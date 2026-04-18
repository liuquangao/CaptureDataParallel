# CaptureData

这个目录现在是一个基于 Isaac Sim 的数据采集与调试工程，围绕 `SAGE-3D .usda` 场景和 `InteriorGS occupancy` 地图工作。

当前代码的核心目标不是复刻旧 Habitat 流程，而是把下面这条链路打通并验证清楚：

- 打开 SAGE-3D 的 `.usda` 场景
- 读取对应的 InteriorGS occupancy map
- 把 occupancy 栅格映射到 Isaac 世界坐标
- 在 occupancy free-space 上采样人物点和候选相机点
- 渲染 RGB / depth / ground mask / score map
- 保存每帧图像、深度、分数和相机位姿
- 提供调试脚本验证 occupancy 对齐、像素反投影和 score 投影是否正确

这份 README 只描述当前代码实际在做什么，以及已经验证过的坐标系约定。

## 目录结构

主要文件如下：

- [collector/run_collector.py](/home/leo/FusionLab/CaptureData/collector/run_collector.py)
  - 主入口。负责打开场景、加载 occupancy、采样人物和相机、执行采集、写出 `metadata.json`
- [collector/occupancy.py](/home/leo/FusionLab/CaptureData/collector/occupancy.py)
  - occupancy map 读取，以及 occupancy 栅格和 Isaac 世界坐标之间的互转
- [collector/camera_capture.py](/home/leo/FusionLab/CaptureData/collector/camera_capture.py)
  - 相机创建、RGB/depth 抓取、ground mask 生成、score map 投影
- [collector/score_field.py](/home/leo/FusionLab/CaptureData/collector/score_field.py)
  - 在 occupancy 上生成候选视角点，做无遮挡判断和可见性打分
- [collector/debug_viz.py](/home/leo/FusionLab/CaptureData/collector/debug_viz.py)
  - 调试人物加载、站立 pose、scene query 支持
- [configs/sage3d.yaml](/home/leo/FusionLab/CaptureData/configs/sage3d.yaml)
  - 当前主要配置文件
- [debug_render_occupancy_in_scene.py](/home/leo/FusionLab/CaptureData/debug_render_occupancy_in_scene.py)
  - 把 occupancy 直接渲染到 Isaac 场景中，用来验证 occupancy 和场景是否对齐
- [debug_backproject_pixels_to_scene.py](/home/leo/FusionLab/CaptureData/debug_backproject_pixels_to_scene.py)
  - 对指定 RGB 像素做 depth 反投影，把结果同时画到 RGB、occupancy 和 Isaac 场景中
- [overlay_score_map_on_rgb.py](/home/leo/FusionLab/CaptureData/overlay_score_map_on_rgb.py)
  - 把 `score_map/*.npy` 叠加到 RGB 上，检查投影是否合理
- [overlay_room_on_occupancy.py](/home/leo/FusionLab/CaptureData/overlay_room_on_occupancy.py)
  - 按和官方 `semantic_map_builder.py` 一致的方式，把房间 polygon 栅格化并叠到 occupancy 上

## 数据源关系

当前工程同时使用两套数据：

- SAGE-3D `.usda` 场景
  - 路径来自 `dataset.stage_root`
- InteriorGS occupancy / structure / annotation
  - 路径来自 `dataset.interiorgs_root`

它们通过 `scene_id` 关联。

例如：

- SAGE-3D 场景：`.../SAGE-3D_InteriorGS_usda/839875.usda`
- InteriorGS 场景目录：`.../InteriorGS/*_839875/`

在代码里，`load_interiorgs_occupancy_map()` 会根据 `stage_url` 自动找到对应的 InteriorGS 目录。

## 主流程

主流程入口是 [collector/run_collector.py](/home/leo/FusionLab/CaptureData/collector/run_collector.py)。

当前链路如下：

1. 读取配置
2. 启动 Isaac Sim
3. 打开 `.usda` 场景
4. 读取对应的 InteriorGS occupancy
5. 在 occupancy free-space 中放置一个调试人物
6. 在人物周围的 occupancy 环带上采样候选相机点
7. 用 segmentation 可见性给候选点打分
8. 选择分数合适的相机点
9. 从这些相机点渲染：
   - RGB
   - depth png
   - depth npy
   - ground mask
   - score map
10. 保存每帧 observation 到 `metadata.json`

## 输出结构

每个 scene 的输出在：

- `outputs/<scene_id>/pos_<idx>/`

典型结构：

- `rgb/*.png`
- `depth/*.png`
- `depth/*.npy`
- `ground_mask/*.png`
- `score_map/*.npy`
- `score_field_overlay.png`
- `metadata.json`
- `scores.json`

当前 `metadata.json` 每条 observation 会保存：

- `rgb_path`
- `depth_path`
- `depth_npy_path`
- `ground_mask_path`
- `score_map_path`
- `reid_score`
- `sampling_score`
- `visible_person_pixels`
- `total_person_pixels`
- `camera_position`
- `camera_orientation_wxyz`

这样后续可以直接从某一帧反查相机位姿和深度文件。

## Occupancy 加载逻辑

InteriorGS occupancy 读取在 [collector/occupancy.py](/home/leo/FusionLab/CaptureData/collector/occupancy.py)。

对当前项目，occupancy 图像像素语义是：

- `255` -> free
- `0` -> occupied
- `127` -> unknown

读取后构造成 `OccupancyMap`：

- `resolution = occupancy.json["scale"]`
- `origin_xy = occupancy.json["min"][:2]`
- `width / height = occupancy.png 尺寸`

并生成：

- `free_mask`
- `occupied_mask`
- `unknown_mask`

## Occupancy 与 Isaac 世界坐标的对应

这是当前项目里最关键的一步。

### 1. 栅格到 map 坐标

`grid_to_world(row, col)` 先把 occupancy cell 转成 map 坐标中心：

```python
x_map = origin_x + (col + 0.5) * resolution
y_map = origin_y + (height - row - 0.5) * resolution
```

这里的含义是：

- occupancy 数组的 `row=0` 在图像最上方
- map 坐标的 `y` 是向上为正
- 所以数组行号转 map `y` 时，要先做一次上下翻转

### 2. map 坐标到 Isaac 世界坐标

然后 `_map_to_isaac_xy()` 按配置里的三个开关继续转换：

- `occupancy_flip_x`
- `occupancy_flip_y`
- `occupancy_negate_xy`

当前默认配置见 [configs/sage3d.yaml](/home/leo/FusionLab/CaptureData/configs/sage3d.yaml)：

```yaml
occupancy_flip_x: true
occupancy_flip_y: true
occupancy_negate_xy: true
```

所以当前 occupancy 到 Isaac 的真实链路是：

```text
occupancy grid
-> map coordinates
-> flip_x / flip_y / negate_xy
-> Isaac world coordinates
```

### 已验证结论

通过 [debug_render_occupancy_in_scene.py](/home/leo/FusionLab/CaptureData/debug_render_occupancy_in_scene.py) 已经验证：

- 当前 `grid_to_world()` 生成的 Isaac 世界坐标
- 和实际场景地面对齐

也就是说：

- `occupancy -> Isaac world` 这条链目前是可信的

## 图像如何和空间对齐

这部分最容易混。

### occupancy 图像层面

occupancy 本质上是一个二维数组：

- 左上角是数组坐标原点
- `col` 向右增大
- `row` 向下增大

但它表达的 map 空间不是“向下为正”，而是常规二维空间：

- `x` 向右为正
- `y` 向上为正

所以从 occupancy 数组到 map 坐标时，必须做一次上下翻转，这也是 `height - row - 0.5` 的来源。

### 为什么还能和官方 semantic_map_builder 对齐

官方 `semantic_map_builder.py` 在中间数组写入时用了 `i_flip / j_flip`，本质上也是在处理：

- 空间坐标按直角坐标系理解
- NumPy 图像数组按左上角原点存储

所以你现在这个项目里 occupancy 的空间解释，和官方语义地图的坐标思路是一致的，只是实现入口不完全相同。

## Occupancy 在 Isaac 场景中的渲染

[debug_render_occupancy_in_scene.py](/home/leo/FusionLab/CaptureData/debug_render_occupancy_in_scene.py) 做的事情是：

1. 打开指定 `.usda`
2. 读取对应 occupancy
3. 对每个 occupancy cell 计算 `grid_to_world(row, col)`
4. 用 `UsdGeom.PointInstancer` 在场景里实例化薄立方体

默认颜色：

- `occupied` -> 红
- `unknown` -> 黄
- `free` -> 绿，只有 `--show-free` 时显示

这些是实际的 USD 几何体，所以可以被 Isaac 相机正常 RGB 渲染。

这个脚本验证的是：

- occupancy 栅格解释是否正确
- occupancy 和场景地面是否对齐

## 当前相机与图像坐标定义

这里一定要和 OpenCV 习惯区分开。

### 图像像素坐标

图像仍然是标准像素坐标：

- 左上角是原点
- 向右 `u` 增大
- 向下 `v` 增大

### 当前代码里经过验证后的相机局部坐标

通过 [debug_backproject_pixels_to_scene.py](/home/leo/FusionLab/CaptureData/debug_backproject_pixels_to_scene.py) 做 A/B 对照后，目前代码采用的相机局部坐标更符合：

- 左方 `+X`
- 上方 `+Y`
- 前方 `+Z`

这不是 OpenCV 常见的那套：

- 右方 `+X`
- 下方 `+Y`
- 前方 `+Z`

所以当前项目里图像反投影和正向投影使用的是：

```python
lateral = -(u - cx) * depth / fx
vertical = -(v - cy) * depth / fy

u = cx - fx * lateral / forward
v = cy - fy * vertical / forward
```

也就是说：

- 图像向右，不对应当前相机局部坐标的 `+X`
- 图像向下，不对应当前相机局部坐标的 `+Y`
- 当前实现里横向和纵向都带负号

### 这个定义是怎么验证出来的

验证方式是：

- 对同一批像素，按两套公式反投影到世界坐标
- 再把结果在 Isaac 场景里画成小球
- 比较哪一组小球和场景中的实际位置一致

实验结果是：

- 原始横向公式得到的球会左右镜像
- 横向取反后的小球和场景一致

所以现在正式代码已经采用修正后的版本。

## Ground Mask 当前逻辑

`ground_mask` 生成在 [collector/camera_capture.py](/home/leo/FusionLab/CaptureData/collector/camera_capture.py)。

当前逻辑是：

1. 从 depth 把每个像素反投影到世界点 `(world_x, world_y, world_z)`
2. 先筛出：
   - 深度有效
   - `abs(world_z - floor_z) <= ground_tolerance_m`
3. 再把这些世界点映射回 occupancy：
   - `occupancy_map.world_to_grid(world_x, world_y)`
4. 只有落在 `occupancy_map.free_mask` 上的像素才保留

所以最终：

```text
ground_mask = near_floor ∩ occupancy_free
```

这表示 ground mask 不是单纯“几何上靠近地面”，而是：

- 靠近地面
- 且该地面点在 occupancy 里属于 free-space

## Score Field 当前逻辑

score field 在 [collector/score_field.py](/home/leo/FusionLab/CaptureData/collector/score_field.py)。

它的候选点不是从图像里来的，而是：

1. 在人物周围环带上采样 `(x, y)`
2. 把这些点映射回 occupancy
3. 只有落在 `free_mask` 里的候选点才保留
4. 如果 occupancy 上两侧视线无遮挡，直接给满分
5. 否则用 segmentation visibility 批量打分

因此：

- `score_field` 本身是 occupancy 约束下生成的
- `score_map` 则是把这些世界点再次投回 RGB 上

## 像素反投影闭环验证

[debug_backproject_pixels_to_scene.py](/home/leo/FusionLab/CaptureData/debug_backproject_pixels_to_scene.py) 用来验证下面这条链：

```text
RGB pixel + depth
-> Isaac world point
-> occupancy cell
-> Isaac scene marker
```

脚本会：

1. 读取指定 RGB
2. 自动从同目录 `metadata.json` 找到对应 observation
3. 读取：
   - `depth_npy_path`
   - `camera_position`
   - `camera_orientation_wxyz`
4. 对指定像素做反投影
5. 同时输出：
   - RGB 上的像素标记图
   - occupancy 上的对应点图
   - Isaac 场景中的小球

这个脚本专门用来定位：

- 像素 -> 世界
- 世界 -> occupancy

到底是哪一段出了问题。

## 房间 polygon 与 occupancy 的对齐验证

[overlay_room_on_occupancy.py](/home/leo/FusionLab/CaptureData/overlay_room_on_occupancy.py) 当前不是简单画线，而是按和官方 `semantic_map_builder.py` 一样的方式做：

- 用房间 polygon 的世界坐标
- 栅格化成 occupancy 上的 mask
- 使用和官方一致的 `i_flip / j_flip` 逻辑
- 再叠到 occupancy 上

这个脚本用于验证：

- `structure.json` 里的房间轮廓
- 和 occupancy 栅格
- 是否真的一致

## Score Map 叠加到 RGB

[overlay_score_map_on_rgb.py](/home/leo/FusionLab/CaptureData/overlay_score_map_on_rgb.py) 用来把：

- `score_map/*.npy`

叠加到：

- `rgb/*.png`

作为热力图查看。

适合快速检查：

- score 点有没有落到不合理区域
- ground mask 和投影链是否正常

## 当前配置文件

主要配置在 [configs/sage3d.yaml](/home/leo/FusionLab/CaptureData/configs/sage3d.yaml)。

最常改的几项：

### 数据选择

```yaml
dataset:
  selection:
    mode: single | random | all
    scene_id: 839875
```

### occupancy 到 Isaac 的坐标变换

```yaml
scene:
  occupancy_flip_x: true
  occupancy_flip_y: true
  occupancy_negate_xy: true
```

### 相机参数

```yaml
camera:
  resolution: [600, 400]
  camera_height: 0.4
  focal_length: 1.8
  horizontal_aperture: 3.6
  vertical_aperture: 2.4
```

### score field 采样参数

```yaml
score_field:
  min_radius_m: 1.2
  max_radius_m: 4.5
  grid_step_m: 0.2
  capture_score_min: 0.1
  capture_score_max: 0.65
```

## 常用命令

### 运行采集

```bash
~/FusionLab/isaacsim/_build/linux-x86_64/release/python.sh \
  /home/leo/FusionLab/CaptureData/collector/run_collector.py \
  --config /home/leo/FusionLab/CaptureData/configs/sage3d.yaml
```

### 在场景里渲染 occupancy

```bash
/home/leo/FusionLab/CaptureData/debug_render_occupancy_in_scene.py
```

显示 free-space：

```bash
/home/leo/FusionLab/CaptureData/debug_render_occupancy_in_scene.py --show-free
```

### 验证像素反投影

```bash
/home/leo/FusionLab/CaptureData/debug_backproject_pixels_to_scene.py
```

指定像素：

```bash
/home/leo/FusionLab/CaptureData/debug_backproject_pixels_to_scene.py \
  --pixels 120,330 280,320 470,310
```

### 把 score_map 叠到 RGB 上

```bash
python3 /home/leo/FusionLab/CaptureData/overlay_score_map_on_rgb.py
```

### 检查房间轮廓和 occupancy 是否对齐

```bash
python3 /home/leo/FusionLab/CaptureData/overlay_room_on_occupancy.py
```

## 目前已经确认的结论

1. `occupancy -> Isaac world` 这条链当前是对的  
   通过 `debug_render_occupancy_in_scene.py` 已验证

2. 相机反投影之前存在水平镜像问题  
   通过 `debug_backproject_pixels_to_scene.py` 的对照实验已定位

3. 正式代码里已经修正了相机横向符号  
   当前 `camera_capture.py` 使用的是验证后的版本

4. 当前 ground mask 已重新加回 occupancy free-space 过滤  
   所以 `ground_mask` 表示的是：
   - 近地面像素
   - 且 occupancy free-space

## 仍需继续关注的问题

- 虽然 occupancy 与场景对齐已经验证通过，但 world <-> occupancy 的更多边界情况还值得继续看
- 如果后续继续出现 score 点落在奇怪区域，需要优先区分：
  - `score_field` 本身生成错
  - 还是 `score_map` 投影错
- 当前相机局部坐标约定不是 OpenCV 默认那套，后续再写投影相关代码时不要直接照搬 OpenCV 公式
