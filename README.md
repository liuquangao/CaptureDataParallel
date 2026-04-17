# Isaac Sim Point-Nav ReID Collector

这个目录用于搭建一个基于 Isaac Sim 的点导航 ReID 数据采集器，目标是在 NuRec 这类 USD/USDZ 场景中，采集与现有 Habitat 版本近似的数据：

- RGB 图像
- Depth 图像
- 相机位姿
- 行人位姿
- 可见性/遮挡分数
- 候选目标列表
- 与后续 ReID 训练兼容的元数据

这个版本不依赖 Habitat 的 `scene_dataset_config`、`navmesh`、`pathfinder`，而是使用 Isaac Sim 的：

- USD 场景加载
- 相机传感器
- PhysX 碰撞与射线检测
- NuRec 自带 occupancy map

## 目标

我们希望最终得到一个脚本，能够：

1. 打开一个 NuRec 场景
2. 加载一个 humanoid
3. 在可行走区域内采样相机点和人物点
4. 检查两者是否合理可见
5. 渲染 RGBD
6. 保存与现有数据流程尽量兼容的输出结构

## 推荐优先场景

优先使用这些 NuRec 场景做第一版：

- `nova_carter-galileo`
- `nova_carter-cafe`
- `nova_carter-wormhole`
- `hand_hold-endeavor-andoria`
- `hand_hold-endeavor-livingroom`
- `hand_hold-endeavor-wormhole`

原因：

- 官方说明中这些场景属于 stereo workflow
- 通常包含更完整的 mesh / occupancy 信息
- 更适合做碰撞、可见性和导航区域采样

不建议一开始就用：

- `zh_lounge`
- `zh_fourth_floor_iphone`

因为这两类 mono 场景更偏神经渲染资产，物理几何基础不如前面几套稳。

## 开发路线

建议按下面顺序逐步搭建。每一步都应该先做到“最小可运行”，再进入下一步。

### Step 1. 项目骨架

目标：

- 建立目录结构
- 明确脚本入口
- 约定配置文件与输出目录格式

要做的事：

- 新建 `collector/` 代码目录
- 新建 `configs/` 配置目录
- 新建主入口脚本，例如 `run_collector.py`
- 约定输出目录结构，例如：
  - `scene_xxx/pos_000/rgb/*.png`
  - `scene_xxx/pos_000/depth/*.png`
  - `scene_xxx/pos_000/meta/*.json`

验收标准：

- 可以运行一个空脚本
- 能读取命令行参数或 yaml 配置
- 能创建输出目录

### Step 2. 场景加载

目标：

- 在 Isaac Sim 中正确打开 NuRec 场景
- 确认物理场景与碰撞可用

要做的事：

- 启动 `SimulationApp`
- 打开 `stage.usdz` 或 `usda`
- 获取当前 stage
- 检查是否存在 `UsdPhysics.Scene`
- 必要时将物理更新模式设置为 synchronous
- 对于缺少地面碰撞的场景，允许附加 ground plane

验收标准：

- 指定场景能够稳定打开
- 时间轴能正常播放和更新
- 能在 stage 中找到主要 prim

### Step 3. Occupancy Map 读取

目标：

- 从 NuRec 的 `occupancy_map.yaml/png` 中恢复可行走区域

要做的事：

- 读取 `occupancy_map.yaml`
- 读取 `occupancy_map.png`
- 解析分辨率、原点、坐标系
- 实现栅格坐标与世界坐标的相互转换
- 提供 “采样一个 free-space 点” 的函数

验收标准：

- 能从 occupancy map 中随机采样一批候选地面点
- 这些点在场景里大致落在可走区域

备注：

- 如果个别场景没有 occupancy map，需要单独做 fallback
- fallback 可以是手工给定区域，或后续基于碰撞查询采样

### Step 4. 相机系统

目标：

- 在 Isaac Sim 中创建可控相机，输出 RGB 和 Depth

要做的事：

- 创建相机 prim
- 设置内参或近似视场角
- 支持相机位姿控制
- 输出彩色图
- 输出 depth 图
- 明确 depth 的单位和保存格式

验收标准：

- 能在固定视角下保存一张 RGB 和一张 depth
- 图像尺寸、格式、深度单位都可控

备注：

- 这一阶段先不追求与 Habitat 完全一致
- 先保证“能稳定渲染并保存”

### Step 5. Humanoid 加载

目标：

- 在 Isaac Sim 场景中放置一个人形角色

要做的事：

- 选定 humanoid 资产来源
- 支持加载一个基础 humanoid
- 支持设置位置、朝向
- 如果已有姿态文件，支持后续挂接

验收标准：

- 能在场景中成功加载一个人
- 人能稳定站在地面上，不穿模、不悬空太明显

备注：

- 第一版可以先只支持静态站姿
- 后续再扩展到多个 pose

### Step 6. 地面贴合与碰撞合法性检查

目标：

- 确保相机点和人物点真实可站立

要做的事：

- 向下射线检测地面高度
- 检查相机是否嵌入物体
- 检查人物周围是否有足够净空
- 必要时为场景添加辅助 collision plane

验收标准：

- 采样点落地后，角色和相机不会明显卡进墙体或家具

### Step 7. 相机-人物配对采样

目标：

- 生成有效的观察关系：相机看向人物

要做的事：

- 从 occupancy map 采样人物点
- 在人物周围环带中采样相机点
- 控制距离范围，例如 `0.5m ~ 4.5m`
- 计算相机朝向，使人物落在画面中
- 可加入 yaw 抖动，避免人物总在正中间

验收标准：

- 能稳定生成一批“人物在画面内”的视角

### Step 8. 可见性与遮挡检测

目标：

- 过滤掉看不见、被墙挡住或严重遮挡的样本

要做的事：

- 从相机向人物中心打射线
- 从相机向多个身体关键点打射线
- 统计可见关键点比例
- 结合距离得到 visibility score

验收标准：

- 明显被墙挡住的样本会被过滤
- 明显无遮挡的样本能保留

### Step 9. 元数据输出

目标：

- 输出训练和分析所需的结构化信息

要做的事：

- 为每帧保存：
  - 图像路径
  - depth 路径
  - 相机位姿
  - 人物位姿
  - 投影坐标
  - visibility score
  - 场景名
- 保存每个位置的 `meta.json`
- 视需要生成 `reid_scores.json`

验收标准：

- 输出格式清晰稳定
- 后续脚本能直接读取

### Step 10. 数据格式对齐现有流程

目标：

- 尽量兼容当前 `prepare_habitat` 产出的下游数据结构

要做的事：

- 对齐文件命名
- 对齐目录层级
- 对齐关键 JSON 字段
- 明确哪些字段保留，哪些字段改名

验收标准：

- 下游脚本只需极少改动，或无需改动即可消费数据

### Step 11. 批量采集

目标：

- 从单个样本扩展到整场景批量采集

要做的事：

- 支持每场景多个人物点
- 支持每个人物点多个相机点
- 支持随机种子
- 支持失败重试和日志

验收标准：

- 能持续稳定采集，不因少量非法点中断

### Step 12. 可视化与调试工具

目标：

- 提升调试效率，快速定位采样失败原因

要做的事：

- 可视化采样点
- 可视化射线检测结果
- 可视化 occupancy map 与世界坐标对应关系
- 保存少量 debug 截图

验收标准：

- 出问题时能快速判断是坐标系、碰撞、渲染还是采样逻辑出错

## 第一阶段建议的最小范围

第一阶段只做最小可行版本，建议限定为：

- 只支持一个场景：`nova_carter-galileo`
- 只支持一个 humanoid
- 只支持静态站姿
- 只输出 RGB + depth + 基本 meta
- 只做单进程单场景采样

原因：

- 这样最容易先把整条链打通
- 一旦最小链路成立，后续扩展会轻松很多

## 建议目录结构

建议后续把这个目录组织成：

```text
CaptureData/
  README.md
  configs/
    nurec_galileo.yaml
  collector/
    __init__.py
    run_collector.py
    scene_loader.py
    occupancy.py
    camera.py
    humanoid.py
    sampling.py
    visibility.py
    writer.py
    debug_viz.py
```

## 每一步的落地原则

后续实现时，遵循这几个原则：

1. 先做单场景，再做多场景
2. 先做静态人，再做姿态扩展
3. 先做单样本成功，再做批量采集
4. 先保证坐标和碰撞正确，再优化视觉质量
5. 每新增一个模块，都要有最小可验证输出

## 我们接下来怎么推进

推荐按这个顺序继续：

1. 搭 Step 1 项目骨架
2. 完成 Step 2 场景加载
3. 完成 Step 3 occupancy map 解析
4. 完成 Step 4 相机 RGBD 输出
5. 再进入 humanoid 与可见性逻辑

如果按这个计划走，下一步最适合先做：

- 建目录结构
- 写 `run_collector.py`
- 写第一个配置文件
- 让它能加载 `nova_carter-galileo`
