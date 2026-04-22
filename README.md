# CaptureDataParallel

`CaptureDataParallel` is a lightweight parallel data collection pipeline built on top of Isaac Sim and Replicator. It targets large-scale synthetic data generation for indoor scenes by sampling many candidate camera views around a placed character, scoring those views, and then capturing the selected observations in parallel.

The project is intentionally simple:

- one main entry script: `run_collector.py`
- one YAML config: `configs/sage3d_parallel.yaml`
- a small set of `utils/` modules for scene selection, occupancy-map handling, camera batching, scoring, and output writing

## What It Does

For each scene, the pipeline:

1. resolves which scene(s) to collect from
2. loads the matching InteriorGS occupancy map
3. places one person in a valid free-space location
4. samples camera candidates in a ring around that person
5. scores candidate views using occupancy visibility and segmentation visibility
6. selects views within a target score range
7. captures RGB, depth, masks, score maps, yaw maps, and person bounding boxes
8. writes all outputs to a structured per-scene, per-position folder

This is an Isaac Sim SDG workflow, but implemented as a custom parallel collector rather than a stock example script.

## Main Components

- [run_collector.py](/home/leo/FusionLab/CaptureDataParallel/run_collector.py)
  Main pipeline entry. Starts `SimulationApp`, loops over scenes and positions, runs multi-pass scoring and capture, and writes final outputs.

- [configs/sage3d_parallel.yaml](/home/leo/FusionLab/CaptureDataParallel/configs/sage3d_parallel.yaml)
  Main experiment config. Controls scene selection, camera settings, sampling parameters, render settings, and output path.

- [utils/scene_selection.py](/home/leo/FusionLab/CaptureDataParallel/utils/scene_selection.py)
  Resolves scene lists from the dataset config and skips scenes whose occupancy map looks unbounded.

- [utils/occupancy_map.py](/home/leo/FusionLab/CaptureDataParallel/utils/occupancy_map.py)
  Loads InteriorGS occupancy maps, handles map-to-Isaac coordinate conversion, and samples valid free-space positions.

- [utils/person_placement.py](/home/leo/FusionLab/CaptureDataParallel/utils/person_placement.py)
  Places or repositions the character in the scene with spacing and obstacle-clearance constraints.

- [utils/ring_sampling.py](/home/leo/FusionLab/CaptureDataParallel/utils/ring_sampling.py)
  Generates camera candidates around the person and scores visibility shortcuts directly from occupancy maps.

- [utils/replicator_tools.py](/home/leo/FusionLab/CaptureDataParallel/utils/replicator_tools.py)
  Builds and tears down the multi-camera pool used for batched parallel rendering.

- [utils/capture_outputs.py](/home/leo/FusionLab/CaptureDataParallel/utils/capture_outputs.py)
  Converts raw capture outputs into saved RGB, depth, ground mask, score map, valid mask, yaw map, and bbox files.

## Data Assumptions

The current config expects a SpatialVerse-style dataset layout:

- `dataset.stage_root`: directory of `.usda` scene files
- `dataset.interiorgs_root`: directory of matching InteriorGS occupancy-map folders

The provided example config points to:

- `/home/leo/FusionLab/DataSets/spatialverse/SAGE-3D_InteriorGS_usda`
- `/home/leo/FusionLab/DataSets/spatialverse/InteriorGS`

Each selected scene must exist both as a `.usda` stage and as a matching InteriorGS occupancy-map directory.

## How To Run

Run the collector with Isaac Sim's bundled Python:

```bash
~/FusionLab/isaacsim/_build/linux-x86_64/release/python.sh \
  run_collector.py \
  --config configs/sage3d_parallel.yaml
```

This is important because `run_collector.py` imports Isaac Sim and `omni.replicator` modules, which require the Isaac Sim runtime environment provided by `python.sh`.

## Config Overview

The example config contains the main controls you will usually change:

- `launch_config`
  Isaac Sim launch settings such as renderer mode and `headless`.

- `dataset.selection`
  Scene selection mode. Supported values are:
  - `single`
  - `random`
  - `all`

- `num_positions`
  Number of person placements collected per scene.

- `num_cameras`
  Number of cameras rendered in parallel in each batch.

- `sampling`
  Seeds, minimum spacing between placements, obstacle clearance, and retry limits.

- `score_field`
  Ring radius range, candidate grid step, camera obstacle clearance, score thresholds, and final capture count.

- `camera`
  Resolution, camera height, focal length, apertures, and clipping settings.

- `backend_params.output_dir`
  Root output directory for all generated data.

## Pipeline Stages

The collector runs in a small number of clear stages.

### 1. Scene Resolution and Filtering

The pipeline resolves the scene list from the config, loads the occupancy map first, and skips scenes that look too open or unbounded based on `scene_filter.max_free_ratio`.

### 2. Person Placement

For each scene, the collector places a single character in free space. Placement is constrained by:

- obstacle clearance
- room-free occupancy area
- minimum distance from previous placements in the same scene

### 3. Score-Field Construction

The collector samples a ring of candidate camera poses around the person. Candidate poses are split into:

- certain views: full-width occupancy visibility says the person is definitely visible
- uncertain views: require rendering-based scoring

Uncertain views are processed in two batched passes:

- Pass 1: hide scene mesh and count total visible person pixels
- Pass 2: show scene mesh and count non-occluded visible person pixels

The final score is:

```text
visible_person_pixels / total_person_pixels
```

All candidate poses and scores are written to `score_field.json`, and an overlay image is written to `score_field_overlay.png`.

### 4. Final View Selection and Capture

Views whose scores fall inside the configured score range are selected for final capture. The collector then switches to higher render quality and runs:

- Capture Pass A: RGB + semantic segmentation
- Capture Pass B: depth

These captures are processed into the final per-view outputs.

## Output Structure

Outputs are written under:

```text
<backend_params.output_dir>/<scene_id>/pos_{idx:03d}/
```

Each `pos_xxx` directory contains:

- `score_field.json`
- `score_field_overlay.png`
- `rgb/{idx:03d}.png`
- `depth/{idx:03d}.png`
- `depth/{idx:03d}.npy`
- `ground_mask/{idx:03d}.png`
- `score_map/{idx:03d}.npy`
- `valid_mask/{idx:03d}.npy`
- `yaw_map/{idx:03d}.npy`
- `person_bbox/{idx:03d}.json`
- `metadata.json`
- `scores.json`

### File Meanings

- `score_field.json`
  All sampled camera candidates around the person, including score and scoring mode.

- `score_field_overlay.png`
  Occupancy-map visualization with the person position, sampled candidates, and selected final views.

- `rgb`
  Final RGB observations.

- `depth`
  Depth saved both as 16-bit PNG in millimeters and as `float32` NumPy arrays in meters.

- `ground_mask`
  Pixels that correspond to visible ground near floor height and within occupancy-map free space.

- `score_map`
  Per-pixel projection of score-field values onto the captured image.

- `valid_mask`
  Boolean mask indicating which score-map pixels are valid projections.

- `yaw_map`
  Two-channel orientation target storing relative yaw as cosine and sine.

- `person_bbox`
  Normalized `xyxy` person bounding box derived from semantic segmentation.

- `metadata.json`
  Scene id, camera parameters, person position, timing, and the list of captured observations.

- `scores.json`
  Filename-to-score mapping for final captured views.

## Current Default Setup

The bundled config currently uses:

- `num_cameras: 16`
- `num_positions: 4`
- random scene selection
- `random_count: 640`
- output directory: `/home/leo/FusionLab/CaptureDataParallel/outputs_parallel`

The default person asset is an Isaac Sim hosted character USD.

## Notes

- This project is designed to run inside an Isaac Sim Python environment, not a system Python.
- The collector assumes scene collision geometry is available at `/World/scene_collision`.
- The collector also assumes the visible scene mesh lives at `/World/gauss`.
- A fresh camera pool is created per scene to avoid stale render-product handles after stage changes.
- If `close_app_after_run` is set to `false`, the Isaac Sim GUI stays open after collection.
