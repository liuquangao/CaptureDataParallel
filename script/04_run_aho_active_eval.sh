#!/usr/bin/env bash

# ── 可调参数 ────────────────────────────────────────────────────────────────
CHECKPOINT=/home/leo/FusionLab/ActivateReID/outputs/exp_20260511_154541_scoremap_avg/final_model/model.safetensors
OUTPUT_DIR=/home/leo/FusionLab/AHO/CaptureDataParallel/outputs_aho_active_eval
SCENE_LIST=/home/leo/FusionLab/AHO/CaptureDataParallel/configs/scene_list_test.txt
MAX_SCENES=32
# ────────────────────────────────────────────────────────────────────────────

export GLIBC_TUNABLES='glibc.malloc.arena_max=1:glibc.malloc.mmap_max=0:glibc.malloc.mmap_threshold=2147483647'

exec /home/leo/FusionLab/isaacsim/_build/linux-x86_64/release/python.sh \
  /home/leo/FusionLab/AHO/CaptureDataParallel/run_aho_active_eval.py \
  --config /home/leo/FusionLab/AHO/CaptureDataParallel/configs/aho_active_eval.yaml \
  --set "aho_inference.checkpoint=${CHECKPOINT}" \
  --set "backend_params.output_dir=${OUTPUT_DIR}" \
  --set "dataset.selection.scene_list_path=${SCENE_LIST}" \
  --set "eval.max_scenes=${MAX_SCENES}"
