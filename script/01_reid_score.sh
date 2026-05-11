#!/usr/bin/env bash

python3 /home/leo/FusionLab/AHO/CaptureDataParallel/script/compute_reid_scoremap.py \
    --dataset_dir /home/leo/FusionLab/AHO/CaptureDataParallel/outputs_parallel-v2 \
    --reference_dir /home/leo/FusionLab/AHO/CaptureDataParallel/outputs_parallel-v2-raycasted/reference \
    --force
