#!/usr/bin/env bash

export GLIBC_TUNABLES='glibc.malloc.arena_max=1:glibc.malloc.mmap_max=0:glibc.malloc.mmap_threshold=2147483647'

exec /home/leo/FusionLab/isaacsim/_build/linux-x86_64/release/python.sh \
  /home/leo/FusionLab/AHO/CaptureDataParallel/script/capture_reference_views.py \
  --output_dir /home/leo/FusionLab/AHO/CaptureDataParallel/outputs_parallel-v2-raycasted/reference
