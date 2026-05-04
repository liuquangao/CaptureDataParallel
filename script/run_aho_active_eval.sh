#!/usr/bin/env bash

export GLIBC_TUNABLES='glibc.malloc.arena_max=1:glibc.malloc.mmap_max=0:glibc.malloc.mmap_threshold=2147483647'

exec /home/leo/FusionLab/isaacsim/_build/linux-x86_64/release/python.sh \
  /home/leo/FusionLab/AHO/CaptureDataParallel/run_aho_active_eval.py \
  --config /home/leo/FusionLab/AHO/CaptureDataParallel/configs/aho_active_eval.yaml
