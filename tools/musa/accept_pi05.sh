#!/usr/bin/env bash
set -euo pipefail
cd /workspace/Evo-RL
while [[ ! -f /workspace/storage/logs/fast-pi05-r2.exit ]]; do sleep 10; done
[[ $(cat /workspace/storage/logs/fast-pi05-r2.exit) == 0 ]]
[[ -f /workspace/storage/models/paligemma-tokenizer/tokenizer_config.json ]]
python tools/musa/gpu_guard.py --devices 4
MUSA_VISIBLE_DEVICES=4 python tools/musa/train_smoke.py --task pi05 --steps 100 --gpus 1 --resume-check \
  --output /workspace/storage/outputs/accept-pi05-1gpu
python tools/musa/gpu_guard.py --devices 4,5
MUSA_VISIBLE_DEVICES=4,5 python tools/musa/train_smoke.py --task pi05 --steps 20 --gpus 2 \
  --output /workspace/storage/outputs/accept-pi05-2gpu
