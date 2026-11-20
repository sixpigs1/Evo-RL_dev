#!/usr/bin/env bash
set -euo pipefail
cd /workspace/Evo-RL
while [[ ! -f /workspace/storage/assets-value_vision.json || ! -f /workspace/storage/assets-value_language.json ]]; do sleep 10; done
python tools/musa/gpu_guard.py --devices 5
MUSA_VISIBLE_DEVICES=5 python tools/musa/train_smoke.py --task value --steps 100 --gpus 1 --resume-check \
  --output /workspace/storage/outputs/accept-value-1gpu
python tools/musa/gpu_guard.py --devices 4,5
MUSA_VISIBLE_DEVICES=4,5 python tools/musa/train_smoke.py --task value --steps 20 --gpus 2 \
  --output /workspace/storage/outputs/accept-value-2gpu
