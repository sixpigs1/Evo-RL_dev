#!/usr/bin/env bash
set -euo pipefail
cd /workspace/Evo-RL
while [[ ! -f /workspace/storage/assets-internvl3.json ]]; do sleep 10; done
for task in stage1 stage2; do
  checkpoint=()
  if [[ $task == stage2 ]]; then checkpoint=(--stage1-checkpoint /workspace/storage/outputs/accept-evo1-stage1-1gpu/checkpoints/last/pretrained_model); fi
  python tools/musa/gpu_guard.py --devices 4
  MUSA_VISIBLE_DEVICES=4 python tools/musa/train_smoke.py --task "evo1-$task" --steps 100 --gpus 1 --resume-check \
    --output "/workspace/storage/outputs/accept-evo1-$task-1gpu" "${checkpoint[@]}"
  python tools/musa/gpu_guard.py --devices 4,5
  MUSA_VISIBLE_DEVICES=4,5 python tools/musa/train_smoke.py --task "evo1-$task" --steps 20 --gpus 2 \
    --output "/workspace/storage/outputs/accept-evo1-$task-2gpu" "${checkpoint[@]}"
done
