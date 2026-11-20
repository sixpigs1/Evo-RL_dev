#!/bin/bash
set -uo pipefail
cd /workspace/Evo-RL
root=/workspace/storage/benchmarks/20261120
mkdir -p "$root"
export PYTHONUNBUFFERED=1
for task in evo1-stage2 pi05; do
 profile=evo1; [ "$task" = pi05 ] && profile=pi05
 for gpus in 1 2 4; do
  devices=2; [ "$gpus" = 2 ] && devices=2,3; [ "$gpus" = 4 ] && devices=2,3,4,5
  for batch in 1 2 4 8; do
   name=$task-g${gpus}-b${batch}
   echo "$(date -Is) START $name" | tee -a "$root/status.log"
   MUSA_VISIBLE_DEVICES=$devices tools/musa/run.sh "$profile" python tools/musa/benchmark.py --task "$task" --gpus "$gpus" --batch-size "$batch" --steps 35 --stage1-checkpoint /workspace/storage/outputs/accept-evo1-stage1-1gpu/checkpoints/000100/pretrained_model --output "$root/$name" >"$root/$name.log" 2>&1
   rc=$?
   echo "$(date -Is) END $name rc=$rc" | tee -a "$root/status.log"
  done
 done
done
