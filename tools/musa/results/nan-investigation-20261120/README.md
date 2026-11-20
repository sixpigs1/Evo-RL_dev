# Pi0.5 NaN investigation — 2026-11-20

Status: initial NaN trigger remains under investigation. The propagation bug is confirmed and protected; this is not a claim that long training is numerically fixed.

## Confirmed original failure

Original experiment: https://swanlab.cn/@sixpigs/RWorld-RL/runs/xb24nabv . Original output: `/workspace/storage/outputs/foldtowel40-pi05-4gpu-20261120`.

- Four S5000 GPUs (physical 2–5), batch 32 per rank, global batch 128, BF16 full fine-tuning, checkpointing enabled, no compile. LR 2.5e-5, warmup 500, schedule horizon 9,275 updates.
- Step 650 remained finite: logged 10-step mean loss 0.066, gradient norm 0.491; cloud last-batch loss 0.069424808.
- First logged NaN: step 660 at **21:44:25 Asia/Shanghai** (trainer timestamp 13:44:25 UTC). Logging every 10 steps bounds the first abnormal update to **651–660**, not an exact update.
- Both loss and gradient norm then remained NaN through step 900. Only this Pi0.5 launcher was terminated; ValueFunction was left running.
- The original update loop called backward, gradient clipping, and optimizer.step without testing finiteness. A non-finite norm is not repaired by clipping. The loop therefore continued corrupting model and optimizer state.

| Checkpoint | Model tensors with NaN/Inf | Optimizer tensors with NaN/Inf | Recovery |
|---|---:|---:|---|
| 000371 | 0 / 812 | 0 / 2415 | Verified finite recovery point |
| 000742 | 805 / 812 | 1610 / 2415 | Invalid; do not resume |

**`checkpoints/last` still refers to the original 000742. It must not be used for recovery.** Original files and symlink were preserved as evidence.

## Checks completed

- All 47,398 actual action and observation.state rows: zero NaN/Inf. Quantile statistics finite; no collapsed action/state q01–q99 ranges. This does not prove every internal activation is safe.
- Native MUSA Beta sampling: 13,107,200 samples across four seeds, all finite.
- Actual B32-shaped Gaussian noise, Beta time sampling, and FP64 sinusoidal time encoding: 20,000 rounds, no non-finite values. This covers 640,000 times, 1,024,000,000 Gaussian values, and 655,360,000 output encoding elements. It does not prove that a rare or context-dependent operator fault is impossible.
- CPU single-process numerical guard, CPU two-rank collective guard, and the real training update on CPU DDP: passed.
- Real training update on two S5000s using Accelerate/BF16/MCCL: passed. Injecting NaN loss or Inf gradient on only rank 1 stopped both ranks before optimizer/scheduler mutation; diagnostic inputs and the rank-zero pre-update model/optimizer snapshot loaded successfully. The finite path still updated weights.
- Readable host kernel journal had no entries in the 21:40–21:50 interval. Inline ECC counters were N/A, so they cannot rule out hardware errors. No driver settings or other workloads were changed.

## Protection implemented

`lerobot_train.update_policy` checks distributed loss finiteness before backward and gradient norm finiteness before optimizer.step. Every rank participates in the decision. On failure, every rank saves its processed batch and RNG state from before forward; rank zero saves model and optimizer state before the rejected update. Peers wait for the snapshot before raising. No nan_to_num, skipped bad batches, global CPU fallback, or silent LR reduction is used.

Reproducible checks, from `/workspace/Evo-RL`:

```bash
tools/musa/run.sh pi05 torchrun --standalone --nproc_per_node=2 tools/musa/check_numerical_safety.py
tools/musa/run.sh pi05 torchrun --standalone --nproc_per_node=2 tools/musa/check_update_safety.py
# Only use idle devices; the MUSA check performs intentional numerical failure injection.
MUSA_VISIBLE_DEVICES=0,1 tools/musa/run.sh pi05 accelerate launch \
  --config_file tools/musa/configs/2gpu.yaml --main_process_port 29624 \
  tools/musa/check_update_safety.py --musa \
  --failure-dir /workspace/storage/runs/guard-check-NEW
# Exit status 1 is expected when any scanned checkpoint is contaminated.
tools/musa/run.sh pi05 python tools/musa/check_checkpoint_finiteness.py \
  /workspace/storage/outputs/foldtowel40-pi05-4gpu-20261120/checkpoints/000371 \
  /workspace/storage/outputs/foldtowel40-pi05-4gpu-20261120/checkpoints/000742
```

## Isolated reproduction

The protected diagnostic run restores verified checkpoint 000371 with the original hyperparameters and 9,275-step LR horizon. It uses a separate output, logs each update, and is bounded at total update 1,000. Checkpoint saving is disabled for this diagnostic; it is **not** the resumed 25-epoch formal run.

- Runtime state/logs: `/workspace/storage/runs/pi05-nan-diagnosis-v3-20261120/`
- Failure artifacts, if any: `/workspace/storage/outputs/pi05-nan-diagnosis-v3-20261120/numerical_failures/`
- Independent cloud experiment, mirrored from native stdout: https://swanlab.cn/@sixpigs/RWorld-RL/runs/pctp6t1e . Mirrored metrics have the trainer's displayed rounding.
- Existing checkpoints do not serialize the full dataloader cursor/iterator state. The restored run's loss trajectory differs from the original; this is a reproduction attempt with restored weights/optimizer/RNG, not an asserted bitwise replay of update 651.

`diagnosis/` contains a timestamped progress snapshot, not a completed test. The initial trigger must not be attributed to LR, Beta sampling, BF16, hardware, or an individual MUSA operator without further evidence.
