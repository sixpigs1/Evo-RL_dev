# S5000 / MUSA（单卡与双卡验收）

Use branch `codex/musa-s5000`. The server container is `yzw_explore`; source is
`/home/mccuseradmin/yzw_explore/Evo-RL` (container `/workspace/Evo-RL`). Large
artifacts are in `/data/yzw_explore` (container `/workspace/storage`).

## Deploy and install

On the server run `bash tools/musa/deploy.sh`, then `bash tools/musa/enter.sh`.
The deployment script deliberately does not replace an existing container.
The fixed base image contains torch/torch_musa 2.9.0, vendor torchvision 0.22.1,
MCCL 2.11.4 and MUSA toolkit 4.3.5. Host driver is 5.2.0-server; driver and toolkit
version numbers are distinct. No host upgrades are required.

Inside the container:

```bash
cd /workspace/Evo-RL
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
python tools/musa/bootstrap.py value --lock-file tools/musa/locks/value-py310.txt
python tools/musa/bootstrap.py evo1 --lock-file tools/musa/locks/evo1-py310.txt
python tools/musa/bootstrap.py pi05 --lock-file tools/musa/locks/pi05-py310.txt --pi-transformers-revision=dcddb970176382c0fcf4521b0c0e6fc15894dfe0
```

Each task has its own environment inheriting the immutable vendor stack. Python
packages for training shadow serving packages from the image. Full `pip check`
output and `pip freeze` are saved per environment; inherited serving-package
conflicts must not be mistaken for task validation. Vendor versions are constrained
and checked after install. NumPy 1.26 is required by the supplied torch binary.
Video decoding defaults to PyAV in MUSA examples. TorchCodec and Rerun are optional
extras (`video-torchcodec`, `visualization`); the existing `all` extra retains both.

Run a command in the proper environment:

```bash
tools/musa/run.sh value python tools/musa/doctor.py --device musa:4
tools/musa/run.sh pi05 accelerate launch --config_file tools/musa/configs/1gpu.yaml \
  -m lerobot.scripts.lerobot_train --help
```

## Data and models

```bash
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
python tools/musa/prepare_assets.py
# Download only selected public assets with --only=dataset|internvl3|pi05|value_vision|value_language.
tools/musa/run.sh value python tools/musa/make_subset.py
tools/musa/run.sh value python tools/musa/check_dataset.py \
  --root /workspace/storage/data/towel-small/acceptance \
  --repo-id Elvinky/bi_so101_fold_towel_d0_acceptance --require-labels
```

The pinned towel metadata has authentic episode outcome labels. The small subset
preserves them. The download selects the first two episodes' storage shards and
metadata; the full source dataset remains in its original schema. Do not train on
this partial source root without the selected episode list: use the extracted
subset. `assets-*.json` records resolved model/dataset commits. No gated-model
credentials are assumed. InternVL3 remote code is pinned and downloaded locally;
FlashAttention is disabled. Pi0.5 uses the original LeRobot port and modified
Transformers, not the separately converted native InternVL HF model.

## Offline iterations

```bash
tools/musa/run.sh value python tools/musa/iteration.py \
  --config tools/musa/configs/round-pi05.json --dry-run
tools/musa/run.sh value python tools/musa/iteration.py \
  --config tools/musa/configs/round-evo1.json
```

Create a separate JSON/output directory per task and per round. `gpus` supports
1, 2 and 8, `steps` counts optimizer updates, `batch_size` is per process. No
extra accumulation is implied. Use 100 steps for single-card acceptance and 20
for two-card acceptance. Eight-card execution is optional and is not required by the current acceptance scope. Set `MUSA_VISIBLE_DEVICES` before the command to select
idle cards. Inspect `mthreads-gmi` first; do not stop other people's jobs.

The runner copies the input, checks real labels, trains value, annotates ACP,
trains SFT (Evo1 stage1), trains ACP (Evo1 stage2), reloads the policy and checks
real-batch actions. `state.json` records stage commands and results, while each
stage has its own log. Completed artifacts are reused on restart. For a failed
training stage, use its saved training config with `--resume=true`; the runner
will not silently overwrite its partial output. `resume_pretrain=true` for stage
handoff is weights-only, deliberately distinct from optimizer/RNG resume.

To ingest the next data round, set `dataset_root` to the original unannotated
pool and add `additional_datasets: [{"repo_id": "...", "root": "..."}]`. The
repository merge implementation reindexes episode/frame/video metadata. Replayed
acceptance data must set `replayed_data: true`, never be described as new robot
experience. A new iteration gets a fresh output directory and independent labels.

`deployment.json` gives the checkpoint for `lerobot-human-inloop-record`. Supply
the actual robot/camera/calibration arguments on the robot host; this tooling
never moves a robot automatically. Return recordings as LeRobot datasets, verify
labels using `check_dataset.py`, and pass their roots to `additional_datasets`.
Sync code with `sync.sh SSH_HOST [SSH_OPTIONS...]`; transfer dataset files using
your authenticated SSH route, without storing credentials in scripts.

## Checks

```bash
tools/musa/run.sh value python -m pytest -q tests/musa tests/value tests/training/test_acp_pi05_prompt_pipeline.py
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=2 tools/musa/distributed_smoke.py
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 tools/musa/distributed_smoke.py
tools/musa/run.sh value python tools/musa/check_operators.py
```

Use the test report for actual status; configuration templates alone do not imply
model-level training has passed. Production training convergence and robot
success-rate improvement require separate experiments.

For full resume of a checkpoint produced with `resume_pretrain=true`, pass
`--resume_pretrain=false --resume=true --config_path=.../train_config.json`.
The two modes are mutually exclusive. Training accepts indexed MUSA devices and
per-rank RNG files. This preserves optimizer/scheduler/RNG state, but the original
training loop restarts the dataloader iterator: bitwise continuation of the sample
order across a restart is not claimed.

`train_smoke.py` runs real-weight training and verifies a checkpoint reached the
requested step. `--resume-check` reloads the checkpoint at that step and performs
five more updates. GPU acceptance jobs lock their selected physical cards and check physical-card
utilization after acquiring it. `--batch-size=8 --gpus=1` provides an effective
batch matching `--batch-size=1 --gpus=8`; compare those configurations rather than
reporting a scaling ratio from different global batch sizes.

The towel camera mapping for Pi0.5 is explicit in `round-pi05.json`: left/right
wrist map to the corresponding pretrained wrist keys, and right-front maps to
`base_0_rgb`. Training saves this mapping and action inference reloads it. The
pinned official base checkpoint contains disabled `relative_actions_processor`
and `absolute_actions_processor` entries; compatibility steps accept **only**
`enabled=false`. They do not silently implement or discard enabled conversions.

For remote deployment, the server supports both `pi05` and `evo1`:

```bash
tools/musa/run.sh evo1 python -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=8080
```

Use the matching task environment for Pi0.5. The server reads the requested MUSA
device into the policy configuration before allocating the model, runs inference
without gradients, and returns CPU actions. Run over an authenticated SSH tunnel;
the existing protocol uses pickle and is intended for trusted clients. Configure
`robot_client` on the robot host with `--policy_type=evo1` (or `pi05`),
`--policy_device=musa`, the exported checkpoint, correct camera rename map and
actual robot arguments. The hardware client is not started by acceptance tests.

## Training speed measurements

[`BENCHMARK_20261120.md`](BENCHMARK_20261120.md) records real-data Evo-1 Stage 2
and Pi0.5 full fine-tuning measurements on 1/2/4 S5000 cards, with historical
4090 evidence and explicit comparison limits. `benchmark.py` runs one fresh
configuration; `benchmark_matrix.sh` runs the batch sweep;
`summarize_benchmark.py` excludes warmup and reports global samples/second.
These benchmarks disable checkpoint writes and do not replace training acceptance.

ValueFunction speed results for full SigLIP + Qwen2.5-0.5B training at per-card
batch 16/32 are in [`BENCHMARK_VALUE_20261120.md`](BENCHMARK_VALUE_20261120.md).
Select this sweep with `EVORL_BENCH_TASKS=value EVORL_BENCH_BATCHES="16 32"`;
pass `--tasks value --batches 16 32` to the summarizer.

## Formal FoldTowel training

[`FOLDTOWEL40_TRAINING.md`](FOLDTOWEL40_TRAINING.md) documents the approved
four-card Pi0.5 / two-card ValueFunction jobs, file-hash data verification,
SwanLab dependency pins and secret-free launch plans. `training_job.py` receives
the API key on stdin, keeps it in process memory, refuses duplicate starts, and
runs detached supervisors with separate GPU groups and rendezvous ports.
