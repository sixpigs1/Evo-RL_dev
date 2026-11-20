"""Run real model acceptance (pretrained weights required); no random-weight fallback."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument("--task", choices=["pi05", "evo1-stage1", "evo1-stage2", "value"], required=True)
p.add_argument("--gpus", type=int, choices=[1, 2, 8], default=1)
p.add_argument("--steps", type=int, default=100)
p.add_argument("--batch-size", type=int, default=1)
p.add_argument("--storage", default="/workspace/storage")
p.add_argument("--stage1-checkpoint")
p.add_argument("--output", required=True)
p.add_argument("--resume-check", action="store_true")
a = p.parse_args()
storage, out = Path(a.storage), Path(a.output)
if a.task == "pi05" and not (storage / "assets-pi05.json").exists():
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/musa/prepare_assets.py"),
            "--only=pi05",
            "--storage",
            str(storage),
        ],
        check=True,
    )
storage.mkdir(parents=True, exist_ok=True)
devices = os.environ.get("MUSA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
gpu_locks = [(storage / f".gpu-test-{i}.lock").open("w") for i in sorted(map(int, devices.split(",")))]
while True:
    acquired = []
    try:
        for gpu_lock in gpu_locks:
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired.append(gpu_lock)
        idle = subprocess.run(
            ["python", str(ROOT / "tools/musa/gpu_guard.py"), "--timeout", "0", "--devices", devices],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if idle.returncode == 0:
            break
    except BlockingIOError:
        pass
    for gpu_lock in acquired:
        fcntl.flock(gpu_lock, fcntl.LOCK_UN)
    time.sleep(5)
profile = "evo1" if a.task.startswith("evo1") else a.task
script = "lerobot_value_train" if profile == "value" else "lerobot_train"
values = {
    "dataset.repo_id": "Elvinky/bi_so101_fold_towel_d0_acceptance",
    "dataset.root": storage / "data/towel-small/acceptance",
    "dataset.video_backend": "pyav",
    "dataset.image_transforms.enable": False,
    "batch_size": a.batch_size,
    "num_workers": 0,
    "steps": a.steps,
    "save_freq": max(1, a.steps // 2),
    "log_freq": 1,
    "output_dir": out,
    "save_checkpoint": True,
    "wandb.enable": False,
}
if profile == "value":
    values |= {
        "value.type": "pistar06",
        "value.device": "musa",
        "value.dtype": "bfloat16",
        "value.use_gradient_checkpointing": True,
        "value.push_to_hub": False,
        "value.vision_repo_id": storage / "models/value_vision",
        "value.language_repo_id": storage / "models/value_language",
    }
else:
    values |= {"policy.device": "musa", "policy.push_to_hub": False, "eval_freq": 0}
    if profile == "pi05":
        values["rename_map"] = json.dumps(
            {
                "observation.images.left_wrist": "observation.images.left_wrist_0_rgb",
                "observation.images.right_wrist": "observation.images.right_wrist_0_rgb",
                "observation.images.right_front": "observation.images.base_0_rgb",
            }
        )
        values |= {
            "policy.path": storage / "models/pi05",
            "policy.dtype": "bfloat16",
            "policy.gradient_checkpointing": True,
            "policy.compile_model": False,
            "policy.use_amp": True,
            "policy.tokenizer_name": storage / "models/paligemma-tokenizer",
        }
    else:
        values |= {
            "policy.type": "evo1",
            "policy.training_stage": a.task.split("-")[1],
            "policy.vlm_model_name": storage / "models/internvl3",
            "policy.use_flash_attn": False,
            "policy.enable_vlm_gradient_checkpointing": a.task.endswith("stage2"),
        }
        if a.task.endswith("stage2"):
            if not a.stage1_checkpoint:
                raise ValueError("Stage2 acceptance requires a real Stage1 checkpoint")
            values.pop("policy.type")
            values |= {"policy.path": a.stage1_checkpoint, "resume_pretrain": True}


def opts(options):
    return [f"--{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in options.items()]


prefix = [
    str(ROOT / "tools/musa/run.sh"),
    profile,
    "accelerate",
    "launch",
    "--config_file",
    str(ROOT / f"tools/musa/configs/{a.gpus}gpu.yaml"),
    "-m",
    f"lerobot.scripts.{script}",
]
log_dir = storage / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
report = {
    "task": a.task,
    "gpus": a.gpus,
    "steps": a.steps,
    "batch_size_per_rank": a.batch_size,
    "output": str(out),
    "started": time.time(),
}
report_path = log_dir / f"{out.name}-result.json"
try:
    checkpoint = out / "checkpoints/last/pretrained_model"
    report["reused_checkpoint"] = (checkpoint / "config.json").exists()
    if not report["reused_checkpoint"]:
        with (log_dir / f"{out.name}.log").open("w") as log:
            subprocess.run(prefix + opts(values), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    assert (checkpoint / "config.json").exists()
    saved_step = json.loads((checkpoint.parent / "training_state/training_step.json").read_text())["step"]
    if saved_step < a.steps:
        raise RuntimeError(
            f"Checkpoint only reached step {saved_step}, expected {a.steps}; resume this run before acceptance"
        )
    if a.resume_check:
        for candidate in sorted((out / "checkpoints").iterdir()):
            step_file = candidate / "training_state/training_step.json"
            if (
                candidate.name != "last"
                and step_file.exists()
                and json.loads(step_file.read_text())["step"] == a.steps
            ):
                checkpoint = candidate / "pretrained_model"
                break
        config_file = checkpoint / ("value_train_config.json" if profile == "value" else "train_config.json")
        with (log_dir / f"{out.name}-resume.log").open("w") as log:
            subprocess.run(
                prefix
                + opts(
                    {
                        "config_path": config_file,
                        "resume": True,
                        **({"resume_pretrain": False} if profile != "value" else {}),
                        "steps": a.steps + 5,
                    }
                ),
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    report["status"] = "passed"
except Exception as error:
    report.update(status="failed", error=str(error))
    raise
finally:
    report["finished"] = time.time()
    report_path.write_text(json.dumps(report, indent=2))
