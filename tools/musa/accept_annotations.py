# Process-lifetime CLI lock; the OS releases it on exit.
# ruff: noqa: SIM115
"""Infer on independent dataset copies and compare single- and eight-card coverage."""

import argparse
import fcntl
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument("--multi-devices", default="6,7")
p.add_argument(
    "--checkpoint", default="/workspace/storage/outputs/accept-value-1gpu/checkpoints/000100/pretrained_model"
)
p.add_argument("--world-sizes", type=int, choices=[1, 2, 8], nargs="+", default=[1, 2])
p.add_argument("--output", default="/workspace/storage/outputs/annotation-acceptance")
a = p.parse_args()
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
lock = Path("/workspace/storage/.gpu-tests.lock").open("w")
for n in a.world_sizes:
    data = out / f"data-{n}"
    if not data.exists():
        shutil.copytree("/workspace/storage/data/towel-small/acceptance", data)
    env = os.environ.copy()
    env["MUSA_VISIBLE_DEVICES"] = "5" if n == 1 else (a.multi_devices if n == 2 else "0,1,2,3,4,5,6,7")
    subprocess.run(
        ["python", str(ROOT / "tools/musa/gpu_guard.py"), "--devices", env["MUSA_VISIBLE_DEVICES"]],
        check=True,
    )
    values = {
        "dataset.repo_id": "Elvinky/bi_so101_fold_towel_d0_acceptance",
        "dataset.root": data,
        "runtime.device": "musa",
        "runtime.batch_size": 4,
        "runtime.num_workers": 0,
        "inference.checkpoint_path": a.checkpoint,
        "acp.enable": "true",
        "acp.value_field": "complementary_info.value_musa",
        "acp.advantage_field": "complementary_info.advantage_musa",
        "acp.indicator_field": "complementary_info.acp_indicator_musa",
        "output_dir": out / f"infer-{n}",
    }
    command = [
        str(ROOT / "tools/musa/run.sh"),
        "value",
        "accelerate",
        "launch",
        "--config_file",
        str(ROOT / f"tools/musa/configs/{n}gpu.yaml"),
        "-m",
        "lerobot.scripts.lerobot_value_infer",
    ]
    fcntl.flock(lock, fcntl.LOCK_EX)
    with (out / f"infer-{n}.log").open("w") as log:
        subprocess.run(
            command + [f"--{k}={v}" for k, v in values.items()],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    fcntl.flock(lock, fcntl.LOCK_UN)
multi = 2 if (out / "data-2").exists() else 8
if (out / "data-1").exists() and (out / f"data-{multi}").exists():
    subprocess.run(
        [
            str(ROOT / "tools/musa/run.sh"),
            "value",
            "python",
            str(ROOT / "tools/musa/compare_annotations.py"),
            str(out / "data-1"),
            str(out / f"data-{multi}"),
            "--output",
            str(out / "comparison.json"),
        ],
        check=True,
    )
