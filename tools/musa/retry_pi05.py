"""Launch the protected FoldTowel Pi0.5 continuation. Read SwanLab key from stdin."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STORAGE = Path("/workspace/storage")
RUN = STORAGE / "runs/pi05-1484-protected"
CHECKPOINT = STORAGE / "outputs/foldtowel40-pi05-full-resume-20261121/checkpoints/001484"
BASE_PLAN = STORAGE / "runs/foldtowel40-20261120/pi05/plan.json"


def supervise():
    launcher = ROOT / "tools/musa/run.sh"
    with (RUN / "checkpoint-scan.jsonl").open("w") as output:
        subprocess.run(
            [str(launcher), "pi05", "python", str(ROOT / "tools/musa/check_checkpoint_finiteness.py"),
             str(CHECKPOINT)], stdout=output, check=True,
        )
    config_path = CHECKPOINT / "pretrained_model/train_config.json"
    config = json.loads(config_path.read_text())
    assert config["batch_size"] == 32 and config["steps"] == 9275
    plan = json.loads(BASE_PLAN.read_text())
    assert plan["devices"] == [2, 3, 4, 5]
    plan["port"] = 29626
    run_id = "null"
    snapshot = None
    retried_steps = set()
    env = os.environ.copy()
    env["EVORL_PI05_GRAD_DIAGNOSTICS"] = "1"
    attempt = 0
    while True:
        directory = RUN / f"attempt-{attempt:03d}"
        directory.mkdir()
        output_dir = STORAGE / "outputs/pi05-1484-protected" / directory.name
        plan["options"] = {
            "config_path": str(config_path), "resume": True,
            "dataset.root": config["dataset"]["root"], "output_dir": str(output_dir),
            "wandb.enable": True, "wandb.run_id": run_id,
            "job_name": "foldtowel40-pi05-4x32-protected-resume1484",
        }
        if snapshot:
            env["EVORL_PI05_REPLAY"] = str(snapshot)
        path = directory / "plan.json"
        path.write_text(json.dumps(plan, indent=2))
        print(f"Launching {directory.name}, replay={snapshot}", flush=True)
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools/musa/training_job.py"), "--worker", str(path)],
            cwd=ROOT, env=env,
        )
        if result.returncode == 0:
            print("Training completed", flush=True)
            return
        failures = list((output_dir / "numerical_failures").glob("step-*"))
        if len(failures) != 1 or not (failures[0] / "complete.json").is_file():
            raise RuntimeError("Training failed without a complete numerical snapshot; automatic retry stopped")
        snapshot = failures[0]
        manifest = json.loads((snapshot / "complete.json").read_text())
        names = {"pre-update-state.pt", *(f"rank-{rank}.pt" for rank in range(4))}
        if set(manifest["sizes"]) != names or any(
            size <= 0 or (snapshot / name).stat().st_size != size
            for name, size in manifest["sizes"].items()
        ):
            raise RuntimeError(f"Incomplete snapshot files: {snapshot}")
        step = manifest["step"]
        if step in retried_steps:
            print(f"Non-finite update {step} reproduced; stopped. Artifacts: {snapshot}", flush=True)
            return
        retried_steps.add(step)
        state = json.loads((directory / "state.json").read_text())
        run_id = state["swanlab_url"].rstrip("/").rsplit("/", 1)[-1]
        attempt += 1


if __name__ == "__main__":
    os.setgid(100)
    os.umask(0o002)
    if sys.argv[1:] == ["--supervise"]:
        supervise()
    else:
        key = sys.stdin.readline().strip()
        if not key:
            raise RuntimeError("Provide the SwanLab key on stdin")
        RUN.mkdir()
        with (RUN / "supervisor.log").open("w") as log:
            process = subprocess.Popen(
                [sys.executable, __file__, "--supervise"], start_new_session=True,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "SWANLAB_API_KEY": key},
            )
        print(f"Supervisor PID {process.pid}", flush=True)
