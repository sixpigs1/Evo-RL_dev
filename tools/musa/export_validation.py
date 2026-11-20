"""Export small reproducibility artifacts; never copy weights, credentials or caches."""

import json
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[2]
storage = Path("/workspace/storage")
results = root / "tools/musa/results/2026-11-20"
locks = root / "tools/musa/locks"
results.mkdir(parents=True, exist_ok=True)
locks.mkdir(parents=True, exist_ok=True)
for profile in ("pi05", "evo1", "value"):
    env = storage / "environments" / profile
    validation = json.loads((env / "dependency-validation.json").read_text())
    assert validation["status"] == "passed", validation
    lock = (env / "requirements.task-lock.txt").read_text()
    if profile == "pi05":
        lock = (
            "\n".join(
                "transformers @ https://github.com/huggingface/transformers/archive/dcddb970176382c0fcf4521b0c0e6fc15894dfe0.zip"
                if line.startswith("transformers==")
                else line
                for line in lock.splitlines()
            )
            + "\n"
        )
    (locks / f"{profile}-py310.txt").write_text(lock)
    for filename in (
        "manifest.json",
        "dependency-validation.json",
        "vendor-constraints.txt",
        "pip-check.txt",
    ):
        shutil.copy2(env / filename, results / f"{profile}-{filename}")
for name in (
    "doctor.json",
    "model-validation.json",
    "dataset.json",
    "tests-final.log",
    "operators.log",
    "rng.log",
    "mccl-2gpu.log",
    "help-record.exit",
    "help-policy-server.exit",
):
    shutil.copy2(storage / "logs" / name, results / (Path(name).with_suffix(".txt").name if name.endswith(".log") else name))
shutil.copy2(
    storage / "outputs/annotation-acceptance/comparison.json", results / "annotation-comparison.json"
)
for path in storage.glob("assets-*.json"):
    shutil.copy2(path, results / path.name)
for policy in ("pi05", "evo1"):
    for n in (0, 1):
        folder = storage / f"outputs/accept-round{n}-{policy}"
        for name in ("state.json", "actions.json", "deployment.json", "data-report.json"):
            shutil.copy2(folder / name, results / f"round{n}-{policy}-{name}")
        steps = json.loads((folder / "state.json").read_text())["stages"]
        assert all(step["status"] == "passed" for step in steps.values())
        assert json.loads((folder / "actions.json").read_text())["finite"]
shutil.copy2(
    storage / "data/towel-replayed-annotated-round1/provenance.json",
    results / "replayed-round1-provenance.json",
)
print("Exported validation and environment locks:", results)
