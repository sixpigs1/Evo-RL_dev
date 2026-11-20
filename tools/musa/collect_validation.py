# safetensors.safe_open is not a dict and exposes keys(), not iteration.
# ruff: noqa: SIM118
"""Summarize actual checkpoints, optimizer counters, numerical logs and weight updates."""

import ast
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

storage = Path("/workspace/storage")
rows = []
for task in ("pi05", "evo1-stage1", "evo1-stage2", "value"):
    for n in (1, 2):
        name = f"accept-{task}-{n}gpu"
        out = storage / "outputs" / name
        last = out / "checkpoints/last"
        step = json.loads((last / "training_state/training_step.json").read_text())["step"]
        expected = 105 if n == 1 else 20
        assert step >= expected, (name, step)
        with safe_open(
            last / "training_state/optimizer_state.safetensors", framework="pt", device="cpu"
        ) as f:
            keys = [k for k in f.keys() if k.endswith("/step")]
            counters = [int(f.get_tensor(k).item()) for k in keys[:8]]
        assert counters and max(counters) >= expected, (name, counters)
        half = 50 if n == 1 else 10
        full = 100 if n == 1 else 20
        before = out / f"checkpoints/{half:06d}/pretrained_model/model.safetensors"
        after = out / f"checkpoints/{full:06d}/pretrained_model/model.safetensors"
        with (
            safe_open(before, framework="pt", device="cpu") as x,
            safe_open(after, framework="pt", device="cpu") as y,
        ):
            candidates = [
                k
                for k in x.keys()
                if k.endswith("weight") and (not task.startswith("evo1") or "action_head" in k)
            ]
            candidates = candidates[:8]
            changed = sum(not torch.equal(x.get_tensor(k), y.get_tensor(k)) for k in candidates)
        assert changed, (name, candidates)
        logs = "".join(
            (storage / f"logs/{name}{suffix}.log").read_text()
            for suffix in ("", "-resume")
            if (storage / f"logs/{name}{suffix}.log").exists()
        )
        losses = [float(v) for v in re.findall(r"loss:([^\s]+)", logs)]
        assert losses and all(torch.isfinite(torch.tensor(losses)))
        peaks = [
            ast.literal_eval(x)
            for x in re.findall(
                r"MUSA memory \[rank, peak_allocated_bytes, peak_reserved_bytes\]: (\[.*\])", logs
            )
        ]
        result_file = storage / f"logs/{name}-result.json"
        result = json.loads(result_file.read_text())
        assert result["status"] == "passed", result
        rows.append(
            {
                "task": task,
                "world_size": n,
                "last_step": step,
                "optimizer_step_samples": counters,
                "changed_weight_probes": changed,
                "weight_probes": len(candidates),
                "finite_loss_records": len(losses),
                "peak_memory_samples": peaks,
                "parameter_sync_checked": "DDP parameter samples agree" in logs,
                "status": "passed",
            }
        )
report = {"required_world_sizes": [1, 2], "rows": rows, "status": "passed"}
(storage / "logs/model-validation.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
