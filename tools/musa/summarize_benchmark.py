"""Summarize synchronized real-data speed measurements after warmup."""
import argparse
import json
import re
import statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("root", type=Path)
p.add_argument("--warmup", type=int, default=10)
p.add_argument("--steps", type=int, default=35)
a = p.parse_args()
if not 0 <= a.warmup < a.steps:
    p.error("Require 0 <= warmup < steps")
results = []
status_path = a.root / "status.log"
status = status_path.read_text() if status_path.exists() else ""
for task in ("evo1-stage2", "pi05"):
    for world in (1, 2, 4):
        for batch in (1, 2, 4, 8):
            name = f"{task}-g{world}-b{batch}"
            log = a.root / f"{name}.log"
            path = a.root / f"{name}.jsonl"
            rows = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
            measured = [r for r in rows if r["step"] > a.warmup]
            content = log.read_text(errors="replace") if log.exists() else ""
            rc = re.search(rf"END {re.escape(name)} rc=(\d+)", status)
            if rc and rc[1] == "0" and len(rows) == a.steps:
                state = "passed"
            elif "out of memory" in content.lower():
                state = "oom"
            else:
                state = "failed" if rc else "pending"
            result = dict(
                name=name, task=task, gpus=world, batch_per_rank=batch,
                global_batch=batch * world, status=state, completed_steps=len(rows),
                measured_steps=len(measured),
            )
            if measured:
                intervals = [r["interval_s"] for r in measured]
                interval = statistics.mean(intervals)
                result.update(
                    mean_step_s=interval,
                    median_step_s=statistics.median(intervals),
                    std_step_s=statistics.pstdev(intervals),
                    p90_step_s=(statistics.quantiles(intervals, n=10, method="inclusive")[8]
                                if len(intervals) > 1 else interval),
                    mean_update_s=statistics.mean(r["update_s"] for r in measured),
                    samples_per_s=world * batch / interval,
                    peak_allocated_gib=max(r["peak_allocated_gib"] for r in rows),
                    peak_reserved_gib=max(r["peak_reserved_gib"] for r in rows),
                )
            memory = re.search(r"MUSA memory .*: (\[\[.*\]\])", content)
            if memory:
                ranks = json.loads(memory[1])
                result["rank_memory_bytes"] = ranks
                result["peak_allocated_gib"] = max(r[1] for r in ranks) / 1024**3
                result["peak_reserved_gib"] = max(r[2] for r in ranks) / 1024**3
            params = re.search(r"num_learnable_params=(\d+)", content)
            result["trainable_parameters"] = int(params[1]) if params else None
            result["finite_logged_loss_and_grad"] = not bool(
                re.search(r"(?:loss|grdn):(?:nan|[-+]?inf)\b", content, re.I)
            )
            result["ddp_parameter_samples_agree"] = (
                "MUSA DDP parameter samples agree" in content if world > 1 else None
            )
            if state == "passed" and (
                not result["finite_logged_loss_and_grad"]
                or (world > 1 and not result["ddp_parameter_samples_agree"])
            ):
                result["status"] = "failed_validation"
            results.append(result)

(a.root / "summary.json").write_text(
    json.dumps(dict(warmup_steps=a.warmup, results=results), indent=2) + "\n"
)
for result in results:
    metrics = [
        f"{result[key]:.3f}"
        for key in ("mean_step_s", "mean_update_s", "samples_per_s", "peak_allocated_gib")
        if key in result
    ]
    print(result["name"], result["status"], *metrics)
