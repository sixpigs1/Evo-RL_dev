"""Instrument real training updates, including optimizer and DDP synchronization."""
import json
import os
import time
from pathlib import Path

import torch

from lerobot.scripts import lerobot_train as training

original = training.update_policy
count = 0
previous = None


def measured(*args, **kwargs):
    global count, previous
    accelerator = kwargs["accelerator"]
    torch.musa.synchronize()
    start = time.perf_counter()
    result = original(*args, **kwargs)
    torch.musa.synchronize()
    end = time.perf_counter()
    elapsed = end - start
    interval = None if previous is None else end - previous
    previous = end
    count += 1
    timings = torch.tensor(
        [elapsed, interval or elapsed], device=accelerator.device, dtype=torch.float32
    )
    if accelerator.num_processes > 1:
        torch.distributed.all_reduce(timings, op=torch.distributed.ReduceOp.MAX)
    record = dict(
        step=count,
        update_s=timings[0].item(),
        interval_s=timings[1].item(),
        batch_per_rank=int(os.environ["EVORL_BENCH_BATCH"]),
        observed_batch_size=int(args[2]["action"].shape[0]),
        world_size=accelerator.num_processes,
        peak_allocated_gib=torch.musa.max_memory_allocated() / 1024**3,
        peak_reserved_gib=torch.musa.max_memory_reserved() / 1024**3,
    )
    if accelerator.is_main_process:
        dest = Path(os.environ["EVORL_BENCH_OUTPUT"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("a") as f:
            f.write(json.dumps(record) + "\n")
    return result


training.update_policy = measured
training.main()
