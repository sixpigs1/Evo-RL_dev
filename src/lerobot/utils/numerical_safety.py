"""Stop every DDP rank before a non-finite loss or gradient reaches the optimizer."""

import json
import os
import time
from pathlib import Path

import torch
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu(item) for item in value)
    return value


def inspect_gradient_bucket(checks, bucket):
    checks.append(torch.isfinite(bucket.buffer()).all())
    return allreduce_hook(None, bucket)


def _atomic_write(path, write):
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        write(stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def require_finite(
    value, *, phase, accelerator, policy, batch, step=None, failure_dir=None, rng=None, optimizer=None, scheduler=None
):
    # All ranks must participate, including ranks whose local value is finite.
    # Raising on only one rank before backward would strand the other DDP ranks.
    finite = torch.isfinite(value.detach()).all().to(device=accelerator.device, dtype=torch.int32)
    count = accelerator.reduce(finite, reduction="sum")
    if int(count.item()) == accelerator.num_processes:
        return
    rank = accelerator.process_index
    message = f"Non-finite {phase} at update {step}; optimizer update aborted on all ranks"
    if failure_dir is not None:
        directory = Path(failure_dir) / f"step-{step}"
        directory.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(policy)
        bad_gradients = [name for name, p in unwrapped.named_parameters()
                         if p.grad is not None and not torch.isfinite(p.grad).all()]
        artifact = dict(phase=phase, step=step, rank=rank, value=_cpu(value),
                        batch=_cpu(batch), rng_before_forward=rng,
                        nonfinite_gradients=bad_gradients)
        _atomic_write(directory / f"rank-{rank}.pt", lambda stream: torch.save(artifact, stream))
        manifest = directory / "complete.json"
        deadline = time.monotonic() + 900
        if rank == 0:
            snapshot = {"model": _cpu(unwrapped.state_dict()), "phase": phase, "step": step}
            if optimizer is not None:
                snapshot["optimizer"] = _cpu(optimizer.state_dict())
            if scheduler is not None:
                snapshot["scheduler"] = scheduler.state_dict()
            state = directory / "pre-update-state.pt"
            _atomic_write(state, lambda stream: torch.save(snapshot, stream))
            files = [state, *(directory / f"rank-{i}.pt" for i in range(accelerator.num_processes))]
            while not all(file.exists() for file in files):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Incomplete diagnostic snapshot in {directory}")
                time.sleep(1)
            payload = dict(step=step, phase=phase, sizes={file.name: file.stat().st_size for file in files})
            _atomic_write(manifest, lambda stream: stream.write(json.dumps(payload).encode()))
        while not manifest.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Incomplete diagnostic snapshot in {directory}")
            time.sleep(1)
        message += f"; complete diagnostic snapshot in {directory}"
    raise FloatingPointError(message)
