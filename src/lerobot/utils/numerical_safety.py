"""Stop every DDP rank before a non-finite loss or gradient reaches the optimizer."""

from pathlib import Path

import torch


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu(item) for item in value)
    return value


def require_finite(
    value, *, phase, accelerator, policy, batch, step=None, failure_dir=None, rng=None, optimizer=None
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
        torch.save({"phase": phase, "step": step, "rank": rank, "value": _cpu(value),
                    "batch": _cpu(batch), "rng_before_forward": rng,
                    "nonfinite_gradients": bad_gradients}, directory / f"rank-{rank}.pt")
        if rank == 0:
            # Weights have not been updated: retain this exact model for replay,
            # rather than relying on a potentially much older periodic checkpoint.
            snapshot = {"model": _cpu(unwrapped.state_dict())}
            if optimizer is not None:
                snapshot["optimizer"] = _cpu(optimizer.state_dict())
            torch.save(snapshot, directory / "pre-update-state.pt")
        # Keep peers alive while rank zero writes a potentially large snapshot.
        accelerator.reduce(torch.ones((), dtype=torch.int32, device=accelerator.device), reduction="sum")
        message += f"; diagnostic batch/RNG and pre-update state saved in {directory}"
    raise FloatingPointError(message)
