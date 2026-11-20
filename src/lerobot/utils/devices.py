"""Native accelerator helpers; importing LeRobot does not require torch_musa."""

import importlib.util

import torch


def musa_available() -> bool:
    if not hasattr(torch, "musa") and importlib.util.find_spec("torch_musa") is not None:
        import torch_musa  # noqa: F401
    return hasattr(torch, "musa") and torch.musa.is_available()


def require_musa(device: str | torch.device) -> None:
    if str(device).split(":")[0] == "musa":
        if not musa_available():
            raise RuntimeError("MUSA requested but torch_musa or its GPU runtime is unavailable")
        index = torch.device(device).index
        if index is not None and index >= torch.musa.device_count():
            raise ValueError(f"MUSA device index out of range: {device}")


def synchronize(device: str | torch.device) -> None:
    device = torch.device(device)
    if device.type in ("cuda", "musa", "xpu"):
        getattr(torch, device.type).synchronize(device)


def validate_runtime_device(requested: str, actual: torch.device) -> None:
    """Do not silently execute an explicitly requested MUSA job on another backend."""
    require_musa(requested)
    if str(requested).split(":")[0] == "musa" and actual.type != "musa":
        raise RuntimeError(f"Requested {requested}, but Accelerate selected {actual}")


def report_musa_memory(accelerator) -> None:
    """Collect peak training memory across ranks for acceptance reports."""
    if accelerator.device.type != "musa":
        return
    import logging

    values = torch.tensor(
        [[accelerator.process_index, torch.musa.max_memory_allocated(), torch.musa.max_memory_reserved()]],
        dtype=torch.int64,
        device=accelerator.device,
    )
    peaks = accelerator.gather(values).cpu().tolist()
    if accelerator.is_main_process:
        logging.info("MUSA memory [rank, peak_allocated_bytes, peak_reserved_bytes]: %s", peaks)


def report_musa_parameter_sync(model, accelerator) -> None:
    """Compare deterministic parameter samples without gathering entire VLA weights."""
    if accelerator.device.type != "musa" or accelerator.num_processes == 1:
        return
    import logging

    samples = []
    for parameter in accelerator.unwrap_model(model).parameters():
        if parameter.requires_grad and parameter.numel():
            flat = parameter.detach().reshape(-1)
            samples.append(flat[:: max(1, flat.numel() // 8)][:8].float())
    local = torch.cat(samples).unsqueeze(0)
    gathered = accelerator.gather(local)
    if not torch.isfinite(gathered).all() or not torch.equal(gathered, gathered[:1].expand_as(gathered)):
        raise RuntimeError("MUSA DDP parameter samples differ across ranks or contain nonfinite values")
    if accelerator.is_main_process:
        logging.info("MUSA DDP parameter samples agree: %s values per rank", local.numel())
