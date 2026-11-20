"""Emit the actual runtime, library paths and a small BF16 numerical check."""

import argparse
import importlib.metadata as md
import json
import os
import platform
from pathlib import Path

import torch
import torch_musa  # noqa: F401

p = argparse.ArgumentParser()
p.add_argument("--device", default="musa:0")
p.add_argument("--output")
a = p.parse_args()
d = torch.device(a.device)
torch.musa.set_device(d)
torch.manual_seed(123)
x = torch.randn(32, 32)
y = torch.randn(32, 32)
ref = x @ y
with torch.autocast("musa", dtype=torch.bfloat16):
    result = x.to(d) @ y.to(d)
torch.musa.synchronize()
error = (result.float().cpu() - ref).abs().max().item()
assert torch.allclose(result.float().cpu(), ref, atol=0.15, rtol=0.04), error
z = torch.randn(8, 8, device=d, requires_grad=True)
z.square().mean().backward()
assert torch.isfinite(z.grad).all()
loaded = sorted(
    {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if "libmccl" in line}
)
assert loaded and all("stub" not in x.lower() for x in loaded), loaded
report = {
    "platform": platform.platform(),
    "python": platform.python_version(),
    "packages": {n: md.version(n) for n in ("torch", "torch_musa", "torchvision")},
    "runtime_versions": {"torch": torch.__version__, "torch_musa": torch_musa.__version__},
    "device": str(d),
    "count": torch.musa.device_count(),
    "mccl": str(torch.musa.mccl.version()),
    "mccl_paths": loaded,
    "bf16_max_abs_error": error,
    "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
    "status": "passed",
}
print(json.dumps(report, indent=2))
if a.output:
    Path(a.output).write_text(json.dumps(report, indent=2))
