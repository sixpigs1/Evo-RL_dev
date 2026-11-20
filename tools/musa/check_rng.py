"""Verify MUSA RNG checkpoint roundtrip, including an indexed device."""

import tempfile
from pathlib import Path

import torch
import torch_musa  # noqa: F401

from lerobot.utils.random_utils import load_rng_state, save_rng_state, set_seed

torch.musa.set_device(4)
set_seed(123)
with tempfile.TemporaryDirectory(dir="/workspace/storage") as folder:
    path = Path(folder)
    save_rng_state(path)
    expected = torch.rand(16, device="musa:4")
    load_rng_state(path)
    actual = torch.rand(16, device="musa:4")
    assert torch.equal(expected, actual)
print("MUSA RNG safetensors roundtrip passed")
