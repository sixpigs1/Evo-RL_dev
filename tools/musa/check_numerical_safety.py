"""CPU regression, including a failure on just one distributed rank.

Run with: torchrun --standalone --nproc_per_node=2 tools/musa/check_numerical_safety.py
"""
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist

from lerobot.utils.numerical_safety import require_finite


class Collective:
    device = torch.device('cpu')

    def __init__(self):
        self.process_index = dist.get_rank() if dist.is_initialized() else 0
        self.num_processes = dist.get_world_size() if dist.is_initialized() else 1

    def reduce(self, tensor, reduction):
        assert reduction == 'sum'
        if dist.is_initialized():
            dist.all_reduce(tensor)
        return tensor

    def unwrap_model(self, model):
        return model


if 'RANK' in os.environ:
    dist.init_process_group('gloo')
a = Collective()
model = torch.nn.Linear(2, 1)
batch = {'action': torch.ones(2, 2)}
directory = [tempfile.mkdtemp() if a.process_index == 0 else None]
if dist.is_initialized():
    dist.broadcast_object_list(directory, src=0)
root = Path(directory[0])
optimizer = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
common = dict(accelerator=a, policy=model, batch=batch, failure_dir=root,
              optimizer=optimizer, scheduler=scheduler,
              rng={"torch_rng_state": torch.get_rng_state()})
require_finite(torch.tensor(1.0), phase="loss", **common)
for step, (phase, bad_value) in enumerate([("loss", float("nan")), ("gradient norm", float("inf"))], 17):
    value = torch.tensor(bad_value if a.process_index == a.num_processes - 1 else 1.0)
    try:
        require_finite(value, phase=phase, step=step, **common)
    except FloatingPointError:
        failure = root / f"step-{step}"
        manifest = json.loads((failure / "complete.json").read_text())
        assert manifest["step"] == step
        assert len(manifest["sizes"]) == a.num_processes + 1
        assert all((failure / name).stat().st_size == size for name, size in manifest["sizes"].items())
        artifact = torch.load(failure / f"rank-{a.process_index}.pt", weights_only=True)
        assert artifact["phase"] == phase
        assert torch.equal(artifact["batch"]["action"], batch["action"])
        assert "torch_rng_state" in artifact["rng_before_forward"]
        snapshot = torch.load(failure / "pre-update-state.pt", weights_only=True)
        model.load_state_dict(snapshot["model"])
        optimizer.load_state_dict(snapshot["optimizer"])
        scheduler.load_state_dict(snapshot["scheduler"])
    else:
        raise AssertionError("A non-finite rank did not stop every rank")
if dist.is_initialized():
    dist.barrier()
if a.process_index == 0:
    shutil.rmtree(root)
print(json.dumps({'rank':a.process_index, 'world_size':a.num_processes, 'passed':True}), flush=True)
if dist.is_initialized():
    dist.destroy_process_group()
