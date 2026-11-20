"""Run with torchrun: native MCCL collectives and DDP parameter equality."""

import datetime
import json
import os

import torch
import torch.distributed as dist
import torch_musa  # noqa: F401
from torch.nn.parallel import DistributedDataParallel

rank, local, world = (int(os.environ[k]) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
torch.musa.set_device(local)
device = torch.device(f"musa:{local}")
dist.init_process_group("mccl", timeout=datetime.timedelta(seconds=180))
x = torch.tensor([rank + 1.0], device=device)
dist.all_reduce(x)
assert x.item() == world * (world + 1) / 2
parts = [torch.zeros_like(x) for _ in range(world)]
dist.all_gather(parts, torch.tensor([float(rank)], device=device))
assert [v.item() for v in parts] == list(range(world))
torch.manual_seed(1)
model = DistributedDataParallel(torch.nn.Linear(8, 4).to(device), device_ids=[local])
optim = torch.optim.AdamW(model.parameters(), lr=0.001)
for _ in range(3):
    optim.zero_grad()
    model(torch.ones(2, 8, device=device) * (rank + 1)).square().mean().backward()
    optim.step()
flat = torch.cat([p.detach().flatten() for p in model.parameters()])
weights = [torch.empty_like(flat) for _ in range(world)]
dist.all_gather(weights, flat)
assert all(torch.equal(flat, w) for w in weights)
dist.barrier()
print(json.dumps({"rank": rank, "device": str(device), "world_size": world, "status": "passed"}), flush=True)
dist.destroy_process_group()
