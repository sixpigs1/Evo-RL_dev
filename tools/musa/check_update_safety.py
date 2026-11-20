"""Verify the real training update preserves weights/scheduler after NaN/Inf.

CPU: torchrun --standalone --nproc_per_node=2 tools/musa/check_update_safety.py
MUSA: accelerate launch --config_file tools/musa/configs/2gpu.yaml \
      tools/musa/check_update_safety.py --musa --failure-dir /workspace/storage/runs/guard-check
"""
import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from lerobot.scripts.lerobot_train import update_policy


class ToyPolicy(torch.nn.Module):
    def __init__(self, phase, rank):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))
        self.phase, self.rank = phase, rank

    def forward(self, batch):
        loss = (self.weight * batch).square().mean()
        if self.rank == 1 and self.phase == 'loss':
            loss = loss * float('nan')
        if self.rank == 1 and self.phase == 'gradient':
            loss.register_hook(lambda gradient: gradient * float('inf'))
        return loss, {}


class CPUCollective:
    device = torch.device('cpu')
    autocast = staticmethod(nullcontext)
    backward = staticmethod(lambda loss: loss.backward())
    clip_grad_norm_ = staticmethod(torch.nn.utils.clip_grad_norm_)

    def __init__(self):
        self.process_index = dist.get_rank()
        self.num_processes = dist.get_world_size()

    def reduce(self, value, reduction):
        assert reduction == 'sum'
        dist.all_reduce(value)
        return value

    def unwrap_model(self, model, **kwargs):
        return model.module


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--musa', action='store_true')
p.add_argument('--failure-dir', type=Path)
args = p.parse_args()
if args.musa:
    from accelerate import Accelerator

    a = Accelerator(step_scheduler_with_optimizer=False)
    assert a.device.type == 'musa'
else:
    dist.init_process_group('gloo')
    a = CPUCollective()
assert a.num_processes == 2, 'This check requires two ranks'
for index, phase in enumerate(['loss', 'gradient', 'finite'], 1):
    model = ToyPolicy(phase, a.process_index)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    base_scheduler = scheduler
    if args.musa:
        model, optimizer, scheduler = a.prepare(model, optimizer, scheduler)
    else:
        model = torch.nn.parallel.DistributedDataParallel(model)
    before = a.unwrap_model(model).weight.detach().clone()
    scheduler_before = base_scheduler.last_epoch
    try:
        update_policy(SimpleNamespace(), model, torch.ones(2, device=a.device), optimizer, 1.0, a,
                      scheduler, failure_dir=args.failure_dir, update_step=index)
    except FloatingPointError:
        assert phase != 'finite'
        assert torch.equal(before, a.unwrap_model(model).weight.detach())
        assert base_scheduler.last_epoch == scheduler_before
        if args.failure_dir is not None:
            directory = args.failure_dir / f'step-{index}'
            artifact = torch.load(directory / f'rank-{a.process_index}.pt', weights_only=True)
            assert artifact['phase'] == ('loss' if phase == 'loss' else 'gradient norm')
            snapshot = torch.load(directory / 'pre-update-state.pt', weights_only=True)
            assert torch.equal(snapshot['model']['weight'], before.cpu())
            assert 'optimizer' in snapshot
    else:
        assert phase == 'finite'
        assert not torch.equal(before, a.unwrap_model(model).weight.detach())
    a.reduce(torch.zeros(1, device=a.device), reduction='sum')
print(json.dumps({'rank': a.process_index, 'device': str(a.device), 'passed': True,
                  'checked': 'real update loss/gradient/finite'}), flush=True)
dist.destroy_process_group()
