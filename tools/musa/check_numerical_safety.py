"""CPU regression, including a failure on just one distributed rank.

Run with: torchrun --standalone --nproc_per_node=2 tools/musa/check_numerical_safety.py
"""
import json
import os
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
with tempfile.TemporaryDirectory() as directory:
    common = dict(accelerator=a, policy=model, batch=batch, step=17, failure_dir=directory,
                  rng={'torch_rng_state': torch.get_rng_state()})
    require_finite(torch.tensor(1.0), phase='loss', **common)
    for phase, bad_value in [('loss', float('nan')), ('gradient norm', float('inf'))]:
        value = torch.tensor(bad_value if a.process_index == a.num_processes - 1 else 1.0)
        try:
            require_finite(value, phase=phase, **common)
        except FloatingPointError:
            artifact = torch.load(Path(directory)/'step-17'/f'rank-{a.process_index}.pt', weights_only=True)
            assert artifact['phase'] == phase
            assert torch.equal(artifact['batch']['action'], batch['action'])
            assert 'torch_rng_state' in artifact['rng_before_forward']
        else:
            raise AssertionError('A non-finite rank did not stop every rank')
print(json.dumps({'rank':a.process_index, 'world_size':a.num_processes, 'passed':True}), flush=True)
if dist.is_initialized():
    dist.destroy_process_group()
