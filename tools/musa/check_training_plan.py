"""Parse the exact native CLI and validate approved schedules without starting training."""
import argparse
import json
import math
from pathlib import Path
import sys

from lerobot.configs import parser

p = argparse.ArgumentParser()
p.add_argument('plan', type=Path)
a = p.parse_args()
plan = json.loads(a.plan.read_text())
options = [f'--{k}={str(v).lower() if isinstance(v, bool) else v}'
           for k, v in plan['options'].items()]
sys.argv = [sys.argv[0], *options]


def validate(cfg):
    cfg.validate()
    expected = math.ceil(plan['expected_frames'] / (cfg.batch_size * plan['gpus'])) * plan['epochs']
    assert cfg.steps == expected
    assert cfg.scheduler.num_decay_steps == cfg.steps
    assert cfg.wandb.enable and cfg.wandb.project == 'RWorld-RL'
    assert cfg.policy.device == 'musa'
    assert cfg.policy.dtype == 'bfloat16'
    assert cfg.policy.freeze_vision_encoder is False
    if plan['profile'] == 'pi05':
        assert cfg.policy.gradient_checkpointing and not cfg.policy.train_expert_only
        assert cfg.scheduler.num_warmup_steps == 500
    else:
        assert cfg.policy.use_gradient_checkpointing and not cfg.policy.freeze_language_model
        assert cfg.scheduler.num_warmup_steps == 500
    (a.plan.parent / 'parsed-config.json').write_text(json.dumps(cfg.to_dict(), indent=2)+'\n')
    print(json.dumps(dict(profile=plan['profile'], steps=cfg.steps, epochs=plan['epochs'],
                          per_rank_batch=cfg.batch_size, global_batch=cfg.batch_size*plan['gpus'],
                          lr=cfg.optimizer.lr, warmup=cfg.scheduler.num_warmup_steps,
                          decay=cfg.scheduler.num_decay_steps, status='passed')))


if plan['profile'] == 'value':
    from lerobot.scripts import lerobot_value_train  # Register configs and plugins used by the real CLI.
    from lerobot.configs.value_train import ValueTrainPipelineConfig

    @parser.wrap()
    def check(cfg: ValueTrainPipelineConfig):
        validate(cfg)
else:
    from lerobot.scripts import lerobot_train
    from lerobot.configs.train import TrainPipelineConfig

    @parser.wrap()
    def check(cfg: TrainPipelineConfig):
        validate(cfg)

check()
