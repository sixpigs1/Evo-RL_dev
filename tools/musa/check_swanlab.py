"""Exercise the native logger against the installed SDK without cloud writes."""
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import swanlab
from lerobot.rl.wandb_utils import make_logger

cfg = SimpleNamespace(
    wandb=SimpleNamespace(enable=True, project='RWorld-RL', entity='sixpigs',
                          notes='Disabled-mode API compatibility check', mode='disabled', run_id=None),
    output_dir=Path('/workspace/storage/logs/swanlab-disabled-api-check'),
    job_name='disabled-api-check', env=None, policy=SimpleNamespace(type='pi05'),
    seed=1000, resume=False, dataset=SimpleNamespace(repo_id='sixpigs/FoldTowel_1119'),
    to_dict=lambda: {'purpose': 'disabled-api-compatibility-check'},
)
cfg.output_dir.mkdir(parents=True, exist_ok=True)
with patch.dict(os.environ, {'SWANLAB_API_KEY': 'disabled-test-placeholder'}):
    with patch.object(swanlab, 'login', return_value=True) as login:
        logger = make_logger(cfg)
        login.assert_called_once_with(api_key='disabled-test-placeholder', save=False)
        logger.log_dict({'loss': 1.0, 'steps': 1}, step=1)
        logger._run.finish()
    with patch.dict('sys.modules', {'swanlab': None}):
        try:
            make_logger(cfg)
        except ImportError:
            pass
        else:
            raise AssertionError('Explicit SwanLab selection must fail closed if the SDK is missing')
print('Native SwanLab API initialization/logging, no-save login, and missing-SDK failure: passed')
