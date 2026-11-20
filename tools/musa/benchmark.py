"""Benchmark real pretrained models; synchronized timing, no checkpoint writes."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument('--task', choices=['pi05', 'evo1-stage2', 'value'], required=True)
p.add_argument('--gpus', type=int, choices=[1, 2, 4], default=1)
p.add_argument('--steps', type=int, default=35)
p.add_argument('--batch-size', type=int, default=1, help='Per-rank batch; accumulation is one')
p.add_argument('--storage', default='/workspace/storage')
p.add_argument('--dataset-root')
p.add_argument('--dataset-repo-id', default='Elvinky/bi_so101_fold_towel_d0_acceptance')
p.add_argument('--stage1-checkpoint')
p.add_argument('--output', required=True)
a = p.parse_args()
storage, out = Path(a.storage), Path(a.output)
if a.steps < 1 or a.batch_size < 1:
    p.error('steps and batch-size must be positive')
if a.task == 'evo1-stage2' and not a.stage1_checkpoint:
    p.error('Stage2 requires a real Stage1 checkpoint')
if out.exists() or Path(str(out) + '.jsonl').exists():
    p.error('Use a fresh output path to avoid mixing measurements')
devices = os.environ.get('MUSA_VISIBLE_DEVICES', ','.join(map(str, range(a.gpus))))
physical = sorted(map(int, devices.split(',')))
if len(set(physical)) != a.gpus or len(physical) != a.gpus or any(i < 0 for i in physical):
    p.error('MUSA_VISIBLE_DEVICES must select exactly --gpus distinct physical cards')
os.environ['MUSA_VISIBLE_DEVICES'] = devices
storage.mkdir(parents=True, exist_ok=True)
locks = [(storage / f'.gpu-test-{i}.lock').open('w') for i in physical]
while True:
    acquired = []
    try:
        for lock in locks:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired.append(lock)
        idle = subprocess.run([sys.executable, str(ROOT / 'tools/musa/gpu_guard.py'),
                               '--timeout', '0', '--devices', devices], capture_output=True)
        if idle.returncode == 0:
            break
        if b'TimeoutError' not in idle.stderr:
            raise RuntimeError(idle.stderr.decode(errors='replace'))
    except BlockingIOError:
        pass
    for lock in acquired:
        fcntl.flock(lock, fcntl.LOCK_UN)
    time.sleep(5)

profile = 'evo1' if a.task == 'evo1-stage2' else a.task
values = {
    'dataset.repo_id': a.dataset_repo_id,
    'dataset.root': a.dataset_root or storage / 'data/towel-small/acceptance',
    'dataset.video_backend': 'pyav', 'dataset.image_transforms.enable': False,
    'batch_size': a.batch_size, 'num_workers': 0, 'steps': a.steps,
    'save_freq': a.steps, 'log_freq': 1, 'output_dir': out,
    'save_checkpoint': False, 'wandb.enable': False,
}
if profile != 'value':
    values |= {'eval_freq': 0, 'policy.device': 'musa', 'policy.push_to_hub': False,
               'policy.use_peft': False}
if profile == 'value':
    values |= {
        'value.type': 'pistar06', 'value.device': 'musa', 'value.dtype': 'bfloat16',
        'value.use_amp': True, 'value.use_gradient_checkpointing': True,
        'value.push_to_hub': False, 'value.freeze_vision_encoder': False,
        'value.freeze_language_model': False,
        'value.vision_repo_id': storage / 'models/value_vision',
        'value.language_repo_id': storage / 'models/value_language',
    }
elif profile == 'pi05':
    values |= {
        'policy.path': storage / 'models/pi05', 'policy.dtype': 'bfloat16',
        'policy.gradient_checkpointing': True, 'policy.compile_model': False,
        'policy.use_amp': True, 'policy.freeze_vision_encoder': False,
        'policy.train_expert_only': False,
        'policy.tokenizer_name': storage / 'models/paligemma-tokenizer',
        'rename_map': json.dumps({
            'observation.images.left_wrist': 'observation.images.left_wrist_0_rgb',
            'observation.images.right_wrist': 'observation.images.right_wrist_0_rgb',
            'observation.images.right_front': 'observation.images.base_0_rgb',
        }),
    }
else:
    values |= {
        'policy.path': a.stage1_checkpoint, 'resume_pretrain': True,
        'policy.training_stage': 'stage2', 'policy.use_flash_attn': False,
        'policy.enable_vlm_gradient_checkpointing': True,
        'policy.vlm_model_name': storage / 'models/internvl3',
    }
os.environ['EVORL_BENCH_TASK'] = a.task
os.environ['EVORL_BENCH_OUTPUT'] = str(out) + '.jsonl'
os.environ['EVORL_BENCH_BATCH'] = str(a.batch_size)
options = [f'--{k}={str(v).lower() if isinstance(v, bool) else v}' for k, v in values.items()]
subprocess.run([str(ROOT / 'tools/musa/run.sh'), profile, 'accelerate', 'launch',
                '--config_file', str(ROOT / f'tools/musa/configs/{a.gpus}gpu.yaml'),
                str(ROOT / 'tools/musa/benchmark_step.py'), *options], cwd=ROOT, check=True)
