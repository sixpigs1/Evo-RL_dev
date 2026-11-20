"""Detach approved training jobs; credentials arrive on stdin and remain in memory."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

# Keep the tested root runtime while granting the host account's users group file access.
if os.geteuid() == 0:
    os.setgid(100)
os.umask(0o002)
ROOT = Path(__file__).resolve().parents[2]


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def validate(plan):
    dataset = Path(plan['options']['dataset.root'])
    info = json.loads((dataset / 'meta/info.json').read_text())
    ready = json.loads((dataset / 'transfer-validation.json').read_text())
    if info['total_frames'] != 47398 or info['total_episodes'] != 40 or not ready['verified']:
        raise RuntimeError('Formal training requires the verified full 40-episode dataset')
    if len(plan['devices']) != plan['gpus']:
        raise ValueError('Device count does not match world size')


def worker(plan_path):
    plan = json.loads(plan_path.read_text())
    validate(plan)
    if not os.environ.get('SWANLAB_API_KEY'):
        raise RuntimeError('SWANLAB_API_KEY must be supplied in the process environment')
    secret = os.environ['SWANLAB_API_KEY']
    storage = Path('/workspace/storage')
    locks = []
    for device in sorted(plan['devices']):
        lock = (storage / f'.gpu-test-{device}.lock').open('w')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    devices = ','.join(map(str, plan['devices']))
    subprocess.run([sys.executable, str(ROOT / 'tools/musa/gpu_guard.py'),
                    '--devices', devices, '--timeout', '0'], check=True)
    out = Path(plan['options']['output_dir'])
    if out.exists():
        raise FileExistsError(f'Fresh output directory required: {out}')
    options = [f'--{k}={str(v).lower() if isinstance(v, bool) else v}'
               for k, v in plan['options'].items()]
    module = 'lerobot_value_train' if plan['profile'] == 'value' else 'lerobot_train'
    command = [str(ROOT / 'tools/musa/run.sh'), plan['profile'], 'accelerate', 'launch',
               '--config_file', str(ROOT / f"tools/musa/configs/{plan['gpus']}gpu.yaml"),
               '--main_process_port', str(plan['port']), '-m', f'lerobot.scripts.{module}', *options]
    env = os.environ.copy()
    env.update(MUSA_VISIBLE_DEVICES=devices, PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1',
               SWANLAB_DISABLE_GIT='True', SWANLAB_DISABLE_UPDATE_CHECK='True')
    state_path = plan_path.parent / 'state.json'
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors='replace', bufsize=1)
    state = dict(status='running', pid=proc.pid, supervisor_pid=os.getpid(),
                 started_at=time.time(), plan=str(plan_path), log=str(plan_path.parent/'train.log'))
    write_json(state_path, state)
    with (plan_path.parent / 'train.log').open('w', buffering=1) as log:
        for line in proc.stdout:
            line = line.replace(secret, '[REDACTED]')
            log.write(line)
            if 'step:' in line or 'https://swanlab.cn/' in line:
                state['updated_at'] = time.time()
                state['last_training_line'] = line.strip()
                urls = re.findall(r'https://swanlab.cn/[^\s\x1b]+', line)
                if urls:
                    state['swanlab_url'] = urls[-1]
                write_json(state_path, state)
    rc = proc.wait()
    state.update(status='completed' if rc == 0 else 'failed', returncode=rc, finished_at=time.time())
    write_json(state_path, state)
    return rc


p = argparse.ArgumentParser()
g = p.add_mutually_exclusive_group(required=True)
g.add_argument('--start', type=Path, help='Directory containing per-job plan.json files')
g.add_argument('--worker', type=Path)
a = p.parse_args()
if a.worker:
    try:
        raise SystemExit(worker(a.worker))
    except Exception as error:
        write_json(a.worker.parent / 'state.json', dict(status='failed', error=str(error), time=time.time()))
        raise
else:
    if sys.stdin.isatty():
        from getpass import getpass

        key = getpass('SwanLab API key: ').strip()
    else:
        key = sys.stdin.readline().strip()
    if not key:
        raise RuntimeError('Provide API key on stdin; it is never saved to disk')
    start_lock = (a.start / '.start.lock').open('w')
    fcntl.flock(start_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (a.start / 'launch.json').exists():
        raise FileExistsError('Jobs have already been launched; inspect launch.json')
    plans = sorted(a.start.glob('*/plan.json'))
    if len(plans) != 2:
        raise ValueError('Expected the Pi0.5 and ValueFunction plans')
    jobs = [json.loads(path.read_text()) for path in plans]
    devices = [d for job in jobs for d in job['devices']]
    if len(set(devices)) != len(devices) or len({job['port'] for job in jobs}) != 2:
        raise ValueError('Training jobs must have disjoint GPUs and rendezvous ports')
    for job in jobs:
        validate(job)
        if Path(job['options']['output_dir']).exists():
            raise FileExistsError('Fresh output directories required for both jobs')
    if any((path.parent / 'state.json').exists() for path in plans):
        raise FileExistsError('A job has already been launched; inspect all states before retrying')
    launch = []
    for path, job in zip(plans, jobs):
        env = os.environ.copy()
        env['SWANLAB_API_KEY'] = key
        with (path.parent / 'supervisor.log').open('w') as log:
            proc = subprocess.Popen([str(ROOT / 'tools/musa/run.sh'), job['profile'], 'python',
                                     str(Path(__file__).resolve()), '--worker', str(path)],
                                    cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        launch.append(dict(profile=job['profile'], supervisor_pid=proc.pid,
                           devices=job['devices'], plan=str(path)))
    write_json(a.start / 'launch.json', launch)
    print(json.dumps(launch, indent=2))
