"""Verify live cloud run status and scalar summaries without persisting credentials."""
import argparse
import json
from pathlib import Path
import sys

import swanlab

p = argparse.ArgumentParser()
p.add_argument('--runs-dir', type=Path, required=True)
a = p.parse_args()
if sys.stdin.isatty():
    from getpass import getpass
    key = getpass('SwanLab API key: ').strip()
else:
    key = sys.stdin.readline().strip()
if not key:
    raise RuntimeError('Missing API key on stdin')
api = swanlab.Api(api_key=key)
reports = []
for profile in ('pi05', 'value'):
    state = json.loads((a.runs_dir/profile/'state.json').read_text())
    run_id = state['swanlab_url'].rstrip('/').rsplit('/', 1)[-1]
    run = api.run('sixpigs/RWorld-RL/'+run_id)
    reports.append(dict(profile=profile, run_id=run_id, url=run.url, state=run.state,
                        summary=run.summary(keys=['train/loss','train/steps','train/grad_norm','train/lr'])))
print(json.dumps(reports, indent=2))
(a.runs_dir/'cloud-verification.json').write_text(json.dumps(reports, indent=2)+'\n')
