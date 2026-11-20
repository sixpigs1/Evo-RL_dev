"""Verify the full approved source copy, then atomically publish its training root."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import time

import pyarrow.parquet as pq

p = argparse.ArgumentParser()
p.add_argument('--incoming', type=Path, required=True)
p.add_argument('--manifest', type=Path, required=True)
p.add_argument('--destination', type=Path, required=True)
a = p.parse_args()
if a.destination.exists():
    raise FileExistsError('Destination already exists; refusing to replace a dataset')
items = [line.split(None, 1) for line in a.manifest.read_text().splitlines()]
if len(items) != 58:
    raise ValueError('Expected 58 source files for this dataset revision')
size = 0
for expected, name in items:
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Invalid manifest path')
    path = a.incoming / relative
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
            size += len(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f'Source hash mismatch: {relative}')
info = json.loads((a.incoming / 'meta/info.json').read_text())
if (info['total_episodes'], info['total_frames']) != (40, 47398):
    raise ValueError('Dataset metadata differs from the approved training recipe')
rows = [row for file in (a.incoming/'meta/episodes').rglob('*.parquet')
        for row in pq.read_table(file, columns=['episode_index', 'length', 'episode_success']).to_pylist()]
if len(rows) != 40 or sum(r['length'] for r in rows) != 47398:
    raise ValueError('Episode table does not match frame metadata')
if any(r['episode_success'] not in ('success', 'failure', True, False) for r in rows):
    raise ValueError('Missing genuine episode outcome labels')
report = dict(verified=True, files=len(items), bytes=size, episodes=40, frames=47398,
              outcome_counts=dict(Counter(str(r['episode_success']) for r in rows)),
              manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
              checked_at=time.time(), source='4090:FoldTowel_1119/meta,data,videos')
shutil.copyfile(a.manifest, a.incoming/'source.sha256')
(a.incoming/'transfer-validation.json').write_text(json.dumps(report, indent=2)+'\n')
a.incoming.rename(a.destination)
print(json.dumps(report, indent=2))
