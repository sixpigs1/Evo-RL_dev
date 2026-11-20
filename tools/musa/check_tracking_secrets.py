"""Check that the supplied runtime key is absent from managed text artifacts."""
import json
from pathlib import Path
import sys

if sys.stdin.isatty():
    from getpass import getpass
    key = getpass('SwanLab API key: ').strip().encode()
else:
    key = sys.stdin.readline().strip().encode()
if not key:
    raise RuntimeError('Provide key only on stdin')
roots = [Path('/workspace/storage/runs/foldtowel40-20261120'),
         Path('/workspace/storage/outputs/foldtowel40-pi05-4gpu-20261120'),
         Path('/workspace/storage/outputs/foldtowel40-value-2gpu-20261120')]
files = [p for root in roots for p in root.rglob('*')
         if p.is_file() and p.stat().st_size < 8*1024*1024]
files += [p for p in (Path('/root/.netrc'),) if p.exists()]
hits = [str(p) for p in files if key in p.read_bytes()]
report = dict(status='passed' if not hits else 'failed', scanned_files=len(files), matches=hits)
(roots[0]/'credential-artifact-check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
if hits:
    raise SystemExit(1)
