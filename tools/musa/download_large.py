"""Resumable HTTP range transfer with size and SHA256 verification for public LFS files."""

import argparse
import hashlib
import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from huggingface_hub import get_hf_file_metadata, hf_hub_url

p = argparse.ArgumentParser()
p.add_argument("--repo", required=True)
p.add_argument("--revision", required=True)
p.add_argument("--filename", default="model.safetensors")
p.add_argument("--output", required=True)
p.add_argument("--workers", type=int, default=12)
a = p.parse_args()
url = hf_hub_url(a.repo, a.filename, revision=a.revision)
meta = get_hf_file_metadata(url)
size, sha = meta.size, meta.etag
if not size or not sha or len(sha) != 64:
    raise ValueError("Expected LFS file with SHA256 and content length")
dest = Path(a.output)
dest.parent.mkdir(parents=True, exist_ok=True)
part, journal = dest.with_suffix(".ranges-part"), dest.with_suffix(".ranges.json")
chunk = 16 * 1024 * 1024
lock = threading.Lock()
state = json.loads(journal.read_text()) if journal.exists() else {"sha256": sha, "size": size, "done": []}
assert state["sha256"] == sha and state["size"] == size
if not part.exists():
    candidates = list((dest.parent / ".cache/huggingface/download").glob(f"*.{sha}.*.incomplete"))
    if candidates:
        seed = max(candidates, key=lambda x: x.stat().st_size)
        shutil.copyfile(seed, part)
        state["done"] = list(range(seed.stat().st_size // chunk))
    else:
        part.touch()
fd = os.open(part, os.O_RDWR)
os.ftruncate(fd, size)


def transfer(index):
    start, end = index * chunk, min((index + 1) * chunk, size) - 1
    for attempt in range(5):
        try:
            location = get_hf_file_metadata(url).location
            with requests.get(
                location, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=(30, 90)
            ) as response:
                if response.status_code != 206:
                    raise RuntimeError(f"Range unsupported: HTTP {response.status_code}")
                if not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
                    raise RuntimeError("Incorrect Content-Range")
                offset = start
                for block in response.iter_content(1024 * 1024):
                    if offset + len(block) > end + 1:
                        raise RuntimeError("Oversized response")
                    os.pwrite(fd, block, offset)
                    offset += len(block)
                if offset != end + 1:
                    raise RuntimeError("Truncated response")
            with lock:
                state["done"].append(index)
                temp = journal.with_suffix(".tmp")
                temp.write_text(json.dumps(state))
                temp.replace(journal)
                print(f"{len(state['done'])}/{(size + chunk - 1) // chunk} chunks verified", flush=True)
            return
        except Exception:
            if attempt == 4:
                raise
            time.sleep(min(2**attempt, 10))


try:
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(transfer, [i for i in range((size + chunk - 1) // chunk) if i not in state["done"]]))
finally:
    os.close(fd)
h = hashlib.sha256()
with part.open("rb") as f:
    while block := f.read(16 * 1024 * 1024):
        h.update(block)
if h.hexdigest() != sha:
    raise ValueError("SHA256 verification failed; do not use partial weights")
part.replace(dest)
print(json.dumps({"path": str(dest), "sha256": sha, "size": size, "status": "passed"}), flush=True)
