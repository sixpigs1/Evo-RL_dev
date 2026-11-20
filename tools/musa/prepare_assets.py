"""Download pinned public inputs; never invent episode outcome labels."""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

p = argparse.ArgumentParser()
p.add_argument("--storage", default="/workspace/storage")
p.add_argument(
    "--only", choices=["dataset", "internvl3", "pi05", "value_vision", "value_language", "all"], default="all"
)
a = p.parse_args()
storage = Path(a.storage)
sources = json.loads(Path(__file__).with_name("sources.json").read_text())
api = HfApi()
manifest = {}
for key in ("dataset", "internvl3", "pi05", "value_vision", "value_language"):
    if a.only not in ("all", key):
        continue
    source = sources[key]
    repo = source["repo_id"]
    kind = "dataset" if key == "dataset" else "model"
    info = api.repo_info(repo, repo_type=kind, revision=source.get("revision"))
    dest = storage / ("data/source-towel" if key == "dataset" else f"models/{key}")
    patterns = None
    if key == "dataset":
        # First two episodes occupy files 000 and 001 in this pinned dataset.
        patterns = [
            "meta/*",
            "meta/**",
            "data/chunk-000/file-00[01].parquet",
            "videos/*/chunk-000/file-00[01].mp4",
        ]
    snapshot_download(
        repo,
        repo_type=kind,
        revision=info.sha,
        local_dir=dest,
        allow_patterns=patterns,
        ignore_patterns=["*.bin", "*.msgpack", "*.h5", "*.onnx"],
        max_workers=4,
    )
    manifest[key] = {"repo_id": repo, "revision": info.sha, "path": str(dest)}
    (storage / f"assets-{key}.json").write_text(json.dumps(manifest[key], indent=2))
    print(json.dumps(manifest[key]), flush=True)
