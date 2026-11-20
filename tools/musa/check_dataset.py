"""Fail closed on unlabeled value data and preserve original episode labels."""

import argparse
import json
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.recording_annotations import normalize_episode_success_label

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--repo-id", required=True)
p.add_argument("--require-labels", action="store_true")
p.add_argument("--output")
p.add_argument("--require-acp", action="store_true")
a = p.parse_args()
ds = LeRobotDataset(a.repo_id, root=a.root, video_backend="pyav")
episodes = ds.meta.episodes.with_format(None)
labels = []
if "episode_success" in episodes.column_names:
    labels = [normalize_episode_success_label(x) for x in episodes["episode_success"]]
valid = len(labels) == len(episodes) and all(x is not None for x in labels)
report = {
    "repo_id": a.repo_id,
    "root": a.root,
    "episodes": len(episodes),
    "frames": len(ds),
    "valid_outcome_labels": valid,
    "label_counts": {str(k): labels.count(k) for k in set(labels)},
    "features": list(ds.meta.features),
}
print(json.dumps(report, indent=2))
if a.output:
    Path(a.output).write_text(json.dumps(report, indent=2))
if a.require_labels and not valid:
    raise ValueError("Value training requires genuine episode_success labels; no defaults are synthesized")

if a.require_acp:
    import numpy as np

    raw = ds.hf_dataset.with_format(None)
    for column in (
        "complementary_info.value_musa",
        "complementary_info.advantage_musa",
        "complementary_info.acp_indicator_musa",
    ):
        if column not in raw.column_names:
            raise KeyError(column)
        if not np.isfinite(np.asarray(raw[column])).all():
            raise ValueError(f"Nonfinite annotation: {column}")
    if not set(np.asarray(raw["complementary_info.acp_indicator_musa"]).reshape(-1).tolist()) <= {0, 1}:
        raise ValueError("ACP indicators must be binary")
