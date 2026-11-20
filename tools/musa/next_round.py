"""Create an explicitly replayed acceptance pool; never label it new robot experience."""

import argparse
import json
from pathlib import Path

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

p = argparse.ArgumentParser()
p.add_argument("--source", default="/workspace/storage/data/towel-small/acceptance")
p.add_argument("--output", default="/workspace/storage/data/towel-replayed-round1")
a = p.parse_args()
ds = LeRobotDataset("Elvinky/bi_so101_fold_towel_d0_acceptance", root=a.source, video_backend="pyav")
merged = merge_datasets([ds, ds], output_repo_id="local/towel-replayed-round1", output_dir=a.output)
assert merged.meta.total_episodes == 2 * ds.meta.total_episodes
assert merged.num_frames == 2 * ds.num_frames
raw = merged.hf_dataset.with_format(None)
assert len(set(raw["index"])) == merged.num_frames
assert sorted(set(raw["episode_index"])) == list(range(merged.meta.total_episodes))
labels = merged.meta.episodes.with_format(None)["episode_success"]
assert all(label == "success" for label in labels)
Path(a.output, "provenance.json").write_text(
    json.dumps(
        {
            "replayed_data": True,
            "new_robot_experience": False,
            "source": a.source,
            "episodes": merged.meta.total_episodes,
            "frames": merged.num_frames,
        },
        indent=2,
    )
)
print("Replay merge passed:", merged.num_frames)
