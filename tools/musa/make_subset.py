"""Copy/reindex selected real episodes through the repository dataset implementation."""

import argparse
import json
from pathlib import Path

from lerobot.datasets.dataset_tools import split_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset

p = argparse.ArgumentParser()
p.add_argument("--source", default="/workspace/storage/data/source-towel")
p.add_argument("--output", default="/workspace/storage/data/towel-small")
p.add_argument("--episodes", default="0,1")
a = p.parse_args()
indices = [int(x) for x in a.episodes.split(",")]
ds = LeRobotDataset("Elvinky/bi_so101_fold_towel_d0", root=a.source, episodes=indices, video_backend="pyav")
result = split_dataset(ds, {"acceptance": indices}, output_dir=a.output)["acceptance"]
Path(a.output, "subset-manifest.json").write_text(
    json.dumps(
        {
            "source": a.source,
            "episodes": indices,
            "root": str(result.root),
            "repo_id": result.repo_id,
            "labels": "copied from original metadata",
        },
        indent=2,
    )
)
print(result.root)
