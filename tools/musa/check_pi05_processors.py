"""Validate official checkpoint processor compatibility without allocating the VLA."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.factory import make_pre_post_processors

checkpoint = "/workspace/storage/models/pi05"
config = PreTrainedConfig.from_pretrained(checkpoint)
config.device = "cpu"
config.tokenizer_name = "/workspace/storage/models/paligemma-tokenizer"
ds = LeRobotDataset(
    "Elvinky/bi_so101_fold_towel_d0_acceptance",
    root="/workspace/storage/data/towel-small/acceptance",
    video_backend="pyav",
)
features = dataset_to_policy_features(ds.meta.features)
config.output_features = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
mapping = json.loads(Path("tools/musa/configs/round-pi05.json").read_text())["rename_map"]
pre, post = make_pre_post_processors(
    config,
    pretrained_path=checkpoint,
    preprocessor_overrides={
        "device_processor": {"device": "cpu"},
        "tokenizer_processor": {"tokenizer_name": config.tokenizer_name},
        "rename_observations_processor": {"rename_map": mapping},
        "normalizer_processor": {
            "stats": ds.meta.stats,
            "features": config.input_features | config.output_features,
            "norm_map": config.normalization_mapping,
        },
    },
    postprocessor_overrides={
        "unnormalizer_processor": {
            "stats": ds.meta.stats,
            "features": config.output_features,
            "norm_map": config.normalization_mapping,
        }
    },
)
batch = pre(next(iter(DataLoader(ds, batch_size=1, num_workers=0))))
assert batch["observation.language.tokens"].shape == (1, 200)
print("Official Pi0.5 processors passed on a real dataset batch")

action = post(torch.zeros(1, ds.meta.features["action"]["shape"][0]))
assert action.shape == (1, 12) and torch.isfinite(action).all()
