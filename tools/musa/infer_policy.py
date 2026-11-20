"""Load a saved policy and its processors and check real-batch action inference."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.devices import require_musa, synchronize

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--dataset-root", required=True)
p.add_argument("--repo-id", required=True)
p.add_argument("--device", default="musa:0")
p.add_argument("--output", required=True)
a = p.parse_args()
require_musa(a.device)
cfg = PreTrainedConfig.from_pretrained(a.checkpoint)
cfg.device = a.device
cfg.pretrained_path = Path(a.checkpoint)
ds = LeRobotDataset(a.repo_id, root=a.dataset_root, video_backend="pyav")
train_config = Path(a.checkpoint) / "train_config.json"
rename_map = json.loads(train_config.read_text()).get("rename_map", {}) if train_config.exists() else {}
policy = make_policy(cfg, ds_meta=ds.meta, rename_map=rename_map).eval()
overrides = {"device_processor": {"device": a.device}}
if cfg.type == "pi05":
    overrides["tokenizer_processor"] = {"tokenizer_name": cfg.tokenizer_name}
pre, post = make_pre_post_processors(cfg, pretrained_path=a.checkpoint, preprocessor_overrides=overrides)
batch = next(iter(DataLoader(ds, batch_size=1, num_workers=0)))
policy.reset()
with (
    torch.inference_mode(),
    torch.autocast(
        torch.device(a.device).type,
        dtype=torch.bfloat16,
        enabled=cfg.use_amp and torch.device(a.device).type in ("cuda", "musa"),
    ),
):
    action = post(policy.select_action(pre(batch)))
synchronize(a.device)
assert action.shape == (1, ds.meta.features["action"]["shape"][0]), action.shape
assert torch.isfinite(action).all()
Path(a.output).write_text(
    json.dumps(
        {
            "checkpoint": a.checkpoint,
            "device": a.device,
            "shape": list(action.shape),
            "finite": True,
            "action": action.cpu().tolist(),
        },
        indent=2,
    )
)
