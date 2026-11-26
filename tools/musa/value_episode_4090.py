"""Read-only CUDA inference and visualization for a complete exported PiStar06 checkpoint."""
import argparse
import csv
import json
from pathlib import Path
from unittest.mock import patch

import draccus
import numpy as np
import torch
from torch.utils.data import default_collate
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.value_train import ValueTargetsConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.value_infer_viz import _export_overlay_videos
from lerobot.values.pistar06 import modeling_pistar06 as pistar



def load_policy(checkpoint, backbones):
    info = json.loads((checkpoint / "pistar06_save_info.json").read_text())
    if info["weights_mode"] != "full" or info["excluded_prefixes"]:
        raise ValueError("This loader requires a full checkpoint")
    raw_config = json.loads((checkpoint / "config.json").read_text())
    raw_config.update(
        device="cuda",
        vision_repo_id=str(backbones / "value_vision"),
        language_repo_id=str(backbones / "value_language"),
    )
    config = draccus.decode(PreTrainedConfig, raw_config)

    def vision_from_config(repo_id, *, revision, torch_dtype):
        config = AutoConfig.from_pretrained(repo_id, revision=revision, local_files_only=True)
        return AutoModel.from_config(config, torch_dtype=torch_dtype)

    def language_from_config(repo_id, revision, dtype):
        config = AutoConfig.from_pretrained(repo_id, revision=revision, local_files_only=True)
        if any(name.endswith("ForCausalLM") for name in config.architectures or []):
            return AutoModelForCausalLM.from_config(config, torch_dtype=dtype).model
        return AutoModel.from_config(config, torch_dtype=dtype)

    # The full export includes both backbones. Strict loading replaces every parameter.
    with patch.object(pistar, "_load_language_model", language_from_config):
        with patch.object(pistar.AutoModel, "from_pretrained", side_effect=vision_from_config):
            policy = pistar.Pistar06Policy.from_pretrained(checkpoint, config=config, strict=True)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=checkpoint,
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "tokenizer_processor": {"tokenizer_name": config.language_repo_id},
        },
    )
    return policy, preprocessor, config


def episode_batches(dataset, episode, batch_size):
    # Decode nearby frames together to avoid seeking back to the keyframe for every sample.
    window_size = 16 * batch_size
    for start in range(0, len(dataset), window_size):
        items = [dataset.hf_dataset[i] for i in range(start, min(start + window_size, len(dataset)))]
        if any(int(item["episode_index"]) != episode for item in items):
            raise RuntimeError("Unexpected episode in selected frames")
        timestamps = [float(item["timestamp"]) for item in items]
        videos = dataset._query_videos({key: timestamps for key in dataset.meta.video_keys}, episode)
        for offset, item in enumerate(items):
            for key, frames in videos.items():
                item[key] = frames if len(items) == 1 else frames[offset]
            item["task"] = dataset.meta.tasks.iloc[int(item["task_index"])].name
        for offset in range(0, len(items), batch_size):
            yield default_collate(items[offset:offset + batch_size])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backbones", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    print(f"CUDA device: {torch.cuda.get_device_name()}", flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    dataset = LeRobotDataset(
        "sixpigs/FoldTowel_1119", root=args.dataset, episodes=[args.episode],
        download_videos=False, video_backend="pyav",
    )
    policy, preprocessor, config = load_policy(args.checkpoint, args.backbones)
    frames, values = [], []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in episode_batches(dataset, args.episode, args.batch_size):
            frames.extend(batch["frame_index"].reshape(-1).tolist())
            predictions = policy.predict_value(preprocessor(batch))
            values.extend(predictions.float().cpu().reshape(-1).tolist())
            if len(values) % 128 == 0 or len(values) == len(dataset):
                print(f"Predicted {len(values)}/{len(dataset)} frames", flush=True)
    values = np.asarray(values)
    expected = int(dataset.meta.episodes[args.episode]["length"])
    if frames != list(range(expected)) or len(values) != expected or not np.isfinite(values).all():
        raise RuntimeError("Expected one finite value for every frame in order")
    seconds = np.asarray(frames) / dataset.fps
    training_config = json.loads((args.checkpoint / "value_train_config.json").read_text())
    targets_config = draccus.decode(ValueTargetsConfig, training_config["targets"])
    if config.reward.mode != "legacy":
        raise ValueError("This evaluation expects the recorded legacy target definition")
    targets = policy._legacy_training_targets(
        dataset, targets_config, np.full(expected, args.episode), np.asarray(frames)
    )
    difference = values - targets
    with (args.output / "values.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_index", "time_seconds", "value", "training_target", "residual"])
        writer.writerows(zip(frames, seconds, values, targets, difference, strict=True))
    summary = {
        "episode": args.episode, "frames": expected, "fps": dataset.fps,
        "device": torch.cuda.get_device_name(), "first": float(values[0]),
        "last": float(values[-1]), "min": float(values.min()), "max": float(values.max()),
        "mean": float(values.mean()), "checkpoint": str(args.checkpoint),
        "target_first": float(targets[0]), "target_last": float(targets[-1]),
        "mae": float(np.abs(difference).mean()),
        "rmse": float(np.sqrt(np.mean(difference ** 2))),
        "min_frame": int(np.argmin(values)), "max_frame": int(np.argmax(values)),
        "success_label": dataset.meta.episodes[args.episode][targets_config.success_field],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    field = "complementary_info.value"
    if field in dataset.hf_dataset.column_names:
        dataset.hf_dataset = dataset.hf_dataset.remove_columns([field])
    dataset.hf_dataset = dataset.hf_dataset.add_column(field, values.tolist())
    _export_overlay_videos(
        dataset=dataset, value_field=field, advantage_field="", indicator_field="",
        viz_episodes=str(args.episode), video_key="observation.images.right_top",
        video_keys=None, output_dir=args.output / "viz", overwrite=False,
        vcodec="h264", frame_storage_mode="disk", smooth_window=1,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
