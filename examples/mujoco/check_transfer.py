"""Check checkpoint input and action parity on the same recorded real observation."""

import argparse
import json
import time
from pathlib import Path

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.simulation.backend import control_names
from lerobot.simulation.compatibility import check_policy_features
from lerobot.simulation.config import MujocoRobotConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--embodiment", required=True)
    parser.add_argument("--rename-map", type=json.loads, default={})
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--eager", action="store_true", help="Disable optional PI model compilation")
    args = parser.parse_args()
    dataset = LeRobotDataset(args.dataset, root=args.dataset_root)
    expected_names = control_names(MujocoRobotConfig(embodiment=args.embodiment))
    for key in ("action", "observation.state"):
        if tuple(dataset.features[key]["names"]) != expected_names:
            raise ValueError(f"Recorded joint order differs for {key}")
    overrides = [f"--device={args.device}"]
    if args.eager:
        overrides.append("--compile_model=false")
    config = PreTrainedConfig.from_pretrained(args.checkpoint, cli_overrides=overrides)
    config.pretrained_path = args.checkpoint
    config.device = args.device
    features = dataset_to_policy_features(dataset.features)
    check_policy_features(
        config, {args.rename_map.get(key, key): feature for key, feature in features.items()}
    )

    frame = dataset[args.frame]
    raw = {"observation.state": frame["observation.state"].numpy()}
    for key in dataset.meta.camera_keys:
        image = (frame[key].permute(1, 2, 0) * 255).round().clamp(0, 255).to(torch.uint8)
        raw[key] = image.numpy()
    robot = prepare_observation_for_inference(
        raw.copy(), torch.device(args.device), frame["task"], dataset.meta.robot_type
    )
    gym = preprocess_observation(
        {
            "agent_pos": raw["observation.state"],
            "pixels": {key.removeprefix("observation.images."): raw[key] for key in dataset.meta.camera_keys},
        }
    )
    gym["task"] = [frame["task"]]
    for key in raw:
        torch.testing.assert_close(robot[key].cpu(), gym[key], rtol=0, atol=0)

    policy = make_policy(config, ds_meta=dataset.meta, rename_map=args.rename_map).eval()
    overrides = {
        "device_processor": {"device": args.device},
        "rename_observations_processor": {"rename_map": args.rename_map},
    }
    if args.tokenizer:
        overrides["tokenizer_processor"] = {"tokenizer_name": str(args.tokenizer.resolve())}
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=args.checkpoint,
        preprocessor_overrides=overrides,
    )

    def predict(observation):
        torch.manual_seed(42)
        policy.reset()
        with torch.inference_mode():
            return postprocessor(policy.select_action(preprocessor(observation))).cpu()

    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)
    started = time.perf_counter()
    first = predict(robot)
    warmup_s = time.perf_counter() - started
    started = time.perf_counter()
    second = predict(gym)
    policy_step_s = time.perf_counter() - started
    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)
    print(
        json.dumps(
            {
                "raw_input_keys": sorted(raw),
                "action_shape": list(first.shape),
                "max_abs_action_difference": (first - second).abs().max().item(),
                "first_action_s": warmup_s,
                "second_action_s": policy_step_s,
                "peak_cuda_mib": (
                    torch.cuda.max_memory_allocated(args.device) / 1024**2
                    if torch.device(args.device).type == "cuda"
                    else None
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
