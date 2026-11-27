"""Run a trained policy on a robot without collecting a dataset."""

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.robots import RobotConfig, bi_so_follower, make_robot_from_config  # noqa: F401
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.joint_target_filter import JointTargetFilter


@dataclass
class DeployConfig:
    robot: RobotConfig
    task: str
    reference_info: Path
    policy: PreTrainedConfig | None = None
    rename_map: dict[str, str] = field(default_factory=dict)
    fps: float = 30
    joint_target_alpha: float = 1.0

    def __post_init__(self):
        path = parser.get_path_arg("policy")
        if not path:
            raise ValueError("Provide --policy.path")
        self.policy = PreTrainedConfig.from_pretrained(path, cli_overrides=parser.get_cli_overrides("policy"))
        self.policy.pretrained_path = Path(path)
        if not self.task.strip() or not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Provide a task and a positive finite FPS")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


def check_features(features, trained_features, policy, rename_map):
    for key in (OBS_STATE, ACTION):
        if features[key]["names"] != trained_features[key]["names"]:
            raise ValueError(f"Live {key} joint names/order differ from training data")
    live_images = {name for name in features if name.startswith("observation.images.")}
    expected = {name for name in policy.input_features if name.startswith("observation.images.")}
    mapped = {rename_map.get(name, name) for name in live_images}
    if mapped != expected or len(live_images) != len(expected):
        raise ValueError(f"Live camera mapping {sorted(mapped)} differs from expected {sorted(expected)}")


@parser.wrap()
def deploy(cfg: DeployConfig):
    logging.basicConfig(level=logging.INFO)
    robot = make_robot_from_config(cfg.robot)
    features = {
        **hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=False),
        **hw_to_dataset_features(robot.action_features, ACTION, use_video=False),
    }
    reference = json.loads(cfg.reference_info.read_text())
    check_features(features, reference["features"], cfg.policy, cfg.rename_map)
    is_evo1 = cfg.policy.type == "evo1"
    if is_evo1 and not math.isclose(cfg.fps, reference["fps"]):
        raise ValueError(f"Evo-1 control FPS {cfg.fps} differs from reference FPS {reference['fps']}")
    if is_evo1:
        for name, feature in features.items():
            if not name.startswith("observation.images."):
                continue
            policy_name = cfg.rename_map.get(name, name)
            height, width, channels = feature["shape"]
            trained_shape = cfg.policy.input_features[policy_name].shape
            if (channels, height, width) != tuple(trained_shape):
                raise ValueError(f"Live image shape for {name} differs from Evo-1 checkpoint shape {trained_shape}")
    joint_filter = JointTargetFilter(features[ACTION]["names"], cfg.joint_target_alpha)
    autocast_dtype = torch.bfloat16 if is_evo1 else None
    device = torch.device(cfg.policy.device)
    policy = make_policy(cfg.policy, ds_meta=SimpleNamespace(features=features), rename_map=cfg.rename_map)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )
    try:
        robot.connect()
        logging.info("Policy active. Press Ctrl+C to stop.")
        period = 1 / cfg.fps
        last_overrun_warning = -math.inf
        while True:
            start = time.perf_counter()
            observation = build_dataset_frame(features, robot.get_observation(), prefix=OBS_STR)
            action = predict_action(
                observation, policy, device, pre, post, cfg.policy.use_amp, cfg.task, robot.robot_type,
                autocast_dtype=autocast_dtype,
            )
            if action.numel() != len(features[ACTION]["names"]) or not bool(torch.isfinite(action).all()):
                raise FloatingPointError("Policy output has an invalid shape or non-finite values")
            if cfg.joint_target_alpha < 1:
                target = action.detach().cpu().reshape(-1).numpy()
                action = torch.from_numpy(joint_filter.step(target, observation[OBS_STATE]))
            robot.send_action(make_robot_action(action, features))
            now = time.perf_counter()
            elapsed = now - start
            if elapsed > period and now - last_overrun_warning >= 5:
                logging.warning("Control tick took %.1f ms (budget %.1f ms)", elapsed * 1000, period * 1000)
                last_overrun_warning = now
            time.sleep(max(0, period - elapsed))
    except KeyboardInterrupt:
        logging.info("Deployment stopped by operator.")
    finally:
        if robot.is_connected:
            robot.disconnect()
        elif isinstance(robot, bi_so_follower.BiSOFollower):
            # A failed second connection can leave the first arm connected.
            for arm in (robot.left_arm, robot.right_arm):
                if arm.is_connected:
                    arm.disconnect()


def main():
    register_third_party_plugins()
    deploy()
