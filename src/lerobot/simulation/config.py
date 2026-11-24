"""Configuration shared by recording, evaluation and online RL."""

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.robots.config import RobotConfig


@dataclass
class SimCamera:
    width: int = 320
    height: int = 240
    mount: str = "world"
    native: str | None = None
    position: tuple[float, float, float] = (0.85, -0.85, 0.65)
    xyaxes: tuple[float, float, float, float, float, float] = (0.8, 0.6, 0, -0.25, 0.33, 0.91)
    fovy: float = 50


@dataclass
class JointMapping:
    """Two corresponding endpoints in policy units and simulator units, in that order."""

    policy: tuple[float, float]
    simulation: tuple[float, float]


@RobotConfig.register_subclass("mujoco")
@dataclass(kw_only=True)
class MujocoRobotConfig(RobotConfig):
    embodiment: str = "so101"
    task: str = "pick_place"
    instruction: str | None = None
    fps: int = 30
    episode_steps: int = 300
    seed: int = 0
    assets: Path | None = None
    so101_use_degrees: bool = False
    camera_views: dict[str, SimCamera] = field(default_factory=dict)
    joint_mappings: dict[str, JointMapping] = field(default_factory=dict)
    base_positions: list[tuple[float, float, float]] = field(default_factory=list)
    base_yaws: list[float] = field(default_factory=list)
    calibration_version: str = "menagerie-default"
    state_dir: Path | None = None
    scripted_expert: bool = False
    randomization: float = 0.015

    def __post_init__(self):
        if self.embodiment not in {"so101", "bi_so101", "piper", "bi_piper"}:
            raise ValueError(f"Unsupported MuJoCo embodiment: {self.embodiment}")
        if self.fps <= 0 or self.episode_steps <= 0 or self.randomization < 0:
            raise ValueError("fps and episode_steps must be positive and randomization nonnegative")
        for name, camera in self.camera_views.items():
            if min(camera.width, camera.height) <= 3 or not 0 < camera.fovy < 180:
                raise ValueError(f"Invalid camera configuration for {name!r}")
