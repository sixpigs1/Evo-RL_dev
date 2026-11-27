"""Causal smoothing for absolute joint targets."""

import math

import numpy as np


class JointTargetFilter:
    """Apply an EMA to arm joints while preserving gripper commands."""

    def __init__(self, joint_names: list[str], alpha: float):
        if not math.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("Joint target alpha must be in (0, 1]")
        self.alpha = alpha
        self.arm_joints = np.array([not name.endswith("gripper.pos") for name in joint_names], dtype=bool)
        self.previous: np.ndarray | None = None

    def reset(self) -> None:
        self.previous = None

    def step(self, target: np.ndarray, measured_position: np.ndarray) -> np.ndarray:
        target = np.asarray(target, dtype=np.float32)
        measured_position = np.asarray(measured_position, dtype=np.float32)
        if target.shape != self.arm_joints.shape or measured_position.shape != target.shape:
            raise ValueError("Joint targets and measured positions must match the configured joint order")
        if not np.isfinite(target).all() or not np.isfinite(measured_position).all():
            raise ValueError("Joint targets and measured positions must be finite")
        previous = measured_position if self.previous is None else self.previous
        result = target.copy()
        joints = self.arm_joints
        result[joints] = previous[joints] + self.alpha * (target[joints] - previous[joints])
        self.previous = result.copy()
        return result
