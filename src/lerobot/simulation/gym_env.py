"""Gymnasium adapters for evaluation and absolute joint-target HIL-SERL."""

import gymnasium as gym
import numpy as np

from lerobot.processor import InterventionActionProcessorStep, TransitionKey
from lerobot.teleoperators.utils import TeleopEvents

from .robot import MujocoRobot


class JointTargetEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, robot, instruction, episode_steps, *, simulated, teleop=None, reset_pose=None):
        super().__init__()
        self.robot = robot
        self.simulated = simulated
        self.teleop = teleop
        self.reset_pose = reset_pose
        self.task = instruction
        self._max_episode_steps = episode_steps
        self.render_mode = "rgb_array" if simulated else None
        self.metadata = {**self.metadata, "render_fps": robot.config.fps if simulated else 30}
        self.names = tuple(robot.action_features)
        self.images = {
            key: shape for key, shape in robot.observation_features.items() if isinstance(shape, tuple)
        }
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (len(self.names),), dtype=np.float32)
        observation = {"agent_pos": gym.spaces.Box(-np.inf, np.inf, (len(self.names),), dtype=np.float32)}
        if self.images:
            observation["pixels"] = gym.spaces.Dict(
                {key: gym.spaces.Box(0, 255, shape, dtype=np.uint8) for key, shape in self.images.items()}
            )
        self.observation_space = gym.spaces.Dict(observation)
        self.steps = 0
        self._joint_positions = {}
        self.listener = None
        self.events = {}
        self.intervening = False
        self.target_ranges = None

    def set_target_ranges(self, ranges):
        if not ranges:
            return
        missing = set(self.names) ^ ranges.keys()
        if missing:
            raise ValueError(f"Joint-target ranges differ from follower keys: {sorted(missing)}")
        endpoints = np.array([ranges[name] for name in self.names], dtype=np.float32)
        if endpoints.shape != (len(self.names), 2) or not np.isfinite(endpoints).all():
            raise ValueError("Joint-target ranges require two finite endpoints per joint")
        lower, upper = endpoints.T
        if (upper <= lower).any():
            raise ValueError("Joint-target range upper endpoints must exceed lower endpoints")
        self.target_ranges = lower, upper
        self.action_space = gym.spaces.Box(-1, 1, (len(self.names),), dtype=np.float32)

    def encode_targets(self, targets):
        if self.target_ranges is None:
            return targets
        lower, upper = self.target_ranges
        return np.clip(2 * (targets - lower) / (upper - lower) - 1, -1, 1)

    @classmethod
    def for_simulation(cls, config, teleop=None, *, target_ranges=None):
        from .tasks import TASKS

        env = cls(
            MujocoRobot(config),
            config.instruction or TASKS[config.task].instruction,
            config.episode_steps,
            simulated=True,
            teleop=teleop,
        )
        env.set_target_ranges(target_ranges)
        return env

    def configure_intervention(self):
        if not callable(getattr(self.teleop, "get_teleop_events", None)):
            from lerobot.utils.control_utils import init_keyboard_listener

            self.listener, self.events = init_keyboard_listener(
                episode_success_key="s", episode_failure_key="f"
            )
            if self.listener is None:
                raise RuntimeError(
                    "Joint teleop requires an event-capable device or an available keyboard listener"
                )

    def intervention_events(self):
        if self.teleop is None:
            return {}
        get_events = getattr(self.teleop, "get_teleop_events", None)
        if callable(get_events):
            return get_events()
        if self.events.pop("toggle_intervention", False):
            self.intervening = not self.intervening
        outcome = self.events.pop("episode_outcome", None)
        terminated = self.events.pop("exit_early", False) or outcome is not None
        return {
            TeleopEvents.IS_INTERVENTION: self.intervening,
            TeleopEvents.SUCCESS: outcome == "success",
            TeleopEvents.TERMINATE_EPISODE: terminated,
        }

    def task_description(self):
        return self.task

    def _observe(self):
        raw = self.robot.get_observation()
        self._joint_positions = {name: raw[name] for name in self.names}
        result = {"agent_pos": np.array(list(self._joint_positions.values()), dtype=np.float32)}
        if self.images:
            result["pixels"] = {key: raw[key] for key in self.images}
        return result

    def get_raw_joint_positions(self):
        return self._joint_positions.copy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if not self.robot.is_connected:
            self.robot.connect()
        self.steps = 0
        self.intervening = False
        if self.simulated:
            self.robot.reset(seed)
            if options and "snapshot" in options:
                self.robot.sim.restore(options["snapshot"])
                self.steps = self.robot.sim.steps
                self.robot._start_trace()
        elif self.reset_pose is not None:
            from lerobot.utils.robot_utils import precise_sleep

            if len(self.reset_pose) != len(self.names):
                raise ValueError("The physical reset pose must contain one target per joint")
            observation = self.robot.get_observation()
            start = [observation[name] for name in self.names]
            for targets in np.linspace(start, self.reset_pose, 51)[1:]:
                self.robot.send_action(dict(zip(self.names, targets, strict=True)))
                precise_sleep(0.02)
        return self._observe(), {}

    def step(self, action):
        action = np.asarray(action)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError(f"Expected finite joint targets with shape {self.action_space.shape}")
        if self.target_ranges is not None:
            lower, upper = self.target_ranges
            action = lower + (np.clip(action, -1, 1) + 1) * (upper - lower) / 2
        executed = self.robot.send_action(dict(zip(self.names, action, strict=True)))
        self.steps += 1
        success = bool(self.simulated and self.robot.sim.success)
        truncated = not success and self.steps >= self._max_episode_steps
        info = {
            "is_success": success,
            "executed_action": np.array([executed[k] for k in self.names], np.float32),
        }
        info["executed_action"] = self.encode_targets(info["executed_action"])
        if self.simulated:
            info["phase"] = self.robot.sim.phase
        info["failure_reason"] = "timeout" if truncated else ""
        return self._observe(), float(success), success, truncated, info

    def render(self):
        if self.simulated:
            return self.robot.sim.image()
        obs = self.robot.get_observation()
        return obs[next(iter(self.images))] if self.images else None

    def close(self):
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
        try:
            if self.robot.is_connected:
                self.robot.disconnect()
        finally:
            if self.teleop is not None and self.teleop.is_connected:
                self.teleop.disconnect()


class JointTargetIntervention(InterventionActionProcessorStep):
    """Apply the existing HIL event semantics to named absolute joint targets."""

    def __init__(self, env, terminate_on_success):
        super().__init__(terminate_on_success=terminate_on_success)
        self.env = env

    def __call__(self, transition):
        info = {**transition[TransitionKey.INFO]}
        for key in (
            TeleopEvents.IS_INTERVENTION,
            TeleopEvents.SUCCESS,
            TeleopEvents.TERMINATE_EPISODE,
            TeleopEvents.RERECORD_EPISODE,
        ):
            info.pop(key, None)
        info.update(self.env.intervention_events())
        transition = {**transition, TransitionKey.INFO: info}
        if info.get(TeleopEvents.IS_INTERVENTION, False) and self.env.teleop is not None:
            action = self.env.teleop.get_action()
            missing = set(self.env.names) - action.keys()
            if missing:
                raise ValueError(f"Teleoperator is missing joint targets: {sorted(missing)}")
            complementary = dict(transition[TransitionKey.COMPLEMENTARY_DATA])
            targets = np.asarray([action[key] for key in self.env.names], np.float32)
            complementary["teleop_action"] = self.env.encode_targets(targets)
            transition[TransitionKey.COMPLEMENTARY_DATA] = complementary
        result = super().__call__(transition)
        action = result[TransitionKey.ACTION].reshape(-1)
        result[TransitionKey.ACTION] = action
        result[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"] = action
        return result
