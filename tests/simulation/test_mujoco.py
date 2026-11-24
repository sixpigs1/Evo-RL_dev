import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from lerobot.simulation.backend import MujocoSimulation
from lerobot.simulation.config import MujocoRobotConfig, SimCamera
from lerobot.simulation.gym_env import JointTargetEnv
from lerobot.simulation.tasks import TASKS

ROBOTS = ("so101", "bi_so101", "piper", "bi_piper")


@pytest.fixture
def assets():
    value = os.environ.get("LEROBOT_MENAGERIE_ROOT")
    if not value:
        pytest.skip("Set LEROBOT_MENAGERIE_ROOT to the pinned Menagerie assets")
    return Path(value)


@pytest.mark.parametrize("robot", ROBOTS)
def test_mapping_snapshot_and_isolation(robot, assets):
    cfg = MujocoRobotConfig(embodiment=robot, assets=assets, task="pick_place", episode_steps=3)
    sim = MujocoSimulation(cfg)
    other = MujocoSimulation(cfg)
    try:
        assert len(sim.names) == (6 if robot.endswith("so101") else 7) * len(sim.prefixes)
        before = sim.snapshot()
        command = dict(zip(sim.names, (m.policy[1] for m in sim.mappings), strict=True))
        sim.step(command)
        expected = sim.snapshot()["integration"]
        assert expected != before["integration"]
        assert other.snapshot()["integration"] == before["integration"]
        sim.restore(before)
        sim.step(command)
        np.testing.assert_allclose(sim.snapshot()["integration"], expected, atol=1e-10, rtol=0)
        sim.reset(41)
        first = sim.snapshot()["integration"]
        sim.reset(41)
        np.testing.assert_array_equal(first, sim.snapshot()["integration"])
    finally:
        sim.close()
        other.close()


@pytest.mark.parametrize(
    "robot,task",
    [
        (robot, name)
        for robot in ROBOTS
        for name, task in TASKS.items()
        if robot.startswith("bi_") or task.arms == 1
    ],
)
def test_task_reset_and_timeout(robot, task, assets):
    env = JointTargetEnv.for_simulation(
        MujocoRobotConfig(embodiment=robot, assets=assets, task=task, episode_steps=1)
    )
    try:
        obs, _ = env.reset(seed=12)
        assert env.observation_space.contains(obs)
        obs, _, terminated, truncated, info = env.step(obs["agent_pos"])
        assert terminated or truncated
        assert info["is_success"] == terminated
        assert env.observation_space.contains(obs)
    finally:
        env.close()


@pytest.mark.parametrize("robot", ROBOTS)
def test_rgb_headless(robot, assets):
    if not os.environ.get("MUJOCO_GL"):
        pytest.skip("Set MUJOCO_GL to the installed headless backend")
    env = JointTargetEnv.for_simulation(
        MujocoRobotConfig(
            embodiment=robot, assets=assets, camera_views={"front": SimCamera(width=80, height=64)}
        )
    )
    try:
        obs, _ = env.reset()
        image = obs["pixels"]["front"]
        assert image.shape == (64, 80, 3)
        assert image.dtype == np.uint8 and image.std() > 0
    finally:
        env.close()


def test_joint_target_online_processors(assets):
    import torch

    from lerobot.envs.configs import HILSerlRobotEnvConfig
    from lerobot.processor import TransitionKey, create_transition
    from lerobot.rl.gym_manipulator import make_processors, make_robot_env, step_env_and_process_transition
    from lerobot.teleoperators.utils import TeleopEvents

    cfg = HILSerlRobotEnvConfig(
        name="joint_targets",
        robot=MujocoRobotConfig(assets=assets, task="reach", episode_steps=2),
    )
    env, teleop = make_robot_env(cfg)
    try:
        obs, info = env.reset(seed=4)
        obs_processor, action_processor = make_processors(env, teleop, cfg)
        transition = obs_processor(create_transition(observation=obs, info=info))
        requested = torch.full((1, len(env.names)), 200.0)
        result = step_env_and_process_transition(env, transition, requested, obs_processor, action_processor)
        assert not result[TransitionKey.INFO][TeleopEvents.IS_INTERVENTION]
        assert not result[TransitionKey.TRUNCATED]
        assert torch.all(result[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"] == 100.0)
        result = step_env_and_process_transition(env, result, requested, obs_processor, action_processor)
        assert result[TransitionKey.TRUNCATED]
    finally:
        env.close()


def test_joint_target_takeover_preserves_named_order(assets):
    import torch

    from lerobot.processor import TransitionKey, create_transition
    from lerobot.simulation.gym_env import JointTargetIntervention
    from lerobot.teleoperators.utils import TeleopEvents

    class JointLeader:
        def get_teleop_events(self):
            return {TeleopEvents.IS_INTERVENTION: True}

        def get_action(self):
            return {key: float(i) for i, key in reversed(list(enumerate(env.names)))}

    env = JointTargetEnv.for_simulation(MujocoRobotConfig(embodiment="bi_piper", assets=assets))
    env.teleop = JointLeader()
    requested = torch.zeros((1, len(env.names)))
    result = JointTargetIntervention(env, True)(create_transition(action=requested))
    torch.testing.assert_close(
        result[TransitionKey.ACTION], torch.arange(len(env.names), dtype=torch.float32)
    )
