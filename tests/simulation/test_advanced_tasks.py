import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")

from lerobot.simulation.advanced_tasks import SEAT_X, drawer_closed, fixture_held, plug_seated
from lerobot.simulation.backend import MujocoSimulation
from lerobot.simulation.config import MujocoRobotConfig
from lerobot.simulation.expert import ScriptedExpert

CASES = (
    ("keyed_plug_insert", "bi_so101"),
    ("drawer_stow_close", "bi_piper"),
    ("bimanual_stabilized_insert", "bi_piper"),
)


@pytest.fixture
def assets():
    root = os.environ.get("LEROBOT_MENAGERIE_ROOT")
    if not root:
        pytest.skip("Set LEROBOT_MENAGERIE_ROOT to the pinned asset tree")
    return Path(root)


def _simulation(assets, task, robot, **kwargs):
    return MujocoSimulation(MujocoRobotConfig(task=task, embodiment=robot, assets=assets, **kwargs))


def _pose(sim, name, position, quaternion=(1, 0, 0, 0)):
    sim.data.joint(f"{name}_free").qpos[:] = (*position, *quaternion)
    sim.data.joint(f"{name}_free").qvel[:] = 0
    sim.mj.mj_forward(sim.model, sim.data)


@pytest.mark.parametrize("task,robot", CASES)
@pytest.mark.parametrize("difficulty", ("nominal", "precision"))
def test_scene_reset_and_reference_control(task, robot, difficulty, assets):
    sim = _simulation(assets, task, robot, task_difficulty=difficulty)
    try:
        sim.reset(seed=12)
        snapshot = sim.snapshot()
        assert sim.model.opt.timestep <= sim.task.max_physics_dt
        assert sim.substeps * sim.model.opt.timestep == pytest.approx(1 / sim.cfg.fps)
        assert set(sim.observation()) == set(sim.names)
        expert = ScriptedExpert(sim)
        for _ in range(15):
            sim.step(expert.action())
        assert sim.data.time == pytest.approx(15 / sim.cfg.fps)
        assert sum(sim.stage_frames) == 15
        progress = sim.snapshot()
        sim.restore(snapshot)
        assert sim.steps == 0
        sim.restore(progress)
        assert sim.stage_frames == progress["stage_frames"]
        sim.reset(seed=12)
        np.testing.assert_array_equal(sim.snapshot()["integration"], snapshot["integration"])
    finally:
        sim.close()


@pytest.mark.parametrize(
    "offset,rotation",
    [
        ((-0.01, 0, 0), (1, 0, 0, 0)),
        ((0, 0.008, 0), (1, 0, 0, 0)),
        ((0, 0, 0), (0, 1, 0, 0)),
    ],
)
def test_plug_rejects_half_insert_one_pin_and_reversed_key(assets, offset, rotation):
    sim = _simulation(assets, *CASES[0], randomization=0)
    try:
        mouth = sim.data.site("socket_mouth").xpos.copy()
        _pose(sim, "object", mouth - [SEAT_X, 0, 0] + offset, rotation)
        assert not plug_seated(sim)
    finally:
        sim.close()


@pytest.mark.parametrize("difficulty", ("nominal", "precision"))
def test_released_connector_remains_seated_under_gravity(assets, difficulty):
    sim = _simulation(assets, *CASES[0], randomization=0, task_difficulty=difficulty)
    try:
        mouth = sim.data.site("socket_mouth").xpos.copy()
        _pose(sim, "object", mouth - [SEAT_X, 0, 0])
        sim.phase = 2
        action = sim.read_joints()
        for _ in range(20):
            sim.step(action)
            if sim.done:
                break
        assert sim.success
        assert plug_seated(sim)
        assert any(
            "socket_lower_lip" in (sim.model.geom(c.geom1).name, sim.model.geom(c.geom2).name)
            for c in sim.data.contact
        )
    finally:
        sim.close()


def test_insert_hold_is_consecutive_and_replays(assets):
    sim = _simulation(assets, *CASES[0], randomization=0)
    try:
        mouth = sim.data.site("socket_mouth").xpos.copy()
        sim.phase = 2  # Exercise the final predicate after its prerequisites.
        _pose(sim, "object", mouth - [SEAT_X, 0, 0])
        assert plug_seated(sim)
        for _ in range(9):
            sim.advance_task()
        before = sim.snapshot()
        assert not sim.success and sim.stage_hold == 9
        _pose(sim, "object", mouth - [SEAT_X + 0.01, 0, 0])
        sim.advance_task()
        assert not sim.success and sim.stage_hold == 0
        sim.restore(before)
        sim.advance_task()
        assert sim.success
    finally:
        sim.close()


def test_drawer_requires_correct_item_full_containment_and_close_order(assets):
    sim = _simulation(assets, *CASES[1], randomization=0)
    try:
        closed = sim.position("drawer").copy()
        sim.advance_task()
        assert sim.phase == 0
        sim.data.joint("drawer_slide").qpos[0] = 0.12
        sim.mj.mj_forward(sim.model, sim.data)
        sim.advance_task()
        assert sim.phase == 1
        tray = sim.position("drawer").copy()
        _pose(sim, "distractor", tray + [0, 0, 0.015])
        sim.advance_task()
        assert sim.phase == 1
        _pose(sim, "distractor", [0.3, -0.1, 0.010])
        _pose(sim, "object", tray + [0.056, 0, 0.016])
        sim.advance_task()
        assert sim.phase == 1  # Centre inside, corners outside.
        _pose(sim, "object", tray + [0, 0, 0.016])
        sim.advance_task()
        assert sim.phase == 2 and not drawer_closed(sim)
        sim.data.joint("drawer_slide").qpos[0] = 0.02
        _pose(sim, "object", closed + [-0.02, 0, 0.016])
        assert not drawer_closed(sim)
        _pose(sim, "object", [0.3, -0.1, 0.011])
        sim.advance_task()
        assert sim.phase == 0 and sim.stage_restarts == 1
        sim.data.joint("drawer_slide").qpos[0] = 0.12
        sim.mj.mj_forward(sim.model, sim.data)
        sim.advance_task()
        _pose(sim, "object", sim.position("drawer").copy() + [0, 0, 0.016])
        sim.advance_task()
        sim.data.joint("drawer_slide").qpos[0] = 0
        _pose(sim, "object", closed + [0, 0, 0.016])
        assert drawer_closed(sim)
        for _ in range(10):
            sim.advance_task()
        assert sim.success
    finally:
        sim.close()


def test_drawer_reference_completes_all_stages(assets):
    sim = _simulation(assets, *CASES[1], episode_steps=900)
    try:
        sim.reset(seed=0)
        expert = ScriptedExpert(sim)
        while not sim.done:
            sim.step(expert.action())
        assert sim.success and drawer_closed(sim)
        assert all(sim.stage_frames)
        assert sim.stage_restarts == 0
    finally:
        sim.close()


def test_mobile_fixture_cannot_pass_without_physical_support(assets):
    sim = _simulation(assets, *CASES[2], randomization=0)
    try:
        mouth = sim.data.site("socket_mouth").xpos.copy()
        _pose(sim, "object", mouth - [SEAT_X, 0, 0])
        assert plug_seated(sim) and not fixture_held(sim)
        sim.phase = 3
        sim.stage_hold = 9
        sim.advance_task()
        assert sim.phase == 0 and sim.stage_hold == 0 and not sim.success
    finally:
        sim.close()
