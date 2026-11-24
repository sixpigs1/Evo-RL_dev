"""Reference joint controllers for contact-rich demonstrations."""

import numpy as np

from .advanced_tasks import (
    DRAWER_TRAVEL,
    ITEM_HALF,
    SEAT_X,
    drawer_open,
    fixture_held,
    item_inside,
    item_stowed,
    plug_seated,
    socket_local,
)

UP = np.array([0.0, 0.0, 1.0])
PIPER_PAD_REACH = 0.045
PIPER_DOWN = np.diag([-1.0, 1.0, -1.0])
PIPER_HANDLE_DOWN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


class AdvancedExpert:
    def __init__(self, sim):
        self.sim = sim
        self.lift_goal = None
        self.stage = self.age = self.retries = 0
        self.grasp_transform = None
        self.insertion_goal = -0.025
        self.fixture_stable = False
        self.drawer_handle = None
        self.right_home = None
        self.ik = sim.mj.MjData(sim.model)

    def _goto(self, stage):
        self.stage, self.age = stage, 0

    def _track(self, arm, position, opened, *, forward=None, rotation=None, grip_fraction=None):
        sim, model = self.sim, self.sim.model
        prefix = sim.prefixes[arm]
        width = len(sim.names) // len(sim.prefixes)
        start, stop = arm * width, (arm + 1) * width
        joints = sim.joints[start : stop - 1]
        qa, va = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
        site = model.site(prefix + "gripperframe").id
        tool_axis = 0 if sim.robot == "so101" else 2
        data = self.ik
        data.qpos[:] = sim.data.qpos
        lo, hi = model.jnt_range[joints].T
        position = np.asarray(position)
        jpos, jrot = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        for _ in range(12):
            sim.mj.mj_fwdPosition(model, data)
            sim.mj.mj_jacSite(model, data, jpos, jrot, site)
            jac = jpos[:, va]
            err = position - data.site_xpos[site]
            current = data.site_xmat[site].reshape(3, 3)
            if rotation is not None:
                angular = np.cross(current.T, rotation.T).sum(axis=0) / 2
                jac = np.vstack((jac, 0.055 * jrot[:, va]))
                err = np.r_[err, 0.055 * angular]
            elif forward is not None:
                axis = current[:, tool_axis]
                jac = np.vstack((jac, 0.055 * np.cross(jrot[:, va].T, axis).T))
                err = np.r_[err, 0.055 * (forward - axis)]
            if np.linalg.norm(err) < 0.00035:
                break
            delta = np.linalg.solve(jac.T @ jac + np.eye(len(qa)) * 0.0002, jac.T @ err)
            data.qpos[qa] = np.clip(data.qpos[qa] + np.clip(delta, -0.14, 0.14), lo, hi)
        targets = np.clip(data.qpos[qa], sim.data.qpos[qa] - 0.18, sim.data.qpos[qa] + 0.18)
        action = {}
        for name, val, mapping in zip(
            sim.names[start : stop - 1], targets, sim.mappings[start : stop - 1], strict=True
        ):
            direction = 1 if mapping.simulation[1] > mapping.simulation[0] else -1
            action[name] = float(np.interp(val, mapping.simulation[::direction], mapping.policy[::direction]))
        low, high = sim.mappings[stop - 1].policy
        fraction = float(opened) if grip_fraction is None else grip_fraction
        action[sim.names[stop - 1]] = low + fraction * (high - low)
        return action

    def _near(self, arm, goal, tolerance=0.012):
        return np.linalg.norm(self.sim.tcp(arm) - goal) < tolerance

    def _capture_plug(self, arm):
        sim = self.sim
        site = sim.data.site(sim.prefixes[arm] + "gripperframe")
        plug = sim.data.body("object")
        orientation = plug.xmat.reshape(3, 3)
        self.grasp_transform = (
            orientation.T @ (site.xpos - plug.xpos),
            orientation.T @ site.xmat.reshape(3, 3),
        )

    def _plug_pose(self, arm, approach):
        self._capture_plug(arm)
        socket = self.sim.data.site("socket_mouth")
        orientation = socket.xmat.reshape(3, 3)
        tool_pos, tool_rotation = self.grasp_transform
        origin = socket.xpos + orientation @ np.array([approach - SEAT_X, 0.0, 0.0])
        target = origin + orientation @ tool_pos
        error = np.clip(target - self.sim.tcp(arm), -0.012, 0.012)
        return self._track(arm, target + 0.25 * error, False, rotation=orientation @ tool_rotation)

    def _plug(self, arm):
        sim = self.sim
        grasp = sim.data.site("plug_grasp").xpos.copy()
        # PiPER's pads extend in front of the configured TCP.
        pickup_orientation = {"forward": -UP}
        if sim.robot == "piper":
            grasp += PIPER_PAD_REACH * UP
            pickup_orientation = {"rotation": PIPER_DOWN}
        if self.stage == 0:
            hover = grasp + 0.045 * UP
            if self._near(arm, hover) or self.age > 65:
                self._goto(1)
            return self._track(arm, hover, True, **pickup_orientation)
        if self.stage == 1:
            if self._near(arm, grasp, 0.008) or self.age > 50:
                self._goto(2)
            return self._track(arm, grasp, True, **pickup_orientation)
        if self.stage == 2:
            if sim.grasping(arm) and self.age >= 4:
                self._capture_plug(arm)
                self.lift_goal = sim.tcp(arm).copy() + 0.032 * UP
                self._goto(3)
            elif self.age > 28:
                self.retries += 1
                self._goto(0)
            return self._track(arm, grasp, False, **pickup_orientation)
        if self.stage < 6 and not sim.gripper_touches(arm, "object") and self.age > 5:
            self.retries += 1
            self._goto(0)
            return self._track(arm, grasp + 0.04 * UP, True, **pickup_orientation)
        if self.stage == 3:
            if self.age > 23:
                self._goto(4)
            return self._track(arm, self.lift_goal, False, **pickup_orientation)
        if self.stage == 4:
            seat = socket_local(sim, "plug_seat")
            if (self.age >= 20 and np.linalg.norm(seat - [-0.025, 0, 0]) < 0.012) or self.age > 65:
                self.insertion_goal = -0.025
                self._goto(5)
            return self._plug_pose(arm, -0.025)
        if self.stage == 5:
            if plug_seated(sim):
                self._goto(6)
            elif self.age > 115:
                self.retries += 1
                self._goto(4)
                return self._plug_pose(arm, -0.045)
            seat = socket_local(sim, "plug_seat")
            # Pause axial motion if the housing has not tracked the preceding target.
            if seat[0] > self.insertion_goal - 0.007 and np.linalg.norm(seat[1:]) < 0.007:
                self.insertion_goal = min(0.0002, self.insertion_goal + 0.00055)
            return self._plug_pose(arm, self.insertion_goal)
        site = sim.data.site(sim.prefixes[arm] + "gripperframe")
        return self._track(arm, site.xpos, True, rotation=site.xmat.reshape(3, 3))

    def _fixture(self):
        sim = self.sim
        target = sim.data.site("fixture_grasp").xpos.copy() + PIPER_PAD_REACH * UP
        held = sim.grasping(0, "socket", "fixture_handle")
        if held:
            slide = sim.data.joint("fixture_slide").qpos[0]
            yaw = sim.data.joint("fixture_yaw").qpos[0]
            goal = target + np.array([0.09 * np.sin(yaw), -np.clip(slide, -0.025, 0.025), 0.0])
            self.fixture_stable = fixture_held(sim)
            return self._track(0, goal, False, rotation=PIPER_DOWN)
        self.fixture_stable = False
        near = self._near(0, target, 0.013)
        descend = self.age > 36 or np.linalg.norm(sim.tcp(0)[:2] - target[:2]) < 0.028
        goal = target if near or descend else target + 0.045 * UP
        return self._track(0, goal, not near, rotation=PIPER_DOWN)

    def _drawer(self):
        sim = self.sim
        handle = sim.data.site("drawer_handle_center").xpos.copy()
        if self.drawer_handle is None:
            self.drawer_handle = handle.copy()
        original = self.drawer_handle
        arm = 0
        if self.right_home is None:
            self.right_home = {key: value for key, value in sim.read_joints().items() if key.startswith("right_")}
        if self.stage < 4:
            grip = handle + PIPER_PAD_REACH * UP
            goals = [
                grip + 0.045 * UP,
                grip,
                grip,
                original + np.array([-DRAWER_TRAVEL + 0.005, 0.0, PIPER_PAD_REACH]),
            ]
            goal = goals[self.stage]
            opening = self.stage < 2
            if self.stage < 2 and (self._near(arm, goal) and self.age > 10 or self.age > 48):
                self._goto(self.stage + 1)
            elif self.stage == 2:
                if sim.grasping(arm, "drawer", "drawer_handle") and self.age >= 5:
                    self._goto(3)
                elif self.age > 45:
                    self.retries += 1
                    self._goto(0)
            elif self.stage == 3:
                if drawer_open(sim):
                    self._goto(4)
                elif self.age > 90:
                    self.retries += 1
                    self._goto(0)
            return self._track(
                arm,
                goal,
                opening,
                rotation=PIPER_HANDLE_DOWN,
                grip_fraction=0.6 if opening else None,
            )
        if self.stage == 4:
            park = np.array([0.16, 0.22, 0.24])
            if self._near(arm, park, 0.028) or self.age > 45:
                self._goto(5)
            return self._track(arm, park, True)
        if self.stage == 11:
            right_tcp = sim.tcp(1)
            if self.age >= 25 and right_tcp[1] < -0.16 and right_tcp[2] > 0.24:
                self._goto(12)
            return self.right_home
        if self.stage < 11:
            arm = 1
            grip = sim.position("object").copy() + PIPER_PAD_REACH * UP
            drop = sim.position("drawer").copy() + [-0.018, 0, ITEM_HALF[2] + PIPER_PAD_REACH + 0.010]
            goals = [
                grip + 0.064 * UP,
                grip,
                grip,
                self.lift_goal,
                drop + 0.065 * UP,
                drop,
            ]
            if self.stage == 10:
                retreat = drop + 0.13 * UP
                if self._near(arm, retreat, 0.035) and item_stowed(sim):
                    self._goto(11)
                elif self.age > 100:
                    self.retries += 1
                    self._goto(5 if drawer_open(sim) else 0)
                return self._track(arm, retreat, True)
            goal = goals[self.stage - 5]
            if self.stage == 7:
                if sim.grasping(arm) and self.age >= 5:
                    self.lift_goal = sim.tcp(arm).copy() + 0.075 * UP
                    self._goto(8)
                elif self.age > 30:
                    self.retries += 1
                    self._goto(5)
                return self._track(arm, goal, False, rotation=PIPER_DOWN)
            if (self._near(arm, goal) and self.age >= 14) or self.age > 48:
                self._goto(self.stage + 1)
            return self._track(arm, goal, self.stage < 7 or self.stage >= 10, rotation=PIPER_DOWN)
        grip = handle + PIPER_PAD_REACH * UP
        if not item_inside(sim):
            self.retries += 1
            self.drawer_handle = None
            self._goto(0)
            return self._track(0, grip + 0.045 * UP, True, rotation=PIPER_HANDLE_DOWN)
        if self.stage == 12:
            goal = grip + 0.045 * UP
            if self._near(0, goal) or self.age > 50:
                self._goto(13)
        elif self.stage == 13:
            goal = grip
            if self._near(0, goal) and self.age >= 10 or self.age > 48:
                self._goto(14)
        elif self.stage == 14:
            goal = grip
            if sim.grasping(0, "drawer", "drawer_handle") and self.age >= 5:
                self._goto(15)
            elif self.age > 35:
                self.retries += 1
                self._goto(12)
        else:
            goal = original + np.array([0.003, 0.0, PIPER_PAD_REACH])
            if self.age > 90 and not sim.done:
                self.retries += 1
                self._goto(12)
        opening = self.stage < 14
        return self._track(
            0,
            goal,
            opening,
            rotation=PIPER_HANDLE_DOWN,
            grip_fraction=0.6 if opening else None,
        )

    def action(self):
        self.age += 1
        task = self.sim.cfg.task
        if task == "drawer_stow_close":
            return self._drawer()
        if task == "keyed_plug_insert":
            return self._plug(0)
        fixture = self._fixture()
        if not self.fixture_stable and self.stage == 0:
            return fixture
        return {**fixture, **self._plug(1)}
