"""Reference joint controllers for contact-rich demonstrations."""

import numpy as np

from .advanced_tasks import (
    DRAWER_TRAVEL,
    ITEM_HALF,
    SEAT_X,
    drawer_open,
    fixture_held,
    item_stowed,
    plug_seated,
)

UP = np.array([0.0, 0.0, 1.0])
FORWARD = np.array([1.0, 0.0, 0.0])


class AdvancedExpert:
    def __init__(self, sim, drawer_track):
        self.sim = sim
        self.drawer_track = drawer_track
        self.lift_goal = None
        self.stage = self.age = self.retries = 0
        self.grasp_transform = None
        self.fixture_target = None
        self.drawer_handle = None
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
        error = np.clip(target - self.sim.tcp(arm), -0.016, 0.016)
        return self._track(arm, target + 0.38 * error, False, rotation=orientation @ tool_rotation)

    def _plug(self, arm):
        sim = self.sim
        grasp = sim.data.site("plug_grasp").xpos.copy()
        # PiPER's pads extend in front of the configured TCP.
        if sim.robot == "piper":
            grasp += 0.021 * UP
        if self.stage == 0:
            hover = grasp + 0.045 * UP
            if self._near(arm, hover) or self.age > 65:
                self._goto(1)
            return self._track(arm, hover, True, forward=-UP)
        if self.stage == 1:
            if self._near(arm, grasp, 0.008) or self.age > 50:
                self._goto(2)
            return self._track(arm, grasp, True, forward=-UP)
        if self.stage == 2:
            if sim.grasping(arm) and self.age >= 4:
                self._capture_plug(arm)
                self.lift_goal = sim.tcp(arm).copy() + 0.032 * UP
                self._goto(3)
            elif self.age > 28:
                self.retries += 1
                self._goto(0)
            return self._track(arm, grasp, False, forward=-UP)
        if self.stage < 6 and not sim.gripper_touches(arm, "object") and self.age > 5:
            self.retries += 1
            self._goto(0)
            return self._track(arm, grasp + 0.04 * UP, True, forward=-UP)
        if self.stage == 3:
            if self.age > 23:
                self._goto(4)
            return self._track(arm, self.lift_goal, False, forward=-UP)
        if self.stage == 4:
            if self.age > 35:
                self._goto(5)
            return self._plug_pose(arm, -0.025)
        if self.stage == 5:
            if plug_seated(sim):
                self._goto(6)
            elif self.age > 75:
                self.retries += 1
                self._goto(4)
                return self._plug_pose(arm, -0.045)
            return self._plug_pose(arm, 0.0002)
        site = sim.data.site(sim.prefixes[arm] + "gripperframe")
        return self._track(arm, site.xpos, True, rotation=site.xmat.reshape(3, 3))

    def _fixture(self):
        sim = self.sim
        target = sim.data.site("fixture_grasp").xpos.copy() + 0.021 * UP
        if fixture_held(sim):
            if self.fixture_target is None:
                self.fixture_target = sim.tcp(0).copy()
            return self._track(0, self.fixture_target, False, forward=-UP)
        self.fixture_target = None
        near = self._near(0, target, 0.013)
        descend = self.age > 36 or np.linalg.norm(sim.tcp(0)[:2] - target[:2]) < 0.028
        goal = target if near or descend else target + 0.045 * UP
        return self._track(0, goal, not near, forward=-UP)

    def _drawer(self):
        sim = self.sim
        handle = sim.data.site("drawer_handle_center").xpos.copy()
        if self.drawer_handle is None:
            self.drawer_handle = handle.copy()
        original = self.drawer_handle
        arm = 0
        if self.stage < 4:
            goals = [
                original + [-0.012, 0, 0.06],
                original + [0, 0, 0.007],
                original + [0, 0, 0.007],
                original + [-DRAWER_TRAVEL + 0.004, 0, 0.007],
            ]
            goal = goals[self.stage]
            opening = self.stage < 2
            if self.stage < 3 and (self._near(arm, goal) and self.age > 10 or self.age > 48):
                self._goto(self.stage + 1)
            elif self.stage == 3:
                if drawer_open(sim):
                    self._goto(4)
                elif self.age > 90:
                    self.retries += 1
                    self.drawer_handle = None
                    self._goto(0)
            return self.drawer_track(arm, goal, opening)
        if self.stage == 4:
            park = np.array([0.16, 0.22, 0.24])
            if self._near(arm, park, 0.028) or self.age > 45:
                self._goto(5)
            return self._track(arm, park, True)
        if self.stage < 11:
            arm = 1
            grip = sim.position("object").copy() + 0.022 * UP
            drop = sim.position("drawer").copy() + [-0.018, 0, ITEM_HALF[2] + 0.033]
            goals = [
                grip + 0.064 * UP,
                grip,
                grip,
                grip + 0.1 * UP,
                drop + 0.065 * UP,
                drop,
                drop + 0.085 * UP,
            ]
            goal = goals[self.stage - 5]
            if self.stage == 10 and item_stowed(sim):
                self._goto(11)
            elif self.stage < 10 and (self._near(arm, goal) and self.age >= 14 or self.age > 48):
                self._goto(self.stage + 1)
            elif self.stage == 10 and self.age > 65:
                self.retries += 1
                self._goto(5 if drawer_open(sim) else 0)
            return self._track(arm, goal, self.stage < 7 or self.stage >= 10, forward=-UP)
        if not item_stowed(sim):
            self.retries += 1
            self.drawer_handle = None
            self._goto(0)
            return self._track(0, handle - 0.05 * FORWARD + 0.05 * UP, True, forward=FORWARD)
        prepush = handle + [-0.04, 0, 0.006]
        if self.stage == 11 and (self._near(0, prepush) or self.age > 50):
            self._goto(12)
        goal = prepush if self.stage == 11 else original + [0.012, 0, 0.006]
        return self._track(0, goal, False, forward=FORWARD)

    def action(self):
        self.age += 1
        task = self.sim.cfg.task
        if task == "drawer_stow_close":
            return self._drawer()
        if task == "keyed_plug_insert":
            return self._plug(0)
        fixture = self._fixture()
        if self.fixture_target is None and self.stage == 0:
            return fixture
        return {**fixture, **self._plug(1)}
