"""Motion planning and closed-loop insertion for the dual SO-101 plug scene."""

import math
from copy import copy, deepcopy

import numpy as np

from .advanced_tasks import SEAT_X, plug_arm, plug_seated, socket_local

UP = np.array([0.0, 0.0, 1.0])
INSERTION_PROFILES = (
    # Pitch weight, jaw-axis weight, lateral tolerance and axial step in metres.
    (0.025, 0.035, 0.0018, 0.0016),
    (0.036, 0.030, 0.0020, 0.0016),
)


class KeyedPlugExpert:
    """Plan in joint space and observe the plug between smooth motion segments."""

    def __init__(self, sim):
        from scipy.optimize import least_squares

        self.sim = sim
        self.arm = plug_arm(sim)
        joints = np.asarray(sim.joints[self.arm * 6 : (self.arm + 1) * 6])
        self.addresses = sim.model.jnt_qposadr[joints]
        self.low, self.high = sim.model.jnt_range[joints[:-1]].T
        self.site = sim.model.site(sim.prefixes[self.arm] + "gripperframe").id
        self.scratch = sim.mj.MjData(sim.model)
        self.scratch.qpos[:] = sim.data.qpos
        self.command = sim.data.qpos[self.addresses].copy()
        self.solve = least_squares
        self.retries = 0
        self.trajectory = self._trajectory()

    def _inverse(self, position, rotation, initial, pitch_weight=0.016, jaw_weight=0.045):
        def residual(joints):
            self.scratch.qpos[self.addresses[:-1]] = joints
            self.sim.mj.mj_fwdPosition(self.sim.model, self.scratch)
            site = self.scratch.site(self.site)
            axes = site.xmat.reshape(3, 3)
            return np.concatenate(
                (
                    site.xpos - position,
                    pitch_weight * (axes[:, 0] - rotation[:, 0]),
                    jaw_weight * (axes[:, 2] - rotation[:, 2]),
                    0.0004 * (joints - initial),
                )
            )

        return self.solve(
            residual,
            np.clip(initial, self.low + 1e-6, self.high - 1e-6),
            bounds=(self.low, self.high),
            max_nfev=38,
            ftol=1e-6,
        ).x

    def _move(self, joints, gripper, frames):
        start = self.command.copy()
        target = np.r_[joints, gripper]
        delta = target - start
        distance = np.max(np.abs(delta))
        # Quintic time scaling gives zero velocity and acceleration at each junction.
        seconds = max(frames / 30, (32 / 30) * distance, math.sqrt(5.78 * distance / 12))
        steps = max(1, math.ceil(seconds * self.sim.cfg.fps - 1e-9))
        for i in range(1, steps + 1):
            t = i / steps
            self.command = start + delta * (t**3 * (10 + t * (-15 + 6 * t)))
            yield self.command

    def _dock(self, target, frames, pitch_weight=0.034, jaw_weight=0.044):
        sim = self.sim
        tool = sim.data.site(self.site)
        plug = sim.data.body("object")
        socket = sim.data.site("socket_mouth")
        plug_rotation = plug.xmat.reshape(3, 3)
        socket_rotation = socket.xmat.reshape(3, 3)
        transfer = socket_rotation @ plug_rotation.T
        position = socket.xpos + socket_rotation @ (np.asarray(target) - [SEAT_X, 0, 0])
        position += transfer @ (tool.xpos - plug.xpos)
        rotation = transfer @ tool.xmat.reshape(3, 3)
        joints = self._inverse(position, rotation, self.command[:-1], pitch_weight, jaw_weight)
        yield from self._move(joints, -0.13, frames)

    def _trajectory(self):
        sim = self.sim
        plug = sim.data.body("object")
        jaw_axis = plug.xmat.reshape(3, 3)[:, 1]
        rotation = np.column_stack((-UP, np.cross(jaw_axis, -UP), jaw_axis))
        position = plug.xpos + [-0.008, 0, 0.008] + 0.003 * jaw_axis
        pinch = self._inverse(position, rotation, np.array([0, -0.28, 0.4, 1.35, 0.6]))
        hover = self._inverse(position + 0.046 * UP, rotation, pinch)
        lift = self._inverse(position + 0.041 * UP, rotation, pinch)
        yield from self._move(hover, 0.62, 37)
        yield from self._move(pinch, 0.62, 27)
        yield from self._move(pinch, -0.13, 22)
        yield from self._move(lift, -0.13, 27)
        if not sim.grasping(self.arm):
            return

        for target, frames in (
            ((-0.065, 0, 0.043), 33),
            ((-0.045, 0, 0), 26),
            ((-0.025, 0, 0), 20),
            ((-0.022, 0, 0), 14),
            ((-0.020, 0, 0), 14),
        ):
            yield from self._dock(target, frames)
        profile = self._choose_insertion_profile()
        yield from self._insert(profile)

    def _insert(self, profile):
        pitch_weight, jaw_weight, lateral_tolerance, axial_step = profile
        sim = self.sim
        bias = np.zeros(2)
        depth = -0.020
        for _ in range(32):
            if plug_seated(sim):
                yield from self._move(self.command[:-1], 0.70, 24)
                return
            if not sim.grasping(self.arm):
                return
            seat = socket_local(sim, "plug_seat")
            bias = np.clip(bias - 0.65 * seat[1:], -0.012, 0.012)
            if np.linalg.norm(seat[1:]) < lateral_tolerance and seat[0] > depth - 0.003:
                depth = min(0.0003, depth + axial_step)
            yield from self._dock((depth, *bias), 9, pitch_weight, jaw_weight)

    def _choose_insertion_profile(self):
        # Preview contact on independent MuJoCo data. The live episode resumes with
        # the chosen feedback controller and advances through its normal action path.
        snapshot = self.sim.snapshot()
        preview = copy(self.sim)
        preview.data = preview.mj.MjData(preview.model)
        preview.rng = deepcopy(self.sim.rng)
        planner = copy(self)
        planner.sim = preview
        planner.scratch = preview.mj.MjData(preview.model)
        planner.scratch.qpos[:] = self.scratch.qpos
        best = INSERTION_PROFILES[0]
        best_error = math.inf
        for profile in INSERTION_PROFILES:
            preview.restore(snapshot)
            planner.command = self.command.copy()
            for command in planner._insert(profile):
                preview.step(planner._policy_action(command))
                if preview.done:
                    break
            # The task requires a quiet, released connector after the opening move.
            for _ in range(preview.task.hold_frames):
                if preview.done:
                    break
                preview.step(planner._policy_action(planner.command))
            if preview.success:
                return profile
            seat = socket_local(preview, "plug_seat")
            error = np.linalg.norm(seat) + (0.02 if not preview.grasping(self.arm) else 0)
            if error < best_error:
                best, best_error = profile, error
            self.retries += 1
        return best

    def action(self):
        return self._policy_action(next(self.trajectory, self.command))

    def _policy_action(self, command):
        start = self.arm * 6
        result = {}
        for name, value, mapping in zip(
            self.sim.names[start : start + 6], command, self.sim.mappings[start : start + 6], strict=True
        ):
            direction = 1 if mapping.simulation[0] < mapping.simulation[1] else -1
            result[name] = float(np.interp(value, mapping.simulation[::direction], mapping.policy[::direction]))
        return result
