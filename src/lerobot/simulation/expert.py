"""Scripted joint-target collection using simulator-only Cartesian goals."""

import numpy as np

from .advanced_expert import AdvancedExpert
from .keyed_expert import KeyedPlugExpert


class ScriptedExpert:
    def __init__(self, sim):
        self.sim = sim
        self.stage = 0
        self.stage_steps = 0
        self.drawer_plan = None
        if sim.cfg.task == "keyed_plug_insert":
            self.advanced = KeyedPlugExpert(sim)
        else:
            self.advanced = AdvancedExpert(sim) if sim.task.stage_names else None

    def _move(self, arm, goal, opening):
        sim = self.sim
        prefix = sim.prefixes[arm]
        width = len(sim.names) // len(sim.prefixes)
        start = arm * width
        stop = start + width
        joint_ids = sim.joints[start : stop - 1]
        positions = sim.model.jnt_qposadr[joint_ids]
        dofs = sim.model.jnt_dofadr[joint_ids]
        jacobian = np.zeros((3, sim.model.nv))
        angular = np.zeros_like(jacobian)
        site = sim.model.site(prefix + "gripperframe").id
        sim.mj.mj_jacSite(sim.model, sim.data, jacobian, angular, site)
        jacobian = jacobian[:, dofs]
        error = goal - sim.tcp(arm)
        if sim.cfg.task not in {"reach", "handover", "cooperative_carry"}:
            rotation = sim.data.site_xmat[site].reshape(3, 3)
            axis = rotation[:, 0 if sim.robot == "so101" else 2]
            direction = (
                np.array([1, 0, 0])
                if sim.cfg.task == "open_drawer"
                else np.array([0, 0, -1])
            )
            derivative = np.cross(angular[:, dofs].T, axis).T
            jacobian = np.vstack([jacobian, 0.09 * derivative])
            error = np.concatenate([error, 0.09 * (direction - axis)])
        delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + np.eye(len(error)) * 0.0025, error)
        q = sim.data.qpos[positions] + np.clip(delta, -0.12, 0.12)
        result = {}
        for name, value, mapping in zip(
            sim.names[start : stop - 1], q, sim.mappings[start : stop - 1], strict=True
        ):
            direction = 1 if mapping.simulation[0] < mapping.simulation[1] else -1
            result[name] = np.interp(value, mapping.simulation[::direction], mapping.policy[::direction])
        grip = sim.mappings[stop - 1]
        result[sim.names[stop - 1]] = grip.policy[int(opening)]
        return result

    def action(self):
        if self.advanced is not None:
            return self.advanced.action()
        sim = self.sim
        task = sim.cfg.task
        if task == "reach":
            return self._move(0, sim.data.site("goal").xpos, True)
        if task == "open_drawer":
            if self.drawer_plan is None:
                origin = sim.position("drawer").copy()
                grip = origin + [-0.078, 0, 0.007]
                self.drawer_plan = [
                    (origin + [-0.09, 0, 0.06], True),
                    (grip, True),
                    (grip, False),
                    (origin + [-0.17, 0, 0.007], False),
                ]
            return self._run(self.drawer_plan)
        obj = sim.position("object").copy()
        if task == "handover":
            transfer = np.array([0.19, 0, 0.13])
            if sim.phase == 0:
                return self._run(
                    [(obj + [0, 0, 0.06], True), (obj, True), (obj, False), (obj + [0, 0, 0.07], False)]
                )
            right = self._move(1, obj + [0, -0.012, 0], np.linalg.norm(sim.tcp(1) - obj) > 0.025)
            return {**self._move(0, transfer, sim.grasping(1)), **right}
        target = sim.position("bottom" if task == "stack" else "target").copy()
        goal_height = 0.037 if task == "stack" else 0.013 if task == "insert" else 0.022
        target[2] += goal_height
        if task == "cooperative_carry":
            actions = {}
            for arm, offset in enumerate((0.063, -0.063)):
                handle = obj + [0, offset, 0]
                goal = target + [0, offset, 0.10 if not sim.phase else 0]
                ready = all(
                    np.linalg.norm(sim.tcp(i) - (obj + [0, o, 0])) < 0.025
                    for i, o in enumerate((0.063, -0.063))
                )
                grasped = sim.grasping(0) and sim.grasping(1)
                actions.update(
                    self._move(
                        arm,
                        goal if grasped else handle,
                        not ready and not grasped or sim.phase > 0 and np.linalg.norm(obj - target) < 0.03,
                    )
                )
            return actions
        plan = [
            (obj + [0, 0, 0.065], True),
            (obj, True),
            (obj, False),
            (obj + [0, 0, 0.1], False),
            (target + [0, 0, 0.10], False),
            (target, False),
            (target, True),
        ]
        return self._run(plan)

    def _run(self, plan):
        goal, opening = plan[self.stage]
        self.stage_steps += 1
        reached = np.linalg.norm(self.sim.tcp(0) - goal) < 0.026
        if self.stage < len(plan) - 1 and (reached and self.stage_steps >= 12 or self.stage_steps >= 55):
            self.stage += 1
            self.stage_steps = 0
        return self._move(0, goal, opening)
