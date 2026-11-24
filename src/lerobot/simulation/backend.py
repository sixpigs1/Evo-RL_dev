"""Shared MuJoCo physics, control, camera and episode state."""

import json
import math
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from lerobot.utils.piper_sdk import PIPER_ACTION_KEYS

from .assets import MODEL_FILES, REVISION, asset_root, verify_assets
from .config import JointMapping, MujocoRobotConfig
from .tasks import TASKS, TabletopTask

SO_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def control_names(cfg):
    joints = [f"{name}.pos" for name in SO_JOINTS] if cfg.embodiment.endswith("so101") else PIPER_ACTION_KEYS
    prefixes = ("left_", "right_") if cfg.embodiment.startswith("bi_") else ("",)
    return tuple(prefix + joint for prefix in prefixes for joint in joints)


@lru_cache(maxsize=8)
def _verified_model(root: Path, robot: str) -> Path:
    verify_assets(root, robot)
    return root / MODEL_FILES[robot]


class MujocoSimulation:
    def __init__(self, cfg: MujocoRobotConfig, task: TabletopTask | None = None):
        import mujoco

        self.mj = mujoco
        self.cfg = cfg
        if task is None and cfg.task not in TASKS:
            raise ValueError(f"Unknown MuJoCo task {cfg.task!r}")
        self.task = task or TASKS[cfg.task]
        if not self.task.supports(cfg.embodiment):
            raise ValueError(f"{cfg.task!r} does not support {cfg.embodiment!r}")
        self.substeps = 8
        if self.task.max_physics_dt:
            self.substeps = max(self.substeps, math.ceil(1 / (cfg.fps * self.task.max_physics_dt)))
        self.robot = cfg.embodiment.removeprefix("bi_")
        self.prefixes = ("left_", "right_") if cfg.embodiment.startswith("bi_") else ("",)
        self.names = control_names(cfg)
        self.rng = np.random.default_rng(cfg.seed)
        self.renderers = {}
        self.model = self._make_model()
        self.data = mujoco.MjData(self.model)
        self.object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.actuators, self.joints, self.mappings = self._control_layout()
        self.reset()

    def _make_model(self):
        cfg = self.cfg
        root = ET.Element("mujoco", model=cfg.task)
        ET.SubElement(
            root,
            "option",
            timestep=str(1 / (cfg.fps * self.substeps)),
            integrator="implicitfast",
            cone="elliptic",
        )
        visual = ET.SubElement(root, "visual")
        ET.SubElement(
            visual,
            "global",
            offwidth=str(max([640, *(c.width for c in cfg.camera_views.values())])),
            offheight=str(max([480, *(c.height for c in cfg.camera_views.values())])),
        )
        world = ET.SubElement(root, "worldbody")
        ET.SubElement(world, "light", pos=".25 -.2 1", dir="0 0 -1", directional="true")
        ET.SubElement(
            world,
            "geom",
            name="table",
            type="box",
            pos=".16 0 -.025",
            size=".65 .5 .025",
            rgba=".48 .49 .52 1",
        )
        ET.SubElement(world, "camera", name="overview", pos=".8 -.78 .65", xyaxes=".8 .6 0 -.25 .33 .91")
        if self.task.stage_names:
            self.task.build(world, cfg.task_difficulty)
        else:
            self.task.build(world)
        spec = self.mj.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
        path = _verified_model((cfg.assets or asset_root()).resolve(), self.robot)
        positions = cfg.base_positions or (
            [(0, 0.19, 0), (0, -0.19, 0)] if len(self.prefixes) == 2 else [(0, 0, 0)]
        )
        yaws = cfg.base_yaws or ([-0.45, 0.45] if len(self.prefixes) == 2 else [0])
        if len(positions) != len(self.prefixes) or len(yaws) != len(self.prefixes):
            raise ValueError("Provide one base position and yaw per arm")
        spec.copy_during_attach = True
        for prefix, position, yaw in zip(self.prefixes, positions, yaws, strict=True):
            arm = self.mj.MjSpec.from_file(str(path))
            if self.robot == "piper":
                arm.body("link6").add_site(name="gripperframe", pos=[0, 0, 0.09], size=[0.004, 0, 0])
            frame = spec.worldbody.add_frame(pos=position, quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
            spec.attach(arm, prefix=prefix, frame=frame)
        for name, camera in cfg.camera_views.items():
            if camera.native is not None:
                continue
            body = spec.worldbody if camera.mount == "world" else spec.body(camera.mount)
            if body is None:
                raise ValueError(f"Camera {name!r} has unknown mount {camera.mount!r}")
            body.add_camera(
                name=f"observation_{name}", pos=camera.position, xyaxes=camera.xyaxes, fovy=camera.fovy
            )
        model = spec.compile()
        for camera in cfg.camera_views.values():
            if camera.native is not None:
                model.camera(camera.native)
        return model

    def _control_layout(self):
        actuator_names = (
            SO_JOINTS if self.robot == "so101" else (*[f"joint{i}" for i in range(1, 7)], "gripper")
        )
        ids = [self.model.actuator(prefix + name).id for prefix in self.prefixes for name in actuator_names]
        joints = [int(self.model.actuator_trnid[i, 0]) for i in ids]
        mappings = []
        unknown = self.cfg.joint_mappings.keys() - set(self.names)
        if unknown:
            raise ValueError(f"Unknown joint mappings: {sorted(unknown)}")
        for key, actuator in zip(self.names, ids, strict=True):
            lo, hi = self.model.actuator_ctrlrange[actuator]
            if key.endswith("gripper.pos"):
                policy = (0, 100) if self.robot == "so101" else (0, 2000 * hi)
            elif self.robot == "so101" and not self.cfg.so101_use_degrees:
                policy = (-100, 100)
            else:
                policy = tuple(np.rad2deg([lo, hi]))
            mapping = self.cfg.joint_mappings.get(key, JointMapping(policy, (lo, hi)))
            p, q = np.asarray(mapping.policy), np.asarray(mapping.simulation)
            if (
                not np.isfinite([p, q]).all()
                or p[1] <= p[0]
                or q[1] == q[0]
                or min(q) < lo - 1e-8
                or max(q) > hi + 1e-8
            ):
                raise ValueError(f"Invalid policy/simulation endpoints for {key}: {mapping}")
            mappings.append(mapping)
        return ids, joints, mappings

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.mj.mj_resetData(self.model, self.data)
        self.phase = self.steps = 0
        self.success = False
        self.failure_reason = ""
        self.stage_frames = [0] * len(self.task.stages)
        self.stage_restarts = 0
        self.stage_hold = 0
        for prefix in self.prefixes:
            if self.robot == "piper":
                self.data.joint(prefix + "joint2").qpos[0] = 1.57
                self.data.joint(prefix + "joint3").qpos[0] = -1.3485
        if self.robot == "piper" and self.task.stage_names:
            for arm, prefix in enumerate(self.prefixes):
                self.data.joint(prefix + "joint1").qpos[0] = 0.65 if arm == 0 else -0.65
        if self.robot == "so101":
            for prefix in self.prefixes:
                self.data.joint(prefix + "gripper").qpos[0] = 0.65
        if self.task.reset is not None:
            self.task.reset(self)
        else:
            self._randomize_free_joints()
        self.data.ctrl[self.actuators] = self.data.qpos[self.model.jnt_qposadr[self.joints]]
        self.mj.mj_forward(self.model, self.data)

    def _randomize_free_joints(self):
        shift = min(self.cfg.randomization, 0.005) if self.task.stage_names else self.cfg.randomization
        for index in range(self.model.njnt):
            if self.model.jnt_type[index] == self.mj.mjtJoint.mjJNT_FREE:
                address = self.model.jnt_qposadr[index]
                self.data.qpos[address : address + 2] += self.rng.uniform(-shift, shift, 2)

    def _to_policy(self, values):
        result = {}
        for key, value, mapping in zip(self.names, values, self.mappings, strict=True):
            direction = 1 if mapping.simulation[0] < mapping.simulation[1] else -1
            result[key] = float(
                np.interp(value, mapping.simulation[::direction], mapping.policy[::direction])
            )
        return result

    def read_joints(self):
        return self._to_policy(self.data.qpos[self.model.jnt_qposadr[self.joints]])

    def step(self, action, *, evaluate_task=True):
        if evaluate_task and self.done:
            raise RuntimeError("Reset the simulation before starting another episode")
        if not action or action.keys() - set(self.names):
            raise ValueError("Action must contain known joint position keys")
        targets = []
        for key, actuator, mapping in zip(self.names, self.actuators, self.mappings, strict=True):
            if key not in action:
                continue
            value = float(action[key])
            if not np.isfinite(value):
                raise ValueError(f"Nonfinite action for {key}")
            targets.append((actuator, np.interp(value, mapping.policy, mapping.simulation)))
        for actuator, value in targets:
            self.data.ctrl[actuator] = value
        self.mj.mj_step(self.model, self.data, nstep=self.substeps)
        self.mj.mj_forward(self.model, self.data)
        self.steps += 1
        if evaluate_task:
            self.advance_task()
            if (
                not self.success
                and self.object_body_id >= 0
                and self.data.xpos[self.object_body_id, 2] < -0.04
            ):
                self.failure_reason = "object_dropped"
            elif not self.success and self.steps >= self.cfg.episode_steps:
                self.failure_reason = "timeout"
        if not np.isfinite(self.data.qpos).all():
            raise RuntimeError("MuJoCo produced nonfinite joint state")
        return self._to_policy(self.data.ctrl[self.actuators])

    def advance_task(self):
        if self.task.recover is not None:
            phase = self.task.recover(self)
            if phase != self.phase:
                self.phase = phase
                self.stage_restarts += 1
                self.stage_hold = 0
        self.stage_frames[self.phase] += 1
        self.stage_hold = self.stage_hold + 1 if self.task.stages[self.phase](self) else 0
        required = self.task.hold_frames if self.phase == len(self.task.stages) - 1 else 1
        if self.stage_hold >= required:
            self.phase += 1
            self.stage_hold = 0
            self.success = self.phase == len(self.task.stages)

    def task_progress(self):
        if not self.task.stage_names:
            return {}
        idx = min(self.phase, len(self.task.stage_names) - 1)
        return {
            "task_stage": "complete" if self.success else self.task.stage_names[idx],
            "stage_frames": np.asarray(self.stage_frames, dtype=np.int32),
            "stage_restarts": self.stage_restarts,
        }

    @property
    def done(self):
        return self.success or bool(self.failure_reason) or self.steps >= self.cfg.episode_steps

    def position(self, name):
        return self.data.body(name).xpos

    def tcp(self, arm):
        return self.data.site(self.prefixes[arm] + "gripperframe").xpos

    def settled(self, obj):
        return np.linalg.norm(self.data.joint(f"{obj}_free").qvel) < 0.18

    def _finger_contacts(self, arm, target, target_geom=None):
        prefix = self.prefixes[arm]
        fingers = ("gripper", "moving_jaw_so101_v1") if self.robot == "so101" else ("link7", "link8")
        bodies = {self.model.body(prefix + finger).id for finger in fingers}
        target_id = self.model.body(target).id
        geom_id = self.model.geom(target_geom).id if target_geom else None
        contacts = set()
        for contact in self.data.contact:
            if contact.dist > 0:
                continue
            one, two = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if one == target_id and (geom_id is None or contact.geom1 == geom_id):
                contacts.add(two)
            if two == target_id and (geom_id is None or contact.geom2 == geom_id):
                contacts.add(one)
        return bodies, contacts

    def grasping(self, arm, target="object", target_geom=None):
        bodies, contacts = self._finger_contacts(arm, target, target_geom)
        return bodies <= contacts

    def gripper_touches(self, arm, target):
        bodies, contacts = self._finger_contacts(arm, target)
        return bool(bodies & contacts)

    def image(self, camera="overview", width=320, height=240):
        size = (width, height)
        if size not in self.renderers:
            self.renderers[size] = self.mj.Renderer(self.model, height=height, width=width)
        renderer = self.renderers[size]
        renderer.update_scene(self.data, camera=camera)
        return renderer.render()

    def observation(self):
        return {
            **self.read_joints(),
            **{
                name: self.image(camera.native or f"observation_{name}", camera.width, camera.height)
                for name, camera in self.cfg.camera_views.items()
            },
        }

    def scene_signature(self):
        config = asdict(self.cfg)
        for key in ("assets", "state_dir", "scripted_expert", "id", "calibration_dir", "seed"):
            config.pop(key)
        return json.loads(json.dumps(config))

    def snapshot(self):
        state = np.empty(self.mj.mj_stateSize(self.model, self.state_spec))
        self.mj.mj_getState(self.model, self.data, state, self.state_spec)
        return {
            "integration": state.tolist(),
            "rng": self.rng.bit_generator.state,
            "phase": self.phase,
            "stage_frames": self.stage_frames.copy(),
            "stage_restarts": self.stage_restarts,
            "stage_hold": self.stage_hold,
            "steps": self.steps,
            "success": self.success,
            "failure_reason": self.failure_reason,
            "mujoco": self.mj.__version__,
            "menagerie": REVISION,
            "scene_version": f"tabletop-v{self.task.scene_version}",
            "config": self.scene_signature(),
        }

    def restore(self, snapshot):
        state = np.asarray(snapshot["integration"], dtype=np.float64)
        if snapshot["menagerie"] != REVISION or snapshot["mujoco"] != self.mj.__version__:
            raise ValueError("Snapshot simulator or asset version differs")
        if snapshot.get("scene_version", "tabletop-v1") != f"tabletop-v{self.task.scene_version}":
            raise ValueError("Snapshot scene version differs")
        config = {"task_difficulty": "nominal", **snapshot["config"]}
        if config != self.scene_signature():
            raise ValueError("Snapshot scene or calibration configuration differs")
        self.mj.mj_setState(self.model, self.data, state, self.state_spec)
        self.rng.bit_generator.state = snapshot["rng"]
        self.phase = snapshot["phase"]
        self.stage_frames = list(snapshot.get("stage_frames", [0] * len(self.task.stages)))
        self.stage_restarts = snapshot.get("stage_restarts", 0)
        self.stage_hold = snapshot.get("stage_hold", 0)
        self.steps = snapshot["steps"]
        self.success = snapshot["success"]
        self.failure_reason = snapshot.get("failure_reason", "")
        self.mj.mj_forward(self.model, self.data)

    def close(self):
        for renderer in self.renderers.values():
            renderer.close()
        self.renderers.clear()
