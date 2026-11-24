"""Reusable tabletop task definitions for MuJoCo."""
from dataclasses import dataclass
from typing import Callable
from xml.etree import ElementTree as ET

import numpy as np


def box(parent, name, position, size, *, free=False, color="0.7 0.25 0.15 1"):
    body = ET.SubElement(parent, "body", name=name, pos=position)
    if free:
        ET.SubElement(body, "freejoint", name=f"{name}_free")
    ET.SubElement(body, "geom", name=name, type="box", size=size, rgba=color, friction="1.2 .01 .001")
    return body


def _reach(world):
    ET.SubElement(world, "site", name="goal", pos=".19 0 .14", size=".012", rgba="0 1 0 .5")


def _pick(world):
    box(world, "object", ".16 -.05 .018", ".018 .018 .018", free=True)
    box(world, "target", ".22 .07 .004", ".035 .035 .004", color=".2 .65 .3 1")


def _stack(world):
    _pick(world)
    box(world, "bottom", ".22 .07 .027", ".019 .019 .019", free=True)


def _insert(world):
    box(world, "object", ".16 -.05 .028", ".009 .009 .027", free=True)
    socket = ET.SubElement(world, "body", name="target", pos=".22 .07 .014")
    for i, (pos, size) in enumerate([
        (".018 0 0", ".005 .023 .014"), ("-.018 0 0", ".005 .023 .014"),
        ("0 .018 0", ".013 .005 .014"), ("0 -.018 0", ".013 .005 .014"),
    ]):
        ET.SubElement(socket, "geom", name=f"socket_{i}", type="box", pos=pos, size=size, rgba=".2 .65 .3 1")


def _drawer(world):
    box(world, "cabinet", ".27 0 .075", ".045 .055 .009")
    drawer = box(world, "drawer", ".26 0 .045", ".04 .048 .016")
    ET.SubElement(drawer, "joint", name="drawer_slide", type="slide", axis="-1 0 0",
                  range="0 .11", damping="1")
    ET.SubElement(drawer, "geom", name="drawer_handle", type="capsule", size=".008",
                  fromto="-.054 -.025 0 -.054 .025 0")


def _handover(world):
    box(world, "object", ".19 .115 .018", ".017 .017 .018", free=True)


def _carry(world):
    box(world, "object", ".2 0 .015", ".019 .09 .015", free=True)
    box(world, "target", ".32 0 .004", ".035 .1 .004", color=".2 .65 .3 1")


def _reach_done(sim):
    return np.linalg.norm(sim.tcp(0) - sim.data.site("goal").xpos) < .025


def _placed(sim):
    obj, target = sim.position("object"), sim.position("target")
    return np.linalg.norm(obj[:2] - target[:2]) < .026 and .016 < obj[2] < .036 and sim.settled("object")


def _stacked(sim):
    offset = sim.position("object") - sim.position("bottom")
    return np.linalg.norm(offset[:2]) < .018 and .03 < offset[2] < .047 and sim.settled("object")


def _inserted(sim):
    offset = sim.position("object") - sim.position("target")
    upright = abs(sim.data.body("object").xmat[8]) > .95
    return np.linalg.norm(offset[:2]) < .004 and 0 < offset[2] < .02 and upright and sim.settled("object")


def _opened(sim):
    return sim.data.joint("drawer_slide").qpos[0] > .075


def _handover_left(sim):
    return sim.position("object")[2] > .05 and sim.grasping(0)


def _handover_right(sim):
    return sim.position("object")[2] > .05 and sim.grasping(1) and not sim.grasping(0)


def _lifted_together(sim):
    return sim.position("object")[2] > .06 and sim.grasping(0) and sim.grasping(1)


@dataclass(frozen=True)
class TabletopTask:
    instruction: str
    build: Callable
    stages: tuple[Callable, ...]
    arms: int = 1


TASKS = {
    "reach": TabletopTask("Move the gripper to the green target.", _reach, (_reach_done,)),
    "pick_place": TabletopTask("Place the red block on the green target.", _pick, (_placed,)),
    "stack": TabletopTask("Stack the red block on the other block.", _stack, (_stacked,)),
    "insert": TabletopTask("Insert the peg into the socket.", _insert, (_inserted,)),
    "open_drawer": TabletopTask("Pull the drawer open.", _drawer, (_opened,)),
    "handover": TabletopTask("Pass the block from the left arm to the right arm.",
                             _handover, (_handover_left, _handover_right), 2),
    "cooperative_carry": TabletopTask("Carry the long block together and place it on the target.",
                                     _carry, (_lifted_together, _placed), 2),
    "lift_then_place": TabletopTask("Lift the block and place it on the green target.",
                                    _pick, (_handover_left, _placed)),
}
