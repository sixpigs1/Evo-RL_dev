"""Example of composing a task with a new scene object and an explicit success condition."""

from xml.etree import ElementTree as ET

import numpy as np

from lerobot.simulation.backend import MujocoSimulation
from lerobot.simulation.config import MujocoRobotConfig
from lerobot.simulation.expert import ScriptedExpert
from lerobot.simulation.tasks import TabletopTask, box


def build(world):
    box(world, "pedestal", ".20 0 .04", ".035 .035 .04", color=".2 .4 .7 1")
    ET.SubElement(world, "site", name="goal", pos=".20 0 .145", size=".01", rgba="0 1 0 .6")


def above_pedestal(sim):
    return np.linalg.norm(sim.tcp(0) - sim.data.site("goal").xpos) < 0.025


if __name__ == "__main__":
    task = TabletopTask("Move the gripper above the blue pedestal.", build, (above_pedestal,))
    sim = MujocoSimulation(MujocoRobotConfig(task="reach"), task=task)
    try:
        expert = ScriptedExpert(sim)
        while not sim.done:
            sim.step(expert.action())
        print({"success": sim.success, "steps": sim.steps})
    finally:
        sim.close()
