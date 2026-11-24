"""Replay saved simulation states or compare joint tracking against a LeRobot episode."""

import argparse
import json
from pathlib import Path

import numpy as np

from .backend import MujocoSimulation
from .config import JointMapping, MujocoRobotConfig, SimCamera


def config_from_metadata(data):
    return MujocoRobotConfig(
        **{
            **data,
            "camera_views": {key: SimCamera(**value) for key, value in data["camera_views"].items()},
            "joint_mappings": {key: JointMapping(**value) for key, value in data["joint_mappings"].items()},
        }
    )


def compare_dataset(sim, dataset, episode):
    action_names = tuple(dataset.features["action"]["names"])
    state_names = tuple(dataset.features["observation.state"]["names"])
    if action_names != sim.names or state_names != sim.names or dataset.fps != sim.cfg.fps:
        raise ValueError("Dataset joint order or control rate differs from the simulation configuration")
    table = dataset.hf_dataset.with_format(None)
    frames = table.filter(lambda row: row["episode_index"] == episode)
    if len(frames) < 2:
        raise ValueError("The selected episode needs at least two frames")
    sim.reset()
    initial = np.array(frames[0]["observation.state"])
    for joint, mapping, value in zip(sim.joints, sim.mappings, initial, strict=True):
        sim.data.qpos[sim.model.jnt_qposadr[joint]] = np.interp(value, mapping.policy, mapping.simulation)
    if sim.robot == "piper":
        for prefix in sim.prefixes:
            sim.data.joint(prefix + "joint8").qpos[:] = -sim.data.joint(prefix + "joint7").qpos
    sim.mj.mj_forward(sim.model, sim.data)
    errors = []
    for action, reference in zip(frames["action"][:-1], frames["observation.state"][1:], strict=True):
        if sim.done:
            break
        sim.step(dict(zip(sim.names, action, strict=True)))
        errors.append(np.array(list(sim.read_joints().values())) - reference)
    if not errors:
        raise ValueError("No valid replay transitions")
    return {
        "dataset": dataset.repo_id,
        "episode": episode,
        "compared_transitions": len(errors),
        "reference_transitions": len(frames) - 1,
        "rmse_in_policy_units": dict(
            zip(sim.names, np.sqrt(np.mean(np.square(errors), axis=0)), strict=True)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()
    trace = json.loads(args.trace.read_text())
    state = trace[0] if args.frame == 0 else trace[args.frame]["state"]
    sim = MujocoSimulation(config_from_metadata(state["config"]))
    try:
        sim.restore(state)
        if args.image:
            from PIL import Image

            Image.fromarray(sim.image()).save(args.image)
        if args.dataset:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            dataset = LeRobotDataset(args.dataset, root=args.dataset_root)
            print(json.dumps(compare_dataset(sim, dataset, args.episode), indent=2))
        else:
            print(json.dumps({"step": sim.steps, "phase": sim.phase, "joints": sim.read_joints()}, indent=2))
    finally:
        sim.close()


if __name__ == "__main__":
    main()
