"""Measure a seeded task sweep at simulator control frequency."""

import argparse
import json
import time
from collections import Counter

from lerobot.simulation.backend import MujocoSimulation
from lerobot.simulation.config import MujocoRobotConfig, SimCamera
from lerobot.simulation.expert import ScriptedExpert
from lerobot.simulation.tasks import TASKS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument("--robot", choices=("so101", "bi_so101", "piper", "bi_piper"))
    parser.add_argument("--task", choices=sorted(TASKS))
    args = parser.parse_args()
    for robot in (args.robot,) if args.robot else ("so101", "bi_so101", "piper", "bi_piper"):
        for name, task in TASKS.items():
            if args.task and args.task != name or task.arms == 2 and not robot.startswith("bi_"):
                continue
            config = MujocoRobotConfig(embodiment=robot, task=name, episode_steps=args.steps)
            if args.rgb:
                config.camera_views["front"] = SimCamera()
            sim = MujocoSimulation(config)
            try:
                successes = 0
                failures = Counter()
                frames = 0
                resets = []
                start = time.perf_counter()
                for seed in range(args.episodes):
                    reset_start = time.perf_counter()
                    sim.reset(seed)
                    resets.append(time.perf_counter() - reset_start)
                    expert = ScriptedExpert(sim)
                    while not sim.done:
                        sim.step(expert.action())
                        if args.rgb:
                            sim.observation()
                    frames += sim.steps
                    successes += sim.success
                    if sim.failure_reason:
                        failures[sim.failure_reason] += 1
                print(
                    json.dumps(
                        {
                            "robot": robot,
                            "task": name,
                            "episodes": args.episodes,
                            "successes": successes,
                            "failures": dict(failures),
                            "control_frames_per_s": frames / (time.perf_counter() - start),
                            "max_reset_ms": max(resets) * 1000,
                        }
                    ),
                    flush=True,
                )
            finally:
                sim.close()


if __name__ == "__main__":
    main()
