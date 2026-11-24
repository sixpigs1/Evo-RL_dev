# Precision and long-horizon MuJoCo tasks

These scenarios use the locked Menagerie robot models and the same follower joint targets, RGB streams and task text as the existing simulator. Geometry and contact parameters are simulation design values. A specific physical station can supply measured robot, camera, fixture, actuator and friction parameters.

| Task | Embodiment | Suggested horizon at 30 Hz | Nominal setup |
| --- | --- | ---: | --- |
| `keyed_plug_insert` | `bi_so101` | 600 steps | Fixed horizontal receptacle, passive plug cradle, two 5 mm square pins on 16 mm centres, asymmetric key and 20 mm pin reach |
| `drawer_stow_close` | `bi_piper` | 900 steps | 125 mm physical slide, cabinet, tray floor and four walls, handle and two coloured objects |
| `bimanual_stabilized_insert` | `bi_piper` | 900 steps | Same plug and mating cross-section, with a constrained lateral slide and yaw joint in the receptacle fixture |

## Contact and success

- The plug has separate collision geometry for its housing, shortened and narrowed leading tips, pin shafts and key. The receptacle contains three physical passages and a rear stop. Each pin passage has 1.2 mm per-side clearance in nominal mode and 0.6 mm in precision mode. Insertion physics runs at a maximum step of 1 ms. Success checks axial depth, each pin's transverse alignment, key position and pose. The connector must remain seated and released for ten consecutive control frames.
- The drawer opens at least 100 mm, then the red object's full geometry must enter below the rim and be released. Closing requires the slide within 5 mm of its stop, a settled target still inside, and the blue distractor on the tabletop. The final condition must persist for ten frames. If the target leaves during closing, the task returns to an earlier stage. Precision mode increases slide damping.
- The mobile fixture requires both left jaw bodies to contact the handle. The fixture must be within 12 mm and 8° of its nominal position and below velocity limits. Losing support after the first stage resets progress and the final hold counter. The right arm must release the seated plug while the left arm keeps supporting the fixture.
- Video presets can use existing robot-specific profiles. Camera locations, robot base placement and joint mappings are configurable. The task presets include small seeded pickup and receptacle tolerances. These parameters are starting points for physical fixture calibration.

Each task uses normal MuJoCo actuator commands throughout a rollout. Reference controllers use simulator state for Cartesian targeting, contact checking and retries. A policy receives the existing RGB, joint state and task text. Gymnasium info and state traces record the current stage, time in each stage and stage restarts. The benchmark also reports reference controller retries.

## Run on the 4090

```sh
export PYTHONPATH=src
export MUJOCO_GL=egl
.venv/bin/python examples/mujoco/benchmark.py --robot bi_so101 --task keyed_plug_insert --episodes 10 --steps 600
.venv/bin/python examples/mujoco/benchmark.py --robot bi_piper --task drawer_stow_close --episodes 10 --steps 900
.venv/bin/python examples/mujoco/benchmark.py --robot bi_piper --task bimanual_stabilized_insert --episodes 10 --steps 900
```

Add `--difficulty precision` to run the tighter variant, or `--rgb` for RGB rendering. For recording, choose the corresponding `--robot.embodiment`, `--robot.task`, `--robot.episode_steps` and language instruction. Use `--robot.scripted_expert=true` for reference rollouts and `--robot.state_dir` for snapshots. Evaluation uses the same fields under `--env.robot`. See [backend commands](mujoco_backend.md).
