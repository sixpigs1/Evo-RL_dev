# Precision and long-horizon MuJoCo tasks

These scenarios use the locked Menagerie robot models and the same follower joint targets, RGB streams and task text as the existing simulator. Geometry and contact parameters are simulation design values. A specific physical station can supply measured robot, camera, fixture, actuator and friction parameters.

| Task | Embodiment | Suggested horizon at 30 Hz | Nominal setup |
| --- | --- | ---: | --- |
| `keyed_plug_insert` | `bi_so101` | 600 steps | Fixed horizontal receptacle, passive plug cradle, two 5 mm square pins on 16 mm centres, asymmetric key and 20 mm pin reach |
| `drawer_stow_close` | `bi_piper` | 900 steps | 125 mm physical slide, cabinet, tray floor and four walls, handle and two coloured objects |
| `bimanual_stabilized_insert` | `bi_piper` | 900 steps | Same plug and mating cross-section, with a constrained lateral slide and yaw joint in the receptacle fixture |

## Contact and success

- The plug has separate collision geometry for its housing, shortened and narrowed leading tips, pin shafts and key. The receptacle contains three physical passages, a rear stop, and a lower shroud lip that supports the released housing. Each pin passage has 1.2 mm per-side clearance in nominal mode and 0.6 mm in precision mode. Insertion physics runs at a maximum step of 1 ms. Success checks axial depth, each pin's transverse alignment, key position and pose. The connector must remain seated and released for ten consecutive control frames.
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

## 4090 acceptance

MuJoCo 3.4.0, locked Menagerie assets, 30 Hz, nominal difficulty and seeds 0–9. Success counts are from the scripted reference controller.

| Task | Successful rollouts | Observed outcome |
| --- | ---: | --- |
| `drawer_stow_close` | 10/10 | Completed in 485 control steps per seed. |
| `keyed_plug_insert` | 1/10 | Seed 7 completed in 312 control steps. The other nine runs timed out. |
| `bimanual_stabilized_insert` | 0/10 | Reached the insertion stage in some runs. Loss of support and plug grip interrupted reference rollouts. |

The drawer meets the 8/10 reference-controller gate. The two connector tasks can be run for evaluation and teleoperation, and their scripted collection remains experimental. The nominal and precision fixed socket both kept an already seated, released connector in place under gravity. The three precision scenes ran and rendered in headless EGL. All 64 simulator and recorder regression tests passed on the 4090. Geometry, friction and camera alignment still need measurements at the intended physical station.

The 4090 worktree stores two decoded, success-labelled demonstrations and their physics traces under `outputs/mujoco_advanced_acceptance/`. The drawer demonstration used one 320×240 RGB stream. The keyed plug demonstration used three 320×240 streams (`left_wrist`, `right_wrist`, `right_top`). From trace frame 0, replay with the saved actuator controls reproduced every saved MuJoCo position exactly in the two demonstrations. Replay of the LeRobotDataset actions from that same initial state had maximum per-joint RMSE of 2.79×10⁻⁶ for the drawer and 3.20×10⁻⁶ for the plug, in each joint’s policy units.

To record a successful drawer demonstration with a success label and replay states:

```sh
.venv/bin/python -m lerobot.scripts.lerobot_record \
  --robot.type=mujoco --robot.embodiment=bi_piper --robot.task=drawer_stow_close \
  --robot.seed=0 --robot.episode_steps=900 --robot.scripted_expert=true \
  --robot.camera_views='{front: {width: 320, height: 240}}' \
  --robot.state_dir=outputs/mujoco_drawer/states \
  --dataset.repo_id=local/mujoco_drawer_advanced --dataset.root=outputs/mujoco_drawer/dataset \
  --dataset.single_task='Open the drawer, put the red block inside, and close the drawer.' \
  --dataset.num_episodes=1 --dataset.episode_time_s=32 --dataset.fps=30 --dataset.push_to_hub=false \
  --dataset.vcodec=h264 --enable_episode_outcome_labeling=true \
  --require_episode_success_label=true --display_data=false --play_sounds=false
```

## Wide initialization for the fixed socket

The fixed socket scene now defaults to a 60 mm sampling half-range on the
tabletop X and Y axes for both the connector and the receptacle. It samples
one of two mirrored work areas at reset. The connector cradle follows the
connector's sampled position. The sampler rejects fixtures that are too close
or outside the selected arm's inverse-kinematics range for the approach, grasp,
pre-insertion and insertion waypoints. Each episode uses the arm on the side
of the sampled station. Both arms use the same task success predicates.

Set `--robot.randomization=0` to reproduce the fixed default left-side layout.
Set an explicit value in metres to change the sampling half-range. The
workspace filter makes the accepted distribution nonuniform. The old fixed
socket benchmarks above used a smaller distribution and a different scene
version.

## Smoothed fixed-plug expert

The fixed-plug expert plans joint waypoints for approach, pinch, lift and
transport. Quintic interpolation starts and ends each segment with zero
command velocity and acceleration. The commanded speed is limited to about
1.76 rad/s and acceleration to 12 rad/s². During insertion, the controller
reads the plug pose again after each short segment. It corrects lateral and
vertical error before allowing additional depth. It previews two feedback
profiles on independent MuJoCo state and runs the chosen profile on the live
simulation through the standard joint-target interface. Previewing pauses the
expert's computation without advancing the live simulation.

4090 results at 30 Hz with the new ±60 mm reset distribution, 600 steps per
episode and seeds 150–249 reserved for evaluation:

| Difficulty | All | Left | Right |
| --- | ---: | ---: | ---: |
| Nominal | 76/100 | 38/46 | 38/54 |
| Precision | 62/100 | 29/46 | 33/54 |

For seeds 0–19, results were 17/20 nominal and 12/20 precision. Across the
200 held-out episodes, measured peak actuator command speed and acceleration
were 1.757 rad/s and 11.707 rad/s². These bounds apply to the commands. Physical
gripper acceleration can spike at contact or release. In the successful
sampled trajectories, the active TCP moved 0.54–0.76 mm in the first 0.1 s.
On the parallel validation run, median peak planner latency per episode was
0.29 s nominal and 0.63 s precision.

The evaluation uses the same scene, contact parameters, outcome checks and
seed distribution as the wide-initialization task. The 4090 worktree keeps
raw per-seed results under `outputs/mujoco_keyed_expert_validation/`.
