# MuJoCo backend

The MuJoCo backend uses the same joint-target and camera feature names as the Evo-RL robot drivers. Select one of `so101`, `bi_so101`, `piper`, or `bi_piper` with `embodiment`. Each robot uses the same simulation implementation in recording, evaluation, and joint-target HIL-SERL.

## Install and verify assets

```sh
pip install -e '.[mujoco]'
python -m lerobot.simulation.assets
python -m lerobot.simulation.assets --verify
export MUJOCO_GL=egl
```

Menagerie revision `eea4c7431b63da8278b0ff31638e0b0e941cec0e` is locked by per-file SHA-256. The manifest includes each model's license. For an offline host, copy the verified cache directory and point `LEROBOT_MENAGERIE_ROOT` or the robot's `assets` field to it. Startup checks the files against the manifest. For CI with no GL, leave `camera_views` empty and disable evaluation video output.

## Collect and train

```sh
lerobot-record \
  --robot.type=mujoco --robot.embodiment=so101 --robot.task=reach \
  --robot.scripted_expert=true --robot.state_dir=outputs/mujoco/states \
  --robot.camera_views='{front: {width: 320, height: 240}}' \
  --dataset.repo_id=local/mujoco_reach --dataset.root=outputs/mujoco/reach \
  --dataset.single_task='Move the gripper to the green target.' \
  --dataset.num_episodes=3 --dataset.fps=30 --dataset.push_to_hub=false \
  --display_data=false --play_sounds=false
```

Use the repository's `lerobot-dataset-report`, `lerobot-value-train`, `lerobot-value-infer` and `lerobot-train` commands with this dataset root and ID. Use `--policy.path` with the same robot, cameras and task text to collect the next round. With a compatible teleoperator, `lerobot-human-inloop-record` uses the regular keyboard takeover state machine. Scripted collection is identified as `scripted_expert`. For visual training, camera feature keys, dimensions and frame rate should match the real recording configuration.

Run `python -m lerobot.simulation.replay TRACE.json --frame 20 --image frame.png` to inspect a snapshot. Add `--dataset REPO_ID --dataset-root PATH --episode INDEX` to compare an existing recording's one-step-aligned joint tracking against a rollout from its initial state.

## Evaluate and switch backends

For visual simulation evaluation, use `lerobot-eval --env.type=mujoco --env.robot.embodiment=so101 --env.robot.task=reach --env.robot.camera_views='{front: {width: 320, height: 240}}' --policy.path=CHECKPOINT`. The Gym observation has `agent_pos` and optional `pixels`. Its action is an absolute joint-target vector in follower order. Async vector evaluation uses spawned workers.

Keep the checkpoint, task text, FPS, and feature keys fixed when switching to hardware. For recording, switch from `--robot.type=mujoco` to the matching follower type, then provide its port and hardware camera configuration. For example, use `--robot.type=so101_follower --robot.port=/dev/serial/by-id/DEVICE --robot.cameras='{front: {type: opencv, index_or_path: 0, width: 320, height: 240, fps: 30}}'` with the same dataset task and checkpoint. Use `robot.joint_mappings` to map real endpoint values onto calibrated simulator endpoints and provide the measured `camera_views`, `base_positions`, `base_yaws` and `calibration_version`. Menagerie defaults are approximate. The PiPER model controls one finger through an equality constraint, with total opening represented in millimetres at the policy interface. PiPER-X requires a separate asset and calibration.

HIL-SERL accepts `env.name: joint_targets` with either the MuJoCo robot or a real follower. Configure the SAC action feature to have one element per joint target and use the same state and camera features as the selected follower. Configure the policy's action scaling for the joint units or absolute outputs. For real success rewards use the existing reward classifier or teleoperator events. An event-capable joint teleoperator can supply takeover events, or a local keyboard can toggle takeover on a joint leader. Preserve the joint-target configuration when moving the checkpoint between backends.

## Task coverage

| Task | SO-101 | Dual SO-101 | PiPER | Dual PiPER |
| --- | --- | --- | --- | --- |
| Reach, pick and place, stack, insert, open drawer, lift then place | Declared | Declared | Declared | Declared |
| Handover and cooperative carry | — | Declared | — | Declared |

The table describes scene and task availability. Use `examples/mujoco/benchmark.py` to measure scripted completion rates with fixed seeds, and `examples/mujoco/custom_task.py` for a custom scene and success predicate.
