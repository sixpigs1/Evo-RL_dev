# MuJoCo backend validation on the RTX 4090

The validation uses Python 3.10, MuJoCo 3.4.0, EGL and the locked Menagerie assets. The branch starts from fork commit `b100924`.

## Simulator and datasets

- Four embodiments compiled and produced headless RGB. Automated coverage includes action ordering, partial dual-arm commands, clipped targets, snapshot replay, reset isolation, all 28 declared embodiment/task combinations, two spawned evaluation workers and the online action processor.
- Dual SO-101 produced eight episodes and 80 frames at 30 Hz with three 640 × 480 H.264 streams through `lerobot-record`. The recording was loaded by the value trainer. A separate two-episode state-only recording exercised trace export.
- A single dual SO-101 reach environment with one 320 × 240 camera completed 30 of 30 scripted episodes at 663 simulator control frames/s, with a maximum reset of 0.204 ms. This timing covers physics and RGB generation. Dataset encoding and VLA inference have separate costs.

## Scripted task sweep

Each cell reports successful episodes out of three generated with seeds 0–2 and a 300-step horizon. Every numeric cell also passed a reset-and-step check. Scripted completion requires additional work for contact tasks that timed out.

| Task | SO-101 | Dual SO-101 | PiPER | Dual PiPER |
| --- | ---: | ---: | ---: | ---: |
| Reach | 3/3 | 3/3 | 3/3 | 3/3 |
| Pick and place | 3/3 | 1/3 | 0/3 | 0/3 |
| Stack | 3/3 | 1/3 | 0/3 | 0/3 |
| Insert | 0/3 | 0/3 | 0/3 | 0/3 |
| Open drawer | 0/3 | 0/3 | 0/3 | 0/3 |
| Handover | — | 0/3 | — | 0/3 |
| Cooperative carry | — | 0/3 | — | 0/3 |
| Lift then place | 3/3 | 1/3 | 0/3 | 0/3 |

## Transfer and learning

- The supplied dual SO-101 real recording matches all 12 names and their order in both state and action metadata at 30 Hz. A replay of its first episode compared 1,026 action/next-state pairs. With the default Menagerie mapping, arm joint RMSE ranged from 0.92 to 4.65 in the recorded normalized units. Left and right gripper RMSE were 1.54 and 2.11. Camera reprojection and physical command latency need calibrated correspondences or timing logs.
- HIL-SERL sent 120 transitions from a MuJoCo actor to the learner, transferred policy parameters, wrote optimizer and replay-buffer state, and resumed optimization. A separate processor test injected a joint-leader takeover and verified named action ordering. Physical leader takeover requires a station session.
- PiStar06 completed a GPU optimizer step from the RGB simulation dataset and saved a checkpoint. The visual and text backbones were reconstructed from the supplied value checkpoint tensors using their archived architecture configurations.

See [setup and commands](mujoco_backend.md) for recording, evaluation, replay and the backend switch. The task sweep is reproducible with `examples/mujoco/benchmark.py --episodes 3`. The available real-data coverage is dual SO-101.
