# MuJoCo expert verification on the RTX 4090

Tested commit `2d36f66` with MuJoCo 3.4.0, the pinned Menagerie assets, 30 Hz and seeds 0–9. The sweep ran 620 episodes across nominal, extended-horizon and precision configurations. Each episode passed initial and terminal snapshot restoration. Each declared success still satisfied the task's final predicate after restoration. The simulator and recorder regression suite passed 64 tests.

## Nominal completion

Cells count complete expert successes out of 10. The basic tasks used 300 steps. The keyed plug used 600 steps and the other advanced tasks used 900 steps.

| Task | SO-101 | Dual SO-101 | PiPER | Dual PiPER |
| --- | ---: | ---: | ---: | ---: |
| `reach` | 10 | 10 | 10 | 10 |
| `pick_place` | 10 | 4 | 0 | 0 |
| `stack` | 10 | 4 | 0 | 0 |
| `insert` | 0 | 2 | 0 | 0 |
| `open_drawer` | 0 | 0 | 0 | 10 |
| `handover` | — | 2 | — | 0 |
| `cooperative_carry` | — | 0 | — | 0 |
| `lift_then_place` | 10 | 4 | 0 | 0 |
| `drawer_stow_close` | — | — | — | 10 |
| `keyed_plug_insert` | — | 1 | — | — |
| `bimanual_stabilized_insert` | — | — | — | 0 |

Nine of 31 combinations meet the 8/10 gate. All 203 nominal failures were timeouts. Extending all 28 basic combinations to 900 steps produced the same per-combination success counts.

## Precision runs

The drawer completed 10/10. The fixed plug completed 0/10, with nine timeouts in alignment and one in insertion or release. The mobile socket completed 0/10, with six timeouts in stabilization and four in alignment. In nominal mode, the fixed plug's nine failures ended in alignment. The mobile socket's nominal failures ended in stabilization seven times and alignment three times.

The server worktree contains per-seed results, recordings and audit scripts in `outputs/mujoco_expert_verification_2d36f66/`. The public [benchmark commands](mujoco_advanced_tasks.md) can rerun the three advanced combinations.

## Recording and replay

One nominal rollout per combination was recorded with the standard LeRobot recorder. The selection used the lowest successful seed when available and seed 0 otherwise. All 31 episodes (15 successes, 16 failures) reopened, decoded their first and last 320 × 240 RGB frames, preserved the outcome and scripted expert source, and matched the expected terminal state and failure reason. Reapplying the saved actuator controls reproduced all 7,453 recorded MuJoCo position states exactly.
