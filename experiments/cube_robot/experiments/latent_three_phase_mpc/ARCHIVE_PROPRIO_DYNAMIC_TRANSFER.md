# Proprioceptive Dynamic Transfer Archive

This archive records the final Cube experiment variant that augments the
latent actor and semantic head with the robot end-effector XYZ state.

## Experiment

- Entry point: `run_recovery_experiment.py`
- Configuration: `recovery_config.yaml`
- Initial checkpoint: `outputs/checkpoints/recovery_cube_semantic_actor.pt`
- Task: move the cube from `[0.40, -0.15, 0.02]` to
  `[0.35, 0.20, 0.02]`
- Online episodes: 10
- Historical result: 9/10 episodes entered TRANSFER after grasping; 5/10
  episodes reached the dynamic transfer latent goal.
- Successful historical episodes: 0, 2, 3, 4, and 5.

The environment truncated failed episodes at 200 steps even though the
experiment configuration allowed 300 online steps.

## Architecture Changes

- `VisualSemanticNet` receives previous/current end-effector XYZ and their
  delta in addition to RGB latents.
- `RecoveryBlockActor` receives current end-effector XYZ together with the
  current, phase-target, and task-goal latents.
- Candidate action chains update a predicted proprioceptive state during LeWM
  rollout.
- ALIGN and GRASP continue using the learned semantic target mechanism.
- TRANSFER uses a task-specific RGB reference rendered once at reset with the
  cube grasped inside the requested goal region. Runtime completion uses only
  RGB latent MSE plus the learned grasp predicate.

No cube or goal coordinates are supplied to the actor during closed-loop
inference. Simulator coordinates are used only to construct the one-time
transfer reference and to report oracle metrics.

## Evidence

`outputs/archive/proprio_dynamic_transfer_2026-08-13/original_5of10/full_run.log`
is the preserved console log from the original run. It contains all ten
per-episode records and the final 5/10 summary.

`original_5of10/transfer_goal_rgb.png` is the exact transfer reference from
that run. The original ten MP4 files were untracked and were removed when the
working branch changed, so they could not be recovered from Git objects.

`reproduction_1of10/` contains a later restoration-check rerun. It verifies
that the recovered code executes end to end, but its stochastic result was
1/10 and must not be presented as the historical 5/10 run.

## Command

```bash
/publicworkspace/envs/le-wm-py310/bin/python \
  experiments/cube_robot/experiments/latent_three_phase_mpc/run_recovery_experiment.py \
  --stage adapt \
  --resume-checkpoint experiments/cube_robot/experiments/latent_three_phase_mpc/outputs/checkpoints/recovery_cube_semantic_actor.pt \
  --run-name dynamic_transfer_goal_from_initial_actor_round1
```
