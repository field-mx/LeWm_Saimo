# Latent Three-Phase MPC

This experiment decomposes cube manipulation into three visual phases:

1. `grasp`: approach, descend, and close the gripper.
2. `transfer`: lift and move the grasped cube toward the goal.
3. `release`: lower, open the gripper, and finish.

Privileged MuJoCo coordinates are used only while collecting teacher labels and
computing training targets. Closed-loop planning and online adaptation use RGB
frames, frozen LeWM latents, and learned networks. Environment success is kept
only as an evaluation metric and unavoidable episode termination signal.

## Components

- `VisualPhaseNet`: predicts the current phase and visual completion probability.
- `PhaseTargetNet`: predicts the latent endpoint for the active phase.
- `PhaseBlockActor`: proposes five blocks of five-dimensional actions.
- `PhaseValueEnsemble`: estimates long-term value and epistemic uncertainty.
- LeWM MPC: rolls out 96 candidate action chains for 5 blocks (25 raw actions),
  scores them against the phase target and critic, then executes 1/3/1 actions
  in the grasp/transfer/release phases before replanning.

Online adaptation accepts an actor target only when both the planner score and
the critic value measured from the next real RGB frame improve. A strong
behavior-cloning anchor and a small learning rate prevent rapid policy drift.

## Commands

Run the complete experiment:

```bash
cd /home/muxiang/work/LeWm_Saimo
/publicworkspace/envs/le-wm-py310/bin/python \
  experiments/cube_robot/experiments/latent_three_phase_mpc/run_experiment.py \
  --stage all --regenerate-data
```

Run a quick interface smoke test:

```bash
/publicworkspace/envs/le-wm-py310/bin/python \
  experiments/cube_robot/experiments/latent_three_phase_mpc/run_experiment.py \
  --stage all --smoke --regenerate-data
```

Reuse the offline checkpoint and run only online adaptation plus evaluation:

```bash
/publicworkspace/envs/le-wm-py310/bin/python \
  experiments/cube_robot/experiments/latent_three_phase_mpc/run_experiment.py \
  --stage adapt
```

Run frozen inference only:

```bash
/publicworkspace/envs/le-wm-py310/bin/python \
  experiments/cube_robot/experiments/latent_three_phase_mpc/run_experiment.py \
  --stage evaluate
```

## Outputs

- `outputs/videos/online_training_selected.mp4`: successful selected trajectory
  produced while the actor is being adapted.
- `outputs/videos/frozen_evaluation_selected.mp4`: successful selected trajectory
  produced with the saved actor frozen.
- `outputs/checkpoints/three_phase_actor_offline.pt`: offline checkpoint.
- `outputs/checkpoints/three_phase_actor_final.pt`: selected online checkpoint.
- `outputs/logs/*.jsonl`: per-step planning, training, and timing records.
- `outputs/summary.json`: episode-level results and LeWM timing.
- `outputs/round1_baseline/`: unmodified first-round comparison artifacts.

Important log fields:

- `phase_before` / `phase_after`: visual phase transition.
- `planner_advantage`: selected chain score minus actor-chain score.
- `real_rgb_value_progress`: critic progress after a real MuJoCo RGB observation.
- `online_sample_accepted`: whether this transition entered online replay.
- `adaptation_loss`: online actor update loss.
- `world_model_inference_seconds`: LeWM rollout time for that planning cycle.
- `oracle_success_for_metrics_only`: simulator success used only for reporting.

The current formal run succeeds in 2/4 online-adaptation episodes and 2/4
frozen-evaluation episodes. The selected successful trajectory completes all
three phases in 46 environment steps.
