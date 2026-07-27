# Actor + Semantic-Gated LeWM MPC

This experiment preserves the state-based behavior-cloned Actor and changes the online
planner in three ways:

1. Candidate chains follow phase-aware pick-and-place templates instead of repeating
   one action for 25 steps.
2. LeWM still selects a candidate by final goal-latent cost, but only the first action
   is executed.
3. The executed action is distilled into the Actor only after real state semantics
   confirm progress in the current phase.

Training uses seeds 1042 through 1053. Frozen evaluation uses seeds 42 through 51.

## Run

```bash
cd /home/muxiang/work/LeWm_Saimo/experiments/cube_robot/experiments/actor_semantic_mpc
./run_all.sh
```

Run a short interface test:

```bash
./run_all.sh --smoke
```

Run stages independently:

```bash
/publicworkspace/envs/le-wm-py310/bin/python run_experiment.py --stage train
/publicworkspace/envs/le-wm-py310/bin/python run_experiment.py --stage evaluate
```

Run the direct-goal ablation with the same trained Actor and no intermediate phases:

```bash
/publicworkspace/envs/le-wm-py310/bin/python run_experiment.py \
  --config config_no_phases.yaml --stage evaluate
```

## Outputs

- `outputs/checkpoints/semantic_actor.pt`: Actor after verified online distillation.
- `outputs/logs/training.jsonl`: every training planning round and semantic decision.
- `outputs/logs/evaluation.jsonl`: frozen-policy evaluation planning rounds.
- `outputs/training_summary.json`: accepted/rejected actions and LeWM timing.
- `outputs/evaluation.json`: final real-environment success rate.
- `outputs/videos/semantic_mpc_episode_0.mp4`: first frozen evaluation episode.
- `outputs_ablation_no_phases/videos/semantic_mpc_episode_0.mp4`: same-seed
  direct-goal ablation without phase-aware action templates.
- `outputs_ablation_no_phases/videos/phase_vs_no_phases_seed42.mp4`: side-by-side
  comparison, phase-aware on the left and direct-goal on the right.
- `RESULTS.md`: formal metrics and the first-round ablation note.

An action updates the Actor only when both `planner_accepted` and
`real_semantic_accepted` are true. Inspect `phase_before`, `phase_after`,
`semantic_progress`, `contact_before`, `contact_after`, and `adaptation_loss` to see
why each action was accepted or rejected.

The final formal run achieved 12/12 training successes and 10/10 frozen evaluation
successes. See `RESULTS.md` for timing and interpretation.
