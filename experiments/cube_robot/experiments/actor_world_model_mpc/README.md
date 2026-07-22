# Experiment 2: Actor-Guided World-Model MPC

This experiment keeps the behavior-cloned 33-D state Actor and adds a LeWM planning
loop around it:

1. The Actor proposes the center action from the current real state and fixed goal.
2. Temporally correlated noise creates multiple 25-action candidate chains.
3. LeWM predicts the final latent for each chain and scores its goal-latent cost.
4. A candidate is accepted only when it sufficiently improves on the repeated BC proposal.
5. The accepted chain executes one action in MuJoCo before replanning.
6. Accepted elite actions supervise a small online Actor update with a frozen BC anchor.
7. The process repeats from the new real state until success or the step limit.

This is planner-guided policy improvement, not PPO. The original frozen Actor supplies
an anchor loss so online adaptation cannot immediately erase behavior-cloning skills.

## Run

```bash
cd /home/muxiang/work/LeWm_Saimo/experiments/cube_robot/experiments/actor_world_model_mpc
./run_all.sh
```

Run the four-step interface test:

```bash
./run_all.sh --smoke
```

## Outputs

- `outputs/logs/planning.jsonl`: one line per MPC planning round.
- `outputs/checkpoints/adapted_actor.pt`: Actor after elite-action updates.
- `outputs/evaluation.json`: real-environment success and aggregate LeWM time.
- `outputs/videos/actor_world_model_mpc_episode_0.mp4`: first episode.
- `outputs/resolved_config.json`: exact settings used for the run.

The smoke test writes below `outputs/smoke/`.

## Reading logs

- `base_cost` and `best_cost`: BC-proposal and selected goal-latent MSE.
- `planner_accepted`: whether the candidate passed the relative-improvement gate.
- `relative_cost_improvement`: predicted gain over the BC proposal.
- `median_cost` and `worst_cost`: candidate spread and exploration quality.
- `adaptation_loss`: disagreement between the Actor and elite first action.
- `executed_steps`: real actions used before the next replanning round.
- `world_model_inference_seconds`: synchronized time for encoding, rollout and scoring.
- `cumulative_world_model_inference_seconds`: total LeWM planning time so far.

Judge the planner by `success_rate`, not by latent cost alone. Falling latent cost with
failed real episodes is direct evidence of world-model or representation mismatch.
