# Cube World-Model RL Experiments

- `latent_ppo_reward`: learned latent Reward network followed by PPO imagination.
- `actor_world_model_mpc`: behavior-cloned Actor proposals, LeWM scoring and online
  elite-action distillation.

Both directories own their configuration, logs, checkpoints, videos and results. They
reuse the existing Cube assets without modifying the PushT experiment or installed
`stable_worldmodel` package.

Formal baseline and experiment results are summarized in `RESULTS.md`.
