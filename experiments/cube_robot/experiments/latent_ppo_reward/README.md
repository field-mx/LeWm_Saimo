# Experiment 1: Latent Reward + PPO

This experiment treats LeWM as a learned latent environment. It has five stages:

1. Collect short real Cube rollouts and label each five-action block with task progress.
2. Predict the next latent with LeWM and train `RewardNetwork(z_pred, z_goal)`.
3. Distill behavior data into a latent action-block Actor.
4. Run PPO in imagined latent rollouts while retaining a frozen BC anchor.
5. Evaluate the trained Actor in the real MuJoCo Cube environment.

The PPO Actor outputs five raw environment actions in `[-1, 1]`. They are standardized
only at the LeWM boundary. One PPO transition therefore corresponds to five MuJoCo
actions.

## Run

```bash
cd /home/muxiang/work/LeWm_Saimo/experiments/cube_robot/experiments/latent_ppo_reward
./run_all.sh
```

Run the small interface test:

```bash
./run_all.sh --smoke --force-collect
```

Run one stage at a time:

```bash
/publicworkspace/envs/le-wm-py310/bin/python run_experiment.py --stage collect --force-collect
/publicworkspace/envs/le-wm-py310/bin/python run_experiment.py --stage reward
/publicworkspace/envs/le-wm-py310/bin/python run_experiment.py --stage ppo
/publicworkspace/envs/le-wm-py310/bin/python run_experiment.py --stage evaluate
```

## Outputs

- `outputs/data/reward_rollouts.npz`: latent transitions and reward labels.
- `outputs/data/reward_rollouts.json`: collection quality and LeWM timing.
- `outputs/checkpoints/reward_best.pt`: best validation Reward network.
- `outputs/checkpoints/latent_ppo_actor.pt`: PPO Actor and Critic.
- `outputs/logs/reward_training.jsonl`: one record per Reward epoch.
- `outputs/logs/ppo_bc_initialization.jsonl`: latent Actor initialization loss.
- `outputs/logs/ppo_training.jsonl`: one record per PPO update.
- `outputs/evaluation.json`: real-environment task success.
- `outputs/videos/latent_ppo_episode_0.mp4`: first evaluation episode.

The smoke test writes the same layout below `outputs/smoke/`.

## Reading logs

For Reward training, compare `train_mse` and `validation_mse`. Both should fall;
a widening gap indicates overfitting. `mean_prediction_mse` in the collection metadata
measures LeWM transition error and is not the task reward.

For PPO, `imagined_reward_mean` should improve without `value_loss` diverging.
`done_fraction` shows imagined resets. `world_model_inference_seconds` is synchronized
GPU time spent on LeWM transitions in that update. Final authority remains the real
MuJoCo `success_rate` in `evaluation.json`.

This is a model-based RL baseline. High imagined reward with low real success indicates
Reward-model exploitation or accumulated LeWM error.
