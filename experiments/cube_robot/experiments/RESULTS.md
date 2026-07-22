# Formal Results

All evaluations use `cube-single-play-v0`, seeds 42 through 51, and a 200-step
environment limit.

## Baseline

The unchanged behavior-cloned state Actor succeeded in 4 of 10 episodes. Its result is
stored in `baseline_actor_eval/evaluation.json`.

## Experiment 1: Latent Reward + PPO

- Reward collection: 709 five-action blocks from 20 episodes.
- Successful terminal blocks: 6.
- Mean LeWM one-block prediction MSE: 0.13315.
- Reward network best validation MSE: 0.00903 at epoch 21.
- Reward-data LeWM time: 7.64 seconds over 1,458 calls.
- Constrained PPO LeWM time: 19.82 seconds over 3,200 calls.
- Imagined mean reward: 0.531 at update 1, 0.753 at update 100.
- Real MuJoCo result: 0 of 10 successes.

The imagined objective improves while real performance remains zero. The reward data
contains very few successful examples, and the latent Actor must compress five
closed-loop state-dependent actions into one open-loop block. The final BC
initialization MSE is 0.112, so the latent policy does not preserve the original
state Actor's 40% baseline capability.

## Experiment 2: Actor-Guided LeWM MPC

- Candidate chains per planning round: 128.
- Predicted chain length: 25 environment actions.
- Executed actions before replanning: 1.
- Planning rounds: 2,000.
- Planner acceptance: 1,895 rounds, or 94.75%.
- Mean accepted predicted-cost improvement over the BC proposal: 47.77%.
- Minimum reported latent cost: 10.17.
- Total synchronized LeWM planning time: 68.18 seconds.
- Mean LeWM planning time: 0.0341 seconds per round.
- Real MuJoCo result: 0 of 10 successes.

The world model repeatedly reports large improvements over the BC proposal, but the
real success rate falls from 40% to zero. For this checkpoint and task horizon, final
latent MSE is not a calibrated action-ranking signal. Online distillation then copies
those ranking errors into the Actor.

## Interpretation

Both implementations and timing paths execute correctly, but neither method currently
improves the real task. The next technically justified iteration is to train a
task-aware value or success model from substantially more balanced real transitions,
add uncertainty or ensemble disagreement to reject out-of-distribution imagined
rollouts, and keep a real-environment validation gate before updating the Actor.

The first unconstrained runs are preserved in each experiment's
`outputs_round1_unconstrained/` directory.
