# Results

The final run uses 12 online-training episodes on seeds 1042 through 1053 and
10 frozen-policy evaluation episodes on seeds 42 through 51.

## Training

- Successes: 12 of 12.
- Completion range: 32 to 54 environment steps.
- LeWM calls: 517.
- Synchronized LeWM inference time: 18.77 seconds.
- Model-accepted actions: 501.
- Actions admitted by both model and real semantic gates: 500.

The Actor is updated only from the executed first action after real state semantics
confirm progress. The replay contains approach, descend, grasp, lift, transport, and
place transitions.

## Frozen Evaluation

- Successes: 10 of 10.
- Completion range: 42 to 68 environment steps.
- LeWM calls: 539.
- Synchronized LeWM inference time: 19.86 seconds.
- Mean LeWM inference time: 0.0368 seconds per planning round.
- Actor updates during evaluation: zero.

The first evaluation episode completed in 64 steps. Its 65-frame, 20 FPS video is
stored at `outputs/videos/semantic_mpc_episode_0.mp4`.

## Ablation Note

The first implementation checked transport height before destination XY alignment.
That caused a `place -> lift -> place` oscillation and produced zero successes in
frozen evaluation. Its complete outputs are preserved in
`outputs_round1_place_oscillation/`.

Changing the phase priority so destination-aligned cubes remain in `place` resolved
the oscillation. This is a phase-aware hybrid controller: LeWM ranks coherent
candidate chains, while true simulator semantics gate online distillation. The result
should not be presented as a purely latent or purely learned policy.

## No-Phase Direct-Goal Ablation

The same trained Actor, LeWM checkpoint, seed 42, 128 candidates, 25-action planning
horizon, and one-step execution policy were evaluated with `control_mode: direct_goal`.
This removes intermediate targets, phase-specific gripper intent, and phase templates.

- Successes: 0 of 1.
- Environment steps: 200, ending at the environment time limit.
- Model-accepted actions: 185 of 200.
- Synchronized LeWM inference time: 9.04 seconds.
- Cube-goal distance: 0.42895 m initially and 0.42888 m finally.
- Maximum contact value: 0.445, below the 0.50 grasp threshold.

The separate ablation video is stored at
`outputs_ablation_no_phases/videos/semantic_mpc_episode_0.mp4`. The side-by-side
comparison is stored at
`outputs_ablation_no_phases/videos/phase_vs_no_phases_seed42.mp4`, with the
phase-aware run on the left and the direct-goal run on the right.
