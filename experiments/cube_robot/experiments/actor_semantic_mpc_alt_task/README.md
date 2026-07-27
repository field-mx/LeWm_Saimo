# Alternate Start/Goal Online Adaptation

该实验复用 `actor_semantic_mpc` 的完整架构，并通过 `task_info` 显式指定任务：

```text
名义起点: (0.5000, -0.2000, 0.0200) m
固定终点: (0.3500,  0.2000, 0.0200) m
名义距离: 0.4272 m
```

OGBench会对起点XY加入最多1 cm扰动；种子固定为4042，因此实际起点可复现。

Actor从原始BC权重开始。在一个连续episode内，每一步执行：

```text
Actor基础动作
-> 语义阶段候选链
-> LeWM评分
-> MuJoCo执行第一步
-> 真实语义验证
-> 有效动作进入回放池
-> Actor在线更新
```

运行：

```bash
./run_online_training.sh
```

主要输出：

```text
outputs/videos/semantic_mpc_training_episode_0.mp4
outputs/logs/training.jsonl
outputs/training_summary.json
outputs/checkpoints/semantic_actor.pt
```
