# Proprio V2：基于冻结 LeWM 的机械臂抓取与转移

本目录保留当前 Cube 机械臂实验的最终版本：**Proprio V2 三阶段闭环规划系统**。系统不重新训练 LeWM，而是在冻结视觉世界模型上训练语义识别、阶段目标、动作策略与价值网络；在线控制以 RGB 图像、LeWM latent 和机械臂本体状态为输入。

视频均由 MuJoCo 真实仿真渲染。LeWM 不生成视频，只预测候选动作链在 latent 空间中的短期结果，并参与 MPC 动作评分。

## 核心结果：抓握确认率 7/10

正式在线实验运行 10 个 episode，结果如下：

| 指标 | 结果 | 含义 |
| --- | ---: | --- |
| 抓握确认率 | **7/10** | 状态机满足抓握判定并进入 `TRANSFER` 的 episode 数。 |
| 系统任务成功率 | 2/10 | 当前 latent 与 transfer-goal latent 连续达到完成阈值。 |
| MuJoCo 物理成功率 | 3/10 | 仿真器物理 success 为真；Episode 8 物理完成但没有通过连续 latent 阈值。 |
| LeWM 调用次数 | 1697 | 全部 10 个 episode 的合计。 |
| LeWM 平均推理时间 | 31.70 ms/次 | 世界模型总推理时间 53.80 秒。 |

当前主要瓶颈已经转移到抓握后的物块搬运与目标放置。

结果来自固定任务和有限样本，不能直接视为跨任务泛化能力。

视频与结果：

- [全部 10 个在线实验视频](outputs/proprio_v2/videos/proprio_v2_round1/)
- [Episode 1：系统判定任务成功](outputs/proprio_v2/videos/proprio_v2_round1/proprio_v2_episode_1_success.mp4)
- [Episode 6：系统判定任务成功](outputs/proprio_v2/videos/proprio_v2_round1/proprio_v2_episode_6_success.mp4)
- [任务专属 transfer goal RGB](outputs/proprio_v2/videos/proprio_v2_round1/transfer_goal_rgb.png)
- [完整结果汇总](outputs/proprio_v2/proprio_v2_round1_summary.json)

## 任务与运行时信息

| 项目 | 设置 |
| --- | --- |
| 物块初始位置 | `[0.40, -0.15, 0.02]` |
| 物块目标位置 | `[0.35, 0.20, 0.02]` |
| 图像尺寸 | `224 x 224` |
| LeWM latent 维度 | 192 |
| 阶段 | `ALIGN -> GRASP -> TRANSFER` |

在线模型只接收当前/上一 RGB、任务目标 RGB 和机械臂自身 6 维本体状态。6 维本体状态为末端 XYZ、yaw 的 `cos/sin` 和夹爪闭合程度。物块坐标、目标距离和接触信息只用于离线训练标签与最终评测，不作为在线输入。

## 系统架构

```text
当前 RGB、上一帧 RGB、任务目标 RGB
                |
        冻结 LeWM encoder
                |
 z_prev, z_t, z_goal (192D) + robot proprio (6D)
                |
  SemanticNet + TransitionNet -> 当前阶段判断
                |
  TargetNet -> ALIGN / GRASP 的短期目标 latent
                |
  Actor -> 5 x 5 基准动作块
                |
  MPC 采样 96 条候选动作链
                |
  冻结 LeWM rollout + Value Ensemble 评分
                |
  执行一个真实动作，读取新 RGB/本体状态，再次规划
```

| 模块 | 作用 |
| --- | --- |
| `ProprioSemanticNet` | 从 latent 时序和本体时序预测 `aligned`、`grasped`。 |
| `ProprioTransitionNet` | 根据视觉、本体、语义概率和当前阶段提出阶段切换建议。 |
| `ProprioTargetNet` | 为 ALIGN、GRASP 阶段预测短期目标 latent。 |
| `ProprioBlockActor` | 输出 `5 x 5` 的基准动作块。 |
| `ProprioValueEnsemble` | 给候选状态估计长期价值和不确定性。 |
| 冻结 LeWM | 从 latent 和动作块预测未来 latent。 |

模型定义见 [proprio_v2_models.py](proprio_v2_models.py)，入口见 [run_proprio_v2_experiment.py](run_proprio_v2_experiment.py)。

## 状态机

### ALIGN

`aligned >= 0.65` 连续 2 帧，且阶段网络有足够置信度后，进入 GRASP。最多执行 90 步；夹爪动作被约束为打开。

### GRASP

GRASP 阶段不因对齐分数的短暂波动而回退。当下列条件连续 1 帧成立时，立即确认抓握并进入 TRANSFER：

```text
grasped probability >= 0.55
AND gripper closure >= 0.45
```

若 55 步内未确认抓握，才返回 ALIGN。该阶段强制夹爪闭合。

### TRANSFER

确认抓握后，系统不再调用 TargetNet；阶段目标改为该任务在 MuJoCo 中渲染出的最终 RGB 所编码的 `transfer_goal_latent`。夹爪保持闭合，MPC 持续规划搬运动作。

当前 latent 与 `transfer_goal_latent` 的 MSE 不高于 `0.017` 且连续 2 帧满足时，系统判定完成。该视觉阈值与物理成功并不完全一致，Episode 8 即为物理完成但系统未确认的例子。

## MPC 与执行

Actor 输出 5 个 5 维动作。MPC 围绕它采样 96 条候选动作链；每条链包含 5 个动作块，因此 LeWM 评估的短期视野为：

```text
5 blocks x 5 actions = 25 raw actions
```

候选链评分为：

```text
score = -0.85 * short_target_cost
        +0.70 * terminal_value
        +0.15 * trajectory_value
        -0.20 * uncertainty
        -5.00 * behavior_deviation
        -0.02 * smoothness
```

每轮仅执行最佳候选的第一个真实动作，再从真实新画面重新编码、重新规划。这避免一次性开环执行 25 步世界模型预测。

## 训练与在线更新

规则型 `PrivilegedTeacher` 尝试 67 条轨迹并保留 64 条成功轨迹，共 3007 帧。训练/验证按完整 episode 划分，所有帧均使用，不做帧采样。

训练样本包含前一/当前/下一 latent、本体状态、动作块、阶段目标、语义标签、奖励与终止标记。离线标签为：

- `aligned`：未接触物块，XY 误差不超过 4 cm，XYZ 误差不超过 3.5 cm。
- `grasped`：接触分数不低于 0.50，夹爪闭合程度不低于 0.45。

离线目标：

```text
L = 2.0 * semantic_loss
  + 1.0 * transition_loss
  + 1.5 * target_loss
  + 2.0 * actor_loss
  + 0.5 * critic_loss
```

随后运行 10 个 episode 的在线更新。在线阶段仅更新 Actor；语义网络、阶段网络、TargetNet、Critic 和 LeWM 均冻结。只有满足规划优势、语义进度和安全条件的真实状态-动作对才会写入 replay buffer。

## 关键文件

```text
latent_three_phase_mpc/
├── README.md                         # 本文档
├── run_proprio_v2_experiment.py      # 训练 + 在线推理入口
├── proprio_v2_models.py              # 五类网络定义
├── proprio_v2_config.yaml            # 任务、训练、MPC、状态机配置
├── privileged_teacher.py             # 专家轨迹与离线标签
├── run_experiment.py                 # LeWM/环境/视频公共工具
├── models.py                         # run_experiment.py 的导入依赖
└── outputs/proprio_v2/
    ├── checkpoints/
    │   ├── proprio_v2_offline.pt
    │   └── proprio_v2_round1_actor_final.pt
    ├── data/proprio_v2_task_continuous.npz
    ├── logs/
    ├── proprio_v2_round1_summary.json
    └── videos/proprio_v2_round1/
```

冻结 LeWM 权重位于：

```text
experiments/cube_robot/model/lewm-cube-mapped/
```

## 运行

从仓库根目录执行：

```bash
PYTHON=/publicworkspace/envs/le-wm-py310/bin/python

$PYTHON experiments/cube_robot/experiments/latent_three_phase_mpc/run_proprio_v2_experiment.py \
  --stage adapt \
  --run-name proprio_v2_validation
```

该命令从 `outputs/proprio_v2/checkpoints/proprio_v2_offline.pt` 加载模型，输出视频到：

```text
outputs/proprio_v2/videos/proprio_v2_validation/
```

使用最终在线 Actor 继续更新：

```bash
$PYTHON experiments/cube_robot/experiments/latent_three_phase_mpc/run_proprio_v2_experiment.py \
  --stage adapt \
  --run-name proprio_v2_continue \
  --resume-checkpoint experiments/cube_robot/experiments/latent_three_phase_mpc/outputs/proprio_v2/checkpoints/proprio_v2_round1_actor_final.pt
```

`--stage train` 会重新采集专家轨迹并训练上层网络；完整训练还需要本地 OGBench 动作统计数据。现有 checkpoint 足以直接执行 `--stage adapt` 推理与在线更新。

## 当前局限

- 抓握确认显著改善，但 TRANSFER 仍是完整任务成功的主要瓶颈。
- latent MSE 是视觉目标接近信号，不等价于真实物理完成度，需要继续校准成功判定。
- 结果来自固定任务和 10 个 episode，不能代表跨位置、跨物块或真实机器人泛化能力。
