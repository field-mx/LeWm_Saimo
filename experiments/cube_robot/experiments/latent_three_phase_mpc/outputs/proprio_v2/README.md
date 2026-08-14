# Proprio V2 实验归档

本目录归档 `latent_three_phase_mpc` 最新实验。系统使用冻结的 LeWM 预测候选动作的短期 latent 轨迹，并结合机械臂本体状态、三阶段状态机、Actor 和价值网络完成 Cube 任务。视频均为 MuJoCo 真实仿真渲染，LeWM 不负责生成视频。

实验基于 [LeWorldModel](https://arxiv.org/abs/2603.19312)，没有重新训练世界模型。

## 任务

- 物块初始位置：`[0.40, -0.15, 0.02]`
- 目标位置：`[0.35, 0.20, 0.02]`
- 阶段：`ALIGN -> GRASP -> TRANSFER`
- 在线推理不读取物块坐标、目标真实距离或仿真接触信息

## 系统架构

### 状态表示

LeWM 将 `224 x 224` RGB 编码为 192 维 latent。运行时还输入 6 维机械臂状态：末端 XYZ、yaw 的 `cos/sin`、夹爪闭合程度。

基础 LeWM 权重：

- `experiments/cube_robot/model/lewm-cube-mapped/config.json`
- `experiments/cube_robot/model/lewm-cube-mapped/weights.pt`

### ProprioSemanticNet

```text
[z_prev, z_t, z_t-z_prev, p_prev, p_t, p_t-p_prev]
    -> [aligned_logit, grasped_logit]
```

输入维度 `3*192 + 3*6 = 594`，用于识别是否对齐和是否抓住物块。

### ProprioTransitionNet

输入视觉时序特征、本体时序特征、两个语义概率及当前阶段 one-hot，共 599 维，输出 `ALIGN/GRASP/TRANSFER` 三类 logits。它提供阶段分类和切换建议，最终转换仍由显式状态机约束。

### ProprioTargetNet

```text
[z_prev, z_t, z_goal, z_t-z_goal, mode_one_hot, p_t]
    -> z_phase (192 dimensions)
```

输入 777 维，为 ALIGN 和 GRASP 生成短期目标。确认抓握后绕过该网络，TRANSFER 直接使用本任务在 MuJoCo 中渲染的最终 RGB 对应 latent。

### ProprioBlockActor

```text
[z_prev, z_t, z_phase, z_goal,
 z_t-z_phase, z_t-z_goal, mode_one_hot, p_t]
    -> [5 actions, 5 dimensions per action]
```

输入 1161 维，输出经 `tanh` 约束的 `5 x 5` 动作块。

### ProprioValueEnsemble

两个 ValueNet 使用与 Actor 相同的 1161 维输入，各输出一个标量。规划器使用均值表示长期价值，标准差表示不确定性。

所有网络采用 `Linear -> LayerNorm -> SiLU -> Linear -> SiLU -> Linear`，隐藏维度为 384。实现见 [proprio_v2_models.py](../../proprio_v2_models.py)。

## 训练数据

数据由 OGBench MuJoCo 中的 `PrivilegedTeacher` 采集。规则专家产生成功轨迹；机械臂/物块坐标和接触信号只用于离线监督标签。

- 尝试 67 条轨迹，丢弃 3 条失败轨迹
- 保留 64 条完整成功轨迹
- 逐帧使用，不抽样
- 共 3007 帧
- 每条 38 至 59 帧，平均 46.98 帧
- 按完整轨迹划分，训练 2375 帧，验证 632 帧

每帧包含前一/当前/下一 latent、前一/当前/下一 6 维本体状态、目标 latent、阶段关键帧 latent、`5 x 5` 动作块、标签、奖励和终止标志。

离线标签：

- `aligned`：未接触物块，XY 误差不超过 4 cm，XYZ 误差不超过 3.5 cm
- `grasped`：接触分数不低于 `0.50`，夹爪闭合程度不低于 `0.45`

这些特权信息不进入在线推理。

## 离线训练

```text
L = 2.0 * L_semantic
  + 1.0 * L_transition
  + 1.5 * L_target
  + 2.0 * L_actor
  + 0.5 * L_value
```

- Semantic：加权 BCE
- Transition：阶段分类交叉熵
- Target：阶段目标 latent MSE
- Actor：专家动作块 MSE
- Value：TD MSE

训练参数为 80 epochs、batch size 128、学习率 `2e-4`、weight decay `1e-5`、TD `gamma=0.97`、target critic 软更新 `tau=0.01`。

最佳验证损失 `0.580548`（epoch 77），对应语义准确率 `0.966927`、阶段准确率 `0.982604`。优化器训练耗时 37.256 秒，不含采集与 RGB 编码。

离线权重：[proprio_v2_offline.pt](checkpoints/proprio_v2_offline.pt)。

## 状态机

### ALIGN

- 仅本阶段检测对齐
- `aligned >= 0.65` 连续 2 帧，并满足阶段建议置信度后进入 GRASP
- 最多 90 步
- 强制夹爪打开，夹爪动作不高于 `-0.35`

### GRASP

- 不再因实时对齐分数波动而回退
- `grasped >= 0.55` 且闭合程度 `>= 0.45`，确认 1 帧后立即进入 TRANSFER
- 不要求 TransitionNet 二次确认
- 55 步内未抓住才超时返回 ALIGN
- 强制夹爪闭合，夹爪动作不低于 `+0.35`

### TRANSFER

- 抓握确认后锁定，不再退回 ALIGN/GRASP
- 目标切换为任务专属 `transfer_goal_latent`
- 始终强制夹爪闭合
- 目标 latent MSE 不高于 `0.017`，连续 2 帧后判定成功

锁定可避免语义分类抖动导致突然松爪；局限是确认抓握后若物块真实掉落，本版本不会自动重新抓取。

## 世界模型 MPC

每个真实环境步骤重新规划：

1. Actor 生成基准动作块，周围采样 96 条候选链。
2. 每条链含 5 个动作块，每块 5 个 5 维动作，张量为 `[96,5,5,5]`。
3. LeWM 预测 5 个未来 latent，得到 `[96,5,192]`。
4. 综合短期误差、价值、不确定性、行为偏离与平滑性评分。
5. 从 8 条 elite 候选中按温度 `0.05` 选择。
6. 只执行第一个真实动作，读取新 RGB/本体状态后重规划。

```text
score = -0.85 * short_target_cost
        +0.70 * terminal_value
        +0.15 * trajectory_value
        -0.20 * uncertainty
        -5.00 * behavior_deviation
        -0.02 * smoothness
```

探索噪声初值 `0.08`，衰减系数 `0.98`。

## 在线自更新

10 个 episode 从离线 Actor 开始，其他模块冻结；Actor 更新会跨 episode 保留。通过筛选的真实状态动作对进入 replay buffer：

- planner advantage `>= 0.005`
- semantic progress `>= 0.002`
- capacity 4096，warmup 8，batch size 64
- Actor 学习率 `1e-6`
- BC anchor `1.0`
- 每个环境步骤最多更新一次

最终 Actor：[proprio_v2_round1_actor_final.pt](checkpoints/proprio_v2_round1_actor_final.pt)。

## 结果

- 10/10 确认抓握并进入 TRANSFER
- 系统成功 2/10：Episode 1、Episode 6
- MuJoCo 物理成功 3/10：Episode 1、Episode 6、Episode 8
- Episode 8 已物理完成，但未连续满足 latent MSE 阈值，因此系统记为失败
- LeWM 调用 1697 次，总耗时 53.798 秒，平均约 31.70 ms/次

展示：

- [Episode 1 成功视频](videos/proprio_v2_round1/proprio_v2_episode_1_success.mp4)
- [Episode 6 成功视频](videos/proprio_v2_round1/proprio_v2_episode_6_success.mp4)
- [全部视频](videos/proprio_v2_round1/)
- [任务目标 RGB](videos/proprio_v2_round1/transfer_goal_rgb.png)
- [在线结果汇总](proprio_v2_round1_summary.json)
- [训练汇总](complete64_round1_summary.json)

加入本体状态与锁定式抓握转换后，抓握已不是本轮主要瓶颈。当前问题主要是 TRANSFER 规划，以及 latent 成功阈值与物理成功之间的校准。失败回合实际在环境 200 步截断，虽然在线配置上限为 300 步。

结果仅针对固定任务和有限回合，不代表任意起终点泛化能力。

## 复现

在仓库根目录运行：

```bash
PYTHON=/publicworkspace/envs/le-wm-py310/bin/python
$PYTHON experiments/cube_robot/experiments/latent_three_phase_mpc/run_proprio_v2_experiment.py \
  --stage adapt \
  --run-name proprio_v2_reproduction
```

`adapt` 从 checkpoint 读取动作归一化参数，不依赖未归档的训练 `.npz`。

从最终 Actor 继续在线更新：

```bash
$PYTHON experiments/cube_robot/experiments/latent_three_phase_mpc/run_proprio_v2_experiment.py \
  --stage adapt \
  --run-name proprio_v2_continue \
  --resume-checkpoint experiments/cube_robot/experiments/latent_three_phase_mpc/outputs/proprio_v2/checkpoints/proprio_v2_round1_actor_final.pt
```

重新采集并训练：

```bash
$PYTHON experiments/cube_robot/experiments/latent_three_phase_mpc/run_proprio_v2_experiment.py \
  --stage train \
  --run-name complete64_retrain
```

重新训练需要 OGBench/MuJoCo 和配置指定的 Cube 原始动作数据。原始轨迹未归档，训练权重已保留。`--stage all` 可连续训练和在线实验。

主要依赖：Python 3.10、PyTorch/CUDA、`stable_worldmodel`、OGBench、MuJoCo、torchvision、scikit-learn、PyYAML、imageio、FFmpeg。

## 关键文件

- [实验入口](../../run_proprio_v2_experiment.py)
- [模型结构](../../proprio_v2_models.py)
- [配置](../../proprio_v2_config.yaml)
- [规则专家](../../privileged_teacher.py)
- [环境与 LeWM 工具](../../run_experiment.py)
- [离线权重](checkpoints/proprio_v2_offline.pt)
- [最终 Actor](checkpoints/proprio_v2_round1_actor_final.pt)
- [结果汇总](proprio_v2_round1_summary.json)
- [视频目录](videos/proprio_v2_round1/)
