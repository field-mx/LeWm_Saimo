# RGB-Latent MPC Cube 实验归档

> 归档日期：2026-08-05<br>
> 状态：实验研究完成，代码和产物保留；当前结果不代表稳定可部署策略。
> 当前脚本缺少一个未归档的教师工具模块，详见“复现实验/当前可执行性”。

本目录记录了使用冻结 LeWM、视觉语义状态机、Actor、Critic 和 MPC 完成
MuJoCo Cube 抓取转移任务的主要实验。训练数据可使用 MuJoCo 坐标生成监督标签，
但在线规划闭环以 RGB 图像及其 LeWM latent 为输入；仿真器坐标和原生 success
只用于生成训练标签与最终评测，不作为规划器的在线输入。

视频均为 **MuJoCo 真实仿真渲染**。LeWM 不生成最终视频，只预测候选动作链的
latent 演化并参与评分。

## 归档结论

1. 冻结 LeWM 可以作为 25 个原始动作左右的短期预测器，并辅助 Actor 在候选动作链中选择动作。
2. 仅使用终点 latent MSE 不能稳定表示任务进度。专家轨迹分析表明，该误差随时间并非单调下降。
3. 分阶段语义控制对抓取任务很重要。原始三阶段实验达到 `2/4`，去掉 Critic 后为 `1/4`。
4. 五阶段可恢复状态机的历史最佳在线结果为 `3/10`；按相同起始检查点重新运行得到 `1/10`，说明规划采样、视觉判定和在线更新仍有明显不稳定性。
5. 三目标状态机初版为 `2/10`；增加显式抓握超时与开爪重置后为 `0/10`。重试避免了部分停滞，但没有解决精细对齐和抓握本身。
6. 主要失败点是对齐、闭爪时机和抓握保持，而不是单纯缺少执行步数。当前方案适合作为研究原型，不应将小样本成功率理解为泛化能力。

## 最终系统流程

一次闭环规划按以下顺序运行：

```text
当前 RGB + 目标 RGB
        |
冻结 LeWM encoder -> 当前 latent、目标 latent
        |
视觉语义网络 -> 当前阶段/语义谓词
        |
阶段目标网络 -> 当前阶段的短期目标 latent
        |
Actor -> 基准动作链（5 blocks x 5 actions = 25 raw actions）
        |
在基准动作链附近采样 96 条候选链
        |
LeWM -> 为每条候选链预测 latent 链
        |
短期目标误差 + Critic 长期价值 + 不确定性 + 动作先验/平滑项
        |
选择最优链，执行当前阶段配置的 1 个动作，再读取新 RGB 并重新规划
        |
满足在线样本筛选条件时，将真实 RGB 转移加入 replay 并更新 Actor
```

这里的 `action_block: 5` 表示 LeWM 每个 block 接收 5 个原始动作；
`horizon_blocks: 5` 表示模型评估 5 个 block，因此每条候选链覆盖 25 个原始动作。
正式五阶段配置只执行候选链的第 1 个动作，然后使用真实新画面闭环重规划。

### 网络职责

- `RecoverySemanticNet`：从 RGB latent 预测 `aligned`、`grasped`、`near_goal`、`released`、`retreated`、`complete` 六个视觉谓词。
- `RecoveryTargetNet`：根据当前 latent、最终目标 latent 和当前状态，给出阶段短期目标 latent。
- `RecoveryBlockActor`：输出动作链的基准值，MPC 在其周围采样候选链。
- `RecoveryValueEnsemble`：估计候选 latent 状态的长期价值和模型不确定性。
- 冻结 LeWM：预测候选动作执行后的 latent 链，为 MPC 提供短期动力学模型。

Critic 在训练时使用 TD 目标更新；规划时参与候选动作链评分。Actor 不是直接通过
LeWM 反向传播更新，而是把筛选后的高质量状态-动作目标加入 replay，再以行为克隆
锚定的方式在线蒸馏，减少策略快速漂移。

## 状态机版本

### 原始三阶段

脚本：`run_experiment.py`，配置：`config.yaml`。

```text
grasp -> transfer -> release
```

这是最早的可用版本，没有显式 `align` 和 `retreat` 状态，在线训练和冻结评测各运行 4 回合。

### 五阶段恢复状态机（推荐归档基线）

脚本：`run_recovery_experiment.py`，配置：`recovery_config.yaml`，模型：`recovery_models.py`。

```text
align -> grasp -> transfer -> release -> retreat
```

状态由视觉谓词决定，连续 2 帧满足阈值才切换；状态超时后可回退恢复。例如抓握超时
回到 `align`，转移阶段丢失抓握则回到 `grasp` 或 `align`。最终 success 必须同时满足：

```text
视觉 complete 连续达到阈值
AND
MuJoCo info["success"] 为真（仅用于最终评测）
```

当前正式配置的每状态最大步数为 `[45, 30, 55, 25, 30]`，每回合总上限为 180 步。

### 三目标状态机（消融实验）

脚本：`run_three_goal_experiment.py`，配置：`three_goal_config.yaml`。

```text
go_align <-> go_grasp -> go_complete
```

该版本把阶段压缩为三个过程目标，并测试了抓握失败后的强制开爪重置。它改善了
“夹爪闭合后不再动作”的控制逻辑，但没有提高最终成功率，因此没有替代五阶段基线。

## 实验结果

以下数字直接来自相应 `summary.json`，样本量较小，只用于本项目内部比较。

| 实验 | 更新方式 | 成功率 | LeWM 调用/平均耗时 | 权威结果文件 |
| --- | --- | ---: | ---: | --- |
| 原始三阶段 | 在线更新 | `2/4` | `226 / 35.56 ms` | `outputs/summary.json` |
| 原始三阶段 | 冻结 Actor | `2/4` | `177 / 36.36 ms` | `outputs/summary.json` |
| 三阶段去 Critic | 冻结 Actor | `1/4` | `194 / 26.98 ms` | `outputs/no_critic_summary.json` |
| 五阶段恢复首轮 | 在线更新 | `2/10` | `1479 / 30.53 ms` | `outputs/recovery_summary.json` |
| 五阶段继续训练 round 2（历史最佳） | 在线更新 | `3/10` | `1286 / 33.51 ms` | `outputs/recovery_state_machine_continue_round2_summary.json` |
| 三目标初版 | 在线更新 | `2/10` | `1633 / 32.09 ms` | `outputs/three_goal_state_machine_v1/summary.json` |
| 三目标 + 抓握重置 | 在线更新 | `0/10` | `1800 / 31.07 ms` | `outputs/three_goal_grasp_reset_v1/summary.json` |
| 五阶段恢复最新复跑 | 在线更新 | `1/10` | `1659 / 31.19 ms` | `outputs/recovery_state_machine_restored_latest_summary.json` |

历史最佳 round 2 成功回合为 3、5、8，分别执行 54、46、51 步。最新复跑仅回合 6
成功，执行 142 步。两次运行都从 `recovery_actor_online_final.pt` 开始，差异说明即使
固定任务和配置，候选采样、GPU 数值行为以及逐回合在线更新也会使后续轨迹分叉。

## 目录与关键文件

```text
latent_three_phase_mpc/
├── README.md                         # 本归档文档
├── config.yaml                       # 原始三阶段配置
├── models.py                         # 原始三阶段网络
├── run_experiment.py                 # 原始三阶段训练/在线适应/冻结评测
├── run_no_critic_evaluation.py       # 去 Critic 消融
├── recovery_config.yaml              # 五阶段正式配置
├── recovery_models.py                # 五阶段语义、Actor、Target、Critic 网络
├── run_recovery_experiment.py        # 五阶段数据、训练、在线恢复实验
├── three_goal_config.yaml            # 三目标状态机配置
├── run_three_goal_experiment.py      # 三目标实验
├── analyze_round_latents.py          # 真实 latent 与 LeWM 预测误差分析
└── outputs/
    ├── checkpoints/                  # 模型检查点
    ├── data/                         # 教师 latent 数据
    ├── logs/                         # 每步 JSONL 日志
    ├── videos/                       # 原始/五阶段/消融视频
    ├── analysis/                     # latent 分析图与统计
    └── *_summary.json                # 回合级权威结果
```

依赖的外部资产：

- LeWM Cube 模型：`experiments/cube_robot/model/lewm-cube-mapped`
- OGBench 动作数据：`experiments/cube_robot/data/ogbench_state/cube-single-play-v0.npz`
- 教师工具模块（脚本导入时需要）：`experiments/cube_robot/experiments/rgb_latent_value_mpc/privileged_teacher.py`（当前归档中缺失）
- 坐标 Actor 初始化（仅重新收集教师数据时需要）：`experiments/cube_robot/experiments/actor_semantic_mpc/outputs/checkpoints/semantic_actor.pt`（当前归档中缺失）

## 复现实验

### 当前可执行性

当前仓库保留了数据、检查点、日志、视频和核心实验脚本，但缺少未被 Git 跟踪的
`rgb_latent_value_mpc/privileged_teacher.py`。`run_experiment.py` 在模块导入阶段加载它，
`run_recovery_experiment.py` 和 `run_three_goal_experiment.py` 又会导入 `run_experiment.py`，
因此当前冷启动运行会先报 `ModuleNotFoundError: privileged_teacher`，包括 `--help`。

下面命令是本次实验实际使用的调用方式，必须先从原实验备份恢复该模块。若还要
`--regenerate-data`，还需恢复上面列出的坐标 Actor checkpoint。仅查看 summary、日志、
视频和加载现有 `.pt`/`.npz` 产物不受影响。

所有命令从仓库根目录执行：

```bash
cd /home/muxiang/work/LeWm_Saimo
```

使用服务器现有 Python 环境：

```bash
PYTHON=/publicworkspace/envs/le-wm-py310/bin/python
```

### 原始三阶段完整流程

```bash
$PYTHON experiments/cube_robot/experiments/latent_three_phase_mpc/run_experiment.py \
  --stage all
```

该命令复用已归档的 `outputs/data/three_phase_latents.npz`，不依赖缺失的坐标 Actor。

仅冻结评测：

```bash
$PYTHON experiments/cube_robot/experiments/latent_three_phase_mpc/run_experiment.py \
  --stage evaluate
```

### 五阶段恢复实验

使用已归档教师数据执行离线训练和在线训练：

```bash
$PYTHON experiments/cube_robot/experiments/latent_three_phase_mpc/run_recovery_experiment.py \
  --stage all --run-name recovery_archive
```

该命令复用 `outputs/data/recovery_semantic_latents.npz`。若要使用
`--regenerate-data` 从 MuJoCo 重新收集教师数据，必须先恢复配置中引用但当前归档缺失的
`actor_semantic_mpc/outputs/checkpoints/semantic_actor.pt`；否则教师 `PrivilegedTeacher`
初始化会失败。`--stage adapt` 不使用该坐标 Actor checkpoint，但当前代码仍需
`privileged_teacher.py` 才能完成模块导入。

从现有 Actor 继续 10 回合在线更新：

```bash
$PYTHON experiments/cube_robot/experiments/latent_three_phase_mpc/run_recovery_experiment.py \
  --stage adapt \
  --resume-checkpoint experiments/cube_robot/experiments/latent_three_phase_mpc/outputs/checkpoints/recovery_actor_online_final.pt \
  --run-name recovery_continue_archive
```

从历史最佳 round 2 参数继续：

```bash
$PYTHON experiments/cube_robot/experiments/latent_three_phase_mpc/run_recovery_experiment.py \
  --stage adapt \
  --resume-checkpoint experiments/cube_robot/experiments/latent_three_phase_mpc/outputs/checkpoints/recovery_state_machine_continue_round2_actor_final.pt \
  --run-name recovery_from_best_archive
```

`--run-name` 会隔离日志、视频、summary 和最终检查点，避免覆盖旧实验。
`--smoke` 可缩短数据量和回合数，只验证接口，不代表正式结果。

### 三目标实验

```bash
$PYTHON experiments/cube_robot/experiments/latent_three_phase_mpc/run_three_goal_experiment.py \
  --stage all --rebuild-data --run-name three_goal_archive
```

### Latent 误差分析

```bash
$PYTHON experiments/cube_robot/experiments/latent_three_phase_mpc/analyze_round_latents.py
```

分析结果保存在 `outputs/analysis/`。其中真实 RGB 经 encoder 得到的 latent 与阶段目标
latent 的距离用于观察语义进度；LeWM 预测 latent 与同一时刻真实 latent 的距离用于区分
世界模型误差和规划误差。不能把单一 latent MSE 直接当作单调 reward。

## 日志阅读

正式日志为 `outputs/logs/<run-name>/online_training.jsonl`，每行是一个规划周期或回合总结。

- `state_before` / `state_after`：状态机切换前后状态。
- `predicate_probabilities`：六个视觉语义谓词的预测概率。
- `planner_advantage`：所选候选链相对 Actor 基准链的得分改进。
- `real_rgb_value_progress`：执行后真实 RGB latent 的 Critic 价值变化。
- `real_rgb_semantic_progress`：执行后阶段语义进度变化。
- `online_sample_accepted`：该真实转移是否进入在线 replay。
- `adaptation_loss`：当前在线 Actor 更新损失。
- `world_model_inference_seconds`：本次候选链 LeWM 推演耗时。
- `forced_recovery` / `rollback`：是否由超时或语义退化触发恢复。
- `visually_complete`：视觉网络是否连续判定完成。
- `oracle_success_for_metrics_only`：仿真器 success，仅用于最终报告。

归档统计以 summary 中的 `success` 为准。五阶段脚本中的 `success` 同时要求视觉完成和
仿真器成功，因此仅有 `visually_complete: true` 仍可能被记为失败。

## 权威产物

### 历史最佳五阶段结果（`3/10`）

- Summary：`outputs/recovery_state_machine_continue_round2_summary.json`
- Resolved config：`outputs/recovery_state_machine_continue_round2_resolved_config.json`
- Checkpoint：`outputs/checkpoints/recovery_state_machine_continue_round2_actor_final.pt`
- Logs：`outputs/logs/recovery_state_machine_continue_round2/online_training.jsonl`
- Videos：`outputs/videos/recovery_state_machine_continue_round2/`

### 最新五阶段复跑（`1/10`）

- Summary：`outputs/recovery_state_machine_restored_latest_summary.json`
- Resolved config：`outputs/recovery_state_machine_restored_latest_resolved_config.json`
- Checkpoint：`outputs/checkpoints/recovery_state_machine_restored_latest_actor_final.pt`
- Logs：`outputs/logs/recovery_state_machine_restored_latest/online_training.jsonl`
- Videos：`outputs/videos/recovery_state_machine_restored_latest/`

归档注意：按最后一次实验要求，`outputs/three_goal_grasp_reset_v1/videos/` 中的 10 个视频
已被最新五阶段复跑视频覆盖；该目录内原有 `summary.json` 和 checkpoint 仍属于三目标
抓握重置实验（`0/10`），与当前视频不一致。需要做统计或复现时，应使用上面列出的
五阶段权威目录，不要用这个镜像目录反推实验配置。

## 已知限制

- latent 欧氏/MSE 距离不是任务语义进度的单调函数，不能独立承担长期 reward。
- LeWM 的可靠范围主要是训练覆盖的短期动作链；较长 rollout 会增加模型偏差。
- 视觉语义网络可能提前切换、跳过状态或误判 complete，且错误会改变后续阶段目标。
- 抓取要求厘米级对齐和准确闭爪时机，Actor 的动作先验及 96 条局部采样候选可能覆盖不到有效动作。
- 在线 Actor 更新来自筛选后的自身轨迹，早期误判可能造成数据偏置和策略漂移。
- Critic 提升了当前小样本结果，但其价值估计仍受教师数据分布和视觉判定误差限制。
- 当前成功率来自单一任务、少量回合和固定初始/目标位置，尚未证明跨目标泛化。
- 同一检查点复跑结果差异较大；正式比较应增加随机种子和回合数，并报告均值、方差及置信区间。

## 后续建议

若继续研究，建议从历史最佳五阶段 checkpoint 出发，先冻结 Actor 做多随机种子基线，
再分别验证视觉谓词、LeWM rollout 和 MPC 候选覆盖率。优先提升对齐/抓握语义监督与
候选动作多样性，再考虑增加在线更新强度；否则更强的在线学习容易放大错误反馈。
