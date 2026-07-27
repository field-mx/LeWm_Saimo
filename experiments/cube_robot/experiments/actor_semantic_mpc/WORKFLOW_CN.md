# Actor + Semantic MPC + LeWM 完整实验流程

## 1. 实验目标

该实验在 OGBench `cube-single-play-v0` 仿真环境中完成单方块抓取与放置任务。
系统由四个部分组成：

1. **BC Actor**：根据当前机器人状态和最终目标状态输出一个基础动作。
2. **语义阶段控制器**：根据末端、方块和目标之间的几何关系，动态生成结构化动作链。
3. **LeWM 世界模型**：预测候选动作链的最终 latent，并计算它与最终目标 latent 的代价。
4. **MPC 闭环**：每轮只执行候选链的第一个动作，然后读取真实状态并重新规划。

这里使用的是 Actor 引导的随机射击 MPC，不是标准 CEM。系统没有在每轮规划中
更新高斯分布参数，而是一次生成 128 条候选链，由 LeWM 评分后选择一条。

## 2. 模型和数据

### 2.1 Actor

Actor 由行为克隆预训练，其输入维度为 33：

```text
当前状态 state:       [28]
最终目标特征 goal:     [5]
Actor 输入 observation:[33]
```

目标特征由目标方块的三维位置和朝向组成：

```text
goal = [goal_x, goal_y, goal_z, cos(goal_yaw), sin(goal_yaw)]
```

Actor 网络结构为：

```text
[33]
  -> Linear(33, 256)
  -> LayerNorm(256)
  -> Tanh
  -> Linear(256, 256)
  -> Tanh
  -> Linear(256, 5)
  -> Tanh
  -> action [5]
```

五维动作表示：

```text
[delta_x, delta_y, delta_z, delta_yaw, gripper]
```

动作被限制在 `[-1, 1]`。代码中的单步动作尺度为：

```text
[0.05 m, 0.05 m, 0.05 m, 0.30 rad, 1.0]
```

### 2.2 LeWM

LeWM 使用当前图像、最终目标图像和候选动作链，预测候选链执行后的 latent。
规划代价可表示为：

\[
J(U_i)=D\left(\hat z_{T}^{(i)},z_g\right)
\]

其中：

- \(U_i\) 是第 \(i\) 条候选动作链；
- \(\hat z_T^{(i)}\) 是 LeWM 预测的最终 latent；
- \(z_g\) 是最终目标图像经过编码器得到的目标 latent；
- \(D\) 是模型内部使用的 latent 距离。

中间阶段目标不会替换 `goal_pixels`。LeWM 从始至终都按照最终目标图像评分，
阶段控制器只负责生成具有合理动作顺序的候选链。

## 3. 语义状态提取

每个真实动作执行后，系统从28维仿真状态中提取：

```text
末端位置:       state[12:15]
末端朝向:       state[15:17]
夹爪状态:       state[17]
接触状态:       state[18]
方块位置:       state[19:22]
方块朝向:       state[26:28]
目标方块位置:   goal_state[19:22]
```

这些量被转换为实际米制坐标，组成 `SemanticState`。阶段划分依赖真实仿真状态，
因此当前实验属于使用 privileged state 的混合控制实验，不是纯视觉部署方案。

## 4. 中间阶段的动态划分

阶段不是按照固定帧数预先切分，而是每执行一个真实动作后重新判断：

| 阶段 | 进入条件 | 中间目标 |
| --- | --- | --- |
| `approach` | 未接触，末端和方块XY误差大于4 cm | 方块上方16 cm |
| `descend` | XY已对齐，三维距离仍大于2.5 cm | 方块位置 |
| `grasp` | 末端已对准，但尚未形成稳定接触 | 夹紧方块 |
| `lift` | 已接触，方块高度低于14 cm | 原地抬高 |
| `transport` | 已抬高，方块尚未到目标XY位置 | 目标上方14 cm |
| `place` | 方块和目标XY误差不超过4 cm | 最终目标位置 |
| `complete` | 方块和目标三维距离不超过4 cm | 任务结束 |

阶段优先级很重要。目标XY已经对齐时，必须优先进入 `place`。如果先检查运输
高度，方块下降时会重新进入 `lift`，形成 `place -> lift -> place` 振荡。

## 5. 一轮闭环规划的完整流程

### 步骤1：环境初始化

环境重置后返回：

```text
state:         [28]
goal_state:    [28]
current_image: [H, W, 3]
goal_image:    [H, W, 3]
```

目标图像在一个 episode 内保持不变，当前图像每个真实动作后更新。

### 步骤2：Actor生成基础动作

系统从 `goal_state` 取目标方块的XYZ和朝向，与当前28维状态拼接：

```text
actor_input = concat(state[28], goal_features[5]) -> [33]
```

输入经过训练集均值和标准差归一化后送入 Actor：

```text
base_action = Actor(actor_input) -> [5]
```

### 步骤3：判断当前阶段

`detect_phase()` 根据末端位置、方块位置、目标位置、方块高度和接触状态，
判断当前属于 `approach`、`descend`、`grasp`、`lift`、`transport` 或 `place`。

### 步骤4：构造25动作语义模板

当前配置为：

```yaml
horizon_blocks: 5
action_block: 5
```

因此一条候选链包含：

```text
5 blocks * 5 actions/block = 25 actions
```

语义控制器根据当前阶段生成第一个动作，再用轻量级近似状态更新器估计下一状态，
重新判断阶段并生成下一个动作，循环25次。由此得到：

```text
template: [25, 5]
```

模板内部可以自然地包含多个阶段，例如：

```text
approach -> descend -> grasp -> lift
```

### 步骤5：生成128条候选动作链

当前配置生成128条候选：

```text
candidates: [128, 25, 5]
```

- 候选0：把 Actor 的基础动作重复25次，作为基线；
- 候选1：不加噪声的语义模板；
- 候选2至127：语义模板加时间相关高斯噪声。

噪声满足近似的一阶相关过程：

\[
\epsilon_t=\rho\epsilon_{t-1}
+\sqrt{1-\rho^2}\sigma\xi_t
\]

当前 `temporal_correlation=0.65`，使相邻动作平滑相关。阶段控制器还会固定夹爪
方向，避免尚未对准就夹紧，或者搬运过程中意外松开。

### 步骤6：动作归一化和张量重排

候选动作先使用数据集动作均值和标准差归一化：

```text
normalized = (candidates - action_mean) / action_scale
```

然后重排成 LeWM 所需格式：

```text
原始候选:       [128, 25, 5]
按block重排:    [128, 5, 25]
加入batch维度:  [1, 128, 5, 25]
```

最后一个维度 `25` 表示一个 block 中5个五维动作展平后的结果。

图像经过 ImageNet 归一化和224像素缩放：

```text
current_pixels: [1, 1, 3, 224, 224]
goal_pixels:    [1, 1, 3, 224, 224]
```

为128条候选扩展后：

```text
current_pixels: [1, 128, 1, 3, 224, 224]
goal_pixels:    [1, 128, 1, 3, 224, 224]
actions:        [1, 128, 5, 25]
```

### 步骤7：LeWM为候选链评分

`model.get_cost()` 对128条候选并行推演，输出：

```text
costs: [1, 128]
```

系统取代价最小的候选：

\[
i^*=\arg\min_i J(U_i)
\]

同时计算它相对Actor基线的改进：

\[
\eta=\frac{J(U_0)-J(U_{i^*})}{\max(|J(U_0)|,10^{-6})}
\]

只有在最优候选不是基线且 `eta >= 0.05` 时，才接受LeWM选择；否则退回Actor
基线动作链。

### 步骤8：只执行第一个真实动作

虽然LeWM评分的是25动作链，但MPC只执行选中链的第一个动作：

```text
executed_action = candidates[selected_index, 0] -> [5]
```

仿真器返回：

```text
next_state, terminated, truncated, info
```

随后丢弃剩余24个规划动作，读取新的真实状态并重新开始下一轮规划。因此这是
每个真实动作都重新规划的闭环控制。

### 步骤9：真实语义进度检查

训练阶段不会仅凭LeWM分数更新Actor。真实动作执行后还要检查：

1. 是否进入了更靠后的阶段；
2. 当前阶段几何误差是否至少下降0.5 mm；
3. `grasp`阶段是否提高夹爪/接触状态；
4. `lift`和`transport`阶段是否保持接触；
5. 是否发生阶段倒退。

只有同时满足：

```text
planner_accepted == true
real_semantic_accepted == true
selected_index != 0
```

该真实执行动作才进入Actor的验证回放池。

### 步骤10：Actor在线适应

Actor训练目标由真实验证过的动作构成：

\[
L_{\text{verified}}
=\|\pi_\theta(o)-a_{\text{verified}}\|_2^2
\]

为避免Actor偏离原始行为克隆策略，还加入锚定损失：

\[
L=L_{\text{verified}}
+0.25\|\pi_\theta(o_{\text{anchor}})
-\pi_{\text{BC}}(o_{\text{anchor}})\|_2^2
\]

当前正式训练运行12个episode。LeWM保持冻结，更新的只有Actor。

## 6. 训练与验证的区别

### 训练阶段

```text
BC Actor -> 候选动作链 -> LeWM评分 -> 真实执行
         -> 真实语义验证 -> 写入回放池 -> 更新Actor
```

### 验证阶段

```text
冻结Actor -> 候选动作链 -> LeWM评分 -> 真实执行 -> 重新规划
```

验证时：

- Actor不更新；
- LeWM不更新；
- 真实语义状态仍用于阶段判断；
- `verified_training_actions`必须为0。

## 7. 停止机制与视频输出

每个episode在以下任一条件满足时停止：

1. `info["success"] == true`；
2. 环境返回 `terminated == true`；
3. 环境返回 `truncated == true`；
4. 达到配置的 `max_steps`。

OGBench环境本身可能在200步截断，因此即使配置写250步，实际也可能在200步停止。

视频记录初始画面以及每个真实动作后的画面：

```text
视频帧数 = 实际执行动作数 + 1
```

正式阶段版种子42执行64步，因此视频包含65帧；视频帧率为20 FPS。

## 8. 无中间阶段消融版本

无阶段版本设置：

```yaml
planner:
  control_mode: direct_goal
```

该版本保持以下条件不变：

- 同一个已训练Actor；
- 同一个LeWM权重；
- 同一个环境种子42；
- 128条候选；
- 每条候选25个动作；
- 每轮只执行1个真实动作；
- 最终目标图像不变。

唯一变化是移除：

- 中间空间目标；
- 阶段状态机；
- 阶段夹爪约束；
- 语义动作模板。

无阶段候选链直接围绕Actor当前动作采样：

```text
base template: repeat(base_action, 25) -> [25, 5]
candidates: base template + correlated noise -> [128, 25, 5]
```

LeWM仍然按照最终目标latent选择动作链。因此该消融测试的是：仅依靠Actor局部
动作和最终latent代价，能否自行发现“接近、下降、夹取、抬升、搬运、放置”的
完整动作顺序。

## 9. 对比结果

| 指标 | 阶段版本 | 无阶段版本 |
| --- | ---: | ---: |
| 种子 | 42 | 42 |
| 成功 | 是 | 否 |
| 执行动作数 | 64 | 200 |
| LeWM调用数 | 64 | 200 |
| LeWM接受动作数 | 64 | 185 |
| 初始方块目标距离 | 0.43079 m | 0.42895 m |
| 最终方块目标距离 | 0.03465 m | 0.42888 m |
| 最大接触值 | 1.000 | 0.445 |
| 最大方块高度 | 0.15925 m | 0.01996 m |

无阶段版本中LeWM接受了185次候选，说明规划器确实持续工作，但方块几乎没有
向目标移动，也未达到0.50接触阈值。问题不是推理停止，而是最终latent目标对
早期抓取动作提供的搜索信号不足，随机候选很难形成完整且因果一致的动作链。

## 10. 日志说明

逐步日志中的主要字段：

| 字段 | 含义 |
| --- | --- |
| `phase_before/after` | 动作执行前后的语义阶段 |
| `template_first_phases` | 当前25步模板前5步对应阶段 |
| `base_action` | Actor输出动作 |
| `template_action` | 语义模板第一个动作 |
| `executed_action` | 实际执行动作 |
| `base_cost/best_cost` | Actor基线和最优候选的LeWM代价 |
| `planner_accepted` | 是否接受LeWM最优候选 |
| `real_semantic_accepted` | 真实状态是否确认动作有效 |
| `semantic_progress` | 当前阶段的真实进度 |
| `contact_before/after` | 动作前后的接触值 |
| `adaptation_loss` | 当前Actor在线更新损失 |
| `world_model_inference_seconds` | 本轮LeWM推理耗时 |

## 11. 运行命令

阶段版完整训练与验证：

```bash
cd /home/muxiang/work/LeWm_Saimo/experiments/cube_robot/experiments/actor_semantic_mpc
./run_all.sh
```

只运行冻结Actor验证：

```bash
/publicworkspace/envs/le-wm-py310/bin/python run_experiment.py --stage evaluate
```

运行无阶段对照：

```bash
/publicworkspace/envs/le-wm-py310/bin/python run_experiment.py \
  --config config_no_phases.yaml \
  --stage evaluate
```

## 12. 输出文件

阶段版视频：

```text
outputs/videos/semantic_mpc_episode_0.mp4
```

无阶段视频：

```text
outputs_ablation_no_phases/videos/semantic_mpc_episode_0.mp4
```

左右并排视频，左侧为阶段版，右侧为无阶段版：

```text
outputs_ablation_no_phases/videos/phase_vs_no_phases_seed42.mp4
```

实验汇总：

```text
outputs/evaluation.json
outputs_ablation_no_phases/evaluation.json
```

逐步日志：

```text
outputs/logs/evaluation.jsonl
outputs_ablation_no_phases/logs/evaluation.jsonl
```

## 13. 结论

中间阶段的主要价值不是改变LeWM目标，而是向动作搜索空间注入任务结构：

1. 把长时序任务转换为因果顺序明确的动作链；
2. 限制夹爪和末端动作，减少无效候选；
3. 为真实动作验证和Actor更新提供局部进度标准；
4. 配合每步重规划，及时用真实状态修正偏差。

消融结果表明，在当前Actor、LeWM和128条候选规模下，仅使用最终目标latent
不足以可靠发现完整抓取放置序列。当前成功结果应表述为“语义阶段控制器引导的
Actor + LeWM MPC混合系统”，不应表述为纯世界模型独立完成长时序规划。
