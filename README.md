
# LeWorldModel
### Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

> **本地实验归档（2026-08-05）**
>
> 本仓库在上游 LeWM 代码之上增加了 PushT 长时序验证、Cube 机械臂规划、
> Actor/LeWM MPC、视觉语义状态机和在线 Actor 更新等实验。当前主要归档实验为
> [`latent_three_phase_mpc`](experiments/cube_robot/experiments/latent_three_phase_mpc/README.md)，
> 其中记录了可复现命令、配置、检查点、视频、成功率和已知限制。下方先给出
> 本地实验演示与总结，随后英文内容保留为上游项目说明。

## 本地 Cube 机械臂实验演示

### 成功回合视频

下面的视频来自五阶段恢复状态机历史最佳实验的第 8 回合。画面由 MuJoCo
真实仿真环境渲染，LeWM 仅用于预测候选动作链的 latent 演化并辅助 MPC 评分。

<video controls width="800">
  <source src="experiments/cube_robot/experiments/latent_three_phase_mpc/outputs/videos/recovery_state_machine_continue_round2/recovery_online_episode_8_success.mp4" type="video/mp4">
</video>

若当前 Markdown 渲染器不支持内嵌视频，可直接
[播放或下载成功回合 MP4](experiments/cube_robot/experiments/latent_three_phase_mpc/outputs/videos/recovery_state_machine_continue_round2/recovery_online_episode_8_success.mp4)。

### 成功回合 Latent 诊断

![五阶段恢复状态机第 8 回合 latent 分析](experiments/cube_robot/experiments/latent_three_phase_mpc/outputs/analysis/recovery_state_machine_continue_round2_latent_analysis/episode_08.png)

该图用于对照每个阶段的真实 RGB latent、阶段目标 latent，以及 LeWM 预测 latent
与执行后真实 latent 的误差，帮助区分规划器选错动作和世界模型预测偏差。

### 长时序 Latent 非单调现象

![专家轨迹逐帧 latent 与最终目标 latent 的 MSE](expert_latent_goal_mse.png)

横轴是专家轨迹步长，纵轴是每帧 RGB 经冻结 LeWM encoder 编码后，与最终目标帧
latent 的 MSE。多条专家轨迹及平均曲线都不是均匀单调下降：专家正在执行正确动作时，
latent 距离也可能暂时增大。因此，`latent MSE 更小` 不能直接等价为
`动作更接近长期任务成功`。

## 工作总结

### 完成的工作

1. 复现并分析了 LeWM 在 PushT 和 Cube 机械臂任务中的短期规划流程，明确了
   `action_block`、规划 horizon、CEM/MPC 候选采样、真实动作执行和闭环重规划之间的关系。
2. 构建了 Cube RGB 闭环实验：在线推理只读取当前 RGB 和目标 RGB，经冻结 LeWM
   encoder 得到 latent，不把仿真器精确坐标输入规划器。
3. 实现并比较了原始三阶段、去 Critic、五阶段恢复状态机、三目标状态机和抓握重置等方案，
   保存了配置、检查点、逐步 JSONL 日志、LeWM 耗时、分析图和 MuJoCo 视频。
4. 实现了 Actor 在线适应：MPC 选择的优质真实转移进入 replay，再使用行为克隆锚定
   蒸馏 Actor，避免直接通过不可靠的长时序 latent loss 更新策略。
5. 历史最佳五阶段在线实验完成 `3/10`，成功回合为 3、5、8；同源检查点复跑为
   `1/10`，说明方法已能完成完整任务，但随机性和鲁棒性仍需改进。

### 发现的问题

**1. LeWM 的可靠视野主要是短期动作链。**

当前模型按约 25 个原始动作的窗口训练和使用。把遥远最终目标直接交给规划器时，
LeWM 可以持续做短期预测，但短期 latent 最优并不保证长期任务最优。

**2. Latent 空间的任务精细度有待商榷。**

LeWM latent 首先服务于视觉动力学预测，并未被显式训练为机器人任务的单调距离函数。
背景、机械臂姿态、夹爪状态和物块位置共同影响 latent；两个 latent 很接近，不一定代表
物块已精确对齐或夹爪已经可靠抓住。

**3. 长步骤任务中的目标 latent 误差并非单调下降。**

上图显示，即使是成功专家轨迹，当前 latent 到最终目标 latent 的 MSE 也会反复升降。
若 CEM/MPC 只最小化该误差，就可能拒绝“短期变远、长期必要”的动作，例如先抬升、
绕行、重新对齐或暂时离开最终位姿。

**4. 规划器容易陷入局部最优。**

CEM/MPC 在 Actor 基准动作链附近有限采样。当候选动作的长期收益无法由最终 latent
MSE 正确表达时，候选得分趋同，规划器容易选择停滞、重复开合夹爪或接近目标后再次离开。
增加采样数量和迭代次数只能扩大局部搜索，不能修复评价目标本身。

### 解决方案

本项目没有重新训练 LeWM，而是把它限制在更可靠的短期预测职责中，并在其外部增加
任务语义和长期价值：

```text
最终长任务
   -> 可恢复五阶段状态机：align -> grasp -> transfer -> release -> retreat
   -> 每阶段生成约 25 步范围内的短期目标 latent
   -> Actor 提供动作链先验，MPC 采样候选动作链
   -> 冻结 LeWM 只预测候选动作链的短期 latent 链
   -> 阶段目标误差 + Critic 长期价值 + 不确定性 + 动作先验共同评分
   -> 仅执行第一个真实动作，读取新 RGB 后重新判定阶段并规划
   -> 失败时允许状态回退和重新尝试
```

这一设计不再要求 `当前 latent 到最终 latent 的距离必须单调下降`，从结构上解决了
单一长期 latent MSE 将规划器引入局部最优的问题：

- 阶段目标负责 25 步范围内的可达性；
- Critic 补充动作对完整任务的长期价值；
- 语义谓词负责对齐、抓握、转移、释放和完成判定；
- 闭环 MPC 每执行一步就使用真实 RGB 修正世界模型误差；
- 恢复状态机允许抓握丢失或阶段误判后回退重试。

因此，该方案能够在**不重新训练 LeWM**的条件下，把原生短视野世界模型用于完整
长时序机械臂任务。它解决了长任务规划信号的组织方式，但当前小样本成功率仍表明
视觉精细判定、候选动作覆盖和在线更新稳定性需要继续提升。

完整配置、实验表、权威产物和复现限制见
[RGB-Latent MPC Cube 实验归档](experiments/cube_robot/experiments/latent_three_phase_mpc/README.md)。

[Lucas Maes*](https://x.com/lucasmaes_), [Quentin Le Lidec*](https://quentinll.github.io/), [Damien Scieur](https://scholar.google.com/citations?user=hNscQzgAAAAJ&hl=fr), [Yann LeCun](https://yann.lecun.com/) and [Randall Balestriero](https://randallbalestriero.github.io/)

**Abstract:** Joint Embedding Predictive Architectures (JEPAs) offer a compelling framework for learning world models in compact latent spaces, yet existing methods remain fragile, relying on complex multi-term losses, exponential moving averages, pretrained encoders, or auxiliary supervision to avoid representation collapse. In this work, we introduce LeWorldModel (LeWM), the first JEPA that trains stably end-to-end from raw pixels using only two loss terms: a next-embedding prediction loss and a regularizer enforcing Gaussian-distributed latent embeddings. This reduces tunable loss hyperparameters from six to one compared to the only existing end-to-end alternative. With ~15M parameters trainable on a single GPU in a few hours, LeWM plans up to 48× faster than foundation-model-based world models while remaining competitive across diverse 2D and 3D control tasks. Beyond control, we show that LeWM's latent space encodes meaningful physical structure through probing of physical quantities. Surprise evaluation confirms that the model reliably detects physically implausible events.

<p align="center">
   <b>[ <a href="https://arxiv.org/pdf/2603.19312v1">Paper</a> | <a href="https://huggingface.co/collections/quentinll/lewm">Checkpoints &amp; Data</a> | <a href="https://le-wm.github.io/">Website</a> ]</b>
</p>

<br>

<p align="center">
  <img src="assets/lewm.gif" width="80%">
</p>

If you find this code useful, please reference it in your paper:
```
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

## Using the code
This codebase builds on [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) for environment management, planning, and evaluation, and [stable-pretraining](https://github.com/galilai-group/stable-pretraining) for training. Together they reduce this repository to its core contribution: the model architecture and training objective.

**Installation:**
```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

## Data

Datasets use the HDF5 format for fast loading. Download the data from [HuggingFace](https://huggingface.co/collections/quentinll/lewm) and decompress with:

```bash
tar --zstd -xvf archive.tar.zst
```

Place the extracted `.h5` files under `$STABLEWM_HOME` (defaults to `~/.stable-wm/`). You can override this path:
```bash
export STABLEWM_HOME=/path/to/your/storage
```

Dataset names are specified without the `.h5` extension. For example, `config/train/data/pusht.yaml` references `pusht_expert_train`, which resolves to `$STABLEWM_HOME/pusht_expert_train.h5`.

## Training

`jepa.py` contains the PyTorch implementation of LeWM. Training is configured via [Hydra](https://hydra.cc/) config files under `config/train/`.

Before training, set your WandB `entity` and `project` in `config/train/lewm.yaml`:
```yaml
wandb:
  config:
    entity: your_entity
    project: your_project
```

To launch training:
```bash
python train.py data=pusht
```

Checkpoints are saved to `$STABLEWM_HOME` upon completion.

For baseline scripts, see the stable-worldmodel [scripts](https://github.com/galilai-group/stable-worldmodel/tree/main/scripts/train) folder.

## Planning

Evaluation configs live under `config/eval/`. Set the `policy` field to the checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix:

```bash
# ✓ correct
python eval.py --config-name=pusht.yaml policy=pusht/lewm

# ✗ incorrect
python eval.py --config-name=pusht.yaml policy=pusht/lewm_object.ckpt
```

## Pretrained Checkpoints

Pretrained LeWM checkpoints for each environment are mirrored on the Hugging Face
Hub (model repos), alongside the datasets (dataset repos) in the same collection:

- [`quentinll/lewm-pusht`](https://huggingface.co/quentinll/lewm-pusht)
- [`quentinll/lewm-cube`](https://huggingface.co/quentinll/lewm-cube)
- [`quentinll/lewm-tworooms`](https://huggingface.co/quentinll/lewm-tworooms)
- [`quentinll/lewm-reacher`](https://huggingface.co/quentinll/lewm-reacher)

The full baseline checkpoint suite (PLDM, LeJEPA, IVL, IQL, GCBC, DINO-WM, DINO-WM-noprop)
is available on [Google Drive](https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e):

<div align="center">

| Method | two-room | pusht | cube | reacher |
|:---:|:---:|:---:|:---:|:---:|
| pldm | ✓ | ✓ | ✓ | ✓ |
| lejepa | ✓ | ✓ | ✓ | ✓ |
| ivl | ✓ | ✓ | ✓ | — |
| iql | ✓ | ✓ | ✓ | — |
| gcbc | ✓ | ✓ | ✓ | — |
| dinowm | ✓ | ✓ | — | — |
| dinowm_noprop | ✓ | ✓ | ✓ | ✓ |

</div>

## Loading a checkpoint

### From the Drive archive

Each tar archive contains two files per checkpoint:
- `<name>_object.ckpt` — a serialized Python object for convenient loading; this is what `eval.py` and the `stable_worldmodel` API use
- `<name>_weight.ckpt` — a weights-only checkpoint (`state_dict`) for cases where you want to load weights into your own model instance

Place the extracted files under `$STABLEWM_HOME/` and load via:

```python
import stable_worldmodel as swm

# Load the cost model (for MPC)
cost = swm.policy.AutoCostModel('pusht/lewm')
```

`AutoCostModel` accepts:
- `run_name` — checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix
- `cache_dir` — optional override for the checkpoint root (defaults to `$STABLEWM_HOME`)

The returned module is in `eval` mode with its PyTorch weights accessible via `.state_dict()`.

### From the Hugging Face mirror

The HF model repos ship the LeWM checkpoint as a `weights.pt` (state dict) plus a
`config.json` describing the model. Convert once to produce the `_object.ckpt`
that `eval.py` expects:

```bash
# download weights.pt + config.json
hf download quentinll/lewm-pusht --local-dir $STABLEWM_HOME/hf_pusht

# convert to object checkpoint under $STABLEWM_HOME/pusht/lewm_object.ckpt
python - <<'PY'
import json, torch, stable_pretraining as spt
from pathlib import Path
from jepa import JEPA
from module import ARPredictor, Embedder, MLP
import stable_worldmodel as swm

src = Path(swm.data.utils.get_cache_dir(), "hf_pusht")
out = Path(swm.data.utils.get_cache_dir(), "pusht", "lewm_object.ckpt")

cfg = json.loads((src / "config.json").read_text())
encoder = spt.backbone.utils.vit_hf(
    cfg["encoder"]["size"],
    patch_size=cfg["encoder"]["patch_size"],
    image_size=cfg["encoder"]["image_size"],
    pretrained=False, use_mask_token=False,
)
mlp = lambda k: MLP(input_dim=cfg[k]["input_dim"], output_dim=cfg[k]["output_dim"],
                    hidden_dim=cfg[k]["hidden_dim"], norm_fn=torch.nn.BatchNorm1d)
model = JEPA(
    encoder=encoder,
    predictor=ARPredictor(**cfg["predictor"]),
    action_encoder=Embedder(**cfg["action_encoder"]),
    projector=mlp("projector"),
    pred_proj=mlp("pred_proj"),
)
sd = torch.load(src / "weights.pt", map_location="cpu", weights_only=False)
model.load_state_dict(sd, strict=True)
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(model, out)
PY
```

After conversion, load via `swm.policy.AutoCostModel('pusht/lewm')` as usual.

## Contact & Contributions
Feel free to open [issues](https://github.com/lucas-maes/le-wm/issues)! For questions or collaborations, please contact `lucas.maes@mila.quebec`
