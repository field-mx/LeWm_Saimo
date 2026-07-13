# LeWM 训练流程说明

这份文档用于汇报当前 `LeWm_Saimo` 项目的训练流程，重点说明两件事：

1. 如何配置和启动训练。
2. 训练时数据如何从 HDF5 数据集流到模型、损失函数和 checkpoint。

## 1. 训练目标概览

当前代码训练的是 LeWorldModel / LeWM，一个基于 JEPA 的 latent world model。模型从像素序列和动作序列中学习：

- 把每一帧图像编码成 latent embedding。
- 根据历史 latent embedding 和动作 embedding 预测下一步 latent embedding。
- 用 SIGReg 正则约束 latent 分布，避免表示坍塌。

训练入口是：

```bash
python train.py
```

主训练配置在：

```text
config/train/lewm.yaml
```

默认配置会加载：

```yaml
defaults:
  - launcher: local
  - data: pusht
  - model: lewm
```

也就是说，默认训练任务是 `pusht` 数据，模型结构使用 `config/train/model/lewm.yaml`，运行方式使用本地 launcher。

## 2. 环境与数据准备

### 2.1 环境

README 中的基础安装方式是：

```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

当前机器上已有训练环境时，可以直接进入项目并激活环境：

```bash
cd /home/muxiang/work/LeWm_Saimo
source /publicworkspace/envs/le-wm-py310/bin/activate
```

### 2.2 数据

数据使用 HDF5 格式。以当前 `pusht` 配置为例：

```yaml
dataset:
  name: /home/muxiang/work/LeWm_Saimo/data/pusht_expert_train.h5
  num_steps: ${eval:'${num_preds} + ${history_size}'}
  frameskip: 5
  keys_to_load:
    - pixels
    - action
    - proprio
    - state
  keys_to_cache:
    - action
    - proprio
    - state
```

关键点：

- `name` 指向训练 HDF5 文件。
- `num_steps = num_preds + history_size`。当前默认 `num_preds=1`、`history_size=3`，因此每个样本会取 4 个时间步。
- `frameskip=5`，动作输入维度会按 `frameskip * action_dim` 设置。
- `pixels` 和 `action` 是当前训练 forward 真正用到的输入。
- `proprio`、`state` 会被加载并做归一化，但当前 `JEPA.encode()` 和训练 loss 不直接使用它们。

如果要切换数据集，可以通过 Hydra 覆盖：

```bash
python train.py data=tworoom
python train.py data=dmc
python train.py data=ogb
python train.py data=pusht
```

对应配置位于：

```text
config/train/data/
```

## 3. 训练配置

主配置 `config/train/lewm.yaml` 中比较重要的训练参数如下：

```yaml
output_model_name: lewm
train_split: 0.9
seed: 3072
img_size: 224
embed_dim: 192
history_size: 3
num_preds: 1

trainer:
  max_epochs: 2
  devices: 8
  accelerator: gpu
  precision: bf16
  gradient_clip_val: 1.0

loader:
  batch_size: 64
  num_workers: 6
  persistent_workers: True
  prefetch_factor: 3
  pin_memory: True

optimizer:
  type: AdamW
  lr: 5e-5
  weight_decay: 1e-3

loss:
  sigreg:
    weight: 0.09
```

含义：

- `history_size=3`：使用 3 个历史 latent 作为上下文。
- `num_preds=1`：预测向后偏移 1 步的 latent。
- `img_size=224`：图像输入统一 resize 到 224。
- `embed_dim=192`：图像 embedding、动作 embedding、预测 embedding 都映射到 192 维。
- `trainer.devices=8`：默认使用 8 张 GPU。
- `precision=bf16`：使用 BF16 混合精度。
- `loss.sigreg.weight=0.09`：SIGReg 正则项权重。

## 4. 启动训练

### 4.1 单机 8 卡训练

当前 `train.py` 中记录的 8 卡启动方式如下：

```bash
cd /home/muxiang/work/LeWm_Saimo
source /publicworkspace/envs/le-wm-py310/bin/activate

NCCL_DEBUG=INFO NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python train.py output_model_name=pusht/lewm_8gpu
```

这里的参数含义：

- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`：指定使用 8 张 GPU。
- `NCCL_P2P_DISABLE=1`、`NCCL_IB_DISABLE=1`：关闭 NCCL 的 P2P / IB 路径，常用于规避某些机器上的多卡通信问题。
- `output_model_name=pusht/lewm_8gpu`：指定保存模型时的 run name。

如果只想先做快速验证，可以覆盖 epoch、设备数和 batch size：

```bash
CUDA_VISIBLE_DEVICES=0 \
python train.py trainer.devices=1 trainer.max_epochs=1 loader.batch_size=16 output_model_name=debug/lewm
```

## 5. 模型结构

模型配置在 `config/train/model/lewm.yaml`，核心组件如下：

```yaml
encoder:
  _target_: stable_pretraining.backbone.utils.vit_hf
  size: tiny
  patch_size: 14
  image_size: ${img_size}
  pretrained: false

predictor:
  _target_: module.ARPredictor
  num_frames: ${history_size}
  input_dim: ${embed_dim}
  hidden_dim: ${embed_dim}
  output_dim: ${embed_dim}
  depth: 6
  heads: 16

action_encoder:
  _target_: module.Embedder
  input_dim: ???
  emb_dim: ${embed_dim}

projector:
  _target_: module.MLP

pred_proj:
  _target_: module.MLP
```

模块职责：

- `encoder`：ViT tiny，从图像中提取 CLS token embedding。
- `projector`：把 ViT 输出再映射到训练使用的 latent 空间。
- `action_encoder`：把原始动作序列编码成动作 embedding。
- `predictor`：自回归 Transformer，根据历史 latent 和动作 embedding 预测未来 latent。
- `pred_proj`：对 predictor 输出再做一次投影，得到最终预测 embedding。

注意：`action_encoder.input_dim` 在 `train.py` 中动态设置：

```python
cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")
```

因此动作输入维度取决于数据集中单步动作维度和 `frameskip`。

## 6. 训练时的数据流

整体数据流可以概括为：

```text
HDF5 数据集
  -> stable_worldmodel.load_dataset()
  -> 图像预处理 + 非图像列 z-score 归一化
  -> train/val split
  -> PyTorch DataLoader
  -> Lightning / stable_pretraining DataModule
  -> JEPA.encode()
  -> JEPA.predict()
  -> pred_loss + SIGReg loss
  -> AdamW 反向传播
  -> checkpoint / weights 保存
```

### 6.1 数据加载

`train.py` 首先读取数据配置：

```python
dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
dataset_name = dataset_cfg.pop("name")
dataset = swm.data.load_dataset(
    dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
)
```

对于 `pusht`，每个 batch 主要包含：

```text
batch["pixels"]  # 图像序列
batch["action"]  # 动作序列
batch["proprio"] # 已加载和归一化，但当前 forward 不使用
batch["state"]   # 已加载和归一化，但当前 forward 不使用
```

默认 `history_size=3`、`num_preds=1`，所以时间长度 `T=4`。

### 6.2 预处理

图像预处理来自 `utils.get_img_preprocessor()`：

```python
transforms = [
    get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
]
```

它会对 `pixels` 做：

- `ToImage`：转为模型需要的图像张量，并使用 ImageNet 统计量。
- `Resize(224)`：统一缩放到 `img_size`。

对非图像列，例如 `action`、`proprio`、`state`，代码会根据整个数据列计算 mean/std，然后做 z-score：

```python
normalizer = get_column_normalizer(dataset, col, col)
transforms.append(normalizer)
```

归一化时会过滤 NaN 行，训练 forward 中还会对 action 再做一次保护：

```python
batch["action"] = torch.nan_to_num(batch["action"], 0.0)
```

### 6.3 划分 train / val

训练集和验证集按 `train_split=0.9` 随机划分，随机种子是 `seed=3072`：

```python
rnd_gen = torch.Generator().manual_seed(cfg.seed)
train_set, val_set = spt.data.random_split(
    dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
)
```

之后分别构建 DataLoader：

```python
train = DataLoader(train_set, shuffle=True, drop_last=True)
val = DataLoader(val_set, shuffle=False, drop_last=False)
```

## 7. 模型内部的数据流

### 7.1 图像编码

进入 `JEPA.encode()` 后，`pixels` 会先从序列 batch 拉平成普通图像 batch：

```python
pixels = info["pixels"].float()
b = pixels.size(0)
pixels = rearrange(pixels, "b t ... -> (b t) ...")
```

假设原始图像序列形状为：

```text
pixels: (B, T, C, H, W)
```

拉平后变成：

```text
pixels: (B*T, C, H, W)
```

然后送入 ViT encoder：

```python
output = self.encoder(pixels, interpolate_pos_encoding=True)
pixels_emb = output.last_hidden_state[:, 0]
```

这里使用 ViT 的 CLS token 作为每一帧图像的表示：

```text
pixels_emb: (B*T, D)
```

再经过 projector：

```python
emb = self.projector(pixels_emb)
info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)
```

最后得到：

```text
emb: (B, T, D)
```

当前默认 `D=192`。

### 7.2 动作编码

如果 batch 中有 `action`，会进入 `action_encoder`：

```python
info["act_emb"] = self.action_encoder(info["action"])
```

`Embedder` 内部先用 `Conv1d(kernel_size=1)` 做通道映射，再用 MLP 映射到 embedding 维度：

```text
action:  (B, T, action_input_dim)
act_emb: (B, T, D)
```

其中：

```text
action_input_dim = frameskip * dataset.get_dim("action")
D = embed_dim = 192
```

### 7.3 构造上下文和预测目标

训练 forward 在 `lejepa_forward()` 中定义：

```python
ctx_len = cfg.history_size
n_preds = cfg.num_preds

ctx_emb = emb[:, :ctx_len]
ctx_act = act_emb[:, :ctx_len]

tgt_emb = emb[:, n_preds:]
pred_emb = self.model.predict(ctx_emb, ctx_act)
```

默认配置下：

```text
history_size = 3
num_preds = 1
T = 4
```

因此：

```text
emb:      [z0, z1, z2, z3]
ctx_emb:  [z0, z1, z2]
tgt_emb:  [z1, z2, z3]
```

模型学习的是：

```text
给定历史 latent 和对应动作，预测下一步 latent。
```

也就是让：

```text
pred_emb[:, 0] 接近 z1
pred_emb[:, 1] 接近 z2
pred_emb[:, 2] 接近 z3
```

### 7.4 Predictor 预测

`JEPA.predict()` 会调用 `ARPredictor`：

```python
preds = self.predictor(emb, act_emb)
preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
```

`ARPredictor` 的逻辑是：

1. 给 latent 加时间位置编码。
2. 用 dropout。
3. 送入带动作条件的 Transformer。
4. 输出每个时间步对应的下一步 latent 预测。

动作条件通过 `ConditionalBlock` 中的 AdaLN-zero modulation 注入 Transformer block：

```text
latent embedding x: (B, T, D)
action embedding c: (B, T, D)
          |
          v
Conditional Transformer
          |
          v
predicted embedding: (B, T, D)
```

## 8. Loss 计算

训练总 loss 有两项：

```python
output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]
```

### 8.1 Prediction loss

```text
pred_loss = MSE(pred_emb, tgt_emb)
```

作用：让模型预测的下一步 latent 接近真实下一帧图像编码出来的 latent。

### 8.2 SIGReg loss

```text
sigreg_loss = SIGReg(emb)
```

作用：约束 latent embedding 的整体分布接近各向同性 Gaussian，降低表示坍塌风险。

### 8.3 总 loss

当前默认：

```text
loss = pred_loss + 0.09 * sigreg_loss
```

训练时会把 loss 写入日志：

```python
self.log_dict(losses_dict, on_step=True, sync_dist=True)
```

多卡训练时 `sync_dist=True` 会同步各卡日志。

## 9. 优化器与训练管理

优化器配置如下：

```python
optimizers = {
    "model_opt": {
        "modules": "model",
        "optimizer": dict(cfg.optimizer),
        "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
        "interval": "epoch",
    },
}
```

当前使用：

```text
AdamW, lr=5e-5, weight_decay=1e-3
```

训练由 Lightning Trainer 和 stable-pretraining Manager 管理：

```python
trainer = pl.Trainer(...)
manager = spt.Manager(
    trainer=trainer,
    module=world_model,
    data=data_module,
    ckpt_path=ckpt_path if ckpt_path.exists() else None,
)
manager()
```

如果指定位置已有 checkpoint，Manager 会从该 checkpoint 恢复训练。

## 10. Checkpoint 与输出

训练开始前会创建 run 目录并保存配置：

```python
run_id = cfg.get("subdir") or ""
run_dir = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), run_id)
OmegaConf.save(cfg, run_dir / "config.yaml")
```

模型权重保存由 `SaveCkptCallback` 完成：

```python
object_dump_callback = SaveCkptCallback(
    run_name=cfg.output_model_name,
    cfg=cfg.model,
    epoch_interval=1,
)
```

每个 epoch 结束时，global rank 0 会调用：

```python
save_pretrained(
    model,
    run_name=self.run_name,
    config=self.cfg,
    filename=f"weights_epoch_{epoch}.pt",
)
```

因此 `output_model_name` 会影响最终保存路径。例如：

```bash
python train.py output_model_name=pusht/lewm_8gpu
```

会按 stable-worldmodel 的 cache 规则保存到 `pusht/lewm_8gpu` 对应目录下。

## 11. 汇报时可以使用的一句话总结

这套训练流程本质上是：从 HDF5 读取像素和动作序列，对图像做 ImageNet 风格预处理、对动作等低维状态做 z-score 归一化；然后用 ViT 把每一帧图像编码成 latent，用 action encoder 把动作编码成同维度 embedding；再用带动作条件的自回归 Transformer 根据历史 latent 预测下一步 latent；训练 loss 由下一步 latent 的 MSE 预测误差和防止 latent 坍塌的 SIGReg 正则组成，最后通过 Lightning 多卡训练并按 epoch 保存权重。

## 12. 训练数据流图

```text
                    HDF5 dataset
                         |
                         v
              swm.data.load_dataset()
                         |
                         v
      +---------------- transforms ----------------+
      |                                            |
      v                                            v
pixels: ToImage + Resize + ImageNet stats   action/proprio/state: z-score
      |                                            |
      +--------------------+-----------------------+
                           |
                           v
                  train / val split
                           |
                           v
                    PyTorch DataLoader
                           |
                           v
                 batch["pixels"], batch["action"]
                           |
              +------------+------------+
              |                         |
              v                         v
      ViT encoder + projector       action encoder
              |                         |
              v                         v
        emb: (B,T,D)              act_emb: (B,T,D)
              |                         |
              +------------+------------+
                           |
                           v
       ctx_emb = emb[:, :history_size]
       ctx_act = act_emb[:, :history_size]
       tgt_emb = emb[:, num_preds:]
                           |
                           v
             ARPredictor / Conditional Transformer
                           |
                           v
                 pred_emb: (B, history_size, D)
                           |
                           v
          pred_loss = MSE(pred_emb, tgt_emb)
          sigreg_loss = SIGReg(emb)
                           |
                           v
          loss = pred_loss + 0.09 * sigreg_loss
                           |
                           v
                 backward + AdamW update
                           |
                           v
                   epoch checkpoint
```
