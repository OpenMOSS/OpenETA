# TactileACT 使用说明书

本文档说明 TactileACT 的**输出定义**、**模型训练**与**仿真评测**三部分，便于复现与二次开发。

---

## 一、输出定义与部署接口

### 1.1 动作空间（模型输出）

- **维度**：8 维向量，与训练/数据一致。
- **含义**：**绝对关节目标**（absolute target qpos）
  - 前 7 维：机械臂 7 个关节目标位置（弧度）。
  - 第 8 维：夹爪开合目标（单指标量，与采集的 `joint[7]` 一致）。
- **单位**：关节为弧度，夹爪为 [0,1] 或与仿真/机器人约定一致。
- **控制方式**：仿真中每步调用 `task.take_action(action_tensor, action_type='qpos')`，即把 8D 向量作为**目标关节位置**下发给底层控制器，由环境做插值/跟踪。

### 1.2 输入（观测）

策略每步需要的观测来自 `task._get_observations()`，`encode_obs` 会转换为：

| 键 | 形状/来源 | 说明 |
|----|-----------|------|
| `qpos` | (8,) float32 | 当前关节位置，来自 `observation["embodiment"]["joint"][:8]` |
| `cam_high` | (3, 256, 256) | 头部相机 RGB，Resize 256×256 + ImageNet 归一化 |
| `cam_left_tactile` | (3, 256, 256) | 左触觉 pad 图像，Resize 256×256 + 数据集 gelsight mean/std 归一化 |
| `cam_right_tactile` | (3, 256, 256) | 右触觉 pad 图像，同上 |
| `cam_wrist` | (3, 256, 256) | 仅当 `task_settings.json` 中该任务 `camera_type: "all"` 时存在（如 lift_can、insert_tube） |

归一化统计来自训练数据目录下的 `norm_stats.json` 或 checkpoint 目录下的 `dataset_stats.pkl`（推理时自动加载）。

### 1.3 推理频率与时序聚合

- **chunk_size**：模型一次前向输出未来 `chunk_size` 步的动作序列（默认 20）。
- **query_frequency**：每 `query_frequency` 步才重新调用一次模型；默认与 `chunk_size` 相同（即每 20 步更新一次 chunk）。
- **temporal_agg**：若为 `true`，会对当前时刻可用的多段 chunk 做指数加权平均再取当前步动作，使轨迹更平滑。
- 因此**每步只执行一个 8D 动作**，但该动作可能来自“每若干步更新一次”的 chunk 或时序聚合结果。

### 1.4 部署时 checkpoint 与配置

- **Checkpoint 路径**（固定）：  
  `UniVTAC/policy/TactileACT/act-ckpt/{task_name}-{task_config}/`  
  其中须包含：
  - `policy_best.ckpt`：评估用权重；
  - `dataset_stats.pkl` 或可由 `norm_stats.json` 推导的统计（用于 qpos/action/gelsight 归一化）。
- **deploy 配置**：使用 `policy/TactileACT/deploy.yml`，其中 `task_name`、`task_config` 须与训练一致，`state_dim: 8`、`chunk_size`、`temporal_agg` 等与训练一致，否则易出现维度错误或表现异常。

---

## 二、模型训练

### 2.1 数据准备

- **原始数据**：由 UniVTAC 采集得到，位于  
  `UniVTAC/data/{task_name}/{task_config}/hdf5/*.hdf5`。
- **数据处理**（在 `UniVTAC/policy/TactileACT/` 下执行）：

```bash
conda activate uni
cd UniVTAC/policy/TactileACT

bash process_data.sh <task_name> <task_config> <n_episodes>
```

示例：

```bash
bash process_data.sh lift_bottle demo 50
```

- **输出目录**：`data/{task_name}-{task_config}-{n_episodes}/`，例如 `data/lift_bottle-demo-50/`。  
  内含：
  - `episode_0.hdf5`, `episode_1.hdf5`, ...（每条轨迹一个文件）；
  - `norm_stats.json`（qpos/action/触觉图像的均值和方差，训练与推理共用）。

### 2.2 训练命令（无预训练 backbone）

不加载 CLIP/预训练权重时，直接使用随机初始化的 ResNet-18：

```bash
conda activate uni
cd UniVTAC/policy/TactileACT

python imitate_episodes.py \
    --config config.json \
    --save_dir data/<task_name>-<task_config>-<n_episodes> \
    --name <task_name>-<task_config> \
    --gpu <GPU_ID>
```

示例：

```bash
python imitate_episodes.py \
    --config config.json \
    --save_dir data/insert_hole-demo-50 \
    --name insert_hole-demo \
    --gpu 0
```

若使用 `train.sh`，需保证预训练权重路径存在，否则改为上述命令并加上：

```bash
--gelsight_backbone_path none --vision_backbone_path none
```

### 2.3 主要超参数（可与 config.json 或命令行一致）

| 参数 | 建议/常用 | 说明 |
|------|-----------|------|
| `--batch_size` | 32 | 批大小 |
| `--num_epochs` | 6000 | 训练轮数 |
| `--chunk_size` | 20 | 动作 chunk 长度 |
| `--kl_weight` | 10.0 | KL 权重 |
| `--hidden_dim` | 512 | Transformer 隐藏维 |
| `--temporal_agg` | true | 训练时与推理一致，建议开启 |

### 2.4 训练产物

保存在 `act-ckpt/{task_name}-{task_config}/`：

- `policy_best.ckpt`：验证集最优，用于评测与部署；
- `policy_last.ckpt`：最后一轮；
- `dataset_stats.pkl`：归一化统计（推理加载）；
- `args.json`：完整训练参数。

---

## 三、仿真评测

### 3.1 前置条件

- 已完成训练，且 `act-ckpt/{task_name}-{task_config}/policy_best.ckpt` 存在；
- `deploy.yml` 中 `task_name`、`task_config` 与训练一致，`state_dim: 8`、`chunk_size` 等与训练一致。

### 3.2 配置 deploy.yml

在 `UniVTAC/policy/TactileACT/deploy.yml` 中确认或修改：

```yaml
policy_name: TactileACT
task_name: lift_bottle    # 与训练一致
task_config: demo         # 与训练一致

state_dim: 8
chunk_size: 20
temporal_agg: true
backbone: "clip_backbone"

seed: 0
instruction_type: seen
```

其他 DETR/backbone 参数与训练保持一致即可。

### 3.3 运行评测

在 **UniVTAC 项目根目录** 执行：

```bash
conda activate uni
cd UniVTAC

bash eval_policy.sh <task_name> <task_config> TactileACT/deploy <GPU_ID> --max_steps 200
```

示例：

```bash
bash eval_policy.sh lift_bottle demo TactileACT/deploy 0
bash eval_policy.sh lift_bottle demo TactileACT/deploy 0 --max_steps 200
```

- `eval_policy.sh` 会调用 `scripts/eval_policy.py`，加载 `policy/TactileACT/deploy.yml` 和对应 task 的配置；
- 每个 episode 先 `task.reset(seed)`，再循环：取观测 → `policy.eval(task, observation)` → `take_action`，直到成功、提前终止或达到 `task.cfg.step_lim`（默认 250 步）；如指定 `--max_steps`，还会在 `env.step_count >= max_steps` 时强制停止该 episode（默认 200 步，可设为 0 关闭此限制）。

### 3.4 评测结果位置

结果目录：

```
UniVTAC/eval_result/TactileACT/{task_name}/{timestamp}_{commit}/
├── log.log
├── metadata.json              # 该 task 本次评测汇总
├── metadata/
│   ├── seeds.json             # 单进程 expert_check 的 seed 状态
│   └── worker_0.json          # worker 级逐 seed 结果
├── scene/
│   └── worker_0/
└── video/
    └── worker_0/              # 按 video_frequency 保存的评测视频
```

### 3.5 可选：并行评测

若存在 `parallel_eval.sh` 或类似脚本，可按其用法进行多进程评测，例如：

```bash
bash parallel_eval.sh <task_name> <task_config> TactileACT/deploy <GPU_ID> <num_processes> <total_num>
```

具体以仓库内脚本说明为准。

---

## 四、端到端流程速查

```bash
# 1. 环境与数据
conda activate uni
cd UniVTAC/policy/TactileACT
bash process_data.sh lift_bottle demo 50

# 2. 训练
python imitate_episodes.py --config config.json \
    --save_dir data/lift_bottle-demo-50 --name lift_bottle-demo --gpu 0

# 3. 仿真评测（回到 UniVTAC 根目录）
cd ../..
bash eval_policy.sh lift_bottle demo TactileACT/deploy 0
```

---

## 五、任务与相机配置

| 任务名 | camera_type | 说明 |
|--------|-------------|------|
| lift_bottle, insert_hole, insert_HDMI, pull_out_key, put_bottle_in_shelf, grasp_classify | head | 仅头部相机 |
| lift_can, insert_tube | all | 头部 + 腕部相机 |

相机类型由 `policy/task_settings.json` 决定，数据处理与推理会自动对齐。

---

## 六、常见问题

1. **推理时找不到归一化统计**  
   确保 `act-ckpt/{task_name}-{task_config}/` 下有 `dataset_stats.pkl`（训练时会生成），或 `data/{task_name}-*/norm_stats.json` 存在且包含 qpos/action/left_tac/right_tac 的 mean/std。

2. **state_dim 报错或维度不匹配**  
   deploy 配置中必须设置 `state_dim: 8`，且与训练时一致；若未设置，部分代码会从 checkpoint 的 `args.json` 或 `dataset_stats.pkl` 推断。

3. **前几十步机械臂几乎不动**  
   TactileACT 输出为绝对 qpos，且带 temporal aggregation，前期可能输出接近当前 qpos，导致位移很小；属策略特性，可尝试调整 chunk_size / temporal_agg 或增加数据多样性。

4. **CUDA tensor 转 numpy 报错**  
   观测中的 `observation["embodiment"]["joint"]` 可能是 CUDA tensor，需先 `.cpu().numpy()` 再取 `[:8]`；若使用已修复的 `encode_obs`，应已处理该情况。

以上为 TactileACT 的**输出定义**、**训练**与**仿真评测**使用说明，便于按流程复现与排查问题。
