# OpenETA Environment Registry Specification

## 1. Motivation

RLinf 提供了 17 种仿真/真机环境的 `gym.Env` 封装，但目前创建环境的唯一途径是：

```
手写 YAML → OmegaConf DictConfig → get_env_cls(env_type) → EnvClass(cfg, num_envs, ...)
```

对 interactive agent 和开放 benchmark 评估来说太重了。我们需要**一条命令即可创建环境**。

---

## 2. Gym ID 格式

```
openeta/<env_type>/[<suite>/]<task_slug>-v0
```

| 字段 | 说明 |
|------|------|
| `openeta` | 固定命名空间前缀，避免与系统内其他 gym 注册冲突 |
| `env_type` | RLinf `SupportedEnvType` 值（小写），如 `maniskill`、`behavior`、`libero` |
| `suite` | **仅多任务环境需要**，标识 task suite / benchmark / scene |
| `task_slug` | URL-safe 的自然语言任务标识符，见 §2.1 |
| `v0` | 版本号 |

### 2.1 task_slug 生成规则

`task_slug` 是**人类可读的自然语言描述，经过 URL-safe 转换**。优先级：

1. 使用各 benchmark 原生提供的短 task name（如 BEHAVIOR 的 `turning_on_radio`）
2. 如无短名，从 task 的自然语言描述生成：取前 6-8 个词 → 小写 → 空格替换为下划线 → 移除特殊字符
3. 确保在同一 `env_type`（+ `suite`）下唯一

```
# BEHAVIOR：task_name 已足够短
"turning_on_radio"        → "Turn on the radio receiver..."

# Libero：从 task.language 生成
"pick_up_the_black_bowl"  → "pick up the black bowl from the table"

# MetaWorld：使用原生 env_name
"assembly"                → "Pick up a nut and screw it onto a peg"
```

### 2.2 示例汇总

```
# ─── 单任务环境 ───
openeta/maniskill/PickCube-v1
openeta/behavior/turning_on_radio-v0
openeta/genesis/cube_pick-v0
openeta/frankasim/pick_cube-v0
openeta/d4rl/halfcheetah-medium-v2
openeta/robotwin/open_drawer-v0
openeta/roboverse/pick_cube-v0
openeta/habitat/vlnce_r2r-v0

# ─── 多任务环境（两级路径）───
openeta/libero/LIVING_ROOM_SCENE1/pick_up_the_black_bowl-v0
openeta/libero/LIVING_ROOM_SCENE1/put_the_black_bowl_on_the_plate-v0
openeta/libero/KITCHEN_SCENE2/open_the_bottom_drawer-v0

openeta/metaworld/MT50/assembly-v0
openeta/metaworld/MT50/pick_place-v0

openeta/calvin/scene_A/rotate_blue_block_right-v0
openeta/calvin/scene_D/push_blue_block_left-v0

openeta/robocasa/BasicPickAndPlace-v0
openeta/robocasa/OpenDoor-v0
```

---

## 3. 复合环境分解

Libero、MetaWorld、Calvin 等环境在 RLinf 中通过 `task_suite` + `reset_state_ids` 在多个 task 之间切换。**一个 `gym.make()` 应该对应一个确定的 task**。

### 3.1 分解策略

每个独立 task 注册为一个 gym env，通过以下机制锁定任务：

| env_type | 锁定机制 | 示例 |
|----------|---------|------|
| `libero` | `cfg.specific_reset_id=task_id` 或 `cfg.task_id_filter=[task_id]` | 单 env 只跑一个 task |
| `metaworld` | `cfg.specific_reset_id` 固定 task 索引 | 同上 |
| `calvin` | `cfg.specific_reset_id` 固定 task 索引 | 同上 |
| `robocasa` | `cfg.task_names=[single_task]` | 注册时只包含一个 task |

### 3.2 `num_envs` 参数

- **默认 `num_envs=1`**：适合 interactive agent / 评估
- **可通过 `gym.make(id, num_envs=N)` 覆盖**：用于并行训练
- 注册的 env 始终是单任务；`num_envs` 只控制同一任务的并行实例数

```python
# 单 env（agent 交互）
env = gym.make("openeta/libero/LIVING_ROOM_SCENE1/pick_up_the_black_bowl-v0")

# 4 个并行实例（训练），跑同一个 task 的不同 trial
env = gym.make("openeta/libero/LIVING_ROOM_SCENE1/pick_up_the_black_bowl-v0", num_envs=4)
```

### 3.3 多任务 env 可选保留

如果需要跑整个 suite，可以额外注册一个不带 task 的 ID：

```python
# 整个 suite（10 个 task 轮转，num_envs=N）
env = gym.make("openeta/libero/LIVING_ROOM_SCENE1_suite-v0", num_envs=10)
```

但**这不是第一期目标**。第一期只注册单任务 env。

---

## 4. 具身（Robot）处理

经过对全部 17 个 env_type 的分析，具身信息与任务的关系分三种模式：

| 模式 | env_type | 说明 |
|------|----------|------|
| **A: robot 嵌入 task** | `maniskill`, `isaaclab`, `genesis`, `polaris` | 换了 task 就换了 robot，无法分开 |
| **B: robot 固定** | `libero`(Franka), `metaworld`(Sawyer), `calvin`(Franka), `frankasim`(Franka), `d4rl`(N/A), `habitat`(N/A) | 整个 benchmark 只有一种 robot |
| **C: robot 可配置** | `behavior`, `robocasa`, `roboverse`, `robotwin` | 同一 task 可以换 robot |

**结论**：robot 不在 ID 中，而是 `gym.make()` 的关键字参数（仅模式 C 有效）：

```python
# 模式 A/B：robot 已确定，无需指定
env = gym.make("openeta/maniskill/PickCube-v1")
env = gym.make("openeta/libero/LIVING_ROOM_SCENE1/pick_up_the_black_bowl-v0")

# 模式 C：robot 可配置
env = gym.make("openeta/behavior/turning_on_radio-v0", robot="franka_panda")
env = gym.make("openeta/roboverse/pick_cube-v0", robot="fetch")
```

---

## 5. 环境元数据

```python
@dataclass
class EnvSpec:
    id: str                       # 完整 gym ID
    env_type: str                 # SupportedEnvType 值
    task_slug: str                # URL-safe 任务标识符
    task_description: str         # 完整自然语言任务描述
    suite: str | None             # 所属 suite（多任务环境），单任务为 None
    default_robot: str | None     # 默认机器人型号，None 表示不适用
    available_robots: list[str]   # 可切换的机器人列表（空 = 不可切换）
    action_dim: int | None        # 动作维度
    max_episode_steps: int        # 默认最大步数
    requires_gpu: bool            # 是否需要 GPU
    requires_sim_install: bool    # 是否需要额外仿真器
```

---

## 6. API 设计

```python
# 顶层 API（放在 adapter/ 下）
import openeta.env_registry as registry

# ─── 创建环境 ───
env = gym.make("openeta/behavior/turning_on_radio-v0", seed=0)
env = gym.make("openeta/behavior/turning_on_radio-v0", seed=0, robot="franka_panda")

# ─── 发现环境 ───
specs = registry.list_envs()                           # 所有已注册环境
specs = registry.list_envs(env_type="behavior")        # 按类型过滤
specs = registry.list_envs(env_type="libero", suite="LIVING_ROOM_SCENE1")  # 按 suite 过滤

# ─── 搜索 ───
# 基于 task_description 的语义搜索（返回匹配度排序的列表）
results = registry.search("radio")
results = registry.search("pick up the bowl")

# ─── 信息查询 ───
spec = registry.get_env_spec("openeta/behavior/turning_on_radio-v0")
print(spec.task_description)     # "Turn on the radio receiver that's on the table..."
print(spec.default_robot)        # "franka_panda"
print(spec.available_robots)     # ["franka_panda", "fetch"]
```

---

## 7. 注册流程

### 7.1 静态注册（硬编码）

对**任务数量固定且可枚举**的环境：

```python
# behavior: task 列表来自 behavior_task.jsonl
for entry in _BEHAVIOR_TASKS:
    task_name = entry["task_name"]   # e.g. "turning_on_radio"
    description = entry["task"]      # e.g. "Turn on the radio receiver..."
    register(
        id=f"openeta/behavior/{task_name}-v0",
        entry_point="adapter.env_registry:make_env",
        kwargs={"env_type": "behavior", "task_name": task_name}
    )

# metaworld: 从 MetaWorldBenchmark 枚举所有 env_name
for suite in ["MT50", "ML45_ind", "ML45_ood"]:
    for env_name in MetaWorldBenchmark(suite).get_env_names():
        register(id=f"openeta/metaworld/{suite}/{env_name}-v0", ...)
```

### 7.2 动态注册（运行时发现）

对**任务列表取决于本地安装**的环境：

```python
def _register_maniskill_envs():
    try:
        import mani_skill
        for task_id in mani_skill.get_all_task_ids():
            register(id=f"openeta/maniskill/{task_id}-v0", ...)
    except ImportError:
        pass  # 未安装，跳过
```

### 7.3 `make_env` 工厂函数

```python
def make_env(env_type: str, task_name: str | None = None,
             suite: str | None = None, num_envs: int = 1,
             seed: int = 0, robot: str | None = None, **overrides):
    """
    Build a minimal RLinf-compatible DictConfig and instantiate the env.

    Defaults to num_envs=1 (single env for agent interaction).
    """
    base_cfg = _build_minimal_config(
        env_type, task_name=task_name, suite=suite, robot=robot
    )
    env_cls = get_env_cls(env_type, base_cfg)
    return env_cls(cfg=base_cfg, num_envs=num_envs, seed_offset=seed,
                   total_num_processes=1, worker_info=None)
```

---

## 8. 不纳入注册的环境

| env_type | 原因 |
|----------|------|
| `realworld` | 真机环境，需要硬件，不适用于 `gym.make()` |
| `opensora_wm` / `wan_wm` | World model，用于 rollout 而非交互 |
| `embodichain` | 依赖外部 EmbodiChain 安装路径 |

---

## 9. 实现计划

### Phase 1: 基础设施（`adapter/env_registry.py`）

- [ ] `EnvSpec` dataclass
- [ ] `make_env()` 工厂函数
- [ ] `_build_minimal_config()` — 从最少参数组装各 env_type 的 `DictConfig`
- [ ] 硬编码注册 behavior 的 50 个 task
- [ ] 动态注册 maniskill（如已安装）

### Phase 2: 扩展覆盖

- [ ] Libero 单任务分解注册
- [ ] MetaWorld
- [ ] Genesis（纯 Python，最易集成）
- [ ] IsaacLab / Polaris
- [ ] RoboCasa, RoboTwin, RoboVerse
- [ ] Calvin, FrankaSim, Habitat, D4RL

### Phase 3: 工具链

- [ ] `list_envs()` / `search()` API
- [ ] `openeta` CLI: `openeta env list`, `openeta env info <id>`
- [ ] 端到端测试：每个 env_type 至少一个 task 能完成 `reset()` + `step()`

---

## 10. 设计原则

1. **自然语言可寻址** — task_slug 来自 task 的自然语言描述，人类和 code agent 都能直观理解
2. **一个 gym.make() 一个 task** — 复合环境自动分解为独立的单任务 env
3. **robot 是配置参数，不是 ID 的一部分** — `gym.make(id, robot="...")` 而非 `id/robot/task`
4. **不存在则静默跳过** — 未安装的仿真器对应的 env 不注册，`list_envs()` 只列出可用项
5. **最小配置，延迟构建** — 注册时只存最少参数，首次 `make()` 时组装完整 `DictConfig`
6. **OpenETA 拥有注册表** — 注册代码在 `adapter/` 下，不修改 `sim/rlinf/` vendored 代码
