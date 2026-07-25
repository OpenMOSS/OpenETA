# Simulation Layer

每个仿真环境独立隔离在 `sim/venvs/<name>/` 下，彻底消除依赖冲突。对外提供统一观测结构、gymnasium 注册入口、MCP 远程控制服务和 Web Dashboard。

## 快速开始

```bash
# 一键安装 (任选)
bash scripts/setup_envs.sh metaworld
bash scripts/setup_envs.sh maniskill
bash scripts/setup_envs.sh metaworld,maniskill,libero

# 启动 MCP 服务 + Web Dashboard
python -m sim.mcp_server
# 或: python sim/mcp_server/server.py

# 浏览器打开 http://localhost:8765/
```

直接使用 gymnasium：

```bash
source sim/venvs/maniskill/bin/activate

python -c "
import gymnasium as gym
import sim.env_registry
env = gym.make('openeta/maniskill_PickCube-v1-v0', render_mode='rgb_array')
obs, _ = env.reset()
print(obs['cameras'].keys(), obs['proprio'].keys(), obs['task_description'])
"
```

**只有已安装的 bench 才会注册环境** — 切换 venv 就切换可用环境集。

---

## 统一观测结构

所有后端返回相同的 `obs` 字典：

```python
obs = {
    "cameras": {
        "<name>": {
            "rgb":        np.ndarray  # (H, W, 3) uint8
            "depth":      np.ndarray  # (H, W) float32 | None  # 线性米 (metres)
            "intrinsics": dict | None
            "extrinsics": dict | None
        },
    },
    "proprio": {
        "joint_positions":  np.ndarray  # (N,) float32
        "joint_velocities": np.ndarray  # (N,) float32 | None
        "ee_pose":          np.ndarray  # (7,) float32 | None   # xyz + xyzw
        "gripper_open":     float | None
    },
    "task_description": str,            # 所有后端都保证存在
    "objects":  [{"name": str, "position": [x,y,z]}, ...],
    "metadata": {...},
}
```

### 深度 (depth)

`depth` **已经是线性的、以米为单位的度量深度**，无需任何客户端换算：

- **ManiSkill (SAPIEN)**：原生 int16 毫米 → `/1000` → 米。
- **MuJoCo (MetaWorld / LIBERO / FrankaSim / D4RL)**：原始 z-buffer ∈ [0, 1]，
  按 robosuite 公式线性化，并用 `model.stat.extent` 缩放后的裁剪面转成米：

  ```
  near = model.vis.map.znear * extent
  far  = model.vis.map.zfar  * extent
  z_m  = near / (1 - z_buf * (1 - near / far))
  ```
- **BEHAVIOR-1K (OmniGibson)**：`depth` / `depth_linear` 已是线性米，直接透传。

所有后端在 UnifiedEnv 边界把 NaN、Inf 和负深度转换成 `0.0`（无效像素），
避免非有限值进入 JSON；有效像素始终保留米制，不做逐帧归一化。

经 MCP 传输时编码为 uint16 PNG（毫米），解码用 `depth_m = pixel / 1000.0`。

### 相机内参 / 外参

```python
intrinsics = {
    "fx", "fy",          # 焦距 (像素)
    "cx", "cy",          # 主点 (像素)
    "width", "height",
    "znear", "zfar",     # MuJoCo: 度量裁剪面 (米)，界定有效深度范围
}
extrinsics = {           # 相机在 world 坐标系中的位姿 (非相对末端)
    "pos": [x, y, z],    # 相机位置 (米)
    # MuJoCo: 3×3 旋转矩阵 camera→world，行主序 (row-major) 展平
    "mat": [m00, m01, m02, m10, m11, m12, m20, m21, m22],
    "matrix_layout": "row_major",
    "frame_transform": "camera_to_world",
    "camera_frame": "opengl",       # 相机沿局部 -Z 观察
    # ManiSkill: 四元数 camera→world (已由 SAPIEN 原生 wxyz 重排为 xyzw)
    "quat_xyzw": [x, y, z, w],       # camera_frame="ros"，沿局部 +X 观察
}
```

> ⚠️ **layout 契约**：MuJoCo `mat` 为 **row-major**，用 `R = np.array(mat).reshape(3,3)`
> 直接得到 camera→world 旋转，**不要转置**。请始终读取返回里的
> `matrix_layout` / `frame_transform` / `camera_frame` 标签，不要凭约定猜测。

`R` 的列 = 相机局部轴在 world 中的表示（col0=右 X，col1=上 Y，col2=前 Z）。
相机沿局部 **-Z** 观察，故世界系视线方向为 `-R[:,2]`。变换公式：

```
R = np.array(mat).reshape(3, 3)   # row-major, 无需转置
p_world = R @ p_cam + pos         # camera → world
p_cam   = R.T @ (p_world - pos)   # world → camera
```

### 像素 → 世界坐标 (deprojection)

这是抓取误差的头号来源。针孔反投影得到的是 **OpenCV 光学系** 点（X 右、Y 下、
Z 前），必须先转到相机原生系（由 `camera_frame` 决定）再乘旋转：

```python
x = (u - cx) * d / fx          # d = depth_m (线性米)
y = (v - cy) * d / fy
p_opencv = np.array([x, y, d])

# 光学系 → 相机原生系 (按 camera_frame)：
#   "opengl" (MuJoCo):   p_cam = np.diag([1,-1,-1]) @ p_opencv   # 翻 Y、Z
#   "ros"    (ManiSkill): p_cam = np.array([d, -x, -y])          # 轴置换

R = np.array(mat).reshape(3, 3)   # MuJoCo；ManiSkill 用 quat_xyzw 转 R
p_world = R @ p_cam + pos
```

> ⚠️ 光学→原生 这一步 **必须做且各后端不同**（读 `camera_frame`，别猜）。
> 已验证：正确的往返重投影把物体中心恢复到 ~2-3 cm（残差为表面-中心偏移），
> OpenGL 与 ROS 后端均如此。跳过这步会把抓取点镜像/旋转到错误世界位置。

### 各环境相机命名与深度

| 后端 | 相机名 | 深度 | 说明 |
|------|--------|------|------|
| `dummy` | `dummy_front` | 合成 | 合成 RGBD |
| `metaworld` | `view` | ✅ 米 | MuJoCo `rgbd_tuple` 渲染 (默认 480×480) |
| `maniskill` | `<sensor_name>` | ✅ 米 | MS3 原生传感器 (int16 毫米 → 米) |
| `libero` | `agentview`, `wrist` | ✅ 米 | 固定视角 + 腕部相机 |
| `genesis` | `head` | — | 头部相机 |
| `behavior` | `zed_head`, `wrist_left`, `wrist_right` | 米（契约测试） | ZED + 双腕 Realsense；相机标定和双臂状态由 DirectEnv 注入 |
| `robocasa` | `agentview_left`, `agentview`, `wrist`, `agentview_right` | 米（契约测试） | RoboCasa365 原生同相机 RGB-D，MuJoCo z-buffer 线性化 |
| `d4rl` | *(无)* | — | 纯状态 |

> ✅ 表示已安装并验证深度为正确的线性度量米值（无 NaN/inf）。已验证环境：
> `metaworld_50_assembly-v3`、`maniskill_PickCube-v1`、`libero_libero_10_task0`。

---

## 环境注册 API

```python
from sim.env_registry import list_envs, search, get_env_spec, hot_activate, hot_list_available

# 查看已安装的 bench
hot_list_available()                     # {"metaworld": True, "libero": True, ...}

# 按类型过滤
list_envs(env_type="maniskill")          # → list[EnvSpec]

# 语义搜索
search("pick up")                        # → list[EnvSpec]

# 获取单个环境的详情
get_env_spec("openeta/maniskill_PickCube-v1-v0").task_description

# 运行时动态激活某个 bench 的 venv
hot_activate("libero")                   # → True (成功) / False (未安装)
```

---

## MCP 服务 + Web Dashboard

```bash
python -m sim.mcp_server                # 默认 SSE 模式，端口 8765
python -m sim.mcp_server --transport stdio  # stdio 模式 (本地 Claude)
python -m sim.mcp_server --port 9000    # 自定义端口
```

### Web Dashboard

打开 `http://localhost:8765/` 即可交互式操控仿真环境：
- 左侧面板搜索 / 选择环境
- DPAD 按钮控制末端平移 (X/Y/Z)
- 夹爪开关、手动 action 输入
- Auto step+render 实时画面
- Speed multiplier 调节步长

### REST API

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/benches` | 列出已安装 bench |
| `GET` | `/api/envs?type=&q=` | 搜索/过滤环境 |
| `POST` | `/api/envs` | 创建环境 → `{handle, action_dim, backend}` |
| `POST` | `/api/envs/{handle}/reset` | 重置 → `{image_base64, width, height, proprio, ...}` |
| `POST` | `/api/envs/{handle}/step` | 执行动作 → 同 reset 结构 + `{reward, terminated, truncated}` |
| `POST` | `/api/envs/{handle}/observe` | 返回最近一次观测 |
| `POST` | `/api/envs/{handle}/render` | 渲染当前帧 |
| `DELETE` | `/api/envs/{handle}` | 关闭/销毁环境 |
| `GET` | `/api/sessions` | 列出活跃 session |
| `GET` | `/session/{sid}` | Session 实时 Dashboard |
| `GET` | `/session/{sid}/stream[/{handle}]` | SSE 实时画面流 |

### MCP 工具 (14 个)

Claude Code / Claude Desktop 通过 SSE 连接后自动发现：

| 工具 | 参数 | 说明 |
|------|------|------|
| `hot_activate` | `bench` | 运行时激活 bench venv |
| `list_available_benches` | — | 返回已安装 bench 列表 |
| `list_envs` | `env_type` | 按类型列环境 |
| `search_envs` | `query` | 语义搜索任务 |
| `create_env` | `env_id, render_mode, seed, task, image_width/height, include_objects` | 创建环境 → handle + session_id |
| `reset_env` | `handle, seed` | 重置并返回初始观测 |
| `step_env` | `handle, action, num_steps` | 执行动作并返回观测 |
| `move_to` | `handle, x, y, z, roll/pitch/yaw, enable_collision_check` | 闭环移动末端到绝对位姿 (含 cuRobo 碰撞检测) |
| `gripper_open` | `handle` | 打开夹爪 |
| `gripper_close` | `handle` | 关闭夹爪 |
| `observe_env` | `handle` | 返回最近观测 (不 step) |
| `render_env` | `handle` | 渲染当前帧 (完整多相机观测) |
| `close_env` | `handle` | 关闭环境 |
| `list_active_envs` | — | 列出活跃 handle |

> 观测中的 `cameras[].depth_base64` 为 uint16 PNG（线性米 × 1000），
> `intrinsics`/`extrinsics` 语义见上文「深度」「相机内参 / 外参」。
> `move_to` / gripper 使用 `sim/mcp_server/action_codecs.py` 的显式 backend
> codec；未知 backend 或未声明的动作布局会返回结构化错误，不会猜测动作槽位。
> BEHAVIOR DirectEnv 默认把双臂切换为有界 delta-pose IK，并在创建环境时
> 返回实际 arm/gripper action indices。`enable_collision_check` 当前只在已接入
> checker 的 LIBERO / ManiSkill 上执行；其他后端会明确标记 unavailable。
> `close_env` 幂等，显式关闭和 TTL cleanup 都会释放 worker 引用。

Claude Code 配置 (`.mcp.json`，项目根目录已有)：

```json
{
  "mcpServers": {
    "openeta": {
      "url": "http://localhost:8765/sse"
    }
  }
}
```

远程连接：

```bash
ngrok http 8765
# 把 url 换成 ngrok 的公网地址即可
```

---

## Viewer 接口

```python
env = gym.make("openeta/maniskill_PickCube-v1-v0", render_mode="human")     # 弹窗
env = gym.make("openeta/maniskill_PickCube-v1-v0", render_mode="rgb_array") # numpy
frame = env.render()    # rgb_array → (H,W,3); human → None
```

---

## Adapter 桥接

`sim/adapter.py` 将 UnifiedEnv 桥接到 OpenETA 协议 (`adapter/protocol.py`)：

```python
from sim.adapter import make_sim_adapter

adapter = make_sim_adapter("openeta/metaworld_50_assembly-v3-v0")
obs = adapter.reset(task="assembly")          # → EnvObservation
result = adapter.step(EnvAction(...))          # → StepResult
adapter.close()
```

`EnvObservation` 包含 `CameraFrame` 列表、`RobotState` (joint positions/velocities, EE pose, gripper)、对象摘要和 metadata，可直接喂给 `AgentSimBridge`。

### Episode loop、日志与回放

```python
from adapter import AgentSimBridge
from logger import EpisodeLogger

bridge = AgentSimBridge(simulator=adapter, agent=agent_adapter)
outcome = bridge.run_episode(
    task="pick and place the mug",
    seed=0,
    max_steps=100,
    logger=EpisodeLogger("artifacts/episodes"),
    environment="robocasa",
)
bridge.close()
```

每个 episode 目录包含 `episode.json`、`steps.jsonl` 和 `media/`。JSONL 只保存
RGB/depth 相对路径，RGB 为 PNG，depth 为 uint16 毫米 PNG；每轮记录 observation、
plan、safety/checker 信息、EnvAction、StepResult 和 latency。`response` 不会被送入
仿真的 Box action space。导出回放视频：

```bash
uv run openeta-replay artifacts/episodes/EPISODE_ID \
  --camera agentview --output artifacts/episodes/EPISODE_ID/replay.mp4
```

---

## 环境状态

> 数量为对应 venv 中实际 `gym.registry` 注册数（随 ManiSkill/MetaWorld 版本浮动）。

| 环境 | 数量 | 机器人 | 安装命令 | 状态 |
|------|:----:|--------|----------|:----:|
| `dummy` | 2 | — | built-in | ✅ |
| `metaworld` | 96 | Sawyer | `setup_envs.sh metaworld` | ✅ 深度已验证 |
| `maniskill` | 52 | 13 种² | `setup_envs.sh maniskill` | ✅ 深度已验证 |
| `libero` | 130 | Franka Panda | `setup_envs.sh libero` | ✅ 深度已验证 |
| `robocasa` | 634¹ | PandaOmron | `setup_envs.sh robocasa` | ✅ RoboCasa365 benchmark |
| `genesis` | 1 | Franka | `setup_envs.sh genesis` | ⚠️ ³ |
| `behavior` | 1,016 BDDL definitions (50-task RLinf eval subset) | R1Pro | OmniGibson 3.9 / Isaac Sim 5.1 | integrated; GPU worker required |
| `d4rl` | 9 | — | `setup_envs.sh d4rl` | TODO |
| `calvin` | — | Franka | 需 clone + dataset | TODO |
| `robotwin` | — | 多种 | 需 clone | TODO |
| `roboverse` | — | 多种 | 需 clone + MetaSim | TODO |
| `habitat` | — | — | 需 conda | TODO |
| `isaaclab` | — | 多种 | 需 Isaac Sim | TODO |
| `polaris` | — | 多种 | 需 Isaac Sim + PolaRiS | TODO |
| `embodichain` | — | — | pip install | TODO |
| `frankasim` | — | Franka | Repo not found | TODO |

¹ RoboCasa 的 317 个官方任务分别注册 pretrain/target 两个 split，因此是 634 个 env ID。
² ManiSkill 机器人: panda, SO100, widowxai, allegro_hand, dclaw, tri_finger, ant, hopper, humanoid, cartpole, anymal_c, unitree_g1/h1/go2
³ Genesis 需 CUDA 12.x 驱动 (nvJitLink 与 CUDA 13.2 驱动不兼容)，代码已有但未测试

### 各后端 action 维度

| 后端 | 维数 | 说明 |
|------|:----:|------|
| `metaworld` | 4 | (dx, dy, dz, gripper), 范围 [-1, 1] |
| `maniskill` | 7 | (dx, dy, dz, roll, pitch, yaw, gripper), delta, 范围 [-1, 1] |
| `libero` | 7 | (dx, dy, dz, rx, ry, rz, gripper), OSC_POSE, 范围 [-1, 1] |
| `robocasa` | 12 | PandaOmron: arm `[0:6]`, gripper `6`, base `[7:10]`, torso `10`, mode `11` |
| `behavior` | 运行时声明 | R1Pro controller 顺序；MCP 读取 DirectEnv 返回的 IK / gripper 实际槽位，不硬编码维度 |
| `dummy` | dict | `{"action_type": "code_policy", "code": "..."}` |

### RoboCasa365 快速开始

RoboCasa 使用独立的 Python 3.11 venv；它的官方 LeRobot 依赖固定
`gymnasium<1.0`，不能和 OpenETA 主环境的 `gymnasium>=1.0` 混装。

```bash
./scripts/setup_envs.sh robocasa
source sim/venvs/robocasa/bin/activate
source sim/venvs/robocasa/activate_extra.sh

# 读取官方 task sets，并生成 317 tasks × 50 scenarios 的目标域清单
python -m scripts.robocasa_benchmark task-sets
python -m scripts.robocasa_benchmark manifest \
  --task-set all_tasks --split target --scenarios-per-task 50 --seed 0 \
  --output runs/robocasa/all_tasks_target.json
```

完整的环境 ID、评测恢复、policy runner 接口和 MCP 控制说明见
[ROBOCASA.md](ROBOCASA.md)。

---

## 目录结构

```
sim/
├── README.md                # 本文件
├── SETUP.md                 # 各环境详细安装说明 (参考用)
├── __init__.py              # 公开 UnifiedEnv, UnifiedSimulatorAdapter
├── env_registry.py          # EnvSpec, gymnasium 注册, list_envs, search, hot_activate
├── env_config.py            # 各后端的 DictConfig 构建器
├── unified_env.py           # 统一 obs 结构 + render 接口
├── adapter.py               # SimulatorAdapter — 桥接到 adapter/protocol.py
├── mcp_server/              # MCP 服务 + Web Dashboard + REST API (拆分为子包)
│   ├── __init__.py / __main__.py
│   ├── server.py            # FastMCP + CLI 入口
│   ├── session.py           # session 状态存储与生命周期
│   ├── worker_mgr.py        # per-bench 子进程 worker 管理
│   ├── rest_api.py          # REST API handler + SSE 直播
│   └── dashboard_html.py    # Web Dashboard HTML 模板
├── envs/                    # 按后端分隔的纯 gym.Env 封装
│   ├── metaworld.py
│   ├── maniskill.py
│   ├── libero.py
│   ├── genesis.py
│   └── ...
├── venvs/                   # 每个 bench 独立 uv venv (gitignored)
│   ├── metaworld/
│   ├── maniskill/
│   └── libero/
```

---

## 添加新环境

1. 在 `sim/envs/` 下创建 `newbench.py`，实现 `_make_newbench_direct(cfg)` 函数和 `_register_newbench()` 注册函数
2. 在 `sim/env_config.py` 中添加 `build_newbench_cfg()` DictConfig 构建器
3. 在 `sim/unified_env.py` 中添加 `_normalise_newbench()` 归一化逻辑
4. 在 `sim/env_registry.py` 的 `hot_activate()` 和 `_init_registry()` 中注册
5. 在 `scripts/setup_envs.sh` 中添加 `newbench)` case
