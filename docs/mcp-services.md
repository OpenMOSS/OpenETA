# OpenETA MCP Services

OpenETA can run heavyweight perception and manipulation backends as separate MCP
services. This keeps the default OpenETA runtime lightweight while allowing
local or LAN deployments of GPU-backed tools.

This page covers the service manager:

```bash
python scripts/openeta_mcp_services.py \
  <command> <sam3|anygrasp|anyplace|contact_graspnet|molmopoint|graspgenx|unidepth_v2|all>
```

The manager starts service processes in the background and writes pid/log state
under `<repo>/outputs/mcp_services/` by default.

## Services

- `sam3`: exposes the text-prompt tool `segment` and the pixel point-prompt
  tool `segment_points`.
- `anygrasp`: exposes the AnyGrasp MCP tool `detect_grasps`.
- `anyplace`: exposes the AnyPlace MCP tool `predict_placement`.
- `contact_graspnet`: exposes the independent targeted Contact-GraspNet MCP
  tool `predict_grasps`. It is a non-commercial research/evaluation backend
  for Panda-compatible grippers.
- `molmopoint`: exposes the independent MolmoPoint MCP tool `point_image` for
  deterministic single- and multi-image pixel grounding.
- `graspgenx`: exposes the independent geometry-only GraspGenX MCP tools
  `list_grippers` and `predict_grasps`. The backend is loaded lazily on the
  first prediction request.
- `unidepth_v2`: exposes `estimate_depth` using the official UniDepth V2 ViT-L
  metric-depth model. It returns float32 NPY Base64 depth and relative
  confidence for conservative RGB-D fusion.

AnyGrasp grasp candidates are returned in camera frame with
`ranking=score_descending`, zero-based `rank`, and `backend_index`. The agent
uses rank 0 greedily and owns downstream rejection/fallback state; the MCP
service remains stateless. Candidates include
`camera_frame="opencv"` unless a future tool docstring explicitly says
otherwise. The current MuJoCo simulator MCP observations provide `pos + mat`
camera calibration in OpenGL camera convention with flattened row-major
`mat`. The agent converts AnyGrasp/OpenCV candidates to world-frame control
targets with `camera_pose_to_world`.

The deployed SAM3 `segment_points` contract accepts the original image as
`image_base64`, its `image_format`, and one to 64 `points`. Each point is
`{"x": <pixel>, "y": <pixel>, "label": 0|1}` in the original image with a
top-left origin; `label=1` is foreground, `label=0` is background, and at least
one foreground point is required. Coordinates are not normalized and an
out-of-bounds point rejects the request. It returns exactly three multimask
candidates ranked by predicted mask quality. The score is a ranking hint, so
the Agent still resolves the normal SAM3 selection obligation.

When `OPENETA_OBJECT_MEMORY_BANK_API_KEY` is configured, Agent-side
`retrieve_asset_reference` queries the controlled `/bundle?name=<namespace>/<asset>`
endpoint. It materializes front/side/top views in the session artifact root and
expects `target_object` to contain only the object's identity or appearance,
not a scene relation or destination. For example, a task such as “pick up the
black bowl on the cookie box” uses `target_object="black bowl"`; the location
phrase remains context for the downstream scene localizer.
uses a clean-context VLM sub-agent to locate one foreground point in the
original scene. The host validates the point and draws the audit marker; the
main planner only copies the exact `positive_points` into `sam3`, which routes
to `segment_points`. The endpoint may be overridden with
`OPENETA_OBJECT_MEMORY_BANK_URL`; credentials are never tool parameters.

The simulator adapter preserves the controller's current EEF orientation for
ranked GraspNet-family candidates by default. A deployment with a calibrated
GraspNet-to-Panda mapping can set
`SimulatorMcpToolProxyConfig.forward_grasp_candidate_orientation=True`; that
maps Panda EEF x to GraspNet y (closing), EEF y to GraspNet z (binormal), and
EEF z to GraspNet x (approach). Ordinary non-candidate world poses forward
explicit orientation unchanged. The Agent ToolResult records the selected
behavior as `mcp.target_orientation_mode`.

Contact-GraspNet candidates use the same `camera/opencv` scene frame and are
normalized to the GraspNet grasp-frame convention, while retaining explicit
`source_model="contact_graspnet"`, `gripper_model="panda"`, and
`gripper_depth=0.1034`. The agent tool uses RGB only to verify SAM3 mask
provenance; RGB is not sent to the Contact-GraspNet MCP.

The agent-facing `graspgenx` tool requires aligned RGB, depth, a complete SAM3
mask artifact, intrinsics, an exact `gripper_name` from
`list_graspgenx_grippers`, and a nonzero camera-frame up direction. RGB is used
only for provenance and local overlays; only Base64 depth and mask bytes cross
the GraspGenX MCP boundary. Successful results expose all 1–20 returned
collision-free candidates in normalized `camera/opencv` GraspNet convention.
Rank 0 enters the same greedy runtime policy used by AnyGrasp. The raw audit
record retains the model-native pose for visualization and diagnosis, while
planner-facing candidates do not. A normalized GraspGenX candidate and its
`details.source` can be passed directly to the agent-facing AnyPlace tool.

Default local URLs:

```text
SAM3     http://127.0.0.1:8773/sse
AnyGrasp http://127.0.0.1:8774/sse
AnyPlace http://127.0.0.1:8775/sse
Contact-GraspNet http://127.0.0.1:8776/sse
MolmoPoint       http://127.0.0.1:8777/sse
GraspGenX        http://127.0.0.1:8778/sse
UniDepth V2      http://127.0.0.1:8779/sse
```

Use `--host 0.0.0.0` only when intentionally sharing the service on a trusted
LAN.

## Start

SAM3 uses the official SAM3 builder to load weights through the configured
Hugging Face auth/cache environment. Prepare access before starting the service.
Do not pass Hugging Face tokens on the command line.

```bash
python scripts/openeta_mcp_services.py start sam3 \
  --sam3-python /path/to/sam3-env/bin/python
```

Optional SAM3 cache controls:

```bash
python scripts/openeta_mcp_services.py start sam3 \
  --sam3-python /path/to/sam3-env/bin/python \
  --sam3-hf-home /path/to/hf-home \
  --sam3-cache-dir /path/to/hf-cache
```

AnyGrasp requires its SDK root and detection checkpoint:

```bash
python scripts/openeta_mcp_services.py start anygrasp \
  --anygrasp-python /path/to/anygrasp-env/bin/python \
  --anygrasp-sdk-root /path/to/anygrasp_sdk \
  --anygrasp-checkpoint-path /path/to/checkpoint_detection.tar
```

AnyPlace requires its repository root and model configuration:

```bash
python scripts/openeta_mcp_services.py start anyplace \
  --anyplace-python /path/to/anyplace-env/bin/python \
  --anyplace-root /path/to/anyplace \
  --anyplace-config-path /path/to/anyplace-config.yaml \
  --anyplace-depth-truncation 2.0
```

Contact-GraspNet requires its external PyTorch checkout and checkpoint
directory. The managed service fixes seed `0`, depth range `0.2-1.8 m`, and a
maximum of 20 returned candidates:

```bash
python scripts/openeta_mcp_services.py start contact_graspnet \
  --contact-graspnet-python /path/to/contact-env/bin/python \
  --contact-graspnet-root /path/to/contact_graspnet_pytorch \
  --contact-graspnet-checkpoint-dir /path/to/contact-checkpoint
```

MolmoPoint requires a CUDA service environment and a pre-deployed Hugging Face
snapshot. Runtime requests are strictly local-only and never download weights.
The revision must be a full 40-character commit SHA:

```bash
python scripts/openeta_mcp_services.py start molmopoint \
  --molmopoint-python /path/to/molmopoint-env/bin/python \
  --molmopoint-hf-home /path/to/huggingface-home \
  --molmopoint-model-id allenai/MolmoPoint-8B \
  --molmopoint-model-revision 188130f961c8e0888a34e11121a1423c461a01ba
```

GraspGenX requires its repository, release checkpoint tree, and gripper
description checkout. Requests must choose one gripper returned by
`list_grippers`:

```bash
python scripts/openeta_mcp_services.py start graspgenx \
  --graspgenx-python /path/to/graspgenx-env/bin/python \
  --graspgenx-root /path/to/GraspGenX \
  --graspgenx-checkpoint-root /path/to/checkpoints/release \
  --graspgenx-gripper-descriptions-root /path/to/gripper-descriptions
```

UniDepth V2 runs in a dedicated environment with the official UniDepth package
and a PyTorch build appropriate for the deployment GPU:

```bash
python scripts/openeta_mcp_services.py start unidepth_v2 \
  --unidepth-v2-python /path/to/unidepth-env/bin/python \
  --unidepth-v2-model-id lpiccinelli/unidepth-v2-vitl14 \
  --unidepth-v2-device cuda \
  --unidepth-v2-resolution-level 4
```

The model is licensed CC BY-NC 4.0; review deployment use before commercial or
externally hosted production use.

For an ARM64 NVIDIA DGX Spark deployment, build from fixed `openeta` and
official UniDepth source snapshots. The deployment validated here uses official
UniDepth commit `8d8cfe4c7ee15297099983607febf0d4f32eb3d6`:

```bash
docker build \
  -f openeta/tools/Dockerfile.unidepth-v2-dgx-spark \
  -t openeta/unidepth-v2-mcp:dgx-spark .

docker run --rm \
  -v /path/to/models/hf:/models/hf \
  --entrypoint python \
  openeta/unidepth-v2-mcp:dgx-spark \
  -c 'from huggingface_hub import snapshot_download; snapshot_download(
      repo_id="lpiccinelli/unidepth-v2-vitl14",
      allow_patterns=["config.json", "model.safetensors"],
  )'

docker run -d \
  --name openeta-unidepth-v2-mcp \
  --restart unless-stopped \
  --gpus all \
  -p 8779:8779 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -v /path/to/models/hf:/models/hf:ro \
  openeta/unidepth-v2-mcp:dgx-spark
```

The build context must expose the source snapshots as `openeta/` and
`unidepth/`. Keep the Hugging Face cache on persistent storage and pin the
UniDepth checkout used for the image; the Dockerfile deliberately installs
UniDepth with `--no-deps` so its broad requirements cannot replace the NGC
Torch/CUDA stack.

Start all seven services:

```bash
python scripts/openeta_mcp_services.py start all \
  --sam3-python /path/to/sam3-env/bin/python \
  --anygrasp-python /path/to/anygrasp-env/bin/python \
  --anygrasp-sdk-root /path/to/anygrasp_sdk \
  --anygrasp-checkpoint-path /path/to/checkpoint_detection.tar \
  --anyplace-python /path/to/anyplace-env/bin/python \
  --anyplace-root /path/to/anyplace \
  --anyplace-config-path /path/to/anyplace-config.yaml \
  --anyplace-depth-truncation 2.0 \
  --contact-graspnet-python /path/to/contact-env/bin/python \
  --contact-graspnet-root /path/to/contact_graspnet_pytorch \
  --contact-graspnet-checkpoint-dir /path/to/contact-checkpoint \
  --molmopoint-python /path/to/molmopoint-env/bin/python \
  --molmopoint-hf-home /path/to/huggingface-home \
  --graspgenx-python /path/to/graspgenx-env/bin/python \
  --graspgenx-root /path/to/GraspGenX \
  --graspgenx-checkpoint-root /path/to/checkpoints/release \
  --graspgenx-gripper-descriptions-root /path/to/gripper-descriptions \
  --unidepth-v2-python /path/to/unidepth-env/bin/python
```

Check commands without starting processes:

```bash
python scripts/openeta_mcp_services.py start all --dry-run \
  --sam3-python /path/to/sam3-env/bin/python \
  --anygrasp-python /path/to/anygrasp-env/bin/python \
  --anygrasp-sdk-root /path/to/anygrasp_sdk \
  --anygrasp-checkpoint-path /path/to/checkpoint_detection.tar \
  --anyplace-python /path/to/anyplace-env/bin/python \
  --anyplace-root /path/to/anyplace \
  --anyplace-config-path /path/to/anyplace-config.yaml \
  --contact-graspnet-python /path/to/contact-env/bin/python \
  --contact-graspnet-root /path/to/contact_graspnet_pytorch \
  --contact-graspnet-checkpoint-dir /path/to/contact-checkpoint \
  --molmopoint-python /path/to/molmopoint-env/bin/python \
  --molmopoint-hf-home /path/to/huggingface-home \
  --graspgenx-python /path/to/graspgenx-env/bin/python \
  --graspgenx-root /path/to/GraspGenX \
  --graspgenx-checkpoint-root /path/to/checkpoints/release \
  --graspgenx-gripper-descriptions-root /path/to/gripper-descriptions \
  --unidepth-v2-python /path/to/unidepth-env/bin/python
```

Dry-run output may include SDK, checkpoint, Python, or cache paths. Do not share
dry-run output if it contains machine-specific or private paths.

## Configuration

Python interpreter priority:

```text
CLI argument > environment variable > current Python
```

Environment variables:

```bash
export OPENETA_SAM3_PYTHON=/path/to/sam3-env/bin/python
export OPENETA_ANYGRASP_PYTHON=/path/to/anygrasp-env/bin/python
export OPENETA_ANYGRASP_SDK_ROOT=/path/to/anygrasp_sdk
export OPENETA_ANYGRASP_CHECKPOINT_PATH=/path/to/checkpoint_detection.tar
export OPENETA_ANYPLACE_PYTHON=/path/to/anyplace-env/bin/python
export OPENETA_ANYPLACE_ROOT=/path/to/anyplace
export OPENETA_ANYPLACE_CONFIG_PATH=/path/to/anyplace-config.yaml
export OPENETA_ANYPLACE_DEPTH_TRUNCATION=2.0
export OPENETA_CONTACT_GRASPNET_PYTHON=/path/to/contact-env/bin/python
export OPENETA_CONTACT_GRASPNET_ROOT=/path/to/contact_graspnet_pytorch
export OPENETA_CONTACT_GRASPNET_CHECKPOINT_DIR=/path/to/contact-checkpoint
export OPENETA_MOLMOPOINT_PYTHON=/path/to/molmopoint-env/bin/python
export OPENETA_MOLMOPOINT_HF_HOME=/path/to/huggingface-home
export OPENETA_MOLMOPOINT_MODEL_ID=allenai/MolmoPoint-8B
export OPENETA_MOLMOPOINT_MODEL_REVISION=188130f961c8e0888a34e11121a1423c461a01ba
export OPENETA_GRASPGENX_PYTHON=/path/to/graspgenx-env/bin/python
export OPENETA_GRASPGENX_ROOT=/path/to/GraspGenX
export OPENETA_GRASPGENX_CHECKPOINT_ROOT=/path/to/checkpoints/release
export OPENETA_GRASPGENX_GRIPPER_DESCRIPTIONS_ROOT=/path/to/gripper-descriptions
export OPENETA_UNIDEPTH_V2_PYTHON=/path/to/unidepth-env/bin/python
export OPENETA_UNIDEPTH_V2_MODEL_ID=lpiccinelli/unidepth-v2-vitl14
export OPENETA_UNIDEPTH_V2_DEVICE=cuda
export OPENETA_UNIDEPTH_V2_RESOLUTION_LEVEL=4
```

Ports and state directory can be overridden:

```bash
python scripts/openeta_mcp_services.py start all \
  --sam3-port 8773 \
  --anygrasp-port 8774 \
  --anyplace-port 8775 \
  --contact-graspnet-port 8776 \
  --molmopoint-port 8777 \
  --graspgenx-port 8778 \
  --unidepth-v2-port 8779 \
  --state-dir /path/to/service-state
```

The service manager does not store model weights, license files, or tokens.
Keep checkpoints, licenses, Hugging Face tokens, and machine-specific paths out
of git-tracked files.

## Operations

Check pid and health state:

```bash
python scripts/openeta_mcp_services.py status all
```

`status` exits with code `0` when it can report state. Use the printed `ok`,
`running`, and `health` fields to decide whether a service is healthy.

Check only the HTTP health endpoints:

```bash
python scripts/openeta_mcp_services.py health all
```

Check MCP SSE handshake and `list_tools()` without running model inference:

```bash
python scripts/openeta_mcp_services.py smoke all
```

`smoke` uses an MCP client in the Python environment that runs this manager.
Run it from an OpenETA/runtime environment with the `mcp` package installed.

Stop services with `SIGTERM`:

```bash
python scripts/openeta_mcp_services.py stop all
```

Force stop with `SIGKILL`:

```bash
python scripts/openeta_mcp_services.py stop all --force
```

Restart services:

```bash
python scripts/openeta_mcp_services.py restart all \
  --sam3-python /path/to/sam3-env/bin/python \
  --anygrasp-python /path/to/anygrasp-env/bin/python \
  --anygrasp-sdk-root /path/to/anygrasp_sdk \
  --anygrasp-checkpoint-path /path/to/checkpoint_detection.tar \
  --anyplace-python /path/to/anyplace-env/bin/python \
  --anyplace-root /path/to/anyplace \
  --anyplace-config-path /path/to/anyplace-config.yaml \
  --anyplace-depth-truncation 2.0 \
  --contact-graspnet-python /path/to/contact-env/bin/python \
  --contact-graspnet-root /path/to/contact_graspnet_pytorch \
  --contact-graspnet-checkpoint-dir /path/to/contact-checkpoint \
  --molmopoint-python /path/to/molmopoint-env/bin/python \
  --molmopoint-hf-home /path/to/huggingface-home \
  --graspgenx-python /path/to/graspgenx-env/bin/python \
  --graspgenx-root /path/to/GraspGenX \
  --graspgenx-checkpoint-root /path/to/checkpoints/release \
  --graspgenx-gripper-descriptions-root /path/to/gripper-descriptions
```

Use `--json` for machine-readable output:

```bash
python scripts/openeta_mcp_services.py status all --json
```

## Runtime Boundary

This service manager only starts and checks MCP services. It does not change
OpenETA tool registration, planner behavior, or runtime handler routing.
Connecting OpenETA runtime handlers to local or remote MCP SSE URLs is a
separate integration step.
