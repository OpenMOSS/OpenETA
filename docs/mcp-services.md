# OpenETA MCP Services

OpenETA can run heavyweight perception and manipulation backends as separate MCP
services. This keeps the default OpenETA runtime lightweight while allowing
local or LAN deployments of GPU-backed tools.

This page covers the service manager:

```bash
python scripts/openeta_mcp_services.py \
  <command> <sam3|anygrasp|anyplace|contact_graspnet|all>
```

The manager starts service processes in the background and writes pid/log state
under `<repo>/outputs/mcp_services/` by default.

## Services

- `sam3`: exposes the SAM3 MCP tool `segment`.
- `anygrasp`: exposes the AnyGrasp MCP tool `detect_grasps`.
- `anyplace`: exposes the AnyPlace MCP tool `predict_placement`.
- `contact_graspnet`: exposes the independent targeted Contact-GraspNet MCP
  tool `predict_grasps`. It is a non-commercial research/evaluation backend
  for Panda-compatible grippers.

AnyGrasp grasp candidates are returned in camera frame with
`ranking=score_descending`, zero-based `rank`, and `backend_index`. The agent
uses rank 0 greedily and owns downstream rejection/fallback state; the MCP
service remains stateless. Candidates include
`camera_frame="opencv"` unless a future tool docstring explicitly says
otherwise. The current MuJoCo simulator MCP observations provide `pos + mat`
camera calibration in OpenGL camera convention with flattened row-major
`mat`. The agent converts AnyGrasp/OpenCV candidates to world-frame control
targets with `camera_pose_to_world`.

Contact-GraspNet candidates use the same `camera/opencv` scene frame and are
normalized to the GraspNet grasp-frame convention, while retaining explicit
`source_model="contact_graspnet"`, `gripper_model="panda"`, and
`gripper_depth=0.1034`. The agent tool uses RGB only to verify SAM3 mask
provenance; RGB is not sent to the Contact-GraspNet MCP.

Default local URLs:

```text
SAM3     http://127.0.0.1:8773/sse
AnyGrasp http://127.0.0.1:8774/sse
AnyPlace http://127.0.0.1:8775/sse
Contact-GraspNet http://127.0.0.1:8776/sse
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
  --anyplace-config-path /path/to/anyplace-config.yaml
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

Start all four services:

```bash
python scripts/openeta_mcp_services.py start all \
  --sam3-python /path/to/sam3-env/bin/python \
  --anygrasp-python /path/to/anygrasp-env/bin/python \
  --anygrasp-sdk-root /path/to/anygrasp_sdk \
  --anygrasp-checkpoint-path /path/to/checkpoint_detection.tar \
  --anyplace-python /path/to/anyplace-env/bin/python \
  --anyplace-root /path/to/anyplace \
  --anyplace-config-path /path/to/anyplace-config.yaml \
  --contact-graspnet-python /path/to/contact-env/bin/python \
  --contact-graspnet-root /path/to/contact_graspnet_pytorch \
  --contact-graspnet-checkpoint-dir /path/to/contact-checkpoint
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
  --contact-graspnet-checkpoint-dir /path/to/contact-checkpoint
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
export OPENETA_CONTACT_GRASPNET_PYTHON=/path/to/contact-env/bin/python
export OPENETA_CONTACT_GRASPNET_ROOT=/path/to/contact_graspnet_pytorch
export OPENETA_CONTACT_GRASPNET_CHECKPOINT_DIR=/path/to/contact-checkpoint
```

Ports and state directory can be overridden:

```bash
python scripts/openeta_mcp_services.py start all \
  --sam3-port 8773 \
  --anygrasp-port 8774 \
  --anyplace-port 8775 \
  --contact-graspnet-port 8776 \
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
  --contact-graspnet-python /path/to/contact-env/bin/python \
  --contact-graspnet-root /path/to/contact_graspnet_pytorch \
  --contact-graspnet-checkpoint-dir /path/to/contact-checkpoint
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
