"""Start the dedicated Codex experiment simulator; Mink is the default."""
import argparse
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO))

from sim.controllers.dependency_overlay import validate_mink_dependency_overlay
from tools.codex_controller import CONTROLLER_IDS, DEFAULT_CONTROLLER


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--controller", choices=CONTROLLER_IDS, default=DEFAULT_CONTROLLER)
    p.add_argument("--mink-dependency-path", type=Path, default=REPO / "tmp/codex-mink-deps")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18778)
    p.add_argument("--private-state-dir", type=Path, default=None,
                   help="Operator-only snapshots, outside the model workspace")
    p.add_argument('--no-grip-stabilization', '--no-fixture-grip-stabilization', action='store_true',
                   help='Disable the opt-in held-object/fixture control used by Codex experiments')
    p.add_argument('--no-cartesian-segment', action='store_true',
                   help='Disable bounded Cartesian reference tracking for a paired baseline')
    return p


def launch_spec(args):
    env = dict(os.environ)
    # Explicit experiment selection overrides ambient settings from other runs.
    env["OPENETA_LIBERO_CONTROLLER_PROFILE"] = args.controller
    env['OPENETA_LIBERO_GRIP_STABILIZATION'] = (
        '1' if args.controller == 'mink_joint_velocity' and not args.no_grip_stabilization else '0')
    env.pop('OPENETA_LIBERO_FIXTURE_GRIP_STABILIZATION', None)
    env['OPENETA_LIBERO_CARTESIAN_SEGMENT'] = (
        '1' if args.controller == 'mink_joint_velocity' and not args.no_cartesian_segment else '0')
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    from datetime import datetime
    state_dir = args.private_state_dir or REPO / 'tmp/codex-private-state' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    env['OPENETA_PRIVATE_STATE_DIR'] = str(state_dir.resolve())
    if args.controller == "mink_joint_velocity":
        overlay = validate_mink_dependency_overlay(args.mink_dependency_path)
        env["OPENETA_LIBERO_MINK_DEPENDENCY_PATH"] = str(overlay)
    else:
        env.pop("OPENETA_LIBERO_MINK_DEPENDENCY_PATH", None)
    command = [sys.executable, "-m", "sim.mcp_server", "--host", args.host, "--port", str(args.port)]
    return command, env


def main():
    p = parser()
    args = p.parse_args()
    try:
        command, env = launch_spec(args)
    except RuntimeError as exc:
        p.error(str(exc))
    print(f"Dedicated simulator controller: {args.controller}", flush=True)
    os.chdir(REPO)
    # Replace this process so ordinary server signals/cleanup retain ownership.
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
