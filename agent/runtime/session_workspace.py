"""Session-owned filesystem layout for parallel agent runs."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from adapter.protocol import JsonDict
from agent.runtime.artifact_paths import safe_artifact_component
from agent.runtime.calibration import calibration_profile_sha256
from agent.runtime.calibration_registry import require_grasp_calibration_profile
from agent.runtime.memory_store import JsonMemoryStore
from agent.runtime.skills import BUILTIN_SKILL_DIR, SkillRegistry, load_skill_markdown
from agent.runtime.task_playbooks import DEFAULT_TASK_PLAYBOOK_ROOT
from agent.tools.grasp_strategies import DEFAULT_GRASP_STRATEGY_ROOT
from agent.tools.grasp_strategies import grasp_strategy_tree_sha256


DEFAULT_MEMORY_ROOT = Path(".openeta_memory")
DEFAULT_SESSION_ROOT = DEFAULT_MEMORY_ROOT / "sessions"
DEFAULT_SESSION_WORKSPACE_ROOT = DEFAULT_SESSION_ROOT
LEGACY_SESSION_WORKSPACE_ROOT = DEFAULT_MEMORY_ROOT / "workspaces"


@dataclass(frozen=True, slots=True)
class SessionWorkspace:
    """One canonical session tree plus its shared memory-store root."""

    session_id: str
    root: Path
    skills_dir: Path
    memory_root: Path
    working_dir: Path
    artifacts_dir: Path
    sandbox_dir: Path
    tools_dir: Path
    calibrations_dir: Path
    strategies_dir: Path
    task_playbooks_dir: Path

    @classmethod
    def create(
        cls,
        session_id: str,
        *,
        root: str | Path = DEFAULT_MEMORY_ROOT,
        source_skills: str | Path = BUILTIN_SKILL_DIR,
        source_grasp_profile: str | Path | None = None,
        source_grasp_strategies: str | Path = DEFAULT_GRASP_STRATEGY_ROOT,
        source_task_playbooks: str | Path = DEFAULT_TASK_PLAYBOOK_ROOT,
        environment_id: str = "",
        embodiment_fingerprint: JsonDict | None = None,
    ) -> "SessionWorkspace":
        safe_session = safe_artifact_component(session_id, fallback="session")
        memory_root = Path(root)
        workspace_root = memory_root / "sessions" / safe_session
        workspace = cls(
            session_id=session_id,
            root=workspace_root,
            skills_dir=workspace_root / "skills",
            memory_root=memory_root,
            working_dir=workspace_root / "working",
            artifacts_dir=workspace_root / "artifacts",
            sandbox_dir=workspace_root / "sandbox",
            tools_dir=workspace_root / "tools",
            calibrations_dir=workspace_root / "calibrations",
            strategies_dir=workspace_root / "strategies",
            task_playbooks_dir=workspace_root / "task_playbooks",
        )
        for directory in (
            workspace.root,
            workspace.skills_dir,
            workspace.memory_root,
            workspace.working_dir,
            workspace.artifacts_dir,
            workspace.sandbox_dir,
            workspace.tools_dir,
            workspace.calibrations_dir,
            workspace.grasp_strategy_root,
            workspace.task_playbooks_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        if not any(workspace.skills_dir.glob("*.md")):
            for source in sorted(Path(source_skills).glob("*.md")):
                shutil.copy2(source, workspace.skills_dir / source.name)
        if not workspace.grasp_profile_path.exists():
            selected_profile = (
                Path(source_grasp_profile)
                if source_grasp_profile is not None
                else require_grasp_calibration_profile(
                    environment_id=environment_id,
                    fingerprint=embodiment_fingerprint,
                )
            )
            shutil.copy2(selected_profile, workspace.grasp_profile_path)
        workspace.grasp_profile_path.chmod(0o400)
        if not any(workspace.grasp_strategy_root.rglob("*.json")):
            for status in ("candidate", "validated"):
                source_dir = Path(source_grasp_strategies) / status
                target_dir = workspace.grasp_strategy_root / status
                target_dir.mkdir(parents=True, exist_ok=True)
                for source in sorted(source_dir.glob("*.json")):
                    shutil.copy2(source, target_dir / source.name)
        if not any(workspace.task_playbooks_dir.rglob("*.json")):
            for status in ("candidate", "validated"):
                source_dir = Path(source_task_playbooks) / status
                target_dir = workspace.task_playbooks_dir / status
                target_dir.mkdir(parents=True, exist_ok=True)
                if source_dir.exists():
                    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
        return workspace

    @property
    def grasp_profile_path(self) -> Path:
        return self.tools_dir / "grasp_profile.json"

    @property
    def memory_dir(self) -> Path:
        """Compatibility alias for the shared JsonMemoryStore root."""

        return self.memory_root

    @property
    def grasp_strategy_root(self) -> Path:
        return self.strategies_dir / "grasp"

    @property
    def grasp_profile_sha256(self) -> str:
        return calibration_profile_sha256(self.grasp_profile())

    @property
    def grasp_profile_id(self) -> str:
        return str(self.grasp_profile().get("calibration_id") or "")

    def grasp_profile(self) -> JsonDict:
        payload = json.loads(self.grasp_profile_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("grasp profile must contain one JSON object")
        return payload

    @property
    def grasp_strategy_tree_sha256(self) -> str:
        return grasp_strategy_tree_sha256(self.grasp_strategy_root)

    def to_dict(self) -> JsonDict:
        return {
            "session_id": self.session_id,
            "root": str(self.root),
            "skills_dir": str(self.skills_dir),
            "memory_root": str(self.memory_root),
            "memory_dir": str(self.memory_root),
            "working_dir": str(self.working_dir),
            "artifacts_dir": str(self.artifacts_dir),
            "sandbox_dir": str(self.sandbox_dir),
            "tools_dir": str(self.tools_dir),
            "calibrations_dir": str(self.calibrations_dir),
            "strategies_dir": str(self.strategies_dir),
            "grasp_profile_path": str(self.grasp_profile_path),
            "grasp_profile_sha256": self.grasp_profile_sha256,
            "grasp_profile_id": self.grasp_profile_id,
            "grasp_strategy_root": str(self.grasp_strategy_root),
            "grasp_strategy_tree_sha256": self.grasp_strategy_tree_sha256,
            "task_playbook_root": str(self.task_playbooks_dir),
        }

    def import_legacy_roots(
        self,
        *,
        memory_root: str | Path = "",
        artifact_root: str | Path = "",
    ) -> None:
        """Migrate pre-workspace paused state into this session-owned layout."""

        if str(memory_root):
            source = Path(memory_root)
            if source.exists() and source.resolve() != self.memory_root.resolve():
                source_store = JsonMemoryStore(root=source)
                if source_store.session_exists(self.session_id):
                    self.import_memory_session(
                        self.session_id,
                        source_root=source,
                    )
                elif (source / "trace.jsonl").exists():
                    JsonMemoryStore(root=self.memory_root).start_session(
                        session_id=self.session_id,
                        task="",
                        metadata={"migrated_from_layout": str(source)},
                    )
                    shutil.copytree(source, self.root, dirs_exist_ok=True)

        if str(artifact_root):
            source = Path(artifact_root)
            if source.exists() and source.resolve() != self.artifacts_dir.resolve():
                shutil.copytree(source, self.artifacts_dir, dirs_exist_ok=True)

    def import_memory_session(
        self,
        session_id: str,
        *,
        source_root: str | Path,
    ) -> None:
        """Copy one legacy JSON-memory session into this workspace."""

        source_store = JsonMemoryStore(root=source_root)
        if not source_store.session_exists(session_id):
            raise ValueError(f"Unknown source memory session: {session_id}")
        target_store = JsonMemoryStore(root=self.memory_root)
        source_session_dir = source_store.session_dir(session_id)
        target_session_dir = target_store.session_dir(session_id)
        if source_session_dir.resolve() == target_session_dir.resolve():
            return
        metadata = source_store.load_session_metadata(session_id)
        task = str(metadata.get("task") or "")
        session_metadata = metadata.get("metadata")
        target_store.start_session(
            session_id=session_id,
            task=task,
            metadata=session_metadata if isinstance(session_metadata, dict) else {},
        )
        shutil.copytree(
            source_session_dir,
            target_session_dir,
            dirs_exist_ok=True,
        )

    def skill_registry(self) -> SkillRegistry:
        registry = SkillRegistry()
        for path in sorted(self.skills_dir.glob("*.md")):
            registry.register(load_skill_markdown(path))
        return registry
