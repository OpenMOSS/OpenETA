from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from sim.envs.univtac.isaac51_runtime_validation import (
    build_collect_command,
    environment_spec_sha256,
    inspect_fresh_output_root,
    summarize_collection_gate,
    validate_collect_command,
    validate_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/univtac/isaac51_runtime_validation.yaml"


def _config():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _gate(gate_id: str):
    return next(gate for gate in validate_config(_config()) if gate.gate_id == gate_id)


def test_revised_gate_seeds_labels_and_output_directories_are_frozen() -> None:
    gates = validate_config(_config())
    assert [(gate.gate_id, gate.task, gate.seed, gate.label, gate.output_dir) for gate in gates] == [
        ("G0", "grasp_classify", 0, "official_isaac51_phase1_collection_smoke", "grasp_classify_seed0"),
        ("L0", "lift_can", 1_000_000, "legacy_ftp1_eval_seed_index_aligned", "lift_can_seed1000000"),
        ("C0", "pull_out_key", 1_000_000, "legacy_ftp1_eval_seed_index_aligned", "pull_out_key_seed1000000"),
        ("H0", "insert_hole", 1_000_000, "legacy_ftp1_eval_seed_index_aligned", "insert_hole_seed1000000"),
    ]


def test_all_collection_commands_pin_start_and_max_to_same_seed(tmp_path: Path) -> None:
    for gate in validate_config(_config()):
        command = build_collect_command(
            python=Path("/env/bin/python"),
            source_root=Path("/source"),
            gate=gate,
            collection_root=tmp_path / gate.output_dir,
            gpu="0",
        )
        validate_collect_command(command, gate)
        assert command[command.index("--start_seed") + 1] == str(gate.seed)
        assert command[command.index("--max_seed") + 1] == str(gate.seed)


def test_next_seed_or_missing_seed_flag_is_rejected() -> None:
    gate = _gate("C0")
    command = build_collect_command(
        python=Path("python"),
        source_root=Path("source"),
        gate=gate,
        collection_root=Path("out"),
        gpu="0",
    )
    command[command.index("--max_seed") + 1] = str(gate.seed + 1)
    with pytest.raises(ValueError, match="max_seed"):
        validate_collect_command(command, gate)
    command = [token for token in command if token != "--start_seed"]
    with pytest.raises(ValueError, match="start_seed"):
        validate_collect_command(command, gate)


def test_fresh_output_rejects_suc_map(tmp_path: Path) -> None:
    root = tmp_path / "collection"
    assert inspect_fresh_output_root(root)["fresh_output_root"] is True
    root.mkdir()
    (root / "suc_map.txt").write_text("1", encoding="utf-8")
    result = inspect_fresh_output_root(root)
    assert result["fresh_output_root"] is False
    assert result["preexisting_suc_map"] is True


def test_observed_seed_contract_detects_unexpected_next_seed(tmp_path: Path) -> None:
    gate = _gate("L0")
    freshness = inspect_fresh_output_root(tmp_path / "new")
    process = {"returncode": 0, "timed_out": False, "cleanup_complete": True}
    result = summarize_collection_gate(
        gate=gate,
        collection_root=tmp_path / "new",
        log_text="Starting from seed 1000000.\n[0] Seed 1000000 failed\n[0] Seed 1000001 failed\n",
        process_result=process,
        freshness=freshness,
    )
    assert result["observed_attempted_seeds"] == [1_000_000, 1_000_001]
    assert result["unexpected_seeds"] == [1_000_001]
    assert result["seed_contract_valid"] is False
    assert result["classification"] == "seed_contract_violation"


def test_startup_failure_is_not_mislabeled_as_seed_contract_violation(tmp_path: Path) -> None:
    gate = _gate("G0")
    result = summarize_collection_gate(
        gate=gate,
        collection_root=tmp_path / "new",
        log_text="UIPC device error before task construction",
        process_result={"returncode": 1, "timed_out": False, "cleanup_complete": True},
        freshness=inspect_fresh_output_root(tmp_path / "new"),
    )
    assert result["seed_contract_valid"] is False
    assert result["seed_contract_violation"] is False
    assert result["classification"] == "runtime_error"


def test_environment_spec_hash_is_deterministic_and_semantic() -> None:
    config = _config()
    assert environment_spec_sha256(config) == environment_spec_sha256(copy.deepcopy(config))
    changed = copy.deepcopy(config)
    changed["environment"]["torch_cuda"] = "12.8"
    assert environment_spec_sha256(config) != environment_spec_sha256(changed)


def test_config_rejects_seed_or_semantic_label_drift() -> None:
    changed = _config()
    changed["gates"][1]["seed"] = 0
    with pytest.raises(ValueError, match="protocol contract changed"):
        validate_config(changed)
    changed = _config()
    changed["gates"][2]["label"] = "benchmark_parity"
    with pytest.raises(ValueError, match="protocol contract changed"):
        validate_config(changed)


def test_runtime_config_has_no_machine_absolute_paths() -> None:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    assert "/home/" not in text
    assert "/tmp/" not in text
