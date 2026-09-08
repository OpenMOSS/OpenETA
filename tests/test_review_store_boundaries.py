from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import threading

import pytest

from agent.runtime.self_improvement import SkillReviewProposal, SkillReviewProposalStore


PROPOSAL = SkillReviewProposal("fixture", "patch", "pick", "fixture", "fixture guidance")


@pytest.mark.parametrize("identifier", ["../escaped", "nested/escaped", r"nested\escaped", "", "..", ".", " fixture ", "fixture.json", "x\x00y", "x\ny"])
def test_save_rejects_noncanonical_or_path_ids_before_creating_store(tmp_path, identifier):
    store = SkillReviewProposalStore(tmp_path / "pending")
    with pytest.raises(ValueError):
        store.save(replace(PROPOSAL, proposal_id=identifier))
    assert not store.root.exists()
    assert not (tmp_path / "escaped.json").exists()


@pytest.mark.parametrize("status", ["pending", "approved", "rejected"])
def test_save_never_overwrites_existing_proposal_or_resolution(tmp_path, status):
    store = SkillReviewProposalStore(tmp_path)
    path = store.save(PROPOSAL)
    if status == "approved":
        store.resolve_reviewed("fixture", reviewer="fixture", resolution={"approved": True})
    elif status == "rejected":
        store.reject("fixture", reason="fixture")
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        store.save(replace(PROPOSAL, rationale="replacement"))
    assert path.read_bytes() == before
    assert store.load("fixture.json")["status"] == status


def test_predictable_old_temporary_symlink_is_never_followed(tmp_path):
    store = SkillReviewProposalStore(tmp_path / "pending")
    store.root.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("unchanged fixture")
    trap = store.root / "fixture.json.tmp"
    trap.symlink_to(victim)
    store.save(PROPOSAL)
    assert victim.read_text() == "unchanged fixture"
    assert trap.is_symlink()


def test_load_rejects_embedded_identity_mismatch(tmp_path):
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps({**PROPOSAL.to_dict(), "proposal_id": "other"}))
    store = SkillReviewProposalStore(tmp_path)
    with pytest.raises(ValueError, match="identity"):
        store.load("fixture")
    assert store.list() == []


def test_stored_path_is_not_authoritative_and_symlink_records_are_not_loaded(tmp_path):
    store = SkillReviewProposalStore(tmp_path / "pending")
    store.root.mkdir()
    path = store.root / "fixture.json"
    path.write_text(json.dumps({**PROPOSAL.to_dict(), "path": "untrusted display path"}))
    assert store.load("fixture")["path"] == str(path)
    outside = tmp_path / "linked.json"
    outside.write_text(json.dumps(replace(PROPOSAL, proposal_id="linked").to_dict()))
    (store.root / "linked.json").symlink_to(outside)
    with pytest.raises(ValueError):
        store.load("linked")
    assert len(store.list()) == 1


def test_concurrent_store_instances_publish_one_complete_winner_without_clobber(tmp_path):
    barrier = threading.Barrier(2)

    def save(index):
        store = SkillReviewProposalStore(tmp_path)
        barrier.wait(timeout=2)
        try:
            store.save(replace(PROPOSAL, rationale=f"writer-{index}"))
            return "saved"
        except FileExistsError:
            return "exists"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(save, [1, 2]))
    assert sorted(results) == ["exists", "saved"]
    payload = SkillReviewProposalStore(tmp_path).load("fixture")
    assert payload["rationale"] in {"writer-1", "writer-2"}
    assert [p.name for p in tmp_path.iterdir()] == ["fixture.json"]


def test_repeat_prepared_commit_cannot_reapply_or_reset_existing_record(tmp_path):
    from types import SimpleNamespace
    from agent.runtime.episode import EpisodeResult
    from agent.runtime.self_improvement import SelfImprovementConfig, SelfImprovementReviewer
    from agent.runtime.skills import SkillRegistry

    applied = []
    reviewer = SelfImprovementReviewer(
        config=SelfImprovementConfig(min_tool_calls=0, proposal_root=tmp_path, auto_apply_reviewed=True),
        subagent=SimpleNamespace(review=lambda context: [PROPOSAL]),
        auto_applier=SimpleNamespace(apply=lambda *args, **kwargs: applied.append(1) or {"applied": True}),
    )
    skills = SkillRegistry()
    plan = reviewer.prepare_review(EpisodeResult("fixture", "fixture"), skills=skills)
    reviewer.commit_review(plan, skills=skills)
    before = (tmp_path / "fixture.json").read_bytes()
    with pytest.raises(FileExistsError):
        reviewer.commit_review(plan, skills=skills)
    assert applied == [1]
    assert (tmp_path / "fixture.json").read_bytes() == before
    assert reviewer.store.load("fixture")["status"] == "approved"


def test_failed_atomic_publication_removes_only_owned_temporary_file(tmp_path, monkeypatch):
    from agent.runtime import self_improvement as module
    unrelated = tmp_path / "unrelated.tmp"
    unrelated.write_text("keep fixture")

    def fail_link(*args):
        raise OSError("injected publication failure")

    monkeypatch.setattr(module.os, "link", fail_link)
    with pytest.raises(OSError, match="injected"):
        SkillReviewProposalStore(tmp_path).save(PROPOSAL)
    assert list(tmp_path.iterdir()) == [unrelated]
    assert unrelated.read_text() == "keep fixture"
