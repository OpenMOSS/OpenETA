# Memory store recovery and checkpoint boundary

Status: candidate implementation on the harness refactor branch. Applies to
`JsonMemoryStore` and `AgentMemory.resume_session`, not every JSONL writer in
the repository. Preserve the original files when investigating a failure.

## Authoritative working memory

`sessions/<id>/working/snapshot.json` is the single restore authority after the
first new-format commit. Its `openeta.working_memory_snapshot.v1` envelope contains
the session ID, monotonically increasing generation, journal cursors, checksum,
and all five memory maps plus compact summary in one JSON document:

```json
{
  "schema_version": "openeta.working_memory_snapshot.v1",
  "session_id": "example",
  "generation": 3,
  "journal_cursors": {"trace": 12, "conversation": 5},
  "memory": {
    "facts": {}, "agent_working_state": {}, "agent_artifacts": {},
    "artifacts": {}, "skill_notes": {}, "compact_summary": ""
  },
  "sha256": "<SHA-256 of the canonical envelope without this field>"
}
```

Publication writes a uniquely named temporary in the same directory, flushes
and fsyncs it, replaces the snapshot, then fsyncs the directory. Newly created
directory entries are also synced. A process exit on either side of replacement
exposes the old or new complete generation, not mixed namespace generations.
Temporary files left by process death are not promoted automatically.

The store serializes operations with per-session thread/file locks. A store
instance records the generation it loaded; a save against a newer generation
raises `MemoryStoreConflict` instead of overwriting another writer's snapshot.
Explicitly reload and reconcile the intended change; do not blindly retry an
old in-memory state. This is not a distributed execution lease: run one active
Agent runtime per session. Concurrent appenders are supported, but simultaneous
active planners/resumes are not authorized by these file locks.

Existing `facts.json`, `agent_working_state.json`, `agent_artifacts.json`,
`artifacts.json`, `skill_notes.json`, and `compact_summary.json` remain
compatibility inspection exports. They may lag or mix generations after a
crash; `load_working_memory()` never assembles a restore from them once the new
snapshot is managed. Export failures are reported as `snapshot_export_failed`
without pretending the committed checkpoint was rolled back. Consumers needing
consistent memory must use the store API or read the complete snapshot once.

Old sessions without the new format still load their legacy files. The first
new-format save writes `.snapshot-managed` before publication. If that first
commit is interrupted, legacy data is preserved, but the marker prevents an
automatic fallback that could revive old epoch-bound evidence. A missing,
malformed, wrong-session, wrong-schema, or checksum-invalid managed snapshot
fails closed. There is no automatic rollback to an earlier generation.

## Trace and conversation journals

New records retain their existing top-level content and add `_store_seq` and
`_store_sha256`. Sequence numbers count nonempty records, including an initial
legacy unsequenced prefix; after sequenced records begin, unsequenced records
are rejected. The checksum covers the whole canonical row except its checksum
field. Checksums detect accidental alteration, not malicious host tampering or
cryptographic authenticity. Both trace and canonical conversation use the same
reader and append policy.

- Valid legacy records are not rewritten. A complete final JSON object without
  a newline is preserved; the next append first inserts the missing separator.
- Only invalid JSON/UTF-8 in an **unterminated final line** is a recoverable torn
  tail. Read-only queries return the valid prefix and a logged recovery report;
  they do not change the file.
- Before resuming/appending, exact torn bytes and their offset/hash are saved
  under `recovery/`, and a durable `journal-recovery-required.json` fence is
  published. Only then is the torn suffix truncated. Valid prefix bytes remain
  unchanged. The quarantine and JSON report provide the retained original tail.
- Complete malformed lines, non-object records, checksum errors, missing or
  out-of-order sequences fail with `MemoryStoreCorruption`; none is silently
  skipped. Recovery validates both journals before modifying either.
- Appends flush/fsync complete records. An unchanged file-signature cache avoids
  rescanning the whole trace on each append; different store instances revalidate
  after observed file changes. `load_events(limit=...)` still validates all rows
  but retains only the requested suffix in memory.

Session-index replacement uses the same durable atomic-write primitive.
Malformed index contents fail explicitly rather than silently resetting the
session inventory. The index is not transactionally committed with a journal
append; its event count remains a discovery hint, not replay authority.

## Restored evidence is not automatically current evidence

Checkpoint journal cursors detect a completely lost committed suffix even when
the remaining JSONL is syntactically valid. A journal shorter than the checkpoint
is corruption and prevents loading/saving over the checkpoint. A journal ahead
of it produces `journal_ahead_of_snapshot`: complete newer records remain in
history, but the store does not claim that all their effects were checkpointed.

On resume after tail recovery or uncheckpointed journal records, AgentMemory
advances robot/object evidence epochs using source
`journal_recovery_invalidates_evidence`, retains historical records, and records
`session_journal_recovery` with `requires_fresh_environment_evidence=true`.
Previously feasible IK seeds then fail the existing exact-current-receipt gate.
This revokes stale evidence; it does not assert that a particular physical motion
occurred, choose a task stage, or authorize movement to gather a view.

The torn-tail fence remains sticky. Every later resume of a repaired session
invalidates historical physical evidence again, including after a process loss
between truncation and the epoch checkpoint. Do not delete the fence or managed
marker to force a resume. Full corruption requires investigation or a fresh
session with reacquired environment evidence, not a manual checksum/epoch edit.

## Limits and review

- Working-memory publication is atomic; a complete Agent turn spanning trace,
  conversation, snapshot, index, and remote motion is **not** one transaction.
  There is no general replay reducer that rebuilds every unsaved memory value
  from journal records. Post-checkpoint effects are reported, not invented.
- This does not provide exactly-once remote execution, a live-session lease,
  automatic repair of arbitrary middle corruption, index reconstruction, or
  power-cut/device-failure certification. POSIX filesystem `fsync`/rename
  semantics are required; cross-platform storage support is not verified.
- Legacy namespaces were never a transaction, and historical legacy records
  have no checksums. Their original consistency cannot be retroactively proven.
- New journal metadata and the checkpoint/inspection-export contract are
  runtime-local candidate changes; review them before shared integration.
  Section 5 Command/ToolResult/EnvObservation fields are not renamed here.

Validation lives in `tests/test_memory_store_recovery.py`: actual subprocess
exit around publication, namespace export failure, optimistic-write conflict,
concurrent readers/appenders, legacy migration, UTF-8/JSON tail quarantine,
middle corruption, sequence/checksum/cursor checks, persistent recovery fence,
and real IK gate rejection after recovery. All mutation tests use temporary
fixtures, not the stored Human VLM experiment sessions. Exact test results are
in the [refactor log](harness-refactor-progress-2026-09-05.md).
