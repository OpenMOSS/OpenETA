# Pre-Host-change checkpoint — 2026-09-12

This snapshot preserves the complete current source in the isolated Codex plugin
worktree, including inherited harness changes and the Cartesian-segment controller.
It precedes local fixture contact patches, route batching and Host feedback changes.
No main-worktree edits or remote push are part of this checkpoint.

## Evidence boundaries

- The frozen Astra/high 40-task evaluation achieved 34/40 pass@1 and 36/40
  pass@2 (40 first attempts plus retries of the six failures). Its source manifest,
  original results and initial-state audit remain under
  `tmp/codex-libero40-pass2-20260911/`.
- The current controller was changed AFTER that evaluation. Its paired local
  replay reached 9/13 targets versus 7/13 with the original controller, preserving
  all seven original local successes. This is not a new full-task success rate.
- Original evaluated runtime files are preserved in
  `tmp/codex-trajectory-control-20260912/frozen-runtime.tar.gz`, verified against
  the evaluation source manifest. Runtime archives, images, model/session logs,
  virtual environments and credentials are not included in Git.
- See `codex-cartesian-segment-control-2026-09-12.md` for the controller design,
  exact validation and unresolved cases, and
  `codex-libero40-pass2-protocol-2026-09-11.md` for the historical evaluation.

Future Host changes and evaluations must use a new source snapshot and result
directory. Do not overwrite or resume the historical frozen campaign with new code.
