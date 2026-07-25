# OpenETA Project Memory

This directory is reserved for curated project memory that should travel with
the repository.

Do not write automatic session traces here. Runtime session events and local
working memory live under `.openeta_memory/sessions/<session_id>/`, which is
ignored by git.

Older local stores may contain `.openeta_memory/sessions/<session_id>.jsonl`
and `.openeta_memory/working/`. `JsonMemoryStore` migrates old trace files into
per-session directories and archives the old global working directory under
`.openeta_memory/legacy/working/`; that archived working memory is not
automatically loaded into any new session.

Use this directory only for promoted memory: concise facts, design decisions,
or project-specific guidance that an agent has summarized and a developer has
accepted as useful across machines and branches.

Promotion is explicit. Session-local working memory is useful only inside the
session that produced it unless a developer reviews and promotes it here. Use
the CLI after reviewing working memory:

```bash
/memory facts --json
/promote-memory facts target --target project_memory.md --note "reviewed target fact"
```

Supported promoted-memory files:

- `project_memory.md`: reviewed long-term project notes.
- `tool_lessons.md`: reusable notes about tool behavior and failure modes.
- `skill_lessons.md`: reusable notes that should inform skill updates.
