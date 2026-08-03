# LIBERO evaluation

OpenETA evaluates each task from a fresh simulator environment, Operator
workspace, and Codex home. Every attempt fixes the task identity, simulator
seed, model, reasoning effort, Operator profile, image size, simulator horizon,
and tool contract in its manifest and episode contract.

## Plan

`libero_operator_coverage.py plan-matrix` enumerates every task in one LIBERO
suite for the requested ordered seeds. The default Operator profile is
`openeta-light`; pass `--profile` only when deliberately evaluating another
reviewed release.

~~~bash
PYTHONPATH=. uv run python scripts/embodied/libero_operator_coverage.py \
  plan-matrix \
  --output outputs/libero-spatial \
  --suite libero_spatial \
  --seeds 0 1 2 3 4 \
  --model MODEL \
  --reasoning-effort medium \
  --model-provider openai
~~~

## Execute

~~~bash
PYTHONPATH=. uv run python scripts/embodied/libero_operator_coverage.py \
  run --output outputs/libero-spatial --jobs 8 --base-port 14000
~~~

Each worker has unique simulator and Gateway ports. Calls inside one episode
remain serial. `--jobs` controls independent episode concurrency.

## Success and Pass@k

Task success comes only from the native LIBERO checker. For ordered seeds,
Pass@\(k\) counts a task once if any of its first \(k\) valid attempts succeeds.
`plan-first-success` and `run-first-success` may stop scheduling later seeds
after a success; this changes compute usage, not the metric.

## Infrastructure validity

Provider stream failures, launch failures, mismatched task or seed identity,
missing episode contracts, and unavailable required services are
infrastructure-invalid. They are retained and reported separately rather than
counted as task failures. Budget exhaustion, unsuccessful manipulation, and a
valid simulator termination without native success are task failures.

Complete episode traces, tool results, images, and contracts are retained under
the requested output and artifact roots.
