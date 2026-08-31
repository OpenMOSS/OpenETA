# Main feature integration: Group 2 XML quality gate

## Scope

Group 2 adopts the upstream XML decision wire format for the main planner while
preserving the existing cache-stable static/dynamic context layout,
ToolContract authority, isolated sub-agent JSON contracts, and provider
retry/failover/token-budget behavior.

The fixed comparison floor was the seed-0 LIBERO Object-10 baseline of 2/10.
The Group 1 integration gate reached 3/10.

## Protocol remediation

The first full Group 2 run exposed XML-to-request shape regressions rather than
provider or simulator failures:

- untyped `true`, `false`, and `null` values remained strings;
- a single SAM3 point object was not promoted to the required point array;
- some list containers arrived with recoverable missing closing tags.

The decoder now infers natural scalar values outside reserved text fields,
normalizes singleton point structures, repairs bounded unclosed list
containers, and gives the planner a compact XML grammar. Focused planner tests
and the full host test suite passed after the remediation.

## Evaluation evidence

| Run | Purpose | Objective result |
| --- | --- | ---: |
| `libero-object-10-main-group2-xml-20260830-r04` | Initial fixed-seed Object-10 gate | 0/10 |
| `libero-object-group2-xml-remediation-canary-20260831-r01` | Tasks 1, 3, and 6 after scalar/array normalization | 1/3 |
| `libero-object-10-main-group2-xml-remediation-20260831-r01` | Formal fixed-seed Object-10 rerun | 1/10 |

The formal rerun completed all ten jobs: one objective success, eight task
failures, and one `need_human`. Its XML/type validation failures were materially
lower than the initial run. The remaining observed failures were dominated by
physical execution: empty or unstable contact, Mink constraint termination,
carry-path collision, a carried object dropping, and placement/release not
finishing within the turn budget.

## Manual quality-gate decision

On 2026-08-31, the project owner manually approved Group 2 for promotion despite
the 1/10 formal sample. The decision treats the one-run success-rate decrease as
insufficient evidence that the XML decision format caused a capability
regression: the protocol defects were repaired, while the terminal failures
were in pre-existing stochastic contact, controller, and collision behavior.

This is an explicit human override of the numeric gate, not a reinterpretation
of the recorded result. Group 3 may proceed. Its cuRobo/collision changes are a
plausible treatment for some observed path failures, but that hypothesis must
be evaluated rather than assumed.

## Preserved integration boundaries

- Main planner decisions use XML; isolated VDM/advisor/reviewer responses stay
  on their existing JSON contracts.
- ToolContract request/receipt and bundle authority remain host-owned.
- Static prompt content remains before growing conversation history.
- Provider retry, failover, timeout, and token budgets are unchanged.
- Group 3 must preserve LIBERO/Mink and the evidence-first attachment and
  provenance contracts while adding BEHAVIOR/R1Pro/cuRobo support.
