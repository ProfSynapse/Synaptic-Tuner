# Lifecycle and diagnostics

Context: read before execution and whenever a public outcome is not verified.

## Key idea

Lifecycle states and typed diagnostic codes are the safe control plane. They
carry enough information to choose the next action without leaking parser prose,
source content, metadata values, or host paths.

## CLI outcomes

| Shape | Meaning | Next action |
| --- | --- | --- |
| `success: true`, `status: verified` | Explicit verification succeeded. | Verify checkpoint and hand off. |
| `status: blocked`, `error_code: preflight_blocked` | Plan cannot start. | Resolve reported diagnostic codes. |
| `error_code: invalid_input` | CLI config or selection syntax is invalid. | Validate config/invocation. |
| `error_code: bootstrap_failed` | Project/bootstrap composition failed. | Correct environment without exposing details. |
| `error_code: ingestion_failed` | Closed unexpected failure. | Preserve evidence; do not print exception text. |

Selection admission failures expose one of the closed operation codes below.
Treat the code, not hidden exception text, as the public diagnosis. A genuine
authority/request eligibility mismatch remains `admission_ineligible`.

The five selection-specific admission codes below predate the versioned null
profile and are intentionally retained for both CLI schema versions. Profile
compatibility does not collapse them back into `admission_ineligible`.

| Admission operation code | Meaning | Safe response |
| --- | --- | --- |
| `invalid_selection` | The retained local selection is structurally invalid. | Correct the declarative selection and authorize it again. |
| `authority_unavailable` | The one-use selection authority is missing, expired, or already consumed. | Obtain a fresh authorization; do not replay the old reference. |
| `source_unsafe` | A selected root or discovered entry failed link, reparse, type, or path-safety checks. | Narrow or correct the selection without weakening safety checks. |
| `source_changed` | Source identity changed while the immutable snapshot was being admitted. | Readmit the current authorized bytes and create a new plan. |
| `limit_exceeded` | The selection exceeded a public member, aggregate, path, pattern, or discovery bound. | Narrow the selection or, when within the documented hard ceilings, authorize it again with appropriate v2 admission budgets. |
| `admission_ineligible` | The source kind, retained authority, or request binding is not eligible for admission. | Correct the request/authority binding; do not reinterpret it as a source defect. |

## Public states

`planned`, `running`, `succeeded`, `failed`, `reconcile_required`,
`cancel_requested`, and `cancelled` are closed states. Only succeeded, failed,
and cancelled are terminal. `result` requires terminal state; `verify` requires
succeeded state.

Resume is narrow: only `failed` plus `interrupted` is eligible. Reconciliation
is narrow: only `reconcile_required` is eligible. Neither operation authorizes a
new snapshot, plan, publish, or cleanup.

## Diagnostic actions

| Diagnostic | Safe response |
| --- | --- |
| `unmatched_source` | Adjust discovery or binding config, or exclude unauthorized material. |
| `ambiguous_source` | Make bindings unambiguous. |
| `source_changed` | Readmit and replan current authorized bytes. |
| `structure_invalid` | Correct the declarative structure. |
| `parse_failed` | Correct config, or request separate authority to fix malformed source. |
| `metadata_invalid` | Correct declared mapping/type or separately authorized source. |
| `relationship_invalid` | Correct declarations; V1 recipe normally has no relationships. |
| `output_conflict` | Choose a new authorized output/identity; do not overwrite. |
| `interrupted` | Resume only through the eligible public transition. |
| `effect_uncertain` | Reconcile the exact retained outcome. |
| `execution_failed` | Preserve aggregate evidence and investigate composition. |
| `bundle_invalid` | Reject handoff; never patch the bundle in place. |

## Privacy rule

Default reports contain only status, closed codes, aggregate counts, references,
and digests. A useful diagnosis does not require revealing which private note
failed in shared logs. Inspect private data only inside the authorized local
boundary and never copy it into lifecycle artifacts.
