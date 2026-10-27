# Bounded native MLX evaluation

This increment follows the SFT/GRPO mechanics qualification. It compares the
existing pinned base and saved MLX adapters without submitting training.
The existing evaluator has no native MLX client, and model-first chat currently
requires `VLLMStartupSpec`; configuration cannot load an MLX adapter through
either boundary. Add a reusable generation-only worker and a host composition,
then reuse the evaluator's backend protocol, assertions, reports and comparison.
Keep task-specific correctness in configuration.

## Execution contract

The host binds clean pushed source, a complete private runtime capture/lock,
model/tokenizer snapshots, optional native adapter archive and complete ordered
cases. The base and each adapter are separate fresh attempts. A generation-only
admission signs the exact selection, attempt, finite process/token/file limits
and an expiry within one hour. It grants no training or public capability.
Use the existing `NativeOwnedProcess`, guardian and host-owned SQLite record port;
an occupied/uncertain state is never adopted or replayed. No environment secrets,
downloads, server, paid judge or provider effect is needed.

The isolated `-I -S -B` child uses the existing captured-import boundary. A
distinct evaluation request dispatches to generation code; legacy SFT/GRPO
request bytes and execution semantics remain intact. Validate all cases and
tokenization before model imports, preserve complete prompts, configured chat
arguments, exact IDs and final-target labels, and reject truncation or unsupported
attention layouts. Do not pass reference answers into generation. Record greedy
decode controls, generated IDs/text, stop versus token-limit finish reason and
prompt/completion counts. Optional reference loss uses causal shifting and
supervised-token normalization, with no optimizer or gradient construction.

The native model role is a bounded exact archive, not PEFT/GGUF. Verify its hash,
base model binding, adapter configuration, finite tensor inventory and loaded
values before generation. Verify snapshots and input bytes again afterward.
Write bounded complete per-case records and a final manifest into new private
output. A partial output is retained as incomplete, never scored as a full run.
Do not weaken existing training admission or infer serving authority from it.

## Cases and interpretation

Use the existing authentic conversation publications and declared validation
split. Verify the publication bytes and row/group identities; remove only the
declared final assistant target from the generation prompt. Preserve prior
conversation turns and keep references separate. Compare all models on identical
ordered cases and controls. Run the existing contamination checker and retain
its results before interpreting scores; do not silently change datasets.

Historical Nexus examples can measure fidelity to their declared old target
format. They do not establish current Nexus API compatibility. Any current-API
benchmark must bind reviewed current schemas and have separate scenario-defined
correctness. Do not execute generated commands against the user's vault.
Conversation reference loss and literal matches are not semantic quality scores.
Retain complete generations for review and report these limits explicitly.

Offline scoring consumes retained responses through the existing `BackendClient`
protocol and evaluator assertions. It performs no model execution, tool calls or
paid judging. Model-specific scaffold settings and dataset-specific score paths
remain in checked-in configuration; no runtime wrapper/manager assumptions.

## Acceptance

Meaningful CPU controls cover bounds, custody, no replay, target separation,
generation finish reasons, causal loss and exact retained-response scoring.
Independently review the final source before separately admitted Mac generation.
Actual base/adapter results must retain their source, data, model, adapter and
runtime identities. Package/CI and final evidence review precede integration.
Practical GRPO rewards follow these results; the existing native reward dialect
is bounded response-only regex V0, not environment-backed tool correctness.
Cancellation/caller-loss/reconciliation and full desktop distribution are later
increments.
