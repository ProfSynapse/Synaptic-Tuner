# Restricted native MLX evaluation

Implementation candidate; no live inference/quality qualification is claimed.
The existing evaluator registry and `VLLMStartupSpec` cannot load native MLX
adapters through configuration. The native evaluation example supplies a
generation-only host/worker composition, reusing the existing native guardian,
captured imports, snapshots, SQLite record port and evaluator backend protocol.
It does not submit training, start a server, execute tools or call a paid judge.

Use the checked-in `examples/training_qualification/native_evaluation_cases.py`
on reviewed private immutable conversation publications. The preparation config
binds exact semantic/content hashes, expected train/validation counts, labels,
literal assertions and generic JSONPath values selected from references.
It preserves the declared validation split and complete conversation prefix,
separates the final target, and writes bounded generation cases, scoring cases,
source evidence and train/evaluation text for contamination checks. An occupied
output refuses. Do not change datasets silently to clear contamination reports.

`native_evaluation.py inspect <config>` observes a clean pushed source, captured
runtime/lock, pinned private model/tokenizer, optional exact native model-role
archive, cases and finite limits. A named profile declares exact model/tokenizer
compatibility and generation bounds. Model-specific chat arguments, EOS IDs and
score paths belong in config, never launcher switches or parser assumptions.

`admission-candidate <config> --expires-at <UTC>` renders an unsigned evaluation
selection. It grants no authority. Independently review source and the complete
selection, sign the canonical candidate outside the execution host, retain the
signature and verify only with an externally supplied public-key pin. Use
`execute <config> --trusted-root-key-sha256 <pin>` for the one newly admitted
attempt. Start expiry must be within one hour of the execution-host clock.
The private state and process claim are create-only. Preserve every failed or
uncertain attempt; never adopt its state or replay it. Keep original keys private.
No existing training admission supplies inference authority.

The isolated child requests list IDs explicitly, validates complete prompts and
reference masks before model imports, loads only verified local snapshots and
optional saved native LoRA tensors, and performs greedy generation without an
optimizer. It records exact IDs/text, natural stop versus token limit and token
counts. Optional reference loss is shifted, masked and normalized by supervised
tokens; references never enter generation. Bounded case files plus a final
manifest establish completeness; partial output is not a complete evaluation.
Private responses are ordinary retained artifacts, not diagnostic log text.
Closed exits 150–157 identify evaluation phases only, never root causes.

`readback <config> --trusted-root-key-sha256 <pin>` uses original authority,
even after start expiry; it never loads a model or renews start permission.
`export <config> --trusted-root-key-sha256 <pin> --destination <new-private-root>`
copies complete bounded cases/results, admission/signature/public key and the
host receipt. Transfer it unchanged to the existing CPU evaluator environment;
no evaluator packages are added to the pinned Mac runtime. Its observed manifest
digest binds the scoring config. Portable integrity/admission checks do not
independently authenticate model execution; preserve the stated operator witness.
`native_evaluation_cases.py score <config> --trusted-root-key-sha256 <pin>`
validates complete retained evidence, joins exact prompts/case IDs and invokes
the existing evaluator's assertions/reporting with an explicit validator config.
`NativeMLXRetainedClient` loads no model and authenticates no execution by itself.
The existing `Evaluator.compare` consumes these reports and their comparison
bases. Backend policy identity binds stop/context/tokenizer/source/runtime
controls in addition to the standard generation fields.

Report historical target-format fidelity separately from current API behavior.
Never run generated commands on the user's vault to score them. A nonempty
conversation, natural completion, reference loss or literal target match is not
a semantic quality score. Retain complete generations and failure/length counts,
run contamination checks, and declare the finite small-sample scope before
interpreting changes. Use result evidence to select future practical GRPO
rewards; existing native regex-V0 rewards do not execute environment tool tasks.
