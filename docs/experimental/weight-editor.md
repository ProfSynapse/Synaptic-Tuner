# Weight-editor broker and recorder (experimental)

`synaptic_tuner.experimental.weight_editor` is a model-independent, CPU-only
broker and append-only recorder for agent-led weight-editing research. It is
**experimental and not a Tuner API v1 promise**: nothing here is listed in the
v1 support matrix, and the stable API is unchanged. The core has no dataset
fields, no model names and no provider code. A study (for example the EHR
research) owns its scientific policy and composes the real model adapter and
executor outside this package.

The state machine is `registered -> discovery -> final_patch_frozen ->
independent_evaluation -> closed`. The editor can only move through the first
three; the last two are host actions (`advance`). There is no transition back
to discovery, and the core implements no scientific verdict.

## What it guarantees, and what it does not

Enforced by the host broker, independent of model compliance:

- Requests are admitted only for operations in the config, with arguments that
  match the declared schema. Objects are opaque ids resolved against a
  hash-frozen capsule of development objects. Paths, URLs, commands, imports,
  policy, budget and split fields in a request are denied before any
  reservation or executor call.
- Unknown, withheld (held-out) and wrong-kind ids are denied. The model sees one
  generic message for unknown and withheld ids; the audit trail records which.
- Writes go only to `artifacts/<dispatch-id>/` through a create-only writer.
  A model cannot name a capsule object as a write target. The output root must
  not overlap the capsule or any `read_only_roots`.
- Every dispatch is written ahead (`dispatch_intent`, fsynced) before the
  executor runs. After a crash the broker asks the executor to reconcile by
  dispatch id: `complete` (receipt recovered), `unstarted` (executor attests
  with the dispatch id and a basis; the broker redispatches once), or `unknown`
  (run halts `reconciliation_needed`; the reservation stays charged). Nothing is
  re-executed implicitly, and an executor exception after dispatch is treated
  as unknown.
- Resource limits are optional and owned by the caller. Reservations,
  commitments and denied attempts are folded from the journal, so a resume
  never resets a charge. Charge on completion is `max(reserved, reported)`.
  Negative, NaN and infinite amounts are rejected.
  Limits without a configured operation-cost meter or built-in counter are
  refused. Provider usage is recorded as evidence; it is not an implicit billing
  meter. `dispatch_attempts` is checked before initial dispatch and redispatch.
- The run identity binds config bytes, capsule manifest and object bytes, the
  requested model block, adapter and executor identity (class, source hash,
  declared identity and Python package sources), the broker package sources,
  and a caller-supplied binding. External binaries, dependencies and non-Python
  resources must be pinned by the port identity or caller binding. A resume under any change
  is refused, without calling the adapter or executor, and the refusal is
  journaled.
- A single writer holds an `flock` on the run for the process lifetime; the
  kernel releases it on SIGKILL.
- `provider/events.jsonl` keeps every raw provider event byte-for-byte
  (base64) in a hash chain. `journal/events.jsonl` is the normalized hash-linked
  journal; unknown or unparseable events are kept and mark capture incomplete.
  Every record is fsynced before the next effect.
  `TransportEvent` additionally retains exact `client_input` and `client_stderr`
  bytes on named streams. These streams cannot issue requests or choices.
  Independent pipes are ordered as observed, without claiming a physical total
  order. Diagnostics containing invalid UTF-8 remain available as base64.
- Success needs a validated `final_choice`; final assistant text is never
  success. A run that stops early is `incomplete` and has no terminal manifest.
  Each clean turn needs returned-model evidence; model fields on any normalized
  event are checked. Failed, interrupted, unknown and truncated turns have their
  requests and choices denied before dispatch. An explicit resume may continue
  the conversation, but capture remains unqualified for the lifetime of that run.
  Patch selection requires its validated receipt to have been delivered in an
  earlier model turn. No-session retries and handoffs resend the instructions
  and catalog in the journaled model input.

Not guaranteed:

- **No OS isolation.** The writer is a cooperative API. An adapter or executor
  that touches the filesystem or network itself is outside the guarantee, and
  `declared_isolation` is only what the port says about itself. A normal
  subagent or CLI session is not an isolated environment.
- **Hidden reasoning is neither captured nor claimed.** Only provider-exposed
  events and summaries are retained.
- An internally chained directory detects inconsistent edits, not a malicious
  writer who rewrites the chain and sidecars together. Boundary truncation of an
  incomplete run is also undetectable without an independent anchor. `verify`
  explicitly reports its integrity scope. It does not reparse raw provider bytes
  using a dynamically loaded adapter. At freeze, use `anchor` to retain the final
  selection, terminal, binding and chain hashes outside the run output tree.
  The host must place that hash-only anchor under independent custody (for
  example, a Git commitment) before evaluation. A second mutable file owned by
  the same process provides no protection against an attacker rewriting both.
  `advance` requires the external anchor and verifies all receipts and artifacts.
- Transcript replay reconstructs what was observed. It never re-asks a model.

## Quick start

```bash
# deterministic synthetic smoke with fake ports (no model, no GPU, no network)
python -m synaptic_tuner.experimental.weight_editor smoke --workdir scratch/editor-smoke

python -m synaptic_tuner.experimental.weight_editor validate --config C --ports MOD:FACTORY [--ports-arg K=V ...]
python -m synaptic_tuner.experimental.weight_editor run      --config C --run-id ID --ports ... [--binding-file B.json]
python -m synaptic_tuner.experimental.weight_editor resume   --config C --run-id ID --ports ... [--recover-partial-tail]
python -m synaptic_tuner.experimental.weight_editor verify   --run-dir DIR      # alias: status
python -m synaptic_tuner.experimental.weight_editor anchor   --run-dir DIR --output EXTERNAL_FREEZE.json
python -m synaptic_tuner.experimental.weight_editor verify   --run-dir DIR --anchor-file EXTERNAL_FREEZE.json
python -m synaptic_tuner.experimental.weight_editor replay   --run-dir DIR [--output F] [--force]
python -m synaptic_tuner.experimental.weight_editor advance  --run-dir DIR --to independent_evaluation|closed --reference TEXT --anchor-file EXTERNAL_FREEZE.json
```

Exit codes: `0` ok/complete, `1` refused or invalid input, `3` incomplete with
reconciliation needed, `4` incomplete for another reason (adapter failure,
budget exhausted, model drift, no final choice, capsule drift), `5`
verification or integrity failure, `6` editor completed with unqualified capture.
`capture_qualification.qualified` is distinct from the editor completion status;
evaluation admission and synthetic smoke require qualified capture.

`validate` creates no run and makes no executor or model call. `run` registers a
new run and never overwrites an existing one. `--ports module:factory` is
chosen by the operator on the command line; it is never read from the config or
from a model request. The factory receives `PortContext(args, model)` and
returns `PortSet(adapter, executor)`.

## Config (`weight_editor_config/v1`, strict JSON)

Unknown keys, duplicate keys and `NaN`/`Infinity` are errors.

```json
{
  "schema": "weight_editor_config/v1",
  "model": {"provider": "…", "model": "…", "settings": {"…": "…"}, "allowed_returned": []},
  "capsule": {"manifest": "capsule/manifest.json"},
  "instructions": [{"role": "system", "object": "instr-main"}, {"role": "user", "text": "…"}],
  "operations": {
    "query_development": {
      "description": "…", "selectable": false, "required": ["items"],
      "arguments": {"items": {"type": "ref_list", "kind": "dev_item", "max_items": 4}},
      "cost": {"gpu_units": {"len_of": "items", "per_item": 0.5}}
    }
  },
  "output_root": "runs",
  "read_only_roots": [],
  "resources": {"limits": {"gpu_units": 2}},
  "final_choice": {"allow_no_edit": true, "max_rationale_chars": 20000},
  "withheld_id_digests": ["<sha256 of an id the model must never reach>"]
}
```

- `model` is required, with no default or fallback. The adapter's requested
  identity must equal it exactly. `allowed_returned` lists provider aliases
  accepted as the returned model; any other returned model halts the run as
  `model_drift`.
- Argument types: `ref`, `ref_list` (capsule object ids of one `kind`), `int`,
  `number` (finite, bounded), `enum`, `text` (bounded payload such as a patch;
  never an authority), `artifact_name` (a fresh bare name). Argument names that
  denote host authority (`path`, `command`, `import`, `url`, `policy`,
  `budget`, `target`, …) are rejected at config load.
- `cost` values are constants, `{"from_arg": name}` or
  `{"len_of": name, "per_item": n}`. Each admitted request also reserves one
  `dispatches`. `resources.limits` is optional; **no default money, token or
  time cap is invented**. Built-in meters `requests`, `model_turns` and
  `dispatches` and `dispatch_attempts` can be limited too; every request attempt, including denied or malformed
  requests, is counted.
- `selectable: true` marks operations whose `ok` receipts may be chosen by
  `final_choice`.

## Capsule (`weight_editor_capsule/v1`)

```json
{"schema": "weight_editor_capsule/v1", "capsule_id": "…",
 "objects": [{"id": "w.layer0", "kind": "tensor", "path": "tensors/w.layer0.json",
              "sha256": "…", "split": "development"}]}
```

Every object must have `split: "development"`; anything else is rejected at
load. Paths must be relative, free of `..`, and resolve (including through
symlinks) inside the manifest's directory. Object bytes are re-verified at
registration, resume, and before every dispatch.

## Ports

`ModelAdapter`: `identity()` (requested `provider`/`model`/`settings`, optional
`client_version`, `capabilities.raw_events` must be `true`,
`captures_hidden_reasoning` must not be),
`run_turn(TurnInput) -> Iterator[bytes | TransportEvent]`
(exact raw events), `parse_event(bytes) -> {"kind": …}` with kinds `message`,
`reasoning_summary`, `request`, `final_choice`, `session`, `usage`, `finish`,
`handoff`, `truncation`, `unknown`. Anything else, or any parse error, is stored
as `unknown` with its raw bytes. Adapter exceptions are recorded as
`cancelled`, `timeout` or `provider_error` and halt the run; a later explicit
`resume` starts a fresh turn marked `resumed`.
Only a clean turn can authorize requests or choices. A provider that reports
model identity in usage or finish packets must include that actual `model` in
the normalized event. An adapter must not copy the requested name and call it
returned evidence. Missing identity prevents dispatch. Unknown/truncated traffic
halts before dispatch; raw bytes remain available and capture is unqualified.

`Executor`: `identity()`, `supported_operations()` (a configured operation
outside it is rejected as `unsupported_capability` at validation, as is the case
for `analyze` in the fake executor), side-effect-free `validate()`,
`execute(Dispatch, BoundedWriter) -> Receipt`, and `reconcile(Dispatch) ->
ReconcileResult`. `reconcile` must never cause an effect. Raise
`ExecutorRejection` from `execute` only when no effect began.
`Dispatch.bound` carries resolved capsule paths for the executor only; they are
not journaled or shown to the model.

The shipped `fakes` module provides `ScriptedAdapter`, `FakeEffectExecutor`
(an external fsynced effect ledger with crash-injection hooks) and
`write_fixture()`. All content is synthetic.

## Run directory

```text
<output_root>/<run-id>/
  manifest.json            immutable binding (create-only)
  config.json              exact config bytes;  capsule/manifest.json exact capsule bytes
  lock                     flock single-writer
  provider/events.jsonl    raw provider events (base64, hash chain)
  journal/events.jsonl     normalized hash-chained journal
  requests/<req>/receipt.json
  artifacts/<dispatch-id>/…   artifacts/manifest.json (at completion)
  final_selection.json     hashes proposal, evidence, artifacts, budget and journal/raw heads
  terminal_manifest.json   binds the frozen record, selection, artifacts and capture status
  transcript.md            derived Markdown view
  quarantine/              incomplete tail lines moved aside by audited recovery
```

Journal record types include `run_registered`, `turn_start` (exact model-visible
messages), `provider_event`, `request_received`, `request_denied`,
`request_admitted` (reservation), `dispatch_intent`, `dispatch_result`,
`dispatch_error`, `reconcile_attempt`, `reconcile_result`, `handoff`,
`adapter_error`, `turn_end`, `final_choice_received`, `final_choice_denied`,
`final_patch_frozen`, `run_halted`, `resume_started`, `resume_refused`,
`tail_recovery`, `editor_terminal` and `phase_advanced`. Each record carries
`run_id`, `seq`, `prev`, `digest`, linked request/dispatch/raw ids and its time.

`verify` fails closed on any altered, reordered, missing or non-canonical line,
a broken raw reference, a partial final line, or a changed artifact, receipt,
selection or terminal manifest. A partial final line is only repaired by
`resume --recover-partial-tail`, which moves it to `quarantine/` (hash recorded)
and appends a `tail_recovery` record; corruption elsewhere is never repaired.

## Tests

```bash
python -m pytest tests/experimental/weight_editor -q
```

The tests drive the public CLI in subprocesses with the fake ports, including
real SIGKILL at the dispatch boundary. They write only under
`scratch/test-weight-editor/`.

## Known gaps before a live run

- No real subscription adapter or real executor exists here; the parent composes
  and qualifies them. Whether a given adapter exports every required field and
  restricts tools is unproven until measured.
- `analyze` (CPU Python) is intentionally unsupported by the shipped executor.
- An unresolved `unknown` dispatch has no operator override yet; the run stays
  `reconciliation_needed` until the executor can resolve it.
- Turns are request/response batches. A streaming provider that expects tool
  results mid-turn must be wrapped to that shape.
