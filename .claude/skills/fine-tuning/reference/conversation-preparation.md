# Immutable conversation preparation and CPU SFT support (Stages 1–2)

The reusable host preparation extension accepts caller-declared text conversation
rows and publishes `synaptic-conversation-sft-row/v1` with a canonical manifest.
The compiler, offline parser and native verifier support its text-message
contract in CPU conformance tests. **Qualified runtime releases do not yet admit
this format**. Do not relabel it as `syntunia-sft-row/v2`,
change an old release allowlist, or report a preparation test as training evidence.

Use existing `TrainingAPI.prepare` with `TrainingPreparationConfigV1`. Compose
`TrainingInputPreparationServiceV1` with a host-owned private prepared root and
`PosixPrivatePreparedRootAuthorityV1`, registering:

- `synaptic.conversation-preparation/v1` →
  `ConversationTrainingInputNormalizerV1(allowed_source_roots=(source_root,))`
- `synaptic-conversation-sft-row/v1` →
  `ConversationPreparedInputFormatVerifierV1()`

Adapters live in `tuner.training.conversation_preparation`; the immutable
`ConversationNormalizerConfigV1` lives in
`tuner.dataset_prep.conversation_messages`. No provider, client, source recipe,
or generation retry is needed to prepare. The existing `prepare-dataset` CLI's
v1/v2 ingestion configs are a different input route; it does not accept this config.

## One declarative document

`ConversationNormalizerConfigV1.from_dict(document)` accepts the closed schema
`schemas/synaptic-conversation-preparation-v1.schema.json`. The configuration
contains no absolute source path; pass `LocalTrainingInputPathV1(source_root)`
separately. Its source members are explicit single relative filenames, not an
assumed `.meta.json` suffix.

- `source`: dataset member/hash/byte count, optional metadata member/hash/byte
  count and optional row-count key path, exact raw row count, host-declared
  `provenance` object, and optional `data_outcome`.
- `projection`: bounded `messages_path` key segments, configured role/content
  field names, and explicit source-role mapping to system/user/assistant.
  Selected contents must already be nonempty strings. Auxiliary source fields
  are omitted by this declared projection and remain covered by the raw-row hash.
  Structured content, native tool-call-only payloads and multimodal content are
  rejected; there is no implicit JSON stringification or tool-result conversion.
- `selection`: `rows: null` selects all rows; otherwise sorted unique 1-based
  row numbers. `target_index: null` requires the final message to be assistant;
  otherwise an explicit zero-based assistant index selects it and excludes the
  suffix. Every earlier mapped message is retained in order. The manifest records
  original message count, selected index and raw-row provenance.
- `grouping`: `path: null` groups by the complete mapped conversation hash;
  otherwise select a nonempty logical group value through an explicit key path.
- `split`: either `kind: field` with key path and explicit source-value mapping
  to train/validation, or `kind: heldout_groups` with sorted unique
  `validation_groups`. Both splits must be nonempty. Unknown heldout groups,
  conflicting assignments and cross-split groups fail. Complete mapped source
  conversations and identical selected prefix+target cannot cross splits even
  when unrelated row IDs, metadata or discarded suffixes differ.

One row selects one target. Multiple preceding assistant turns are context; the
SFT prompt-completion path masks them and trains only the selected final target.
This does not provide all-assistant-turn loss or chronological splitting. Group
truth and semantic relationships remain caller responsibilities; exact hashes
cannot prove absence of related paraphrases or derivatives.

## Explicit SFT controls and release boundary

The new format requires `dataset_format: messages`, `use_preassigned_splits: true`,
`split_dataset: false`, `completion_only_loss: true`, `assistant_only_loss: false`,
`prompt_render: prompt_completion`, `packing: false`, and
`require_memory_efficient_loss: true`. Optional `chat_template_kwargs` remain
recipe configuration and retain the admitted template-generation anchor. Rows
must fit without truncation and produce target tokens. The declared dataset
schema travels through offline argv and row loading; mixed schemas and new rows
masquerading as the old v2 format reject.

A CPU parser/native-artifact fixture proves contract behavior, not GPU training,
model quality, image qualification or provider reliability. Existing release
allowlists remain unchanged and reject this format. A future additive candidate
selection needs separate review; do not bypass or edit a historical selection.
The native prepared-path verifier retains its 2 MiB limit for old v1/v2 and the
new conversation format; preparation's larger publication bound is not execution
support.

## Public Data provenance

For Data-origin input use
`ConversationNormalizerConfigV1.from_data_outcome(document, outcome)` with the
actual existing `DataOutcome`. The outcome becomes part of the same canonical
configuration and semantic dataset identity. It requires succeeded status,
complete requested/written counts, matching raw row count, and both exact
`dataset_jsonl` and `dataset_metadata` hash/size commitments. Missing metadata,
partial outcomes or contradictory declarations fail. Generic non-Data inputs
use `data_outcome: null`; they still require exact source hashes/sizes and declared
provenance. Do not omit a Data outcome merely to bypass a mismatch.

## Custody and limits

Linux publication reuses the existing descriptor-backed private publisher,
exclusive staging, no-replace rename, fsync and collision handling. A private
owner-only 0700 root is required. Read-only source capture rejects symlinks,
hardlinks, special files, changed roots/members and mismatched hashes. Config
and manifest are bounded to 2 MiB, source/publication bytes to 64 MiB, each row
(including newline) to 1 MiB, and messages to 128 per row. Explicit finite
structural/depth limits apply; rejection never truncates data.

The current executable preparation adapter rejects publications above **2 MiB**.
A larger publication can be independently verified within the 64 MiB bound; this
is not large-input execution support. Qualified release/profile format compatibility remains a
separate gate even below 2 MiB.

The returned public prepared value contains identity/count-independent digest
fields, not corpus text or host paths. Its host-only retained source revalidates
the immutable publication before each fresh one-use lease. Original raw inputs
can be removed after preparation; the publication must remain unchanged.

The verifier checks publication/manifest consistency, exact content, row IDs,
selection, group/split constraints and committed provenance. Raw-source provenance
is a preparer assertion derived from the admitted read; after deletion a digest
cannot independently reconstruct historical source truth. Rewriting provenance
creates a different semantic identity and cannot satisfy an old expected prepared
identity. Preserve original Data outcome and source commitments for audit.

## Data-to-SFT candidate planning (Stage 3)

The Python API handoff uses actual `DataAPI` generation and improvement outputs,
`DataAPI.validate`, and the conversation normalizer/verifier through
`TrainingAPI.prepare`. Bind the selected run's exact `DataOutcome` when it is
supplied; a generated run's commitments cannot be substituted for an improved
run's artifact. Optional generic provenance remains a preparer declaration.
Retain the prepared source and immutable publication. Raw datasets, metadata,
scenario and generation/preparation configuration copies can then be removed;
subsequent prepared leases and configured `TrainingAPI.resolve` / `plan` use the
retained identity and bytes without regenerating data.

`qwen35-conversation-sft-candidate-v1` is a separate, **unqualified** planning
selection for `synaptic-conversation-sft-row/v1`. It preserves the exact Qwen
model/revision and immutable base inventory; historical Qwen SFT profiles and
release format allowlists remain unchanged. Its source/host-owned build profile
is `Trainers/image_profiles/qwen35_4b_conversation_sft_candidate_v1/profile.yaml`.
It binds the accepted Stage 2 wheel SHA-256
`13aff34e7e0a0414a46770637b54d2393320097a02fc5bc0a42fe1e7d48f8199`.
The build profile is outside that wheel to avoid a circular self-hash. The
candidate runtime-profile and compatibility declarations are installed assets;
the consuming host supplies the build profile explicitly.

The joined CPU contract test uses text-only configured environment fixtures,
real bounded filesystem assertions, and scripted judge/improver corrections.
It does not coerce structured tool calls or `content: null` into text. Its
provider/release metadata and Git observation are explicitly synthetic planning
fixtures, not deployment observations, source admission or build capture.
Actual image construction, CPU/GPU qualification and training remain separate.
This handoff supports Python API resolve/plan only: the ordinary Modal recipe
CLI remains bound to its existing v2 messages path. No model-quality, live
training or new provider capability claim follows from this planning selection.
