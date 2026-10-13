# Immutable conversation preparation (Stage 1)

The reusable host preparation extension accepts caller-declared text conversation
rows and publishes `synaptic-conversation-sft-row/v1` with a canonical manifest.
This is **preparation only**: current compiled SFT loaders and qualified runtime
releases do not admit this format. Do not relabel it as `syntunia-sft-row/v2`,
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
future SFT integration must mask them and train only the selected final target.
This does not provide all-assistant-turn loss or chronological splitting. Group
truth and semantic relationships remain caller responsibilities; exact hashes
cannot prove absence of related paraphrases or derivatives.

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
is not large-input execution support. Runtime format compatibility remains a
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
