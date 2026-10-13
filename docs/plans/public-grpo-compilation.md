# Public GRPO compilation — G1 contract

Status: implementation contract; provider execution remains unavailable until separate GRPO child, closure and provider admission are accepted. SFT canonical documents and default recipe imports remain unchanged.

## Public input

The existing `synaptic-training-input/v1` gains method `grpo`. Existing model, dataset and artifact fields retain their types. GRPO requires `synaptic-grpo-hyperparameters/v1` and a GRPO-only `rewards` object with exactly `items`, `documents`, `schema`. SFT rejects rewards and retains its existing serialization. Unknown fields are rejected.

| Hyperparameter | Finite contract |
| --- | --- |
| batch_size, gradient_accumulation_steps | Exact integers 1..4096; effective batch divisible by num_generations |
| num_generations | Exact integer 2..4096 |
| max_steps | Exact integer 1..100000; explicit step-limited first lane |
| max_seq_length, max_prompt_length, max_completion_length | Exact integers 1..100000; prompt plus completion fits sequence bound |
| learning_rate | Finite number greater than zero, at most 1 |
| temperature | Finite number greater than zero, at most 10 |
| beta, weight_decay | Finite numbers 0..1 |
| warmup_ratio | Finite number 0..1 |
| max_grad_norm | Finite number greater than zero, at most 100 |
| seed | Exact integer 0..2**32-1 |
| logging_steps, save_steps, save_total_limit | Exact integers 1..100000 |
| lora_rank, lora_alpha | Exact integers 1..4096 and 1..65536 respectively |
| lora_dropout | Finite number 0 inclusive to 1 exclusive |
| lora_target_modules | 1..256 unique nonblank strings, each at most 128 UTF-8 bytes |
| use_dora, use_rslora, load_in_4bit, bf16, fp16 | Exact booleans; bf16 and fp16 cannot both be true |
| prompt_column | Explicit nonblank string, at most 128 UTF-8 bytes |
| chat_template_kwargs | Optional existing bounded template-argument contract; omitted stays omitted |

Fixed normalized first-lane behavior: native chat template, no existing adapter, no vLLM/GSPO/pivot/custom Python rewards, no judge/environment rollout, no external reporting, no freeform training extras. Fixed normalized values are optimizer `adamw_8bit`, scheduler `linear`, num_train_epochs `1`, report_to `none`, LoRA bias `none`, gradient_checkpointing `unsloth`, and model dtype `null` (existing runtime precision selection). No init_lora_weights field is invented: the retained GRPO LoRA helper does not forward one. Batch divisibility is conservative single-process admission, not distributed-training parity. Both precision flags false retain the existing core hardware precision selection; they do not promise FP32.

## Declarative rewards

Items are 1..32 unique named selections with finite weights 0..100; at least one weight is positive. Documents contain exactly selected names, with each name matching its document. Names are nonblank and at most 128 UTF-8 bytes. Each document has `name`, `scope: response`, `validation`, and `scoring`; optional `type: rubric` and bounded description are allowed. Validation has 1..32 rules, each exactly `{type: regex, match: <pattern>}`; each pattern is at most 4096 UTF-8 bytes and must compile. Scoring has exactly `strategy: binary`, `on_pass`, `on_fail`, with finite numbers -100..100. The first raw-response lane requires explicit empty `schema: {}`. Total rewards canonical JSON is at most 64 KiB, with finite JSON values and no undeclared fields. This binds declarations, not reward quality or regex execution latency.

## Prepared custody and compilation

Prepared format is `synaptic-grpo-prompts/v1`: exact nonempty UTF-8 JSONL, at most 65536 bytes and 128 rows. Every row is a finite JSON object with string keys; the explicit prompt column is nonempty text or a nonempty message list with exact role/content fields. Remaining finite reward columns are preserved. No assistant targets, SFT splits or invented lineage are added. UTF-8 is decoded strictly; BOM and UTF-16/32 inputs are rejected. If prompt_column differs from `prompt`, an existing other `prompt` column is rejected because the current trainer formatter reserves that derived field. A closed publication sidecar binds format, prompt_column, content hash, byte size and row count; semantic identity includes this preparation basis and excludes model, hyperparameters and rewards so legitimate learning runs reuse the same dataset.

The normalizer and verifier use existing generic preparation ports, immutable identities and retained input sources. Fresh publication is exclusive; independently verified identical publications may be reused, while changed or unverified destinations fail. A small GRPO adapter reuses retained publication primitives because the SFT publication identity requires split and row-ID fields that GRPO does not invent. The optional verifier validate_training_input(path, identity, canonical_training_json) runs after identity construction and before retained lease verification/result return. GRPO rereads the closed manifest and snapshot, checks identity and prompt_column agreement, and checks actual retained GRPO core admission against supplied detached normalized config/reward/model/workload commitments. Existing verifiers without this optional method retain their behavior. Pure compilation binds exact config, dataset and reward hashes; compilation alone does not establish row validity. Model and tokenizer revisions are immutable lower-case 40-hex revisions.

The normalized workload binds method `grpo`, exact model identity, normalized config SHA-256, original prepared dataset SHA-256 and SHA-256 of `{documents, schema}`. The explicit recipe resolved config has exactly schema_version, method grpo, execution `{mode: developer_integration}`, training_input and prepared_input. It works through the existing RecipeRegistry/compile_execution_workload pure compilation path. This is a developer/core declaration, not executable GRPO admission. Packaged projection/entrypoint is deferred to G2/G3; no packaged_grpo_compilation support is claimed here. New config/workload schemas are `synaptic-grpo-config/v1` and `synaptic-grpo-workload/v1`. The recipe is explicitly registered and lazily imported. No executable dispatch or provider readiness is added by G1. Native artifact policy remains the existing five roles; checkpoint retention is unsupported in this lane.

## Acceptance and source boundaries

Source adapter constraints are explicit: Linux retained-descriptor operations, absolute caller-owned 0700 source and output roots, one declared source leaf with private permissions, regular single-link file only; selected source directory must be one of the admitted roots. These constraints are narrower than legacy arbitrary source paths. Failed publication/close retains uncertain owned stage or committed bytes and does not report success. This is local immutable custody, not a sandbox or remote durability claim.

CPU tests cover strict value round trips, legacy SFT bytes, unknown and nonfinite fields, reward mutation, real core admission, prepared mutation and exclusive publication, exact workload hashes and guarded imports. No model download or training is required. Changes to training_input.py and training/__init__.py intersect existing Modal/inference inventories; the latter also intersects the SFT offline closure. Root reviews source-only hash refresh separately, preserving inventories and pins. G1 does not modify the packaged SFT five-member closure.
