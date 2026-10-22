# SFT Training Reference

Supervised Fine-Tuning — teaches the model tool-calling format and behaviors from positive examples.

---

## CLI Flags

```bash
python train_sft.py [options]
```

### Model Selection
| Flag | Description | Default |
|------|-------------|---------|
| `--model-size {3b,7b,13b,20b}` | Use preset configuration | — |
| `--config PATH` | Load custom Python config file | — |

### Complexity Tiers
| Flag | Description |
|------|-------------|
| `--tier {quick,standard,thorough}` | Use preset complexity tier (overrides individual training params) |

| Tier | LoRA Rank | LR | Epochs | Steps | Time | Use Case |
|------|-----------|------|--------|-------|------|----------|
| `quick` | r=8, alpha=16 | 5e-4 | 1 | 200 max | ~5 min | Rapid prototyping, idea validation |
| `standard` | r=64, alpha=128 | 2e-4 | 1 | — | ~30-60 min | Production training |
| `thorough` | r=128, alpha=256 | 1e-4 | 3 | — | ~2-4 hrs | Maximum quality, publication |

Tier configs: `Trainers/sft/configs/tiers/{quick,standard,thorough}.yaml`

Explicit CLI flags (e.g., `--learning-rate`) override tier defaults.

### Training Parameters
| Flag | Description | Default |
|------|-------------|---------|
| `--batch-size N` | Per-device batch size | 2 |
| `--gradient-accumulation N` | Gradient accumulation steps | 2 |
| `--learning-rate FLOAT` | Learning rate | 2e-4 |
| `--num-epochs N` | Number of epochs | 3 |
| `--max-steps N` | Max steps (overrides epochs) | — |
| `--max-seq-length N` | Max sequence length | 2048 |

### Dataset
| Flag | Description | Default |
|------|-------------|---------|
| `--dataset-name STR` | HuggingFace dataset name | config value |
| `--dataset-file STR` | Specific file in HF dataset | config value |
| `--local-file PATH` | Local JSONL file (overrides HF) | — |
| `--split-dataset` | Create train/validation split | false |
| `--test-size FLOAT` | Validation fraction (rows, or groups with a group key) | config value (0.1) |
| `--validation-group-key PATH` | Dot-path into each row; keeps groups on one side of the split | config value (null) |
| `--use-preassigned-splits` | Consume declared raw-text train/validation splits | false |

### Experiment Tracking
| Flag | Description | Default |
|------|-------------|---------|
| `--wandb` | Enable W&B logging | false |
| `--wandb-project STR` | W&B project name | — |
| `--wandb-run-name STR` | W&B run name | — |

### Utility
| Flag | Description |
|------|-------------|
| `--dry-run` | Setup only, don't train |
| `--resume-from-checkpoint PATH` | Resume from checkpoint |
| `--hf-token STR` | HuggingFace token (gated models) |
| `--no-dashboard` | Disable live dashboard, use table output |
| `--quiet` | Suppress verbose library logs |

---

## Key SFT Settings

### What Makes SFT Different
- **No reference model** — simpler than KTO/GRPO
- **Higher learning rate** (2e-4 vs KTO's 1e-6) — more aggressive learning
- **Multiple epochs** (3 vs KTO's 1) — learn patterns thoroughly
- **Positive examples only** — no True/False labels needed
- **Packing support** — 2.5-5x faster training

### Packing (Recommended)
When `packing: true` in config:
1. Multiple examples packed into single sequences
2. **2.5-5x faster** training due to better GPU utilization
3. Dataset auto-preprocessed with chat template

### Completion-Only Loss
For conversational or prompt/completion rows, when
`completion_only_loss: true` (default):
- Loss computed only on assistant response tokens
- User prompt tokens ignored during training
- Prevents model from learning to generate user messages

### Authoritative raw-text rows

Rows are admitted to the direct-text path only when they declare both
`schema_version: syntunia-sft-row/v1` and `format: raw_text`. An arbitrary
`text` field does not opt a dataset into this behavior.

Raw-text SFT deliberately bypasses chat rendering. The trainer tokenizes the
row's `text`, appends the tokenizer-derived EOS token, and labels the complete
sequence. Its recipe must therefore bind the following settings together:

```yaml
model:
  max_seq_length: 16384  # choose from exact tokenizer-profile evidence
dataset:
  local_file: private/prepared-dataset/dataset.jsonl
  schema_version: syntunia-sft-row/v1
  format: raw_text
  use_preassigned_splits: true
  split_dataset: false
  test_size: 0.0
training:
  completion_only_loss: false
  assistant_only_loss: false
  prompt_render: full_conversation
aux_head:
  enabled: false
```

The declared splits must contain a non-empty `train` set and a non-empty
`validation` set, with no other split names. Do not enable random splitting,
assistant/completion-only masks, `prompt_render: prompt_completion`, or
`aux_head.token_position: end_of_prompt` for raw text. Profile the exact pinned
tokenizer first; the sequence length above is illustrative, not a default.

### Authoritative prompt/completion rows and 32K

`syntunia-sft-row/v2` / `messages` rows are a distinct, fail-closed path for an
exact two-turn user prompt and assistant target. The recipe must bind all of the
following together:

```yaml
model:
  max_seq_length: 32768
dataset:
  schema_version: syntunia-sft-row/v2
  format: messages
  use_preassigned_splits: true
  split_dataset: false
training:
  packing: false
  completion_only_loss: true
  assistant_only_loss: false
  prompt_render: prompt_completion
  require_memory_efficient_loss: true
```

Authoritative rows must fit completely; preprocessing rejects over-budget rows
instead of truncating context or target tokens. At 32K, the local runner rejects
the recipe unless the memory-efficient-loss guard is enabled. The trainer then
checks the loaded model's actual loss-function identity against the active
Unsloth causal-LM mapping, preventing a silent stock Transformers fallback that
could materialize full fp32 logits.

The checked-in `Trainers/recipes/qwen35_4b_32k_prompt_completion.yaml` selects
the `qwen35-sft-v1` runtime profile. It supplies the immutable image, complete
installed-distribution inventory, and exact admitted `Qwen/Qwen3.5-4B` revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` for SFT. The recipe remains
`dry_run: true`; its batch size and accumulation are conservative sizing
candidates only. A green provider-free compile/test pass does not qualify GPU
memory, throughput, or a live provider launch.

Consumers using the provider-neutral `TrainingAPI` carry these seven prepared-row
controls as one atomic optional group on `SFTTrainingHyperparametersV1`:
`dataset_format`, `completion_only_loss`, `assistant_only_loss`,
`use_preassigned_splits`, `prompt_render`, `packing`, and
`require_memory_efficient_loss`. Omitting the whole group preserves the original
v1 input document exactly. A prepared v2 artifact requires the group and
`dataset_format: messages`; the consumer resolver transports it unchanged into
the resolved SFT configuration.

Optional `training.chat_template_kwargs` in a Modal recipe flows through the
provider-neutral SFT hyperparameters into the existing `--chat-template-kwargs`
trainer flag. It is bounded finite JSON, not an executable template override;
renderer-control keys are rejected. Omit it to preserve the legacy canonical
document. For prompt/completion rows, inspect the pinned tokenizer's generation
scaffold and use matching template arguments at serving time. Keep model-specific
arguments in the recipe, never in generic engine behavior.
An offline boundary regression can use an already-verified saved tokenizer
archive, without weights or Hub access:

```bash
SYNAPTIC_TEST_QWEN35_TOKENIZER_ARTIFACT=/private/verified/tokenizer.artifact \
  python -B -m pytest -q tests/training/test_chat_template_training_transport.py
```

The artifact-backed case checks the exact pinned template hash, generation
scaffold, prompt mask, and prose-plus-terminal target (the terminal is the
template's end-of-turn token, which is `eos_token_id` for that tokenizer when
the two coincide). Without that explicit local
artifact it skips; the generic contract/transport cases still run. This checks
token placement, not live GPU execution or writing quality.

### Auxiliary Readout Head (`aux_head`, optional)
An optional auxiliary scalar readout head that learns to predict a per-row
target from a base model's hidden state. It runs in two modes:

- **Phase A** (`freeze_base: true`, `lm_loss_weight: 0`): the base/LoRA is frozen
  and only the small head trains; the LM loss is not used.
- **Phase B** (`freeze_base: false`, `lm_loss_weight > 0`): the base stays
  unfrozen (the LoRA params PEFT left trainable are added to the optimizer as a
  second group at the trainer LR) and the head co-trains jointly with the LM
  loss — total loss is `lm_loss + lm_loss_weight * head_loss`.

The feature is **off by default** — when the `aux_head` block is absent or
`enabled: false`, the SFT path is byte-identical to a standard run. The Phase-A
path (frozen base, no LM term, `input_norm: none`) is itself byte-identical
whether or not the Phase-B code is present, because both branches gate on the
config values.

When enabled, every training row must carry a finite numeric target under the
column named by `target_field`; a missing, null, non-numeric, or non-finite
value fails loudly (there is no silent default). After training, the head is
written next to the model as a sidecar (`aux_head.safetensors` +
`aux_head_config.json`) so it can be reloaded for inference independently of the
base weights. The sidecar persists `input_norm`, so a head trained with
normalization reloads and infers identically.

```yaml
aux_head:
  enabled: true            # absent / false => feature fully off
  layer: 35                # required when enabled: which hidden_states index to read (0 = embeddings)
  token_position: last     # "last" (last non-pad) | "mean" | int | "end_of_prompt"
  target_field: target     # per-row dataset column carrying the scalar target
  loss: bce                # "bce" | "brier" (MSE on probability)
  head_type: linear        # "linear" | "mlp"
  hidden_dims: []          # hidden widths when head_type: mlp
  out_activation: sigmoid  # "sigmoid" (prob in [0,1]) | "identity"
  input_norm: none         # "none" (off; default) | "layernorm" (so the head trains at a normal LR)
  freeze_base: true        # Phase A = true; Phase B = false (co-train the unfrozen base)
  lm_loss_weight: 0.0      # Phase A = 0.0; Phase B > 0 (weight on the head loss in the joint sum)
  head_lr: null            # optional dedicated head LR; defaults to trainer LR
```

Two Phase-B knobs:

- **`token_position: end_of_prompt`** reads the last *prompt* token (the position
  right before generation) instead of the last completion token. At train time
  the prompt/completion boundary is recovered from the label mask (prompt tokens
  are masked to `-100`); a row with no prompt span (completion-first or empty
  completion) falls back to the last real token. At inference the input is
  prompt-only, so `end_of_prompt` and `last` coincide.
- **`input_norm: layernorm`** applies a `LayerNorm` to the pulled hidden state
  before the head, so a linear head trains at a normal LR on unnormalized
  activations instead of saturating.

### `prompt_render` (a `training` knob, paired with `end_of_prompt`)

`token_position: end_of_prompt` is only faithful when the row was tokenized so the
prompt ends exactly at the generation anchor. The default render
(`training.prompt_render: full_conversation`) renders the whole conversation and
derives the assistant-only mask by a prefix match — but many chat templates render
the assistant scaffold differently with vs. without `add_generation_prompt` (e.g.
one fewer newline), so the masked boundary token is NOT the generation anchor and
the head reads an off-anchor representation.

Set `training.prompt_render: prompt_completion` for a faithful boundary: the row's
`input_ids` are built from the `add_generation_prompt=True` prompt render followed
by the raw completion plus the chat template's end-of-turn token, with the
prompt segment masked to `-100` when `completion_only_loss: true` (with
`completion_only_loss: false` every token is trained). The terminal is derived
from the template (the special token it renders after assistant content, the
same derivation `doctor sft-mask` uses); only when the template renders none does
it fall back to `eos_token_id`, logged once. Where `eos_token_id` already is the
end-of-turn token the rows are byte-identical to the earlier eos-terminated
construction; where they differ (for example eos `<|endoftext|>` vs turn end
`<|im_end|>`), the completion now ends with the turn terminator. The prompt then
ends exactly at the generation anchor, so the existing `end_of_prompt` read is
faithful.

```yaml
training:
  prompt_render: prompt_completion   # full_conversation (default) | prompt_completion
```

`prompt_render` lives on the `training` config (it replaces the SFT masking
region, so it is not an `aux_head` field), but it is grouped with the aux_head
knobs for launching (see below). It defaults to `full_conversation`, so every
existing recipe is byte-identical. When `aux_head.enabled` and
`token_position: end_of_prompt` are set while `prompt_render` is still
`full_conversation`, the trainer prints a one-line WARNING (not an error — the
combo is legitimate for single-turn / inference-shaped rows whose two renders
coincide).

Complete runnable examples live at
`Trainers/sft/configs/aux_head_example.yaml` (Phase A) and
`Trainers/sft/configs/aux_head_phase_b_example.yaml` (Phase B). `layer` has no
default — choosing the hidden-state index is a deliberate per-run decision, so an
enabled block without it fails fast.

### Launching aux_head (local-run recipe + direct CLI)

The `aux_head` block flows through **both** launch paths:

- **Direct trainer** (`cd Trainers/sft && python train_sft.py ...`): every field
  has a CLI flag — `--aux-head-enabled` / `--no-aux-head-enabled`,
  `--aux-head-layer`, `--aux-head-token-position`, `--aux-head-target-field`,
  `--aux-head-loss`, `--aux-head-head-type`, `--aux-head-out-activation`,
  `--aux-head-input-norm`, `--aux-head-freeze-base` / `--no-aux-head-freeze-base`,
  `--aux-head-lm-loss-weight`, `--aux-head-head-lr`, and `--aux-head-prompt-render`
  (which sets `training.prompt_render`). An unset flag never overrides the loaded
  config; a CLI value takes precedence over the YAML/preset config.
- **local-run recipe** (`python tuner.py local-run --job-config <recipe>.yaml`):
  put an `aux_head:` block (and, when using `end_of_prompt`, a
  `training.prompt_render: prompt_completion`) in the recipe; the runner forwards
  every field to the trainer's `--aux-head-*` flags automatically. Recipes with no
  `aux_head` block and no `training.prompt_render` emit zero new flags, so they
  stay byte-identical. aux_head forwarding is SFT-only.

---

## Training Workflow

1. **Choose runtime**: prefer `python tuner.py local-run --job-config Trainers/recipes/<recipe>.yaml --yes` for repeatable local Docker runs; use direct `cd Trainers/sft && python train_sft.py ...` for tight trainer iteration.
2. **Prepare dataset**: use conversational JSONL for chat semantics, or run
   `prepare-dataset` for a verified `syntunia-sft-row/v1` raw-text artifact.
   Then check loss masking with the target tokenizer:
   `python tuner.py doctor sft-mask --sft-config <trainer-config> --model <tokenizer> --dataset-path <jsonl>`
   (tokenizer only, no GPU; exit 1 on hard failures such as too many dropped
   rows or a wrong end-of-turn token). See `dataset-formats.md` → Validation.
   Preprocessing contract version 2 changed SFT labels (end-of-turn stop,
   dropped untrainable rows, `training.max_dropped_row_fraction`); losses from
   earlier runs are not directly comparable.
3. **Test setup**: set `run.dry_run: true` in local-run YAML or use `python train_sft.py --model-size 7b --tier quick --dry-run`
4. **Quick iteration**: cap `training.max_steps` in local-run YAML or use `--tier quick`
5. **Production run**: remove the step cap and use the intended `training`, `model`, `dataset`, and `lora` settings in YAML
6. **Monitor**: Watch live dashboard or `tail -f logs/training_latest.jsonl`
7. **Upload**: `python3 .skills/upload-deployment/scripts/upload_model.py ./final_model user/repo --save-method merged_16bit`

---

## Typical SFT Performance (7B on RTX 3090)

| Metric | Value |
|--------|-------|
| VRAM usage | ~7-9 GB |
| Speed (with packing) | ~500-800 tokens/sec |
| Time per epoch (~2700 examples) | ~15 min |
| Total (3 epochs) | ~45 min |

---

## Config File

**Direct trainer location:** `Trainers/sft/configs/config.yaml`

**Config-driven local Docker location:** `Trainers/recipes/*.yaml` (recipes with `target: local` or `target: both`)

Local Docker job configs keep runtime choices outside shell history:

```yaml
name: my-sft-run
provider: local_docker
job:
  image: unsloth/unsloth:latest
  pull_policy: missing
  transfer: auto
setup:
  pip: []
run:
  method: sft
  trainer: Trainers/sft/train_sft.py
model:
  name: Qwen/Qwen3.5-2B
  max_seq_length: 2048
  load_in_4bit: false
dataset:
  local_file: Datasets/my_data.jsonl
training:
  batch_size: 2
  gradient_accumulation: 8
  learning_rate: 1.0e-4
  num_epochs: 1
lora:
  r: 64
  alpha: 128
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
artifacts:
  output_root: toolset-training-artifacts/runs/local_docker/sft/{name}
  run_timestamp: "{timestamp}"
```

Key sections:
- `model` — model name, seq length, quantization
- `lora` — rank, alpha, dropout, target modules
- `training` — batch size, LR, epochs, packing, etc.
- `dataset` — source, filtering, split
- `evolutionary` — experimental gradient evolution (disabled by default)
- `aux_head` — optional auxiliary scalar readout head, head-only over a frozen base (Phase A) or co-trained with the LM loss over an unfrozen base (Phase B); disabled by default

For cloud runs, evolutionary SFT is now expressible through `run-experiment` specs or `cloud-pipeline --train-evolutionary-*` overrides. Keep the first run short and capped by `max_steps`; the wrapper adds real per-step overhead.

See `reference/training-config.md` for full config documentation.


## Public training qualification consumer

The checkout-owned `examples/training_qualification/launch.py` exercises the
existing public TrainingAPI and RunsAPI with caller-owned configuration, named
secret references and SQLite stores. See its README and Docker/HF templates.
Run the installed interpreter from a neutral directory using
`python -I /absolute/checkout/examples/training_qualification/launch.py ...`;
the example itself is not a packaged engine module.

`plan` checks existing material/compiler/profile bindings without resolving
credentials or contacting a provider. Full public preflight is deferred until
host composition in `start`. State is create-only: an unknown submission never
authorizes replay. Successful native verification precedes downloading the five
SFT artifacts through the public streams.

Docker cold composition uses retained material/profile/request bytes, original
named keys, qualification assets and local mount roots. It is not source-free
recovery or remote Docker path mapping. HF cold reads use the existing retained
reader and original named keys without restoring input authority. Default
capability refusals remain in force. CPU scripted-provider tests do not qualify
an image, GPU training, physical termination, provider billing or live parity.

### Genuine host-owned HF material preparation

Use `examples/training_qualification/prepare_hf.py` with the declared config
and `hf_raw_text_fixture` when preparing the ordinary HF SFT consumer. The
fixture ingests eight complete synthetic Question/Answer bodies and publishes
`syntunia-sft-row/v1` through the existing service, with six training and two
validation rows. This supported raw-text lane applies full-token loss; it does
not relabel messages/v2 data or apply completion-only loss.

`python examples/training_qualification/prepare_hf.py declare <config.json>`
exclusively writes a credential-free runtime declaration from the selected
profile/image/interpreter, exact HF roots, configured environment and named
requirements. Its canonical bytes are retained in material execution_context;
the ExecutionSource deployment-member digest means this declared configuration
for HF, not an observed deployment or staged remote member.

`python examples/training_qualification/prepare_hf.py prepare <config.json>`
uses genuine clean host/Gitlink inspection, bounded public Git verification and
an explicitly named local source-authentication key distinct from both HF token
references. Each Git read is bounded to 30 seconds, source evidence freshness
to 600 seconds, and prepared bytes to 2 MiB. Supply real bucket, quote, scope,
source and named-key inputs; unfilled example values refuse. Preview does not
verify bucket existence or current pricing. Only actual parsed/rederived
material and successful consumer validation/HF preview publish consumer.json.
Full public preflight remains deferred to warm composition. No Job, bucket,
model download, image/GPU qualification or capability promotion occurs here.
See the checked-in example README for commands and exact fixture coverage.

## Public Docker qualification consumer preparation

The checkout-owned `examples/training_qualification/prepare_docker.py` closes
material setup for the public Docker consumer. With an installed engine,
`plan` validates declared release/model/layout, `prepare-model` first prepares
the pinned public snapshot into a fresh private artifact cache, and `prepare`
then authenticates fresh source and existing immutable v1 input before publishing
canonical material. A private read-only seed cache may reduce downloads only
after pinned upstream metadata and every reused Git/LFS member are verified.
The old seed claim is never adopted. `prepare` binds the complete model receipt;
public start rechecks its bytes before Docker effects. A missing pinned revision
in an otherwise private seed takes the normal cold path; redirects reject.
Cold public reads use the receipt retained with the attempt and do not require
current model weights or the original preparation root. See `examples/training_qualification/README.md`
for the template and exact commands. Full public preflight remains deferred
until host composition; missing image/runtime/model qualification still refuses.
The CPU `probe-filesystem` command measures actual native writer modes and host
readability under a reviewed finite container plan; it is not training, GPU,
provider or cancellation qualification. Source inventory and five-role native
verification remain closed; no default capability or UID policy is promoted.

Before short-lived source preparation, run `model-preflight` with the exact
interpreter selected for `prepare-model`; the base consumer may lack the optional
Hub SDK. This checks only the existing SDK signature contract and creates no
client or claim. `prepare-model` repeats it before its one-use claim. Use an
independently verified helper environment; do not upgrade the pinned trainer.
Create and verify necessary private source/artifact parent directories first.
Preserve any consumed model-preparation claim after failure. New failures retain
the existing allowlisted `model_preparation_stage`; never infer that stage or a
cause for an older generic diagnostic, or replay an uncertain attempt.

### Offline SFT worker rejection

The isolated source worker retains exit code 2 on failure and emits one bounded
diagnostic line to the existing `tracking/trainer.stderr.log`:
`OFFLINE_SFT_WORKER_REJECTED phase=<closed> category=<closed>`.
The phases are `BOOTSTRAP`, `CLOSURE`, `BOOTSTRAP_ORIGIN`, `ARGUMENTS`,
`IMPORT_GUARD`, `TRAINER`, and `POSTCHECK`. Categories are `REJECTION`, `IMPORT`,
`OS`, `VALUE`, `SYSTEM_EXIT`, and `OTHER`. During `TRAINER`, the last valid
existing packaged-trainer hook may append `milestone=<closed>` (for example
`TORCH_IMPORT` or `MODEL_LIBRARY_LOAD`). Unknown hooks clear that milestone.
The fallback remains the bare rejection marker. This diagnostic does not print
exception text, arguments, paths, environment values, or traceback frames;
library-generated stdout/stderr is a separate surface.

Use the labels to choose the next targeted reproduction. An import category
does not identify the missing library, and a model-load milestone does not
establish OOM. Do not interpret an older bare marker retrospectively. Preserve
the consumed claim, roots, container and retained logs; the diagnostics never
authorize replay. See `docs/review/offline-sft-worker-diagnostics.md` for the
October 20 evidence boundary, CPU regressions and fresh-run prerequisites.

For a fresh, credential-free Docker Torch import reproduction, review the exact
command before effects. The checked-in `scripts/diagnose_docker_torch_import.py`
requires `--docker`, an empty `--docker-config`, `--release`, independently
verified `--expected-release-sha256`, `--daemon-id`, and a nonexistent private
`--output-dir`. Guarded mode additionally requires `--source-root`,
`--source-lock`, independently reviewed `--expected-source-lock-sha256`,
`--expected-manifest-sha256`, and `--expected-closure-digest` from the retained
attempt. Direct-worker mode also requires `--expected-engine-commit` set to the
independently reviewed engine gitlink from that source lock; legacy modes default
to the pinned r2 commit when this option is absent. It validates every staged
worker member before executing mounted code;
the result records those commitments. Run from the reviewed checkout so its
trusted closure verifier is the one checked in with the helper.

The helper claims a unique name before Docker create, authenticates the exact
new container before start, and retains it. Docker CLI output is capped at 8 KiB
in memory; the container requests JSON log rotation at 1 MiB with one file.
The import execution has a
120-second host deadline. At expiry the helper issues at most one stop request
to the authenticated new container and separately inspects terminal state;
cleanup takes additional bounded time, and a stop acknowledgment alone is not
quiescence. An uncertain create/start/stop remains consumed in `claim.json` and
`events.jsonl`, with no automatic replay or container removal. The default
probe asks for at most 2 CPUs, 4 GiB, no network/GPU/model mount, and a
read-only root.
Bare mode mounts no source. Guarded mode mounts the independently authenticated
retained source and runs the worker closure check, module guard, trainer path
setup, environment bootstrap and Torch import. Neither runs training. `PASS`
only says Torch imports in the reduced probe; it does not diagnose an earlier
failure. Review the pinned package/runtime source before changing a profile,
image or runtime policy.
The exact r2 CPU evidence and its unresolved runtime difference are in
`docs/review/docker-torch-import-r2-diagnosis.md`.

The guarded-only `--gpu-device 0` option is an import-only comparison that
requires separate exact root review before any live use. It requests only
device 0, verifies Docker retained that device request before start, removes
the CPU probe's visibility suppressors, and records the scope in its claim.
`gpu_used=null` means actual device use was not measured. The same 8-KiB
output budget, 120-second execution deadline, owned cleanup, source
authentication and closed child codes apply. It neither invokes the trainer
nor loads a model; a PASS or failure cannot retrospectively diagnose an
earlier trainer attempt. See the exact proposed command and remaining
environment/user/filesystem gaps in the r2 diagnosis note.

For the worker-context child, keep supplied environment bytes distinct from
the environment observed after isolated Python startup. Require every
supplied key unchanged; the only reviewed addition is
`LC_CTYPE=C.UTF-8`, recorded as a closed delta with separate digests.
Any other addition or change rejects. Do not reset the observed environment
or reinterpret it as a historical capture of the failed r2 child. See the r10
repair and exact unexecuted command in the diagnosis note.

The optional `--worker-context` mode requires independently reviewed
`--expected-workload-sha256`, `--expected-runtime-sha256`, and
`--expected-trainer-sha256` alongside guarded source commitments. It
reconstructs the pinned workload's isolated child environment, launches a
separate `-I` child, loads the authenticated worker closure and guard, and
executes only the authenticated trainer prefix through `import torch`.
The prefix excludes the `UNSLOTH_IMPORT` marker and all later trainer/model
code. It records closed environment, path, file and cwd digests and keeps the
existing claim, output budget, deadline and owned cleanup. Its writable
`/artifacts` is a small private tmpfs with no model weights. Review the exact
workload/source hashes, device scope, host user, caps and filesystem before
each fresh use. A reconstructed child environment is a source-and-plan
deduction; it is not a historical capture of the failed child.

`--worker-context --worker-entry` selects a separate, still import-only
diagnostic. It reconstructs the pinned runtime's trainer argument vector,
including a one-byte sealed stand-in for the prepared dataset descriptor,
then invokes the authenticated worker's real `main()` and nested
`runpy.run_path` trainer entry. A source-bound line trace raises at the
`UNSLOTH_IMPORT` marker after Torch import, while a narrow import-call and
module-finder barrier prevents Unsloth entry even if tracing is disabled or
that module name was cached. A PASS
requires the worker's closed stop label and real trainer `__main__` module
binding. The worker still performs its own argument, closure and import-guard
checks. It records closed environment digests before and after Torch import,
separate from the supplied and interpreter-startup digests. The diagnostic
starts through `-I -c` to install that boundary, so it
does not reproduce the original direct `-I` script startup. The sealed data
byte, reduced filesystem and absent model weights also differ from the failed
attempt. Review and admit a fresh one-use command before any live execution;
do not infer the historical failure cause from either result. See the r12
proposal in the diagnosis note.

`--worker-context --worker-direct --public-parent` keeps the direct worker's
independently reviewed derivative trainer and fixed pre-Unsloth stop, but
starts the checked-in `runtime_v1.py` parent using its canonical workload-file
transport. It requires explicit expected engine and project commits, original
source/manifest/closure/runtime/trainer/workload/model receipt pins, derivative
trainer/manifest/closure pins, and parent environment/command digests. The
normal parent authenticates the derivative closure and prepared input before
starting its ordinary isolated worker; the diagnostic helper does not alter
the production runtime or worker start method. It checks the actual new Docker
parent configuration and mounts before start and at terminal, and accepts
only the parent's generic fixed failure marker plus the worker's fixed closed
`TORCH_IMPORT` import or `UNSLOTH_IMPORT` stop label. It retains the same owned
claim, finite deadline, bounded CLI output, model inventory and cleanup
rules. No command in a diagnosis note is live authority without separate
one-use review; successful import cannot identify the cause of an earlier
failed public run. See the r32 proposal in the diagnosis note.

After an intentional worker source change, check and refresh existing
manifests with the checked-in maintainers in dependency order:
`regenerate_offline_sft_worker_closure.py`,
`regenerate_packaged_training_worker_closure.py`,
`regenerate_grpo_worker_closures.py --kind offline`, then `--kind outer`,
`regenerate_hf_parent_closure.py`, `regenerate_modal_runtime_lock.py`, and
`regenerate_modal_inference_lock.py` (all under `scripts/`). Run each read-only
check first, review its drift, use `--write` only for intentional hash refresh,
then recheck. Preserve inventories and non-hash pins. Sync canonical skills
with `.skills/scripts/sync_skill_trees.py` and verify its `--check`.
