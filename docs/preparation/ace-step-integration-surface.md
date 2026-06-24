# Repo Integration Surface — ACE-STEP Music Pipeline

> **Phase**: PREPARE (research only — no pipeline code).
> **Scope**: HOW a new audio/music training + generation method slots into THIS
> repo (Synthetic Conversations). Sibling research: model internals are covered in
> [`ace-step-model-research.md`](./ace-step-model-research.md) (peer `preparer-acestep`).
> **Audience**: the architect, who will design `Trainers/ace_step/` (or equivalent)
> to match existing conventions.
> All file paths are relative to the repo root. Line numbers verified against the
> working tree on 2026-06-24.

---

## 1. Executive Summary

This repo has a **convention-driven training-method system**: a training method
is registered in one SSOT tuple (`shared/utilities/paths.py:11`
`TRAINING_METHODS`), and a method named `<m>` is expected to live at
`Trainers/<m>/train_<m>.py` with `Trainers/<m>/configs/config.yaml`. Both the
local (RTX/Mac/Docker) and cloud (HF Jobs / Modal / RunPod) backends derive the
trainer script name and directory from the method name by string convention
(`train_{method}.py`, `Trainers/{method}/`). **The `embedding` method (added most
recently) is the best precedent for adding a new method**, and it demonstrates
both the clean seams and the rough edges.

The headline finding: the **training-dispatch wiring is genuinely low-friction**
(a method is mostly "a directory + a tuple entry + a config schema"), BUT the
repo carries **three hardcoded duplicates of the method list** that must be kept
in sync, and the **local-Docker auto-builder only knows `sft`/`dpo`/`kto`** — the
embedding method side-steps it via an explicit `run.command`/`run.trainer`. So a
music method inherits the embedding pattern, not the SFT pattern.

The headline **gap** is modality, not wiring. Every dataset loader, every
evaluator verifier (except the corpus-level `retrieval` one), the LLM-judge, the
SynthChat generator, and the Unsloth model loader assume **text JSONL +
causal-LM**. ACE-STEP consumes **binary audio (`.mp3`/`.wav` + sidecar tag/lyric
text)** preprocessed into latents, trains via an **official PyTorch-Lightning
`trainer.py` (not HF Transformers / not Unsloth)**, and is scored by
**audio-specific metrics (CLAP / FAD)** that have no analog in our text/judge eval.
So: *training orchestration is reusable; data plumbing, model loading, and eval
are net-new* — and the net-new parts mirror existing extension seams
(`retrieval` verifier registration, `ModelLoaderRegistry`, the `lora` save
strategy).

The architect's main decision is **how much of ACE-STEP's own `trainer.py` to
wrap vs. reimplement** — our convention only requires a `train_ace_step.py`
entry point that our backends can `python train_ace_step.py --config ...`; what
that script does internally (call ACE-STEP's Lightning trainer, or a thin
HF-style loop) is an architecture choice. This doc enumerates the touch-points;
it does not pick the shape.

---

## 2. The Embedding Precedent (Template for a New Method)

The `embedding` method (SentenceTransformers bi-encoders) is the most recent
NEW-method addition and the closest structural analog to "a model family that is
not a causal-LM." Studying how it was wired end-to-end gives the template.

| Concern | Where embedding wired it | File |
|---------|--------------------------|------|
| Method registered in SSOT | added `"embedding"` to `TRAINING_METHODS` tuple | `shared/utilities/paths.py:11` |
| Local backend method list | added to hardcoded `get_available_methods()` | `tuner/backends/training/rtx_backend.py:106` |
| Cloud backend method list | added to hardcoded `get_available_methods()` | `tuner/backends/training/cloud/hf_jobs_backend.py:123` |
| Trainer entry point | `train_embedding.py` (convention `train_<m>.py`) | `Trainers/embedding/train_embedding.py` |
| Per-method config | `configs/config.yaml` + a **model registry** YAML | `Trainers/embedding/configs/{config,model_registry}.yaml` |
| Dataset loader | method-local loader (NOT the shared SFT loader) | `Trainers/embedding/src/data_loader.py` |
| Eval | **new `retrieval` verifier** self-registered via `@register` | `shared/verifiers/builtins/retrieval_verifier.py` |
| Local-run dispatch | explicit `run.trainer:`/`method: embedding` recipe, NOT the sft/dpo/kto auto-builder | `Trainers/recipes/embedding_bge_base_smoke.yaml` |

**Key lesson for ACE-STEP**: embedding did NOT shoehorn itself into the
causal-LM SFT machinery. It brought its own loader, its own model registry, its
own loss, and its own (corpus-level) eval verifier — reusing only the
*orchestration* (method dispatch, recipe runner, output layout, upload registry).
A music method should follow the same "bring your own data/model/eval, reuse the
orchestration" shape. This is the reasoning chain behind every recommendation
below.

> **Reasoning chain (embedding precedent → ACE-STEP touch-points)**: embedding
> proved a non-causal-LM method integrates by (a) one tuple entry + two backend
> list edits for dispatch, (b) a `Trainers/<m>/train_<m>.py` + `configs/` for the
> trainer, (c) a method-local dataset loader instead of the shared SFT loader, and
> (d) a self-registering corpus-level verifier instead of per-completion text
> assertions. ACE-STEP is *further* from causal-LM than embedding (binary audio,
> Lightning trainer, audio metrics), so it reuses the SAME dispatch/recipe/upload
> seams but needs MORE net-new in the data/model/eval layers — every seam
> embedding touched, ACE-STEP touches too, plus an audio-data bind-mount and an
> audio metric module that embedding did not need.

---

## 3. Local Training Path

### 3.1 Method registration (the SSOT and its duplicates)

The method registry SSOT is a tuple:

```python
# shared/utilities/paths.py:11
TRAINING_METHODS = ("sft", "kto", "grpo", "dpo", "embedding")
```

From this, `paths.py` derives the canonical trainer dir (`Trainers/<m>/`,
`paths.py:13`), legacy dirs, and output dirs (`<m>_output/`, `paths.py:16`).
`get_trainer_root(method)` (`paths.py:88`) and
`get_primary_training_output_dir(method)` (`paths.py:138`) resolve everything by
convention — **no per-method dispatch code**.

**BUT the method list is duplicated (hardcoded) in three more places** that are
NOT derived from the tuple:

| Hardcoded copy | File:line | Notes |
|----------------|-----------|-------|
| RTX local backend | `tuner/backends/training/rtx_backend.py:106` | `return ["sft","kto","grpo","dpo","embedding"]` |
| HF Jobs cloud backend | `tuner/backends/training/cloud/hf_jobs_backend.py:123` | same literal list |
| Cloud method display labels | `tuner/handlers/cloud_train_handler.py:481` | `_load_method_labels()` returns ONLY `{sft, kto}` — already stale (missing grpo/dpo/embedding) |

> **Constraint flag (config-driven rule)**: `_load_method_labels()`
> (`cloud_train_handler.py:474-485`) is a hardcoded Python dict that already
> omits grpo/dpo/embedding. This is an existing drift from the repo's
> config-driven principle. A music method would need to be added here for a
> friendly label, and the architect may want to flag converting this to read from
> a YAML (or from `TRAINING_METHODS`) as cleanup. The GPU-tier sibling method
> (`_load_gpu_tiers`, `cloud_train_handler.py:486`) DOES read from YAML — so the
> repo's own convention is "load from config"; method labels are the exception.

**Trainer script + config name convention:**

```python
# tuner/backends/training/rtx_backend.py:55-62
def _config_filename_for_method(cls, method): return "env_config.yaml" if method=="grpo" else "config.yaml"
def _script_for_config(...):  return "train_env_grpo.py" if grpo else f"train_{config.method}.py"
```

Same in the cloud command builder
(`tuner/backends/training/cloud/_hf_command_builder.py:37-38`):
`train_{method}.py`, with `grpo` the lone special case. So a method `ace_step`
auto-resolves to `Trainers/ace_step/train_ace_step.py` +
`Trainers/ace_step/configs/config.yaml` with **zero dispatch-code changes** once
it is in the three method lists.

### 3.2 Local-run (Docker) — the auto-builder gap

The local-Docker path is `tuner/handlers/local_run_handler.py`. Its compile step
chooses how to build the trainer command:

```python
# tuner/handlers/local_run_handler.py:615-626
if run_cfg.get("command"):
    command = ...                                    # explicit command list
elif method in ("sft", "dpo", "kto"):
    command, ... = self._build_trainer_command(...)  # auto-built from model/dataset/training/lora cfg
else:
    raise LocalRunError("local-run supports run.method: sft, dpo, kto, or an explicit run.command list.")
```

**FINDING — the auto-builder (`_build_trainer_command`, `:478`) only handles
`sft`/`dpo`/`kto`.** `embedding` and `grpo` are NOT in that branch — they must
ride the explicit `run.command`/`run.trainer` path. The embedding smoke recipe
(`Trainers/recipes/embedding_bge_base_smoke.yaml`) sets `method: embedding` +
`trainer: Trainers/embedding/train_embedding.py`; its flag dialect is embedding-
specific, so it cannot reuse the SFT flag set.

`_build_trainer_command` hardcodes a causal-LM flag dialect (LoRA scalars,
`--load-in-4bit`, `--chat-template-kwargs`, `max_seq_length`,
`--split-dataset`, sft-only dashboard toggles — `local_run_handler.py:504-590`).
**None of these map cleanly to ACE-STEP's PyTorch-Lightning `trainer.py` flags**
(`--dataset_path`, `--lora_config_path`, `--learning_rate`, `--max_steps`,
`--devices` — per the model-research doc §2). So a music method should either:
- (a) ride the **explicit `run.command`** path like embedding does (cleanest,
  zero changes to the auto-builder), or
- (b) extend the auto-builder with a music flag dialect (more code, more risk).

> The architect should prefer (a) for the first integration — it mirrors
> embedding and respects "no special-casing the SFT builder."

### 3.3 Docker bind-mount mechanics (the audio data gap)

`local_run_handler.py` copies/binds the trainer dir, `shared/`, `tuner/`, and the
`dataset.local_file` into `/workspace/repo` (`:636-644`). Today `local_file` is a
small text JSONL inside the repo. **An audio corpus (potentially many GB of
`.mp3`/`.wav`) is NOT a single JSONL and likely lives OUTSIDE the repo tree** —
it needs an explicit new bind mount + a `data_dir` config key + path rewrite.
This is the single biggest local-path infra gap (details in §5).

### 3.4 Other local backends

- **Mac (MLX)**: `tuner/backends/training/mac_backend.py:127-130` — hardcoded to
  `train_sft.py`. MLX is causal-LM-specific; ACE-STEP on Mac/MLX is out of scope
  unless v1.5's MLX backend is targeted (see model-research §1; an Open Question).
- **RTX**: `rtx_backend.py` is the primary local GPU path (24GB consumer GPU,
  which the model-research doc confirms ACE-STEP v1 fits).

---

## 4. Cloud (HF Jobs) Path

### 4.1 How a cloud job launches

Entry handler: `tuner/handlers/cloud_train_handler.py`. Flow:
1. Pick provider → backend (`hf_jobs` / `modal` / `runpod`).
2. `method = backend.get_available_methods()` menu (`cloud_train_handler.py:233`)
   + label lookup (`_load_method_labels`, the stale dict).
3. `config = backend.load_config(method)` (`:249`) → reads
   `Trainers/<m>/configs/config.yaml` (`hf_jobs_backend.py:185-186`).
4. `backend.execute(config, ...)` (`:270`) → builds the cloud command and submits.

HF Jobs command construction:
`tuner/backends/training/cloud/_hf_command_builder.py` — `cd` into
`/workspace/repo/Trainers/{get_canonical_trainer_dir_name(method)}` and run
`python {train_<method>.py} --config configs/config.yaml ...`
(`_hf_command_builder.py:236-238, 269`). Job is submitted via
`huggingface_hub.run_job(image, command, flavor)`
(`hf_jobs_backend.py:9-11, 74-75`). **The scripts run unchanged in the cloud** —
HF Jobs supports "the same methods as local RTX" (`hf_jobs_backend.py:119-120`).

### 4.2 Hardware / GPU selection knobs

| Knob | Where | Default |
|------|-------|---------|
| HF Jobs flavor | `hf_jobs_backend.py:58,214` `flavor = hf_config.get("flavor", DEFAULT_FLAVOR)` | `a10g-small` |
| GPU tiers (config-driven) | `Trainers/cloud/cloud_config.yaml` `gpu_tiers:` | budget=a10g-small, standard=a10g-small, perf=a100-large |
| Tier → provider GPU map | `cloud_train_handler.py:_load_gpu_tiers` (`:486`, reads the YAML) | — |
| Config dataclass fields | `tuner/core/config.py:239-249` `CloudTrainingConfig` | `gpu_type`, `hf_flavor`, `timeout_hours`, `cloud_image`, `pip_packages` |

**The max documented tier is A100 80GB single (`a100-large`, $2.50/hr,
`cloud_config.yaml:43`).** Per the model-research doc, ACE-STEP v1 fine-tunes on
a 24GB consumer GPU, so it fits even the budget tier — but diffusion training can
be slow; the architect may want to confirm whether a perf tier is wanted for
real (non-smoke) runs. GPU tiers are config-driven, so **adding/adjusting a tier
is a YAML edit, not code**.

### 4.3 Adding a method/model to the cloud path

A new method auto-flows through the cloud command builder by the `train_<m>.py`
convention (§3.1) once it is in `hf_jobs_backend.py:123`'s list. **There is no
separate "cloud model preset registry"** — the model is just `model.model_name`
in `Trainers/<m>/configs/config.yaml` (and, for embedding, a method-local
`model_registry.yaml`). ACE-STEP should follow embedding's pattern: an
ACE-STEP-specific model registry/config under `Trainers/ace_step/configs/`,
because its "model" is a HF model folder (DCAE + DiT checkpoints), not a single
`unsloth/...` id.

> **Net-new image concern**: every existing recipe rides `unsloth/unsloth:latest`
> (ships transformers, sentence-transformers overlay for embedding). ACE-STEP
> needs PyTorch-Lightning + ACE-STEP's deps + audio libs (torchaudio, ffmpeg).
> The `job.image` / `setup.pip` overlay seam exists
> (`local_run_handler` setup overlay; `CloudTrainingConfig.cloud_image` /
> `pip_packages`, `config.py:242-244`), so this is a **config/recipe concern, not
> a code concern** — but the exact image + pins are an empirical Open Question
> (the repo's documented pattern is "pin from the first working smoke", per the
> embedding recipe's `TBD-PENDING-SMOKE-TEST` comments).

---

## 5. Dataset Handling (the modality impedance mismatch)

### 5.1 How datasets work today (all text JSONL)

- `Datasets/` holds text JSONL only: `behavior_datasets/`, `tools_datasets/`,
  `essay_datasets/`, `kto/`, `grpo/`, `embedding/examples/`, plus loose
  top-level `.jsonl` files with `.metadata.json` sidecars.
- **SFT loader** (`Trainers/sft/src/data_loader.py`): loads via HF
  `datasets.load_dataset("json", ...)` (`:63,175`), renames
  `conversations`→`messages` (`:82-84`), applies `tokenizer.apply_chat_template`
  (`:109`), emits `text`/`input_ids`/`labels`. Pure causal-LM text.
- **Embedding loader** (`Trainers/embedding/src/data_loader.py`): reads JSONL
  triplets/pairs (`_read_jsonl`, `:34`), emits `anchor`/`positive`/`negative`
  string columns. Pure text.
- **Validation infra** (`shared/validation/`): `structure_validator.py` validates
  string/JSON/XML/regex *text content* — it is "format-agnostic" only in the
  sense of text tool-call formats (qwen/mistral/chatml), NOT media.
- **There is no explicit dataset-format enum** — format is inferred from JSON
  keys at load time.

### 5.2 ACE-STEP's data format (from model-research §2)

ACE-STEP v1: **flat, 3 files per sample, matching basenames**:
```
filename.mp3            # audio (binary), 44.1 kHz
filename_prompt.txt     # comma-separated tags (genre, vocal, instruments, mood, bpm, key)
filename_lyrics.txt     # optional lyrics with verse/chorus structure
```
Preprocessed by ACE-STEP's own `convert2hf_dataset.py` into an HF dataset with
row schema `keys, filename, tags(list), speaker_emb_path, norm_lyrics,
recaption(dict)` — **the audio path stays a reference; the DCAE codec encodes
audio→latent on the fly inside the trainer**. v1.5 instead has an explicit
"Preprocess and Generate Tensor Files" step that caches latent tensors (`.pt` +
`manifest.json`). DCAE f8c8 → a 128-frame latent per ~11.88 s segment (~10.77 Hz).
v1.5 also accepts `.wav/.flac/.ogg/.opus` + optional `song.json`
(caption/bpm/keyscale/timesignature/language).

> **Precise gap shape** (confirmed by peer `preparer-acestep`): the loader gap is
> NOT "feed waveforms straight to a Trainer" — it is "**audio files + text
> sidecars in, DCAE codec-preprocess in the middle, latent out**." So the net-new
> seam is (a) get the audio+sidecar corpus to the trainer (bind-mount), and (b)
> run ACE-STEP's codec-preprocess (reuse `convert2hf_dataset.py` / v1.5 tensor
> caching). Our text-JSONL loaders do neither.

### 5.3 Where text/JSONL assumptions break (exact spots)

| Spot | File:line | Why it breaks for audio |
|------|-----------|--------------------------|
| HF JSON loader hard-wired | `Trainers/sft/src/data_loader.py:63,175`; `Trainers/embedding/src/data_loader.py:46-53` | Waveforms can't be JSON rows; need an audio loader (HF `Audio()` feature or path-reference rows resolving `.mp3`/`.wav`) — or just call ACE-STEP's own `convert2hf_dataset.py`. |
| Chat-template assumption | `Trainers/sft/src/data_loader.py:82-114` | Audio data has no `role`/`content`. |
| Docker bind-mount of dataset | `tuner/handlers/local_run_handler.py:636-644` | Today binds one in-repo JSONL; audio corpus (GBs, outside repo) needs a NEW bind mount + `data_dir` key + path rewrite. **Biggest local gap.** |
| Dataset config keys | `Trainers/sft/configs/config.yaml` (`dataset_name`/`dataset_file`/`local_file`) | Music config needs new keys: audio root dir, sample rate (44.1 kHz), tag/lyric fields, latent-cache dir. |
| Method registry tuple | `shared/utilities/paths.py:11` | Add `ace_step` (and its 2 hardcoded duplicates). |

> **Cleanest data shape (architect to decide)**: reuse ACE-STEP's first-party
> `convert2hf_dataset.py` as the preprocessor (produces an HF dataset our backends
> can already point at by name/path), and add a thin `data_dir` bind-mount +
> config-key seam. This avoids reimplementing audio I/O and mirrors embedding's
> "method-local loader" choice. The net-new piece is the **bind-mount + path
> rewrite for an out-of-repo audio corpus**, which no existing method needs.

---

## 6. Generation & Evaluation

### 6.1 Current eval harness (text/judge, with one corpus-level escape hatch)

- Entry: `Evaluator/__main__.py` → `Evaluator/cli.py`; runner `Evaluator/runner.py`.
- Verifiers self-register via a decorator:
  `shared/verifiers/registry.py:22` `@register(type_name)` populates
  `VERIFIER_FACTORIES`; `build_verifier(spec)` dispatches on `spec["type"]`
  (`:46`). Builtins imported for side-effect in
  `shared/verifiers/builtins/__init__.py:11-19`. Registered types: `substring`,
  `structure`, `llm_judge`, `args_match`, `assertions`, `tool_sequence`,
  `retrieval`.
- **`retrieval_verifier.py` is the exact template for a corpus-level metric
  verifier**: it registers via `@register("retrieval")` to stay discoverable, but
  because it scores corpus-level (not per-completion), it exposes a dedicated
  `evaluate_retrieval()` entry point and makes `verify()` raise
  `NotImplementedError`. Heavy deps (`sentence_transformers`, `faiss`) are lazily
  imported inside the method to keep `shared/` import-pure.
- Runner branches to corpus-level eval when a scenario sets
  `metadata["retrieval_config"]` (`Evaluator/runner.py:317-323` →
  `_evaluate_retrieval_case`, `:476`; config built by `_build_retrieval_config`,
  `:517`). This sits as a **sibling to the per-completion loop**.
- **LLM-judge** (`shared/judge/judge_service.py`): text-rubric based; cannot
  score audio.
- **Generation** (`tuner/handlers/generate_handler.py`): orchestrates SynthChat
  to produce `messages` text conversations. `inference_handler.py`: text
  inference. **No music-generation analog exists.**

### 6.2 Where music-gen + audio-eval attach

- **Audio eval seam (mirror `retrieval` exactly)**: create
  `shared/verifiers/builtins/audio_verifier.py` with `@register("audio")`, expose
  a corpus-level `evaluate_audio()` (CLAP/FAD are dataset-level like retrieval),
  lazily import audio deps. Wire a new branch in `Evaluator/runner.py:317`-style
  keyed on `metadata["audio_config"]` + an `_evaluate_audio_case` sibling. Add an
  audio result field to the eval record + status ladder.
- **Net-new metric impl**: there is no `shared/ml/` audio analog (the existing
  `shared/ml/retrieval_metrics.py` is numpy recall@k/nDCG/MRR). FAD/CLAP/audio
  quality must be net-new.
- **Music generation step**: entirely net-new (no SynthChat/inference analog).
  Per model-research §, ACE-STEP inference renders audio via vocoder→waveform
  (WAV/MP3 via ffmpeg) — a generation handler would shell out to ACE-STEP's
  inference, not our text LLM client.

> **Eval Open Question** (peer `preparer-acestep` confirmed there is **no single
> drop-in scalar** like our LLM-judge, and **no first-party eval script wired for
> fine-tune QA**): the standard audio-domain family is **FAD** (Fréchet Audio
> Distance, distributional realism), **CLAP score** (text↔audio alignment), plus
> lyric/PER alignment and human A/B for musicality; the ACE-STEP paper emphasizes
> lyric alignment + musical coherence + RTF (speed). CLAP/FAD require reference
> distributions + a CLAP/VGGish model. For a first integration, a smoke "did it
> produce valid audio of expected length/sample-rate (44.1 kHz)" check is the
> pragmatic verifier, with FAD/CLAP as a follow-up net-new dependency.

---

## 7. Shared Infra Reuse

| Component | File(s) | Reusable for music? |
|-----------|---------|---------------------|
| **Method dispatch** | `shared/utilities/paths.py:11`; `rtx_backend.py:106`; `hf_jobs_backend.py:123` | **Yes (convention seam)** — add `ace_step` to the tuple + 2 backend lists; `train_ace_step.py` + `configs/config.yaml` then auto-resolve. |
| **Recipe runner / output layout** | `local_run_handler.py`, `paths.py:138` | **Yes** via explicit `run.command`/`run.trainer` (embedding pattern). |
| **GPU tiers** | `Trainers/cloud/cloud_config.yaml` | **Yes** (config-driven; YAML edit only). |
| **Experiment tracking** | `shared/experiment_tracking/registry.py` (`RunRegistry`, `register_run`); `schema.py:33` `RunRecord` | **Yes, fully** — `run_type` is an unvalidated free string (`schema.py:49,77`), so `run_type="ace_step"` works with zero schema changes. |
| **Upload — `lora` strategy** | `shared/upload/strategies/lora.py:14-51` | **Yes (raw adapter push)** — it `shutil.copytree`s adapter files, no model load, no GPU. A music LoRA *adapter* (the unmerged `.safetensors` + config) ships through this as-is. NOTE (peer-confirmed): a *merged* publish differs — see merged row. ACE-STEP's HF model repo is ~8.28 GB of safetensors (DCAE + DiT), so the upload framework's HF-repo plumbing fits, but the merge step does not. |
| **Upload — orchestrator/uploader/registry** | `shared/upload/orchestrator.py:78`; `uploaders/huggingface.py`; `strategies/registry.py:25` | **Yes** if a new no-merge save strategy is registered + GGUF converter skipped. |
| **Upload — merged_16bit / merged_4bit / GGUF** | `shared/upload/strategies/{merged_16bit,merged_4bit}.py`; `converters/` | **Reusable-with-adaptation, NOT as-is** — these assume LoRA-merge into a base causal-LM via `save_pretrained_merged` (Unsloth). ACE-STEP merge is a **diffusion-DiT adapter merge** (different mechanics), so a merged publish needs a net-new ACE-STEP-specific merge step. GGUF is llama.cpp/LM-specific — skip entirely for audio. |
| **Model loading** | `shared/model_loading/unsloth_loader.py`; `registry.py:28` `ModelLoaderRegistry` | **Registry pattern reusable; Unsloth loader NOT** — `unsloth_loader.py` wraps `FastLanguageModel`/`FastVisionModel` + `save_pretrained_merged`. ACE-STEP (diffusion DiT, PyTorch Lightning) needs a net-new loader registered via `ModelLoaderRegistry.register("ace_step", ...)`. |
| **`Trainers/shared/ui`** | `Trainers/shared/ui/training_progress.py` | **Yes** (progress display is method-agnostic). |
| **Config/YAML utilities** | `shared/utilities/{yaml_loader,env,paths}.py` | **Yes, all method-agnostic.** |
| **Dataset loaders** | `Trainers/{sft,embedding}/src/data_loader.py` | **No** (all `load_dataset("json")` text) — net-new audio loader or reuse ACE-STEP's `convert2hf_dataset.py`. |
| **LLM-judge / SynthChat gen** | `shared/judge/`, `tuner/handlers/generate_handler.py` | **No** — text-only; net-new audio gen + scoring. |
| **Method display labels** | `cloud_train_handler.py:481` `_load_method_labels` | **Add `ace_step` entry** (and note: this dict is hardcoded + already stale — config-driven cleanup candidate). |

---

## 8. Integration-Shape Options (for the architect — enumerated, NOT decided)

All options share the **mandatory dispatch edits** (the convention seam):
add `ace_step` to `TRAINING_METHODS` (`paths.py:11`),
`rtx_backend.py:106`, `hf_jobs_backend.py:123`, and a label in
`cloud_train_handler.py:481`; create `Trainers/ace_step/train_ace_step.py` +
`Trainers/ace_step/configs/config.yaml`. They differ in how the trainer and data
are realized.

### Option A — Thin wrapper around ACE-STEP's official `trainer.py` (embedding-style)
- `train_ace_step.py` = a thin adapter that parses our `--config` and shells out
  to ACE-STEP's PyTorch-Lightning `trainer.py` with translated flags
  (`--dataset_path`, `--lora_config_path`, `--learning_rate`, `--max_steps`,
  `--devices`).
- Data: reuse ACE-STEP's `convert2hf_dataset.py` as the preprocessor; add a
  `data_dir` config key + Docker bind-mount for the audio corpus.
- Local-run: explicit `run.command`/`run.trainer` recipe (NOT the sft/dpo/kto
  auto-builder).
- **Touch-points**: dispatch edits (4 files) · `Trainers/ace_step/` (trainer
  wrapper + configs + model registry) · `local_run_handler` bind-mount seam for
  `data_dir` (the one genuine handler edit) · new recipe under `Trainers/recipes/`
  · upload via `lora` strategy (reused) · new `audio` verifier
  (`shared/verifiers/builtins/audio_verifier.py`) · audio metric module.
- **Pros**: least net-new code; rides the most-documented ACE-STEP path; mirrors
  embedding. **Cons**: an extra subprocess layer; ACE-STEP's Lightning trainer's
  output layout must be mapped to our `<m>_output/` + experiment tracking.

### Option B — Native HF-style trainer reimplementation
- `train_ace_step.py` implements the diffusion training loop using ACE-STEP's
  model classes directly (no Lightning), matching our SFT trainer's CLI/output
  shape more closely.
- **Touch-points**: same dispatch + data seams as A, plus a net-new model loader
  registered in `ModelLoaderRegistry` and a hand-written training loop.
- **Pros**: tightest fit to our output/tracking/dashboard conventions.
  **Cons**: most net-new code; re-derives what ACE-STEP's `trainer.py` already
  does; higher correctness risk for a diffusion objective.

### Option C — Auto-builder extension (music flag dialect)
- Add a `music`/`ace_step` branch to `_build_trainer_command`
  (`local_run_handler.py:478`) with an ACE-STEP flag dialect, so recipes don't
  need an explicit `run.command`.
- **Pros**: recipes look uniform with sft/dpo/kto. **Cons**: special-cases the
  SFT-centric builder (against the embedding precedent's "bring your own
  command"); more handler code to maintain. Lower-priority than A.

> **Recommendation framing (not a decision)**: Option A is the lowest-risk first
> integration and the closest analog to how embedding was added. The architect
> should weigh A vs B on how much they want ACE-STEP's Lightning output to conform
> to our experiment-tracking/dashboard conventions; Option C is an ergonomics
> add-on orthogonal to A/B.

---

## 9. Repo Constraints That Affect Integration

- **Config-driven, no hardcoding**: method labels (`cloud_train_handler.py:481`)
  and the duplicated method lists (`rtx_backend.py:106`, `hf_jobs_backend.py:123`)
  are existing hardcoded drift from this rule. Adding `ace_step` perpetuates the
  duplication; the architect may scope a cleanup (derive lists from
  `TRAINING_METHODS`, move labels to YAML). ACE-STEP's own config (model paths,
  data_dir, sample rate, LoRA rank) MUST be YAML, not hardcoded.
- **No backward-compat shims**: when adding the method, edit the three lists
  directly; do not add a compat/alias layer.
- **`.skills/` is canonical**: a new `ace-step-training` (or `music-training`)
  skill should be authored under `.skills/`, then mirrors synced via
  `python3 .skills/scripts/sync_skill_trees.py` — do NOT hand-edit
  `.agents/skills` or `.claude/skills`.
- **No `/tmp` outputs**: training outputs go to `<m>_output/` and artifacts under
  `toolset-training-artifacts/` per existing convention (`paths.py:16`,
  `local_run_handler.py:567-568`).

---

## 10. Reusable vs Net-New (consolidated)

| Layer | Reusable as-is | Net-new for ACE-STEP |
|-------|----------------|----------------------|
| Method dispatch | convention seam (`paths.py:11` + 2 backend lists + 1 label) | `Trainers/ace_step/{train_ace_step.py, configs/config.yaml, configs/model_registry.yaml}` |
| Local-run orchestration | recipe runner via explicit `run.command` (embedding pattern) | `data_dir` bind-mount + path rewrite in `local_run_handler.py:636-644` |
| Cloud (HF Jobs) | command builder, `run_job(image,cmd,flavor)`, GPU tiers (YAML) | ACE-STEP Docker image + pip overlay (empirical pins); confirm perf tier |
| Dataset | — (all text JSONL) | audio loader OR reuse ACE-STEP `convert2hf_dataset.py`; new dataset config keys; out-of-repo audio corpus mount |
| Model loading | `ModelLoaderRegistry` plug-in pattern | `ace_step` loader (Unsloth/causal-LM contract breaks) |
| Eval | verifier `@register` seam + corpus-level sibling-branch pattern (`retrieval` template) | `audio` verifier + CLAP/FAD/audio-quality metrics (no `shared/ml/` analog) |
| Generation | — (text-only) | music-generation step (shell out to ACE-STEP inference) |
| Upload | `lora` copytree strategy (raw adapter push), orchestrator, HF uploader, registry | register no-merge save strategy; net-new diffusion-DiT *merged* publish if wanted; skip GGUF + causal-LM merged_* |
| Experiment tracking | **fully** (`run_type` unvalidated string) | — |
| Shared utilities / progress UI | **fully** | — |

---

## 11. Open Questions (for architect / cross-lane)

1. **v1 vs v1.5 as the trainer target.** Model-research recommends v1 (Apache 2.0,
   most-documented `trainer.py`) for the first integration; v1.5 (MIT, higher
   quality, one-click LoRA + MLX) is a follow-up. The integration shape (Option A
   wrapper) is similar for both, but the exact flags/preprocessor differ. Confirm
   target before architecture.
2. **Reuse `convert2hf_dataset.py` vs write a native audio loader.** Reusing it
   is lowest-effort and produces an HF dataset our backends point at by path, but
   couples us to ACE-STEP's preprocessor + its tensor-cache layout. Architect call.
3. **Audio corpus location + size.** Where does the training audio live (HF
   dataset? local dir?), and how large? This determines the bind-mount/transfer
   design (the `data_dir` seam) and whether `transfer: bind` vs `copy` is viable.
4. **Eval metric for v1.** CLAP/FAD need reference models + distributions;
   model-research did not pin a metric. Decide whether the first verifier is a
   pragmatic "valid-audio-of-expected-length/sample-rate" smoke vs. a full CLAP/FAD
   harness.
5. **Cloud image + pin set.** ACE-STEP needs PyTorch-Lightning + torchaudio +
   ffmpeg + its own deps on top of (or instead of) the Unsloth image. Pins must be
   captured empirically from the first working smoke (repo convention), not
   invented.
6. **Method-list duplication cleanup — in scope?** Should this work also derive
   `get_available_methods()` from `TRAINING_METHODS` and move `_load_method_labels`
   to YAML, or leave that drift and just add `ace_step` to all copies?
7. **MLX/Mac path.** v1.5 has an MLX backend; our Mac backend is hardcoded to
   `train_sft.py` (`mac_backend.py:127`). Is Mac local training in scope, or
   RTX-local + HF-Jobs-cloud only (matching the task's stated targets)?

---

## 12. Source Index (verified file:line references)

- Method SSOT + path conventions: `shared/utilities/paths.py:11,13,16,88,138`
- Local backend method list + script/config naming:
  `tuner/backends/training/rtx_backend.py:55-62,106,128`
- Cloud backend method list + flavor: `tuner/backends/training/cloud/hf_jobs_backend.py:58,123,185-186,214`
- Cloud command builder (script-name convention): `tuner/backends/training/cloud/_hf_command_builder.py:37-38,236-238,269`
- Local-run dispatch + auto-builder gap + bind-mount: `tuner/handlers/local_run_handler.py:478,504-590,615-626,636-644`
- Cloud train handler (method menu, labels, GPU tiers): `tuner/handlers/cloud_train_handler.py:233,474-499`
- GPU tiers config: `Trainers/cloud/cloud_config.yaml:12,22,29,41-43,85`
- Cloud config dataclass: `tuner/core/config.py:203-296`
- Embedding precedent: `Trainers/embedding/{train_embedding.py,src/data_loader.py,configs/*}`, `Trainers/recipes/embedding_bge_base_smoke.yaml`
- SFT data loader (text assumptions): `Trainers/sft/src/data_loader.py:63,82-114,175`
- Verifier registry + retrieval template: `shared/verifiers/registry.py:22,46`, `shared/verifiers/builtins/{__init__.py:11-19,retrieval_verifier.py}`, `Evaluator/runner.py:317-323,476,517`
- Upload framework: `shared/upload/{orchestrator.py:78,strategies/{lora.py:14-51,registry.py:25},uploaders/huggingface.py}`
- Model loading: `shared/model_loading/{unsloth_loader.py,registry.py:28}`
- Experiment tracking: `shared/experiment_tracking/{registry.py,schema.py:33,49,77}`
- Sibling model research: `docs/preparation/ace-step-model-research.md`
