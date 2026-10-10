# Image LoRA on Modal (`image_lora` method)

Status: implemented 2026-10-10 (first run: Qwen-Image-2512, 189 images, H100).
Code: `Trainers/image_lora/`. Tests: `tests/image_lora/` (Modal SDK faked; no spend).

## What it is

A training method that wraps an outside trainer, like `Trainers/ace_step`, instead
of reimplementing training. It trains a diffusion-model LoRA with
[ostris/ai-toolkit](https://github.com/ostris/ai-toolkit) (MIT) on one Modal GPU.
The default base model is **Qwen-Image-2512** (Apache 2.0), which ai-toolkit lists
as `qwen_image:2512` (the `qwen_image` architecture with `Qwen/Qwen-Image-2512` as
`name_or_path`). LoRAs are saved with ComfyUI key prefixes and load in ComfyUI's
Qwen-Image-2512 workflow with `LoraLoaderModelOnly`.

It deliberately does **not** use the packaged LLM Modal path
(`tuner/training/modal_host_*`, the coordinator and Foundation broker). That path
exists to authenticate packaged LLM workloads end to end; an image LoRA from a
private dataset needs a short, auditable lifecycle instead (below).

```
build-dataset  notes + images ──► <dataset>/images/*.png + *.txt, manifest.json, tokens.md
plan           config.yaml + model_registry.yaml + dataset ──► ai-toolkit job + estimate   (no cloud)
probe          L4 smoke test: driver, CUDA, torch, ai-toolkit imports, hf-token key        (~$0.05)
launch         create run Volume, upload dataset, spawn detached ──► run_state.json
status         poll the call, remote status.json, train.log tail (best effort, see below)
fetch          samples + logs + final LoRA (+ chosen checkpoints) ──► sha256 + safetensors check
cleanup        delete run Volume, stop app — refused until fetch verified
run            launch + wait + fetch + cleanup
```

With no subcommand (`python Trainers/image_lora/train_image_lora.py`, which is how
the local training menu calls `train_{method}.py`) it only runs `plan`.

## Resource lifecycle (the cleanup fix)

An audit found that every packaged Modal run leaves behind three Volumes
(control, artifacts and a per-run model cache with its own copy of the base
weights), a deployed app, and two Secrets, one of them a per-run copy of the HF
token (`tuner/training/modal_host_runtime.py: prepare_modal_runtime_for_host`).
`image_lora` keeps the resource set minimal and closes it:

| Resource | Name | Created | Deleted |
|---|---|---|---|
| Base-weights cache Volume | `synaptic-image-lora-base-cache` (config) | `from_name(create_if_missing=True)`; shared by all runs | never by a run |
| Per-run Volume | `synaptic-image-lora-run-<run_id>` | `Volume.objects.create(allow_existing=False)`, so a name clash can never adopt old data | `cleanup`, only after `fetch` verified every requested file |
| HF token | existing `hf-token` Secret, `required_keys=["HF_TOKEN"]` | never | never |
| App | `synaptic-image-lora`, ephemeral, `app.run(detach=True)` | per launch | stops itself when the call ends; `cleanup` stops it if not |

The worker downloads the base model with `snapshot_download(repo, revision=<sha>)`
into `HF_HOME` on the cache Volume (pinned commit recorded in `run_state.json`)
and commits it; later runs on the same revision skip the download (Qwen-Image-2512
is 57.7 GB; the first download took 202 s). Nothing about the HF token reaches
argv, files, logs or the run state.

`cleanup` is idempotent and records each step in `run_state.json`; `--force`
exists for discarding a failed run's resources deliberately.

### Pattern for the LLM methods (not refactored here)

1. One named, long-lived model-cache Volume per environment, keyed by repo and
   pinned revision inside the HF cache layout, instead of a cache Volume per run.
2. Reference existing Secrets by name (`Secret.from_name(..., required_keys=...)`);
   never copy credential values into per-run Secrets.
3. Per-run storage created exclusively, deleted by an explicit step that requires
   a verified download (hashes against a worker-written manifest).
4. Ephemeral detached apps for one-shot jobs; deployed apps only where a stable
   function identity is actually required, with an explicit stop step.
5. A single local state file that names every created resource, so cleanup and
   audits never need to list or guess.

## Spend cap

`launch --max-usd X` refuses if the estimate exceeds `X`, and sets the Modal
function timeout to `X / hourly_rate`, so a runaway job cannot spend more than `X`
on GPU + CPU + memory. The estimate model lives in `configs/config.yaml`
(`estimate:`), with list prices read by `modal billing rates`. A timeout kills the
call before the final commit, so keep the cap above the estimate with margin.

## Dataset builder

`src/dataset_builder.py` is generic; every project-specific rule (note globs,
approved statuses, token naming, aliases, relations, boilerplate patterns,
holdouts, caption fixes) is in a recipe YAML kept with the dataset, outside the
repository (`configs/dataset_recipe.example.yaml` documents the keys). Decisions,
in order, each logged per file in `manifest.json`:

1. Approval: moment images need an approved status and an existing linked image;
   entity images are dropped when the entity status is excluded.
2. Holdouts (evaluation scenes), layer filename patterns, and real alpha
   transparency (compositing layers would teach plain backgrounds).
3. Same-file merges, then perceptual-hash near-duplicates (64-bit DCT pHash).
4. Downscale oversized upscales; convert to grayscale when the image is
   effectively monochrome: low chroma, or a single tint (each channel predicted
   from luminance alone, residual RMS below a threshold — catches sepia/cyanotype).
5. Captions: trigger phrase, entity tokens, relation phrases, then the cleaned
   prompt (structured JSON prompts reduced to scene fields; negative and
   instruction sentences dropped; names replaced by tokens). Linked characters are
   kept only when the prompt actually shows them; objects and locations are also
   detected in the text. Every library image not selected is logged too.

## Known limits

- Mid-run Volume visibility is best effort. `Volume.commit()` reloads afterwards
  and a reload is refused while the container holds files open (ai-toolkit keeps
  some open), so `status` may show an empty log tail; use
  `modal app logs <app_id>` for the live view. The final commit after the trainer
  exits is complete and is what `fetch` verifies.
- `status` reads the call result non-blockingly; a Modal `FunctionTimeoutError`
  (spend cap hit) is reported as failed and leaves outputs uncommitted.
- Sampling during training uses ai-toolkit's sampler at fixed seeds; it is a
  progression view, not the ComfyUI pipeline used for evaluation.
