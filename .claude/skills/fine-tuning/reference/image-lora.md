# Image LoRA (ai-toolkit on Modal)

Contract and resource lifecycle: `docs/architecture/image-lora-modal.md`.
Code: `Trainers/image_lora/` (`train_image_lora.py`, `src/`, `configs/`).

Use this method to teach a diffusion model a set of named characters, objects,
locations and a house style from an illustration library. Default base model:
Qwen-Image-2512 (Apache 2.0, ai-toolkit `qwen_image` arch). Training runs on one
Modal GPU; nothing trains on the host.

## Workflow

```bash
T=Trainers/image_lora/train_image_lora.py
# 1. Dataset (outside the repo; images never go in git). The recipe holds every
#    project-specific rule; start from configs/dataset_recipe.example.yaml.
python $T build-dataset --recipe /data/proj/recipe.yaml --out /data/proj --dry-run --show 30
python $T build-dataset --recipe /data/proj/recipe.yaml --out /data/proj --contact-sheets
#    Read the contact sheets and at least 20 captions against their images; fix
#    rules or add recipe caption_fixes, rebuild. Write /data/proj/sample_prompts.yaml.
# 2. Plan: renders the ai-toolkit job and the estimate (no cloud calls)
python $T plan --dataset /data/proj --show-job
# 3. Probe once per image change (~1 min of L4)
python $T probe --gpu L4
# 4. Launch detached with a spend cap (sets the provider timeout)
python $T launch --dataset /data/proj --output-dir /data/proj/runs/r1 --max-usd 20
python $T status --state /data/proj/runs/r1/run_state.json
modal app logs <app_id from run_state.json>      # live view (Volume tail is best effort)
# 5. Fetch (final LoRA + any checkpoints you pick from the samples), then clean up
python $T fetch --state /data/proj/runs/r1/run_state.json --checkpoint 2500
python $T cleanup --state /data/proj/runs/r1/run_state.json
```

`run` chains launch → wait → fetch → cleanup for unattended use. `cleanup` refuses
until `fetch` verified every requested file (sha256 against the worker manifest and
a safetensors header check).

## Captions and tokens

- Every caption starts with the trigger phrase, then entity tokens, then relation
  phrases, then the cleaned prompt: `myproj ink style, go_char, kanabo_obj,
  go_char holding kanabo_obj. go_char leans on his kanabo_obj like a cane ...`.
- Tokens come from note names (`The Ash Fields` → `ash_fields_loc`,
  `Gō's Kanabō` → `kanabo_obj`); `tokens.md` maps each token to its entity and
  image count. Prompt with the same tokens at inference.
- `cache_text_embeddings: true` means ai-toolkit cannot inject a trigger word, so
  the builder writes it into every caption.
- Moment links name everyone in a story beat; the recipe's
  `mention_required_types` keeps a linked character only when the prompt shows it.

## Sizing (ai-toolkit Qwen-Image guidance, adjusted)

- ai-toolkit's example (`train_lora_qwen_image_24gb.yaml`): rank 16, lr 1e-4,
  2000 steps, buckets 512/768/1024, 3-bit transformer for 24 GB cards.
- On an 80 GB H100/A100 the bf16 transformer fits once text embeddings are cached
  (the encoder is unloaded), so `quantize_transformer: false`.
- Each listed resolution is a separate pass over the dataset
  (`toolkit/config_modules.py` splits datasets by resolution): with N images and
  3 buckets one epoch is 3N steps at batch 1.
- Rank 32 for ~30 entities plus a style on ~200 images; 3-5k steps gives each
  image roughly 15-25 views. Save and sample every 500 steps and pick the
  checkpoint from the fixed-seed samples, not the last step by default.

## Cost

`plan` prints the estimate from `configs/config.yaml: estimate`. Replace
`seconds_per_step` with the measured value after a run. First measured run
(2026-10-10, H100, 189 images, 768/1024/1328 buckets, rank 32): see the
run notes in `docs/architecture/image-lora-modal.md`.

## Gotchas

- Do not return torch objects from Modal methods; the host has no torch to
  unpickle them (`torch.__version__` is a `TorchVersion`).
- The training image pins ai-toolkit and torch in `src/modal_app.py`
  (torch 2.13 cu130 needs a 580+ driver; the probe prints it).
- Exclude compositing layers (transparent or chroma-key backgrounds) from
  datasets; they teach plain backgrounds.
