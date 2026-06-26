# ACE-STEP Model Research (PREPARE phase)

> **Scope**: This document characterizes the **ACE-Step music-generation foundation model itself** —
> architecture, training/fine-tuning support, inference, hardware, licensing, and ecosystem maturity.
> It does **not** cover how to wire ACE-Step into this repo's plumbing (that is the parallel
> `preparer-integration` workstream). All non-obvious claims are cited with live source URLs;
> facts were verified against official repos / model cards / the arXiv paper rather than model memory.
>
> **Researcher**: preparer-acestep · **Date**: 2026-06-24

---

## Executive Summary

**ACE-Step is a real, actively-developed, open-source text-to-music foundation model, and it is
strongly feasible to fine-tune both locally (24 GB consumer GPU) and on cloud GPUs.** Two generations
exist:

| | **ACE-Step v1** | **ACE-Step 1.5** (current) |
|---|---|---|
| Repo | [`ace-step/ACE-Step`](https://github.com/ace-step/ACE-Step) | [`ace-step/ACE-Step-1.5`](https://github.com/ace-step/ACE-Step-1.5) |
| Generative model | ~3.5B diffusion DiT + DCAE + linear transformer | 2B / 4B-XL DiT + optional Qwen3-based LM planner (0.6B–4B) |
| **License** | **Apache 2.0** | **MIT** |
| Training code | **Official `trainer.py` (PyTorch Lightning) + LoRA** | One-click LoRA/LoKr in UI + REST + community Side-Step toolkit |
| Status | Superseded but stable; cleanest documented trainer | Current; faster, multi-backend (CUDA/ROCm/XPU/MLX) |

**The two findings the lead flagged as co-priority #1:**

1. **Official training + LoRA support: CONFIRMED (first-party).** v1 ships an official `trainer.py`
   (PyTorch Lightning) with a documented dataset format and a `convert2hf_dataset.py` preprocessor,
   plus a released reference LoRA (Chinese rap). v1.5 adds one-click LoRA/LoKr training in its
   Gradio UI and a REST training endpoint. This is **not** community-only — it is first-party.
   ([v1 TRAIN_INSTRUCTION.md](https://github.com/ace-step/ACE-Step/blob/main/TRAIN_INSTRUCTION.md),
   [v1.5 LoRA tutorial](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/LoRA_Training_Tutorial.md))

2. **Licensing: GO for commercial use (Synaptic Labs).** Both generations are permissive and
   commercial-friendly — **v1 = Apache 2.0**, **v1.5 = MIT** — verified by fetching both `LICENSE`
   files directly. No non-commercial / research-only clauses, no appended field-of-use restrictions.
   The only caveats are ordinary content-responsibility notes on the model card (don't generate
   copyrighted material, disclose AI involvement). **No licensing blocker; no ALERT warranted.**
   ([v1 LICENSE](https://raw.githubusercontent.com/ace-step/ACE-Step/main/LICENSE),
   [v1.5 LICENSE](https://raw.githubusercontent.com/ace-step/ACE-Step-1.5/main/LICENSE))

**Recommendation for the later build decision**: target **ACE-Step v1 (Apache 2.0, 3.5B)** as the
primary fine-tuning surface for a first integration, because its `trainer.py` + flat
`mp3 / _prompt.txt / _lyrics.txt` dataset format is the **most explicitly documented and
most repo-friendly** (HF-dataset conversion, clear LoRA config). Treat **v1.5** as the higher-quality
forward path once the v1 pipeline is proven — but note v1.5's training is more UI/REST-centric and
its preprocessing (tensor caching, `manifest.json`) is less fully documented in text form. See
Open Questions for what to confirm before committing to v1.5 as the trainer target.

---

## 1. Model Identity & Architecture

**Canonical project (verified):** *ACE-Step: A Step Towards Music Generation Foundation Model*,
**co-led by ACE Studio and StepFun**. The v1 `LICENSE` copyright line reads
*"Copyright [2025] Timedomain Inc. and stepfun"* — so the "ACE Studio" lineage maps to **Timedomain Inc.**,
and **StepFun** is the second party. Paper authors: Junmin Gong, Sean Zhao, Sen Wang, Shengyuan Xu,
Joe Guo (arXiv 2506.00045, submitted 2025-05-28).
([GitHub README](https://github.com/ace-step/ACE-Step/blob/main/README.md),
[arXiv:2506.00045](https://arxiv.org/abs/2506.00045),
[v1 LICENSE](https://raw.githubusercontent.com/ace-step/ACE-Step/main/LICENSE))

### Architecture (v1, ~3.5B)
A **diffusion-based** generator (not an autoregressive LM) trained with a **continuous-time
flow-matching (FM)** objective, composed of:

- **Sana's Deep Compression AutoEncoder (DCAE)** — compresses audio into a compact latent the
  diffusion model operates in. **f8c8 setting (8× compression, channel=8), ≈10.77 Hz temporal
  resolution**; encoder produces a **128-frame** latent for ~11.88 s input segments. HF folder
  `music_dcae_f8c8`, alongside a `music_vocoder`.
- **Lightweight linear transformer (Linear DiT)** — **24 layers** (per the paper; note secondary
  sources said "28" — the paper's "8th of 24" is authoritative), with Simplified Adaptive Layer
  Normalization (AdaLN-single) and 1D-convolutional feed-forward layers. Hidden dim / head count /
  RoPE details are **not stated** in the paper. HF folder `ace_step_transformer`.
- **Semantic alignment (REPA)** via **MERT** (24 kHz mono) and **m-HuBERT** (16 kHz mono)
  representations during training, used to speed convergence (SSL loss λ=1.0, dropped to 0.01 for the
  final 100k fine-tune steps).
- **Text encoder**: `umt5-base` (multilingual mT5) is bundled in the HF repo.

([arXiv HTML §3/Tables 3–4](https://arxiv.org/html/2506.00045v1),
[HF file tree](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B/tree/main))

> **Resolved from the full paper (arXiv HTML)**: the **"~3.5B" total** is stated only in the
> speed-comparison context (§5.2.2); the paper gives **no per-component param breakdown** (DCAE /
> Linear DiT / text-encoder split is unspecified). DCAE compression (f8c8, ≈10.77 Hz, 128-frame
> latent) and the 24-layer DiT are confirmed. **Audio sample rate = 44.1 kHz** (pretrain data
> resampled to 44.1 kHz; MERT/m-HuBERT branches use 24 kHz / 16 kHz internally). Flow-matching
> timestep schedule: **logit-normal, mean=0.0, std=1.0, shift=3.0** (Table 4). See §7a for full specs.

### Architecture (v1.5 — current)
Moves to a **hybrid two-stage design**:
- **LM planner (optional)** — a **Qwen3-based** language model (0.6B / 1.7B / 4B variants) that turns a
  short user query into a full "song blueprint" (metadata, lyrics, captions) via Chain-of-Thought.
- **Diffusion Transformer (DiT) backbone** — the generative decoder, in **2B** (base/sft/turbo) and
  **4B "XL"** (xl-base/xl-sft/xl-turbo, released 2026-04-02) sizes.

([v1.5 README](https://github.com/ace-step/ACE-Step-1.5),
[v1.5 site](https://ace-step.github.io/ace-step-v1.5.github.io/))

### Modalities / tasks (v1)
- Text→music from **tags / descriptions / scenarios**.
- **Lyrics conditioning** (lyric-aligned vocals).
- **Audio2Audio / repainting** with mask constraints (e.g. `Ace_Step_4x_a2a.json` ComfyUI workflow).
- **Lyric editing** via flow-edit.
- LoRA-delivered task heads: **Lyric2Vocal**, **Text2Samples** (instrumentals), **RapMachine** (Chinese rap).
- Roadmap: **StemGen** (multi-track), **Singing2Accompaniment ControlNet**.

([v1 README](https://github.com/ace-step/ACE-Step/blob/main/README.md))

---

## 2. Training & Fine-Tuning Support  *(co-priority #1)*

### v1 — official trainer (most documented, recommended starting surface)
**Framework: PyTorch Lightning.** Official instructions live in
[TRAIN_INSTRUCTION.md](https://github.com/ace-step/ACE-Step/blob/main/TRAIN_INSTRUCTION.md).

**Dataset format (flat, 3 files per sample, matching basenames):**
```
data/
├── filename.mp3            # audio (MP3)
├── filename_prompt.txt     # comma-separated tags: genre, vocal type, instruments, mood, tempo, key
└── filename_lyrics.txt     # optional but recommended; verse/chorus song structure
```
Example prompt line:
`"melodic techno, male vocal, electronic, emotional, minor key, 124 bpm, synthesizer, driving, atmospheric"`

**Preprocess → HF dataset:**
```bash
python convert2hf_dataset.py --data_dir "./data" --repeat_count 2000 --output_name "zh_lora_dataset"
```
Output row schema: `keys` (str), `filename` (str), `tags` (list[str]), `speaker_emb_path` (str, empty),
`norm_lyrics` (str), `recaption` (dict, empty).

**Train (incl. LoRA):**
```bash
python trainer.py
```
Key flags: `--dataset_path ./zh_lora_dataset`, `--learning_rate 1e-4`, `--epochs -1`,
`--max_steps 2000000`, `--devices 1`, `--lora_config_path config/zh_rap_lora_config.json`.

**LoRA config (`config/zh_rap_lora_config.json`):** `r: 16`, `lora_alpha: 32`,
`target_modules: ["linear_q","linear_k","linear_v","to_q","to_k","to_v","to_out.0"]`.

Reference released LoRA: [`ACE-Step/ACE-Step-v1-chinese-rap-LoRA`](https://huggingface.co/ACE-Step/ACE-Step-v1-chinese-rap-LoRA)
(see `ZH_RAP_LORA.md`). Full fine-tune is implied by the same trainer (omit `--lora_config_path`);
the documented, exercised path is **LoRA**.

> **No sample-rate / duration limit is stated** in v1's TRAIN_INSTRUCTION (DCAE consumes whatever the
> codec front-end resamples to). Flag as Open Question.

### v1.5 — one-click LoRA/LoKr + REST + community toolkit
- **In-UI workflow**: scan a dataset dir → review/auto-label with the LM → "Preprocess and Generate
  Tensor Files" → "Train LoRA". Dataset dir is per-file:
  `song.mp3` (or `.wav/.flac/.ogg/.opus`) + `song.lyrics.txt` (required) + optional `song.json`
  (`caption, bpm, keyscale, timesignature, language`) + optional `song.caption.txt`. Preprocessing
  encodes audio→latents (VAE codec) and text→embeddings, caching tensors (a `manifest.json`-indexed
  tensor dir).
- **CLI/REST (LoKr)**: `POST http://localhost:8001/v1/training/start_lokr`.
- LoRA defaults: `learning_rate 1e-4`, `train_batch_size 1` (gradient_accumulation 4),
  epochs heuristic "~100 songs → 500 epochs; 10–20 songs → 800 epochs", `save_every_n_epochs 5`.
  Rank/alpha not stated in the tutorial.
- **Community Side-Step toolkit** ([koda-dernet/Side-Step](https://github.com/koda-dernet/Side-Step)):
  CLI/wizard/GUI training scripts, **LoKr** (Kronecker low-rank) adapters, corrected timestep sampling,
  Fisher-information adaptive per-module ranks (`fisher_map.json`), VRAM optimization.

([v1.5 LoRA tutorial](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/LoRA_Training_Tutorial.md),
[v1.5 training overview (DeepWiki)](https://deepwiki.com/ace-step/ACE-Step-1.5/6.1-training-overview-and-workflow),
[Side-Step](https://github.com/koda-dernet/Side-Step))

---

## 3. Inference / Generation

**v1:**
- **CLI**: `acestep --port 7865` (launches Gradio).
- **Python API**: pip-install from GitHub, import pipeline.
- **Gradio UI** (Win/macOS/Linux), **ComfyUI** node ([`billwuhao/ComfyUI_ACE-Step`](https://github.com/billwuhao/ComfyUI_ACE-Step)),
  hosted on **Replicate** ([lucataco/ace-step](https://replicate.com/lucataco/ace-step)).
- Key params: duration (`-1` = random), inference steps, guidance/CFG scale, seed, scheduler type,
  CFG type, ERG settings, variance/retake. Output: rendered audio (vocoder → waveform; format not
  pinned in README — typically WAV/MP3 via ffmpeg).

**v1.5:** `uv run acestep` (Gradio, :7860), `uv run acestep-api` (REST, :8001), Python `inference.py`,
interactive `cli.py`, plus a VST3 plugin and community ComfyUI nodes. Duration 10s–600s, batch up to 8,
diffusion steps 8 (turbo) / 50 (base/sft), CFG scale, sampling method.

([v1 README](https://github.com/ace-step/ACE-Step/blob/main/README.md),
[v1.5 README](https://github.com/ace-step/ACE-Step-1.5))

**HF model IDs & sizes:**
- v1: [`ACE-Step/ACE-Step-v1-3.5B`](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B) — **total repo ≈ 8.28 GB**,
  components `ace_step_transformer`, `music_dcae_f8c8`, `music_vocoder`, `umt5-base`. Also on ModelScope.
- v1.5 (under the [`ACE-Step` HF org](https://huggingface.co/ACE-Step)): DiT 2B ≈ **4.7 GB**, XL 4B ≈ **9 GB**;
  LM models `acestep-5Hz-lm-0.6B / -1.7B / -4B`. *(v1.5 per-file sizes are approximate — see Open Questions.)*

---

## 4. Hardware Requirements

### Inference
| Device | v1 RTF (27 steps) | 1-min-audio time |
|---|---|---|
| RTX 4090 | 34.5× | 1.74 s |
| A100 | 27.3× | 2.20 s |
| RTX 3090 | 12.8× | 4.70 s |
| M2 Max | 2.27× | 26.4 s |

v1 reduced **max inference VRAM to ~8 GB** (with `--torch_compile --cpu_offload --overlapped_decode`).
v1.5 scales from **≤6 GB** (2B-turbo, INT8 + CPU offload) up to **≥24 GB** (XL-sft + 4B LM, best quality),
and runs on **CUDA / ROCm (AMD) / Intel XPU / Apple MLX**. v1.5 inference: "<2 s/song on A100, <10 s on RTX 3090."
([v1 README RTF table](https://github.com/ace-step/ACE-Step/blob/main/README.md),
[v1.5 README VRAM tiers](https://github.com/ace-step/ACE-Step-1.5))

### Training / LoRA  *(feasibility verdict)*
- **v1 LoRA**: fits comfortably on a **24 GB** consumer GPU (3090/4090); the released reference LoRA
  was trained this way.
- **v1.5 LoRA**: **16 GB minimum** ("longer songs may OOM"), **≥20 GB recommended** (usage ~17 GB).
  Stated "~8 songs, ~1 hr on RTX 3090" for a quick LoRA; **LoKr** can cut that to ~5 min.
- **Cloud (HF Jobs) sizing**: a single **A100 40 GB** (or L40S/A10G-class 24 GB+) is ample headroom for
  LoRA on either generation; full fine-tune of the 3.5B/4B DiT would want **A100 80 GB** or multi-GPU.

**Verdict: local 24 GB LoRA is clearly feasible; cloud LoRA is cheap (single mid-tier GPU). Full
fine-tune is cloud-only (80 GB-class).**
([v1.5 LoRA tutorial — VRAM](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/LoRA_Training_Tutorial.md))

---

## 5. Licensing  *(co-priority #1 — go/no-go gate)*

**Both generations are permissive and commercial-use-OK. Verified by fetching the raw LICENSE files.**

| | License | Copyright line | Commercial use |
|---|---|---|---|
| **v1** | **Apache License 2.0** (standard, no appended field-of-use clauses) | "Copyright [2025] Timedomain Inc. and stepfun" | ✅ Yes |
| **v1.5** | **MIT License** | "Copyright (c) 2026 ACEStep" | ✅ Yes |

- **Model weights**: the v1 HF model card is tagged **Apache 2.0** and states **no explicit
  commercial-use restriction**. v1.5 weights live under the MIT-licensed org.
- **Caveats (ordinary, not blockers)**: the v1 card advises against "generating copyrighted content
  without permission", and asks users to "verify originality" and "disclose AI involvement" — these are
  responsible-use notes, **not** license restrictions.
- **Training-data provenance**: the corpus used to *train* ACE-Step is **not** disclosed in the sources
  reviewed; this is a model-provenance unknown, not a license term on *our* use of the weights. Flag it
  for the user's awareness (music-copyright sensitivity) but it does not gate Apache/MIT usage rights.

**No HALT/ALERT on licensing.** ([v1 LICENSE](https://raw.githubusercontent.com/ace-step/ACE-Step/main/LICENSE),
[v1.5 LICENSE](https://raw.githubusercontent.com/ace-step/ACE-Step-1.5/main/LICENSE),
[v1 HF card](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B))

---

## 6. Ecosystem Maturity & Gotchas

- **Stack**: PyTorch; v1 uses **PyTorch Lightning** for training; v1.5 uses **`uv`** for env/locking
  (`pyproject.toml` + `uv.lock`) and **vLLM** as an LM backend at higher VRAM tiers.
- **Python**: v1 ≥ 3.10; v1.5 **3.11–3.12** (3.12 required for ROCm-on-Windows).
- **Audio deps**: **ffmpeg** required (audio I/O); torchaudio implied. Exact torch/CUDA pins are in the
  repos' lockfiles, **not** spelled out in READMEs (v1 example uses cu126 wheels on Windows).
- **Windows gotcha**: needs `pip install triton-windows`; macOS needs `--bf16 false`.
- **Active & broad ecosystem**: official ComfyUI node, community Windows ports
  ([sdbds/ACE-Step-for-window](https://github.com/sdbds/ACE-Step-for-window),
  [sdbds/ACE-Step-1.5-for-windows](https://github.com/sdbds/ACE-Step-1.5-for-windows)), Replicate hosting,
  a community UI ([fspecii/ace-step-ui](https://github.com/fspecii/ace-step-ui)), the **Side-Step**
  training toolkit, and an [awesome-ace-step](https://github.com/ace-step/awesome-ace-step) index.
- **Two-generation churn**: v1 vs v1.5 differ in trainer ergonomics, env manager (pip vs uv), and
  dataset preprocessing. Picking a target generation **before** architecture design avoids rework — the
  cleanest text-documented trainer is **v1**; the higher-quality/faster model is **v1.5**.
- **Integration-relevant friction for *this* repo** (flagged for `preparer-integration` / architect):
  this codebase is text-LLM-centric (Unsloth/TRL/transformers, JSONL text datasets). ACE-Step adds a
  **diffusion/audio** stack — DCAE codec, vocoder, ffmpeg, audio tensors, PyTorch-Lightning *or* uv/vLLM —
  that does **not** overlap the existing SFT/KTO/GRPO trainers. A **separate pipeline** (as the user
  intends) is the right call; do not try to fold it into the text trainers.

([v1.5 README](https://github.com/ace-step/ACE-Step-1.5),
[v1 README](https://github.com/ace-step/ACE-Step/blob/main/README.md))

---

## 7a. Authoritative Numeric Specs (v1, from arXiv full paper)

Mined from the [arXiv HTML](https://arxiv.org/html/2506.00045v1) (§3 + Tables 3–4 + §5.2.2) to resolve
Open Question #1. Numbers explicitly absent from the paper are marked **not stated**.

| Spec | Value | Source |
|---|---|---|
| Total params | **~3.5B** (stated only in the speed-comparison context §5.2.2) | §5.2.2 |
| Per-component param split | **not stated** (no DCAE / DiT / text-encoder breakdown) | — |
| DCAE compression | **f8c8** — 8× compression, channel=8, **≈10.77 Hz** temporal resolution | Table 3 |
| DCAE latent | **128 frames** for ~**11.88 s** input segments | Table 3 |
| Linear DiT layers | **24 layers** (AdaLN-single, 1D-conv feed-forward) | §3.3.2 |
| DiT hidden dim / heads / RoPE | **not stated** | — |
| Audio sample rate | **44.1 kHz** (pretrain); MERT branch 24 kHz mono, m-HuBERT branch 16 kHz mono | §3 (data) |
| Max duration | up to **4 min**; variable-length training, no hard cap stated | Abstract |
| Training data | **~100,000 hours**, **~1.8M tracks**, **19 languages** (English-majority) | §3 (data) |
| Flow-matching schedule | logit-normal **mean=0.0, std=1.0, shift=3.0**; MSE on latents; SSL λ=1.0 (→0.01 final 100k steps) | Table 4 |
| Training compute | **120× A100** (15 nodes × 8), global batch 120, **460k pretrain + 240k finetune steps**, ~264 h wall-clock (~31,680 A100-GPU-hours) | §3 (training) |

**Correction to §1**: secondary sources said the DiT is "28 layers"; the paper says **24**. Trust 24.

---

## 7. Reasoning Chain (how the findings connect)

1. **Identity verified** → ACE-Step is genuinely open (public repos + HF weights + arXiv paper),
   so per-model research is valid and the integration premise holds.
2. **Two licenses both permissive (Apache 2.0 / MIT, fetched from source)** → the go/no-go gate is
   **GO** for Synaptic Labs commercial use; no ALERT. This removes the single biggest project-killing risk.
3. **Official first-party trainer exists (v1 `trainer.py` + LoRA; v1.5 UI/REST LoRA/LoKr)** → fine-tuning
   is supported without reverse-engineering, so the build is a *pipeline-integration* problem, not a
   *research-a-training-recipe* problem.
4. **VRAM envelope (v1 LoRA on 24 GB; v1.5 LoRA 16–20 GB)** → **local 3090/4090 LoRA is feasible**, and
   **cloud LoRA needs only a single mid-tier GPU**; full fine-tune is cloud-80 GB. This makes the
   feasibility verdict *positive for both local and cloud*.
5. **v1's flat `mp3/_prompt/_lyrics` format + HF-dataset conversion is the most repo-friendly** →
   recommend **v1 as the first integration target**, v1.5 as the forward path once proven.
6. **Audio/diffusion stack is disjoint from the repo's text trainers** → confirms the user's "separate
   music pipeline" framing; architect should design it standalone, sharing only generic infra
   (experiment tracking, HF upload, cloud-job orchestration) — details for `preparer-integration`.

---

## 8. Open Questions / Unknowns

1. **~~Exact v1 numeric specs~~ — RESOLVED** (see §7a). Mined from the arXiv full paper: DCAE f8c8
   (≈10.77 Hz, 128-frame latent), 24-layer Linear DiT, 44.1 kHz audio, ~100k hrs / 1.8M tracks /
   19 languages, 120×A100 / 700k steps. *Residual unknown*: no per-component param breakdown and no
   DiT hidden-dim/head/RoPE numbers are stated in the paper — only the ~3.5B total.
2. **~~Training sample-rate~~ — RESOLVED (44.1 kHz, §7a)**. *Residual*: max-**duration** limit for
   *training* is still unspecified (v1.5 only warns "longer songs may OOM" at 16 GB). Needs a code
   read / empirical check if long-form training is a goal.
3. **v1 full fine-tune (non-LoRA) viability & exact flags** — implied by the same `trainer.py` but not
   explicitly documented; only LoRA is exercised. *Confirm via the trainer source.*
4. **v1.5 per-component checkpoint sizes & `manifest.json` tensor schema** — approximate; the
   preprocessing/tensor-cache format is under-documented in text. *Read `acestep/acestep_v15_pipeline.py`
   / preprocessing code if v1.5 is chosen as the trainer.*
5. **Training-data provenance / copyright** — the corpus ACE-Step was trained on is undisclosed. Not a
   license blocker on *our* use of the Apache/MIT weights, but worth surfacing to the user given
   music-copyright sensitivity.
6. **diffusers / HF Trainer integration** — neither generation appears to use HF `diffusers` or `Trainer`;
   both ship bespoke training (Lightning / uv+REST). *Confirms a separate pipeline; no drop-in HF Trainer reuse.*
7. **Exact torch/CUDA pins** — live in repo lockfiles, not READMEs. *Pull from `uv.lock` /
   `requirements.txt` at integration time to match the repo's CUDA environment.*

---

## Source Index

**Official (highest authority):**
- v1 repo / README — https://github.com/ace-step/ACE-Step · https://github.com/ace-step/ACE-Step/blob/main/README.md
- v1 training instructions — https://github.com/ace-step/ACE-Step/blob/main/TRAIN_INSTRUCTION.md
- v1 LICENSE (Apache 2.0) — https://raw.githubusercontent.com/ace-step/ACE-Step/main/LICENSE
- v1 HF model card / files — https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B · /tree/main
- v1 reference LoRA — https://huggingface.co/ACE-Step/ACE-Step-v1-chinese-rap-LoRA
- v1.5 repo / README — https://github.com/ace-step/ACE-Step-1.5
- v1.5 LoRA tutorial — https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/LoRA_Training_Tutorial.md
- v1.5 LICENSE (MIT) — https://raw.githubusercontent.com/ace-step/ACE-Step-1.5/main/LICENSE
- HF org — https://huggingface.co/ACE-Step
- arXiv paper — https://arxiv.org/abs/2506.00045 · https://arxiv.org/pdf/2506.00045
- Project sites — https://ace-step.github.io/ · https://ace-step.github.io/ace-step-v1.5.github.io/

**Community / ecosystem (lower authority, used for ecosystem mapping):**
- Side-Step training toolkit — https://github.com/koda-dernet/Side-Step
- v1.5 training overview (DeepWiki) — https://deepwiki.com/ace-step/ACE-Step-1.5/6.1-training-overview-and-workflow
- ComfyUI node — https://github.com/billwuhao/ComfyUI_ACE-Step
- Windows ports — https://github.com/sdbds/ACE-Step-for-window · https://github.com/sdbds/ACE-Step-1.5-for-windows
- Community UI — https://github.com/fspecii/ace-step-ui
- awesome-ace-step — https://github.com/ace-step/awesome-ace-step
- Replicate hosting — https://replicate.com/lucataco/ace-step
