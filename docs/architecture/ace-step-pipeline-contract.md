# ACE-STEP v1.5 Pipeline — Build Contract (Phase-0 De-Risk)

> **Status**: Phase-0 contract lock. This is the authoritative interface contract the CODE
> fan-out (backend-coder, devops-engineer, test-engineer) builds against **in parallel,
> without guessing**.
> **Author**: architect · **Date**: 2026-06-24 · **Branch**: `feat/ace-step-v15-music-pipeline`
> **Plan**: [`docs/plans/ace-step-v15-music-pipeline-plan.md`](../plans/ace-step-v15-music-pipeline-plan.md) (APPROVED)
> **Scope rule**: this LOCKS contracts, it does not redesign. Any needed plan change is FLAGGED, never silently applied.
>
> **User decisions locked (2026-06-24)**: (1) full pipeline = train + generate + eval;
> (2) 3-list method-duplication dedup folded in (derive from SSOT, labels → YAML);
> (3) `data_dir` is GENERIC config-driven — reusable by any researcher, NO hardcoded personal path.
>
> All repo `file:line` references verified against this worktree on 2026-06-24.
> The EXACT ACE-STEP v1.5 flag/cache facts (§1.3, §4) are sourced from `preparer-acestep`
> reading the real `ace-step/ACE-Step-1.5` repo file `acestep/training_v2/cli/args.py` — NOT
> asserted from the plan's subcommand-level summary. §1.3 + §4 are **✅ FILLED**; §7 generation
> entry is **✅ RESOLVED to `cli.py`** (only its exact flag spellings are a Phase-4 build-time
> `--help` confirm). **The contract is FULLY LOCKED for the CODE fan-out.** THREE mandatory
> build-time verifies are flagged inline and all assume §9 ACE-STEP-repo provisioning first:
> (a) §1.3 BUILD STEP-0 byte-confirm before training argv (blocking): `train.py fixed --help`
> (subcommand) + `train.py --help` (root parser — `preprocess` is a FLAG, not a subcommand;
> see §1.3 ⚠️); (b) §7 `cli.py --help` before generation argv (Phase 4); (c) §4.2 2–3-sample
> preprocess smoke for `.pt` shapes. **Structural correction baked in**: v1.5 `dataset.json` is a
> SINGLE index file, NOT v1's per-file `.txt` sidecars (§4.1).

---

## 0. How the coders use this doc

| Specialist | Owns (builds against §) | Hard dependency |
|------------|-------------------------|-----------------|
| **backend-coder** | §1 wrapper API + flag table ✅ · §2 config schema ✅ · §3 model registry · §4 dataset-prep ✅ · §5 data_dir/cache_dir (local rewrite) ✅ · §8 registration + dedup ✅ | §1/§4 flag+cache facts ✅ FILLED — **must byte-confirm `train.py {fixed,preprocess} --help` at build (§1.3)** |
| **devops-engineer** | §5 cloud HF-dataset pull seam · §2 (image/pip overlay keys) · Docker image (separate) · **§9 ACE-STEP v1.5 repo provisioning (pinned)** | dep/runtime pins (empirical, pin-from-smoke) |
| **test-engineer** | §6 audio-verifier interface (AudioConfig + evaluate_audio) ✅ · §7 generation contract (entry=`cli.py` ✅; flag spellings Phase-4) · §8 registration tests | §6 AudioConfig shape ✅ locked · §4 fixture sizing ✅ (~12 s clip floor) |

**Config-driven SACROSANCT**: every ACE-STEP param (model variant, sample_rate, max_duration,
rank, alpha, lr, steps, data_dir, cache_dir) is a YAML key — NO hardcoding, NO scenario special-casing.
**No backward-compat shims**: edit the registration sites directly. **`.skills/` is canonical**:
the `ace-step-training` skill is authored under `.skills/`, mirrors synced via
`python3 .skills/scripts/sync_skill_trees.py`.

---

## 1. `train_ace_step.py` Wrapper API + config→flag translation table

**Pattern**: Adapter. `Trainers/ace_step/train_ace_step.py` is a thin subprocess wrapper around
ACE-STEP v1.5's own headless CLI. It does NOT reimplement training (Option A, plan §Architecture).

### 1.1 Our CLI (the surface our backends invoke)

```
python train_ace_step.py --config configs/config.yaml [--data-dir <path>] [--dry-run]
```

| Flag | Required | Meaning |
|------|----------|---------|
| `--config` | yes | Path to our `config.yaml` (§2). Single source of all run params. |
| `--data-dir` | no | Override for `dataset.data_dir` (§5). CLI wins over YAML when both set. |
| `--dry-run` | no | Print the translated ACE-STEP `train.py` argv(s) and EXIT 0 WITHOUT executing. De-risks flag translation with zero GPU/audio cost (plan CODE step 3). |

### 1.2 Wrapper responsibilities (contract)

1. Parse `--config` → our config dataclass (§2).
2. Resolve `data_dir`/`cache_dir` (§5), apply container path-rewrite when running under Docker.
3. **Stage 1 — preprocess**: build + invoke the root-parser **`--preprocess` flag** form, `python train.py --preprocess <flags>` (audio → `.pt` cache) — `preprocess` is a FLAG, not a subcommand (see §1.3 ⚠️). Skip if cache present & `preprocess.force: false`.
4. **Stage 2 — train**: build + invoke `train fixed <flags>` (LoKr/LoRA train, cache → adapter).
5. **Error handling (NON-NEGOTIABLE, plan risk row)**: each subprocess call MUST
   capture returncode, tee stderr to the run log, and **raise on nonzero** so a failed
   `train.py` can NEVER look like success. No silent swallow.
6. **Output remap**: map ACE-STEP's output dir → `ace_step_output/` (convention from
   `paths.py` `CANONICAL_OUTPUT_DIRS`, `<method>_output`) + register a run via
   `shared/experiment_tracking` (`run_type="ace_step"` — free string, reused as-is).

### 1.3 config → `train.py` flag translation table  ✅ FILLED (preparer-acestep, from `acestep/training_v2/cli/args.py`)

> Source: the parsers are centralized in **`acestep/training_v2/cli/args.py`**
> (`build_fixed_standalone_parser` / `build_root_parser`) — `train.py` / `train_fixed.py`
> only import them. Flag names + defaults below are VERBATIM from that file.
>
> ### ⚠️ BUILD STEP-0 (MANDATORY, BLOCKING — backend-coder, before locking ANY subprocess argv)
>
> The flag names/defaults below were read by preparer-acestep via raw-GitHub fetch +
> summarization, **NOT** by executing `--help`. So the FIRST build action (step-0, before any
> wrapper argv is written) is to run, against the **provisioned ACE-STEP v1.5 repo** (§9
> build-dependency note) at the pinned commit:
>
> ```
> python train.py fixed --help     # {fixed} IS a real subcommand → subcommand --help works
> python train.py --help           # preprocess is NOT a subcommand — it's an args.preprocess
>                                   #   FLAG on the ROOT parser; its --help lives at root level
> ```
>
> ⚠️ **`preprocess` is a ROOT-PARSER FLAG, not a subcommand** (devops verified live against
> `vendor/ACE-Step-1.5` @ v0.1.8 `args.py`). Only `{vanilla, fixed, estimate}` are actual
> subcommands. So `python train.py preprocess --help` does NOT behave as a subcommand `--help`
> (argparse will treat `preprocess` as an unexpected positional). The Stage-1 invocation in §1.2
> and the Stage-1 table below are written as `train.py preprocess` shorthand for readability, but
> the REAL argv is the root-parser flag form (e.g. `python train.py --preprocess <flags>`).
> backend-coder-2 (who owns the preprocess argv) MUST byte-confirm the exact flag spelling
> (`--preprocess` vs `preprocess`) and the root-level preprocess flags from `python train.py --help`,
> NOT from a `preprocess` subcommand help that does not exist.
>
> byte-confirm every flag spelling + default in the two tables below. If any flag differs,
> correct the table and ping architect BEFORE locking the translation. The wrapper `--dry-run`
> mode (§1.1) is where this assertion lives. (Verify-by-execution discipline — a recipe arg
> silently dropped at any layer runs at the trainer default and corrupts the run invisibly;
> cf. the `tuner_training_arg_surface` lesson.) **The argv translation is NOT "locked" until
> step-0 passes.**

**Stage 1 — `python train.py fixed --preprocess`** (✅ byte-confirmed backend-coder-2 vs args.py + train.py:79-187; one of `--audio-dir`/`--dataset-json` required; `--checkpoint-dir` required):

> ### ⚠️ STAGE-1 INVOCATION (corrected — the subcommand token IS required)
> `--preprocess` is `store_true` on the common arg group (args.py:318); `train.py:_dispatch`
> (87-88) EARLY-RETURNS to `_run_preprocess(args)` before reading the subcommand. BUT `main()`
> only enters direct-CLI mode `if _has_subcommand()` (train.py:121) — a BARE `python train.py
> --preprocess` drops into the interactive WIZARD instead. So the real stage-1 argv MUST carry a
> subcommand token to satisfy `_has_subcommand()` + the required subparser: **`python train.py
> fixed --preprocess <preprocess-flags>`**. The `fixed` token is inert here (the `--preprocess`
> early-return exits before any training); it exists only to reach direct-CLI mode. Stage-2 train
> is the same `fixed` subcommand WITHOUT `--preprocess`.

| Our config key (§2) | ACE-STEP `preprocess` flag | Default | Notes |
|---------------------|----------------------------|---------|-------|
| `dataset.data_dir` (raw audio) | `--audio-dir` | None | raw audio source |
| `dataset.dataset_json` (labeled) | `--dataset-json` | None | labeled dataset JSON (alt to `--audio-dir`) |
| `dataset.cache_dir` | `--tensor-output` | None | output dir for `.pt` files |
| `model.registry_name` → base ckpts | `--checkpoint-dir` | (required) | base model checkpoints root |
| `model.variant` | `--model-variant` | `"turbo"` | shared flag |
| `preprocess.max_duration` | `--max-duration` | `240.0` | seconds; longer → OOM risk |
| `preprocess.device` | `--device` | `"auto"` | **pin `cuda` for training** (mps broken, see §9) |
| `preprocess.precision` | `--precision` | `"auto"` | choices: auto/bf16/fp16/fp32 |

> Resample to 48 kHz STEREO is **automatic inside preprocess** (so `target_sr`/`channels` in our
> config are documentation/assertion values, NOT passed as flags — there is no `--sr`/`--channels`
> flag). See §4.1.

**Stage 2 — `python train.py fixed`** (`--checkpoint-dir`, `--dataset-dir`, `--output-dir` required):

| Our config key (§2) | ACE-STEP `fixed` flag | Default | Notes |
|---------------------|------------------------|---------|-------|
| `dataset.cache_dir` | `--dataset-dir` | (required) | dir of preprocessed `.pt` files |
| `output.adapter_dir` | `--output-dir` | (required) | LoRA/LoKr output → remapped to `ace_step_output/` |
| `model.registry_name` → base ckpts | `--checkpoint-dir` | (required) | base model checkpoints root |
| `model.variant` | `--model-variant` | `"turbo"` | 2b/4b-xl selected via variant + checkpoint-dir |
| `training.learning_rate` | `--lr` / `--learning-rate` | `1e-4` | |
| `training.epochs` | `--epochs` | `100` | **EPOCHS, not steps — there is NO `--max-steps`** (see §2 note) |
| `training.train_batch_size` | `--batch-size` | `1` | |
| `training.gradient_accumulation` | `--gradient-accumulation` | `4` | |
| `training.seed` | `--seed` | `42` | |
| `training.save_every` | `--save-every` | `10` | checkpoint every N epochs |
| `training.precision` | `--precision` | `"auto"` | choices auto/bf16/fp16/fp32 |
| `training.device` | `--device` | `"auto"` | **pin `cuda`** for training |
| `adapter.type` | `--adapter-type` | `"lora"` | choices: `lora` (PEFT) / `lokr` (LyCORIS). **Our config default = `lokr`** (plan: LoKr ~5min vs LoRA ~1hr) |
| `adapter.rank` (when `lora`) | `--rank` / `-r` | `64` | ⚠️ **LoRA-ONLY** (args.py:285-287 group "used when --adapter-type=lora") — v1.5 default 64 (v1 was 16) |
| `adapter.alpha` (when `lora`) | `--alpha` | `128` | ⚠️ **LoRA-ONLY** — v1.5 default 128 (v1 was 32) |
| `adapter.rank` (when `lokr`) | `--lokr-linear-dim` | `64` | ⚠️ **LoKr** maps `rank`→`--lokr-linear-dim` (args.py:295). `--rank` is IGNORED for lokr |
| `adapter.alpha` (when `lokr`) | `--lokr-linear-alpha` | `128` | ⚠️ **LoKr** maps `alpha`→`--lokr-linear-alpha` (args.py:296). `--alpha` is IGNORED for lokr |
| `adapter.target_modules` | `--target-modules` (nargs=+) | `[q_proj,k_proj,v_proj,o_proj]` | ⚠️ v1.5 PEFT-style names (v1 used linear_q/to_q…) |

> ### ⚠️ DELTA-2 — adapter-type-aware translation (✅ byte-confirmed backend-coder-2 vs args.py)
> `--rank`/`--alpha` live in the **LoRA-only** arg group (args.py:285 "used when
> --adapter-type=lora"). For `--adapter-type=lokr` the knobs are **`--lokr-linear-dim`** (default
> 64) + **`--lokr-linear-alpha`** (default 128) (args.py:294-301). So a translation that emits
> `--rank/--alpha` while `adapter.type==lokr` would be **SILENTLY IGNORED** (the
> `tuner_training_arg_surface` corruption mode).
>
> **MECHANISM — LOCKED (architect, 2026-06-24), independent of the default question below:**
> `config_translation.py` MUST branch on `adapter.type` — `lora` → emit `--rank/--alpha`;
> `lokr` → emit `--lokr-linear-dim/--lokr-linear-alpha` from the SAME two config scalars. The
> config keys are RENAMED `lora.*` → **`adapter.*`** (adapter-agnostic) so a coder cannot misread
> them as LoRA-universal. `target_modules` is shared (not in the lora-only group split). The
> remaining `--lokr-factor`/`--lokr-decompose-both`/`--lokr-use-tucker`/etc. (args.py:297-301)
> stay at upstream defaults — NOT surfaced in our config (add later only if a recipe needs them).
> This branch-on-type translation is correct under EITHER default, so backend-coder-2 can lock the
> translation logic now; only the default literal below is pending.
>
> **DEFAULT — ✅ DECIDED: `lokr` (USER-CONFIRMED 2026-06-24).** The default was genuinely a
> user-facing product choice (LoRA and LoKr produce DIFFERENT trained artifacts — different
> adapter math), so it was escalated to the user rather than pinned by inference. The user RULED
> **LoKr** — rationale: ~5min / smaller / more parameter-efficient (the plan's named fast path),
> chosen over LoRA's standard-PEFT familiarity. **`lora` is retained as the documented alternative**
> via the `adapter.type` branch (set `adapter.type: lora` and the translation emits `--rank/--alpha`
> instead). This matches backend-coder-2's interim → ZERO code churn; DELTA-2 is fully closed
> (mechanism + default both locked).
>
> *(History for the record — the signals that made this a user call, not an inference: LoRA-leaning
> = plan "LoRA/LoKr" ordering L24, "PEFT/LoKr adapter" upload L69, "rank/alpha" YAML vocab L31/L171;
> LoKr-leaning = "LoKr ~5min vs LoRA ~1hr" L27, P1 smoke "Real train (LoKr ~5min)" L114; the plan
> never stated a config default.)*

> **Subcommands**: `vanilla | fixed | estimate` (devops verified live @ v0.1.8). **`preprocess`
> is NOT a subcommand — it is an `args.preprocess` flag on the ROOT parser** (run via
> `python train.py --preprocess …`). We use the `--preprocess` root flag + the `fixed` subcommand
> (FixedLoRATrainer). `vanilla` = alt trainer; `estimate` = Fisher gradient-sensitivity for
> adaptive ranks (optional, not in scope). In-process entry points exist if ever preferred over
> subprocess: `run_fixed(args)->int` and `main()->int` — but subprocess remains the chosen shape
> (Option A; env-island + churn-decoupling rationale stands).

---

## 2. `config.yaml` schema

**Nesting mirrors the embedding precedent** (`Trainers/embedding/configs/config.yaml`):
top-level `model` / `training` / `dataset` sections, plus ACE-STEP-specific `preprocess` /
`lora` / `output`. ALL audio + training params are YAML keys (config-driven SACROSANCT).

```yaml
# Trainers/ace_step/configs/config.yaml  (schema — values illustrative)
method: ace_step

model:
  registry_name: ace-step-v15-2b        # → resolved via model_registry.yaml (§3) → --checkpoint-dir
  variant: "turbo"                        # → --model-variant. choices incl: turbo | sft | base (default "turbo")

dataset:
  data_dir: ""                           # GENERIC out-of-repo audio corpus root (§5) → --audio-dir. Empty → documented default landing dir.
  dataset_json: ""                       # ALT to data_dir: labeled dataset JSON → --dataset-json (one of the two required)
  cache_dir: ""                          # .pt tensor cache dir (§5) → --tensor-output (preprocess) / --dataset-dir (train). Empty → default under data_dir.
  dataset_name: ""                       # CLOUD: HF dataset id pulled at runtime (§5). Mutually-informative with data_dir.

preprocess:
  target_sr: 48000                       # 48 kHz — ASSERTION/doc only; preprocess resamples automatically (NO --sr flag, §4.1)
  channels: 2                            # STEREO — ASSERTION/doc only (auto-resample; NO --channels flag)
  max_duration: 240                      # → --max-duration (default 240.0s); OOM-guard for long songs
  device: "cuda"                         # → --device. PIN cuda for training (mps broken, §9). default upstream "auto"
  precision: "auto"                      # → --precision. choices auto/bf16/fp16/fp32
  force: false                           # wrapper-level: re-run preprocess even if .pt cache present (no upstream flag)

training:
  learning_rate: 1.0e-4                  # → --lr / --learning-rate (default 1e-4)
  epochs: 500                            # → --epochs (default 100). NB: EPOCHS, not steps (no --max-steps). Tutorial: ~100 songs→500ep; 10–20 songs→800ep
  train_batch_size: 1                    # → --batch-size (default 1)
  gradient_accumulation: 4               # → --gradient-accumulation (default 4)
  save_every: 10                         # → --save-every (checkpoint every N epochs, default 10)
  precision: "auto"                      # → --precision (choices auto/bf16/fp16/fp32)
  device: "cuda"                         # → --device (PIN cuda)
  seed: 42                               # → --seed (default 42)

adapter:                                 # (renamed from `lora:` — adapter-agnostic; §1.3 DELTA-2)
  type: "lokr"                           # → --adapter-type. ✅ SHIPPED DEFAULT = lokr (USER-CONFIRMED 2026-06-24, §1.3 DELTA-2); lora retained as documented alt via the type branch. lokr (LyCORIS) | lora (PEFT). upstream default "lora"
  rank: 64                               # type==lora → --rank/-r ; type==lokr → --lokr-linear-dim (translation branches on type!)
  alpha: 128                             # type==lora → --alpha  ; type==lokr → --lokr-linear-alpha (translation branches on type!)
  target_modules: ["q_proj","k_proj","v_proj","o_proj"]  # → --target-modules (shared; v1.5 PEFT-style names; v1 used linear_q/to_q…)

output:
  adapter_dir: ""                        # → --output-dir. empty → ace_step_output/<run> (convention; remapped by wrapper)
```

> **Schema authority**: section/key NAMES above are LOCKED for the coders, and every value now
> maps to a CONFIRMED ACE-STEP flag (§1.3) — except `preprocess.target_sr`/`channels` (assertion-
> only; auto-resample, no flag) and `preprocess.force` (wrapper-level cache skip, no upstream flag).
> **CORRECTION vs plan**: the plan's `training.max_steps` is WRONG — ACE-STEP v1.5 `fixed` has NO
> `--max-steps`; control is by `--epochs`. Schema uses `epochs`. (Flagged to lead.)

---

## 3. `model_registry.yaml` schema

ACE-STEP's "model" is a multi-file HF folder (DCAE/VAE + DiT + optional LM planner), NOT a
single `unsloth/...` id — so it follows embedding's **method-local model registry** precedent
(`Trainers/embedding/configs/model_registry.yaml`), not a single `model_name` string.

```yaml
# Trainers/ace_step/configs/model_registry.yaml  (schema)
models:
  ace-step-v15-2b:
    hf_id: "ACE-Step/Ace-Step1.5"        # RESOLVED #32 — AGGREGATE repo = DEFAULT (2B turbo DiT + 1.7B LM + VAE + text-encoder, all IN-FOLDER)
    variant: "turbo"                     # → --model-variant. 2B choices: turbo|base|sft (args.py:182, VARIANT_DIR_MAP:18)
    revision: "19671f40…"                # RESOLVED #32 — pin to this rev
    components:                          # multi-file folder layout — get exact subfolder
      dit: "<dit subfolder>"             #   names from the PROVISIONED repo tree (not invented)
      vae_dcae: "<vae/dcae subfolder>"
      lm_planner: "<optional lm subfolder>"
  ace-step-v15-xl:                       # XL = 4B — OPT-IN (separate repo, NOT in the aggregate)
    hf_id: "ACE-Step/acestep-v15-xl-turbo"  # RESOLVED #32 — XL is a SEPARATE repo: acestep-v15-xl-{base,sft,turbo}
    variant: "xl_turbo"                  # → --model-variant. XL choices: xl_turbo|xl_base|xl_sft (args.py:182). ⚠️ NOT "4b-xl"
    revision: "<xl-pinned-ref>"          # pin from the xl-* repo at provisioning (NOT the aggregate's 19671f40)
    components: { dit: "...", vae_dcae: "...", lm_planner: "..." }
```

> ### ✅ PINS RESOLVED (preparer-acestep #32, 2026-06-24 — both MIT)
> - **GitHub source repo** (the `train.py`/`cli.py` code, §9 build dependency):
>   `ace-step/ACE-Step-1.5` @ tag **v0.1.8**, SHA **`dce6214…`** — pin devops provisioning + the
>   §1.3/§7 `--help` byte-confirm to THIS SHA.
> - **HF model weights**: repo id **`ACE-Step/Ace-Step1.5`**, revision **`19671f40…`** — use as
>   `hf_id` / `revision`.
>
> ### ✅ 2B-vs-XL DISAMBIGUATION — RESOLVED (#32 relay, my earlier flag answered)
> It is **NOT one repo with both variants in-folder** (my prior assumption, now corrected):
> - **DEFAULT path = the aggregate `ACE-Step/Ace-Step1.5` @ `19671f40…`** — carries the 2B turbo
>   DiT (+ 1.7B LM + VAE + text-encoder) IN-FOLDER. This is the smoke + default path; read the
>   `components.<subfolder>` names from the provisioned tree.
> - **XL (4B) = SEPARATE opt-in repos** `ACE-Step/acestep-v15-xl-{base,sft,turbo}` — NOT in the
>   aggregate. The `ace-step-v15-xl` registry entry points at its own `hf_id` + its own pinned
>   revision (pull at provisioning; do NOT reuse the aggregate's `19671f40`).
> - **No 2b/4b conditional on the default path** (lead confirmed to backend-coder too): the default
>   registry entry is unconditionally the aggregate repo. XL is a clearly-separate opt-in entry.
>
> Cross-ref: model-research §3 (DiT 2B ≈ 4.7 GB, XL 4B ≈ 9 GB).

---

## 4. Dataset-prep `.pt`-cache contract  ✅ FILLED (preparer-acestep, cross-checked)

**Flow (Adapter, two-pass)**: audio (`--audio-dir`) or labeled JSON (`--dataset-json`) →
`train.py --preprocess --tensor-output <cache>` (root-parser FLAG, §1.3 ⚠️) → flat dir of
per-sample `<name>.pt` → consumed by `train.py fixed --dataset-dir <cache>`. The wrapper
orchestrates; ACE-STEP's
preprocess does the VAE→DiT encoding.

### 4.1 INPUT format ✅ (confirmed against `preprocess_discovery.py` + `preprocess.py`)

- Two input modes (one required): `--audio-dir <raw>` (raw audio source) OR
  `--dataset-json <labeled.json>` (a labeled dataset JSON).
- **`dataset.json` is a SINGLE index file, NOT per-file sidecars** (structural correction).
  One JSON document indexes ALL samples. The discovery loader (`preprocess_discovery.py`) does
  NOT read any `.txt`/`.json` sidecars — per-file `.lyrics.txt`/`.caption.txt` are the **Gradio UI
  dataset-builder flow ONLY**. **Per-sample keys actually PARSED by the headless loader**:
  `audio_path`, `filename`, `caption`, `lyrics`, `genre`, `bpm`, `keyscale`, `timesignature`,
  `duration`, `is_instrumental`.
  > NOTE: `language` / `custom_tag` / `prompt_override` appear in the `.pt` *metadata* schema (§4.2)
  > but are **NOT parsed by the discovery loader** — treat them as optional/unused on INPUT.
  > Do NOT carry the v1 `mp3 + _prompt.txt + _lyrics.txt` per-file-sidecar layout into v1.5.
- If `--audio-dir` is given with NO JSON: the filename stem becomes the caption, lyrics default to
  `[Instrumental]`. (So a bare audio dir is a valid minimal input.)
- **Accepted audio extensions (verbatim)**: `{.wav, .mp3, .flac, .ogg, .opus, .m4a}` — note `.m4a` IS included.
- **48 kHz STEREO resample is AUTOMATIC inside preprocess** (`load_audio_stereo(af, _TARGET_SR,
  max_duration)`) — input may be ANY rate/channels; need NOT be pre-resampled. (So our
  `preprocess.target_sr`/`channels` config keys are assertion/doc values, not flags.)
- Two-pass: pass-1 (~3 GB) VAE→`target_latents` + text-encoder→caption/lyric hidden states (writes
  `<name>.tmp.pt`); pass-2 (~6 GB) DiT-encoder→`encoder_hidden_states`/mask + context builder
  (silence-based)→`context_latents` (writes final `<name>.pt`, deletes `.tmp.pt`).
- **Min clip duration**: there is NO minimum-duration or file-size validation in discovery/preprocess
  (only the 240 s UPPER bound). But DCAE ≈10.77 Hz → ~128 latent frames per ~11.88 s, so a
  non-degenerate clip needs **≥~12 s** (one full 128-frame window). Shorter clips aren't code-rejected
  but yield a very short latent — test-engineer: use **≥12 s** fixtures under `scratch/fixtures/ace_step/`.

### 4.2 `.pt` cache schema ✅ (per-sample, self-contained — **NO central manifest.json**)

Each sample is a standalone `<name>.pt` (intermediate `<name>.tmp.pt` from pass-1 is deleted
after pass-2). Output is a **flat dir** of `.pt` files (NOT sharded, NO index file).

**Top-level keys (CONFIRMED — match the plan's asserted set exactly):**

| Key | Notes |
|-----|-------|
| `target_latents` | DiT-target latent (the diffusion target) |
| `attention_mask` | |
| `encoder_hidden_states` | text/caption encoding |
| `encoder_attention_mask` | |
| `context_latents` | DiT-encoder context (pass-2) |
| `metadata` | per-sample meta dict (keys below) |

**`metadata` sub-dict keys:** `audio_path`, `filename`, `caption`, `lyrics`, `duration`,
`bpm`, `keyscale`, `timesignature`, `genre`, `is_instrumental`, `custom_tag`, `prompt_override`.

> These `metadata` keys ARE the effective `dataset.json` field surface (§4.1) — a labeled
> `--dataset-json` row carries `caption`/`lyrics`/`bpm`/`keyscale`/`timesignature`/`genre`/
> `is_instrumental`/`custom_tag`/`prompt_override` (audio_path/filename/duration are derived).
>
> **test-engineer golden-fixture**: assert these 6 top-level keys are present + tensor-shaped,
> and the `metadata` dict carries at least `audio_path`/`filename`/`caption`/`duration`. This is
> the highest-risk contract (plan risk row) — but it is now CONFIRMED from `args.py` + the
> preprocess writer, not guessed. Still: gate on a real 2–3-sample preprocess smoke before
> trusting tensor shapes/dtypes (only the KEY NAMES are byte-confirmed here, not shapes).

---

## 5. Generic `data_dir` / `cache_dir` contract (local bind-mount + cloud HF-dataset)

**SACROSANCT**: generic + reusable. NO hardcoded personal path anywhere. A researcher points
`dataset.data_dir` at THEIR own out-of-repo audio corpus; an empty value falls back to a
documented, **gitignored** default landing dir.

### 5.1 Config keys (locked)

| Key | Lane | Meaning |
|-----|------|---------|
| `dataset.data_dir` | local | Host path to the out-of-repo audio corpus root. CLI `--data-dir` overrides. |
| `dataset.cache_dir` | both | Where the `.pt` cache lives (survives container restarts when bind-mounted). |
| `dataset.dataset_name` | cloud | HF dataset id pulled at runtime (the cloud equivalent of `data_dir`). |

**Documented default landing dir** (when `data_dir` empty): a repo-relative gitignored dir,
e.g. `Datasets/ace_step_corpus/` (add to `.gitignore`) — the skill (§9) documents pointing
`data_dir` anywhere. NOT user-specific.

### 5.2 Local bind-mount seam (the ONE genuine handler edit)

`tuner/handlers/local_run_handler.py` — verified seams:
- Docker run assembles mounts at **`local_run_handler.py:390-392`**: `-v {repo_root}:/workspace/repo`,
  then `_cache_mount_args(plan, home_dir)` appends HF/pip cache `-v` mounts (**`:393`**).
- The plan dict is built at **`:676-704`** (carries `image`, `transfer`, `mount_hf_cache`, etc.).
- Today only `dataset.local_file` (a single in-repo file) is staged — copy-mode at **`:642-644`**,
  flag-rewrite at **`:513-518`**.

**Contract for the edit** (mirror the `_cache_mount_args` precedent, do NOT special-case):
1. Add `data_dir` + `cache_dir` (resolved host paths) to the plan dict (`:676-704`).
2. Add a `_data_dir_mount_args(plan)` helper that, when `data_dir` is set, appends
   `-v {host_data_dir}:/workspace/data` (read-only ok) and `-v {host_cache_dir}:/workspace/cache`;
   call it right after `_cache_mount_args` at **`:393`**.
3. Path-rewrite: inside the container the wrapper sees `data_dir=/workspace/data`,
   `cache_dir=/workspace/cache` (mirror the `local_file` rewrite at `:513-518`).
4. `transfer: bind` is required for an out-of-repo corpus (copy-mode can't pull outside the repo);
   document that constraint (bind is the default on non-Windows, `:630`).

### 5.3 Cloud HF-dataset pull seam

Per plan Key-Decision (devops + preparer-integration verified): cloud input today is only
(a) small in-repo files via git-clone or (b) an HF dataset pulled at runtime
(`HF_HUB_ENABLE_HF_TRANSFER=1`). So for cloud, the corpus is pre-staged as an **HF dataset**
and referenced by `dataset.dataset_name`.

> **⚠️ CORRECTION**: ACE-STEP's `train.py` has **NO `--dataset-name` flag** (confirmed in §1.3 from
> `args.py` — preprocess takes `--audio-dir`/`--dataset-json` + `--tensor-output`; train takes
> `--dataset-dir`). So `dataset_name` is **OUR cloud seam, not an ACE-STEP flag**: the wrapper /
> cloud job pulls the HF dataset to a LOCAL path, then passes that path to `--audio-dir` (preprocess)
> or `--dataset-dir` (train). devops owns the pull-to-local-path step; the ACE-STEP flags it feeds
> are the §1.3 ones, never a `--dataset-name`.

Bucket-as-input is possible (`shared/hf_bucket_sync_helper.py::sync_bucket` is bidirectional) but
net-new/unprecedented — NOT chosen.

---

## 6. Audio-verifier interface (test-engineer builds against this — LOCKED)

**Pattern**: mirror `retrieval_verifier.py` EXACTLY — a corpus-level verifier that registers
for discoverability but is invoked via a dedicated entry point (NOT the per-completion
`verify()` loop). Verified template: `shared/verifiers/registry.py:23` (`@register`),
`shared/verifiers/builtins/retrieval_verifier.py`, `Evaluator/runner.py:322` (sibling branch).

### 6.1 Registration + invocation split

> **⚠️ REGISTRATION FORM (corrected to match the real SSOT):** `@register(type_name)` decorates a
> **FACTORY FUNCTION** `_build_*(spec: Mapping) -> Verifier`, NOT the class. This is the working
> precedent at **`retrieval_verifier.py:144`** (`@register("retrieval")` on
> `_build_retrieval_verifier(spec) -> RetrievalVerifier()`), and `registry.py:23` is the dispatch
> SSOT (`VERIFIER_FACTORIES[type_name] = factory`). The `AudioVerifier` CLASS is a SEPARATE,
> undecorated definition. (An earlier sketch put `@register` on the class — that is wrong; mirror
> the factory-function form below.)

```python
# shared/verifiers/builtins/audio_verifier.py
from typing import Mapping
from shared.verifiers.registry import register
from shared.verifiers.contract import VerifierInput, VerifierOutput


@register("audio")                       # factory-function form (mirrors retrieval_verifier.py:144)
def _build_audio_verifier(spec: Mapping) -> "AudioVerifier":
    """Factory for the ``audio`` verifier type. Stateless + configured per-call via
    AudioConfig (corpus-level), so no spec fields are consumed here — accepted for
    registry uniformity."""
    return AudioVerifier()


class AudioVerifier:                      # SEPARATE class — NOT decorated
    name = "audio"

    def verify(self, sample: VerifierInput) -> VerifierOutput:
        raise NotImplementedError(
            "audio is corpus-level; use evaluate_audio(AudioConfig), "
            "not the per-completion verify(VerifierInput) entry point."
        )   # mirrors RetrievalVerifier.verify — protocol-only, intentionally unused

    def evaluate_audio(self, cfg: "AudioConfig") -> "AudioValidationResult":
        # DECODE: WAV via stdlib `wave` + numpy (NO soundfile/torchaudio) — §7 locks
        # audio_format=wav, soundfile is not in base/CI env, and stdlib keeps this
        # verifier shared/-pure so the whole P0 suite runs in CI with ZERO heavy deps.
        # soundfile is the lazy-imported FALLBACK for non-WAV inputs only (inside this
        # method, so @register import stays cheap). CLAP/FAD heavy deps (P2) also lazy.
        ...
```

> ### ✅ DECODE DEPENDENCY — RATIFIED (architect, 2026-06-24): stdlib `wave`+numpy primary, soundfile fallback
> test-engineer's choice to decode WAV via the stdlib **`wave`** module + numpy (NOT
> soundfile/torchaudio) is BLESSED as the contract shape. Rationale holds: §7 locks
> `audio_format="wav"` (the rendered smoke audio is always WAV), soundfile is not in the base/CI
> env, and stdlib decoding keeps `audio_verifier.py` **`shared/`-pure** so the entire P0 verifier
> suite runs in CI with zero heavy deps. **soundfile is retained as the lazy-imported FALLBACK for
> non-WAV inputs only** (kept inside `evaluate_audio` so the `@register` module import stays
> cheap). This is STRICTLY BETTER than the original §6.1 sketch (which lazy-imported
> soundfile/torchaudio as primary) — it removes the heavy dep from the smoke path entirely. P2
> CLAP/FAD heavy deps stay lazy on the same seam. (Verified: 32 tests green, non-vacuity
> counter-tested.)

### 6.2 `AudioConfig` dataclass (frozen — mirrors `RetrievalConfig` shape at `retrieval_verifier.py:73-100`)

```python
@dataclass(frozen=True)
class AudioThresholds:                    # mirrors RetrievalThresholds (:57)
    min_duration_s: float = 0.0
    max_duration_s: float = 0.0           # 0 = no cap
    require_sr: int = 48000
    require_channels: int = 2             # stereo
    min_rms: float = 0.0                  # non-silence floor

@dataclass(frozen=True)
class AudioConfig:                        # mirrors RetrievalConfig (:74)
    audio_paths: Sequence[str]            # rendered audio to score (required)
    thresholds: AudioThresholds = field(default_factory=AudioThresholds)
    # Phase-1 = structural smoke. Phase-2 drop-ins on the SAME seam:
    reference_set: str | None = None      # for FAD (P2)
    captions: Sequence[str] | None = None # for CLAP text↔audio (P2)
    metrics: Sequence[str] = ()           # e.g. ("fad","clap") — P2, empty in P1
```

### 6.3 `evaluate_audio` return + Phase-1 assertions (test-engineer owns)

`evaluate_audio(cfg) -> AudioValidationResult` (mirror `RetrievalValidationResult`): a
pass/warn/fail ladder over **structural** checks only in Phase-1:
loadable · non-silent (RMS ≥ `min_rms`) · sample-rate == 48000 · channels == 2 (stereo) ·
duration within `[min_duration_s, max_duration_s]`. **Never bit-exact** (diffusion is
non-deterministic — plan risk row). FAD/CLAP are P2 drop-ins on the same `metrics` field.

### 6.4 Runner attach-point (mirror retrieval at `runner.py:317-323`)

`Evaluator/runner.py` — add a SIBLING branch to the per-completion loop, keyed on
`case.metadata.get("audio_config")` (exactly as `retrieval_config` at **`:322`**):

```python
if case.metadata.get("audio_config"):
    return _evaluate_audio_case(case)     # sibling to _evaluate_retrieval_case (:476)
```

`_evaluate_audio_case` mirrors `_evaluate_retrieval_case` (`:476-514`): build an
`AudioConfig` via a `_build_audio_config(raw)` (mirror `_build_retrieval_config` `:517`,
required-key validation), call `AudioVerifier().evaluate_audio(cfg)` once, attach the result
to `EvaluationRecord` (add an `audio` field alongside `retrieval`), surface any exception as
`error` → fail. Builtin import for side-effect goes in
`shared/verifiers/builtins/__init__.py` (alongside the retrieval import).

---

## 7. Music-generation step contract  ✅ FILLED (`cli.py`, full flag surface — preparer-acestep)

Dedicated generation step (net-new — no SynthChat/text-inference analog). Shells out to
ACE-STEP v1.5 inference (NOT our text LLM client).

> **⚠️ ENTRY-POINT MAP (load-bearing — preparer-acestep read all the files):** pick the right one.
> - **top-level `cli.py main()`** = the REAL headless inference entry — fully non-interactive via
>   `python cli.py -c config.toml` (TOML keys OR flags); internally calls `generate_music()`.
>   **THIS is our music-gen entry (SHELL-OUT — chosen).**
> - **`acestep/acestep_v15_pipeline.py main()`** = the `acestep` console-script = **GRADIO UI ONLY**
>   (argparse has only `--port/--share/--server-name/--language/device/offload`; NO prompt/duration/
>   seed). **Do NOT target this for headless generation.**
> - **`acestep-api` (`acestep.api_server`)** = REST :8001 — optional, NOT needed.
> - **`acestep/inference.py`** = the IMPORTABLE library (EXISTS). In-process entry:
>   `generate_music(dit_handler, llm_handler, params: GenerationParams, config: GenerationConfig,
>   save_dir=None, progress=None) -> GenerationResult` (must first init dit_handler/llm_handler +
>   build the `GenerationParams`/`GenerationConfig` dataclasses). **Reserve ONLY for tight loop
>   control / avoiding process spawn — NOT the default.**
> - **⚠️ `inference.py` at REPO ROOT does NOT exist (404)** — model-research §3's phrasing "Python
>   `inference.py`" is misleading; the importable lib is `acestep/inference.py`, the shell entry is
>   top-level `cli.py`. Do not chase a root-level `inference.py`.
>
> So the clean split is: `train.py` = training, **`cli.py` = headless inference (shell-out, chosen)**,
> `acestep/inference.py::generate_music` = in-process fallback, `acestep_v15_pipeline` = UI (skipped),
> `acestep-api` = REST (skipped). §7 is FULLY filled.

### 7.1 Generation invocation + flag surface (driven by our generate handler)

`python cli.py -c <generated.toml>` (or equivalent flags). Our generate step writes a TOML from
config + per-request params, then shells `cli.py`.

| Our config / request key | `cli.py` flag / TOML key | Notes |
|--------------------------|--------------------------|-------|
| caption/prompt | `--caption` / `caption` | the text prompt |
| lyrics (optional) | `--lyrics` / `lyrics` | text or file; `--use_cot_lyrics` to auto-gen |
| duration | `--duration` / `duration` | 10–600 s |
| **seed (reproducibility)** | `--seed <int>` / `seed` | pin an int; `-1` = random. Batch: `--seeds "1,2,3"`. **Do NOT set `--use_random_seed`** for reproducible gen |
| inference steps | `--inference_steps` | |
| guidance | `--guidance_scale` (+ `--use_adg`, `--cfg_interval_start/_end`) | |
| task type | `--task_type` | choices: text2music \| cover \| repaint \| lego \| extract \| complete (we use `text2music`) |
| DiT model | `--config_path` | e.g. `"acestep-v15-turbo"` (ties to §3 variant) |
| LM planner | `--lm_model_path` | optional |
| output format | `--audio_format {mp3,wav,flac}` | default from model config_defaults |
| output dir | `--save_dir` | default `"output"` |

Result is printed as `Path: <file> | Seed: <seed>` — the generate handler parses that line for the
output audio path. Native generation sample rate is 48 kHz (not a CLI flag; inherited from model
defaults — confirm at runtime). Determinism: pin `--seed <int>`, never `--use_random_seed`;
downstream assertions are structural-only (never bit-exact — diffusion).

> **Generation build step-0** (mirrors §1.3, applies at Phase 4): run `python cli.py --help` against
> the provisioned ACE-STEP repo to byte-confirm these spellings/defaults before locking the
> generation argv (preparer read via raw-GitHub+summarization, not `--help` execution).

Example non-interactive TOML the repo accepts:
`task_type="text2music", caption=..., lyrics=..., duration=30, seed=42, save_dir="output",
audio_format="wav", config_path="acestep-v15-turbo"`.

Output audio path feeds directly into §6 `AudioConfig.audio_paths` for the smoke verifier
(generate → eval E2E, plan test scenario).

---

## 8. Method-registration surface (7 gates) + dedup (LOCKED — user decision: FIX NOW)

> **§8 was an UNDERCOUNT.** Earlier revisions named "3 hardcoded lists." backend-coder's #27
> dedup surfaced that the method-registration surface is **≥7 gates** — `ace_step` was only
> HALF-registered (gates 5/6/7 — cloud-exec validation, the `--method` CLI choices, and the
> ExperimentSpec allowlist — all rejected `ace_step` until they were SSOT-derived). The split
> below is the authoritative count. **Train-time gates (1–7) derive from the SSOT;
> serving-time eval tuples (§8.3) are an INTENTIONAL embedding-/ace_step-excluding subset.**

### 8.1 Train-time registration gates — all derive from the `TRAINING_METHODS` SSOT (verified `file:line` in this worktree)

| # | Site | Current (post-#27) | Status |
|---|------|---------|------|
| 1 | `shared/utilities/paths.py:11` | `TRAINING_METHODS = (… "ace_step")` | **SSOT** — add `"ace_step"` here, everything below derives |
| 2 | `tuner/backends/training/rtx_backend.py` `get_available_methods()` | `return list(TRAINING_METHODS)` | SSOT-derived (literal removed by dedup — see §8.2) |
| 3 | `tuner/backends/training/cloud/hf_jobs_backend.py` `get_available_methods()` | `return list(TRAINING_METHODS)` | SSOT-derived (literal removed by dedup — see §8.2) |
| 4 | `tuner/handlers/cloud_train_handler.py:474` `_load_method_labels` | reads YAML (was hardcoded `{sft,kto}`, STALE) | moved to YAML — see §8.2 |
| 5 | `tuner/backends/training/cloud/base_cloud.py:113` `SUPPORTED_METHODS` | `SUPPORTED_METHODS = TRAINING_METHODS` | **SSOT-derived** — cloud-exec guard (`:127`) accepts `ace_step` |
| 6 | `tuner/cli/parser.py:269` `--method` choices | `choices=list(TRAINING_METHODS)` | **SSOT-derived** — CLI accepts `ace_step` |
| 7 | `shared/experiment_tracking/experiment_spec.py:213` allowlist | `self.method not in set(TRAINING_METHODS)` | **SSOT-derived** — ExperimentSpec.validate() accepts `ace_step` |

**Orchestration decision (team-lead, 2026-06-24, aligned with user FIX-NOW dedup intent):** all
three of the formerly-literal train-time gates (5/6/7) derive from `TRAINING_METHODS`. Adding
`ace_step` to gate #1 (the SSOT) propagates to all seven with no further per-gate edits.

`train_ace_step.py` + `configs/config.yaml` then auto-resolve by the `train_{method}.py` /
`Trainers/{method}/` convention (`rtx_backend.py` `_script_for_config`, cloud `_hf_command_builder`)
with zero dispatch-code changes once `ace_step` is in the SSOT-derived list.

### 8.2 Dedup refactor design (the in-scope cleanup — plan Phase 5)

**Goal**: single source of truth. After this, adding a method touches ONLY `paths.py:11` + a YAML label.

1. **Derive `get_available_methods()` from the SSOT.** Both backends
   (`rtx_backend.py:99-106`, `hf_jobs_backend.py:115-125`) currently `return [...]` a hardcoded
   literal. Replace each body with `return list(TRAINING_METHODS)` (import from
   `shared/utilities/paths.py`). Removes touch-points #2 + #3 permanently.
2. **Move method labels to YAML** (mirror the EXISTING `_load_gpu_tiers` precedent at
   `cloud_train_handler.py:486-500`, which reads `Trainers/cloud/cloud_config.yaml` via
   `load_gpu_tiers(config_path)`). Add a `method_labels:` section to a config YAML
   (e.g. `Trainers/cloud/cloud_config.yaml` alongside `gpu_tiers:`, OR a dedicated
   `Trainers/methods.yaml`), and rewrite `_load_method_labels` (`:474-485`) to read it —
   exactly as `_load_gpu_tiers` does. Seed the YAML with ALL methods (fixes the existing
   sft/kto-only staleness) + the new `ace_step` friendly label.
3. **No backward-compat shim** (repo SACROSANCT): delete the hardcoded literals outright.

> **Dedup is SEPARABLE** from the pipeline (plan commit #11, last): if the user later wants
> minimal blast radius, touch-points #2/#3/#4 can revert to a minimal `+ "ace_step"` add.
> But the user decision (2026-06-24) is FIX NOW — so the SSOT-derive is the spec.

### 8.3 Serving-time eval tuple — `ace_step` is INTENTIONALLY EXCLUDED (do NOT SSOT-derive)

The eval/serving backend gate is a SEPARATE surface from the seven train-time gates above and
**must NOT** be folded into the `TRAINING_METHODS` SSOT-derive. The 3-backend serving subset:

| Site | Subset | `ace_step` in it? |
|------|--------|-------------------|
| `tuner/handlers/eval_handler.py:129` | `("llamacpp", "mlc", "unsloth")` | **NO — intentionally excluded** |

This is the same exclusion `embedding` already carries (durable per
`tests/trainers/embedding/test_embedding_method_registration.py` — the R3 exclusion test).
`ace_step` is a **generation method that serves through ACE-STEP's own `cli.py`** (§7), exactly
as `embedding` produces a retrieval embedder served through SentenceTransformers — **neither is
served/quantized through the llamacpp/mlc/unsloth causal-LM serving path.** Leave the literal
`("llamacpp","mlc","unsloth")` tuple as-is and add a documenting comment that the exclusion of
`embedding`/`ace_step` is deliberate (these methods do not produce a causal-LM checkpoint these
backends can load). **DESIGN RATIFICATION (architect, 2026-06-24): CONFIRMED — `ace_step`
belongs in the serving EXCLUSION, not in the serving tuples.**

---

## 9. Cross-cutting locks

- **⚠️ BUILD DEPENDENCY — ACE-STEP v1.5 repo must be PROVISIONED (devops owns).** The wrapper
  (§1) shells `python train.py --preprocess` + `python train.py fixed` (`preprocess` is a
  root-parser FLAG, not a subcommand — §1.3 ⚠️) and the generation step (§7) shells
  `python cli.py` — both require the **`ace-step/ACE-Step-1.5` repo present/installed** at the
  **PINNED ref `v0.1.8` / SHA `dce6214…`** (resolved #32; same ref the §3 model-registry points
  to). devops provisions it:
  in the **Docker image** for local/cloud runs (cloned/installed into the image, NOT vendored
  into THIS repo) and available on the build host for the §1.3/§7 `--help` byte-confirm (step-0).
  The repo is NOT checked into Synthetic Conversations — it is an external dependency (plan
  "External Dependencies: Yes — pin to a known-good commit/tag"). Backend-coder's step-0 `--help`
  runs assume this provisioning is done first.
- **Output dirs**: `ace_step_output/` (auto-derived `CANONICAL_OUTPUT_DIRS`, `paths.py:16`).
  Artifacts under `toolset-training-artifacts/`. **NO `/tmp`** (repo rule). Test fixtures
  under `scratch/fixtures/ace_step/` (plan).
- **Experiment tracking**: reused as-is — `run_type="ace_step"` (free string, no schema change).
- **Upload**: PEFT/LoKr adapter dir → existing `lora` copytree strategy
  (`shared/upload/strategies/lora.py`) AS-IS. Merged-DiT publish deferred (Phase-2); GGUF skipped.
- **Docker image**: SEPARATE purpose-built image (cu128 + apt ffmpeg + Lightning Fabric +
  peft/LyCORIS, transformers<4.58) — do NOT overlay the Unsloth image (verified collision).
  Pins **from first smoke**, not invented (devops; embedding `TBD-PENDING-SMOKE-TEST` precedent).
- **Skill**: `.skills/ace-step-training/` authored, mirrors synced (NEVER hand-edit
  `.agents/skills` / `.claude/skills`). Documents the generic `data_dir` usage (local) +
  `dataset_name` HF-dataset equivalent (cloud).
- **CLAUDE.md note (worktree-gitignored)**: the dedup makes `paths.py:11` `TRAINING_METHODS`
  the SOLE method-registration SSOT — worth a CLAUDE.md pin in the main tree after merge
  (flagged in HANDOFF; not editable here — CLAUDE.md is absent/gitignored in worktrees).

---

## 10. Open items rolled to CODE (gated on this contract)

| Item | Status | Owner | Gates |
|------|--------|-------|-------|
| §1.3 flag tables (Q1+Q2) | ✅ FILLED | preparer-acestep → architect | backend-coder subprocess argv — **gated on §1.3 BUILD STEP-0 `--help` byte-confirm** |
| §4 `.pt` cache schema + input format + min-duration (Q3+Q4) | ✅ FILLED (dataset.json = single index, NOT per-file sidecars) | preparer-acestep → architect | preprocess shim + test golden-fixture sizing |
| §7 generation entry + flag surface (`cli.py`) | ✅ FILLED (Q5 answered — full cli.py flag table; `acestep_v15_pipeline.main()` is UI-only, do NOT target) | preparer-acestep → architect | generation step (Phase 4) — `cli.py --help` byte-confirm at build |
| §3 HF repo ids + pinned revision | ✅ RESOLVED (#32): GH `ace-step/ACE-Step-1.5` @ v0.1.8 SHA `dce6214…`; HF `ACE-Step/Ace-Step1.5` rev `19671f40…`; both MIT | preparer-acestep → architect | model registry + §9 ACE-STEP repo provisioning (pin to SHA `dce6214…`) — one HF repo, 2b/4b-xl via `variant`+`components` (§3 ⚠️) |
| ACE-STEP v1.5 repo provisioning (in image + on build host) | ⚠️ devops action | devops (§9) | step-0 `--help` runs + all subprocess shells |
| cu128 wheel availability (torch 2.7–2.10) | ⚠️ empirical | devops (image smoke) | image build |
| Tensor shapes/dtypes in `.pt` (key NAMES confirmed; shapes not) | ⚠️ smoke-gate | backend/test (2–3 sample preprocess) | golden-fixture shape asserts |

> **Contract status: FULLY LOCKED for the CODE fan-out.** §1–§9 all buildable. §7's generation
> entry is resolved (`cli.py`); only its exact flag spellings are a Phase-4 build-time `--help`
> confirm (Q5 follow-up out) — not on the Phase 1–3 critical path. **Three MANDATORY build-time
> verifies, all called out inline**: (a) §1.3 BUILD STEP-0 byte-confirm BEFORE locking training
> argv (blocking): `python train.py fixed --help` (subcommand) + `python train.py --help` (root
> parser — `preprocess` is a FLAG, not a subcommand; §1.3 ⚠️); (b) §7 `python cli.py --help`
> byte-confirm before locking generation argv (Phase 4); (c) §4.2 2–3-sample preprocess smoke to
> confirm `.pt` tensor shapes/dtypes. All assume the §9 ACE-STEP-repo provisioning is done first.
