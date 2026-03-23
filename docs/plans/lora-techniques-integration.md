# Plan: Wire LoRA Techniques Into All Trainers

## Context

Config templates and reference guide already committed (`.skills/fine-tuning/configs/` and `reference/lora-techniques.md`). This plan covers the code changes needed to make them actually work end-to-end.

## Unsloth Compatibility Findings

Researched the Unsloth repo (`unslothai/unsloth`) to determine what `FastLanguageModel.get_peft_model()` actually accepts:

| Parameter | Supported? | Notes |
|-----------|-----------|-------|
| `use_dora` | Yes | Works via kwarg passthrough |
| `use_rslora` | Yes (fixed May 2025) | Was broken, fixed in PR #2539 — ensure Unsloth is up to date |
| `init_lora_weights` | Partial | Allowed values: `True`, `False`, `"gaussian"`, `"loftq"`, `"corda"` only |
| `loftq_config` | Yes | Dedicated parameter |
| `target_modules="all-linear"` | Unknown | Not validated in Unsloth source; needs runtime test |
| PiSSA (`"pissa"`) | **No** | Not in Unsloth's allowed list — would need PEFT bypass |
| EVA (`"eva"`) | **No** | Not in Unsloth's allowed list — would need PEFT bypass |
| OLoRA (`"olora"`) | **No** | Not in Unsloth's allowed list — would need PEFT bypass |

**Sources:**
- [Bug: use_rslora not working](https://github.com/unslothai/unsloth/issues/2531) — Fixed May 2025
- [Bug: init_lora_weights="corda" not supported](https://github.com/unslothai/unsloth/issues/3693) — Fixed, added "corda" to allowed list
- [Unsloth _utils.py validation](https://github.com/unslothai/unsloth/blob/main/unsloth/models/_utils.py) — `init_lora_weights` must be `[True, False, "gaussian", "loftq", "corda"]`

### Impact on Config Templates

| Template | Status |
|----------|--------|
| `regret_free.yaml` | Needs runtime test for `target_modules="all-linear"` |
| `dora.yaml` | Ready — `use_dora` works |
| `qlora_dora.yaml` | Ready — `use_dora` works |
| `grpo_minimal.yaml` | Ready — `use_rslora` works (post May 2025 fix) |
| `loftq.yaml` | Ready — `init_lora_weights="loftq"` in allowed list |
| `pissa.yaml` | Blocked — not in Unsloth allowed list |
| `eva.yaml` | Blocked — not in Unsloth allowed list |
| `olora.yaml` | Blocked — not in Unsloth allowed list |

---

## Implementation Steps

### Step 1: SFT Config Loader — Add Missing Fields

**File:** `Trainers/sft/configs/config_loader.py`

Add to `LoRAConfig` dataclass:
```python
use_rslora: bool = False
use_dora: bool = False
init_lora_weights: Optional[str] = None  # "loftq", "gaussian", "corda", or True/False
```

Update `target_modules` type from `List[str]` to `Union[List[str], str]` to support `"all-linear"` string.

### Step 2: SFT config.yaml — Add Default Fields

**File:** `Trainers/sft/configs/config.yaml`

Add under `lora:`:
```yaml
use_rslora: false
use_dora: false
```

### Step 3: SFT Model Loader — Accept and Pass Through

**File:** `Trainers/sft/src/model_loader.py`

Update `apply_lora_adapters()` signature to add:
- `use_rslora: bool = False`
- `use_dora: bool = False`
- `init_lora_weights = True` (default keeps current behavior)

Pass all three to `FastLanguageModel.get_peft_model()`.

### Step 4: SFT Trainer — Wire Config to Model Loader

**File:** `Trainers/sft/train_sft.py`

Update the `apply_lora_adapters()` call (~line 771) to pass:
```python
use_rslora=config.lora.use_rslora,
use_dora=config.lora.use_dora,
```

### Step 5: KTO Model Loader — Fix Hardcoded False

**File:** `Trainers/kto/src/model_loader.py`

The `apply_lora_adapters()` function:
1. Add `use_rslora`, `use_dora`, `init_lora_weights` parameters to function signature
2. Remove hardcoded `use_rslora=False` at line 205
3. Pass all three through to `FastLanguageModel.get_peft_model()`

### Step 6: KTO Trainer — Wire Config to Model Loader

**File:** `Trainers/kto/train_kto.py`

Update the `apply_lora_adapters()` call (~line 882) to pass:
```python
use_rslora=config.lora.use_rslora,
use_dora=config.lora.use_dora,
```

### Step 7: Tier Config Maps — Add LoRA Technique Keys

**Files:**
- `Trainers/sft/train_sft.py` (~line 589)
- `Trainers/kto/train_kto.py` (~line 734)

Add to `_tier_config_map`:
```python
"use_dora": ("lora", "use_dora"),
"use_rslora": ("lora", "use_rslora"),
"target_modules": ("lora", "target_modules"),
"init_lora_weights": ("lora", "init_lora_weights"),
```

### Step 8: CLI Flags (Optional but Recommended)

**Files:**
- `Trainers/sft/train_sft.py`
- `Trainers/kto/train_kto.py`

Add CLI arguments:
```python
parser.add_argument("--use-dora", action="store_true", help="Enable DoRA (Weight-Decomposed LoRA)")
parser.add_argument("--use-rslora", action="store_true", help="Enable rsLoRA (rank-stabilized scaling)")
```

Wire in the CLI override section (after tier application, before training).

### Step 9: Update Lora Techniques Reference

**File:** `.skills/fine-tuning/reference/lora-techniques.md`

Update the integration status table and Tier 2 section with Unsloth findings:
- PiSSA/EVA/OLoRA are blocked by Unsloth's validation
- LoftQ is supported via `init_lora_weights="loftq"`
- Mark `use_rslora` bug as fixed (May 2025)
- Add note about `target_modules="all-linear"` needing runtime test

### Step 10: Runtime Smoke Test

After code changes:
1. `--dry-run` with `use_dora: true` on SFT
2. `--dry-run` with `use_rslora: true` on SFT
3. `--dry-run` with `use_dora: true` on KTO
4. Test `target_modules: "all-linear"` string on SFT to see if Unsloth handles it
5. Test `init_lora_weights: "loftq"` on SFT

---

## What NOT to Do

- Don't add PiSSA/EVA/OLoRA support — Unsloth blocks them. Document as "needs PEFT bypass" in the reference.
- Don't add `init_lora_weights` to GRPO — it already uses raw dict access and would work if added to YAML config, but Unsloth restricts the values.
- Don't wire `init_lora_weights` into CLI flags yet — limited to "loftq"/"gaussian"/"corda" in Unsloth, not worth CLI surface area until PiSSA/EVA are unblocked.

## File Change Summary

| File | Change Type |
|------|------------|
| `Trainers/sft/configs/config_loader.py` | Add fields to LoRAConfig |
| `Trainers/sft/configs/config.yaml` | Add use_rslora, use_dora defaults |
| `Trainers/sft/src/model_loader.py` | Add params, pass to get_peft_model |
| `Trainers/sft/train_sft.py` | Wire config, update tier map, add CLI flags |
| `Trainers/kto/src/model_loader.py` | Add params, fix hardcoded False |
| `Trainers/kto/train_kto.py` | Wire config, update tier map, add CLI flags |
| `.skills/fine-tuning/reference/lora-techniques.md` | Update integration status |
| `.skills/fine-tuning/configs/pissa.yaml` | Add "blocked by Unsloth" note |
| `.skills/fine-tuning/configs/eva.yaml` | Add "blocked by Unsloth" note |
| `.skills/fine-tuning/configs/olora.yaml` | Add "blocked by Unsloth" note |
