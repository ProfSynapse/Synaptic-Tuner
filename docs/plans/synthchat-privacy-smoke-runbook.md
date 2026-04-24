# SynthChat Privacy Smoke Runbook

## Purpose

This is the runnable smoke sequence for the privacy preprocess feature once local compute is available.

It covers:

1. standalone sanitize on synthetic fixture docs
2. standalone sanitize on a synthetic JSONL fixture
3. docs-based SynthChat generation with privacy preprocess enabled
4. optional vLLM-backed `llm_polish` smoke

The checked-in fake inputs live under `tests/fixtures/privacy/`.

## Prerequisites

- `opf`, `Faker`, and `pytest` installed in the active Python environment
- an OPF checkpoint available locally
- enough free CPU/GPU/RAM to load the privacy filter model

Recommended environment variables:

```powershell
$env:OPF_CHECKPOINT="F:\Models\privacy-filter"
$env:VLLM_HOST="127.0.0.1"
$env:VLLM_PORT="8000"
```

If `OPF_CHECKPOINT` is unset, OPF will try to download the default checkpoint from Hugging Face.

## 1. Unit Coverage

```powershell
python -B -m pytest tests\test_privacy_preprocess.py tests\synthchat\test_labeling.py -q
```

## 2. Mask-Only Sanitize Smoke

```powershell
python -B -m SynthChat.run sanitize `
  --input tests\fixtures\privacy\raw_seed_docs `
  --output tmp\privacy_mask_only_docs `
  --privacy-profile mask_only
```

```powershell
python -B -m SynthChat.run sanitize `
  --input tests\fixtures\privacy\raw_seed_dataset.jsonl `
  --output tmp\privacy_mask_only_dataset.jsonl `
  --privacy-profile mask_only
```

Expected outcome:

- typed placeholders such as `[PRIVATE_PERSON]` and `[SECRET]`
- `privacy_sanitize_report.json` for docs
- per-record `metadata.privacy_preprocess` for JSONL output

## 3. Realistic Pseudonymization Smoke

```powershell
python -B -m SynthChat.run sanitize `
  --input tests\fixtures\privacy\raw_seed_docs `
  --output tmp\privacy_pseudonyms_docs `
  --privacy-profile realistic_pseudonyms
```

```powershell
python -B -m SynthChat.run sanitize `
  --input tests\fixtures\privacy\raw_seed_dataset.jsonl `
  --output tmp\privacy_pseudonyms_dataset.jsonl `
  --privacy-profile realistic_pseudonyms
```

Expected outcome:

- names, emails, phones, addresses, URLs, and dates replaced with synthetic values
- repeated entities remain consistent within a document
- secrets remain masked when `secret_strategy: mask`

## 4. Docs-Based SynthChat Generation Smoke

This path verifies the integrated preprocess hook, not just standalone sanitize.

```powershell
python -B -m SynthChat.run generate `
  --docs tests\fixtures\privacy\raw_seed_docs `
  --targets-file SynthChat\config\targets_privacy_docs_smoke.json `
  --privacy-profile realistic_pseudonyms `
  --per-doc 1 `
  --max-iterations 1 `
  --output Datasets\synthchat\privacy_docs_smoke.jsonl
```

Inspect the output for:

- `_meta.privacy_preprocess`
- `metadata.source_doc_privacy`
- absence of the original fixture PII strings in prompts and final examples

## 5. Improve / Validate Input Sanitization Smoke

These commands verify that JSONL input can be sanitized before improvement or validation sends content to the model.

```powershell
python -B -m SynthChat.run validate `
  --input tests\fixtures\privacy\raw_seed_dataset.jsonl `
  --privacy-profile realistic_pseudonyms
```

```powershell
python -B -m SynthChat.run improve `
  --input tests\fixtures\privacy\raw_seed_dataset.jsonl `
  --privacy-profile realistic_pseudonyms `
  --max-iterations 1 `
  --output tmp\privacy_improve_smoke.jsonl
```

Inspect:

- `metadata.privacy_preprocess_input`
- `metadata.privacy_preprocess_output` when generated-output sanitization is enabled in settings
- the `.improve_report.json` privacy summary fields

## 6. Optional vLLM Polish Smoke

Use the `realistic_pseudonyms_vllm_polish` profile only after a local OpenAI-compatible vLLM endpoint is running.

```powershell
python -B -m SynthChat.run sanitize `
  --input tests\fixtures\privacy\raw_seed_docs `
  --output tmp\privacy_vllm_polish_docs `
  --privacy-profile realistic_pseudonyms_vllm_polish
```

Success criteria:

- OPF sees raw fixture text locally
- the polish model only sees sanitized text
- synthetic entities remain unchanged
- no masked secrets are expanded into fake secret values
