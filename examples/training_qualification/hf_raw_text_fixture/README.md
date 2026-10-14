# Synthetic raw-text SFT fixture

The eight authored documents each contain one complete Question/Answer pair.
Ingestion selects every `.md` document, preserves its complete body and performs
no chunking. Preparation publishes genuine `syntunia-sft-row/v1` raw-text rows;
it does not relabel the separate messages/v2 metrics fixture.

Run from the engine checkout with the existing CLI:

```sh
python tuner.py ingest --config examples/training_qualification/hf_raw_text_fixture/ingestion.json --select fixture=examples/training_qualification/hf_raw_text_fixture/sources --json
python tuner.py prepare-dataset --config examples/training_qualification/hf_raw_text_fixture/preparation.json --json
```

The selected eight rows cover all eight complete paired questions and literal
answers. Configured hash-rank splitting yields six training and two validation
rows; these independent synthetic pairs make no chronological ancestry claim.
Raw-text training applies full-token loss to both question and answer. There is
no chat-template projection or completion-only masking.

Expected bundle identity is
`cbad3e6be421470a8f3480944f95e52f151ec315d0914472fee27e2449265a1b`,
prepared semantic identity is
`7aa26aba079ac1210466c5c2cdf747f00c764b5bc4ed366e2f67af7ffa0d3770`,
and the 2,821-byte original JSONL has SHA256
`739b3df17bf7fc39d5c9487136a633e8ad2703e700271e224756df8074d2c361`.

The configuration selects a compatible named Qwen SFT profile and three bounded
steps. This fixture and compiler proof do not qualify HF hardware, the image,
model quality, or an actual parameter update.
