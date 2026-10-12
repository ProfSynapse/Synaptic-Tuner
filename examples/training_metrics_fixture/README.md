# Synthetic training metrics fixture

This wholly authored synthetic corpus is for bounded transport qualification,
not model quality measurement. No private source or previous artifact substitute
is selected. Each of eight short question documents pairs with exactly one
answer document; preparation admits all sixteen complete documents, selects only
the eight answers as targets, and uses only the paired question as context.
No target text or derived target guidance enters its prompt. Documents are kept
in declared order without chunking or cleanup transformations.

All question contexts have sequence zero, their own revision family and context
group, and no ancestors. The eight targets have sequences 1–8 in one
`synthetic-metrics` group, their own revision families, and ancestry limited to
the paired question. The configured sequence-tail split must retain target01–06
as train and target07–08 as validation. Those held-out targets or derivatives
never appear as training contexts. Expected coverage: 16 source documents,
8 complete message rows, 6 train and 2 validation. This is explicit synthetic
lineage and ordered allocation, not inferred historical provenance.

The user message consists of `Answer the following harmless practice question.`
then a blank line and the complete paired question. The assistant contains the
complete answer including its newline. Read the sixteen small source files and
all eight prepared rows before any paid approval. The training recipe uses the
same configured Qwen template arguments for prompt/completion as the existing
reviewed recipe; targets contain no thinking scaffold.

## Rebuild locally

From the engine repository root, using a Python environment with the CLI's
lightweight dependencies:

```bash
python tuner.py ingest --config examples/training_metrics_fixture/ingestion.json --select fixture=examples/training_metrics_fixture/sources --json
python tuner.py prepare-dataset --config examples/training_metrics_fixture/preparation.json --json
python tuner.py train --job-config Trainers/recipes/qwen35_4b_modal_metrics_qualification.yaml --plan --json
```

These commands admit only the sixteen explicitly selected synthetic Markdown
files. The preparer binds the exact verified two-member normalized bundle and
publishes the immutable messages dataset under ignored `.tracking/`. A fresh
checkout must run ingestion before preparation, and preparation before planning.
Do not manually rewrite generated bundle or dataset members. Repeated local CLI
builds produced the same bundle digest and byte-identical preparation results.
The expected identity and CPU-only tokenizer review are recorded in
[local-evidence.json](local-evidence.json). That bounded summary contains no
live job evidence or approval.

Use the existing CPU-only doctor before considering a paid attempt:

```bash
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 python tuner.py doctor sft-mask --sft-config examples/training_metrics_fixture/mask-doctor.yaml --dataset-path .tracking/datasets/dataset-58f367d391782375b2fb04c31c5b7c50fec804b35e4a7a54c668433d9018971d/dataset.jsonl --tokenizer-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --sample-size 0 --preview-count 0 --json
```

Unset Hugging Face credential variables for this public tokenizer check. The
local check used an isolated environment with Transformers 5.17.0, Tokenizers
0.23.2, Datasets 4.3.0 and Jinja2 3.1.6, without Torch. After loading the exact
public tokenizer once, the successful final doctor ran with `HF_HUB_OFFLINE=1`.
All eight rows passed: zero hard failures, zero truncation, 83 supervised tokens,
maximum 44 total tokens per row. This checks prompt/completion masking and the
chat-template end-of-turn token; it is neither GPU execution nor model quality.
The doctor config selects preassigned splits and the same loss/render/template
controls as the training recipe. It is not a training launch recipe.

## Bounded training candidate

The recipe preserves `qwen35-sft-v1`, the existing Qwen3.5-4B/model-tokenizer
revision, immutable base image/inventory and LoRA shape. See the retained
[two-step L40S smoke](../../docs/review/modal-volume-root-diagnostic-20260927.md#verified-end-to-end-l40s-smoke-2026-09-29)
for actual earlier execution evidence, and the existing
[reviewed Qwen flight plan](../../docs/review/qwen35-full-training-flight-plan.md#proposed-training-shape)
for the profile/revision/template selection. That earlier source/configuration
qualified its two-step run; it does not qualify this source, dataset, 32-step
run or progress delivery.

This candidate requests L40S × 1, 32 optimizer steps, numeric capture every step,
32 metric records, 900 seconds, and a USD 5 operator maximum (500 minor units).
It retains the five-artifact policy and omits post-training evaluation. A current
scoped quote and its excluded billing dimensions must be reviewed separately;
500 minor units is not a provider billing cap. No training, CPU cloud
qualification or rate query was performed to create this fixture.

Use the [restricted metrics qualification workflow](../../docs/plans/training-progress-streaming.md)
only after its final clean source, candidate admission and exact paid scope are
reviewed and approved. This directory deliberately contains no admission with a
fake source SHA or operational account/workspace identity. Generate an admission
from the final integrated commit; never copy an example digest as authority.
