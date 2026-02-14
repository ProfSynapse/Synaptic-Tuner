# Dataset Organization

This folder is split by data lifecycle so generated files do not pile up in random locations.

## Directory Roles

- `Datasets/behavior_datasets/`: curated behavior datasets used by training/evaluation.
- `Datasets/tools_datasets/`: curated tool-calling datasets.
- `Datasets/gspo_datasets/`: GSPO-specific datasets.
- `Datasets/essay_datasets/`: essay pipeline datasets.
- `Datasets/synthchat/`: direct SynthChat generation outputs.
- `Datasets/archive/legacy_snapshots/`: historical dataset snapshots kept for reproducibility.
- `Datasets/archive/analysis/`: analysis artifacts tied to investigations.

## Working Rules

- Do not place new `.jsonl` files directly under `Datasets/` root.
- Use `Datasets/synthchat/` for new generation runs before curation.
- Move superseded versions to `Datasets/archive/legacy_snapshots/`.
- Keep temporary test fixtures in `scratch/fixtures/`, not in `docs/`.

## Naming

Use descriptive, date-stamped names:

- `<domain>_<purpose>_<MM.DD.YY>.jsonl`
- Example: `tools_sft_train_02.14.26.jsonl`

For metadata files, use the same basename plus `.metadata.json`.
