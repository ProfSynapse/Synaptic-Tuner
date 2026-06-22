---
name: dataset-publishing
description: Publish local dataset artifacts to a Hugging Face dataset repo. Use when uploading a JSONL dataset, pushing a filtered dataset variant, syncing a matching .metadata.json sidecar, or renaming a dataset file in the target repo. This skill is about USING the checked-in dataset publish script via CLI — never ad hoc Python.
allowed-tools: Read, Bash, Write, Grep, Glob
---

# Dataset Publishing

Publish a local dataset JSONL to a Hugging Face dataset repo with the skill-owned script:
`python3 .skills/dataset-publishing/scripts/publish_dataset_to_hf.py`

The script accepts:
- `dataset_path`
- `repo_id`

It also auto-uploads a matching metadata sidecar if present:
- `dataset.jsonl`
- `dataset.metadata.json`

## Quick Reference

| Task | Command |
|------|---------|
| Dry-run a dataset upload | `python3 .skills/dataset-publishing/scripts/publish_dataset_to_hf.py DATASET.jsonl namespace/repo --dry-run` |
| Upload dataset + sidecar | `python3 .skills/dataset-publishing/scripts/publish_dataset_to_hf.py DATASET.jsonl namespace/repo` |
| Upload under a new repo filename | `python3 .skills/dataset-publishing/scripts/publish_dataset_to_hf.py DATASET.jsonl namespace/repo --path-in-repo new_name.jsonl` |
| Upload with explicit metadata file | `python3 .skills/dataset-publishing/scripts/publish_dataset_to_hf.py DATASET.jsonl namespace/repo --metadata-path DATASET.metadata.json` |
| Skip metadata sidecar | `python3 .skills/dataset-publishing/scripts/publish_dataset_to_hf.py DATASET.jsonl namespace/repo --no-metadata` |

## Defaults

- Reads `HF_TOKEN` from the environment or repo `.env`
- Creates the target dataset repo if needed
- Uploads the dataset file to `path_in_repo = basename(dataset_path)`
- Auto-detects `*.metadata.json` sidecars for dotted filenames correctly

## Recommended Workflow

1. Build or filter the dataset locally.
2. Run `--dry-run` first.
3. Run the real upload command.
4. Point the next experiment spec at the uploaded HF dataset file.

## Common Patterns

**Upload a filtered SFT dataset:**
```bash
python3 .skills/dataset-publishing/scripts/publish_dataset_to_hf.py \
  Datasets/synthchat/my_filtered_dataset.jsonl \
  professorsynapse/claudesidian-synthetic-dataset \
  --dry-run

python3 .skills/dataset-publishing/scripts/publish_dataset_to_hf.py \
  Datasets/synthchat/my_filtered_dataset.jsonl \
  professorsynapse/claudesidian-synthetic-dataset
```

**Rename on upload:**
```bash
python3 .skills/dataset-publishing/scripts/publish_dataset_to_hf.py \
  Datasets/synthchat/my_filtered_dataset.jsonl \
  professorsynapse/claudesidian-synthetic-dataset \
  --path-in-repo nonthinking_tools_sft_filtered_03.22.26.jsonl
```

**Upload without a sidecar:**
```bash
python3 .skills/dataset-publishing/scripts/publish_dataset_to_hf.py \
  Datasets/synthchat/my_filtered_dataset.jsonl \
  professorsynapse/claudesidian-synthetic-dataset \
  --no-metadata
```

## CLI Discipline

- Use the checked-in script, not inline Python.
- Run `--dry-run` before the real upload when testing a new dataset variant.
- Keep dataset filenames descriptive and date-stamped.
- If you create a curated filtered variant, keep the rationale in the `.metadata.json` sidecar.
