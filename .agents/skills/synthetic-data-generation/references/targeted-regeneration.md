# Targeted Dataset Regeneration

Use the checked-in SynthChat CLI when selected JSONL rows need another
judge/improve pass. This preserves source line identities and produces a separate
slice for review; the skill does not ship row-extraction or in-place replacement
scripts.

## Select rows directly

```bash
python -m SynthChat.run improve \
  --input path/to/dataset.jsonl \
  --rubrics <rubric-name> \
  --lines 7,12,20-25 \
  --workers 8 \
  --output Datasets/synthchat/regenerated-slice.jsonl
```

## Select rows from a manifest

Put one line number or range per line in a reviewed text file, then run:

```bash
python -m SynthChat.run improve \
  --input path/to/dataset.jsonl \
  --rubrics <rubric-name> \
  --line-file path/to/reviewed-lines.txt \
  --workers 8 \
  --output Datasets/synthchat/regenerated-slice.jsonl
```

The adjacent `.improve_report.json` records original input line numbers. Inspect
the regenerated slice and report before deciding how it joins a downstream
dataset. Validate the slice with:

```bash
python -m SynthChat.run validate \
  --input Datasets/synthchat/regenerated-slice.jsonl \
  --rubrics <rubric-name>
```

To create a new combined dataset without editing the original in place, use the
checked-in `scripts/combine_datasets.sh` helper from this skill. Keep the original
JSONL and regeneration report as provenance. Any bespoke merge or replacement
policy belongs in a reviewed, reusable project workflow rather than an invented
one-off script.

## Review rules

- Work on a bounded selection and keep output separate from the source.
- Confirm the selected rows and rubric before incurring provider cost.
- Inspect both positive and negative examples when labels are present.
- Preserve source line numbers and the improvement report.
- Run the small-sample protocol before regenerating a large selection.
