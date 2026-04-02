---
name: research-reporting
description: Create structured research notes from experiment runs and analysis artifacts. Use when creating a note at run launch, updating it as training/evaluation/loss stages finish, summarizing a finished run, comparing experiment outcomes, extracting hypotheses from eval/loss artifacts, or proposing next-run actions grounded in `.tracking/experiments/<id>/analysis/` outputs. This skill is about turning repo-native experiment evidence into stable, machine-readable markdown.
allowed-tools: Read, Bash, Write, Grep, Glob
---

# Research Reporting

Generate compact research notes that are easy to read and easy to parse later.

## Use This Skill When

- The user wants a research note, experiment summary, post-run analysis, or structured markdown output.
- The source of truth is an experiment bundle under `.tracking/experiments/<id>/`.
- The output should include stable frontmatter and explicit evidence for claims.
- The note should be created early and updated through the lifecycle of one experiment.

## Default Workflow

1. Resolve the experiment id and open `.tracking/experiments/<id>/experiment.json`.
2. If `spec_path` is present, read the experiment spec so the note captures actual config numbers instead of only outcome artifacts.
3. Read primary analysis artifacts in this order:
   - `analysis/experiment_summary.json`
   - `analysis/next_run_candidates.json`
   - `analysis/hypothesis_context.json`
   - `analysis/run_matrix.csv`
4. Read failure slices only if you need representative examples:
   - `analysis/failure_slices/eval_failures.jsonl`
   - `analysis/failure_slices/high_loss_examples.jsonl`
5. Read stage lineage files when you need provenance, timing, commit, hardware, or cost details.
6. Write the note from `assets/research_note_template.md`.

Load [reference/artifact-map.md](reference/artifact-map.md) when you need to know which artifact supports which section.

## Lifecycle Modes

Use the same note template for all three modes:

1. Launch note:
   - Create the note as soon as the experiment is launched or selected.
   - Fill identity, config, and known runtime fields.
   - Leave future metrics and recommendation fields empty.
2. Stage update:
   - Re-open the same note after training, evaluation, loss, or analysis completes.
   - Update only the fields now supported by artifacts.
   - Preserve prior fields unless newer canonical artifacts supersede them.
3. Final note:
   - After analysis/recommendation, ensure the note contains the final status, observed outcomes, hypotheses, and next-run recommendation.

Default policy: one evolving note per experiment, not one note per stage.

## Reporting Rules

- Keep frontmatter keys stable across notes. Prefer `null`, `[]`, or omitted sections over ad hoc placeholder prose.
- Keep the schema general: use grouped maps for metrics and config instead of adding one top-level key per experiment-specific number.
- Support partial completion. A note does not need all stages populated to be valid.
- Separate three things clearly:
  - observed: directly supported by artifacts
  - inferred: reasoned from observed evidence
  - proposed: next-run actions or hypotheses
- Prefer exact values from JSON for frontmatter. Round only in prose if readability improves.
- Preserve config numbers exactly as found in the spec or lineage artifacts. Do not normalize `1.0e-4` into prose-only text and do not drop unset knobs that matter to interpretation.
- Do not cite `experiment_summary.md` as the primary evidence source when the JSON exists.
- Do not invent comparisons, baselines, or causes. If a baseline run is missing, state that it is missing.
- When the analysis bundle includes a ranked recommendation, carry over:
  - `selected_candidate_rank`
  - `selected_candidate_confidence`
  - `recommended_next_action`
- If loss artifacts are absent or failed, keep loss fields `null` and note the missing stage in the body.
- If a run has custom metrics, place them under `metrics.summary` or `metrics.groups.<stage>` rather than forcing them into a fixed eval/loss schema.
- If a run has stage-specific knobs, place them under `config_snapshot.<stage>` rather than flattening them into root keys.
- When updating an existing note, overwrite fields only when the new source is more canonical or more complete than the prior one.
- For in-flight runs, prefer explicit `stage_statuses` and partial sections over vague prose like "still running."

## Note Shape

Use the template exactly once per note and keep these sections in this order:

1. `Summary`
2. `Run Context`
3. `Observed Results`
4. `Failure Analysis`
5. `Hypotheses`
6. `Next Run`
7. `Sources`

The frontmatter is for machine-readable indexing. The body is for human judgment and downstream review.

## Interpretation Heuristics

- Treat `experiment_summary.json` as the canonical top-level snapshot.
- Treat `experiment.json` plus the referenced experiment spec as the canonical source for config intent.
- Treat `next_run_candidates.json` as the canonical source for ranked recommendations and high-loss snapshot summaries.
- Use `hypothesis_context.json` when you need richer tag-level evidence or supporting context behind a recommendation.
- Use `run_matrix.csv` to confirm stage status rather than inferring completion from one artifact alone.
- If schema pass rate is materially higher than behavior pass rate, call out behavior reliability as the likely bottleneck instead of just "tool calling."

## Output Discipline

- Use short paragraphs and flat bullets.
- Name concrete failure families or tags when the artifacts support them.
- Include exact artifact paths in `sources`.
- If the user asks for a comparison note, keep the same template and populate `comparison_experiment_ids`.

## Bundled Resources

- Template: [assets/research_note_template.md](assets/research_note_template.md)
- Artifact guide: [reference/artifact-map.md](reference/artifact-map.md)
