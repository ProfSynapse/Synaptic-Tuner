---
name: synthetic-data-generation
description: Use SynthChat's config-first CLI workflows to generate, judge, improve, validate, project, and export synthetic datasets or structured documents. Use when authoring scenario or rubric YAML, configuring generation models, running small quality smokes before scaling, operating structured-document batch generation, or rendering admitted results into reviewable Markdown. This skill is for using checked-in workflows, not modifying runtime source code.
allowed-tools: Read, Bash, Write, Grep, Glob
---

# Synthetic Data Generation

Use checked-in SynthChat commands and declarative configuration to produce
locally admitted, reviewable data. Keep corpus shape, provider routing, prompts,
schemas, mappings, and quality policy in caller configuration.

## Workflow

1. Classify the task, then read only the matching route:
   - Generate, improve, validate, sanitize, or inspect CLI options:
     `references/cli-commands.md`.
   - Configure providers, models, workers, targets, privacy, or runtime checks:
     `references/settings-config.md`.
   - Author or revise scenarios: `references/scenario-authoring.md`.
   - Author or revise rubrics: `references/rubric-authoring.md`.
   - Understand generation quality gates, repair loops, privacy, and config-first
     architecture: `references/generation-workflow.md`.
   - Regenerate selected rows with checked-in commands:
     `references/targeted-regeneration.md`.
   - Project environment rollouts into SFT, KTO, or GRPO data, or pick the
     `validation_group_key` for training on them:
     `references/rollout-projection.md`.
   - Plan run-size and cost projections: `references/rollout-projection.md`.
   - Export admitted structured results to Markdown:
     `references/structured-document-export.md`.
2. Before any nontrivial or scaled generation, follow
   `protocols/testing-before-scale.md`. You MUST dry-run a small representative
   sample, inspect the raw and rendered artifacts, and obtain the user's approval
   before scaling. Treat judge-less runs as explicit plumbing exceptions.
3. Keep format assumptions in YAML, JSON Schema, rubric text, templates, and
   explicit mappings. NEVER hardcode a current wrapper, corpus, filename rule,
   metadata convention, or dataset shape into runtime code for a generation job.
4. For structured documents, use the checked-in bakeoff lifecycle to generate,
   collect, locally validate, and judge results. Then use the checked-in exporter
   lifecycle in order: preflight, dry-run, execute with explicit write authority,
   verify. NEVER promote staged output into authored locations implicitly.
5. For the single grounded Markdown recipe, load
   `references/recipes/yaml-frontmatter-markdown.md`. Treat its names as
   placeholders and adapt all schema fields and paths through configuration.
6. Validate outputs with the command appropriate to the route. Preserve run
   manifests, usage, local schema failures, judgment dispositions, digests, and
   review gates. Stop on drift, unsafe paths, collisions, or verification failure.

## Boundaries

- Use `source-ingestion` for selecting and normalizing source folders before
  dataset-example construction.
- Use `evaluation` for model evaluation scenarios and comparisons.
- Use `fine-tuning` after the dataset is admitted and ready to train.
- Do not package, publish, deploy, or promote generated artifacts unless the user
  separately authorizes that action.

## Public dataset validation limits

The reference `DataAPI.validate` scans at most 4 MiB per physical line, 64 MiB
per invocation and one million physical lines before parsing rows. The line
ceiling includes terminator bytes. The byte
and line budgets include blank lines, the rescanned cursor prefix and page
lookahead. Later pages beyond these bounds reject with `manifest_invalid`;
oversized or invalid line-number cursors reject with `cursor_invalid`. These
are finite read limits, not filesystem-call deadlines or immutable dataset
verification. LF, CRLF, bare CR and UTF-8 replacement decoding remain supported.
Use the existing preparation workflow for immutable provenance/split identity
before training; structural row validation does not establish that identity.

## Public metered workflow evidence

The checkout-owned `examples/nontraining/qualification.py` is a config-first
public Data/evaluation consumer with `plan`, create-only `execute` and retained
`readback`. Use its README/template rather than rebuilding the composition.
Generation validates before improvement, which consumes its unchanged JSONL
and configured `metadata.resolved_environment_config` projection. The improved
row validates before evaluation. Configuration commitments are rechecked
between stages; unknown upstream outcomes stop downstream calls. Existing state
is inspected, never automatically replayed. The example adds no engine runtime
or provider qualification, active-request recovery, sandbox or invoice cap.

Installed CPU tests join Data generation and improvement with the fixed local
worker, measured scripted judge/improver calls, one spend closure per run and
SQLite retained readback after input removal. Missing usage remains
indeterminate. These controls establish plumbing and accounting semantics;
they do not qualify a live model/account, restore active requests or worker
ownership, or publish structured Markdown. Public evaluation also compares
retained measured reports without reopening a backend; accounting uncertainty
can coexist with comparable scoring and is never repaired by comparison.
