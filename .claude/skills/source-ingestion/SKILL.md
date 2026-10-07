---
name: source-ingestion
description: Ingest user-authorized local Markdown files or folders through Syntunia's public config-first ingestion surface, including optional YAML frontmatter, sanitized diagnosis, deterministic bundle verification, and a dataset-ready handoff. Use for source selection, ingestion configuration, planning, execution, reconciliation, or normalized-bundle handoff; do not use for dataset-example construction, model training, publishing, or unapproved source edits.
---

# Source ingestion

Turn an authorized Markdown selection into a verified, private normalized source
bundle. V1 contains one grounded recipe: Markdown with optional YAML
frontmatter. It does not assign training-example semantics.

## Workflow

1. Establish authority and scope by following
   `protocols/ingest-local-markdown.md`.
2. Configure the one supported recipe with
   `protocols/configure-markdown-recipe.md`.
3. Execute only through the public CLI or API boundary in
   `protocols/execute-ingestion.md`.
4. If execution does not verify, follow `protocols/diagnose-ingestion.md` and
   do not mutate source material without separate authorization.
5. Verify and hand off the normalized source bundle with
   `protocols/verify-and-handoff.md`.
6. Before delivery, run `protocols/validate-and-sync.md`, then finish with
   `protocols/deliver-project-local.md`.

## Map

- `protocols/` contains the ordered procedures. Load only the current step.
- `references/` documents public surfaces, exact config, parsing, lifecycle,
  privacy, bundle, and handoff contracts.
- `templates/` contains the CLI config and aggregate checkpoint to copy.
- `scripts/` validates configs, command results, and this skill tree. Run these
  tools; do not reimplement their checks ad hoc.

## Refine

After a session that exercised this skill, run `protocols/self-refine.md`.
