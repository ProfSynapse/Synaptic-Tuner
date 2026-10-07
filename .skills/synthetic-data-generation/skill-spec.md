# Skill spec: synthetic-data-generation export improvement

Status: APPROVED 2026-09-20 by the user in conversation.

## Purpose

Extend the existing project-local synthetic-data-generation skill with the proven post-generation workflow for exporting admitted structured results into create-only Markdown staging. Keep the guidance generic and agent-ergonomic while documenting one grounded YAML-frontmatter/Markdown recipe.

## Trigger

Use this improvement when an agent has generated or judged structured document artifacts and needs to render them deterministically into local Markdown files, preserve evidence bindings, or stage them safely for human review.

### Should not trigger

- Do not use it for source ingestion or folder selection before dataset/example construction; use the source-ingestion skill.
- Do not use it for fine-tuning, deployment, model upload, or publication.
- Do not use it to promote generated files into authored/canonical locations without a separate review decision.

## Workflow

1. Keep source selection, prompts, schemas, metadata, output mappings, templates, and disposition policies in declarative configuration.
2. Generate through the existing structured-document workflow, locally admit every result, and complete the configured judging pass.
3. Configure explicit document/model-to-relative-path mappings for the structured-document exporter; do not derive filenames from generated prose.
4. Run export preflight and dry-run before requesting or exercising write authority.
5. Execute only into a dedicated create-only staging root, then run the read-only verify phase.
6. Preserve accept, review, and reject dispositions in the manifest; do not overwrite or auto-merge authored files.
7. Record the run and export evidence, including digests, counts, failures, and review gates.

## Inputs

- A completed structured-document batch collection produced by the checked-in bakeoff CLI.
- A complete judgment manifest when judgment mode is required.
- A `structured_document_export/v1` YAML configuration with explicit output mappings, render selection, disposition policy, destination, and limits.
- User authorization for any provider spend and for the destination write.

## Outputs

- Deterministic UTF-8/LF Markdown files in the caller-configured staging root.
- A canonical JSONL manifest binding source, generation, judgment, rendered bytes, disposition, and output path.
- Machine-readable summaries for preflight, dry-run, execute, and verify.
- Project-local run notes or journal updates when the surrounding project requires them.

## Good output

The user's grounded example is the YAML-frontmatter and Markdown-notes workflow: define the structure beforehand, generate and judge the structured content, then convert it into reviewable Markdown through an ergonomic API. The proven result created 229 proposed outline files plus one manifest in a dedicated `_generated` staging root, with no authored overwrite and a successful byte-for-byte verify pass.

## Bad output

The user explicitly does not want a recipe that over-indexes on one file system or use case, made-up recipes that have not been tested, direct writes into authored outline folders, invented tags or ordering conventions, audio-script handling, or a skill packaged or published outside this project.

## Constraints

- The runtime must remain format-agnostic; corpus-specific paths and schemas belong only in private caller configuration.
- Use only checked-in CLIs and config surfaces; do not write an ad hoc copy script.
- Run local schema and expected-output admission before export.
- Use explicit stable output mappings and create-only publication.
- Fail closed on source/result/judge drift, unsafe paths, collisions, partial manifests, and verification mismatch.
- Never overwrite, auto-merge, or silently repair authored files.
- Keep source prose, generated payloads, judgment rationales, and private paths out of CLI errors and summaries.
- Preserve the current generation, improvement, validation, scenario, rubric, and rollout-projection guidance.

## Out of scope

- Designing additional untested recipes.
- Promoting staged files into canonical book-local locations.
- Human repair of review rows.
- Source ingestion, dataset training-row construction, fine-tuning, deployment, publishing, or model serving.
- Packaging the skill as a `.skill` artifact.

## Done

- The skill routes agents to a focused structured-document export reference and the proven YAML-frontmatter/Markdown recipe.
- The reference documents preflight, dry-run, execute, verify, evidence bindings, disposition behavior, recovery, and privacy boundaries.
- Generic skill validation passes, existing domain guidance remains intact, and `.agents/skills` plus `.claude/skills` match the canonical `.skills` tree.
- No packaged artifact is created.

## Delivery

Canonical source: `.skills/synthetic-data-generation/`, with synced project mirrors under `.agents/skills/` and `.claude/skills/`. This is local project guidance only. Do not create, package, install, or publish a `.skill` artifact.

## Open decisions

- Keep exactly one recipe in this improvement: optional YAML frontmatter plus Markdown notes rendered from admitted structured output.
- Treat generated files as proposed staging artifacts even when the judge verdict is `accept`; promotion is a separate authority boundary.

## Change log

- 2026-09-20: User approved the written improvement specification after an end-to-end structured corpus run and verified staging.
- 2026-09-20: After assessment exposed 25 legacy validator errors and an oversized router, the user explicitly approved a delegated full structural repair rather than the baseline-aware minimal option. Preserve domain behavior, make the generic validator fully green, sync mirrors, and continue to omit packaging/publication.
