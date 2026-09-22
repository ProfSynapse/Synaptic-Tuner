# Skill spec: source-ingestion

Status: APPROVED 2026-09-19.

Every field below must be filled before this spec is presented for approval.
`UNRESOLVED` is the marker for a field still owed an answer; the alignment gate
is zero `UNRESOLVED` fields plus explicit user approval.

## Purpose

Teach agents and operators to take user-selected files, folders, archives, or structured exports through Syntunia's source-agnostic ingestion API: declare one or more source structures, apply metadata rules, and produce a verified normalized bundle that later dataset-building work can consume. V1 includes one worked ingestion recipe because it is the current proven need: Markdown notes with YAML frontmatter.

## Trigger

Use this skill when an agent needs to:

- select or upload source material for ingestion;
- define, propose, review, or bind source structures;
- map fields, text projections, metadata, groups, or relationships;
- plan, preflight, execute, observe, resume, reconcile, or verify ingestion;
- use or adapt the checked-in Markdown-with-YAML-frontmatter ingestion recipe;
- diagnose an ingestion plan, source binding, normalization, metadata, or recipe failure.

### Should not trigger

- Fine-tuning, evaluating, merging, quantizing, or deploying a model after a training dataset already exists; use the corresponding training/evaluation/deployment skill.
- Synthetic example generation or improvement when the source bundle and dataset recipe are already settled; use the synthetic-data-generation skill.
- Publishing datasets or models to external services.
- Directly editing or normalizing a user's source files in place.

## Workflow

1. Establish the intake context: local folder/file selection, uploaded batch, archive, or existing immutable snapshot.
2. Inventory the admitted logical sources without assigning semantic meaning.
3. Load or author one or more declarative structure definitions and source bindings.
4. Optionally generate bounded structure proposals; require explicit acceptance or edits before those proposals become authoritative configuration.
5. Define metadata schemas, mappings, stamps, precedence, grouping, relationships, limits, and failure policy.
6. Plan and preview source-to-structure matches, exclusions, diagnostics, and projected normalized items.
7. Preflight the exact immutable plan, then start and observe ingestion through the public API.
8. Resume or reconcile only when the typed lifecycle says it is eligible.
9. Verify the immutable normalized bundle and its structure, binding, item, metadata, and provenance digests.
10. Hand the verified normalized bundle and its declared projections to the appropriate dataset-building workflow without inventing training-example semantics inside ingestion.

## Inputs

- User-authorized local file/folder selections, uploaded file manifests, supported archives, existing source snapshot references, or supported structured exports.
- Optional YAML/JSON structure definitions and bindings.
- Optional metadata schemas, mappings, stamps, path/root/source rules, grouping keys, and relationships.
- An ingestion objective. V1's worked example may additionally receive the declared YAML-frontmatter metadata schema for Markdown notes.
- Host limits and privacy/failure policy.

## Outputs

- Inspectable structure configuration and source bindings.
- A deterministic ingestion plan and typed diagnostics.
- A verified private normalized source bundle with content/provenance digests.
- A stable, documented handoff contract for later dataset construction, including structure identities and named text projections.
- No publication unless a separate publishing workflow is explicitly requested.

## Good output

"I upload my stuff, optionally define metadata and one or more structures beforehand, inspect what will happen, and then easily convert the verified result into trainable data without the system being tied to my filesystem or use case."

## Bad output

"A special pipeline over-indexed on Rose n Thorn, Nexus, Obsidian, one folder layout, or one current file schema; the parser guesses what my data means and silently turns it into training rows."

## Constraints

- Structure-first and config-first: declared or explicitly accepted structures are semantic authority; auto-detection is advisory only.
- Use the public agentic API and checked-in reusable scripts. Do not create ad hoc extraction or conversion scripts when a supported surface exists.
- Keep source intake, normalized structures, and dataset recipes separate.
- Never hardcode a user's paths, filenames, story names, vaults, database schema, tool wrapper, or dataset shape into runtime code.
- Never modify source material in place.
- Keep raw source content, metadata values, absolute host paths, and parser exception prose out of lifecycle records, default logs, diagnostics, and aggregate reports.
- Require deterministic ordering, stable IDs/digests, immutable snapshots/bundles, exact plan/preflight binding, and verification before dataset construction.
- Use safe bounded parsing; no arbitrary code, SQL, shell, templates, macros, external entities, network fetches, or environment expansion from source configuration.
- Canonical skill source lives under `.skills/`; synchronized skill trees must remain identical.

## Out of scope

- Worked ingestion recipes other than Markdown notes with YAML frontmatter. Additional recipes are added only when a concrete use case requires them.
- Dataset-recipe authoring, training-example semantics, segmentation policy, and a speculative catalog of common training recipes.
- V1 source formats beyond UTF-8 text/Markdown, JSON/JSONL/YAML/CSV, and bounded ZIP/TAR/TAR.GZ expansion.
- PDF, DOCX, OCR, images, audio, video, Parquet/Arrow, arbitrary SQLite/database files, live database connections, URLs, or cloud-drive connectors.
- Model training, evaluation, quantization, upload, or deployment.
- External publication of normalized bundles, datasets, or models.

## Done

The skill is complete when:

- an agent can follow it from approved folder/upload intake through a verified normalized bundle without internal-module access;
- it supports the approved Markdown-notes-with-YAML-frontmatter example through a verified normalized bundle;
- structure proposals cannot silently become semantic authority;
- templates cover structure definitions, bindings, metadata policy, and ingestion requests;
- the Markdown-with-YAML-frontmatter recipe is documented, executable through supported repository surfaces, and validated with privacy/provenance checks;
- validators catch malformed structure/binding/recipe configuration and skill-tree drift;
- the end-to-end test proves deterministic repeated output and a clean, structure-bound handoff for later trainable-data construction.

## Delivery

Canonical project-local skill at `.skills/source-ingestion/`, synchronized to `.agents/skills/source-ingestion/` and `.claude/skills/source-ingestion/`. Do not build or deliver a standalone `.skill` package.

## Open decisions

- Decision from the user: recipes are lazy-loaded from concrete needs. V1 ships only the Markdown-notes-with-YAML-frontmatter ingestion recipe and does not invent a general recipe catalog.
- Decision from the user: custom dataset-recipe authoring is deferred until a real dataset-conversion requirement is agreed.
- Decision from the user: the skill is local to this project and is not packaged as a standalone `.skill` artifact.

## Change log

- 2026-09-19, user-approved amendment after Prepare: remove the mixed Markdown-plus-structured-data V1 example; retain only Markdown notes with YAML frontmatter.
- 2026-09-19, user-approved sequencing: pause the skill workflow after Prepare. Resume architecture, creation, and testing only after the public ingestion API has successfully completed the approved Markdown/frontmatter ingestion run, so the skill is written from checked-in ground truth.
- 2026-09-19, user-approved authority boundary: exact frontmatter parsing, metadata precedence, bundle layout, archive limits, and lifecycle vocabulary come from the implemented and validated ingestion API; the skill documents and validates that behavior rather than inventing a parallel contract.
