# Protocol: ingest local Markdown

Context: start here when a user wants authorized Markdown files or folders
normalized through Syntunia ingestion.

## Mission

Establish a bounded, private ingestion objective without treating a filesystem,
vault, or current corpus layout as product semantics.

## Steps

1. Confirm the user-authorized files or folders and the private destination for
   ingestion outputs. Treat folder-picker UI, uploads, Nexus, and vault layouts
   as clients or source locations, not as the ingestion contract.
2. Record selection aliases that reveal no private host path in logical output.
   Use one stable alias per selected root and never place absolute paths in
   configs, checkpoints, logs, or handoff prose.
3. Bound discovery explicitly: list include patterns, exclude patterns, and the
   hidden-file policy. Prefer an exact allowlist when nearby material is outside
   scope.
4. Confirm the semantic authority: V1 supports exactly the declared Markdown
   structure described in `../references/markdown-yaml-frontmatter.md`.
   Auto-detected proposals, if used through the API, remain advisory until a
   human explicitly accepts them into configuration.
5. Confirm that source correction is not authorized by ingestion itself.
   Reading and normalizing the admitted snapshot is allowed; editing source
   requires a separate user instruction.
6. Stop when authority, aliases, discovery policy, output privacy, and the
   declared structure are explicit.

## Guidelines

- Pattern: express corpus-specific choices as config and selection arguments.
- Anti-pattern: encode a vault name, story hierarchy, or one user's metadata
  keys into runtime code or this reusable skill.

## Next

Run `configure-markdown-recipe.md` to create and validate the declarative CLI
configuration.
