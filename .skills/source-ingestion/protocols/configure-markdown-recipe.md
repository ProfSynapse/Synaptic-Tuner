# Protocol: configure the Markdown recipe

Context: authority and discovery scope are settled; now declare the sole V1
Markdown structure and binding.

## Mission

Produce a strict, validated `synaptic-ingestion-cli/v2` JSON config without
inventing source meaning.

## Steps

1. Read `../references/ingestion-cli-config.md` and
   `../references/markdown-yaml-frontmatter.md` completely.
2. Copy `../templates/markdown-frontmatter-ingestion.json` to a private config
   location. Replace request and project identifiers with stable, non-secret
   identifiers; keep source paths out of the file.
3. Set `discovery.include`, `discovery.exclude`, and `include_hidden` to the
   authorized scope. Do not rely on an implicit recursive scan.
4. Set `admission_limits.max_member_bytes` and `max_total_bytes` to bounded
   source-admission budgets appropriate to the authorized selection. These are
   operational limits, not semantic structure declarations.
5. Keep the proven `body` and `path` mappings unless the user has declared
   additional frontmatter fields. For each added field, declare its exact key,
   value kind, required policy, and any metadata exposure. Never infer a field's
   training role.
6. Keep `frontmatter_mode` as `optional` when plain Markdown is admissible.
   Absence then succeeds; a present malformed frontmatter block still fails
   closed.
7. Run
   `python .skills/source-ingestion/scripts/validate_ingestion_config.py CONFIG`.
   Resolve every violation before execution.
8. Stop when the validator prints `VALID` and the user-authorized selection
   arguments are separately available as `ALIAS=PATH` values.

## Guidelines

- Pattern: use JSON config for semantic choices and `--select` for host paths.
- Anti-pattern: copy source paths or source content into the config to make the
  command self-contained.

## Next

Run `execute-ingestion.md` with the validated config and authorized selections.
