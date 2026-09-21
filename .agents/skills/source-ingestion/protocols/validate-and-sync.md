# Protocol: validate and sync

Context: run after creating or changing the canonical project-local skill and
before delivery.

## Mission

Prove the canonical skill is internally coherent and synchronize it through the
repository's supported mirror workflow.

## Steps

1. Run each domain validator from the repository root:

   ```text
   python .skills/source-ingestion/scripts/validate_ingestion_config.py .skills/source-ingestion/templates/markdown-frontmatter-ingestion.json
   python .skills/source-ingestion/scripts/validate_source_ingestion_skill.py .skills/source-ingestion
   ```

2. Exercise `validate_ingestion_result.py` against a real sanitized command
   result when one is in scope. Never check private result content into the
   skill tree.
3. Run the repository's generic skill validation if available. Resolve broken
   links, incomplete files, or router bloat before synchronization.
4. Synchronize only with the checked-in command:

   ```text
   python .skills/scripts/sync_skill_trees.py
   python .skills/scripts/sync_skill_trees.py --check
   ```

5. Confirm `.skills/source-ingestion/`, `.agents/skills/source-ingestion/`, and
   `.claude/skills/source-ingestion/` match. Do not hand-copy mirror trees.
6. Stop when all validators and the sync check exit zero.

## Guidelines

- Pattern: change `.skills/` first and let the repository command own mirrors.
- Anti-pattern: edit a mirror directly or declare delivery from a canonical-only
  tree after the sync phase has begun.

## Next

Run `deliver-project-local.md` to report the validated repository-local result.
