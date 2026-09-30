# Protocol: deliver project-local

Context: validation and mirror synchronization are complete.

## Mission

Deliver the source-ingestion skill as repository source, with no standalone
package and no implied publication.

## Steps

1. Report the canonical path `.skills/source-ingestion/` and confirm the
   `.agents` and `.claude` mirrors passed the repository sync check.
2. Report the validators run and their exit status, plus any known platform skip
   or operational limitation.
3. State that V1 contains exactly one recipe: Markdown with optional YAML
   frontmatter.
4. State that no `.skill` artifact was built, no external publication occurred,
   and dataset construction remains a separate workflow.
5. Commit or publish only when the user has separately authorized that Git or
   external action.
6. Stop after handing back the paths and evidence; this is the terminal delivery
   step.

## Guidelines

- Pattern: make the no-package exception explicit so a future agent does not
  apply Skill Crafter's default artifact path.
- Anti-pattern: describe a canonical tree as delivered before mirror sync, or
  silently create a package because generic guidance normally recommends one.

## Next

This is the terminal protocol. This project-local exception ends with validated,
synchronized source trees and never a standalone `.skill` artifact.
