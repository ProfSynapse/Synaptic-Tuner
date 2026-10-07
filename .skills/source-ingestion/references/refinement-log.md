# Refinement log

Append-only, newest-first record of evidence-backed changes made through
`../protocols/self-refine.md`.

<!-- YYYY-MM-DD | evidence/observation | change made | file(s) touched -->

- 2026-10-02 | A valid rich Markdown note exceeded the aggregate 128-entry YAML
  mapping cap; a narrowly approved 512-entry change passed 41 parser tests,
  one public acceptance test, and a fresh verified 106-source intake | Documented
  the 512-entry aggregate bound and all independent unchanged guards; kept the
  same single Markdown/frontmatter recipe, without source edits or packaging |
  `markdown-yaml-frontmatter.md`.

- 2026-10-02 | One real 106-source intake stopped at preflight with aggregate
  `parse_failed`; inspection confirmed the exited CLI retained no failed member
  or parser code | Clarified the diagnosis limit and bounded read-only
  localization path; no inferred source defect, source mutation, replay, new
  recipe or runtime diagnostic surface | `../protocols/diagnose-ingestion.md`.

- 2026-09-19 | Initial project-local skill created from the accepted public API,
  CLI, deterministic fixture, and real 75-document ingestion evidence | Added
  the single Markdown plus optional YAML-frontmatter recipe and no-package
  delivery workflow | Initial skill tree.
