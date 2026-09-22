# Ingestion checkpoint

Status: `verified | blocked | failed`

## Declared scope

- Recipe: Markdown with optional YAML frontmatter
- Project reference: `<non-secret-project-ref>`
- Selection aliases: `<aliases-only; no host paths>`
- Discovery policy digest or concise rule: `<aggregate-only>`
- Structure: `MarkdownNote@1`
- Text projection: `text -> body`
- Metadata declarations: `path -> path`

## Aggregate result

- Sources admitted: `<count>`
- Sources matched: `<count>`
- Sources unmatched: `<count>`
- Sources ambiguous: `<count>`
- Sources processed: `<count>`
- Documents written: `<count>`
- Diagnostic codes: `<closed codes only>`

## Integrity evidence

- Snapshot reference: `<ref>`
- Snapshot manifest digest: `<sha256>`
- Structure-set digest: `<sha256>`
- Plan fingerprint: `<sha256>`
- Run reference: `<ref>`
- Outcome digest: `<sha256>`
- Bundle reference: `<bundle-ref>`
- Bundle digest: `<sha256>`
- Public verification: `<true|false>`
- Bundle inventory: `manifest.json`, `items.jsonl`

## Handoff boundary

- Private bundle location retained by: `<host/project boundary, no absolute path>`
- Source content or metadata values included here: `no`
- Dataset semantics decided here: `no`
- Authorized next workflow: `<dataset recipe or pending decision>`

## Notes

Record only aggregate, path-free operational facts. Source excerpts, frontmatter
values, parser exception prose, and absolute paths do not belong in this file.
