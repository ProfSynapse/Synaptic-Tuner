# Markdown and YAML frontmatter

Context: read when declaring fields or diagnosing parse behavior for the one V1
recipe.

## Key idea

Each admitted UTF-8 Markdown file is one unit. Optional frontmatter means the
block may be absent; it does not mean a present malformed block is ignored.

## Body and delimiters

The parsing profile is `markdown_yaml_frontmatter_v1` with file boundaries.
`frontmatter_mode` is one of:

- `none`: frontmatter markers are ordinary body text.
- `optional`: a file without a frontmatter opener is valid; a present block must
  be valid.
- `required`: every file must contain a valid block.

Frontmatter delimiters are exact. The parser normalizes supported UTF-8/BOM and
line-ending variants into deterministic field values. It does not execute
templates, environment expansion, macros, SQL, shell, external entities, or
network fetches.

Source-member admission budgets are configured separately from parsing. The
frontmatter block retains its fixed 64 KiB bound even when a larger member
budget is configured.

## Versioned closed YAML subset

`markdown_yaml_frontmatter_v1` preserves the original bounded subset: plain
`null`, `Null`, `NULL`, and `~` are invalid, while an empty plain scalar remains
an empty string. Strict CLI schema v1 continues to bind this profile.

`markdown_yaml_frontmatter_v2` adds JSON-null representation inside arrays and
objects. Plain `null`, `Null`, `NULL`, `~`, and empty mapping or sequence values
normalize to JSON null; quoted null-like text remains a string. Strict CLI
schema v2 binds this profile. A top-level frontmatter field whose value is null
still cannot satisfy a non-null declared field kind.

Both profiles preserve the same depth, node, mapping, sequence, key, scalar,
and 64 KiB frontmatter bounds. Ambiguous or unsafe YAML features fail closed
rather than being guessed or executed. Declared field types must match parsed
values.

Frontmatter keys are not automatically metadata. A key becomes an output field
only through a `frontmatter_field` mapping, and becomes metadata only through an
additional metadata declaration. This preserves declared semantic authority.

## Failure rule

A malformed present block produces the sanitized `parse_failed` diagnostic at
preflight. The operator may correct the structure config if it was wrong. Editing
the source document is a separate action requiring explicit user authority; the
ingestion request itself never grants it. After any correction, readmit and plan
the current bytes rather than reusing stale evidence.

## One recipe, not a catalog

This skill does not define recipes for JSON, databases, archives, PDFs, or media,
even if broader product contracts may later support them. Add another worked
recipe only after a concrete use case is aligned, implemented, and proven.
