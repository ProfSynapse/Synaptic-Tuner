# Structured Document Export

Use this lifecycle after structured-document batch generation has been collected,
locally admitted, and judged. The checked-in exporter is
[SynthChat structured-document exporter](../../../SynthChat/scripts/structured_document_export.py),
and its behavior is exercised by
[structured-document export tests](../../../tests/synthchat/test_structured_document_export.py).
It makes no provider calls.

## Contract

The exporter consumes a `structured_document_export/v1` YAML file. Configuration
declares:

- the bakeoff config and optional or required judgment manifest;
- the JSON Pointer used to read a judgment disposition and its mapping to
  `accept`, `review`, or `reject`;
- a dedicated destination root, manifest path, and which dispositions emit;
- one explicit `(document_id, model) -> relative_path` mapping per result;
- frontmatter selections, body-template variables, and render formats;
- document and byte limits.

Paths are never derived from generated prose. The renderer supports literal
frontmatter values and RFC 6901 selections from the admitted artifact context.
Body variables select values and render them as `text`, `json`, `yaml`, or
`markdown`.

## Lifecycle

Run every command from the repository root.

1. Preflight validates configuration, artifact bindings, expected outputs,
   judgments, render selections, limits, and destination-independent invariants:

   ```bash
   python -m SynthChat.scripts.structured_document_export \
     --config path/to/export.yaml --preflight
   ```

2. Dry-run repeats compilation and inspects the destination for unsafe ancestors,
   collisions, conflicting bytes, or partial prior publication:

   ```bash
   python -m SynthChat.scripts.structured_document_export \
     --config path/to/export.yaml --dry-run
   ```

3. Execute only after the destination and write scope are approved. Publication
   is create-only and protected by an exclusive claim. Existing identical files
   are idempotent; conflicting bytes fail closed:

   ```bash
   python -m SynthChat.scripts.structured_document_export \
     --config path/to/export.yaml --execute
   ```

4. Verify recompiles from the bound inputs and compares every emitted file and
   manifest byte-for-byte:

   ```bash
   python -m SynthChat.scripts.structured_document_export \
     --config path/to/export.yaml --verify
   ```

## Evidence and disposition

The canonical JSONL manifest binds source, generation result, optional judgment,
rendered bytes, disposition, and output path with digests. An `accept` disposition
means the configured judge accepted the result; it does not authorize promotion
from staging into an authored or canonical location. Configure emission policy
explicitly, and preserve review and reject rows in the manifest even when their
Markdown files are not emitted.

## Recovery and privacy

- If preflight or dry-run fails, repair declarative config or regenerate the
  bound upstream artifact. Do not copy files around the checks.
- If execute is interrupted, rerun dry-run. Identical completed files may remain;
  changed or partial state fails closed.
- If verify fails, stop and preserve the destination for diagnosis. Do not
  overwrite or auto-merge it.
- Keep source prose, generated payloads, private paths, and judge rationales out
  of summaries. Retain them only in the explicitly configured local artifacts.
- Use a dedicated staging root. Promotion, publication, or authored-file merge is
  a separate user decision.

For the one documented render recipe, see
`recipes/yaml-frontmatter-markdown.md`.
