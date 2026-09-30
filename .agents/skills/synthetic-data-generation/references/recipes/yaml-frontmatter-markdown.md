# Recipe: YAML Frontmatter and Markdown Notes

This is the skill's single grounded export recipe. It is generic: every angle-
bracketed value is a caller-owned placeholder, and every selected field must be
declared by the caller's structured-output schema. Do not copy its field names as
a universal data model.

## Prerequisites

- A collected structured-document batch whose generation payloads passed the
  configured JSON Schema and expected-output admission.
- A completed judgment manifest when judgments are required.
- Explicit stable output paths chosen independently of generated prose.
- A dedicated create-only staging root.

## Export configuration

```yaml
kind: structured_document_export/v1
input:
  bakeoff_config: <relative-path-to-bakeoff-config.yaml>
  judgments:
    mode: required
    manifest_path: <relative-path-to-judgment-manifest.json>
    disposition_pointer: /judgment/payload/<verdict-field>
    disposition_map:
      <accepted-value>: accept
      <review-value>: review
      <rejected-value>: reject
    when_absent: review
destination:
  root: <dedicated-staging-root>
  manifest: export-manifest.jsonl
  emit:
    accept: true
    review: true
    reject: false
outputs:
  - document_id: <stable-document-id>
    model: <exact-model-id>
    relative_path: <explicit/relative-note-path.md>
render:
  frontmatter:
    artifact_kind:
      $literal: <caller-defined-kind>
    document_id:
      $select: /identity/document_id
    model:
      $select: /identity/model
    disposition:
      $select: /quality/disposition
    source_sha256:
      $select: /hashes/source_sha256
    generation_sha256:
      $select: /hashes/generation_result_sha256
    <metadata-key>:
      $select: /source/metadata/<metadata-key>
  body_template: |-
    # {{ title }}

    {{ body }}
  variables:
    title:
      select: /generation/payload/<title-field>
      format: text
    body:
      select: /generation/payload/<body-field>
      format: markdown
limits:
  max_documents: <positive-integer>
  max_output_bytes_per_document: <positive-integer>
  max_total_output_bytes: <positive-integer>
```

Run preflight and dry-run, review their summaries, execute only with destination
write approval, then verify. Add one explicit `outputs` item for each admitted
document/model identity. Do not infer note paths from the generated title, write
into authored folders, or treat judge acceptance as promotion authority.
