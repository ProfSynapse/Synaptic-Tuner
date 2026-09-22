# Ingestion CLI config

Context: read while authoring or reviewing the strict JSON file consumed by
`tuner.py ingest`.

## Key idea

The CLI config declares discovery and one Markdown structure; `--select`
arguments carry authorized host paths. Unknown fields, duplicate keys, nonfinite
numbers, or wrong exact types fail as `invalid_input`.

## Exact document

The current template uses `synaptic-ingestion-cli/v2`. Its top level has exactly:

| Field | Meaning |
| --- | --- |
| `schema_version` | Must be `synaptic-ingestion-cli/v2`. |
| `project_ref` | Stable non-secret project identity. |
| `admission_request_id` | Identity for this admission request. |
| `request_id` | Identity for this ingestion request. |
| `admission_limits` | Exact member and aggregate source-admission budgets. |
| `discovery` | Exact include/exclude/hidden policy. |
| `structure` | The sole Markdown structure declaration. |
| `binding` | The sole logical-path matcher. |

`discovery` has exactly `include`, `exclude`, and `include_hidden`. Include must
contain 1-128 string patterns; exclude may contain 0-128; hidden is an exact
boolean. Keep patterns relative to selection aliases.

`structure` has exactly `name`, `version`, `frontmatter_mode`, `fields`,
`text_projections`, and `metadata`. V1 accepts one Markdown structure. Limits are
1-64 fields, 1-16 text projections, and 0-64 metadata declarations.

Each field has exactly:

```json
{
  "name": "body",
  "selector": {"kind": "document_body", "key": null},
  "value_kind": "string",
  "required": true
}
```

Selector kinds are `document_body`, `logical_path`, and `frontmatter_field`.
Only `frontmatter_field` has a non-null `key`. Value kinds are `string`,
`boolean`, `int64`, `number`, `array`, and `object`; body and path are strings.

Each text projection and metadata declaration has exactly `name` and
`field_ref`; the reference must name a declared field. A text projection must
refer to a string field.

`binding` has exactly `binding_id` and `pattern`. Its pattern matches admitted
logical paths, which include the selection alias, not absolute host paths.

`admission_limits` has exactly `max_member_bytes` and `max_total_bytes`. Both
are positive exact JSON integers; booleans and fractional values are invalid.
The member budget must not exceed the aggregate budget. Configurable budgets
are capped by fixed safety ceilings of 16 MiB per member and 32 MiB aggregate.
The 64 KiB frontmatter bound is independent and remains fixed.

The legacy `synaptic-ingestion-cli/v1` document remains accepted with its exact
original fields and fixed 256 KiB member and 32 MiB aggregate budgets. It does
not accept `admission_limits`; use v2 when explicit budgets are required. V1
binds `markdown_yaml_frontmatter_v1`; v2 binds
`markdown_yaml_frontmatter_v2`. The parsing profile is derived from the strict
CLI schema version rather than supplied as another config field.

Operational identifiers (`project_ref`, `admission_request_id`, and
`request_id`) use the public ingestion identity grammar: 1-128 ASCII letters,
digits, underscores, or hyphens, beginning with a letter or digit. Discovery
patterns use the same public whole-path glob grammar as bindings and must be
unique within each include or exclude list.

## Identity and editing

Identifiers are operational, not secret, but should be stable enough to audit.
Changing selection bytes, effective admission limits, semantic config, or
request identity creates new operational evidence. Admission limits bind
operational authority, while a bundle made from the same admitted bytes and
semantic structure remains content-identical. Do not reuse an old
plan/preflight after a change.

Validate every edited config with:

```text
python .skills/source-ingestion/scripts/validate_ingestion_config.py CONFIG
```

The shipped template is the minimal proven body-plus-path recipe. Add
frontmatter fields only when their keys and value kinds have been declared for
the current intake.
