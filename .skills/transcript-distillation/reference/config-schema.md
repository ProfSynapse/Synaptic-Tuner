# Config Schema

Every knob in a distill config. Start from `configs/template.yaml`.

## `sources` (map)
Each entry is one source of transcripts. The key (e.g. `claude_main`) is the
`source_kind` stamped on rows and used for `--sources`.

```yaml
sources:
  claude_main:
    format: claude_code     # adapter name (REGISTRY in scripts/adapters/__init__.py)
    enabled: true
    root: "~/.claude/projects"   # ~ is expanded
    glob: "*/*.jsonl"            # glob under root (recursive ** supported)
```

Multiple sources can share a format (e.g. `claude_main` and `claude_subagent`
both use `claude_code` with different globs).

## `scope` (map)
Filter which projects are included.
- `include_substrings`: a transcript is kept only if its scope-key contains one
  of these (case-insensitive). **Empty list = include everything.**
- `exclude_substrings`: ...and none of these.

The scope-key is adapter-defined: `claude_code` uses the project-slug;
`codex` uses the `cwd` recorded in `session_meta`.

## `output` (map)
- `dir`: output directory, relative to the current working directory (or absolute).
- `rows_file`: filename (default `rows.jsonl`).
- `smoke_subdir`: subdir used when `--smoke` is passed (default `smoke`).

## `sanitize` (map)
Deterministic secret scrub, applied to every text field at emit time.
- `enabled`: bool.
- `replacement`: template, `{name}` = pattern name.
- `patterns`: name → regex. Each match replaced with the redaction marker.
- `env_assignment.sensitive_key_re` / `.replacement`: redacts the VALUE in
  `KEY=value` / `KEY: value` lines when the key matches (keeps the key name).

## `context_budget` (map)
- `max_context_tokens`: caps each row's context by estimated tokens. This is the
  **compute knob** — set it to your training sequence length. It bounds context
  DEPTH per row, not the number of rows.
- `chars_per_token`: estimate divisor (4.0 ≈ English; ~3.5 for code-heavy).
- `max_context_messages`: hard ceiling on context messages regardless of tokens.

Context is filled most-recent-first after reserving room for the completion.
A turn whose completion alone exceeds budget is still emitted, flagged
`metadata.oversize=true`.

## `labeling` (map)
Deterministic per-turn label.
- `correction_markers`: the next human turn is a correction (=> reject) only if
  it is short (`max_correction_chars`) and STARTS WITH one of these.
- `max_correction_chars`: length ceiling for a correction.
- `interrupt_markers`: substrings marking a user interrupt.

Result: `label=true` (clean accept), `label=false` (human-corrected reject),
`label=null` (errored/interrupted but no human signal → borderline → send to a judge).

## `session_outcome` (map)
Classifies the WHOLE session by terminal signal, by matching tool commands and
their output. All regexes are case-insensitive.
- `test_command_re` / `build_command_re` / `commit_command_re`: which commands count.
- `pass_re` / `fail_re`: success/failure tokens in command output.
- `approve_re` / `approve_max_chars`: a short final human turn = approval.

Detected outcomes: `tests_passed`, `clean_build`, `committed`, `pr_created`,
`user_approved`. (`pr_created` also comes from adapter session signals, e.g.
Claude's `pr-link` events.)

## `quality_tiers` (map)
- `gold`: list of outcomes that mean code verifiably worked.
- `silver`: outcomes that mean saved/shipped.
- anything else → `bronze`.
- `inherit_parent_for_subagents`: subagents inherit their parent session's
  outcome (their own transcript never contains the commit/PR/test).

## `quality_filter` (map)
- `enabled`: bool.
- `drop_tools`: a turn whose only tool calls are these (ceremony/orchestration)
  and which has no substantive text is dropped.
- `drop_skills`: `Skill()` invocations of these are ceremony.
- `min_completion_tokens`: text-only turns shorter than this are filler → dropped.
- `dedup.enabled`: drop near-duplicate completions (normalized hash).
- `drop_prompt_markers`: drop rows whose prompt starts with a ritual marker
  (off by default — can over-drop).
- `require_outcomes`: empty = tag-only (keep all, tiered). Set to e.g.
  `[tests_passed, clean_build]` to keep ONLY verified-good sessions.
