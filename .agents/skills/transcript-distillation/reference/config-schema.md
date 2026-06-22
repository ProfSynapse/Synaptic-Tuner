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

## `reasoning` (map)
Optional reasoning/thinking capture. Off by default.
- `capture`: when `true`, each row gets a separate `reasoning` field (the
  `completion` stays answer+tool-calls only). The field is secret-scrubbed like
  the others.

What's actually recoverable from the logs:
- **Codex**: the readable reasoning **summary** (the `summary_text` blocks) is
  captured. The verbatim chain-of-thought is in `encrypted_content` — an opaque
  provider-held ciphertext — and is **not** recoverable.
- **Claude Code**: `thinking` blocks store only a `signature` (no plaintext), so
  `reasoning` is empty for Claude rows. Capture is wired generically so it would
  populate if a future format logs plaintext thinking.

## `feedback` (map)
A per-turn **human-reaction** weight, read off the NEXT human turn, stamped as
`metadata.user_feedback` (`praised`/`accepted`/`neutral`/`corrected`) and
`metadata.feedback_score` (`+2`/`+1`/`0`/`-1`). Orthogonal to `quality_tier`:
the tier grades the *session* by execution outcome, this grades the *turn* by
what the human said next. Applies to **all** sources (a praised/accepted coding
turn is extra-good; a corrected one is suspect), and is often the only quality
signal for chat-style sources that never run tests/builds.

- `enabled`: bool.
- `praise_markers`: explicit-approval phrases. Matched anywhere within the first
  `praise_window_chars` of the turn. Keep these task-directed — omit prose-common
  words (beautiful/amazing/love) or they fire on pasted fiction/quotes.
- `accept_markers`: short acceptance/proceed phrases. Only count when the turn is
  ≤ `max_accept_chars` AND starts with one (so "yes but you forgot X" isn't a
  clean accept).
- `reject_markers`: dissatisfaction phrases. Matched anywhere within the first
  `window_chars`. Use specific multi-word phrases; avoid ambiguous bare words.
- `window_chars` (default 100) / `praise_window_chars` (default 50): feedback is
  front-loaded, so markers only count near the start of the turn — this avoids
  matching a marker buried in pasted code/JSON/a long prompt.

Precedence is conservative: `corrected` > `praised` > `accepted`. A turn that
opens with a markdown quote (`>`) is treated as pasted content and scored
`neutral`.

## `domain` (map)
A per-turn **content domain** tag, stamped as `metadata.domain`
(`"coding"` | `"chat"`), so downstream training can split coding from
conversational data. Deterministic/regex — **no LLM**. Decided purely on the
row's concatenated text (`prompt + " " + completion + " " + reasoning`),
**independent of `tool_names` and `source_kind`**.

**`coding` means AUTHORING CODE, not merely invoking a tool.** A chat turn that
calls `web_search` / `analysis` / `artifacts` / any MCP tool is **still
`chat`** — tool-call presence is **not** a coding signal. A turn is `coding`
only when its text contains real code:
- a fenced block tagged with a known language, OR an untagged fence whose body
  has ≥4 programming symbols (`{};=()`) or a syntax marker (`def `, `import `,
  `function `, `class `, `=>`, `fn `, ...);
- a unified diff (`diff --git`, `@@ ... @@`, `+++`/`---` hunks);
- a traceback / panic / segfault;
- a build/test/git command invocation (`pytest`, `npm `, `cargo `, `go test`,
  `git commit`, `pip install`, `make`, ...);
- a code-file extension (`.py`, `.ts`, `.rs`, ...) alongside programming syntax.

Everything else is `chat`. Runs uniformly on all sources — codex/claude_code
turns naturally tag `coding`; pure-planning turns tag `chat` (correct and
intended; domain is per-turn content, not per-source).

- `enabled`: bool (default **true**). When `false`, every row is stamped
  `domain="chat"`.
- `lang_tags`: list of fence language tags that count as code (defaults cover
  python/js/ts/rust/go/java/sql/yaml/etc.).

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
- `drop_content_substrings`: drop rows whose prompt OR completion contains any
  of these substrings (case-insensitive). Use to scrub confidential
  client/topic mentions that bleed across projects, beyond the project-slug
  `scope.exclude_substrings`. Off by default (empty list).
- `require_outcomes`: empty = tag-only (keep all, tiered). Set to e.g.
  `[tests_passed, clean_build]` to keep ONLY verified-good sessions.
