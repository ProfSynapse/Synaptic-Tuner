---
name: transcript-distillation
description: Turn local agent transcripts (Claude Code, Codex CLI, or any format with an adapter) into high-quality fine-tuning rows. Use when someone wants to mine their own coding-agent conversation logs into an SFT/KTO dataset — parse the logs, extract substantive turns, scrub secrets, score each session by outcome (tests passed / built clean / committed / PR'd), dedup, and tier by quality. Config-driven and format-pluggable; point it at a transcript directory and it produces deduped, quality-tiered JSONL. This skill is about USING the checked-in distill engine via CLI + YAML.
allowed-tools: Read, Bash, Write, Edit, Grep, Glob
---

# Transcript Distillation

Mine local agent transcripts into training data. You point it at a directory of
conversation logs; it emits one row per assistant turn, scrubs secrets, labels
each turn (accept / borderline / reject), scores each *session* by how it ended
up (tests passed, clean build, commit, PR, approval), drops ceremony/filler/
duplicates, and stamps a quality tier (gold/silver/bronze) on every row.

**Format-agnostic.** The engine knows nothing about Claude or Codex — that lives
in small pluggable *adapters*. Built-in: `claude_code`, `codex` (filesystem
glob), `codex_sqlite` (index-backed, cross-platform), and `claude_ai_export`
(the claude.ai bulk `conversations.json` export — one file, many conversations,
**with plaintext thinking**). Add more in ~40 lines (see
`reference/writing-adapters.md`).

## Quickstart

```bash
SKILL=.skills/transcript-distillation

# 1. copy the template config and (optionally) edit scope/paths
cp $SKILL/configs/template.yaml my_distill.yaml

# 2. smoke test on a few files first — ALWAYS do this before a full run
python $SKILL/scripts/distill.py --config my_distill.yaml --smoke --limit 20 --show 3

# 3. full run
python $SKILL/scripts/distill.py --config my_distill.yaml
```

Out of the box the template grabs **all** local Claude Code (`~/.claude/projects`)
and Codex (`~/.codex/sessions`) transcripts and writes to
`./transcript_distill_out/rows.jsonl`. Only dependency: `pyyaml`.

## Multi-machine setup (do this when running on a new machine)

Transcript locations vary by OS and CLI version. **Do not assume the template's
paths exist — detect them first**, then enable only the sources that resolve.
The config is yours per-machine (copy `template.yaml` → `my_distill.yaml` and
edit `root:` paths); the skill itself stays path-agnostic.

Probe these, enable what exists:

| Source | Where to look (per OS) |
|---|---|
| **Claude Code** (`claude_code`) | `~/.claude/projects/*/*.jsonl` (+ `*/*/subagents/*.jsonl`). Same on mac/linux/Windows. |
| **Codex** (`codex`) | `~/.codex/sessions/**/*.jsonl` + `~/.codex/archived_sessions/`. **WSL gotcha:** if Codex ran against the Windows home, they're under `/mnt/c/Users/<you>/.codex` instead. |
| **Codex, layout unknown** (`codex_sqlite`) | If you can't find/locate the rollouts, use `codex_sqlite` with `root: ~/.codex/sqlite` — it reads the thread catalog (`state_*.sqlite`) and follows each `rollout_path`, wherever they live. Lower coverage than the glob if the index omits old threads; prefer the glob when you can resolve the path. |
| **Claude Desktop agent mode / Cowork** (`claude_code`) | Desktop bundles Claude Code, so transcripts are `.../Claude/local-agent-mode-sessions/**/.claude/projects/*/*.jsonl`. Base dir per OS: macOS `~/Library/Application Support/Claude`, Linux `~/.config/Claude`, Windows `%APPDATA%/Claude` (from WSL: `/mnt/c/Users/<you>/AppData/Roaming/Claude`). |

Quick detection one-liner (adjust user as needed):

```bash
for p in ~/.claude/projects ~/.codex/sessions ~/.codex/sqlite \
         ~/Library/Application\ Support/Claude/local-agent-mode-sessions \
         ~/.config/Claude/local-agent-mode-sessions \
         /mnt/c/Users/*/.codex /mnt/c/Users/*/AppData/Roaming/Claude/local-agent-mode-sessions; do
  [ -e "$p" ] && echo "FOUND: $p"
done
```

**claude.ai chats (web/Desktop normal chats):** not on disk — they live
server-side; the local IndexedDB holds only draft-composer autosaves, no
assistant turns. Mine them via the **bulk export**: claude.ai → Settings →
Privacy → Export data, which emails you a `conversations.json`. Point the
built-in `claude_ai_export` source at that file:

```yaml
sources:
  claude_ai_export:
    format: claude_ai_export
    root: "conversations.json"   # the exported file (or a dir of exports)
    glob: ""                     # ignored
```

It splits the one file into one session per conversation, and — unlike the
local CLI transcripts — the export carries **plaintext `thinking`**, so set
`reasoning.capture: true` to keep the chain-of-thought. Scope is the
conversation *title*, so `scope.exclude_substrings` filters chats by topic.
These are chats, not coding runs, so most land in `bronze` (no test/build/
commit signal) — that's expected and honest.

## CLI

| Flag | Meaning |
|------|---------|
| `--config PATH` | distill config YAML (required) |
| `--smoke` | write to a `smoke/` subdir (keeps full runs separate) |
| `--limit N` | cap files **per source** (0 = all) — use for smoke tests |
| `--sources a,b` | restrict to named sources from config |
| `--show N` | print N sample rows to eyeball quality |
| `--max-context-tokens N` | override `context_budget.max_context_tokens` for this run |

## Pipeline (all config-driven)

```
adapter.parse(file)            # format-specific -> normalized events
  -> one row per assistant turn
  -> secret scrub at emit time (raw secrets never hit disk)
  -> context filled most-recent-first under a TOKEN budget
  -> deterministic label: accept / borderline(->judge) / reject
  -> quality funnel: drop ceremony tools, trivial filler, duplicates
  -> session-outcome tiering: gold / silver / bronze
  -> JSONL
```

Each knob is documented in `reference/config-schema.md`. Highlights:

- **`scope`** — include/exclude projects by substring (Claude: project slug;
  Codex: recorded cwd). Empty include = everything.
- **`context_budget.max_context_tokens`** — set to your training seq length.
  This caps context DEPTH per row, *not* the number of rows; training cost is
  ~O(seq²), so keep it real (8192 is a good default; 32768 is ~16× the compute).
- **`session_outcome`** — regexes that detect tests-passed / clean-build /
  commit / PR / approval from tool commands + their output. The strongest
  quality signal: it judges the whole path by where it ended up.
- **`quality_tiers`** — map outcomes to gold (verified-correct) / silver
  (saved/shipped) / bronze (no signal). Subagents inherit their parent
  session's tier.
- **`quality_filter`** — drop ceremony tools (orchestration), trivial text
  turns, and duplicate completions. Set `require_outcomes` to keep ONLY
  verified-good sessions.

## Output row schema

One JSON object per line, shaped for SFT/KTO:

```json
{
  "conversations": [{"role": "...", "content": "..."}],
  "prompt": "...", "completion": "...",
  "label": true | false | null,        // accept / reject / borderline(->judge)
  "score_tier": "good|bad|borderline",
  "failure_labels": ["tool_error", "user_corrected", ...],
  "source_example_id": "claude_main:<file>:<turn>",
  "metadata": {
    "quality_tier": "gold|silver|bronze",
    "session_outcome": ["clean_build", "committed"],
    "good_path": true,
    "domain": "coding",                  // coding|chat — turn CONTENT, not tools
    "tool_names": ["Edit"], "est_tokens": 5948, "oversize": false, ...
  }
}
```

`metadata.domain` is "coding" only when the turn's text authors/operates on code
(code fence, diff, traceback, build/test/git command); calling a tool does NOT
make a turn "coding". Split coding from conversational data with it.

Build a dataset from it by filtering on `metadata.quality_tier` and/or `label`:
SFT on `label==true` gold rows; route `label==null` (borderline) rows to an
LLM-as-judge to mint KTO negatives.

## Privacy

- Secret scrubbing is **on by default** and runs at emit time, so raw API
  keys/tokens never reach disk. Verify with a residual grep after a run.
- Transcript-derived output is private. Keep the output dir gitignored.
- A trained model on frontier-agent transcripts is distillation — keep it
  private (provider ToS).

## References

| File | When |
|------|------|
| `reference/config-schema.md` | Every config knob explained |
| `reference/writing-adapters.md` | Add support for a new transcript format |
| `reference/case-study-claude-codex.md` | Worked example: 538k turns → ~102k tiered rows |

## Discipline

- **Always `--smoke --limit` first**, eyeball `--show` samples, then full run.
- Keep everything in config — never hardcode a tool name, scope, or format
  assumption in the engine. Format specifics belong in an adapter.
