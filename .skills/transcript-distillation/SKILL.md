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
in small pluggable *adapters*. Two ship built-in (`claude_code`, `codex`); add
more in ~40 lines (see `reference/writing-adapters.md`).

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
    "tool_names": ["Edit"], "est_tokens": 5948, "oversize": false, ...
  }
}
```

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
