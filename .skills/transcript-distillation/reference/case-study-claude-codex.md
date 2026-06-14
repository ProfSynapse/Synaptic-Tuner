# Case Study: Distilling Claude Code + Codex into a Fine-Tuning Dataset

The worked example this skill was built from — turning ~2,700 local Claude Code
and Codex CLI transcripts into a private SFT/KTO dataset.

## Goal
Fine-tune a private model on the user's own coding-agent behavior. SFT on a
clean high-quality pool first, then KTO refinement on a held-out set + mined
negatives. Reuse the repo's existing training pipeline; the only new work is
extracting good rows from raw logs.

## Config
`configs/transcript_import/default.yaml` (repo root). Differs from the template:
- `scope.include_substrings: ["documents-code-", "/documents/code/"]` — only
  coding projects.
- `scope.exclude_substrings: ["client-a", "project-b"]` — client/sensitive.
- Output to `private/transcripts/` (gitignored).

Run:
```bash
python .skills/transcript-distillation/scripts/distill.py \
  --config configs/transcript_import/default.yaml --smoke --limit 30 --show 3   # smoke
python .skills/transcript-distillation/scripts/distill.py \
  --config configs/transcript_import/default.yaml                              # full
```

## What the data looked like
- 1,793 Claude files = **53 main sessions + 1,740 subagent traces** (nested under
  `<project>/<session>/subagents/`). 923 Codex rollouts (filed by date; scoped by cwd).
- Claude main transcripts use the newer event schema (skip `attachment`,
  `last-prompt`, `queue-operation`, `system`, ...). Subagent prompts carry heavy
  PACT framing.
- Codex encodes exit status as plaintext `Process exited with code N`.

## Full-run result (8192-token budget)
```
538,804 raw turns
  - quality funnel dropped 436,649: duplicate 368,087 · ceremony 55,763 · trivial 12,799
  = 102,155 rows kept
tiers:  gold 79,099 · silver 8,748 · bronze 14,308
labels: accept 98,577 · borderline 3,509 (-> judge) · reject 69
secrets redacted: 98,165   (0 residual on grep)
runtime: ~6 min
```

The headline: **dedup did most of the work** — the repeated session-start ritual
and boilerplate across 1,740 subagent files collapsed 368k duplicate turns. The
"insane" 450k raw became ~102k deduped, ~77% gold.

## Lessons (calibration for future runs)
1. **Per-turn dump is too noisy** — every assistant turn includes acks,
   thinking-only turns, and orchestration ceremony. The funnel (dedup + ceremony
   + filler) is essential, not optional.
2. **Outcome > heuristics.** Judging the whole session by tests-passed/built/
   committed/PR'd is the strongest quality signal. ~77% of funnel-survivors were
   from verifiably-good sessions.
3. **Subagents must inherit the parent outcome** — their own transcript never
   contains the commit/PR/test. Without inheritance they're ~all bronze; with
   it, ~98% gold.
4. **Beware noisy survey heuristics.** An early survey reported "16% Codex error
   rate" — it was matching the word "error" in tool *output*, not real failures.
   True failure rate (nonzero exit) was ~3%. Always confirm a detector fires on
   true positives.
5. **Distillation ceiling:** completions are frontier-model output, so
   deterministic negatives are scarce (~3%). KTO negatives come mostly from the
   judge over the borderline pool, not from rules.
6. **Token budget is the compute knob**, not a coverage knob — it bounds context
   depth per row, not row count. 8192 over 32768 (~16× attention cost) with ~0
   coverage loss.

## Downstream
- **SFT**: `label==true` gold rows.
- **KTO**: `label==null` (borderline) rows → LLM-as-judge
  (`SynthChat validate --rubrics quality_labels`) → confirmed accept/reject,
  interleaved with held-out gold accepts.
