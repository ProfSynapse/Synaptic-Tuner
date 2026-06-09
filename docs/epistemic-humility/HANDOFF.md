# HANDOFF — Epistemic Humility Research Program

**Branch:** `claude/ai-humility-research-experiment-37za2i` · **Updated:** 2026-06-09
**Goal:** two arXiv submissions — (1) a meta-analysis of epistemic humility in LLM
training/fine-tuning, (2) a novel SFT-vs-KTO abstention experiment run in Synaptic Tuner.

This doc is the single re-entry point. Read it on your computer, do the
"UNBLOCK" section, then tell the next session to "read docs/epistemic-humility/HANDOFF.md
and continue."

---

## 1. UNBLOCK — settings only you can change (5 minutes, from a computer)

The cloud session's egress proxy is deny-by-default. On **claude.ai/code →
(this repo's) Environment → Network policy**, add these domains:

```
arxiv.org
export.arxiv.org
ar5iv.labs.arxiv.org
huggingface.co
cdn-lfs.huggingface.co
datasets-server.huggingface.co
```
Optional but useful: `aclanthology.org`, `openreview.net`, `cdn.openai.com`,
`drive.google.com`, `drive.usercontent.google.com` (R-Tuning + Idk train data).

Then start a fresh session on this branch and run:

```bash
# pulls all 61 paper PDFs + abstracts + ar5iv full text into the library
python3 docs/epistemic-humility/library/scripts/fetch_library.py --enrich
```

## 2. What exists already (all committed on this branch)

| Asset | Path | State |
|---|---|---|
| Paper library: manifest (61 papers) + frontmatter notes + fetch script | `library/` | notes stubbed; PDFs pending network |
| 5 deep-research evidence reports (every extracted number + provenance) | `meta-analysis/evidence/raw-reports/` | done; numbers flagged unverified-against-PDF |
| Independent reanalysis of Cheng et al. (2401.13275) method outputs | `meta-analysis/analysis/reanalyze_idk_outputs.py` → `evidence/idk-method-reanalysis.csv` | done (exact refusal metrics) |
| Local datasets (SelfAware, TruthfulQA, 2× sycophancy evals, SaySelf, Cheng outputs) | `datasets/` | done, with licenses + frontmatter notes |
| Normalized evidence table + synthesis stats | `meta-analysis/evidence/effects.csv`, `analysis/` | see git log — being built |
| Experiment protocol (hypotheses, design, metrics, recipes) | `experiment/protocol/` | see git log — being built |

## 3. Headline findings driving the experiment

1. **KTO has never been applied to abstention/IDK/calibration training** (verified
   gap, high confidence, as of 2026-06). Closest prior art: Cheng et al. ICML 2024
   compares Idk-SFT/DPO/PPO/BoN/HIR on Llama-2-7b-chat (no KTO, no 3B, thin OOD).
2. Our reanalysis of their raw outputs (exact, n=11,313): Idk-SFT over-refuses
   42.7% of known questions; DPO halves that to 23.3% (refusal recall 84.1% vs 71.2%).
3. RLHF/preference training degrades token-level calibration ~10× (GPT-4 ECE
   0.007→0.074) while *improving* abstention/factuality — unreconciled tension;
   no paper measures both after the same run. KTO unmeasured on either.
4. SFT on model-unknown facts causally drives hallucination, linearly in fitted
   unknowns (Gekhman 2405.05904) → training data must be split by THIS model's
   knowledge (known/unknown splits are model-specific by construction).

**Planned experiment (pending your sign-off on the formal hypothesis doc):**
SFT vs KTO on model-specific IDK data for Qwen2.5-3B-Instruct (pilot) →
Qwen2.5-7B-Instruct (confirm), measuring truthful rate + over-refusal + ECE +
OOD transfer — directly filling gaps 1, 2, 5, 6 from
`evidence/raw-reports/05-sft-vs-preference-and-gaps.md`.

## 4. Work queue after unblocking (in order)

1. **Enrich library** (command above); spot-verify every number starred/flagged
   `verify` in raw-reports against the PDFs; update note statuses
   `candidate→fetched→verified`; correct `effects.csv` where snippets were wrong.
   Priority verifications: GPT-4 ECE figure-8 values; Sharma 98%/~6% figures;
   AbstentionBench 24%; Cheng per-method truthful rates; R-Tuning AP tables;
   Wei et al. sycophancy percentages.
2. **Download blocked datasets** to `datasets/` (each gets a `dataset.md` with
   frontmatter like the existing ones): TriviaQA (`mandarjoshi/trivia_qa`,
   rc.nocontext), MMLU (`cais/mmlu`), PopQA (`akariasai/PopQA`), KUQ
   (`amayuelas/KUQ`), CoCoNot (`allenai/coconot`), AbstentionBench
   (`facebook/AbstentionBench`); TriviaQA gold aliases unlock the exact
   truthful-rate recomputation in `reanalyze_idk_outputs.py` (replace the
   INTERIM token-F1 proxy).
3. **Finish meta-analysis paper** from `meta-analysis/paper/` skeleton +
   verified `effects.csv` (PRISMA-style search section is documented in the
   raw reports' frontmatter: queries per agent, date, method).
4. **Experiment execution** (local GPU or HF Jobs; see
   `experiment/protocol/`): generate known/unknown splits for Qwen2.5-3B by
   10-sample correctness probing (needs an inference backend — RTX 3090 via
   `tuner.py local-run`, or cloud), build SFT + KTO JSONL per
   `.skills/fine-tuning/reference/dataset-formats.md`, train both arms,
   evaluate with the Evaluator harness, analyze, draft paper 2.

## 5. Constraints discovered (so you don't re-hit them)

- `docs/research/` is **gitignored** — that's why this lives in `docs/epistemic-humility/`.
- Egress proxy blocks everything except github.com / raw.githubusercontent.com /
  pypi.org; WebSearch works (snippets only). WebFetch obeys the same allowlist.
- HF Jobs / training does not run in this container — training happens on your
  RTX 3090 or HF cloud from your machine.
- Datasets in repo so far: ~116 MB. Large downloads (TriviaQA full is GBs) should
  be subset before committing (rc.nocontext validation slice is enough).
- The PACT plugin/skills referenced in CLAUDE.md are not installed in cloud
  sessions; sessions run directly.

## 6. Provenance rules (non-negotiable for arXiv)

Every number in `effects.csv` carries source + URL + `verified` flag. Numbers
failing PDF verification get corrected or dropped from pooled stats — the raw
reports keep the audit trail of what changed.
