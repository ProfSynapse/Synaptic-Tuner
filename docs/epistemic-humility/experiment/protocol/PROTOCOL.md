# Experiment Protocol (pre-registration draft v0.1)

**Working title:** *Loss-Averse Humility: Kahneman-Tversky Optimization for
Teaching Small Language Models to Say "I Don't Know"*

**Status:** DRAFT — awaiting user sign-off on hypothesis selection (H1 vs
adding H2/H3 arms) and compute budget. Derived from gaps verified in
`../../meta-analysis/evidence/raw-reports/05-sft-vs-preference-and-gaps.md`.

---

## 1. Motivation (one paragraph)

The literature establishes: (a) SFT-based IDK-training works but over-refuses
(our exact reanalysis: 42.7% over-refusal for Idk-SFT on Llama-2-7b-chat);
(b) preference optimization (DPO/PPO) halves over-refusal but needs paired
preference data and damages token-level calibration in other settings;
(c) KTO — which takes exactly the unpaired binary (desirable/undesirable)
data that IDK splits naturally produce, and whose asymmetric loss-aversion
weighting mirrors the asymmetric real-world cost of hallucination vs
abstention — has **never been applied to abstention/calibration training**
(verified gap, 2026-06). KTO also requires no reward model and no pair
construction, making it the cheapest preference-class method to deploy.

## 2. Hypotheses

- **H1 (primary):** KTO on model-specific IDK data achieves a truthful rate
  (Ik-Ik + Ik-Idk) >= Idk-SFT while reducing over-refusal by >= 25% relative,
  matching or approaching DPO's over-refusal advantage *without preference
  pairs*.
- **H2 (transfer):** KTO-trained abstention transfers out-of-domain at least
  as well as SFT-trained abstention (no prior OOD evidence exists for ANY
  preference-trained abstention).
- **H3 (loss-aversion dose-response, stretch goal):** KTO's
  lambda_D/lambda_U asymmetry knob gives monotonic control over the
  abstention/over-refusal trade-off — a capability no other method offers
  natively. (3-point sweep; only if budget allows.)
- **Falsifiers:** H1 fails if KTO truthful rate < SFT or over-refusal
  reduction < 10% relative. H2 fails if OOD truthful-rate drop for KTO
  exceeds SFT's drop by > 5 pp.

## 3. Design

### 3.1 Factors

| Factor | Levels |
|---|---|
| Training method | base (no FT), Idk-SFT, Idk-KTO; (optional 4th arm: Idk-DPO for a 3-way) |
| Model | Qwen2.5-3B-Instruct (pilot), Qwen2.5-7B-Instruct (confirm) |
| Eval domain | in-domain (TriviaQA held-out), OOD-near (PopQA), OOD-far (MMLU subsets + SelfAware) |

Seeds: 2 per arm at 3B (variance estimate — addressing the field-wide
no-error-bars problem is part of the paper's contribution), 1 at 7B.

### 3.2 Data construction (the model-specific part)

1. Probe base model on TriviaQA (rc.nocontext) train subset (~20k questions):
   10 samples @ T=1.0 + greedy. P_correct per question (SliCK-style).
2. Split: known (greedy correct & P_correct >= 0.5) / unknown (P_correct = 0)
   / discard ambiguous middle band (cleaner contrast; sensitivity check keeps it).
3. Targets: known -> gold short answer in a fixed response template;
   unknown -> abstention template ("I don't know the answer to that." +
   brief honesty rationale, style-varied via SynthChat to avoid template
   overfitting — hybrid data decision).
4. **SFT set:** all examples as positives (R-Tuning/Cheng style).
   **KTO set (same questions, zero extra labeling):**
   - desirable: known+correct-answer, unknown+abstention
   - undesirable: unknown+model's own hallucinated answer (from the probe
     samples!), known+abstention (anti-over-refusal signal — the novel bit)
   - interleaved per `KTO_TRAINING_REFERENCE.md`; ~10-15k examples each arm.
5. Eval sets are never touched during training; OOD sets share no questions.

### 3.3 Training (Synaptic Tuner native)

- `Trainers/sft/train_sft.py --qwen-3b` and `Trainers/rtx3090_kto/train_kto.py
  --qwen-3b --local-file <kto.jsonl>`; LoRA r=32/alpha=64 (3B preset),
  r=64/128 at 7B; identical LoRA budget across arms (confound control).
- Early stopping on dev loss (Gekhman: prolonged training on unknowns is the
  hallucination driver — we log the dynamics as a secondary analysis).
- Local RTX 3090 or HF Jobs (`tuner.py cloud-pipeline`), per
  `.skills/fine-tuning/SKILL.md`.

### 3.4 Metrics (joint reporting — fixing the field's incommensurability)

1. Truthful rate + 4-quadrant matrix (Cheng) — primary.
2. Refusal recall on unknowns + over-refusal on knowns (our reanalysis metrics).
3. AP over confidence-ranked answers (R-Tuning comparability).
4. Token-level ECE on MMLU multiple-choice (does KTO damage calibration the
   way RLHF/DPO do? — the unreconciled tension, measured for the first time
   on the same run).
5. TruthfulQA MC1/MC2 + over-refusal on its informativeness axis.
6. Accuracy retention on answered questions (capability tax).

Analysis: paired bootstrap CIs over eval questions; McNemar tests between
arms on the same questions; all scripts + outputs committed.

### 3.5 Power / scale sanity

TriviaQA-Idk-style test split ~11k questions: detecting a 3 pp truthful-rate
difference at alpha=.05 has power > 0.95 (two-proportion, n~11k); the binding
constraint is seed variance, hence 2 seeds at 3B.

## 4. Deliverable

arXiv paper #2: intro motivated by meta-analysis findings; methods as above;
the dose-response knob (H3) as the headline if it pans out. All data
construction + analysis scripts in this repo; datasets + adapters to HF Hub.

## 5. Blockers / needs

- TriviaQA + PopQA + MMLU downloads (network allowlist — HANDOFF.md §1).
- Inference backend for the knowledge probe (RTX 3090 local or HF endpoint).
- User sign-off: arm count (2 vs 3 training methods), H3 in/out, budget.
