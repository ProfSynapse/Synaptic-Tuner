# Epistemic Humility in LLM Training & Fine-Tuning — Research Program

Two linked arXiv-bound deliverables, both developed on branch
`claude/ai-humility-research-experiment-37za2i`:

1. **Meta-analysis** (`meta-analysis/`) — PRISMA-style systematic search +
   quantitative synthesis of the literature on epistemic humility as it
   relates to LLM training and fine-tuning: calibration (ECE/Brier/AUROC),
   abstention/IDK fine-tuning, fine-tuning-induced hallucination,
   sycophancy under preference training, and knowledge-boundary
   self-awareness. Formal pooling is run only where variance data exists;
   elsewhere we use normalized descriptive synthesis and moderator
   regression (documented honestly in methods).

2. **Experiment** (`experiment/`) — a novel fine-tuning experiment run in
   Synaptic Tuner, with hypothesis derived from gaps the meta-analysis
   exposes. Design: 3B pilot (Qwen2.5-3B-Instruct) → 7B confirmation
   (Qwen2.5-7B-Instruct), hybrid data (open QA datasets keyed to
   model-specific known/unknown splits + SynthChat-generated response
   styles where needed), SFT and KTO arms via the repo training pipeline.

## Layout

```
docs/epistemic-humility/
├── README.md                  # this file
├── meta-analysis/
│   ├── evidence/              # extracted-numbers tables (CSV) + source log
│   ├── analysis/              # stats scripts + generated figures/tables
│   └── paper/                 # meta-analysis draft (markdown -> LaTeX)
└── experiment/
    ├── protocol/              # preregistered hypothesis, design, power notes
    ├── data/                  # dataset construction scripts + manifests
    └── paper/                 # experiment paper draft
```

## Provenance rules

- Every number in `evidence/*.csv` must carry: paper title, arXiv ID, the
  exact metric definition, eval dataset, model + size, training method,
  and the URL it was extracted from. Numbers that could not be verified
  against the primary source are flagged `verified=false` and excluded
  from pooled statistics.
- Analysis scripts are deterministic and re-runnable: `evidence/*.csv` in,
  figures/tables out. No hand-edited results.
