# Local dataset store — epistemic humility research

Everything here is committed to the repo (the execution container is ephemeral).
Each dataset folder carries a `dataset.md` with Obsidian-style frontmatter for
querying. Re-fetch provenance is recorded per dataset.

## Local now (fetched 2026-06-09, GitHub sources)

| Folder | Dataset | License | Role |
|---|---|---|---|
| `selfaware/` | SelfAware (3,369 Q) | CC-BY-SA-4.0 (data) | Abstention eval |
| `truthfulqa/` | TruthfulQA (817 Q) | Apache-2.0 | Truthfulness eval |
| `sycophancy-eval/` | Sharma et al. sycophancy-eval | MIT per HF mirror (no LICENSE in repo — verify) | Sycophancy eval |
| `anthropic-sycophancy/` | Perez et al. model-written sycophancy evals | CC-BY-4.0 | Sycophancy eval |
| `sayself/` | SaySelf stage-1 SFT (8,603 ex) + small eval sets | MIT | Reference train data |
| `say-i-dont-know-outputs/` | Cheng et al. per-method TriviaQA test outputs (SFT/DPO/PPO/BoN/HIR) | unstated | **Meta-analysis reanalysis** — lets us independently recompute truthful rates |

## Blocked by network allowlist (need huggingface.co / drive.google.com / arxiv.org)

| Dataset | Source | Why we want it |
|---|---|---|
| TriviaQA | HF `mandarjoshi/trivia_qa` | Substrate for model-specific known/unknown splits (train) |
| Natural Questions | HF `google-research-datasets/natural_questions` | OOD eval substrate |
| MMLU | HF `cais/mmlu` | OOD eval + correctness splits |
| PopQA | HF `akariasai/PopQA` | Long-tail "likely unknown" proxy eval |
| KUQ | HF `amayuelas/KUQ` | Known-unknown train/eval (MIT) |
| CoCoNot | HF `allenai/coconot` | Noncompliance + contrast (over-refusal) sets |
| AbstentionBench | HF `facebook/AbstentionBench` | Holistic abstention eval |
| Idk train datasets | Google Drive via `OpenMOSS/Say-I-Dont-Know` | Model-specific Idk SFT + preference data (we will regenerate for Qwen anyway) |
| R-Tuning data | Google Drive via `shizhediao/R-Tuning` | Refusal-aware train sets |
| LACIE preference pairs | `esteng/pragmatic_calibration` `data.tar.gz` (60 MB, cloned to scratch but not committed) | DPO calibration pairs reference |

Note: for our experiment the known/unknown splits MUST be regenerated against our
own base models (Qwen2.5-3B/7B-Instruct) — published splits are model-specific
by construction. The blocked items above are for evaluation substrates and
cross-checks, not as-is training data.
