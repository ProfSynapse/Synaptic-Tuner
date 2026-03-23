# Git-Based Code Fine-Tuning: Research & Feasibility

**Date**: 2026-03-23
**Status**: Research / Exploration
**Context**: Can we train coding ability using git commits as ground-truth, GRPO for RL, and Claude Code/Codex transcripts as trajectory data?

---

## 1. The Core Idea

Train a model to be a better coding agent by:

1. **Extracting training signal from git history**: Each commit represents a (task → solution) pair. The commit message (or linked issue/PR) is the user request; the diff is the ground-truth solution.
2. **Creating sandboxed environments**: Snapshot the repo at the pre-commit state, give the model the task, let it attempt the solution using tools (bash, file edit, grep, etc.).
3. **GRPO reward**: Score the model's attempt based on whether it achieves the committed outcome (tests pass, diff matches, etc.).
4. **Transcript bootstrapping**: Use existing Claude Code / Codex / Cursor transcripts as SFT warm-start data before GRPO.

---

## 2. State of the Art

### 2.1 Git-to-Training-Data Pipelines

**SWE-bench** (Princeton, 2023-present) is the canonical benchmark:
- Extracts (issue description, PR patch) pairs from real GitHub repos
- 2,294 tasks from 12 popular Python repos (Django, Flask, scikit-learn, etc.)
- Each task: issue text + repo snapshot at the merge base → model must produce the correct patch
- **SWE-bench Verified**: Human-validated subset (500 tasks) with confirmed test coverage

**SWE-smith** (NeurIPS 2025 Spotlight) — most mature pipeline:
- Given any Python codebase, constructs Docker-based execution environments
- Synthesizes task instances using 4 strategies: LM-Modify (inject bugs), LM-Rewrite (reimplement functions), Procedural AST mutations (13 operators), PR Mirrors (real diffs)
- **50,137 task instances from 128 GitHub repos** — far richer than raw (commit, diff) pairs
- Each instance: broken code snapshot + NL issue description + execution env + tests
- [GitHub](https://github.com/SWE-bench/SWE-smith) | [HF Dataset](https://huggingface.co/datasets/SWE-bench/SWE-smith)

**R2E-Gym** (COLM 2025) — complementary approach:
- **SWE-GEN** pipeline: generates executable training environments *without* human-written issues or tests
- Uses automated test generation + "back-translation" (generating issue descriptions from diffs)
- 8,100+ tasks across 13 repos
- **DeepSWE found R2E-Gym's curated 4.5K tasks outperformed SWE-smith's 50K for GRPO** — quality > quantity
- [GitHub](https://github.com/R2E-Gym/R2E-Gym)

**SWE-rebench** (NeurIPS 2025):
- Continuously-updating pipeline: 21,000+ interactive SWE tasks from diverse GitHub repos
- Designed specifically for RL training
- Nebius released [67,074 agent trajectories](https://nebius.com/blog/posts/openhands-trajectories-with-qwen3-coder-480b) on SWE-rebench

**SWE-gym** (2024-2025):
- Training environment companion to SWE-bench
- 2,438 real GitHub issues with executable test environments
- Docker-based: each task gets a pre-built container with the repo at the right commit

**CommitPack / StarCoder2** (BigCode):
- 4TB of permissively-licensed git commits across 350+ languages
- (commit message, diff) pairs used for code instruction tuning
- Filtered version "CommitPackFT" for higher-quality instruction-like commits

**Key insight**: Raw (commit message, diff) pairs are considered too noisy. The field has moved toward generating *executable environments* where the model's solution can be tested, not just compared textually to a ground-truth diff.

### 2.2 GRPO / RL for Code

**DeepSWE** (Together AI + Agentica, 2025) — most impressive open result:
- Starting from Qwen3-32B, using *only* GRPO on R2E-Gym environments
- **42.2% Pass@1 on SWE-bench Verified** (59% with test-time scaling at Pass@16)
- Just **200 RL steps** boosted score from 23% → 42%
- Key finding: R2E-Gym worked best for RL — SWE-smith and SWE-Gym showed limited improvements
- [Blog](https://www.together.ai/blog/deepswe)

**MicroCoder-GRPO** (ICLR 2026) — GRPO optimizations for code:
- Conditional truncation masking, diversity-determined temperature, removal of KL loss
- 17.6% relative improvement on LiveCodeBench v6
- 13,300 curated competitive programming problems → 3x larger gains than mainstream datasets in 300 steps

**Self-Play SWE-RL (SSR)** — no human labels needed:
- Single LLM agent trained via RL in self-play to iteratively inject and repair bugs of increasing complexity
- +10.4 points on SWE-bench Verified with zero human-written issues or tests

**DeepSeek-Coder-V2 & R1**:
- Used GRPO with code execution feedback (test pass/fail) as reward
- Binary reward sufficient — no need for fine-grained diff matching

**CodeRL** (ICML 2022) / **RLTF** (2023) / **StepCoder** (2024):
- Progression from outcome-level → multi-granularity → curriculum RL for code
- Key finding: partial credit via test fraction or step-level rewards helps training stability

**Reward signals ranked by effectiveness**:

| Signal | Type | Notes |
|--------|------|-------|
| Unit test pass/fail | Binary execution | Gold standard (DeepSWE, R2E-Gym) |
| Process Reward Models | Step-level | CodePRM, DreamPRM-Code — score individual reasoning steps |
| Execution-free verifiers | Model-based | SWE-RM: 62% SWE-bench Verified *without* running tests |
| AST edit distance | Structural similarity | Partial credit for structurally close solutions |
| Hybrid (exec + model) | Combined | R2E-Gym: combining both significantly outperforms either alone |

### 2.3 Agentic / Tool-Use Code Training

**OpenHands (formerly OpenDevin)** (2024-2025):
- Full coding agent framework with bash, file edit, browser tools
- Training data: collect trajectories of successful task completions
- CodeAct paradigm: model outputs executable code actions, not just text

**SWE-agent** (Princeton, 2024):
- Agent interface for SWE-bench with custom tools (edit, search, scroll, etc.)
- Tool design matters enormously — good tools boost performance 2-3x
- Key insight: the *tool interface* is as important as the model

**Agentless** (UIUC, 2024):
- Shows you don't always need full agent loops — localize then patch
- Two-phase: fault localization → patch generation
- Competitive with agent-based approaches at lower cost

**OpenHands** (ICLR 2025) — leading open platform:
- CodeAct architecture: actions are bash commands and file edits
- Nebius released 67K trajectories; RFT checkpoints: 50.3% (30B), 61.7% (235B) on SWE-bench Verified
- Pipeline: SFT on expert trajectories → rejection fine-tuning → GRPO RL

**Harness coupling insight**: Frontier models are post-trained on their specific tool harnesses. OpenAI's Codex models are tightly coupled with `apply_patch`. This suggests the tool API design matters as much as the training data.

**Training on Transcripts** (the approach you're describing):
- **Emerging practice**: Companies are starting to use coding agent transcripts for fine-tuning
- Claude Code / Codex / Cursor transcripts contain rich multi-turn tool-use trajectories
- These are essentially expert demonstrations (SFT data) for agentic coding
- **Practical path**: Generate your own trajectories using a strong model on SWE-smith/R2E-Gym tasks, then distill
- **Key challenge**: transcripts include tool outputs (file contents, bash results) that are environment-specific — you need to either strip these or make them reproducible

### 2.4 Environment Sandboxing at Scale

**Docker-based** (dominant approach):
- SWE-bench uses per-task Docker images with pre-installed deps
- SWE-gym pre-builds ~2,400 Docker images (one per task)
- Build once, run many: amortize container build cost

**E2B (Code Interpreter SDK)**:
- Cloud sandboxes via API (what this repo already integrates)
- Fast spin-up (~200ms), but cost per execution
- Good for inference/eval, expensive for RL training (many rollouts per example)

**Modal / RunPod** (this repo already supports):
- GPU-attached containers for training
- Can also serve as execution environments for code validation

**Scaling Concern**: GRPO needs multiple rollouts per example. If `num_generations=4` and you have 1,000 tasks, that's 4,000 environment executions per epoch. Docker locally is feasible; cloud sandboxes get expensive.

---

## 3. What This Repo Already Has

The existing infrastructure is **remarkably well-positioned** for this:

| Capability | Status | Location |
|-----------|--------|----------|
| GRPO trainer | **Complete** | `Trainers/grpo/train_grpo.py` |
| Environment-backed GRPO | **Complete** | `Trainers/grpo/train_env_grpo.py` |
| Sandbox runtime (local) | **Complete** | `shared/environments/local_runtime.py` |
| Sandbox runtime (E2B) | **Complete** | `shared/environments/e2b_runtime.py` |
| Tool execution framework | **Complete** | `shared/environments/tool_executor.py` |
| Multi-step rollout | **Complete** | `Trainers/grpo/src/env_rollout.py` |
| Environment reward function | **Complete** | `Trainers/grpo/src/env_rewards.py` |
| Cloud training (HF Jobs) | **Complete** | `Trainers/grpo/src/env_runtime.py` |
| Flywheel auto-retrain loop | **Complete** | `shared/flywheel/` |
| SFT/KTO baselines | **Complete** | `Trainers/sft/`, `Trainers/kto/` |
| Git repo → training data | **Missing** | Needs new pipeline |
| Code execution tools (bash, python) | **Missing** | Current tools are file-management only |
| Transcript ingestion | **Missing** | Needs parser for Claude Code/Codex format |
| Test-pass reward signal | **Missing** | Current rewards are assertion-based, not test-runner |

**The gap is not infrastructure — it's the data pipeline and code-specific tools.**

---

## 4. Proposed Architecture

### 4.1 Data Pipeline: Git → Training Examples

```
Git Repository
    │
    ▼
┌─────────────────────┐
│  Commit Extractor    │  For each commit:
│  ─────────────────   │  - Parse commit message / linked issue / PR body
│  git log --format    │  - Extract pre-commit snapshot (git checkout HEAD~1)
│  git diff HEAD~1     │  - Extract post-commit diff
│  git stash           │  - Identify affected test files
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Environment Builder │  For each (task, snapshot) pair:
│  ─────────────────   │  - Create fixture from repo tree at pre-commit
│  repo → fixture      │  - Identify test commands (pytest, npm test, etc.)
│  test discovery      │  - Build assertions (tests must pass after changes)
│  dep resolution      │  - Record ground-truth diff for reward scoring
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Quality Filter      │  Filter out:
│  ─────────────────   │  - Merge commits, version bumps, dependency updates
│  heuristic + LLM     │  - Commits without meaningful test coverage
│  judge filter        │  - Commits with unclear/empty messages
│                      │  - Too-large diffs (>500 lines initially)
└─────────────────────┘
    │
    ▼
  Training Dataset (JSONL)
```

### 4.2 Transcript Pipeline: Claude Code → Training Examples

```
Claude Code Transcript (.jsonl / .json)
    │
    ▼
┌─────────────────────┐
│  Transcript Parser   │  Extract:
│  ─────────────────   │  - User request (initial prompt)
│  Parse tool calls    │  - Tool-use sequence (bash, edit, read, grep, etc.)
│  Extract outcomes    │  - Tool outputs / results
│                      │  - Final outcome (success/failure)
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Replay Validator    │  For each transcript:
│  ─────────────────   │  - Reconstruct repo state at session start
│  Verify replayable   │  - Check if tools + outputs are reproducible
│  Check determinism   │  - Flag non-deterministic steps (API calls, etc.)
└─────────────────────┘
    │
    ▼
  Two output paths:
  ├── SFT Dataset: Full (prompt, tool_calls, response) trajectories
  └── GRPO Dataset: (prompt, environment_fixture, assertions) for RL
```

### 4.3 Extended Environment Runtime

The current `EnvironmentRuntime` supports file operations only. For code training, extend with:

```python
class CodeEnvironmentRuntime(EnvironmentRuntime):
    """Extended runtime with code execution capabilities."""

    def run_command(self, cmd: str, timeout: int = 30) -> CommandResult:
        """Execute a shell command in the sandbox."""

    def run_tests(self, test_cmd: str = "auto") -> TestResult:
        """Run project tests, auto-detecting framework."""

    def apply_patch(self, patch: str) -> PatchResult:
        """Apply a unified diff patch."""

    def git_diff(self) -> str:
        """Get current diff vs. initial state."""

    def check_syntax(self, file_path: str) -> SyntaxResult:
        """Check file for syntax errors."""
```

### 4.4 Reward Design

Multi-signal reward combining:

| Signal | Weight | Source |
|--------|--------|--------|
| **Tests pass** | 0.5 | `run_tests()` — binary, most reliable |
| **Diff similarity** | 0.2 | Compare model's changes to ground-truth diff (AST-aware for code) |
| **Patch applies cleanly** | 0.1 | No syntax errors, no conflicts |
| **Efficiency** | 0.1 | Fewer tool calls = better (step penalty) |
| **Scope accuracy** | 0.1 | Changed the right files (not too many, not too few) |

**Important**: Start with just test-pass as the reward. DeepSeek and SWE-gym showed binary test-pass reward is surprisingly effective for GRPO. Add other signals only if needed.

### 4.5 Training Progression

```
Phase 1: SFT Warm-Start
├── Data: Claude Code transcripts + successful SWE-bench trajectories
├── Format: Multi-turn (user request → tool calls → results → response)
└── Goal: Teach tool-use patterns and code reasoning

Phase 2: GRPO with Test Feedback
├── Data: Git commits with test coverage
├── Environment: Docker sandbox per task (pre-commit snapshot)
├── Reward: Tests pass after model's changes
├── num_generations: 4-8 (more = better signal, higher cost)
└── Goal: Learn to actually solve coding tasks

Phase 3: Flywheel (Optional)
├── Deploy model via proxy
├── Collect real usage data (inference logs)
├── Auto-retrain on successful interactions
└── Goal: Continuous improvement from production use
```

---

## 5. Practical Considerations

### 5.1 Model Size

| Size | Feasibility | Notes |
|------|-------------|-------|
| 1.5-4B | Research / experimentation | MicroCoder-GRPO experiments on Qwen3-1.7B and 4B |
| 7-8B | Fast iteration, single GPU | Deepcoder-14B (60.6% LiveCodeBench). Fits RTX 3090 w/ LoRA. |
| 14B | Strong practical choice | Deepcoder-14B trained with GRPO on rLLM framework |
| 32B | Current sweet spot for SWE | DeepSWE (42.2%), SWE-agent-LM-32B (40.2%), OpenHands RFT (50.3%) |
| 72B+ | Best results, multi-GPU | 39% SWE-bench with Qwen2.5-72B long-context RL |

**Recommendation**: Start with 7-8B (Qwen2.5-Coder-7B) for iteration speed. **Qwen2.5-Coder** is the dominant base model family across all recent SWE-agent work (Apache 2.0, 0.5B-32B). Scale to 32B for production quality.

### 5.2 Dataset Size

| Phase | Examples Needed | Source |
|-------|----------------|--------|
| SFT warm-start | 5,000-50,000 trajectories | Transcripts + curated trajectories (SWE-smith scale) |
| GRPO | 500-4,500 tasks with tests | R2E-Gym style curated tasks |
| KTO refinement | 1,000-3,000 pairs | Generated from GRPO attempts (pass/fail) |

**Key data point**: DeepSWE used only 4,500 R2E-Gym tasks over 200 RL steps to jump from 23% → 42%. GRPO can work with surprisingly few examples when reward signals are clean. **Quality vastly outweighs quantity** — R2E-Gym's 4.5K beat SWE-smith's 50K for RL.

### 5.3 Environment Cost

| Approach | Cost per Rollout | Scaling |
|----------|-----------------|---------|
| Local Docker | ~Free (CPU/time) | ~10-30s per rollout depending on tests |
| E2B | ~$0.01-0.05 | Fast but adds up at GRPO scale |
| Pre-built containers | ~Free after build | Best for repeated tasks (SWE-gym approach) |

**Recommendation**: Pre-build Docker images for your target repos. Run locally for GRPO training. Use E2B/cloud only for inference-time eval.

### 5.4 Reward Signal Reliability

| Signal | Reliability | Notes |
|--------|-------------|-------|
| Tests pass/fail | **High** | Binary, unambiguous. Best primary signal. |
| Exact diff match | **Low** | Many valid solutions per task. Avoid as primary. |
| AST diff similarity | **Medium** | Structural comparison. Good secondary signal. |
| Lint/type check pass | **Medium** | Catches regressions but not correctness. |
| LLM judge | **Medium** | Flexible but noisy. Use for non-testable aspects. |

---

## 6. Practical GRPO Recipe (from DeepSWE / MicroCoder / Swift docs)

```yaml
# Proven hyperparameters for code GRPO
model: Qwen2.5-Coder-7B-Instruct  # or 32B for best results
max_completion_length: 8192
temperature: 1.2               # during generation (higher = more exploration)
per_device_train_batch_size: 1
gradient_accumulation_steps: 64  # effective batch = 64
num_generations: 8              # samples per query
learning_rate: 1e-6
beta: 0.04
reward: binary_test_pass        # 0 or 1
```

**Key training insights from the literature**:
- **MicroCoder**: Remove KL loss + high clipping ratios for code tasks
- **DeepSWE**: 200 RL steps sufficient for massive gains
- **Temperature**: Use 1.0-1.2 during generation for exploration, 0.0 at eval
- **Scaling**: CPU is the bottleneck (running tests), not GPU. Use Kubernetes for parallel sandbox execution.

---

## 7. Quick-Win: Transcript-First Approach

The fastest path to value:

1. **Collect transcripts**: You already have Claude Code / Codex transcripts.
2. **Parse into ChatML**: Convert tool-use sequences into conversation format.
3. **SFT**: Train on successful transcripts (tool-use patterns, code reasoning).
4. **Evaluate**: Run on SWE-bench Lite or your own tasks.
5. **Then GRPO**: Only add RL if SFT plateaus.

This skips the hardest part (environment sandboxing) and gets you a coding-capable model faster. The GRPO phase can be added incrementally.

---

## 8. What Needs to Be Built

### Must-Have (MVP)

1. **Git commit extractor**: Script to extract (message, pre-commit-snapshot, diff, test-cmd) tuples from a repo
2. **Code execution runtime**: Extend `EnvironmentRuntime` with `run_command()` and `run_tests()`
3. **Test-pass reward function**: New reward in `env_rewards.py` that scores based on test execution
4. **Transcript parser**: Convert Claude Code JSONL transcripts into the repo's conversation format

### Nice-to-Have (Phase 2)

5. **Docker image builder**: Auto-build per-task containers from repo snapshots
6. **AST-aware diff reward**: Partial credit for structurally similar solutions
7. **SWE-bench integration**: Import SWE-bench tasks directly into the pipeline
8. **Flywheel integration**: Auto-collect successful coding interactions for retraining

### Already Done (Leverage Existing)

- GRPO trainer + env-backed variant
- Sandbox runtime (local + E2B)
- Tool execution framework
- Multi-step rollout infrastructure
- Reward function framework
- Cloud training (HF Jobs)
- SFT/KTO trainers
- Experiment tracking

---

## 9. Open Questions

1. **Transcript format**: What exact format are your Claude Code / Codex transcripts in? This determines the parser complexity.
2. **Target repos**: Which repos to mine for git commits? Your own projects, or popular OSS repos (like SWE-bench)?
3. **Language scope**: Python-only initially, or multi-language from the start?
4. **Test infrastructure**: Do target repos have good test coverage? (Commits without tests aren't useful for GRPO)
5. **Model base**: Start from a code-specific base (Qwen2.5-Coder) or general-purpose?

---

## 10. References

### Data Pipelines & Benchmarks
- **SWE-bench**: [swe-bench.github.io](https://swe-bench.github.io/)
- **SWE-smith**: [github.com/SWE-bench/SWE-smith](https://github.com/SWE-bench/SWE-smith) — 50K tasks, NeurIPS 2025 Spotlight
- **R2E-Gym**: [github.com/R2E-Gym/R2E-Gym](https://github.com/R2E-Gym/R2E-Gym) — 8.1K tasks, COLM 2025
- **SWE-rebench**: [openreview.net](https://openreview.net/forum?id=nMpJoVmRy1) — 21K tasks, NeurIPS 2025
- **CommitPack**: 4TB git commits from BigCode/StarCoder2

### RL / GRPO for Code
- **DeepSWE**: [together.ai/blog/deepswe](https://www.together.ai/blog/deepswe) — 42.2% SWE-bench Verified with GRPO
- **MicroCoder-GRPO**: [arxiv.org/abs/2603.07777](https://arxiv.org/abs/2603.07777) — ICLR 2026
- **Self-Play SWE-RL**: [arxiv.org/abs/2512.18552](https://arxiv.org/abs/2512.18552) — No human labels needed
- **CodeRL**: RL from unit test feedback (ICML 2022)
- **RLTF**: Multi-granularity RL from test feedback (2023)
- **StepCoder**: Curriculum RL for code (2024)
- **DeepSeek-R1**: GRPO applied to code reasoning

### Agents & Frameworks
- **OpenHands**: [github.com/All-Hands-AI/OpenHands](https://github.com/All-Hands-AI/OpenHands) — ICLR 2025
- **SWE-agent**: [github.com/princeton-nlp/SWE-agent](https://github.com/princeton-nlp/SWE-agent)
- **ScaleBox**: [github.com/icip-cas/ScaleBox](https://github.com/icip-cas/ScaleBox) — Distributed sandbox + RL
- **rLLM**: [github.com/agentica-project/rLLM](https://github.com/agentica-project/rLLM) — Ray-based RL framework (Deepcoder-14B)

### Reward Models
- **SWE-RM**: [arxiv.org/pdf/2512.21919](https://www.arxiv.org/pdf/2512.21919) — Execution-free verifier, 62% SWE-bench
- **CodePRM**: [ACL 2025](https://aclanthology.org/2025.findings-acl.428/) — Step-level process rewards

### Tutorials & Guides
- **GRPO Explainer**: [Cameron Wolfe](https://cameronrwolfe.substack.com/p/grpo)
- **Swift GRPO Code Training**: [swift.readthedocs.io](https://swift.readthedocs.io/en/latest/BestPractices/GRPO-Code-Training.html)
- **TRL GRPOTrainer**: [huggingface.co/docs/trl](https://huggingface.co/docs/trl/en/grpo_trainer)
- **DeepLearning.AI GRPO Course**: [deeplearning.ai](https://www.deeplearning.ai/short-courses/reinforcement-fine-tuning-llms-grpo/)
