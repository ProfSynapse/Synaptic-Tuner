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

**SWE-gym** (2024-2025):
- Training environment companion to SWE-bench
- 2,438 real GitHub issues with executable test environments
- Docker-based: each task gets a pre-built container with the repo at the right commit
- Pre-built validation: `run_tests.sh` per task that executes the relevant test suite
- Used to train open-weight models that approach frontier performance on SWE-bench

**R2E (Repo2Env)** (2024):
- Automated pipeline to convert GitHub repos into executable environments
- Handles dependency resolution, build systems, test framework detection
- Designed specifically for creating RL training environments from repos

**CommitPack / StarCoder2** (BigCode):
- 4TB of permissively-licensed git commits across 350+ languages
- (commit message, diff) pairs used for code instruction tuning
- Filtered version "CommitPackFT" for higher-quality instruction-like commits

**Key insight**: The community has validated that git history is a rich source of training signal. The harder problem is environment reproducibility, not data extraction.

### 2.2 GRPO / RL for Code

**DeepSeek-Coder-V2 & R1** demonstrated GRPO works for code:
- Used GRPO with code execution feedback (test pass/fail) as reward
- Achieved SOTA on code benchmarks with relatively modest RL training
- Key: binary reward (tests pass or not) is sufficient — no need for fine-grained diff matching

**CodeRL** (Le et al., 2022):
- RL from unit test feedback for code generation
- Reward = fraction of tests passed (partial credit)
- Showed RL substantially improves over SFT-only baselines

**RLTF (Reinforcement Learning from Test Feedback)** (2023):
- Multi-granularity reward: line-level, function-level, test-level
- Fine-grained credit assignment via test error localization

**StepCoder** (2024):
- Curriculum RL for code generation
- Starts with easier tasks, progressively harder
- Uses Code Compiler feedback as reward signal

**SWE-gym's RL Results** (2024-2025):
- Applied GRPO directly to SWE-bench-style tasks
- Models trained with RL on SWE-gym significantly outperform SFT-only
- **Open-weight models achieved ~30%+ on SWE-bench Verified** using this approach
- Process: SFT on successful trajectories → GRPO with test-pass reward

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

**Training on Transcripts** (the approach you're describing):
- **Emerging practice**: Companies are starting to use coding agent transcripts for fine-tuning
- Claude Code / Codex / Cursor transcripts contain rich multi-turn tool-use trajectories
- These are essentially expert demonstrations (SFT data) for agentic coding
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
| 1-3B | Good for RL training | Fast rollouts, many iterations. Limited ceiling. |
| 7-8B | Sweet spot | SWE-gym showed strong results at this scale. Fits RTX 3090 with LoRA. |
| 14B | Possible with LoRA | Slower rollouts but higher ceiling. |
| 32B+ | SFT only (locally) | GRPO too expensive without multi-GPU or cloud. |

**Recommendation**: Start with 7-8B (Qwen2.5-Coder-7B or similar). This repo's existing GRPO config targets similar-sized models.

### 5.2 Dataset Size

| Phase | Examples Needed | Source |
|-------|----------------|--------|
| SFT warm-start | 1,000-5,000 trajectories | Transcripts + curated commits |
| GRPO | 500-2,000 tasks with tests | Filtered git commits |
| KTO refinement | 1,000-3,000 pairs | Generated from GRPO attempts (pass/fail) |

**SWE-gym** used ~2,400 tasks and achieved strong results. You don't need millions of examples — hundreds of high-quality tasks with good test coverage outperform thousands of noisy ones.

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

## 6. Quick-Win: Transcript-First Approach

The fastest path to value:

1. **Collect transcripts**: You already have Claude Code / Codex transcripts.
2. **Parse into ChatML**: Convert tool-use sequences into conversation format.
3. **SFT**: Train on successful transcripts (tool-use patterns, code reasoning).
4. **Evaluate**: Run on SWE-bench Lite or your own tasks.
5. **Then GRPO**: Only add RL if SFT plateaus.

This skips the hardest part (environment sandboxing) and gets you a coding-capable model faster. The GRPO phase can be added incrementally.

---

## 7. What Needs to Be Built

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

## 8. Open Questions

1. **Transcript format**: What exact format are your Claude Code / Codex transcripts in? This determines the parser complexity.
2. **Target repos**: Which repos to mine for git commits? Your own projects, or popular OSS repos (like SWE-bench)?
3. **Language scope**: Python-only initially, or multi-language from the start?
4. **Test infrastructure**: Do target repos have good test coverage? (Commits without tests aren't useful for GRPO)
5. **Model base**: Start from a code-specific base (Qwen2.5-Coder) or general-purpose?

---

## 9. References

- **SWE-bench**: [swe-bench.github.io](https://swe-bench.github.io/) — Benchmark for real-world software engineering
- **SWE-gym**: Training environment for SWE-bench-style RL
- **SWE-agent**: [github.com/princeton-nlp/SWE-agent](https://github.com/princeton-nlp/SWE-agent) — Agent interface for code tasks
- **CommitPack**: Large-scale git commit dataset from BigCode
- **CodeRL**: RL from unit test feedback (ICML 2022)
- **RLTF**: Multi-granularity RL from test feedback (2023)
- **StepCoder**: Curriculum RL for code (2024)
- **DeepSeek-R1**: GRPO applied to code reasoning
- **OpenHands**: [github.com/All-Hands-AI/OpenHands](https://github.com/All-Hands-AI/OpenHands) — Coding agent framework
- **R2E**: Repo-to-Environment conversion pipeline
