# Code Review: Karpathy-Inspired Training Optimizations

> Reviewed: 2026-03-21 | Reviewers: architect, test-engineer, backend-coder
> Scope: 36 files, +5,980 lines across 6 implementations + integration tests

## Overall Assessment

The implementation is well-structured with clean protocol-based abstractions, good separation of concerns, and graceful degradation throughout. The test suite has good breadth (178 tests) with thoughtful edge-case coverage. However, **two critical integration bugs** would cause runtime crashes when components are wired together, and the surgery CLI handler is a no-op.

## CRITICAL Issues (3 — all reviewers agree)

### C1: Duplicate EvalBackend protocols with incompatible signatures
**Files**: `shared/eval_backend.py:28-31` vs `shared/evolutionary/lora_surgery.py:57-62`
**Agreement**: All 3 reviewers flagged this independently.

The canonical `EvalBackend` protocol defines `run_eval() -> EvalResult`, while `lora_surgery.py` defines its own `EvalBackend` with `evaluate() -> float`. A `LocalEvalBackend` passed to `LoRASurgeon` would crash with `AttributeError`.

**Fix**: Delete the duplicate protocol in `lora_surgery.py`. Import from `shared/eval_backend.py`. Add a thin wrapper to extract the float score from `EvalResult`.

### C2: CheckpointEvaluator constructor mismatch in experiment_loop.py
**Files**: `shared/checkpoint_eval.py:60` vs `shared/flywheel/experiment_loop.py:679`
**Agreement**: Architect + backend-coder.

`experiment_loop.py` calls `CheckpointEvaluator()` with no args and then `evaluator.evaluate(...)`, but the real constructor requires `(run_dir, eval_backend, eval_scenario)` and the method is `evaluate_checkpoints()`. Would crash at runtime.

**Fix**: Update the call site to match the real constructor signature, or add a convenience method.

### C3: Surgery work directory never cleaned up
**File**: `shared/evolutionary/lora_surgery.py:344`
**Agreement**: Backend-coder.

Adapter copies accumulate across all operations (potentially hundreds of copies for layer_scaling alone). With large adapters, this exhausts disk space.

**Fix**: Add cleanup after each operation, or use `tempfile.TemporaryDirectory` with context manager.

## IMPORTANT Issues (8)

| # | Issue | File(s) | Reviewer |
|---|-------|---------|----------|
| I1 | `SurgeryHandler` validates but never runs surgery (no-op CLI) | `surgery_handler.py:121-136` | Architect, Backend |
| I2 | `fitness_reward` creates new `FitnessEvaluator` per call (wasteful) | `rewards.py:567-575` | Backend, Architect |
| I3 | Conditional assertions in surgery tests silently pass | `test_lora_surgery.py:381,431,477,561` | Test Engineer |
| I4 | Deprecated `asyncio.get_event_loop().run_until_complete()` in surgery tests | `test_lora_surgery.py` (20 occurrences) | Test Engineer |
| I5 | No timeout on cloud eval — stuck jobs block forever | `eval_backend.py:126-131` | Backend |
| I6 | Hard-coded 1-hour subprocess timeout kills thorough-tier experiments | `experiment_loop.py:614` | Architect |
| I7 | `checkpoint_eval.py:184` reads entire log file into memory | `checkpoint_eval.py:184-186` | Backend |
| I8 | DARE division by zero when `drop_rate >= 1.0` | `lora_surgery.py:711` | Backend |

## SUGGESTIONS (12)

| # | Suggestion | Reviewer |
|---|-----------|----------|
| S1 | Split `lora_surgery.py` (1035 lines) into config/operations/helpers/surgeon | Backend |
| S2 | Expose `SurrogateModel.is_fitted` property instead of accessing `_pipeline` | Architect, Backend |
| S3 | `_flatten_config` only goes one level deep — document or fix | Backend, Architect |
| S4 | `_extract_yaml` regex matches non-YAML fenced blocks | Backend |
| S5 | `LocalEvalBackend` uses `"python"` not `sys.executable` | Backend |
| S6 | Add `__all__` exports to new modules | Architect |
| S7 | Tier YAMLs lack `weight_decay` and `max_grad_norm` | Architect |
| S8 | `ExperimentConfig.validate()` doesn't check `eval_backend` values | Architect |
| S9 | Factor surgery operation boilerplate into template method | Backend |
| S10 | Integration tests verify wiring, not actual data flow | Test Engineer |
| S11 | No tests for `_extract_training_loss` or `_evaluate_experiment` | Test Engineer |
| S12 | `sys.path` manipulation in `test_fitness_reward.py` instead of conftest | Test Engineer |

## Test Coverage Gaps (from Test Engineer)

**High priority missing tests:**
1. `_extract_training_loss()` — missing log, fallback glob, malformed JSON
2. `_evaluate_experiment()` — ImportError fallback, loss-inversion
3. `ExperimentLoop.run()` end-to-end with mocked subprocess
4. `CheckpointEvaluator` when ALL evaluations fail
5. Fix conditional assertions in surgery tests

## Strengths (all reviewers agree)

- Clean protocol-based abstractions (`EvalBackend`, `CloudProvider`)
- Graceful degradation everywhere (LLM advisor, surrogate, surgery deps)
- Consistent config patterns (dataclass + from_dict/from_yaml + validate)
- Good logging discipline throughout
- Defensive weight handling in surgery operations
- Phased search strategy (random → LLM → LLM+surrogate) is well-designed
- Operation registry pattern in LoRASurgeon for easy extension

## Recommended Fix Priority

1. **C1 + C2**: Unify EvalBackend protocol + fix CheckpointEvaluator integration (blocks real usage)
2. **C3**: Add work directory cleanup (disk space risk)
3. **I1**: Complete SurgeryHandler (CLI is a no-op)
4. **I2**: Cache FitnessEvaluator in closure (perf in GRPO training)
5. **I3 + I4**: Fix surgery test quality (silent passes + deprecated async)
6. Everything else as time allows
