# SynthChat Decomposition Plan Validation

**Date**: 2026-04-03
**Validator**: pact-architect (arch-synthchat)
**Plan**: `docs/plans/synthchat-solid-refactoring-plan.md`
**Code state**: branch `synthchat-solid-refactor`, worktree at `.worktrees/synthchat-solid-refactor`

## Verdict: Plan Validated -- Minor Adjustments Documented

The plan is accurate and implementable. Line counts, method listings, external consumers, and dependency order all match current code. The adjustments below are refinements, not blockers.

---

## 1. Line Counts and File Sizes

| File | Plan States | Actual | Status |
|------|-------------|--------|--------|
| `generator.py` | 3,384 lines | 3,384 lines | Match |
| `run.py` | 1,060 lines | 1,060 lines | Match |
| `test_synthchat_generator.py` | 2,943 lines | 2,943 lines | Match |

No drift since the plan was written.

## 2. Method Listings Verified

### generator.py -- All methods accounted for

The plan's responsibility map lists methods across 10 responsibility clusters. Cross-referencing with `grep -n 'def '` output confirms all 54 methods/functions are present and at their expected line numbers.

**Methods not explicitly assigned to an extraction target in the plan** (these remain in generator.py as orchestrator core):

| Method | Line | Role |
|--------|------|------|
| `_resolve_environment_mode` | 1561 | Orchestration logic (determines generation strategy) |
| `_generate_environment_spec` | 1575 | Orchestration logic (drives environment generation stage) |
| `_generate_assistant_response` | 1626 | Orchestration logic (drives assistant response stage) |
| `_build_user_context` | 2027 | Context builders for stage prompts |
| `_build_assistant_context` | 2035 | Context builders for stage prompts |
| `_build_loop_assistant_context` | 2054 | Context builder for agentic loop |
| `_improve_stage` | 1814 | Orchestration logic (delegates to ImprovementEngine) |
| `_log_stage` | 1848 | Logging utility |

**Assessment**: Correct. These methods are orchestration glue -- they call into the extracted modules (llm, parsing, schemas, workspace) and compose the generation pipeline. They belong in the slimmed-down generator.py. The plan's ~600 line estimate for the post-refactoring generator.py is consistent with these methods plus `__init__`, `generate_batch`, `prepare_seed_bundle`, and `generate_single`.

### run.py -- All functions accounted for

All 16 functions/classes listed in the plan's responsibility map are present at their expected locations. The `_normalize_target_spec` duplicate is confirmed at line 880 (identical to generator.py line 3329).

## 3. External Consumer Verification

| Consumer | Plan Lists | Actual | Status |
|----------|-----------|--------|--------|
| `SynthChat/__init__.py` | `SynthChatGenerator, ScenarioLoader, GenerationResult` | Confirmed (line 23) | Match |
| `SynthChat/run.py` | `SynthChatGenerator, ScenarioLoader, _extract_shared_seed_spec` | Confirmed (line 28) | Match |
| `tuner/handlers/generate_handler.py` | `ScenarioLoader, SynthChatGenerator` | Confirmed (lines 121, 284) | Match |
| `tuner/handlers/synthchat_handler.py` | `SynthChatGenerator` | Confirmed (line 210) | Match |
| `tests/test_synthchat_generator.py` | `SynthChatGenerator, _build_use_tools_*`, `_apply_stage_review_result` | Confirmed (lines 8-9, 2791) | Match |

**Additional consumers not in the plan** (no impact on refactoring):

| Consumer | Imports | Impact |
|----------|---------|--------|
| `tuner/handlers/synthchat_handler.py` | `ImprovementEngine` (line 211) | From `SynthChat.engine`, unchanged |
| `tuner/handlers/synthchat_handler.py` | `RubricRunner` (line 325) | From `SynthChat.services`, unchanged |
| `tuner/handlers/synthchat_handler.py` | `DatasetScanner` (line 492) | From `SynthChat.utils`, unchanged |
| `tuner/handlers/improve_handler.py` | `RubricRunner, DatasetScanner, load_config, ImproveLogger` | From `SynthChat.services` and `SynthChat.utils`, unchanged |

These imports target modules that are not being refactored (`engine.py`, `services/`, `utils/`), so they are unaffected.

## 4. DRY Violation Confirmed

`_normalize_target_spec` at `generator.py:3329` and `run.py:880` are functionally identical (same logic, same error messages, same return structure). The only cosmetic difference is that the `run.py` version omits the docstring. The plan's extraction to `SynthChat/targets.py` is the correct fix.

## 5. Directory Structure -- No New Files

The SynthChat directory contains no new files since the plan was written. The existing subdirectories (`config/`, `scenarios/`, `rubrics/`, `services/`, `utils/`, `scripts/`, `output/`) are all accounted for in the plan as "unchanged".

## 6. Dependency Order Validation

The 9-PR roadmap dependency graph is correct:

```
PR 1 (template_utils + targets) -- no dependencies, foundation
  |
  +-- PR 2 (workspace)     -- uses template_utils (confirmed: _render_template_object used in workspace rendering)
  +-- PR 3 (schemas)        -- independent of PR 2 (confirmed: no workspace imports in schema code)
  +-- PR 4 (labeling)       -- independent of PR 2/3 (confirmed: pure computation)
  +-- PR 5 (parsing)        -- independent of PR 2/3/4 (confirmed: pure transformation)
       |
       +-- PR 6 (llm)       -- follows parsing (confirmed: _call_llm_structured calls _parse_json_object at line 2013)
            |
            +-- PR 7 (review)   -- uses llm client pool (confirmed: _run_stage_review calls _call_llm at line 1132)
            +-- PR 8 (agentic)  -- uses llm + parsing (confirmed: _synthchat_loop_response calls _call_llm)
       |
       +-- PR 9 (run.py)    -- depends on targets (PR 1), independent of PRs 6-8
```

**One refinement**: PR 5 (parsing) does not actually depend on PR 1 (template_utils). The parsing functions (`_parse_assistant_response`, `_parse_json_object`, `_normalize_generated_environment`) do not use any template utilities. The plan shows PR 5 at the same tier as PRs 2-4, which is correct in practice. The dependency arrow in the plan's ASCII diagram is slightly misleading (it groups PRs 2-5 as children of PR 1, implying all depend on it). Only PR 2 (workspace) truly depends on PR 1. PRs 3, 4, and 5 could technically start before PR 1, but starting them after PR 1 is still the right call since PR 1 is low-risk and establishes the extraction pattern.

## 7. Backward Compatibility Re-export Strategy

The plan's re-export strategy covers all current consumers:

| Import Path | Used By | Re-export Location |
|------------|---------|-------------------|
| `SynthChat.generator._extract_shared_seed_spec` | `run.py` | `generator.py` re-exports from `targets.py` |
| `SynthChat.generator._build_use_tools_generation_prompt` | `test_synthchat_generator.py` | `generator.py` re-exports from `schemas/tool_response_schema.py` |
| `SynthChat.generator._build_use_tools_response_schema` | `test_synthchat_generator.py` | `generator.py` re-exports from `schemas/tool_response_schema.py` |
| `SynthChat.generator._apply_stage_review_result` | `test_synthchat_generator.py` | `generator.py` re-exports from `targets.py` |
| `SynthChat.generator.SynthChatGenerator` | Multiple | Stays in `generator.py` (no re-export needed) |
| `SynthChat.generator.ScenarioLoader` | Multiple | Stays in `generator.py` (no re-export needed) |

**Assessment**: Complete. No missing re-exports.

## 8. Test Coupling Analysis

The test file (`tests/test_synthchat_generator.py`, 2,943 lines) uses these coupling patterns:

1. **Class-level imports** (lines 8-9): `SynthChatGenerator`, `_build_use_tools_generation_prompt`, `_build_use_tools_response_schema` -- covered by re-exports.

2. **Inline imports in test functions** (line 2791): `_apply_stage_review_result` -- covered by re-export.

3. **Monkeypatching**: Tests use `_FakeLLMClient` and `_FakeLogger` passed as constructor args, not monkeypatching module attributes. This means the extraction of LLM client management (PR 6) is safe -- tests don't patch `SynthChat.generator._call_llm` directly; they inject fakes through the constructor.

4. **`task_derivation` imports** (lines 2417, 2462, 2505, 2541, 2808): These import from `SynthChat.task_derivation`, which is not being refactored. No impact.

**Assessment**: Test coupling is well-managed. The re-export strategy is sufficient. No test file modifications are required for PRs 1-5. PRs 6-8 may require test updates only if internal method signatures change (the plan correctly keeps the public interface stable).

## 9. Adjustments (Minor)

### A. `_improve_stage` placement

The plan's responsibility map lists `_improve_stage` under "Stage review & improvement" but the PR 7 extraction target for `review/` does not include it. This is correct -- `_improve_stage` is a thin orchestration wrapper around `self.engine.run()` (3 lines of logic) and belongs in the generator orchestrator, not in the review module. No change needed; just noting the implicit decision is sound.

### B. Context builder methods

`_build_user_context`, `_build_assistant_context`, and `_build_loop_assistant_context` are not listed in any extraction target. These are small context-formatting methods (6-12 lines each) tightly coupled to the generation pipeline. Keeping them in generator.py is correct. They could be candidates for a future "context builders" extraction if generator.py grows again, but that is out of scope.

### C. `_generate_assistant_response` dependencies on schemas

`_generate_assistant_response` (line 1626) calls `_build_use_tools_response_schema`, `_build_use_tools_generation_prompt`, `_resolve_allowed_tool_names`, `_resolve_context_defaults`, and `_tool_wrapper_name` -- all of which move to `schemas/` in PR 3. After PR 3, generator.py will need to import these from `SynthChat.schemas.tool_response_schema`. The plan accounts for this ("Import from `schemas/` subpackage"). Confirmed correct.

### D. PR 9 scope consideration

`run.py` line 149 defines `create_environment_validator` which imports from `shared.environments`. The plan keeps this in run.py's slimmed-down form. This is fine, but note that `_create_environment_validator_from_options` (line 802, for parallel workers) moves to `parallel/executor.py`. These two validator-creation functions serve different purposes (CLI setup vs. per-worker reconstruction) so the split is appropriate.

---

## Summary

The plan is ready for implementation. All method listings, line numbers, external consumers, dependency order, and re-export strategy are verified against the current code. No blocking issues found. The minor adjustments above are documentation clarifications, not design changes.

**Recommended implementation order**: Follow the plan's 9-PR sequence exactly as written. PRs 2-5 can be parallelized after PR 1. PRs 7-8 depend on PR 6. PR 9 can run in parallel with PRs 6-8.
