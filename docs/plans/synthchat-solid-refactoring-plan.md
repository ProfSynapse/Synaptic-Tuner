# SynthChat SOLID Refactoring Plan

## Executive Summary

`SynthChat/generator.py` (3,384 lines) is a god class with ~35 methods spanning 6+ distinct responsibilities. `SynthChat/run.py` (1,060 lines) mixes CLI argument parsing, three mode functions, parallel execution, and serialization. This plan decomposes both into focused modules with clear SRP boundaries, incremental PR-safe steps, and backward-compatible re-exports.

---

## Current State Analysis

### generator.py Responsibility Map

| Responsibility | Methods | Lines (approx) | Coupling |
|---|---|---|---|
| **Batch orchestration** | `generate_batch`, `generate_single`, `prepare_seed_bundle` | ~600 | High -- calls everything |
| **LLM client management** | `_get_or_create_llm_client`, `_normalize_stage_llm_spec`, `_get_stage_llm_clients`, `_call_llm`, `_call_llm_structured` | ~300 | Medium -- used by all generation methods |
| **Response parsing** | `_parse_assistant_response`, `_parse_json_object`, `_normalize_generated_environment`, `_normalize_generated_assertion`, `_stringify_assistant_message` | ~200 | Low -- pure transformation |
| **Environment/workspace rendering** | `_render_mocked_workspace_system_prompt`, plus ~500 lines of module-level helpers (`_workspace_structure_from_fixture`, `_vault_structure_text_from_fixture`, `_render_available_*`, `_build_*_section`, `_note_entries_from_fixture`, `_render_note_contents`, `_render_extra_sections`) | ~550 | Low -- only called from `_render_mocked_workspace_system_prompt` |
| **Schema builders** | `_scalar_schema`, `_assertion_schema`, `_build_canonical_environment_schema`, `_build_canonical_environment_generation_prompt`, `_build_use_tools_response_schema`, `_build_use_tools_generation_prompt` | ~350 | Low -- pure data construction |
| **Stage review & improvement** | `_run_stage_review`, `_run_configured_stage_judge`, `_build_stage_judge_template_vars`, `_improve_stage`, `_build_environment_generation_review_payload` | ~200 | Medium -- needs LLM client + engine |
| **Agentic episode generation** | `_generate_agentic_episode`, `_build_turn_judge`, `_build_turn_judge_template_vars`, `_synthchat_loop_response`, `_validate_agentic_synthchat_response` | ~250 | Medium -- uses LLM + environment validator |
| **Metadata labeling** | `_build_metadata_labels`, `_classify_environment_issue`, `_derive_kto_candidate_label`, `_slugify_label` | ~150 | Low -- pure transformation |
| **Template rendering** | `_render_template_object`, `_task_context_template_vars`, `_user_generation_style_instructions`, `_deep_merge_dicts`, `_make_json_safe`, `_clean_path` | ~150 | Low -- pure utilities |
| **Target spec handling** | `_normalize_target_spec`, `_extract_shared_seed_spec`, `_apply_stage_review_result` | ~80 | Low -- pure transformation (also duplicated in run.py) |

### run.py Responsibility Map

| Responsibility | Functions | Lines (approx) |
|---|---|---|
| **CLI argument parsing** | `main()`, parser setup | ~100 |
| **Generate mode** | `generate_mode()` | ~300 |
| **Improve mode** | `improve_mode()` | ~110 |
| **Validate mode** | `validate_mode()` | ~120 |
| **Parallel execution** | `_run_parallel_generation`, `_generate_single_example`, `_create_worker_generator`, `_serialize_environment_options`, `_create_environment_validator_from_options` | ~160 |
| **Output/serialization** | `StreamingResultWriter`, `_save_results`, `_print_summary`, `_generate_output_path` | ~120 |
| **Setup helpers** | `load_settings`, `create_llm_client`, `create_environment_validator` | ~90 |
| **Duplicate code** | `_normalize_target_spec` (identical to generator.py version) | ~20 |

### External Consumers (Import Surface)

| Consumer | Imports |
|---|---|
| `SynthChat/__init__.py` | `SynthChatGenerator`, `ScenarioLoader`, `GenerationResult` |
| `SynthChat/run.py` | `SynthChatGenerator`, `ScenarioLoader`, `_extract_shared_seed_spec` |
| `tuner/handlers/generate_handler.py` | `ScenarioLoader`, `SynthChatGenerator` |
| `tuner/handlers/synthchat_handler.py` | `SynthChatGenerator` |
| `tests/test_synthchat_generator.py` | `SynthChatGenerator`, `_build_use_tools_generation_prompt`, `_build_use_tools_response_schema`, `_apply_stage_review_result` |

### DRY Violations

1. **`_normalize_target_spec`**: Identical implementations in `generator.py:3329` and `run.py:880`

---

## Proposed Module Structure

```
SynthChat/
├── __init__.py                    # Public API (unchanged exports)
├── generator.py                   # SynthChatGenerator (orchestrator, ~600 lines)
├── engine.py                      # ImprovementEngine (unchanged)
├── run.py                         # CLI arg parsing + mode dispatch (~150 lines)
├── stage_gates.py                 # Existing (unchanged)
├── task_derivation.py             # Existing (unchanged)
├── task_requirements.py           # Existing (unchanged)
├── task_selectors.py              # Existing (unchanged)
│
├── llm/                           # NEW: LLM client management
│   ├── __init__.py
│   ├── client_pool.py             # LLMClientPool class (~200 lines)
│   └── caller.py                  # _call_llm, _call_llm_structured (~200 lines)
│
├── parsing/                       # NEW: Response parsing and normalization
│   ├── __init__.py
│   ├── response_parser.py         # _parse_assistant_response, _parse_json_object (~150 lines)
│   └── environment_normalizer.py  # _normalize_generated_environment, _normalize_generated_assertion (~100 lines)
│
├── workspace/                     # NEW: Mocked workspace rendering
│   ├── __init__.py
│   ├── renderer.py                # render_mocked_workspace_system_prompt (~80 lines)
│   ├── sections.py                # Section builders (_build_*_section, _render_*) (~200 lines)
│   └── fixture_helpers.py         # Fixture->structure conversion helpers (~200 lines)
│
├── schemas/                       # NEW: JSON schema builders
│   ├── __init__.py
│   ├── environment_schema.py      # Canonical environment schema + prompt (~150 lines)
│   └── tool_response_schema.py    # use_tools response schema + prompt (~150 lines)
│
├── review/                        # NEW: Stage review and judging
│   ├── __init__.py
│   ├── stage_reviewer.py          # StageReviewer class (~150 lines)
│   └── judge_templates.py         # Template variable builders (~100 lines)
│
├── agentic/                       # NEW: Agentic episode generation
│   ├── __init__.py
│   └── episode.py                 # AgenticEpisodeRunner (~250 lines)
│
├── labeling/                      # NEW: Metadata and label construction
│   ├── __init__.py
│   └── metadata_labels.py         # _build_metadata_labels + helpers (~150 lines)
│
├── targets.py                     # NEW: Target spec handling (deduplicated)
│                                  # _normalize_target_spec, _extract_shared_seed_spec (~80 lines)
│
├── template_utils.py              # NEW: Template rendering utilities
│                                  # _render_template_object, _deep_merge_dicts, _make_json_safe, etc. (~150 lines)
│
├── modes/                         # NEW: CLI mode implementations
│   ├── __init__.py
│   ├── generate.py                # generate_mode + work item building (~250 lines)
│   ├── improve.py                 # improve_mode (~110 lines)
│   └── validate.py                # validate_mode (~120 lines)
│
├── parallel/                      # NEW: Parallel execution infrastructure
│   ├── __init__.py
│   └── executor.py                # _run_parallel_generation, worker helpers (~160 lines)
│
├── output/                        # NEW: Output and serialization
│   ├── __init__.py
│   └── writer.py                  # StreamingResultWriter, _save_results, _print_summary (~120 lines)
│
├── config/                        # Existing config directory (unchanged)
├── scenarios/                     # Existing (unchanged)
├── rubrics/                       # Existing (unchanged)
├── services/                      # Existing (unchanged)
└── utils/                         # Existing (unchanged)
```

---

## Dependency Diagram

```
                    SynthChat/__init__.py
                           │
                           ▼
                    ┌─────────────┐
                    │  generator   │  SynthChatGenerator (orchestrator)
                    └──────┬──────┘
                           │
          ┌───────┬────────┼────────┬───────────┬──────────┐
          ▼       ▼        ▼        ▼           ▼          ▼
      ┌───────┐ ┌──────┐ ┌──────┐ ┌─────────┐ ┌────────┐ ┌──────────┐
      │  llm/ │ │parse/│ │work- │ │ review/ │ │agentic/│ │ labeling/│
      │       │ │      │ │space/│ │         │ │        │ │          │
      └───┬───┘ └──────┘ └──┬───┘ └────┬────┘ └───┬────┘ └──────────┘
          │                  │          │          │
          │                  ▼          │          ▼
          │            ┌──────────┐     │    ┌──────────────┐
          │            │ schemas/ │     │    │ template_utils│
          │            └──────────┘     │    └──────────────┘
          │                             │
          └─────────────────────────────┘
                  (review and agentic use llm/)

                        run.py (CLI dispatch)
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ modes/   │    │parallel/ │    │ output/  │
    │generate  │    │executor  │    │ writer   │
    │improve   │    └──────────┘    └──────────┘
    │validate  │
    └──────────┘
          │
          ▼
     ┌──────────┐
     │ targets  │  (shared by generator + modes)
     └──────────┘
```

### Dependency Rules

1. **No circular dependencies**: Each extracted module depends only on modules below it in the diagram (or on `shared/` and external libraries)
2. **generator.py remains the facade**: It imports from extracted modules, not the reverse
3. **targets.py is shared**: Used by both generator.py and modes/generate.py, eliminating the DRY violation
4. **template_utils.py is shared**: Pure functions used by generator, workspace, and schemas

---

## Implementation Roadmap

### PR 1: Extract `template_utils.py` and `targets.py` (Low Risk)

**Goal**: Extract pure utility functions, fix the DRY violation.

**Extract to `SynthChat/template_utils.py`**:
- `_deep_merge_dicts`
- `_make_json_safe`
- `_render_template_object`
- `_task_context_template_vars`
- `_clean_path`
- `_user_generation_style_instructions`

**Extract to `SynthChat/targets.py`**:
- `_normalize_target_spec`
- `_extract_shared_seed_spec`
- `_apply_stage_review_result`

**Changes**:
- `generator.py`: Replace local definitions with imports from new modules
- `run.py`: Replace `_normalize_target_spec` duplicate with import from `SynthChat.targets`
- Re-export from `generator.py` for backward compatibility of test imports

**Risk**: Low. Pure functions with no state, no side effects. Tests verify behavior, not location.

**Test impact**: Update import in `test_synthchat_generator.py:2791` for `_apply_stage_review_result`. Other test imports work via re-exports.

---

### PR 2: Extract `workspace/` (Low Risk)

**Goal**: Move the ~550 lines of workspace rendering into a focused subpackage.

**Extract to `SynthChat/workspace/renderer.py`**:
- `_render_mocked_workspace_system_prompt` (becomes `render_mocked_workspace_system_prompt`)

**Extract to `SynthChat/workspace/sections.py`**:
- `_build_session_context_section`
- `_build_wrapped_section`
- `_build_selected_workspace_section`
- `_render_available_workspaces`
- `_render_available_prompts`
- `_render_available_tools`
- `_tool_wrapper_name`
- `_render_extra_sections`

**Extract to `SynthChat/workspace/fixture_helpers.py`**:
- `_merged_fixture_from_config`
- `_workspace_structure_from_fixture`
- `_vault_structure_text_from_fixture`
- `_note_entries_from_fixture`
- `_render_note_contents`

**Changes**:
- `generator.py`: Replace method with import; `SynthChatGenerator._render_mocked_workspace_system_prompt` delegates to `workspace.renderer.render_mocked_workspace_system_prompt`
- All module-level workspace helpers removed from generator.py

**Risk**: Low. These functions form a self-contained cluster with no dependencies on SynthChatGenerator state. The only connection is `self.environment_validator.tool_schema`, which is passed as a parameter.

**Test impact**: Minimal. Tests call `SynthChatGenerator` methods which internally delegate. Existing tests continue to pass unchanged.

---

### PR 3: Extract `schemas/` (Low Risk)

**Goal**: Move JSON schema construction into a dedicated subpackage.

**Extract to `SynthChat/schemas/environment_schema.py`**:
- `_scalar_schema`
- `_assertion_schema`
- `_build_canonical_environment_schema`
- `_build_canonical_environment_generation_prompt`

**Extract to `SynthChat/schemas/tool_response_schema.py`**:
- `_build_use_tools_response_schema`
- `_build_use_tools_generation_prompt`
- `_resolve_allowed_tool_names`
- `_resolve_context_defaults`

**Changes**:
- `generator.py`: Import from `schemas/` subpackage
- Re-export `_build_use_tools_generation_prompt` and `_build_use_tools_response_schema` from `generator.py` for test backward compatibility

**Risk**: Low. Pure data-returning functions with no side effects.

**Test impact**: `test_synthchat_generator.py` imports `_build_use_tools_generation_prompt` and `_build_use_tools_response_schema` from `SynthChat.generator`. Re-exports maintain compatibility.

---

### PR 4: Extract `labeling/` (Low Risk)

**Goal**: Move metadata label construction into a focused module.

**Extract to `SynthChat/labeling/metadata_labels.py`**:
- `_build_metadata_labels` (becomes standalone function, receives all data as params)
- `_classify_environment_issue`
- `_derive_kto_candidate_label`
- `_slugify_label`

**Changes**:
- `generator.py`: `SynthChatGenerator._build_metadata_labels` delegates to imported function

**Risk**: Low. Pure computation with no dependencies on generator state.

**Test impact**: None -- tested through `SynthChatGenerator` integration tests.

---

### PR 5: Extract `parsing/` (Low Risk)

**Goal**: Move response parsing and normalization into a focused subpackage.

**Extract to `SynthChat/parsing/response_parser.py`**:
- `_parse_assistant_response` (becomes standalone, receives scenario as param)
- `_parse_json_object`
- `_stringify_assistant_message`

**Extract to `SynthChat/parsing/environment_normalizer.py`**:
- `_normalize_generated_environment`
- `_normalize_generated_assertion`

**Changes**:
- `generator.py`: Methods delegate to imported functions

**Risk**: Low. Parsing is pure transformation; the only state dependency is the scenario dict (passed as parameter).

**Test impact**: None -- tested through SynthChatGenerator integration.

---

### PR 6: Extract `llm/` (Medium Risk)

**Goal**: Extract LLM client management into a dedicated subpackage.

**Extract to `SynthChat/llm/client_pool.py`**:
- `LLMClientPool` class encapsulating:
  - `_llm_client_cache` (the cache dict)
  - `_get_or_create_llm_client`
  - `_normalize_stage_llm_spec`
  - `_get_stage_llm_clients`

**Extract to `SynthChat/llm/caller.py`**:
- `call_llm` (from `_call_llm`)
- `call_llm_structured` (from `_call_llm_structured`)

**Changes**:
- `generator.py`: `SynthChatGenerator.__init__` creates an `LLMClientPool`; all methods use `self.client_pool` instead of direct cache access
- LLM calling methods become functions that accept a client pool

**Risk**: Medium. The client pool holds mutable state (cache dict) and the calling methods have retry logic with logging. Need to verify thread safety for parallel execution.

**Test impact**: Tests use `_FakeLLMClient` mock. The extraction doesn't change the mock interface, but tests that verify retry behavior or client caching need to ensure the pool is properly initialized in test fixtures.

---

### PR 7: Extract `review/` (Medium Risk)

**Goal**: Extract stage review and judging into a dedicated subpackage.

**Extract to `SynthChat/review/stage_reviewer.py`**:
- `StageReviewer` class encapsulating:
  - `_run_stage_review`
  - `_run_configured_stage_judge`
  - `_build_environment_generation_review_payload`

**Extract to `SynthChat/review/judge_templates.py`**:
- `_build_stage_judge_template_vars`

**Changes**:
- `generator.py`: Creates a `StageReviewer` in `__init__`, delegates review calls

**Risk**: Medium. The reviewer needs access to the LLM client pool and the logger. Must pass these as constructor dependencies, not reach back into generator state.

**Test impact**: Moderate. Tests like `test_environment_generation_stage_review_runs_when_configured` and `test_final_stage_judge_runs_when_configured` exercise review through the generator. They should continue to pass, but new unit tests for `StageReviewer` in isolation would improve coverage.

---

### PR 8: Extract `agentic/` (Medium Risk)

**Goal**: Extract agentic episode generation into a focused module.

**Extract to `SynthChat/agentic/episode.py`**:
- `AgenticEpisodeRunner` class or standalone functions:
  - `_generate_agentic_episode`
  - `_build_turn_judge`
  - `_build_turn_judge_template_vars`
  - `_synthchat_loop_response`
  - `_validate_agentic_synthchat_response`

**Changes**:
- `generator.py`: Delegates agentic generation to extracted module
- Must pass `environment_validator`, `llm_client_pool`, and `logger` as dependencies

**Risk**: Medium. Agentic episode generation has the most complex interaction with external dependencies (`run_environment_episode`, `AgenticTurnJudge`, `validate_assistant_response`). The optional imports at the top of `generator.py` need careful handling.

**Test impact**: Tests like `test_synthchat_generator_can_run_shared_agentic_loop` and `test_generate_single_can_use_turn_judge_and_require_final_text` exercise this path. They pass through the generator facade, so they should work unchanged, but the optional import pattern needs to be preserved.

---

### PR 9: Refactor `run.py` into `modes/`, `parallel/`, `output/` (Medium Risk)

**Goal**: Decompose run.py from 1,060 lines into focused modules.

**Extract to `SynthChat/modes/generate.py`**:
- `generate_mode` function
- Work item building logic (currently inline in generate_mode)

**Extract to `SynthChat/modes/improve.py`**:
- `improve_mode` function

**Extract to `SynthChat/modes/validate.py`**:
- `validate_mode` function

**Extract to `SynthChat/parallel/executor.py`**:
- `_run_parallel_generation`
- `_generate_single_example`
- `_create_worker_generator`
- `_serialize_environment_options`
- `_create_environment_validator_from_options`

**Extract to `SynthChat/output/writer.py`**:
- `StreamingResultWriter`
- `_save_results`
- `_print_summary`
- `_generate_output_path`

**Result: `run.py` becomes ~150 lines**:
- `load_settings`
- `create_llm_client`
- `create_environment_validator`
- `main()` (argparse + dispatch)

**Risk**: Medium. The mode functions are large and interleave setup, execution, and output. The parallel execution code creates per-thread generator instances.

**Test impact**: No existing tests for `run.py` functions were found. This is a risk -- the refactoring should not introduce regressions in untested code. The streaming writer and parallel executor are critical paths that would benefit from new tests.

---

## Risk Assessment

### High Risk Items

| Risk | Impact | Mitigation |
|---|---|---|
| **Breaking external imports** | `tuner/handlers/` imports `SynthChatGenerator`, `ScenarioLoader` by path | `SynthChat/__init__.py` and `generator.py` maintain all current public exports |
| **Test imports of private functions** | Tests import `_build_use_tools_*`, `_apply_stage_review_result` from `generator.py` | Re-export from generator.py; eventually update test imports |
| **Thread safety of LLMClientPool** | Parallel workers create per-worker generators, but shared seed bundles cross threads | Current code already handles this (each worker creates its own generator). Pool extraction must not change this pattern. |

### Medium Risk Items

| Risk | Impact | Mitigation |
|---|---|---|
| **Optional imports (agentic deps)** | `AgenticTurnJudge`, `run_environment_episode`, etc. are try/except imported | Preserve optional import pattern in `agentic/episode.py` |
| **Monkeypatch in tests** | Tests monkeypatch `SynthChat.generator` module attributes | Re-exports ensure monkeypatching the facade still works |
| **run.py has no tests** | Mode functions are untested | Run.py refactoring is mechanical (move functions, update imports). Manual smoke test the CLI after. |

### Low Risk Items

| Risk | Impact | Mitigation |
|---|---|---|
| **Circular imports** | New subpackages might create cycles | Dependency diagram enforces one-way flow; template_utils and targets have no SynthChat imports |
| **Module initialization cost** | More files = more import time | Python caches imports; the additional overhead is negligible |

---

## Testing Strategy

### Existing Test Coverage (2,943 lines)

The test file `tests/test_synthchat_generator.py` is comprehensive with ~40 test functions. The tests primarily exercise `SynthChatGenerator` through its public interface, which means the extraction of internal methods into separate modules should not break tests as long as:

1. `SynthChatGenerator` maintains the same public API
2. Private functions re-exported from `generator.py` remain importable at the same paths
3. Mock/fake objects (`_FakeLLMClient`, `_FakeLogger`) continue to satisfy the interfaces

### Recommended New Tests per PR

| PR | New Tests |
|---|---|
| PR 1 (template_utils, targets) | Unit tests for `_deep_merge_dicts`, `_normalize_target_spec` edge cases in new locations |
| PR 2 (workspace) | Unit test for `render_mocked_workspace_system_prompt` in isolation |
| PR 3 (schemas) | Already well-covered by existing tests |
| PR 6 (llm) | Unit test for `LLMClientPool` caching behavior |
| PR 7 (review) | Unit test for `StageReviewer` with mock judge |
| PR 9 (run.py modes) | Integration test for `generate_mode` with mock LLM |

### Regression Verification

After each PR:
1. Run `pytest tests/test_synthchat_generator.py` -- all 40+ tests must pass
2. Verify `from SynthChat import SynthChatGenerator, ScenarioLoader, GenerationResult` works
3. Verify `from SynthChat.generator import _extract_shared_seed_spec` works (used by run.py)
4. Verify `python -m SynthChat.run --help` works (CLI not broken)

---

## Implementation Order and Dependencies

```
PR 1 (template_utils + targets)   ←── No dependencies, DRY fix
  │
  ├── PR 2 (workspace)            ←── Depends on template_utils
  ├── PR 3 (schemas)              ←── No dependency on PR 2
  ├── PR 4 (labeling)             ←── No dependency on PR 2/3
  └── PR 5 (parsing)              ←── No dependency on PR 2/3/4
       │
       ├── PR 6 (llm)             ←── Should follow parsing (parsing uses _parse_json_object)
       │    │
       │    ├── PR 7 (review)     ←── Depends on llm client pool
       │    └── PR 8 (agentic)    ←── Depends on llm + parsing
       │
       └── PR 9 (run.py)          ←── Depends on targets (PR 1), can run in parallel with PRs 6-8
```

PRs 2-5 can be done in parallel after PR 1. PRs 7-8 depend on PR 6. PR 9 can start any time after PR 1.

---

## Post-Refactoring File Sizes

| Module | Lines (approx) |
|---|---|
| `generator.py` (orchestrator) | ~600 |
| `llm/client_pool.py` | ~200 |
| `llm/caller.py` | ~200 |
| `parsing/response_parser.py` | ~150 |
| `parsing/environment_normalizer.py` | ~100 |
| `workspace/renderer.py` | ~80 |
| `workspace/sections.py` | ~200 |
| `workspace/fixture_helpers.py` | ~200 |
| `schemas/environment_schema.py` | ~150 |
| `schemas/tool_response_schema.py` | ~150 |
| `review/stage_reviewer.py` | ~150 |
| `review/judge_templates.py` | ~100 |
| `agentic/episode.py` | ~250 |
| `labeling/metadata_labels.py` | ~150 |
| `template_utils.py` | ~150 |
| `targets.py` | ~80 |
| `run.py` (CLI shell) | ~150 |
| `modes/generate.py` | ~250 |
| `modes/improve.py` | ~110 |
| `modes/validate.py` | ~120 |
| `parallel/executor.py` | ~160 |
| `output/writer.py` | ~120 |

All modules are under the 500-600 line maintainability threshold.

---

## Backward Compatibility Contract

### Must Preserve

1. `from SynthChat import SynthChatGenerator, ScenarioLoader, GenerationResult` -- public API
2. `from SynthChat.generator import SynthChatGenerator` -- direct import
3. `from SynthChat.generator import _extract_shared_seed_spec` -- used by run.py
4. `from SynthChat.generator import _build_use_tools_generation_prompt, _build_use_tools_response_schema` -- used by tests
5. `from SynthChat.generator import _apply_stage_review_result` -- used by tests
6. `python -m SynthChat.run generate|improve|validate` -- CLI entry point

### Re-export Strategy

In `generator.py`, after extracting functions to new modules, add re-exports:
```python
# Backward compatibility re-exports
from .targets import _normalize_target_spec, _extract_shared_seed_spec, _apply_stage_review_result
from .schemas.tool_response_schema import _build_use_tools_generation_prompt, _build_use_tools_response_schema
```

These re-exports can be removed in a future cleanup PR once all consumers are updated to import from the canonical location.
