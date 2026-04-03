# Evaluator Module SOLID Refactoring Plan

**Status**: Proposed
**Scope**: `Evaluator/config_validator.py` (1,087 lines) + `tuner/handlers/eval_handler.py` (929 lines)
**Goal**: Apply Strategy Pattern and DRY extraction to reduce boilerplate, improve extensibility, and lower the cost of adding new checks/table displays.

---

## Part 1: config_validator.py — Strategy Pattern + Declarative Check Registry

### Current State Analysis

`ConfigDrivenValidator` has 17 `_check_*` methods (lines 652-1064) registered in `_register_checks()` (line 104). Every method shares the identical signature:

```python
def _check_*(self, parsed: ParsedResponse, params: Dict[str, Any],
             issues: List[ValidationIssue], check_name: str) -> None:
```

And the identical failure pattern:

```python
issues.append(ValidationIssue(
    level="error", check=check_name,
    message=f"...", expected=..., actual=...,
))
```

#### Boilerplate vs Unique Logic Breakdown

| Check Method | Unique Lines | Boilerplate Lines | Category |
|---|---|---|---|
| `_check_tool_called` | 3 | 6 | tool-set |
| `_check_any_tool_called` | 4 | 7 | tool-set |
| `_check_all_tools_called` | 4 | 7 | tool-set |
| `_check_tool_not_called` | 3 | 6 | tool-set |
| `_check_tool_count` | 4 | 12 | comparison |
| `_check_tool_params_present` | 4 | 8 | tool-field |
| `_check_tool_sequence` | 5 | 14 | tool-set |
| `_check_fields_present` | 4 | 8 | field-check |
| `_check_fields_min_length` | 4 | 8 | field-check |
| `_check_field_equals` | 3 | 8 | field-check |
| `_check_text_contains` | 2 | 6 | text-check |
| `_check_text_contains_any` | 2 | 6 | text-check |
| `_check_text_not_contains` | 3 | 7 | text-check |
| `_check_text_matches` | 2 | 6 | text-check |
| `_check_text_min_length` | 3 | 6 | comparison |
| `_check_text_max_length` | 3 | 6 | comparison |
| `_check_batch_structure` | 2 | 6 | comparison |
| `_check_batch_strategy` | 2 | 6 | comparison |

**Total**: ~58 unique lines vs ~128 boilerplate lines across 17 methods (413 total lines including docstrings, blanks).

#### Structural Patterns Identified

Five distinct check "shapes" emerge:

1. **Tool-set checks** (4 methods): Build `called = {tc.name for tc in parsed.tool_calls}`, then do a set operation (in, intersection, difference, not-in). Differ only in which set operation and error message.

2. **Comparison checks** (4 methods): Extract a numeric value (text length, tool count), compare against min/max from params. Identical structure with different extraction.

3. **Text-content checks** (4 methods): Operate on `parsed.text_content` with string contains/regex. Nearly identical — only the predicate differs.

4. **Field-check checks** (3 methods): Iterate `parsed.tool_calls`, access `tc.context` or `tc.params`, check field presence/value/length.

5. **Special checks** (2 methods): `_check_tool_sequence` and `_check_tool_params_present` have slightly more logic but still fit the pattern.

### Proposed Architecture

#### Option A: Declarative Check Descriptors (Recommended)

Replace each `_check_*` method with a declarative `CheckDescriptor` that separates "what to check" from "how to report failures". Each descriptor defines:
- **extractor**: How to get the value being checked from `ParsedResponse` + `params`
- **predicate**: A condition function that returns `(passed: bool, actual: Any)`
- **message_template**: How to format the error message

```python
@dataclass
class CheckDescriptor:
    """Declarative description of a validation check."""
    name: str
    extractor: Callable[[ParsedResponse, Dict[str, Any]], Any]
    predicate: Callable[[Any, Dict[str, Any]], Tuple[bool, Any]]
    message_template: str
    expected_template: str = ""
```

The runner becomes a single generic method:

```python
def _run_check(self, descriptor: CheckDescriptor, parsed, params, issues, check_name):
    value = descriptor.extractor(parsed, params)
    passed, actual = descriptor.predicate(value, params)
    if not passed:
        issues.append(ValidationIssue(
            level="error", check=check_name,
            message=descriptor.message_template.format(**params, actual=actual),
            expected=descriptor.expected_template.format(**params),
            actual=actual,
        ))
```

**Registration becomes data**:

```python
CHECKS = {
    "tool_called": CheckDescriptor(
        name="tool_called",
        extractor=lambda p, params: {tc.name for tc in p.tool_calls},
        predicate=lambda called, params: (params["tool"] in called, list(called)),
        message_template="Tool '{tool}' was not called",
        expected_template="{tool}",
    ),
    "text_min_length": CheckDescriptor(
        name="text_min_length",
        extractor=lambda p, params: len(p.text_content.strip()),
        predicate=lambda length, params: (length >= params.get("min", 0), length),
        message_template="Text too short: {actual} < {min}",
        expected_template=">= {min}",
    ),
    # ... etc
}
```

**Why this approach**:
- Adding a new check = adding one `CheckDescriptor` instance (no new method, no registration step)
- The `_run_check` method is testable independently
- Check descriptors can be composed (e.g., a "negate" wrapper for `tool_not_called`)
- The existing `behaviors.yaml` already uses declarative check references — this aligns the Python layer with the YAML layer
- Reduces 413 lines of check implementations to ~100 lines of descriptors + 15 lines of runner

#### Option B: Keep Methods, Extract Boilerplate Helpers

Less ambitious: keep `_check_*` methods but extract the common boilerplate into helpers:

```python
def _fail(self, issues, check_name, message, expected=None, actual=None):
    issues.append(ValidationIssue(level="error", check=check_name,
                                   message=message, expected=expected, actual=actual))

def _get_called_tools(self, parsed) -> Set[str]:
    return {tc.name for tc in parsed.tool_calls}
```

This reduces each method by ~3-4 lines but doesn't address the fundamental issue: every new check type still requires a new method + registration entry. **Not recommended** as the primary approach, but could serve as an intermediate step.

#### Recommendation: Option A

Option A is the right choice because the check system is already data-driven from the YAML side (behaviors.yaml defines checks declaratively). The Python implementation should mirror that philosophy. This eliminates the impedance mismatch where YAML declares "what" but Python requires hand-coded "how" for each check type.

### Migration Strategy

The existing `_register_checks` dict and the proposed `CHECKS` dict share the same key space (check names). This means migration can be incremental: the runner checks `CHECKS` first, falls back to `self.check_functions` for not-yet-migrated methods.

### Handling Complex Checks

Two methods have more complex logic than the simple extractor/predicate model:

1. **`_check_tool_sequence`** (lines 793-829): Checks multiple conditions (first_any_of, not_first) from the same params dict. This can be handled by making the predicate return multiple issues or by decomposing into sub-checks.

2. **`_check_tool_params_present`** (lines 769-791): Iterates tool calls and checks for a specific tool match before validating params. The extractor can handle this by filtering to the matching tool call first.

Neither requires abandoning the declarative approach — they need slightly richer extractors.

### Composability: Negation and OR

The codebase already handles OR logic at the behavior level (`_validate_behavior`, line 567). The check system should support composable operations:

- **Negation**: `_check_tool_not_called` is just `_check_tool_called` with inverted predicate. A `NegatedCheck` wrapper avoids duplicating descriptors.
- **Min/Max pairs**: `_check_text_min_length` and `_check_text_max_length` share the same extractor. A `BoundCheck` descriptor parameterized by direction eliminates this duplication.

---

## Part 2: eval_handler.py — Generic Table Display

### Current State Analysis

`EvalHandler` has 7 `_display_*_table` methods spanning lines 178-577 (400 lines). All follow the same skeleton:

```python
def _display_X_table(self, ...) -> None:
    if RICH_AVAILABLE:
        from rich.table import Table
        from rich import box as rich_box
        table = Table(title=TITLE, box=rich_box.ROUNDED, border_style=COLORS["cello"])
        table.add_column(...)  # N columns
        for i, item in enumerate(items, 1):
            # extract fields from item
            table.add_row(str(i), ...)
        console.print()
        console.print(table)
        console.print()
    else:
        print(f"\n{title}:")
        for i, item in enumerate(items, 1):
            print(f"  [{i}] {format_item(item)}")
        print()
```

#### Variation Points by Method

| Method | Title | Columns | Row Extraction | Extra Logic |
|---|---|---|---|---|
| `_display_models_table` | `"Available {backend} Models"` | `#, Model` | `str(i), m` | None |
| `_display_gguf_models_table` | `"Available GGUF Models"` | `#, Name, Quant, Size, Type` | `backend.get_model_info()` → name, quant, size_gb, trainer | Format size as `{:.1f}GB` |
| `_display_lora_models_table` | `"Available LoRA Adapters"` | `#, Run, Base Model, Type, Size` | `backend.get_model_info()` → timestamp, base_model_short, trainer, size_mb | Format size as `{:.0f}MB` |
| `_display_mlc_models_table` | `"Available MLC/WebGPU Models"` | `#, Name, Arch, Quant, Type` | `backend.get_model_info()` → name, architecture, quantization, trainer | None |
| `_display_training_runs_table` | `"Available {type} Training Runs"` | `#, Run, Has Final, Checkpoints` | `run_path.name`, final check, checkpoint count | Filesystem checks |
| `_display_checkpoints_table` | `"Available Checkpoints"` | `#, Checkpoint, Step, Loss, [KTO cols], Epoch` | `cp.step`, `cp.metrics.*` | Conditional columns by trainer_type |
| `_display_scenarios_table` | `"Available Test Scenarios"` | `#, Name, Description, Tests` | `info.name`, `info.description`, `info.count` | None |

**Key observations**:
- 6 of 7 methods have static column definitions. Only `_display_checkpoints_table` has conditional columns (KTO adds KL+Margin; GRPO adds Reward).
- Row extraction ranges from trivial (pass-through) to moderate (filesystem checks in training_runs, metric formatting in checkpoints).
- The Rich Table creation + console.print boilerplate is identical across all 7.
- The plain-text fallback is identical across all 7 (only the format string changes).

### Proposed Architecture

#### A `TableSpec` Data Class + Generic Renderer

```python
@dataclass
class ColumnSpec:
    """Specification for a single table column."""
    header: str
    style: str = "white"
    width: Optional[int] = None
    justify: str = "left"

@dataclass
class TableSpec:
    """Specification for a display table."""
    title: str
    columns: List[ColumnSpec]
    row_extractor: Callable[[int, Any], List[str]]  # (index, item) -> row values
    plain_formatter: Callable[[int, Any], str]       # (index, item) -> plain text line
```

The generic renderer:

```python
def _display_table(self, items: List[Any], spec: TableSpec) -> None:
    if RICH_AVAILABLE:
        from rich.table import Table
        from rich import box as rich_box

        table = Table(title=spec.title, box=rich_box.ROUNDED,
                      border_style=COLORS["cello"])

        # Always add # column first
        table.add_column("#", style=COLORS["orange"], width=4, justify="center")
        for col in spec.columns:
            table.add_column(col.header, style=col.style,
                             width=col.width, justify=col.justify)

        for i, item in enumerate(items, 1):
            row = spec.row_extractor(i, item)
            table.add_row(str(i), *row)

        console.print()
        console.print(table)
        console.print()
    else:
        print(f"\n{spec.title}:")
        for i, item in enumerate(items, 1):
            print(f"  [{i}] {spec.plain_formatter(i, item)}")
        print()
```

Each current method becomes a thin wrapper that builds a `TableSpec` and calls `_display_table`:

```python
def _display_models_table(self, backend: str, models: List[str]) -> None:
    spec = TableSpec(
        title=f"Available {backend.title()} Models",
        columns=[ColumnSpec("Model")],
        row_extractor=lambda i, m: [m],
        plain_formatter=lambda i, m: m,
    )
    self._display_table(models, spec)
```

#### Handling Conditional Columns (Checkpoints)

`_display_checkpoints_table` adds different columns based on `trainer_type`. Two approaches:

1. **Build the TableSpec dynamically** — construct the columns list based on trainer_type before calling `_display_table`. The row_extractor closure captures trainer_type. This is the simplest approach and keeps the generic renderer unchanged.

2. **Column visibility flags** — add an `Optional[Callable[[], bool]]` to `ColumnSpec` for conditional visibility. Over-engineering for a single use case.

**Recommendation**: Option 1. Build the spec dynamically in the thin wrapper. The generic renderer stays simple.

### Line Count Impact

| Component | Before | After |
|---|---|---|
| 7 display methods | ~400 lines | ~7 thin wrappers at ~8 lines each = ~56 lines |
| Generic renderer | 0 | ~30 lines |
| Data classes | 0 | ~15 lines |
| **Total** | **~400 lines** | **~101 lines** |

**Net reduction**: ~300 lines (75%).

### Where to Place the Generic Renderer

Two options:

1. **In `eval_handler.py` itself** — simplest, no new files, keeps the refactor self-contained.
2. **In `shared/ui/`** — if other handlers also display selection tables (check for reuse).

**Recommendation**: Start in `eval_handler.py`. If other handlers adopt the pattern, extract to `shared/ui/` in a follow-up.

---

## Part 3: Implementation Roadmap

### Phase 1: eval_handler.py DRY Extraction (Low Risk)

**Priority**: Do this first — it's a straightforward mechanical refactor with no behavioral changes.

1. Add `ColumnSpec`, `TableSpec` dataclasses and `_display_table` method to `EvalHandler`
2. Convert `_display_models_table` → thin wrapper (simplest case, validates approach)
3. Convert `_display_scenarios_table` → thin wrapper (second simplest)
4. Convert `_display_gguf_models_table`, `_display_mlc_models_table`, `_display_lora_models_table` → thin wrappers (moderate: each needs a `backend.get_model_info()` call in extractor)
5. Convert `_display_training_runs_table` → thin wrapper (moderate: filesystem logic in extractor)
6. Convert `_display_checkpoints_table` → thin wrapper with dynamic column construction (most complex)
7. Run existing tests; verify no regressions

**Estimated scope**: ~150 net lines removed. All call sites (`handle()` method) remain unchanged since method names and signatures are preserved.

### Phase 2: config_validator.py Check Descriptors (Medium Risk)

**Priority**: Second — more impactful but requires careful migration to preserve behavior.

1. Add `CheckDescriptor` dataclass and `_run_check` method
2. Create descriptor module constant `BUILTIN_CHECKS` dict
3. Migrate simplest checks first:
   - `text_min_length`, `text_max_length` (pure comparison)
   - `batch_structure`, `batch_strategy` (pure comparison)
4. Migrate text checks:
   - `text_contains`, `text_contains_any`, `text_not_contains`, `text_matches`
5. Migrate tool-set checks:
   - `tool_called`, `any_tool_called`, `all_tools_called`, `tool_not_called`
6. Migrate field checks:
   - `fields_present`, `fields_min_length`, `field_equals`
7. Migrate complex checks:
   - `tool_count` (min/max dual comparison)
   - `tool_params_present` (tool-filtered field check)
   - `tool_sequence` (multi-condition)
8. Update `_register_checks` to merge `BUILTIN_CHECKS` with any remaining method-based checks
9. Remove migrated `_check_*` methods
10. Verify all behaviors.yaml-referenced checks still work

**Estimated scope**: ~300 net lines removed. External interface (`validate()`, `ValidationResult`) unchanged.

### Phase 3: Testing (Required for Both Phases)

No dedicated test files exist for either `config_validator.py` or `tuner/handlers/eval_handler.py` (only `tests/cloud/test_cloud_eval_handler.py` tests the cloud variant). This refactoring should include:

1. **config_validator.py tests**: Unit tests for each `CheckDescriptor` — test the extractor and predicate independently, then test the full `_run_check` integration. The declarative approach makes individual checks trivially testable.
2. **eval_handler.py tests**: Snapshot tests for `_display_table` output. Mock `RICH_AVAILABLE` to test both rich and plain-text paths.
3. **Regression tests**: Run existing evaluation scenarios through `ConfigDrivenValidator.validate()` with known inputs/outputs to verify behavioral equivalence.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Check behavior changes subtly during migration | Medium | High | Migrate one check at a time; test each against existing behaviors.yaml scenarios |
| Complex checks (tool_sequence) don't fit descriptor model cleanly | Low | Medium | Allow composite descriptors or keep as method-based fallback |
| Table display regressions in terminal output | Low | Low | Visual inspection + snapshot tests |
| Conditional checkpoint columns break | Low | Medium | Dynamic spec construction handles this; test both trainer types |

---

## Reasoning Chain

1. The orchestrator identified two files exceeding maintainability thresholds. I analyzed both to quantify the actual redundancy.

2. For config_validator.py: The existing `_register_checks` dict is already a registry mapping names to methods — but the methods themselves are boilerplate. The insight is that the check *logic* (extractor + predicate) is typically 2-5 lines, but it's wrapped in 10-15 lines of identical signature/append scaffolding. A `CheckDescriptor` dataclass captures only the unique parts and lets a generic runner handle the scaffolding. This aligns with the existing YAML-driven philosophy where `behaviors.yaml` already describes checks declaratively.

3. For eval_handler.py: The seven display methods are almost copy-paste. The variation points (title, columns, row extraction) are small and well-defined. A `TableSpec` + generic renderer captures the variation while eliminating the ~30-line Rich Table skeleton repeated seven times. The only complexity is `_display_checkpoints_table` with conditional columns, which is handled by building the spec dynamically rather than complicating the renderer.

4. I prioritized eval_handler.py first because it's a lower-risk mechanical refactor (same signatures, same behavior, just less code). config_validator.py comes second because it's higher-impact but requires more careful migration testing.

5. Both refactors preserve the public interface completely — no callers need to change. The config_validator.py refactor additionally preserves the YAML config format, so no test scenario files need updating.
