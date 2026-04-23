# PR #85 — Backend Implementation Review

**Reviewer**: backend-coder-reviewer
**Scope**: `Trainers/shared/callbacks/` package + three per-trainer shims
(`Trainers/{sft,kto,grpo}/src/training_callbacks.py`).
**Commit under review**: `d1cd904`
**Mode**: Review-only. No code changes.

## Verdict

Close to mergeable. Two blocking concerns around **undocumented behavior
drift in SFT** (health-check cadence + interval_time semantics) and **a
silent precedence flip in GRPO's JSONL row** that the architect doc
names but the risk matrix doesn't surface as a diff risk. Everything
else is minor/cosmetic.

---

## Blocking

### B1. SFT: health-check cadence + `interval_time` semantics silently shifted

**What changed.** Old `Trainers/sft/src/training_callbacks.py`
(pre-refactor) gated the **entire** `on_log` body on
`state.global_step % self.log_every_n_steps != 0` (old
`training_callbacks.py:123-124`). That early return meant:

- `self.last_log_time = current_time` was updated only at interval
  multiples, so `interval_time` = wall-clock gap between **printed
  rows** (which the row label `Time/5s` advertises).
- `_check_training_health(...)` was called only at interval multiples,
  so SFT warnings were throttled to 1-per-N-steps.

The new `BaseMetricsCallback.on_log`
(`Trainers/shared/callbacks/base.py:175-223`) is structured differently:

- `self.last_log_time = current_time` updates on **every** `on_log`
  (line 181) — before the interval gate.
- `self.health_checker.check(...)` fires on **every** `on_log`
  (line 203) — before the interval gate at line 205.

For KTO + GRPO this matches old behavior (they already updated every
call). For **SFT this is a behavior change**:

- `interval_time` now measures on_log-to-on_log deltas, not
  row-to-row deltas. Since HF Trainer's `logging_steps` and this
  callback's `log_every_n_steps` are independent knobs, when
  `logging_steps < log_every_n_steps` the printed `Time/5s` column
  will drop to small values. Users reading the column by its label
  will be misled.
- SFT warnings (unusual loss, high grad norm, loss-still-high-after-50)
  print more often than before. In the worst case — HF Trainer logging
  per step — warnings become per-step noise rather than per-5-step.

The architect design doc (§5 *Divergence Matrix*) lists
"elapsed/steps-per-sec/samples-per-sec calc" as moved-to-base but is
silent on these two cadence changes. The risk matrix (§9) does not
mention SFT-specific health-check cadence or `interval_time`
redefinition.

**Recommendation.** Either:

1. Gate both `self.last_log_time = current_time` and
   `self.health_checker.check(...)` on `should_write_jsonl` (i.e.,
   `log_every_write=True OR on interval`) so SFT keeps its old
   interval-only semantics while KTO/GRPO still update every call
   (they set `log_every_write=True`). This is the least-diff fix and
   preserves both trainers' prior behavior exactly.
2. Or: explicitly document this as an intentional additive change, add
   a risk-matrix entry, and add a test asserting SFT warnings fire at
   expected cadence.

Option 1 is preferred — no behavior change is the stated
non-negotiable (design doc §1).

**Evidence**: pre-refactor `Trainers/sft/src/training_callbacks.py:117-124`
(early return at top of on_log) vs `Trainers/shared/callbacks/base.py:175-206`
(gate only at line 205 after `last_log_time` and `health_checker.check`).

---

### B2. GRPO JSONL: dict-merge precedence flipped from "ours wins" to "logs wins"

**What changed.** Pre-refactor GRPO `on_log`
(old `training_callbacks.py:113-120`) built the JSONL row as:

```python
entry = dict(logs)                    # logs is the base dict
entry["step"] = int(state.global_step)
entry["timestamp"] = current_time.isoformat()
entry["interval_seconds"] = interval_time
entry["elapsed_seconds"] = ...
entry["steps_per_second"] = ...
entry["samples_per_sec"] = ...
entry.update(capacity_snapshot)        # capacity overrides
```

Precedence: `{...logs, ...our_timing (overrides), ...capacity (overrides)}`.
**Our fields win over logs.**

New `BaseMetricsCallback._write_log_row`
(`Trainers/shared/callbacks/base.py:237-246`):

```python
entry = {
    "step": int(step),
    "timestamp": current_time.isoformat(),
    self.interval_key_name: interval_time,
    "elapsed_seconds": ...,
    "steps_per_second": ...,
    "samples_per_sec": ...,
    **capacity_snapshot,
    **logs,
}
```

Precedence: `{...our_fields, ...capacity, ...logs}`.
**Logs wins over our fields.**

For SFT and KTO this matches their previous `{...ours, **logs}`
pattern, so no change. For **GRPO this flips precedence** on any key
that appears in both our fields and HF Trainer's `logs`.

**In practice the blast radius is small.** HF Trainer's standard
emitted log keys (`loss`, `learning_rate`, `epoch`, `grad_norm`) don't
collide with our field names (`step`, `timestamp`, `interval_seconds`,
`elapsed_seconds`, `steps_per_second`, `samples_per_sec`, or capacity
keys like `gpu_memory_gb`). So in the default GRPO training path no
row content actually changes.

**However**: this is a silent precedence flip on the exact format the
PR description asks reviewers to verify byte-exact
(`interval_seconds`). If a downstream consumer — or a custom
`GRPOTrainer` subclass, or TRL's own eval logs — ever puts `step`,
`timestamp`, or a capacity key into `logs`, GRPO's behavior changes
without warning. The architect doc §5 claims "moves to base …
`_append_final_training_summary` helper (byte-identical in all three)"
— which is true for the final-summary row — but doesn't surface that
the per-step row's merge precedence has changed for GRPO.

**Recommendation.** Either:

1. Keep the new unified `{...ours, **logs}` precedence (matches the
   architect's stated unification), but document it explicitly in the
   PR description + §9 risk matrix with an explicit "GRPO: per-row
   dict-merge precedence flipped from ours-wins to logs-wins; no
   current HF Trainer log key collides, so no observed diff in
   default training, but any custom trainer emitting `step` /
   `timestamp` / capacity-shaped keys in `logs` would now see those
   values land in JSONL instead of our computed ones."
2. Or: change base to preserve GRPO's old precedence with
   `{..logs, **our_fields, **capacity}` — but this would break
   SFT/KTO's prior `{...ours, **logs}` semantics.

Option 1 is the correct direction. The PR just needs to own the
change explicitly.

**Evidence**: old `Trainers/grpo/src/training_callbacks.py:113-120`
vs new `Trainers/shared/callbacks/base.py:237-246`.

---

## Minor

### M1. SFT `on_train_end` drops one leading `=` * 100 decorator line

Old SFT `on_train_end` printed a leading `"=" * 100` line before the
`train_end` JSONL write (old `training_callbacks.py:268`), then
`"\n" + "=" * 100` + banner. New `BaseMetricsCallback.on_train_end`
(`base.py:268-287`) writes the JSONL row first then goes straight to
`"\n" + "=" * 100`. Visual output now missing one row of `=` dashes
before the banner. Cosmetic.

### M2. SFT `LiveDashboardCallback` fallback banner reshape

Old SFT no-dashboard fallback banner:
`"TRAINING STARTED - {training_type.upper()}"`
(old `training_callbacks.py:455`).

New `BaseLiveDashboardCallback.on_train_begin` fallback banner:
`"{training_type_attr.upper()} TRAINING STARTED"` (`base.py:368`).

Text order flipped. Low-impact (fallback path only, no rich
dashboard). Call out if the design goal is byte-exact output
preservation.

### M3. Four modules each call `sys.path.insert(0, <repo-root>)` at import

`Trainers/shared/callbacks/base.py:21`,
`Trainers/shared/callbacks/log_suppression.py:10`, and all three
`Trainers/<trainer>/src/training_callbacks.py` shims (line 17 each)
mutate global `sys.path` at import time. Pre-refactor files did the
same (old `training_callbacks.py:18`), so no regression, but it's four
copies of the same hack in the new layout where a single package-level
`__init__.py` injection would have sufficed. Consider consolidating
into one place (e.g., `Trainers/shared/callbacks/__init__.py`).

### M4. `resolve_cloud_provider` + `CLOUD_GPU_TYPE` env lookup: not DRY

`_annotate_cloud` (`base.py:47-54`) wraps `resolve_cloud_provider` +
`CLOUD_GPU_TYPE` env read into one helper — good. But the helper is
called from **two** places (`BaseMetricsCallback.on_log:188` and
`BaseLiveDashboardCallback.on_log:384`) and the logic inside is only
two `setdefault` calls. Fine as-is, but worth a comment that both
callbacks must call it — easy to forget when adding a third callback
type.

### M5. `BaseLiveDashboardCallback.on_log` has no `log_write_swallow_errors` switch

`BaseMetricsCallback._write_log_row` (`base.py:247-252`) honors
`log_write_swallow_errors`. `BaseLiveDashboardCallback.on_log`
writes JSONL (`base.py:397-398`) with no such gate. Pre-refactor,
neither `LiveDashboardCallback` (SFT/KTO/GRPO) swallowed errors on
this write path, so no regression — but if the design intent is
symmetry, consider adding the knob to the live-dashboard path too.
GRPO's `MetricsTableCallback` swallows errors; GRPO's
`LiveDashboardCallback` does not. Intentional divergence should be
noted.

### M6. `total_epochs` default of 1 shadows `num_train_epochs` for very-early failures

`BaseLiveDashboardCallback.__init__:344` sets `self.total_epochs = 1`
as a sentinel before `on_train_begin` overwrites it with
`args.num_train_epochs`. If `on_log` were ever called before
`on_train_begin` (which HF Trainer shouldn't do), JSONL rows would
record `total_epochs=1`. Pre-refactor code had the same sentinel —
no regression. Non-issue in practice but worth a brief comment.

### M7. Epoch float cast ONLY applied to dashboard `update(...)`, not to JSONL row

The `float(logs.get("epoch", 0.0) or 0.0)` cast
(`base.py:401`) is used for `dashboard.update(epoch=...)`. The JSONL
row written at `base.py:386-396` uses `**logs` — so `epoch` in the
JSONL entry is whatever HF emitted (already a float from HF's side).
This is fine, but the PR description mentions the fix lives in "one
place"; technically it's one **call site**, and the JSONL row relies
on HF's native float emission. If HF ever changed that, the JSONL
reader would break even with the dashboard fix in place. Call out in
PR description for clarity.

---

## Future

### F1. `CheckpointMonitorCallback.on_save` is a no-op — why does it exist?

`Trainers/shared/callbacks/checkpoints.py:11-12`:
```python
def on_save(self, args, state, control, **kwargs):
    pass
```

Comment in pre-refactor file says "This is already handled by
MetricsTableCallback but we keep this for extensibility". Refactor
preserves this verbatim. Dead code — consider deleting in a follow-up.
Not this PR.

### F2. `HealthChecker._print_warnings` helper is module-private but used from subclasses

`Trainers/shared/callbacks/health_checks.py:26` — the leading
underscore suggests private, yet both `SFTHealthChecker.check` and
`KTOHealthChecker.check` call it from the module scope (same file, so
it's legal Python). Fine as-is. If `health_checks.py` is ever split,
rename to `_grad_norm_warning` / `_print_warnings` at module scope is
the same file — no actual boundary crossed today.

### F3. `NoOpHealthChecker.check` returns `None` explicitly

`health_checks.py:88`: `return` — redundant in a function whose
default return is `None`. Cosmetic.

### F4. `resolve_cloud_provider` is exported but only `_annotate_cloud` uses it

`__init__.py:35` and `base.py:39` export `resolve_cloud_provider` as
public API. Nothing outside the package (in the shim modules or the
train_*.py callers) imports it today. Fine to expose for future use;
just noting.

---

## Verification Details

### Task-description check items (✓ / ✗)

- ✓ **Code quality**: callbacks are idiomatic Python — class attrs as
  configuration knobs (`log_every_write`, `log_write_swallow_errors`,
  `print_checkpoint_on_save`, `print_completion_banner`,
  `interval_key_name`), strategy-pattern `HealthChecker`, and thin
  subclass hooks (`_print_header`, `_print_row`, `_dashboard_metrics`,
  `_fallback_row`). Typing is consistent (`Optional[str]`,
  `Dict[str, Any]`, `List[Dict[str, Any]]`). Naming is clear.

- ✓ **Shim thinness**: all three shims are ≤113 lines. SFT shim is
  the thickest (113 lines) because of the `training_type` ctor param
  plumbing; KTO (102) and GRPO (126) are in the same range. No
  business logic leaked into the shims — they only (a) subclass and
  set class attrs, (b) override `_print_header` / `_print_row` /
  `_dashboard_metrics` / `_fallback_row`, (c) re-export hoisted
  names. Good.

- ⚠ **Error handling (`log_write_swallow_errors`)**: correct
  per-trainer — GRPO sets True (swallow), SFT/KTO default to False
  (raise). Matches pre-refactor behavior. *But* see M5 — the
  `LiveDashboardCallback` write path has no equivalent switch, and
  the JSONL write there is unguarded on all three trainers.

- ⚠ **Dict-merge precedence `{...ours, **logs}` = logs wins**:
  implemented correctly and uniformly in both
  `BaseMetricsCallback._write_log_row` and
  `BaseLiveDashboardCallback.on_log`. **But for GRPO this is a
  precedence flip** — see B2.

- ✓ **Import cleanliness**: no circular imports. Smoke test:
  `from Trainers.shared.callbacks import *` + each of the three shim
  imports succeed cleanly.

- ⚠ **Silent drops from old files**: see M1 (SFT banner decorator
  line) and M2 (SFT fallback banner reshape). Minor, cosmetic, but
  technically silent output diffs.

- ✓ **`interval_seconds` GRPO byte-exact**: GRPO shim sets
  `interval_key_name = "interval_seconds"` at
  `Trainers/grpo/src/training_callbacks.py:35`; base dereferences
  `self.interval_key_name` at `base.py:240`. Confirmed byte-exact.

- ✓ **Epoch float cast — one place only**: located at
  `base.py:401` inside `BaseLiveDashboardCallback.on_log`. No other
  call site casts epoch. The `MetricsTableCallback` path prints
  `epoch` via `logs.get("epoch", 0.0)` (`sft/src/training_callbacks.py:66`)
  which stays as HF's native type — this is the console-row side,
  independent of the dashboard bug. Per M7 above, the JSONL row
  relies on HF's emission.

### Files reviewed

- `Trainers/shared/callbacks/__init__.py`
- `Trainers/shared/callbacks/base.py`
- `Trainers/shared/callbacks/health_checks.py`
- `Trainers/shared/callbacks/lr_schedules.py`
- `Trainers/shared/callbacks/checkpoints.py`
- `Trainers/shared/callbacks/log_suppression.py`
- `Trainers/sft/src/training_callbacks.py`
- `Trainers/kto/src/training_callbacks.py`
- `Trainers/grpo/src/training_callbacks.py`
- `docs/architecture/training-callbacks-refactor.md` (for context)
- pre-refactor versions of all three `training_callbacks.py` files
  (extracted via `git show d1cd904^:...`)

### Call sites confirmed unbroken

- `Trainers/sft/train_sft.py:51-57` → imports
  `MetricsTableCallback`, `CheckpointMonitorCallback`,
  `LiveDashboardCallback`, `suppress_training_logs`,
  `DASHBOARD_AVAILABLE`. All five are re-exported by the SFT shim.
- `Trainers/kto/train_kto.py:50` → imports
  `LiveDashboardCallback`, `MetricsTableCallback`,
  `CheckpointMonitorCallback`, `TwoStageLRCallback`,
  `DASHBOARD_AVAILABLE`, `RICH_AVAILABLE`. All six re-exported.
- `Trainers/grpo/train_grpo.py:62` and
  `Trainers/grpo/train_env_grpo.py:34` → import
  `LiveDashboardCallback`, `MetricsTableCallback`,
  `DASHBOARD_AVAILABLE`, `RICH_AVAILABLE`. All four re-exported.

No caller edits required. ✓

---

## Summary for Lead

- **B1 and B2** should be resolved before merge — either by tightening
  the `on_log` flow in `base.py` (preferred) or by explicitly
  documenting the drift in the PR description + risk matrix.
- **M1-M7** are minor cosmetic / ergonomic comments; none are merge
  blockers.
- **F1-F4** are follow-up cleanup, not in scope for this PR.
- The bug fix itself (epoch float cast) is correctly implemented at
  exactly one location in `BaseLiveDashboardCallback.on_log`.
- The refactor is structurally sound: strategy pattern for health
  checks, class-attr knobs for cadence/banner/interval-key
  divergence, thin per-trainer subclasses, and clean re-exports keep
  all four call sites untouched.
