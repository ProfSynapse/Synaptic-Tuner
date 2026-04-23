# PR #85 — Architect Review

**Scope**: Design-conformance review of the DRY callback refactor against
`docs/architecture/training-callbacks-refactor.md`. Implementation review
and test coverage are the sibling reviewers' charge.

**Commit**: `d1cd904 fix(trainers): epoch counter in TUI dashboard + DRY callback refactor`

**Verdict**: **Blocking** — two real SFT behavior changes slipped past the
"zero behavior change" non-negotiable. One fix is a two-line reorder; the
other is a one-class-attribute addition. Refactor shape is otherwise clean
and matches the design doc.

---

## Findings

### BLOCKING-1 — SFT health check now fires on every on_log (was Nth)

**Location**: `Trainers/shared/callbacks/base.py:175-206`

**What changed**: The original SFT `on_log` (see `git show main:Trainers/sft/src/training_callbacks.py`, lines 117-124) early-returned at
`if state.global_step % self.log_every_n_steps != 0: return` **before**
`_check_training_health` and `_save_metrics_to_file`. The new
`BaseMetricsCallback.on_log` computes the capacity snapshot, writes JSONL
(now gated via `should_write_jsonl`), calls
`self.health_checker.check(...)` at line 203, and **only then** checks the
modulo for the print-row gate at line 205.

For SFT, `log_every_write = False`, so JSONL cadence is preserved (modulo
still gates `should_write_jsonl`). But `SFTHealthChecker.check` now runs
on **every** HF Trainer log callback instead of every Nth, producing
N× more grad-norm / loss-still-high / reward-collapse warnings.

**Why this is blocking**: Design doc §7 ("No-Public-API-Change
Confirmation") and the dispatch non-negotiable ("zero behavior change")
both required preserving observable behavior. SFT health-warning
frequency is observable and used by the operator during training.

**Fix**: Move the health-check call below the modulo gate, OR gate it on
the same modulo:

```python
# replace lines 203-206 with:
if state.global_step % self.log_every_n_steps != 0:
    return
self.health_checker.check(logs, state.global_step, args.max_grad_norm)
```

This preserves SFT behavior exactly. For KTO it also matches the
original (original KTO called `_check_training_health` before the modulo
gate — so KTO's health check *should* remain pre-gate). Resolution: make
the gating configurable, e.g. class attr `health_check_every_on_log: bool`
(KTO/GRPO: True; SFT: False, or absent since NoOp). Simpler alternative:
keep pre-gate for all trainers and move SFT's rule into
`SFTHealthChecker` with an internal modulo guard — but that leaks the
log_every_n_steps knob into the health checker. The class-attr approach
is cleaner and consistent with the design doc's divergence-via-class-attrs
pattern.

---

### BLOCKING-2 — SFT `last_log_time` / `interval_time` semantics changed

**Location**: `Trainers/shared/callbacks/base.py:180-181`

**What changed**: Original SFT assigned `self.last_log_time = current_time`
**after** the modulo early-return (line 129 in the old file), so
`interval_time` measured time between **logged rows**. New base.py
assigns `self.last_log_time = current_time` on every on_log (line 181),
before the modulo gate. Consequence for SFT: the `interval_time` /
`Time/5s` column and the `interval_time` JSONL field now reflect time
since the **previous on_log callback**, not since the previous logged
row. On the N-1 off-boundary steps the value is silently reset.

Net effect: printed `Time/5s` and JSONL `interval_time` for SFT become
meaningless (effectively zero-ish or very small), because the previous
on_log tick happened 1 training step ago, not `log_every_n_steps` steps
ago.

KTO's original also updated `last_log_time` unconditionally — so KTO
semantics are preserved. GRPO same. This is SFT-only.

**Why this is blocking**: silently breaks an operator-facing metric
(interval time and samples-per-5s). Small code change, meaningful
display regression.

**Fix**: Gate the `last_log_time` reassignment on the same `should_write_jsonl`
or print-modulo that SFT previously used. Cleanest: make it track the
print-row cadence (modulo), so `interval_time` always reflects
row-to-row wall time. Example:

```python
current_time = datetime.now()
interval_time = (current_time - self.last_log_time).total_seconds() if self.last_log_time else 0.0
# (do NOT reassign last_log_time here yet)
elapsed = ...
...
should_write_jsonl = self.log_every_write or (state.global_step % self.log_every_n_steps == 0)
if should_write_jsonl:
    self._write_log_row(..., interval_time=interval_time, ...)
if state.global_step % self.log_every_n_steps != 0:
    return
self.last_log_time = current_time  # moved: only update at print boundary
# ...print row...
```

Caveat: KTO/GRPO write JSONL every on_log (`log_every_write=True`), so
their `interval_time` value *does* need to update every on_log to stay
meaningful. Two options: (a) a second class attr
`interval_time_updates_every_on_log: bool` (KTO/GRPO True, SFT False),
or (b) update `last_log_time` inside `_write_log_row` when
`log_every_write` is True, and at the print-row boundary when False.
Option (a) is more explicit and matches the design doc's pattern.

---

### MINOR-1 — `BaseMetricsCallback.on_train_end` double-banner preserved

**Location**: `Trainers/shared/callbacks/base.py:279-280`

```python
print("=" * 100)
print("\n" + "=" * 100)
```

Preserved byte-identically from original SFT (which had the same bug).
Not introduced by the refactor but now visible as shared code. Safe to
leave for this PR (strict byte parity on that path was an implicit goal);
flag for a follow-up cleanup.

---

### MINOR-2 — `BaseLiveDashboardCallback` no-dashboard start banner format drift

**Location**: `Trainers/shared/callbacks/base.py:367-368`

Original SFT (dashboard fallback) printed `"SFT TRAINING STARTED"` by
concatenating `training_type.upper() + " TRAINING STARTED"`. New base
does the same — **but** for GRPO the original printed `"GRPO TRAINING STARTED"`
via the same template, so parity holds. I'm flagging this only because
the original KTO LiveDashboard no-fallback path (original
`Trainers/kto/src/training_callbacks.py:206` region) prints
`"KTO TRAINING STARTED"` — verify in the coder-review that KTO parity is
intact. Not blocking unless that check fails.

---

### MINOR-3 — `on_train_end` elapsed-time format inconsistency between base classes

**Location**: `Trainers/shared/callbacks/base.py:283` vs `:436`

`BaseMetricsCallback.on_train_end` uses `format_time(elapsed)`.
`BaseLiveDashboardCallback.on_train_end` uses an inline
`elapsed // 3600 ... % 60` expression. Both are preserved from originals,
but the sibling-base inconsistency is a DRY smell. Follow-up: point both
at `format_time()`.

---

### MINOR-4 — `BaseLiveDashboardCallback` JSONL write lacks `log_write_swallow_errors`

**Location**: `Trainers/shared/callbacks/base.py:397-398`

```python
with open(self.log_file, "a", encoding="utf-8") as f:
    f.write(json.dumps(log_entry) + "\n")
```

No try/except. Original per-trainer LiveDashboard callbacks also had no
try/except here, so byte-parity holds. Design inconsistency only: the
sibling `_write_log_row` on `BaseMetricsCallback` respects
`log_write_swallow_errors`; this path does not. If GRPO LiveDashboard
is ever invoked in a read-only FS / quota-exhausted environment, it will
crash where GRPO's `MetricsTableCallback` would have silently swallowed.
Safe to defer; flag for a follow-up unification.

---

### MINOR-5 — Per-trainer module docstrings call themselves "shims"

**Location**: `Trainers/{sft,kto,grpo}/src/training_callbacks.py:1-7`

Docstrings say "Per-trainer [SFT|KTO|GRPO] callback shims." CLAUDE.md
rule: **"No backward-compat shims — This codebase has no external
consumers..."** These are NOT backward-compat shims — they are the
canonical concrete-class home (per design doc §7). Rename docstring to
something like "Per-trainer concrete subclasses and re-exports" to avoid
confusing future readers who may grep for "shim" during cleanup.

---

### FUTURE-1 — `sys.path.insert(...)` repeated in 4 files

`base.py:21`, `log_suppression.py`, and the three per-trainer modules
all do `sys.path.insert(0, str(Path(__file__).resolve().parents[N]))`.
Consistent with the repo's existing convention, but worth consolidating
via a single bootstrap module or a proper `pyproject.toml` package entry
in a future refactor. Not in scope for this PR.

---

### FUTURE-2 — HealthChecker strategy is good; consider a no-op subclass sentinel

The `NoOpHealthChecker` is used as both the default (in
`BaseMetricsCallback.__init__`) and the explicit GRPO assignment. If the
default covers GRPO, the explicit assignment in the GRPO module can go.
Purely stylistic; leave as-is for clarity of intent.

---

## Design-Doc Conformance — PASS items

- **§1 Package Layout**: `Trainers/shared/callbacks/{base,health_checks,lr_schedules,checkpoints,log_suppression,__init__}.py` — matches exactly.
- **§3 Class-attr divergence knobs**: `log_every_write`, `log_write_swallow_errors`, `interval_key_name`, `default_output_dir`, `start_banner`, `completion_banner`, `print_checkpoint_on_save`, `print_completion_banner` — all present on `BaseMetricsCallback` and used correctly by the three subclasses.
- **§4 HealthChecker Strategy**: ABC with single `check(logs, step, max_grad_norm)` method, three concrete subclasses (`SFTHealthChecker`, `KTOHealthChecker`, `NoOpHealthChecker`), shared `_grad_norm_warning` helper. Minimal and clean.
- **§6 Cloud-provider resolution**: Unified on env-first + `getattr(args, "cloud_provider", None)` fallback via `resolve_cloud_provider()`. Matches the design-doc callout; KTO and GRPO correctly inherit the additive env-first behavior that the design-doc §6 flagged as an intentional minor behavior change.
- **§7 No-public-API-change**: Per-trainer concrete classes and re-exports preserved at `Trainers/{sft,kto,grpo}/src/training_callbacks`. The four documented call sites (`train_sft.py`, `train_kto.py`, `train_grpo.py`, `train_env_grpo.py`) can continue importing by existing paths.
- **§8 Migration sequence**: SFT-first / KTO-second / GRPO-third order evident; each per-trainer shim is thin and byte-faithful on its specific divergence axes (modulo the two BLOCKING findings above).

---

## Summary for Team Lead

- 5 non-blocking design smells, 2 future cleanups.
- **2 blocking SFT behavior regressions** that the refactor introduced
  by "unifying" what was actually divergent between SFT and KTO/GRPO.
  Both are small local fixes on `base.py`.
- Recommended remediation: add one or two class attributes
  (`health_check_every_on_log`, `interval_time_updates_every_on_log`) —
  consistent with the class-attr-divergence pattern already used in the
  refactor — and move the two statements below the appropriate gates.
  Coder's lift is small; tests should grow to cover the SFT Nth-step
  health-check and interval-time cadence so this regression doesn't
  recur.

---

## Open Questions for Team Lead

1. Accept BLOCKING-1 and BLOCKING-2 as must-fix-before-merge? (My
   recommendation: yes, both. Each is a ~5-line fix plus a test.)
2. MINOR-1 (double-banner bug preserved from original): fix in this PR
   or leave for follow-up? Byte-parity goal suggests leave; hygiene
   suggests fix while the code is open.
3. MINOR-5 (docstrings say "shim"): rename in this PR or leave?
   Low-risk one-liner per file.
