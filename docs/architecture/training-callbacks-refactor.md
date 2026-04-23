# Training Callbacks DRY Refactor — Architecture

Pure code-organization refactor of the three per-trainer `training_callbacks.py`
modules (sft, kto, grpo) into a shared base + per-trainer strategies.

- **Non-negotiable**: zero behavior change. No public API change on the four
  callback classes or the `DASHBOARD_AVAILABLE` / `RICH_AVAILABLE` /
  `suppress_training_logs` symbols. Existing call sites in
  `Trainers/{sft,kto,grpo}/train_*.py` import unchanged.
- **Scope**: `Trainers/{sft,kto,grpo}/src/training_callbacks.py` + new
  `Trainers/shared/callbacks/` package.
- **Out of scope**: `shared/ui/dashboard.py`, `Trainers/shared/ui/training_progress.py`.

## 1. Target Package Layout

| Module | Responsibility / Exports |
|---|---|
| `Trainers/shared/callbacks/__init__.py` | Public re-exports + `DASHBOARD_AVAILABLE` / `RICH_AVAILABLE` |
| `base.py` | Lifecycle skeleton (shared timing/capacity/cloud-provider capture, JSONL writing, symlink mgmt, final-summary footer). Exports `BaseMetricsCallback`, `BaseLiveDashboardCallback`, `append_final_training_summary`, `resolve_cloud_provider`, `build_log_entry` |
| `health_checks.py` | `HealthChecker` ABC + `SFTHealthChecker` / `KTOHealthChecker` / `NoOpHealthChecker` |
| `lr_schedules.py` | `TwoStageLRCallback` (hoisted, byte-identical across sft + kto) |
| `checkpoints.py` | `CheckpointMonitorCallback` (hoisted, byte-identical) |
| `log_suppression.py` | `suppress_training_logs` (currently sft-only) |

Each `Trainers/<trainer>/src/training_callbacks.py` shrinks to:
- 2 thin concrete subclasses (`MetricsTableCallback`, `LiveDashboardCallback`) binding per-trainer strategies.
- Re-exports of hoisted classes/symbols under existing names (see §7).

## 2. `BaseMetricsCallback` Contract

Shared `TrainerCallback` subclass owning the table-output and JSONL-logging
lifecycle. Per-trainer classes subclass it and inject strategies — they do
**not** override `on_train_begin` / `on_train_end`.

**Class attributes (overridable)**: `default_output_dir: str` (e.g. `"./sft_output"`),
`log_every_write: bool` (kto/grpo: True; sft: False), `training_type_label: str`
(header/banner name), `log_write_swallow_errors: bool` (grpo: True; others: False).

**Constructor**: `(log_every_n_steps=5, output_dir=None, previous_log_entries=None)`.
`previous_log_entries` is accepted by all trainers; GRPO currently omits it, base
adds uniformly (default `None` no-ops — no behavior change).

**Lifecycle (base-owned, subclasses must not override)**:

- `on_train_begin`: set `start_time` / `last_log_time`; `reset_capacity_peaks(torch)`;
  recreate `training_latest.jsonl` symlink (swallow OSError/NotImplementedError);
  print start banner using `training_type_label`.
- `on_log`: compute elapsed / steps-per-sec / samples-per-sec / interval_time;
  build capacity_snapshot via `capture_runtime_capacity_snapshot(torch)`; annotate
  with `resolve_cloud_provider(args)` + `CLOUD_GPU_TYPE` env. Then delegate:
  (1) `_write_log_row(entry)` gated by `log_every_write` vs. interval;
  (2) `self.health_checker.check(logs, step, args.max_grad_norm)`;
  (3) every Nth step: `_print_row(...)` (subclass hook).
- `on_save`: print checkpoint line (identical across trainers).
- `on_train_end`: `append_final_training_summary(...)`; print completion banner.

**Subclass hooks**: `_print_header(self)`, `_print_row(self, *, step, state, logs,
capacity_snapshot, interval_time, samples_per_sec, eta)`, `health_checker: HealthChecker`
(set by subclass `__init__`).

## 3. `BaseLiveDashboardCallback` Contract

Wraps `shared.ui.LiveDashboard` (when available) with the same lifecycle shape.
Subclass provides `training_type` and a `_dashboard_metrics(logs, capacity_snapshot)`
method returning the kwargs passed to `self.dashboard.update(...)`.

| Hook | SFT | KTO | GRPO |
|---|---|---|---|
| `training_type` | `self.training_type` (ctor param, defaults `"sft"`) | `"kto"` | `"grpo"` |
| `_dashboard_metrics` | `loss`, `learning_rate`, `kl`, `margin` (rewards/margins), `gpu_memory_gb` | same + `kl` falls back to `logps/rejected` | `reward`/`reward_std`/`kl_penalty`/`advantage` (with multi-key fallbacks) |
| `_fallback_row` | prints step/loss/lr/gpu | step/loss/margin/gpu | step/loss/reward |

SFT keeps its `training_type` constructor arg (currently `"sft"`/`"kto"`); base exposes
it via subclass default, preserving the signature.

## 4. `HealthChecker` Strategy

```python
class HealthChecker(ABC):
    @abstractmethod
    def check(self, logs: dict, step: int, max_grad_norm: float | None) -> None: ...
```

Concrete classes print identical-format warning blocks (`"!" * 100` bracket lines +
`"Consider: ..."` footer), preserving exact output today:

- `SFTHealthChecker`: loss-range, grad-norm-clip, loss-high-after-50-steps.
- `KTOHealthChecker`: loss-range, KTO margin < −1.0, reward-collapse (both ≈0 after
  step 10), grad-norm-clip.
- `NoOpHealthChecker`: GRPO — no checks today.

Grad-norm-clip logic lives in a shared `_grad_norm_warning(logs, max_grad_norm)`
helper in `health_checks.py` used by both SFT and KTO checkers.

## 5. Divergence Matrix — Stays Per-Trainer vs. Moves to Base

**Moves to base** (identical today across ≥2 trainers): start banner + JSONL
symlink-to-latest (swallow OSError/NotImplementedError); elapsed/steps-per-sec/
samples-per-sec calc; `capture_runtime_capacity_snapshot` + cloud-provider +
cloud-gpu-type annotation; `append_final_training_summary` helper (byte-identical
in all three); `_format_time` helper (identical sft + kto); completion banner
shape; `on_save` checkpoint-line print; `TwoStageLRCallback` + `CheckpointMonitorCallback`
(both byte-identical sft + kto; grpo doesn't use them — unchanged).

**Stays per-trainer** (concrete subclass / strategy / class attr):

- Table header + row format (SFT: loss/LR/gradnorm/epoch/gpu/samples/eta;
  KTO: loss/LR/chosen/reject/margin/gpu/samples/eta; GRPO: compact loss/reward/LR/gpu).
- Dashboard metric extraction (`rewards/margins` vs. `reward`/`kl_penalty`/`advantage`
  multi-key lookups).
- Health-check rules (via strategy, §4).
- `default_output_dir` and completion banner label (per-trainer class attrs).
- **JSONL write cadence**: KTO + GRPO write every `on_log` (`log_every_write=True`);
  SFT writes only at `log_every_n_steps` boundary (`log_every_write=False`).
  Preserved via class attr — single knob, grep-visible.

## 6. Cloud-Provider Resolution — Unify on SFT's Helper

**Current divergence**:

| Trainer | Source |
|---|---|
| sft | `_resolve_cloud_provider(args)` — env `CLOUD_PROVIDER` then `getattr(args, "cloud_provider", None)` |
| kto | `args.cloud_provider` directly |
| grpo | `os.environ.get("CLOUD_PROVIDER")` only (no args fallback) |

**Decision**: base uses sft's helper (`resolve_cloud_provider(args)` —
env-first, then `getattr(args, "cloud_provider", None)`). All three trainers
converge on it.

> **Intentional, additive behavior change — not an accidental diff.** Kto
> and grpo gain a fallback path they did not have before. This is called out
> explicitly so PR reviewers do not treat it as an unintended regression.
> Precedence is fixed: env-first, args fallback (via `getattr(..., None)`).
> `cloud_provider` is metadata attached to JSONL log rows for downstream
> experiment tracking — never a control-flow input.

**Behavior-change audit**:
- **sft**: unchanged — helper originates here.
- **kto**: *additive* — previously `args.cloud_provider` only; now env-first,
  args fallback. In practice `CLOUD_PROVIDER` env is set by HF-Jobs + local
  docker runners that also set `args.cloud_provider`; env and args are
  expected to agree. If they ever disagreed, env now wins — same-or-stricter
  since `cloud_provider` is metadata, not control flow. **Flag for test
  engineer**: add a kto test asserting env-wins-over-args-when-set.
- **grpo**: *additive* — previously `os.environ.get("CLOUD_PROVIDER")` only;
  now gains `getattr(args, "cloud_provider", None)` fallback when env is
  unset. Current behavior: None when env unset. New behavior: None unless
  args carries it (GRPO's `args` is TRL `GRPOConfig` which may or may not
  have the attribute — `getattr(..., None)` default makes this safe).
  Strictly an improvement in metadata completeness, no control-flow change.

### 6a. JSONL dict-merge precedence — preserved per-trainer (correction note)

> **Correction to the coder's original HANDOFF.** An earlier handoff
> note said the refactor would "unify on SFT/KTO style
> `{...our_fields, **logs}` (logs wins)" for all three trainers. That
> was based on a misread of GRPO's pre-refactor code. Pre-refactor
> GRPO's `on_log` built the JSONL row as
> `entry = dict(logs); entry[k] = v; entry.update(capacity)` — i.e.,
> logs is the base dict and **our fields + capacity override logs on
> key collisions** (fields-win). Pre-refactor SFT and KTO built the
> dict as `{**our_fields, **capacity, **logs}` — logs-win. Flipping
> base to a single unified precedence would have regressed whichever
> two trainers were on the other side.
>
> **Resolution**: per-trainer class attr
> `BaseMetricsCallback.fields_win_on_collision: bool = False` (default
> matches SFT/KTO majority). GRPO's `MetricsTableCallback` overrides
> to `True`. `_write_log_row` branches on the attr and emits the
> corresponding spread order. All three trainers now preserve their
> pre-refactor JSONL row content byte-exact.
>
> Practical blast radius of either direction is small — HF Trainer's
> standard emitted log keys (`loss`, `learning_rate`, `epoch`,
> `grad_norm`) don't collide with our field names (`step`,
> `timestamp`, `interval_time` / `interval_seconds`,
> `elapsed_seconds`, `steps_per_second`, `samples_per_sec`,
> `gpu_memory_gb`). But custom trainers or future TRL key additions
> could collide, so preservation matters.

### 6b. SFT cadence — interval gate restored via class attrs (correction note)

> **Correction to the coder's original implementation.** The first cut
> of `BaseMetricsCallback.on_log` unconditionally updated
> `self.last_log_time` and called
> `self.health_checker.check(...)` on every `on_log` invocation. This
> matches KTO and GRPO's pre-refactor behavior but regresses SFT,
> which gated the entire `on_log` body on
> `state.global_step % log_every_n_steps != 0` (early return at the
> top of the function). As a result, in the first cut: SFT's
> printed-row `Time/5s` column silently redefined from "time between
> printed rows" to "time between on_log calls", and SFT health-check
> warnings fired more often than before.
>
> **Resolution**: two per-trainer class attrs
> `BaseMetricsCallback.health_check_every_on_log: bool = True` and
> `BaseMetricsCallback.interval_time_updates_every_on_log: bool = True`
> (defaults match KTO/GRPO). SFT's `MetricsTableCallback` overrides
> both to `False`. `on_log` gates the two lines on
> `(attr or on_interval)` so SFT only fires them at interval
> multiples — matching its pre-refactor early-return behavior —
> while KTO/GRPO retain every-call behavior.

## 7. No-Public-API-Change Confirmation

Callers import from the **per-trainer module**, not from a shared package:

- `Trainers/sft/train_sft.py:51–54` →
  `from src.training_callbacks import MetricsTableCallback, CheckpointMonitorCallback, LiveDashboardCallback`
- `Trainers/kto/train_kto.py:50` →
  `from src.training_callbacks import LiveDashboardCallback, MetricsTableCallback, CheckpointMonitorCallback, TwoStageLRCallback, DASHBOARD_AVAILABLE, RICH_AVAILABLE`
- `Trainers/grpo/train_grpo.py:62` + `Trainers/grpo/train_env_grpo.py:34` →
  `from src.training_callbacks import LiveDashboardCallback, MetricsTableCallback, DASHBOARD_AVAILABLE, RICH_AVAILABLE`

> **Critical — per-trainer re-export required.** A package-level
> `Trainers/shared/callbacks/__init__.py` re-export is **not sufficient**.
> Every name these callers reference must be importable from
> `Trainers/<trainer>/src/training_callbacks` at its original path. Coder
> **must not** delete or empty `Trainers/{sft,kto,grpo}/src/training_callbacks.py`
> or merely point to the shared package from callers. The per-trainer module
> stays; it becomes thin.

This is not a "backward-compat shim" (which CLAUDE.md forbids). The per-trainer
module is the *canonical* home for these class names: sft's `MetricsTableCallback`
is a different class from kto's (different health checker, different row format).
The shared package provides the base machinery; the per-trainer module is where
the concrete, trainer-specific class lives.

After refactor, each `Trainers/<trainer>/src/training_callbacks.py`:

1. **Defines** (subclasses `BaseMetricsCallback` / `BaseLiveDashboardCallback`,
   wiring in the trainer's `HealthChecker`, `default_output_dir`, `log_every_write`,
   `training_type_label`, `_print_header` / `_print_row`, and `_dashboard_metrics` /
   `_fallback_row`):
   `MetricsTableCallback`, `LiveDashboardCallback`.
2. **Re-exports from the shared package** (so per-trainer import path is preserved):
   `TwoStageLRCallback` (sft + kto), `CheckpointMonitorCallback` (sft + kto),
   `DASHBOARD_AVAILABLE`, `RICH_AVAILABLE`, and `suppress_training_logs` (sft only).

The result: every symbol in each caller's import line still resolves from
`Trainers/<trainer>/src/training_callbacks` at unchanged signatures, with no
changes to the train_*.py files.

## 8. Migration Sequence

Each step is independently testable; later steps are gated by the previous
step's tests being green.

1. **Land base package, no imports yet.** Create
   `Trainers/shared/callbacks/{__init__.py,base.py,health_checks.py,lr_schedules.py,checkpoints.py,log_suppression.py}`.
   Base classes compile; strategies compile; nothing imports them yet. Gate:
   package imports cleanly under `python -c "from Trainers.shared.callbacks import *"`.
2. **Port SFT first — sft's behavior is unchanged by the refactor.** SFT is the
   richest file, and since the cloud-provider helper originated in sft, porting
   it first exercises the full base-class surface with zero behavior-change risk
   (any regression must be a refactoring bug, not the intentional additive
   change from §6). Replace `Trainers/sft/src/training_callbacks.py` body with
   thin subclasses + re-exports (per §7 — do **not** empty the module). Gate:
   `train_sft.py` imports succeed; smoke-run a 5-step SFT training; verify log
   file + symlink + banner + health-warning output match pre-refactor.
3. **Port KTO — introduces the additive cloud-provider change.** Gate: kto
   smoke-run; verify JSONL-every-on_log cadence preserved (diff pre/post sample
   log files on same 5-step run); verify `CLOUD_PROVIDER` env value appears in
   JSONL rows when env is set (new behavior per §6).
4. **Port GRPO — introduces the additive args fallback.** Gate: `train_grpo.py`
   + `train_env_grpo.py` imports succeed; grpo smoke-run; verify no health-check
   output (`NoOpHealthChecker`); verify `cloud_provider` appears when only args
   (not env) carries it (new behavior per §6).
5. **Delete duplicated helpers.** The per-trainer `_append_final_training_summary`
   / `_resolve_cloud_provider` (sft only) / `_format_time` are now dead. Gate:
   grep confirms zero callers outside the shared package.

Rollback unit: each step is one or two files; `git revert` restores behavior.

## 9. Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Import path breakage — callers use `from src.training_callbacks import …` | Low | High | Per-trainer module must define concrete subclasses + re-export hoisted names (see §7 — package-level `__init__.py` alone is insufficient); verified 4 call sites, all use same-file imports |
| Symlink-behavior regression on WSL | Low | Low | Base preserves identical try/except OSError+NotImplementedError pattern; test step 2 checks symlink on WSL |
| Dashboard callback sequence (on_log vs on_save order with LiveDashboard) | Low | Medium | Base does not reorder hooks; pure extraction. Test engineer: run dashboard side-by-side diff on same training_latest.jsonl |
| KTO health-check output format drift (spaces, punctuation) | Medium | Low | `HealthChecker.check` prints identical bracket lines + "Consider:" footer; snapshot-test output |
| Cloud-provider resolution: KTO gains env-first fallback (**intentional additive change**, guarded by args precedence, §6) | Low | Low | Called out explicitly so PR reviewers do not flag as accidental; test engineer adds env-wins-over-args assertion |
| Cloud-provider resolution: GRPO gains args fallback (**intentional additive change**, guarded by args precedence, §6) | Low | Low | Called out explicitly; previously `None` when env unset, now `getattr(args, "cloud_provider", None)`; improvement-only for metadata completeness |
| JSONL-write cadence silently flipped for a trainer | Low | High | `log_every_write` class attr is the single knob; step-2/3/4 gates diff log files before/after |
| `previous_log_entries` semantics for GRPO (new param) | Low | Low | Default `None` is a no-op; preserves current behavior |
| `suppress_training_logs` only lived in sft — don't accidentally break kto/grpo | Low | Low | Hoist to `log_suppression.py`; sft re-exports; kto/grpo don't import it, no-op for them |
| `CLOUD_GPU_TYPE` env read — currently in all three files | Low | Low | Move to base alongside cloud-provider capture — identical code |

## 10. Open Questions (for Coder)

- GRPO's `MetricsTableCallback` swallows all log-write exceptions (`try/except Exception: pass`);
  SFT + KTO raise. Preserve via `log_write_swallow_errors` class attr (default False;
  GRPO overrides True) — **recommendation: preserve**, zero behavior change.
- SFT's `LiveDashboardCallback` `training_type="sft"` ctor param is actually used
  to switch dashboard rendering mode. Keep on sft's subclass only; don't propagate
  to kto/grpo subclasses — **recommendation: keep as-is**.
