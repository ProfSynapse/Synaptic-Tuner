# PR #85 — Test Coverage Review (test-engineer)

**PR**: `fix(trainers): epoch counter in TUI dashboard + DRY callback refactor`
**Branch**: `fix/tui-epoch-counter` → `main` (+1538 / −1291, net −383 LOC + tests + docs)
**Reviewer scope**: test quality, coverage rigor, regression scope, mock fairness, edge cases.

---

## Headline

The PR bundles **two changes**: (1) a one-line bug fix for the TUI epoch
counter (`int(logs.get('epoch', 0))` → `float(logs.get("epoch", 0.0) or 0.0)`
at `Trainers/shared/callbacks/base.py:401`) and (2) a DRY callback refactor
that extracts shared lifecycle into `Trainers/shared/callbacks/`.

**The 21 new tests in `tests/trainers/test_callback_refactor.py` cover the
refactor's 4 stated uncertainties. They do NOT cover the headline bug fix.**
This is the dominant finding of the review.

---

## Findings

### Blocking

#### B1. The epoch-fix regression guard is missing

The PR title promises an epoch-counter fix, the commit message names it, and
the PR body's test plan moves byte-level confirmation to a manual post-merge
SFT run. But no automated test will fail if a future refactor reintroduces
`int(logs.get("epoch", ...))` on the path that feeds
`LiveDashboard.update(epoch=…)`.

Concrete evidence:

- Pre-fix, at `HEAD~1`, all three trainers (`sft`, `kto`, `grpo`) passed
  `epoch=int(logs.get('epoch', 0))` into `self.dashboard.update(...)`
  (see `Trainers/{sft,kto,grpo}/src/training_callbacks.py:500/475/283` at
  the parent commit).
- Post-fix, `base.py:401` uses `float(logs.get("epoch", 0.0) or 0.0)`.
- `LiveDashboard.update(epoch: float = None)` at `shared/ui/dashboard.py:157`
  stores the value as `self.metrics.epoch: float = 0.0` and formats it via
  `f"{self.epoch:.2f}/{total_epochs}"` at `dashboard.py:91`. Passing `int(0)`
  when real epoch is `0.34` renders `0.00/3` until the integer flips to `1`.
  That is the user-visible bug.
- My `TestSftJsonlShape::test_sft_row_has_required_keys_and_types` asserts
  `row["epoch"] == 0.25` — but this tests `MetricsTableCallback.on_log` →
  JSONL, NOT `LiveDashboardCallback.on_log` → `dashboard.update(epoch=…)`.
  The JSONL row path spreads `**logs` verbatim, so this assertion would
  have passed pre-fix as well. **It does not exercise the bug.**

**Recommendation**: Add a dashboard-targeted unit test. Mock `LiveDashboard`,
drive `LiveDashboardCallback.on_log` with `logs={"epoch": 0.34, "loss": …}`,
assert the captured `dashboard.update` kwargs contain `epoch=0.34` (float) —
not `0` (int). One test per trainer (sft, kto, grpo) since the bug lived in
all three. This is one small patch, ~30 LOC, and closes the regression door
on the PR's headline promise.

### Minor

#### M1. Test uncertainty-traceability is one-to-one, but the dict-merge test is semantically broader than the stated uncertainty

Mapping my 21 tests to the 4 uncertainties:

| Uncertainty | Test class | Coverage |
|---|---|---|
| [MEDIUM] KTO env-wins cloud-provider | `TestKtoCloudProviderEnvWins` (4 tests) | env-overrides-args, args-fallback, empty-env-fallback, neither-set |
| [MEDIUM] `log_every_write` cadence knob | `TestLogEveryWriteCadence` (4 tests) | SFT skips non-boundary, KTO/GRPO write every, class-attr grep-visible |
| [MEDIUM] GRPO dict-merge order | `TestDictMergeOrder` (3 tests) | logs.step wins, loss/lr/reward verbatim, logs can override steps_per_second |
| [HIGH] GRPO `interval_seconds` key | `TestGrpoIntervalKey` (4 tests) | GRPO has, SFT has interval_time, KTO has interval_time, class-attr grep-visible |
| (bonus — direct helper) | `TestResolveCloudProvider` (5 tests) | env wins, env absent, neither, empty, whitespace |
| (bonus — SFT JSONL shape) | `TestSftJsonlShape` (1 test) | shape-only parity |

Classification: tight 1:1 on uncertainties 1, 2, 4. Uncertainty 3 (dict-merge
order) is *semantically broader* than the lead's restatement — the lead's
guidance was specifically "GRPO logs.step wins over state.global_step", which
is covered. My two additional assertions (`loss`/`learning_rate`/`reward`
verbatim; `steps_per_second` overrideable) are not gratuitous: they document
the commutative property of the merge order so a future reader can reason
about it without re-reading `base.py`.

**Recommendation**: No action — this is fine. Note the breadth in the HANDOFF
so the architect reviewing design conformance can confirm intent.

#### M2. SimpleNamespace mocks omit several real TrainerState/TrainingArguments attributes — none currently read, but fragile

My stubs only set:

- State: `global_step`, `max_steps`, `epoch`
- Args: `per_device_train_batch_size`, `gradient_accumulation_steps`,
  `max_grad_norm`, `output_dir`, and optionally `cloud_provider`.

Real HF `TrainerState` carries `log_history`, `best_model_checkpoint`,
`is_hyper_param_search`, etc. Real `TrainingArguments` carries ~200 fields.
A future base-class change that reads any unmocked attribute (e.g.,
`args.num_train_epochs` — which `BaseLiveDashboardCallback.on_train_begin`
DOES read at `base.py:352`) will surface as `AttributeError` in the test,
which is still a loud failure. No silent pass risk.

**However**: my tests only exercise `BaseMetricsCallback` (JSONL/table path),
NOT `BaseLiveDashboardCallback` (which reads `args.num_train_epochs`). The
dashboard path is the one where the bug lived. So the mock narrowness
compounds with B1.

**Recommendation**: When adding the B1 dashboard test, extend the args stub
to include `num_train_epochs`. Keep the pattern: stub only what the code
under test reads, and let `AttributeError` fail loudly when that invariant
breaks.

#### M3. `_dashboard_metrics` fallback chains are entirely uncovered

The divergence matrix (design §5) says per-trainer dashboard metric
extraction stays trainer-specific. My tests cover none of it:

- `Trainers/kto/src/training_callbacks.py:82` — KTO's `kl` falls back to
  `logps/rejected` when `kl` is absent.
- `Trainers/grpo/src/training_callbacks.py:91-99` — GRPO tries
  `reward` → `rewards` → `rewards/mean` → `mean_reward` and similar chains
  for `reward_std`, `kl_penalty`, `advantage`.

Neither is asserted anywhere. A fallback key rename (e.g. TRL releasing a
new `rewards/std_dev` field) would silently degrade the dashboard's live
reward-std display to 0 without any test red.

**Recommendation**: When writing the B1 dashboard test, add three small
cases asserting each fallback chain. Drive `on_log` with `logs` missing the
preferred key, assert the dashboard captures the fallback. Combined with
B1's fix, this becomes ~60 LOC of test code total.

#### M4. Shape-only JSONL parity is rigorous enough for STANDARD risk — do not block on byte parity

A captured pre-refactor baseline would give byte-identity, but:

- Capturing it requires running real training (deferred per §8 migration
  sequence).
- The dict-merge order test (M1) already locks down the spread invariant.
- The `interval_key_name` test (uncertainty #4) locks down the one known
  GRPO divergence.
- The `log_every_write` test locks down the cadence divergence.
- Shape-only parity (required keys + types) catches silent field drift.

Between those four locks, a silent row-shape regression would have to bypass
all of them. That's acceptable for a STANDARD-tier refactor.

**Recommendation**: Keep shape-only. If byte parity is demanded later, the
right path is a fixtures-based golden test driven by a ~50-step real SFT run
captured to JSONL, diffed as structured rows (not bytes — timestamps and
capacity vary run-to-run).

#### M5. Regression scope was adequate — no hidden callback-touching tests

Exhaustive grep across `tests/` for callback symbols (`MetricsTableCallback`,
`LiveDashboardCallback`, `BaseMetricsCallback`, `BaseLiveDashboardCallback`,
`TwoStageLRCallback`, `CheckpointMonitorCallback`, `suppress_training_logs`,
`HealthChecker`, `training_callbacks`) finds 3 files:

1. `tests/trainers/test_callback_refactor.py` — the new file.
2. `tests/cloud/test_dashboard.py` — only constructs `LiveDashboard`, does
   not touch callbacks.
3. `tests/cloud/test_hf_jobs_backend.py` — stubs `mock_shared_ui.LiveDashboard
   = DummyDashboard` at line 523, does not touch callbacks.

Both 2 and 3 pass (45/45). Other matches for `from Trainers` are unrelated
imports (`rewards`, `runpod_sync`, `filter_lora_adapter`, `manage_space`).

1812 total tests collect cleanly at the worktree root, confirming no
callback-refactor import-path breakage elsewhere.

### Future

#### F1. `capture_runtime_capacity_snapshot` behaviour under GPU vs CPU is indirectly exercised

My tests run on a no-GPU box; the snapshot returns CPU-only fields. The
GPU-present branch at `shared/training_capacity.py:210+` is not exercised by
any test in this repo. Not a refactor concern (code unchanged), but a
candidate for future snapshot-based coverage if callback JSONL rows ever
become load-bearing for experiment tracking decisions.

#### F2. `suppress_training_logs` hoist from SFT → shared has no direct test

`Trainers/shared/callbacks/log_suppression.py` is new; SFT re-exports it;
KTO/GRPO don't use it. No test calls `suppress_training_logs()`. The design
calls this "no-op for kto/grpo" and the module's function presumably still
does what it always did. If stdout/stderr redirection semantics ever drift,
no test would catch it. Low-priority given the extraction was byte-identical
per the design matrix.

#### F3. Health-check output format drift has no snapshot test

Per design §9 risk matrix: "KTO health-check output format drift (spaces,
punctuation) — Medium likelihood, Low impact — snapshot-test output." No
such snapshot exists. The `HealthChecker` ABC is not exercised directly by
any test. If you add any, `capsys` plus a fixed-logs input would snapshot
the banner lines in under 20 LOC.

#### F4. No byte-level parity baseline captured

Per M4, shape-only is acceptable. Noting the absence here for
future-tier tests if this code ever moves up from STANDARD to HIGH risk.

---

## Edge Cases — Probed, Clean

| Case | Behaviour | Covered? |
|---|---|---|
| `logs=None` | `on_log` returns early (base.py:177 `if not logs: return`) | Implicit — no test, but the guard is trivial |
| `logs={}` | Same early return (empty dict is falsy) | Implicit — same guard |
| `state.global_step=0` | `0 % log_every_n_steps == 0` → SFT writes row at step 0; KTO/GRPO always write. Edge: `steps_per_sec=0/elapsed` division guarded (`if elapsed > 0`) | Not explicitly tested — would pass on inspection |
| `args.output_dir=None` | Base uses `self.default_output_dir` via `output_dir if output_dir is not None else self.default_output_dir` at base.py:138. Safe. | Not explicitly tested |
| `CLOUD_PROVIDER=" "` (whitespace) | `.strip()` falsy → falls back to args | Covered by `test_whitespace_env_falls_back_to_args` |
| `logs["step"]` collision | logs wins | Covered |
| `interval_key_name` collision — what if logs contained `"interval_seconds"`? | logs wins, overrides base's computed interval | Not tested — edge case, likely harmless |

None of these are blockers. `logs={}` / `logs=None` guards are one-liners
the current tests implicitly rely on (the `_begin` helper feeds non-empty
logs). A single `test_on_log_with_empty_logs_returns_early` would be a nice
belt-and-braces addition.

---

## Signal

| Field | Value |
|---|---|
| Risk Tier | **HIGH** (elevated from STANDARD — headline bug fix is untested; any regression goes to production dashboards for all three trainers) |
| Signal | 🟡 **YELLOW** |
| Coverage | 21/21 new tests pass; 1/1 HIGH + 3/3 MEDIUM refactor uncertainties tested; 0/1 bug-fix regression guard |
| Uncertainty Coverage | 1 of 1 HIGH flagged by architect/coder tested (interval_seconds) |
| Blockers | **B1**: add a dashboard-path test locking `epoch: float` into `LiveDashboard.update` |
| Minors | M1–M5 addressed above |
| Future | F1–F4 addressed above |

Route back to a coder to add the B1 test (and optionally M3 fallback
coverage) — do not block the merge on anything else. Once B1 lands, re-run
the refactor test file to confirm GREEN and re-emit signal.

---

## Proposed remediation (for whoever picks up B1)

```python
# Add to tests/trainers/test_callback_refactor.py

class TestEpochDashboardRegression:
    """Lock `float` epoch into `LiveDashboard.update` kwargs. Guards the
    headline bug this PR fixes: int-cast truncated sub-epoch progress to 0
    in the TUI, breaking the `X.XX/N` epoch display until epoch completion."""

    def _drive(self, CallbackCls, tmp_path, logs_epoch):
        from unittest.mock import MagicMock
        cb = CallbackCls(log_every_n_steps=1, output_dir=str(tmp_path))
        cb.dashboard = MagicMock()  # bypass real LiveDashboard construction
        cb.start_time = datetime.now()
        cb.total_steps = 100
        cb.total_epochs = 3
        args = _make_args()
        args.num_train_epochs = 3
        cb.on_log(args, _make_state(global_step=1), _make_control(),
                  logs={"loss": 1.0, "learning_rate": 1e-5, "epoch": logs_epoch})
        return cb.dashboard.update.call_args

    def test_sft_dashboard_receives_float_epoch(self, tmp_path):
        from Trainers.sft.src.training_callbacks import LiveDashboardCallback
        call = self._drive(LiveDashboardCallback, tmp_path / "sft", 0.34)
        assert call.kwargs["epoch"] == 0.34
        assert isinstance(call.kwargs["epoch"], float)

    def test_kto_dashboard_receives_float_epoch(self, tmp_path):
        from Trainers.kto.src.training_callbacks import LiveDashboardCallback
        call = self._drive(LiveDashboardCallback, tmp_path / "kto", 0.67)
        assert call.kwargs["epoch"] == 0.67
        assert isinstance(call.kwargs["epoch"], float)

    def test_grpo_dashboard_receives_float_epoch(self, tmp_path):
        from Trainers.grpo.src.training_callbacks import LiveDashboardCallback
        call = self._drive(LiveDashboardCallback, tmp_path / "grpo", 0.12)
        assert call.kwargs["epoch"] == 0.12
        assert isinstance(call.kwargs["epoch"], float)
```

~40 LOC, no GPU, no real HF Trainer, mocks the dashboard so the test
survives CI environments without rich/LiveDashboard.
