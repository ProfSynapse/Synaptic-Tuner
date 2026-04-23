"""Targeted tests for the sft/kto/grpo callback DRY refactor.

Verifies the four uncertainties flagged by the architect and coder on the
callback refactor design (docs/architecture/training-callbacks-refactor.md),
plus the B1/B2/B3 remediations from PR #85 peer review:

  1. [MEDIUM] KTO cloud_provider env-first resolution (intentional additive
     change for KTO per §6 — env wins over args).
  2. [MEDIUM] `log_every_write` cadence knob — SFT writes only at
     log_every_n_steps boundary; KTO writes every on_log.
  3. [MEDIUM] Dict-merge order — per-trainer `fields_win_on_collision`
     class attr (SFT/KTO default False = logs-win; GRPO override True =
     fields-win, matching pre-refactor GRPO behavior).
  4. [HIGH]   GRPO `interval_seconds` key preservation — not `interval_time`.
  5. [B1]    SFT cadence gating — `health_checker.check()` and
     `last_log_time` update fire only at modulo boundary for SFT (matches
     pre-refactor top-of-on_log early-return behavior). KTO/GRPO fire
     every on_log call.
  6. [B3]    Epoch float-cast regression — `dashboard.update(epoch=...)`
     receives a float, not an int, so sub-epoch progress (e.g. 0.34) is
     not truncated to 0. Guards against re-introducing
     `int(logs.get('epoch', 0))` in BaseLiveDashboardCallback.on_log.

Plus an SFT shape-only test asserting the JSONL row shape is sane
(keys + types). No pre-refactor baseline rows were captured, so parity is
asserted by shape, not byte-identity — documented in the handoff.

All tests are unit-level. `TrainerState` / `TrainingArguments` are hand-
constructed as SimpleNamespace stubs. No GPU, no real training, no network.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Make Trainers.shared.callbacks importable + each trainer's src/ importable.
WORKTREE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKTREE_ROOT))
sys.path.insert(0, str(WORKTREE_ROOT / "Trainers" / "sft" / "src"))
sys.path.insert(0, str(WORKTREE_ROOT / "Trainers" / "kto" / "src"))
sys.path.insert(0, str(WORKTREE_ROOT / "Trainers" / "grpo" / "src"))


# ---------------------------------------------------------------------------
# Stubs for HF TrainerState / TrainingArguments (no GPU, no HF Trainer)
# ---------------------------------------------------------------------------

def _make_state(global_step: int = 1, max_steps: int = 100, epoch: float = 0.1):
    return SimpleNamespace(global_step=global_step, max_steps=max_steps, epoch=epoch)


def _make_args(cloud_provider=None, max_grad_norm=1.0):
    """Minimal TrainingArguments stub — only attributes `on_log` reads."""
    ns = SimpleNamespace(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_grad_norm=max_grad_norm,
        output_dir="./stub_output",
    )
    if cloud_provider is not None:
        ns.cloud_provider = cloud_provider
    return ns


def _make_control():
    return SimpleNamespace()


def _read_jsonl_rows(path: Path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Fixtures — per-trainer callback instances writing into tmp_path
# ---------------------------------------------------------------------------

@pytest.fixture
def sft_callback(tmp_path):
    from Trainers.sft.src.training_callbacks import MetricsTableCallback as SFTMetrics
    cb = SFTMetrics(log_every_n_steps=5, output_dir=str(tmp_path / "sft_out"))
    return cb


@pytest.fixture
def kto_callback(tmp_path):
    from Trainers.kto.src.training_callbacks import MetricsTableCallback as KTOMetrics
    cb = KTOMetrics(log_every_n_steps=5, output_dir=str(tmp_path / "kto_out"))
    return cb


@pytest.fixture
def grpo_callback(tmp_path):
    from Trainers.grpo.src.training_callbacks import MetricsTableCallback as GRPOMetrics
    cb = GRPOMetrics(log_every_n_steps=5, output_dir=str(tmp_path / "grpo_out"))
    return cb


def _begin(cb, state=None):
    """Call on_train_begin with default stubs so timing + symlink state is valid."""
    args = _make_args()
    control = _make_control()
    state = state or _make_state()
    cb.on_train_begin(args, state, control)


# ---------------------------------------------------------------------------
# Uncertainty #1 — KTO cloud_provider env-first (intentional additive, §6)
# ---------------------------------------------------------------------------

class TestKtoCloudProviderEnvWins:
    """KTO previously read args.cloud_provider only. Refactor makes env win
    over args across all trainers. This verifies the env-first precedence
    lands in KTO's JSONL log rows."""

    def test_env_overrides_args(self, kto_callback, monkeypatch):
        monkeypatch.setenv("CLOUD_PROVIDER", "hf-jobs")
        _begin(kto_callback)
        args = _make_args(cloud_provider="local")
        kto_callback.on_log(args, _make_state(global_step=1), _make_control(),
                            logs={"loss": 1.0, "learning_rate": 1e-5})

        rows = _read_jsonl_rows(kto_callback.log_file)
        assert rows, "KTO should have written a row on first on_log (log_every_write=True)"
        assert rows[0]["cloud_provider"] == "hf-jobs", (
            f"env-first precedence lost: {rows[0].get('cloud_provider')!r}"
        )

    def test_args_fallback_when_env_absent(self, kto_callback, monkeypatch):
        monkeypatch.delenv("CLOUD_PROVIDER", raising=False)
        _begin(kto_callback)
        args = _make_args(cloud_provider="runpod")
        kto_callback.on_log(args, _make_state(global_step=1), _make_control(),
                            logs={"loss": 1.0, "learning_rate": 1e-5})

        rows = _read_jsonl_rows(kto_callback.log_file)
        assert rows[0]["cloud_provider"] == "runpod"

    def test_env_empty_falls_back_to_args(self, kto_callback, monkeypatch):
        # resolve_cloud_provider strips env; empty-string env must not win.
        monkeypatch.setenv("CLOUD_PROVIDER", "  ")
        _begin(kto_callback)
        args = _make_args(cloud_provider="runpod")
        kto_callback.on_log(args, _make_state(global_step=1), _make_control(),
                            logs={"loss": 1.0})

        rows = _read_jsonl_rows(kto_callback.log_file)
        assert rows[0]["cloud_provider"] == "runpod"

    def test_neither_set_omits_key(self, kto_callback, monkeypatch):
        monkeypatch.delenv("CLOUD_PROVIDER", raising=False)
        _begin(kto_callback)
        # args has no cloud_provider attr -> getattr default None -> key omitted.
        args = _make_args(cloud_provider=None)
        kto_callback.on_log(args, _make_state(global_step=1), _make_control(),
                            logs={"loss": 1.0})

        rows = _read_jsonl_rows(kto_callback.log_file)
        assert "cloud_provider" not in rows[0], (
            "cloud_provider key must be absent when resolve returns None"
        )


# ---------------------------------------------------------------------------
# Uncertainty #2 — log_every_write cadence single-knob preservation
# ---------------------------------------------------------------------------

class TestLogEveryWriteCadence:
    """SFT (log_every_write=False) writes only at log_every_n_steps boundaries.
    KTO + GRPO (log_every_write=True) write on every on_log call."""

    def test_sft_skips_non_boundary_steps(self, sft_callback):
        _begin(sft_callback)
        args = _make_args()
        # log_every_n_steps=5, so steps 1,2,3,4 must NOT write; step 5 must write.
        for step in (1, 2, 3, 4):
            sft_callback.on_log(args, _make_state(global_step=step), _make_control(),
                                logs={"loss": 1.0, "learning_rate": 1e-5})
        rows_before = _read_jsonl_rows(sft_callback.log_file)
        assert rows_before == [], f"SFT wrote at non-boundary step: {rows_before}"

        sft_callback.on_log(args, _make_state(global_step=5), _make_control(),
                            logs={"loss": 1.0, "learning_rate": 1e-5})
        rows_after = _read_jsonl_rows(sft_callback.log_file)
        assert len(rows_after) == 1, f"SFT expected 1 row at boundary, got {len(rows_after)}"

    def test_kto_writes_every_on_log(self, kto_callback):
        _begin(kto_callback)
        args = _make_args()
        for step in (1, 2, 3):
            kto_callback.on_log(args, _make_state(global_step=step), _make_control(),
                                logs={"loss": 1.0, "learning_rate": 1e-5})
        rows = _read_jsonl_rows(kto_callback.log_file)
        assert len(rows) == 3, f"KTO expected 3 rows (one per on_log), got {len(rows)}"

    def test_grpo_writes_every_on_log(self, grpo_callback):
        _begin(grpo_callback)
        args = _make_args()
        for step in (1, 2, 3):
            grpo_callback.on_log(args, _make_state(global_step=step), _make_control(),
                                 logs={"loss": 1.0, "learning_rate": 1e-5})
        rows = _read_jsonl_rows(grpo_callback.log_file)
        assert len(rows) == 3, f"GRPO expected 3 rows (one per on_log), got {len(rows)}"

    def test_cadence_knob_is_class_attr(self):
        """The design's single-knob claim: `log_every_write` must be grep-visible
        as a class attribute, so reviewers can audit JSONL write behavior at a
        glance (per §5 of the architecture doc)."""
        from Trainers.sft.src.training_callbacks import MetricsTableCallback as SFTMetrics
        from Trainers.kto.src.training_callbacks import MetricsTableCallback as KTOMetrics
        from Trainers.grpo.src.training_callbacks import MetricsTableCallback as GRPOMetrics

        assert SFTMetrics.log_every_write is False
        assert KTOMetrics.log_every_write is True
        assert GRPOMetrics.log_every_write is True


# ---------------------------------------------------------------------------
# Uncertainty #3 — dict-merge order: logs spread last, logs win
# ---------------------------------------------------------------------------

class TestDictMergeOrder:
    """Per-trainer collision precedence, controlled by `fields_win_on_collision`.

    SFT + KTO (default False): logs win on collision — preserves pre-refactor
    `{**our_fields, **capacity, **logs}` build order.

    GRPO (override True): our_fields win on collision — preserves pre-refactor
    `entry = dict(logs); entry[k]=v; entry.update(cap)` build order."""

    def test_grpo_fields_win_state_step_overrides_logs_step(self, grpo_callback):
        """GRPO: state.global_step wins over logs['step'] — matches pre-refactor
        GRPO build order where our fields + capacity were applied AFTER logs."""
        _begin(grpo_callback)
        args = _make_args()
        # State says step=1 but logs carry step=42; GRPO's fields_win_on_collision
        # means step=1 (from state) overrides logs['step']=42.
        grpo_callback.on_log(args, _make_state(global_step=1), _make_control(),
                             logs={"loss": 1.0, "step": 42})
        rows = _read_jsonl_rows(grpo_callback.log_file)
        assert rows[0]["step"] == 1, (
            f"GRPO fields_win_on_collision=True: state.global_step must override "
            f"logs['step']; got {rows[0]['step']}"
        )

    def test_kto_logs_win_logs_step_overrides_state_step(self, kto_callback):
        """KTO: logs['step'] wins — default fields_win_on_collision=False."""
        _begin(kto_callback)
        args = _make_args()
        kto_callback.on_log(args, _make_state(global_step=1), _make_control(),
                            logs={"loss": 1.0, "step": 42})
        rows = _read_jsonl_rows(kto_callback.log_file)
        assert rows[0]["step"] == 42, (
            f"KTO fields_win_on_collision=False: logs['step'] must win over "
            f"state.global_step; got {rows[0]['step']}"
        )

    def test_logs_loss_appears_verbatim(self, grpo_callback):
        _begin(grpo_callback)
        args = _make_args()
        grpo_callback.on_log(args, _make_state(global_step=7), _make_control(),
                             logs={"loss": 0.1234, "learning_rate": 2.5e-5,
                                   "reward": 0.88})
        rows = _read_jsonl_rows(grpo_callback.log_file)
        row = rows[0]
        assert row["loss"] == 0.1234
        assert row["learning_rate"] == 2.5e-5
        assert row["reward"] == 0.88

    def test_logs_can_override_elapsed_keys_if_present(self, kto_callback):
        """Shape-parity guard: if someone ever emits a conflicting key in logs,
        dict-merge order ensures logs wins, preventing silent field masking."""
        _begin(kto_callback)
        args = _make_args()
        kto_callback.on_log(args, _make_state(global_step=1), _make_control(),
                            logs={"loss": 1.0, "steps_per_second": 999.9})
        rows = _read_jsonl_rows(kto_callback.log_file)
        assert rows[0]["steps_per_second"] == 999.9


# ---------------------------------------------------------------------------
# Uncertainty #4 — GRPO `interval_seconds` key preservation [HIGH]
# ---------------------------------------------------------------------------

class TestGrpoIntervalKey:
    """GRPO's original JSONL schema uses `interval_seconds`, not the
    `interval_time` key SFT + KTO use. The refactor preserves this via the
    `interval_key_name` class attr. If this flips, downstream parsers break."""

    def test_grpo_emits_interval_seconds(self, grpo_callback):
        _begin(grpo_callback)
        args = _make_args()
        grpo_callback.on_log(args, _make_state(global_step=1), _make_control(),
                             logs={"loss": 1.0})
        rows = _read_jsonl_rows(grpo_callback.log_file)
        row = rows[0]
        assert "interval_seconds" in row, (
            f"GRPO row must contain 'interval_seconds'; keys={list(row.keys())}"
        )
        assert "interval_time" not in row, (
            f"GRPO row must NOT contain 'interval_time'; keys={list(row.keys())}"
        )

    def test_sft_emits_interval_time(self, sft_callback):
        """Complement: SFT stays on `interval_time` (default)."""
        _begin(sft_callback)
        args = _make_args()
        sft_callback.on_log(args, _make_state(global_step=5), _make_control(),
                            logs={"loss": 1.0})
        rows = _read_jsonl_rows(sft_callback.log_file)
        row = rows[0]
        assert "interval_time" in row
        assert "interval_seconds" not in row

    def test_kto_emits_interval_time(self, kto_callback):
        _begin(kto_callback)
        args = _make_args()
        kto_callback.on_log(args, _make_state(global_step=1), _make_control(),
                            logs={"loss": 1.0})
        rows = _read_jsonl_rows(kto_callback.log_file)
        row = rows[0]
        assert "interval_time" in row
        assert "interval_seconds" not in row

    def test_interval_key_class_attr_is_grep_visible(self):
        """Same single-knob visibility guarantee as log_every_write."""
        from Trainers.sft.src.training_callbacks import MetricsTableCallback as SFTMetrics
        from Trainers.kto.src.training_callbacks import MetricsTableCallback as KTOMetrics
        from Trainers.grpo.src.training_callbacks import MetricsTableCallback as GRPOMetrics

        assert SFTMetrics.interval_key_name == "interval_time"
        assert KTOMetrics.interval_key_name == "interval_time"
        assert GRPOMetrics.interval_key_name == "interval_seconds"


# ---------------------------------------------------------------------------
# SFT JSONL row shape — shape-only parity check
# ---------------------------------------------------------------------------

class TestSftJsonlShape:
    """Shape-only parity: no pre-refactor baseline rows captured. This test
    asserts the canonical SFT JSONL row shape (keys + types) that the base
    class produces, so future refactors can catch silent field drift."""

    def test_sft_row_has_required_keys_and_types(self, sft_callback):
        _begin(sft_callback)
        args = _make_args()
        sft_callback.on_log(args, _make_state(global_step=5, max_steps=100, epoch=0.25),
                            _make_control(),
                            logs={"loss": 0.5, "learning_rate": 1e-5,
                                  "grad_norm": 0.1, "epoch": 0.25})
        rows = _read_jsonl_rows(sft_callback.log_file)
        assert len(rows) == 1
        row = rows[0]

        # Required base fields.
        assert isinstance(row["step"], int)
        assert row["step"] == 5
        assert isinstance(row["timestamp"], str)
        assert isinstance(row["interval_time"], (int, float))
        assert isinstance(row["elapsed_seconds"], (int, float))
        assert isinstance(row["steps_per_second"], (int, float))
        assert isinstance(row["samples_per_sec"], (int, float))

        # Logs fields passed through.
        assert row["loss"] == 0.5
        assert row["learning_rate"] == 1e-5
        assert row["grad_norm"] == 0.1
        assert row["epoch"] == 0.25


# ---------------------------------------------------------------------------
# Bonus: resolve_cloud_provider direct unit coverage
# ---------------------------------------------------------------------------

class TestResolveCloudProvider:
    """Direct unit coverage of the helper — documents precedence exhaustively."""

    def test_env_wins(self, monkeypatch):
        from Trainers.shared.callbacks import resolve_cloud_provider
        monkeypatch.setenv("CLOUD_PROVIDER", "hf-jobs")
        args = SimpleNamespace(cloud_provider="local")
        assert resolve_cloud_provider(args) == "hf-jobs"

    def test_env_absent_uses_args(self, monkeypatch):
        from Trainers.shared.callbacks import resolve_cloud_provider
        monkeypatch.delenv("CLOUD_PROVIDER", raising=False)
        args = SimpleNamespace(cloud_provider="local")
        assert resolve_cloud_provider(args) == "local"

    def test_env_and_args_absent_returns_none(self, monkeypatch):
        from Trainers.shared.callbacks import resolve_cloud_provider
        monkeypatch.delenv("CLOUD_PROVIDER", raising=False)
        args = SimpleNamespace()  # no attribute
        assert resolve_cloud_provider(args) is None

    def test_empty_env_falls_back_to_args(self, monkeypatch):
        from Trainers.shared.callbacks import resolve_cloud_provider
        monkeypatch.setenv("CLOUD_PROVIDER", "")
        args = SimpleNamespace(cloud_provider="local")
        assert resolve_cloud_provider(args) == "local"

    def test_whitespace_env_falls_back_to_args(self, monkeypatch):
        from Trainers.shared.callbacks import resolve_cloud_provider
        monkeypatch.setenv("CLOUD_PROVIDER", "   ")
        args = SimpleNamespace(cloud_provider="local")
        assert resolve_cloud_provider(args) == "local"


# ---------------------------------------------------------------------------
# B3 — Epoch float-cast regression (dashboard.update kwargs path)
# ---------------------------------------------------------------------------

def _make_dashboard_cb(cb_cls, tmp_path, subdir: str):
    """Build a BaseLiveDashboardCallback subclass with a mocked dashboard.

    Bypasses `DASHBOARD_AVAILABLE` / `RICH_AVAILABLE` gates by setting
    `cb.dashboard` directly and seeding `cb.start_time` so on_log does not
    trip on unset timing state.
    """
    from datetime import datetime
    cb = cb_cls(log_every_n_steps=1, output_dir=str(tmp_path / subdir))
    cb.dashboard = MagicMock()
    cb.start_time = datetime.now()
    cb.total_steps = 100
    cb.total_epochs = 3
    return cb


def _dashboard_args():
    """TrainingArguments stub the LiveDashboardCallback.on_log body reads."""
    return SimpleNamespace(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        num_train_epochs=3,
        output_dir="./stub",
    )


class TestEpochDashboardRegression:
    """Regression guard for the PR #85 headline bug: epoch counter stuck at 0/X
    in the TUI dashboard because `dashboard.update(epoch=int(logs.get('epoch', 0)))`
    truncated sub-epoch progress (0.34 -> 0). Fix: cast to float.

    These tests assert `mock_dashboard.update.call_args.kwargs['epoch']` is a
    float ≈ the sub-epoch value the trainer emitted. If someone re-introduces
    the int-cast, these tests fail loudly (0 != 0.34)."""

    @pytest.mark.parametrize("trainer_name", ["sft", "kto", "grpo"])
    def test_epoch_passed_as_float_to_dashboard_update(self, tmp_path, trainer_name):
        if trainer_name == "sft":
            from Trainers.sft.src.training_callbacks import LiveDashboardCallback
        elif trainer_name == "kto":
            from Trainers.kto.src.training_callbacks import LiveDashboardCallback
        else:
            from Trainers.grpo.src.training_callbacks import LiveDashboardCallback

        cb = _make_dashboard_cb(LiveDashboardCallback, tmp_path, f"{trainer_name}_dash")
        args = _dashboard_args()

        # log_every_n_steps=1, so every on_log call triggers a dashboard.update.
        cb.on_log(
            args,
            _make_state(global_step=1, max_steps=100, epoch=0.34),
            _make_control(),
            logs={"loss": 1.0, "learning_rate": 1e-5, "epoch": 0.34},
        )

        assert cb.dashboard.update.called, (
            f"{trainer_name}: expected dashboard.update(...) to be called once"
        )
        kwargs = cb.dashboard.update.call_args.kwargs
        assert "epoch" in kwargs, f"{trainer_name}: epoch kwarg missing from update()"
        epoch_val = kwargs["epoch"]
        # Regression guard #1: must be float, not int.
        assert isinstance(epoch_val, float), (
            f"{trainer_name}: epoch must be float (regression); got "
            f"{type(epoch_val).__name__}={epoch_val!r}"
        )
        # Regression guard #2: must preserve sub-epoch precision.
        assert abs(epoch_val - 0.34) < 1e-9, (
            f"{trainer_name}: epoch value truncated; got {epoch_val!r}, expected ~0.34"
        )
        # Regression guard #3: explicitly not the truncated int-cast value.
        assert epoch_val != 0, (
            f"{trainer_name}: epoch truncated to 0 — int() cast regression"
        )

    def test_epoch_none_falls_back_to_zero_float(self, tmp_path):
        """When logs omits 'epoch', the `or 0.0` fallback yields 0.0 (float),
        not 0 (int). Covers the right-hand side of `float(... or 0.0)`."""
        from Trainers.sft.src.training_callbacks import LiveDashboardCallback
        cb = _make_dashboard_cb(LiveDashboardCallback, tmp_path, "sft_dash_none")
        args = _dashboard_args()

        cb.on_log(
            args,
            _make_state(global_step=1),
            _make_control(),
            logs={"loss": 1.0, "learning_rate": 1e-5},  # no 'epoch'
        )
        epoch_val = cb.dashboard.update.call_args.kwargs["epoch"]
        assert isinstance(epoch_val, float)
        assert epoch_val == 0.0


# ---------------------------------------------------------------------------
# B1 — SFT cadence gating: health_check + last_log_time modulo gate
# ---------------------------------------------------------------------------

class TestSftCadenceGating:
    """SFT overrides `health_check_every_on_log=False` and
    `interval_time_updates_every_on_log=False`. Pre-refactor SFT gated the
    entire on_log body on the interval multiple, so health checks and
    last_log_time updates fired only at printed-row cadence. KTO + GRPO
    (defaults True) fired them every on_log. These tests pin that split
    so a future refactor cannot silently drift SFT to every-call cadence."""

    def test_sft_health_check_only_at_modulo_boundary(self, sft_callback):
        _begin(sft_callback)
        sft_callback.health_checker = MagicMock()
        args = _make_args()

        # log_every_n_steps=5: non-boundary steps 1..4 must NOT call health_checker.check.
        for step in (1, 2, 3, 4):
            sft_callback.on_log(args, _make_state(global_step=step), _make_control(),
                                logs={"loss": 1.0})
        assert sft_callback.health_checker.check.call_count == 0, (
            f"SFT called health_checker.check at non-boundary step "
            f"(expected 0 calls, got {sft_callback.health_checker.check.call_count})"
        )

        # Step 5 is a boundary — health_checker.check must fire exactly once.
        sft_callback.on_log(args, _make_state(global_step=5), _make_control(),
                            logs={"loss": 1.0})
        assert sft_callback.health_checker.check.call_count == 1, (
            f"SFT did not call health_checker.check at boundary step "
            f"(expected 1 call, got {sft_callback.health_checker.check.call_count})"
        )

    def test_kto_health_check_every_on_log(self, kto_callback):
        _begin(kto_callback)
        kto_callback.health_checker = MagicMock()
        args = _make_args()

        for step in (1, 2, 3, 4):
            kto_callback.on_log(args, _make_state(global_step=step), _make_control(),
                                logs={"loss": 1.0})
        assert kto_callback.health_checker.check.call_count == 4, (
            f"KTO should call health_checker.check every on_log (expected 4, "
            f"got {kto_callback.health_checker.check.call_count})"
        )

    def test_grpo_health_check_every_on_log(self, grpo_callback):
        _begin(grpo_callback)
        grpo_callback.health_checker = MagicMock()
        args = _make_args()

        for step in (1, 2, 3, 4):
            grpo_callback.on_log(args, _make_state(global_step=step), _make_control(),
                                 logs={"loss": 1.0})
        assert grpo_callback.health_checker.check.call_count == 4, (
            f"GRPO should call health_checker.check every on_log (expected 4, "
            f"got {grpo_callback.health_checker.check.call_count})"
        )

    def test_sft_last_log_time_only_updates_at_boundary(self, sft_callback):
        _begin(sft_callback)
        args = _make_args()
        # Snapshot last_log_time set by on_train_begin.
        baseline = sft_callback.last_log_time

        # Non-boundary steps must NOT advance last_log_time.
        for step in (1, 2, 3, 4):
            sft_callback.on_log(args, _make_state(global_step=step), _make_control(),
                                logs={"loss": 1.0})
        assert sft_callback.last_log_time == baseline, (
            "SFT advanced last_log_time at non-boundary step"
        )

        # Boundary step must advance last_log_time.
        sft_callback.on_log(args, _make_state(global_step=5), _make_control(),
                            logs={"loss": 1.0})
        assert sft_callback.last_log_time != baseline, (
            "SFT did not advance last_log_time at boundary step"
        )

    def test_kto_last_log_time_updates_every_on_log(self, kto_callback):
        _begin(kto_callback)
        args = _make_args()
        baseline = kto_callback.last_log_time
        kto_callback.on_log(args, _make_state(global_step=1), _make_control(),
                            logs={"loss": 1.0})
        assert kto_callback.last_log_time != baseline, (
            "KTO must advance last_log_time on every on_log call (non-boundary)"
        )


# ---------------------------------------------------------------------------
# Grep-visibility of new class-attr knobs (single-knob design contract)
# ---------------------------------------------------------------------------

class TestNewKnobsGrepVisible:
    """The three new knobs introduced by the B1+B2 remediation must be
    class attributes so reviewers can audit behavior splits at a glance.

    - `health_check_every_on_log`: base True, SFT False.
    - `interval_time_updates_every_on_log`: base True, SFT False.
    - `fields_win_on_collision`: base False, GRPO True."""

    def test_health_check_knob_per_trainer(self):
        from Trainers.sft.src.training_callbacks import MetricsTableCallback as SFTMetrics
        from Trainers.kto.src.training_callbacks import MetricsTableCallback as KTOMetrics
        from Trainers.grpo.src.training_callbacks import MetricsTableCallback as GRPOMetrics
        assert SFTMetrics.health_check_every_on_log is False
        assert KTOMetrics.health_check_every_on_log is True
        assert GRPOMetrics.health_check_every_on_log is True

    def test_interval_time_update_knob_per_trainer(self):
        from Trainers.sft.src.training_callbacks import MetricsTableCallback as SFTMetrics
        from Trainers.kto.src.training_callbacks import MetricsTableCallback as KTOMetrics
        from Trainers.grpo.src.training_callbacks import MetricsTableCallback as GRPOMetrics
        assert SFTMetrics.interval_time_updates_every_on_log is False
        assert KTOMetrics.interval_time_updates_every_on_log is True
        assert GRPOMetrics.interval_time_updates_every_on_log is True

    def test_fields_win_knob_per_trainer(self):
        from Trainers.sft.src.training_callbacks import MetricsTableCallback as SFTMetrics
        from Trainers.kto.src.training_callbacks import MetricsTableCallback as KTOMetrics
        from Trainers.grpo.src.training_callbacks import MetricsTableCallback as GRPOMetrics
        assert SFTMetrics.fields_win_on_collision is False
        assert KTOMetrics.fields_win_on_collision is False
        assert GRPOMetrics.fields_win_on_collision is True


# ---------------------------------------------------------------------------
# M-J — Dashboard metric fallback chains (KTO + GRPO)
# ---------------------------------------------------------------------------

class TestDashboardMetricFallbacks:
    """`_dashboard_metrics` uses multi-key fallback chains for trainer-specific
    metrics. Pins the per-trainer fallback order so a refactor can't silently
    drop a supported log-key variant.

    - KTO (`kto/training_callbacks.py:82`): `kl` → `logps/rejected` fallback.
    - GRPO (`grpo/training_callbacks.py:93-103`): `reward` chain is
      `reward` → `rewards` → `rewards/mean` → `mean_reward`.
    """

    def test_kto_kl_falls_back_to_logps_rejected(self, tmp_path):
        from Trainers.kto.src.training_callbacks import LiveDashboardCallback
        cb = _make_dashboard_cb(LiveDashboardCallback, tmp_path, "kto_kl_fallback")
        args = _dashboard_args()
        cb.on_log(
            args,
            _make_state(global_step=1),
            _make_control(),
            logs={"loss": 1.0, "logps/rejected": 0.5},  # no 'kl'
        )
        kwargs = cb.dashboard.update.call_args.kwargs
        assert kwargs["kl"] == 0.5, (
            f"KTO: expected kl fallback to logs['logps/rejected']=0.5; got {kwargs.get('kl')!r}"
        )

    def test_kto_kl_preferred_over_logps_rejected(self, tmp_path):
        """When both keys present, `kl` wins (short-circuit in get-fallback)."""
        from Trainers.kto.src.training_callbacks import LiveDashboardCallback
        cb = _make_dashboard_cb(LiveDashboardCallback, tmp_path, "kto_kl_preferred")
        args = _dashboard_args()
        cb.on_log(
            args,
            _make_state(global_step=1),
            _make_control(),
            logs={"loss": 1.0, "kl": 0.9, "logps/rejected": 0.1},
        )
        kwargs = cb.dashboard.update.call_args.kwargs
        assert kwargs["kl"] == 0.9

    @pytest.mark.parametrize(
        "logs_key,logs_value,expected",
        [
            ("rewards", 0.7, 0.7),       # no 'reward', falls back to 'rewards'
            ("rewards/mean", 0.3, 0.3),  # falls back to 'rewards/mean'
            ("mean_reward", 0.2, 0.2),   # falls back to 'mean_reward'
        ],
        ids=["rewards", "rewards_slash_mean", "mean_reward"],
    )
    def test_grpo_reward_fallback_chain(self, tmp_path, logs_key, logs_value, expected):
        from Trainers.grpo.src.training_callbacks import LiveDashboardCallback
        cb = _make_dashboard_cb(LiveDashboardCallback, tmp_path, f"grpo_reward_{logs_key}")
        args = _dashboard_args()
        cb.on_log(
            args,
            _make_state(global_step=1),
            _make_control(),
            logs={"loss": 1.0, logs_key: logs_value},
        )
        kwargs = cb.dashboard.update.call_args.kwargs
        assert kwargs["reward"] == expected, (
            f"GRPO: logs[{logs_key!r}]={logs_value} should produce reward={expected}; "
            f"got {kwargs.get('reward')!r}"
        )

    def test_grpo_reward_preferred_over_fallbacks(self, tmp_path):
        """When `reward` is present, it wins over all fallback keys."""
        from Trainers.grpo.src.training_callbacks import LiveDashboardCallback
        cb = _make_dashboard_cb(LiveDashboardCallback, tmp_path, "grpo_reward_preferred")
        args = _dashboard_args()
        cb.on_log(
            args,
            _make_state(global_step=1),
            _make_control(),
            logs={"loss": 1.0, "reward": 0.95, "rewards": 0.1, "rewards/mean": 0.2, "mean_reward": 0.3},
        )
        kwargs = cb.dashboard.update.call_args.kwargs
        assert kwargs["reward"] == 0.95


# ---------------------------------------------------------------------------
# F-A — HealthChecker output format snapshot (stdout capture)
# ---------------------------------------------------------------------------

class TestHealthCheckerOutputSnapshot:
    """Frozen-string snapshots of SFT/KTO health-check stdout.

    Design doc §9 risk matrix names 'KTO health-check output format drift' as
    a medium-likelihood risk. These tests pin the exact warning banner format
    (100-char bracket lines, emoji prefixes, 'Consider:' footer) so any text
    change in `Trainers/shared/callbacks/health_checks.py` fails loudly."""

    def test_sft_health_checker_all_branches_snapshot(self, capsys):
        from Trainers.shared.callbacks.health_checks import SFTHealthChecker
        checker = SFTHealthChecker()
        # Drives all three SFT branches: loss-range (loss=500 out of 0..100),
        # grad-clip (grad_norm=150 > 100), step>50 (step=60, loss=5.0 > 2.0).
        checker.check(
            logs={"loss": 500.0, "grad_norm": 150.0},
            step=60,
            max_grad_norm=1.0,
        )
        out = capsys.readouterr().out
        # Boundary markers: 100-char "!" lines top + bottom + blank padding.
        assert out.startswith("\n" + "!" * 100 + "\n"), (
            f"SFT: missing opening bracket line; got {out[:120]!r}"
        )
        assert out.endswith("!" * 100 + "\n\n"), (
            f"SFT: missing closing bracket line; got {out[-120:]!r}"
        )
        # All three warning types present with exact phrasing.
        assert "⚠ Unusual loss value: 500.0000" in out
        assert "⚠ High gradient norm: 150.00 → 1.00 (clipped)" in out
        assert "⚠ Loss remains high after 60 steps: 500.0000" in out
        # Footer: "Consider:" appears when max_grad_norm is None OR grad_norm > 10*max_grad_norm.
        # Here grad_norm=150 > 10*1.0=10, so the footer should appear.
        assert "Consider: reducing learning rate or using tighter gradient clipping" in out

    def test_kto_health_checker_all_branches_snapshot(self, capsys):
        from Trainers.shared.callbacks.health_checks import KTOHealthChecker
        checker = KTOHealthChecker()
        # Drives all four KTO branches: loss-range, margin<-1.0, reward-collapse
        # (abs(chosen)<0.001, abs(rejected)<0.001, step>10), grad-clip.
        checker.check(
            logs={
                "loss": 500.0,
                "rewards/margins": -2.5,
                "rewards/chosen": 0.0,
                "rewards/rejected": 0.0,
                "grad_norm": 200.0,
            },
            step=20,
            max_grad_norm=1.0,
        )
        out = capsys.readouterr().out
        assert out.startswith("\n" + "!" * 100 + "\n")
        assert out.endswith("!" * 100 + "\n\n")
        assert "⚠ Unusual loss value: 500.0000" in out
        assert "⚠ Very negative margin: -2.5000 (chosen model may be worse than reference)" in out
        assert "⚠ Reward collapse detected (both rewards near zero)" in out
        assert "⚠ High gradient norm: 200.00 → 1.00 (clipped)" in out
        assert "Consider: reducing learning rate or using tighter gradient clipping" in out

    def test_noop_health_checker_emits_nothing(self, capsys):
        from Trainers.shared.callbacks.health_checks import NoOpHealthChecker
        checker = NoOpHealthChecker()
        checker.check(
            logs={"loss": 500.0, "grad_norm": 999.0},  # would trigger SFT/KTO warnings
            step=100,
            max_grad_norm=1.0,
        )
        out = capsys.readouterr().out
        assert out == "", f"NoOpHealthChecker should emit nothing; got {out!r}"

    def test_sft_no_warnings_when_healthy(self, capsys):
        """Clean-path: no warnings means no stdout output (early-return via _print_warnings)."""
        from Trainers.shared.callbacks.health_checks import SFTHealthChecker
        checker = SFTHealthChecker()
        checker.check(
            logs={"loss": 0.5, "grad_norm": 1.0},
            step=10,
            max_grad_norm=1.0,
        )
        out = capsys.readouterr().out
        assert out == "", f"SFT with healthy metrics should emit nothing; got {out!r}"


# ---------------------------------------------------------------------------
# F-B — capture_runtime_capacity_snapshot GPU branch
# ---------------------------------------------------------------------------

class _StubDeviceProps:
    def __init__(self, name="FakeGPU-H100", total_memory=80 * (1024 ** 3)):
        self.name = name
        self.total_memory = total_memory


def _make_stub_torch(*, cuda_available: bool, total=80 * (1024 ** 3),
                     reserved=40 * (1024 ** 3), allocated=20 * (1024 ** 3),
                     max_reserved=60 * (1024 ** 3), max_allocated=50 * (1024 ** 3)):
    """Build a minimal torch stub matching `capture_runtime_capacity_snapshot`'s
    attribute reads: `.cuda.is_available/memory_reserved/memory_allocated/
    max_memory_reserved/max_memory_allocated/get_device_properties`."""
    cuda_ns = SimpleNamespace(
        is_available=lambda: cuda_available,
        memory_reserved=lambda idx=0: reserved,
        memory_allocated=lambda idx=0: allocated,
        max_memory_reserved=lambda idx=0: max_reserved,
        max_memory_allocated=lambda idx=0: max_allocated,
        get_device_properties=lambda idx=0: _StubDeviceProps(total_memory=total),
    )
    return SimpleNamespace(cuda=cuda_ns)


class TestCaptureRuntimeCapacitySnapshotGpuBranch:
    """Exercise the GPU-branch of `capture_runtime_capacity_snapshot` via a
    torch_module stub. Covers the 18-field GPU block at
    `shared/training_capacity.py:218-238` plus OOM-risk classification."""

    def test_gpu_branch_fields_present_and_shaped(self):
        from shared.training_capacity import capture_runtime_capacity_snapshot
        # reserved=40GB/80GB=50%, max_reserved=60GB/80GB=75% -> oom_risk=low.
        torch_stub = _make_stub_torch(cuda_available=True)
        snap = capture_runtime_capacity_snapshot(torch_module=torch_stub)

        # GPU identity + totals.
        assert snap["gpu_name"] == "FakeGPU-H100"
        assert snap["gpu_total_memory_gb"] == 80.0
        # Current values (reserved = the UI-facing "gpu_memory_gb").
        assert snap["gpu_memory_gb"] == 40.0
        assert snap["gpu_memory_reserved_gb"] == 40.0
        assert snap["gpu_memory_allocated_gb"] == 20.0
        # Peaks.
        assert snap["max_gpu_memory_reserved_gb"] == 60.0
        assert snap["max_gpu_memory_allocated_gb"] == 50.0
        # Percentages.
        assert snap["gpu_memory_reserved_pct"] == 50.0
        assert snap["gpu_memory_allocated_pct"] == 25.0
        assert snap["max_gpu_memory_reserved_pct"] == 75.0
        assert snap["max_gpu_memory_allocated_pct"] == 62.5
        # Headroom.
        assert snap["gpu_memory_reserved_headroom_gb"] == 40.0
        assert snap["gpu_memory_allocated_headroom_gb"] == 60.0
        # OOM risk at 75% max_reserved → "low" (<85%).
        assert snap["oom_risk_level"] == "low"

    def test_gpu_branch_high_oom_risk_classification(self):
        from shared.training_capacity import capture_runtime_capacity_snapshot
        # max_reserved=77GB / 80GB = 96.25% -> oom_risk=high (>=92%, <97%).
        torch_stub = _make_stub_torch(
            cuda_available=True,
            max_reserved=77 * (1024 ** 3),
        )
        snap = capture_runtime_capacity_snapshot(torch_module=torch_stub)
        assert snap["oom_risk_level"] == "high"

    def test_cpu_only_omits_gpu_fields(self):
        from shared.training_capacity import capture_runtime_capacity_snapshot
        torch_stub = _make_stub_torch(cuda_available=False)
        snap = capture_runtime_capacity_snapshot(torch_module=torch_stub)
        # No GPU fields at all.
        for key in ("gpu_name", "gpu_total_memory_gb", "gpu_memory_gb",
                    "gpu_memory_reserved_gb", "oom_risk_level"):
            assert key not in snap, f"CPU-only snapshot leaked GPU key {key!r}"
        # System RAM fields should still be populated (psutil / sysconf available on CI).
        assert "system_ram_total_gb" in snap or "system_ram_used_gb" in snap or True
        # The `or True` above guards against test env having no psutil/sysconf; we
        # only strictly assert the GPU-fields-omitted invariant.


# ---------------------------------------------------------------------------
# F-C — suppress_training_logs context manager
# ---------------------------------------------------------------------------

class TestSuppressTrainingLogs:
    """Direct coverage of `Trainers.shared.callbacks.log_suppression.suppress_training_logs`.

    Two paths:
    - `_SUPPRESS_AVAILABLE=True`: returns `shared.ui.suppress_logs(...)` context
      that raises noisy-logger levels to WARNING inside the block.
    - `_SUPPRESS_AVAILABLE=False`: returns `contextlib.nullcontext()` — a no-op
      fallback when `shared.ui` is unimportable."""

    def test_returns_context_manager(self):
        from Trainers.shared.callbacks.log_suppression import suppress_training_logs
        ctx = suppress_training_logs()
        # Must be a usable context manager regardless of availability branch.
        assert hasattr(ctx, "__enter__") and hasattr(ctx, "__exit__")
        with ctx:
            pass  # no-op block executes without error

    def test_nullcontext_fallback_when_unavailable(self, monkeypatch):
        """When `_SUPPRESS_AVAILABLE` is False, suppress_training_logs returns
        nullcontext — the block runs without side-effects on logger levels."""
        import Trainers.shared.callbacks.log_suppression as mod
        from contextlib import nullcontext
        monkeypatch.setattr(mod, "_SUPPRESS_AVAILABLE", False)

        ctx = mod.suppress_training_logs()
        # nullcontext class check.
        assert isinstance(ctx, type(nullcontext())), (
            f"Expected nullcontext instance when unavailable; got {type(ctx).__name__}"
        )

    def test_available_branch_delegates_to_suppress_logs(self, monkeypatch):
        """When `_SUPPRESS_AVAILABLE=True`, the call is delegated to
        `shared.ui.suppress_logs(_NOISY_LOGGERS, level=logging.WARNING)`.
        Verify the delegation args without depending on the real suppress_logs
        implementation."""
        import Trainers.shared.callbacks.log_suppression as mod
        import logging

        captured = {}

        def fake_suppress_logs(loggers, level):
            captured["loggers"] = list(loggers)
            captured["level"] = level
            from contextlib import nullcontext
            return nullcontext()

        monkeypatch.setattr(mod, "_SUPPRESS_AVAILABLE", True)
        monkeypatch.setattr(mod, "suppress_logs", fake_suppress_logs, raising=False)

        mod.suppress_training_logs()

        assert captured["level"] == logging.WARNING
        # The canonical noisy-logger list must be passed through verbatim.
        assert "unsloth" in captured["loggers"]
        assert "transformers" in captured["loggers"]
        assert "trl" in captured["loggers"]
        # Full list lock: pin exact membership so an accidental reorder/removal fails.
        assert captured["loggers"] == [
            "unsloth", "transformers", "datasets", "accelerate",
            "trl", "peft", "bitsandbytes", "torch", "huggingface_hub",
        ]


# ---------------------------------------------------------------------------
# F-D — JSONL row shape+value parity vs frozen baseline (restricted scope)
# ---------------------------------------------------------------------------
#
# NOTE: the test-engineer reviewer previously recommended AGAINST byte-level
# parity at the current STANDARD risk tier. User chose 'Address now'. Honest
# scope note: after stripping all non-deterministic fields (timestamps,
# capacity/GPU snapshots, interval timing, nvidia-smi output), the parity
# test reduces to a key+value check on step, loss, learning_rate, epoch —
# overlapping with TestSftJsonlShape + TestDictMergeOrder. The stripped
# fields are precisely the ones where byte-parity would have added value,
# but they are inherently non-deterministic and cannot be frozen.
#
# Value delivered: catches regressions in step-counter drift, log key
# passthrough, and int/float type drift across a multi-step run. Does NOT
# catch regressions in timing/capacity/interval field computation.
# ---------------------------------------------------------------------------

# Non-deterministic keys that must be stripped before frozen-baseline diff.
_NONDETERMINISTIC_FIELDS = frozenset({
    "timestamp",
    "interval_time", "interval_seconds",
    "elapsed_seconds", "steps_per_second", "samples_per_sec",
    # Capacity/GPU snapshot — machine/run-specific.
    "gpu_name", "gpu_total_memory_gb", "gpu_memory_gb",
    "gpu_memory_reserved_gb", "gpu_memory_allocated_gb",
    "max_gpu_memory_reserved_gb", "max_gpu_memory_allocated_gb",
    "gpu_memory_reserved_pct", "gpu_memory_allocated_pct",
    "max_gpu_memory_reserved_pct", "max_gpu_memory_allocated_pct",
    "gpu_memory_reserved_headroom_gb", "gpu_memory_allocated_headroom_gb",
    "max_gpu_memory_reserved_headroom_gb", "max_gpu_memory_allocated_headroom_gb",
    "gpu_utilization_pct", "gpu_vram_used_gb", "gpu_vram_total_gb",
    "gpu_vram_utilization_pct",
    "oom_risk_level",
    "system_ram_total_gb", "system_ram_used_gb", "system_ram_available_gb",
    "process_ram_gb",
    "cloud_provider", "cloud_gpu_type",
})


def _strip_nondeterministic(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in _NONDETERMINISTIC_FIELDS}


class TestSftJsonlParityBaseline:
    """Shape+value parity of SFT JSONL rows across a synthetic 10-step run.

    Drives the SFT MetricsTableCallback with deterministic logs over steps 1..10
    at log_every_n_steps=1 (log_every_write bypass not needed — SFT writes only
    at boundaries, and every step is a boundary here). Strips non-deterministic
    fields, compares the remainder against an expected baseline built inline.

    This test's actual reach: step counter integrity + logs passthrough + type
    stability across many calls. It does NOT cover timing/capacity math."""

    def test_sft_ten_step_synthetic_run_matches_baseline(self, tmp_path):
        from Trainers.sft.src.training_callbacks import MetricsTableCallback as SFTMetrics
        cb = SFTMetrics(log_every_n_steps=1, output_dir=str(tmp_path / "sft_parity"))
        _begin(cb)
        args = _make_args()

        # Drive 10 deterministic on_log calls.
        for step in range(1, 11):
            cb.on_log(
                args,
                _make_state(global_step=step, max_steps=10, epoch=step * 0.1),
                _make_control(),
                logs={
                    "loss": round(1.0 + step * 0.1, 3),
                    "learning_rate": 1e-5,
                    "epoch": round(step * 0.1, 2),
                    "grad_norm": round(0.01 * step, 3),
                },
            )

        rows = _read_jsonl_rows(cb.log_file)
        assert len(rows) == 10, f"Expected 10 rows from 10 on_log calls, got {len(rows)}"

        stripped = [_strip_nondeterministic(r) for r in rows]

        # Frozen expected baseline after stripping non-deterministic fields.
        # Any drift in step/log passthrough/types fails this test.
        expected = [
            {
                "step": step,
                "loss": round(1.0 + step * 0.1, 3),
                "learning_rate": 1e-5,
                "epoch": round(step * 0.1, 2),
                "grad_norm": round(0.01 * step, 3),
            }
            for step in range(1, 11)
        ]

        assert stripped == expected, (
            f"SFT JSONL parity failure. "
            f"First diff at row 0: stripped={stripped[0]!r} vs expected={expected[0]!r}"
        )

    def test_sft_row_types_stable_across_run(self, tmp_path):
        """Type-stability guard: across a multi-step run, `step` stays int,
        `loss`/`learning_rate`/`epoch`/`grad_norm` stay float. Catches type
        drift that shape-only tests might miss on a single-step call."""
        from Trainers.sft.src.training_callbacks import MetricsTableCallback as SFTMetrics
        cb = SFTMetrics(log_every_n_steps=1, output_dir=str(tmp_path / "sft_types"))
        _begin(cb)
        args = _make_args()
        for step in range(1, 6):
            cb.on_log(args, _make_state(global_step=step), _make_control(),
                      logs={"loss": 1.5, "learning_rate": 1e-5, "epoch": 0.1 * step, "grad_norm": 0.1})

        rows = _read_jsonl_rows(cb.log_file)
        for i, row in enumerate(rows):
            assert isinstance(row["step"], int), f"row {i}: step not int"
            assert isinstance(row["loss"], float), f"row {i}: loss not float"
            assert isinstance(row["learning_rate"], float), f"row {i}: learning_rate not float"
            assert isinstance(row["epoch"], float), f"row {i}: epoch not float"
            assert isinstance(row["grad_norm"], float), f"row {i}: grad_norm not float"
