"""Targeted tests for the sft/kto/grpo callback DRY refactor.

Verifies the four uncertainties flagged by the architect and coder on the
callback refactor design (docs/architecture/training-callbacks-refactor.md):

  1. [MEDIUM] KTO cloud_provider env-first resolution (intentional additive
     change for KTO per §6 — env wins over args).
  2. [MEDIUM] `log_every_write` cadence knob — SFT writes only at
     log_every_n_steps boundary; KTO writes every on_log.
  3. [MEDIUM] GRPO dict-merge order — `**logs` spreads last, so log keys
     override base's fixed fields (coder's unification direction).
  4. [HIGH] GRPO `interval_seconds` key preservation — not `interval_time`.

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
    """Coder unified the entry-dict build so `**logs` is the LAST spread,
    meaning any key present in `logs` overrides base's fixed field. This is
    intentional: trainers produce canonical log rows (e.g. TRL's own `step`
    from its internal log callback) that should shine through."""

    def test_logs_step_overrides_state_step_grpo(self, grpo_callback):
        _begin(grpo_callback)
        args = _make_args()
        # State says step=1 but logs carry step=42 (simulating TRL-emitted logs).
        grpo_callback.on_log(args, _make_state(global_step=1), _make_control(),
                             logs={"loss": 1.0, "step": 42})
        rows = _read_jsonl_rows(grpo_callback.log_file)
        assert rows[0]["step"] == 42, (
            f"logs['step'] must win over state.global_step; got {rows[0]['step']}"
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
