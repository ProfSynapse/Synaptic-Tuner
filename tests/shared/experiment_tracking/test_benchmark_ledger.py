from __future__ import annotations

import csv
import json
from pathlib import Path

import shared.experiment_tracking.benchmark_ledger as benchmark_ledger
from shared.experiment_tracking.experiment import Experiment
from shared.experiment_tracking.schema import LossResult, RunRecord


def _minimal_experiment() -> Experiment:
    return Experiment(
        experiment_id="exp_isolation",
        name="isolation",
        created_at="2026-03-22T20:00:00+00:00",
        dataset_path="repo/dataset_variant.jsonl",
        dataset_hash="hash",
        base_model_name="Qwen/Qwen3-4B",
        provider="hf_jobs",
        method="sft",
        objective="speed_cost_round",
        status="completed",
    )


def test_ledger_dir_env_var_redirects_write_away_from_repo_root(tmp_path, monkeypatch):
    """The BENCHMARK_LEDGER_DIR seam redirects the write off repo_root.

    With the env override set, a real repo_root passed to upsert must NOT be
    written under — the ledger lands in the override dir instead, so a test run
    can never pollute the committed docs/benchmarks CSV. (The autouse
    isolate_benchmark_ledger fixture already sets the var; this test pins the
    contract directly with its own override dir.)
    """
    monkeypatch.setattr(benchmark_ledger, "load_live_hf_hardware_rows", lambda: [])
    override_dir = tmp_path / "override"
    fake_repo_root = tmp_path / "repo_root"
    fake_repo_root.mkdir()
    monkeypatch.setenv(benchmark_ledger.LEDGER_DIR_ENV_VAR, str(override_dir))

    ledger_path = benchmark_ledger.upsert_benchmark_ledger(
        repo_root=fake_repo_root,
        experiment=_minimal_experiment(),
        runs=[],
    )

    # Written under the override dir, not the passed repo_root.
    assert Path(ledger_path) == override_dir / benchmark_ledger.LEDGER_CSV_RELATIVE_PATH
    assert Path(ledger_path).exists()
    assert not (fake_repo_root / benchmark_ledger.LEDGER_CSV_RELATIVE_PATH).exists()


def test_upsert_benchmark_ledger_materializes_row(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(benchmark_ledger, "LEDGER_CSV_RELATIVE_PATH", Path("docs/benchmarks/model_hardware_benchmark_ledger.csv"))
    monkeypatch.setattr(benchmark_ledger, "load_live_hf_hardware_rows", lambda: [])

    training_dir = tmp_path / "train"
    training_dir.mkdir(parents=True)
    (training_dir / "training_lineage.json").write_text(
        json.dumps(
            {
                "training": {
                    "num_epochs": 2,
                    "max_seq_length": 2048,
                    "batch_size": 12,
                    "gradient_accumulation_steps": 3,
                    "effective_batch_size": 36,
                },
                "results": {"training_time_seconds": 123.4, "final_loss": 0.456},
            }
        ),
        encoding="utf-8",
    )

    experiment = Experiment(
        experiment_id="exp_test_benchmark",
        name="benchmark",
        created_at="2026-03-22T20:00:00+00:00",
        dataset_path="repo/dataset_variant.jsonl",
        dataset_hash="hash",
        base_model_name="Qwen/Qwen3-4B",
        provider="hf_jobs",
        method="sft",
        objective="speed_cost_round",
        training_run_id="run-training",
        evaluation_run_id="run-eval",
        loss_run_id="run-loss",
        status="completed",
        stage_statuses={"training": "completed", "evaluation": "completed", "loss": "completed"},
        stage_details={
            "training": {"status": "completed", "started_at": "2026-03-22T20:00:00+00:00", "finished_at": "2026-03-22T20:02:03+00:00"},
            "evaluation": {"status": "completed", "started_at": "2026-03-22T20:02:03+00:00", "finished_at": "2026-03-22T20:04:03+00:00"},
            "loss": {"status": "completed", "started_at": "2026-03-22T20:04:03+00:00", "finished_at": "2026-03-22T20:05:03+00:00"},
        },
    )
    runs = [
        RunRecord(
            run_id="run-training",
            run_type="sft",
            name="training",
            timestamp="2026-03-22T20:02:03+00:00",
            status="completed",
            output_dir=str(training_dir),
            artifact_root=str(training_dir),
            provider="hf_jobs",
            stage="training",
            hardware="a100-large",
        ),
        RunRecord(
            run_id="run-eval",
            run_type="evaluation",
            name="evaluation",
            timestamp="2026-03-22T20:04:03+00:00",
            status="completed",
            output_dir="hf://buckets/test/eval",
            artifact_root="hf://buckets/test/eval",
            provider="hf_jobs",
            stage="evaluation",
            hardware="a100-large",
        ),
        RunRecord(
            run_id="run-loss",
            run_type="loss",
            name="loss",
            timestamp="2026-03-22T20:05:03+00:00",
            status="completed",
            output_dir="hf://buckets/test/loss",
            artifact_root="hf://buckets/test/loss",
            provider="hf_jobs",
            stage="loss",
            hardware="a100x4",
        ),
    ]
    eval_payload = {"summary": {"passed": 10, "failed": 2, "warned": 1, "total": 13, "pass_rate": 10 / 13, "schema_pass_rate": 12 / 13}}
    loss_results = [
        LossResult(index=0, loss=1.0, num_completion_tokens=5, num_total_tokens=10, jsonl_hash="aaa"),
        LossResult(index=1, loss=0.5, num_completion_tokens=6, num_total_tokens=11, jsonl_hash="bbb"),
    ]

    ledger_path = benchmark_ledger.upsert_benchmark_ledger(
        repo_root=tmp_path,
        experiment=experiment,
        runs=runs,
        eval_payload=eval_payload,
        loss_results=loss_results,
    )

    assert Path(ledger_path).exists()
    with Path(ledger_path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["experiment_id"] == "exp_test_benchmark"
    assert rows[0]["train_flavor"] == "a100-large"
    assert rows[0]["loss_flavor"] == "a100x4"
    assert rows[0]["eval_passed"] == "10"


def test_upsert_benchmark_ledger_prefers_stage_lineage_payloads(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(benchmark_ledger, "LEDGER_CSV_RELATIVE_PATH", Path("docs/benchmarks/model_hardware_benchmark_ledger.csv"))
    monkeypatch.setattr(benchmark_ledger, "load_live_hf_hardware_rows", lambda: [])

    training_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval"
    loss_dir = tmp_path / "loss"
    training_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)
    loss_dir.mkdir(parents=True)

    (training_dir / "training_lineage.json").write_text(
        json.dumps(
            {
                "training": {"num_epochs": 2, "batch_size": 27, "gradient_accumulation_steps": 2, "effective_batch_size": 54},
                "results": {"training_time_seconds": 100.0, "final_loss": 0.55},
                "planner": {"resolved_hardware_flavor": "a100-large"},
                "pricing": {"estimated_cost_usd": 2.0},
                "runtime": {"duration_seconds": 100.0},
            }
        ),
        encoding="utf-8",
    )
    (eval_dir / "evaluation_lineage.json").write_text(
        json.dumps(
            {
                "results_summary": {"passed": 45, "failed": 22, "warned": 10, "total": 77, "schema_pass_rate": 55},
                "execution": {"hardware_flavor": "a100-large"},
                "pricing": {"estimated_cost_usd": 1.1},
                "runtime": {"duration_seconds": 90.0},
            }
        ),
        encoding="utf-8",
    )
    (loss_dir / "loss_lineage.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "execution": {"hardware_flavor": "a100x4"},
                "pricing": {"estimated_cost_usd": 3.2},
                "runtime": {"duration_seconds": 45.0},
                "results": {"row_count": 9378, "mean_loss": 0.42},
            }
        ),
        encoding="utf-8",
    )

    experiment = Experiment(
        experiment_id="exp_lineage_preferred",
        name="benchmark",
        created_at="2026-03-22T20:00:00+00:00",
        dataset_path="repo/dataset_variant.jsonl",
        dataset_hash="hash",
        base_model_name="Qwen/Qwen3-4B",
        provider="hf_jobs",
        method="sft",
        objective="speed_cost_round",
        status="completed",
    )
    runs = [
        RunRecord(run_id="train", run_type="sft", name="train", timestamp="2026-03-22T20:00:00+00:00", status="completed", output_dir=str(training_dir), artifact_root=str(training_dir), provider="hf_jobs", stage="training"),
        RunRecord(run_id="eval", run_type="evaluation", name="eval", timestamp="2026-03-22T20:02:00+00:00", status="completed", output_dir=str(eval_dir), artifact_root=str(eval_dir), provider="hf_jobs", stage="evaluation"),
        RunRecord(run_id="loss", run_type="loss", name="loss", timestamp="2026-03-22T20:04:00+00:00", status="completed", output_dir=str(loss_dir), artifact_root=str(loss_dir), provider="hf_jobs", stage="loss"),
    ]

    ledger_path = benchmark_ledger.upsert_benchmark_ledger(
        repo_root=tmp_path,
        experiment=experiment,
        runs=runs,
        eval_payload=None,
        loss_results=None,
    )

    with Path(ledger_path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["eval_passed"] == "45"
    assert rows[0]["loss_examples"] == "9378"
    assert rows[0]["loss_mean"] == "0.42"


def test_neutralize_csv_cell_prefixes_formula_triggers_only():
    """The cell-level helper single-quotes every spreadsheet formula trigger and
    leaves everything else byte-identical (and non-strings untouched)."""
    for trigger in ("=", "+", "-", "@", "\t", "\r"):
        payload = f"{trigger}SUM(A1:A9)"
        assert benchmark_ledger._neutralize_csv_cell(payload) == "'" + payload

    # Normal text, empty string, and non-string scalars must pass through as-is so
    # numeric columns stay numeric and well-formed cells are unchanged.
    assert benchmark_ledger._neutralize_csv_cell("a100-large") == "a100-large"
    assert benchmark_ledger._neutralize_csv_cell("") == ""
    assert benchmark_ledger._neutralize_csv_cell(0.42) == 0.42
    assert benchmark_ledger._neutralize_csv_cell(13) == 13
    assert benchmark_ledger._neutralize_csv_cell(None) is None

    # Numeric strings with a leading sign (read back from a prior CSV row) keep
    # their value — the +/- is a numeric sign, not a formula trigger.
    assert benchmark_ledger._neutralize_csv_cell("-0.5") == "-0.5"
    assert benchmark_ledger._neutralize_csv_cell("-5") == "-5"
    assert benchmark_ledger._neutralize_csv_cell("+1.0") == "+1.0"
    # But a sign-led NON-number (formula payload) is still neutralized.
    assert benchmark_ledger._neutralize_csv_cell("-cmd|calc") == "'-cmd|calc"
    assert benchmark_ledger._neutralize_csv_cell("+SUM(A1)") == "'+SUM(A1)"


def test_upsert_benchmark_ledger_preserves_negative_numeric_across_roundtrip(tmp_path: Path, monkeypatch):
    """A negative numeric cell (e.g. a -0.5 loss_mean) must survive a second
    upsert un-neutralized — the round-trip read-back as the string "-0.5" must
    NOT be mistaken for a formula trigger (regression guard for the leading-'-'
    collision)."""
    monkeypatch.setattr(benchmark_ledger, "load_live_hf_hardware_rows", lambda: [])

    negative_loss = [
        LossResult(index=0, loss=-0.5, num_completion_tokens=5, num_total_tokens=10, jsonl_hash="a"),
    ]
    first = Experiment(
        experiment_id="exp_negative",
        name="neg",
        created_at="2026-03-22T20:00:00+00:00",
        dataset_path="repo/dataset_variant.jsonl",
        dataset_hash="hash",
        base_model_name="Qwen/Qwen3-4B",
        provider="hf_jobs",
        method="sft",
        objective="speed_cost_round",
        status="completed",
    )
    benchmark_ledger.upsert_benchmark_ledger(
        repo_root=tmp_path, experiment=first, runs=[], loss_results=negative_loss
    )

    # A second upsert of a DIFFERENT experiment rewrites the whole file, so the
    # first row's -0.5 is read back as a string and re-passed through the writer.
    second = Experiment(
        experiment_id="exp_other_row",
        name="other",
        created_at="2026-03-22T20:00:00+00:00",
        dataset_path="repo/dataset_variant.jsonl",
        dataset_hash="hash",
        base_model_name="Qwen/Qwen3-4B",
        provider="hf_jobs",
        method="sft",
        objective="speed_cost_round",
        status="completed",
    )
    ledger_path = benchmark_ledger.upsert_benchmark_ledger(
        repo_root=tmp_path, experiment=second, runs=[]
    )

    with Path(ledger_path).open("r", encoding="utf-8", newline="") as handle:
        rows = {r["experiment_id"]: r for r in csv.DictReader(handle)}

    assert rows["exp_negative"]["loss_mean"] == "-0.5"  # NOT "'-0.5"


def test_upsert_benchmark_ledger_neutralizes_formula_injection_in_notes(tmp_path: Path, monkeypatch):
    """A notes value originating from a stage error_message that begins with a
    formula trigger round-trips NEUTRALIZED (single-quote-prefixed), while the
    dedup key and normal cells are unchanged (FS1, PR #1 security review)."""
    monkeypatch.setattr(benchmark_ledger, "load_live_hf_hardware_rows", lambda: [])

    injection = "=SUM(1+9)*cmd|'/c calc'!A1"
    experiment = Experiment(
        experiment_id="exp_injection",
        name="injection",
        created_at="2026-03-22T20:00:00+00:00",
        dataset_path="repo/dataset_variant.jsonl",
        dataset_hash="hash",
        base_model_name="Qwen/Qwen3-4B",
        provider="hf_jobs",
        method="sft",
        objective="speed_cost_round",
        status="failed",
        stage_details={
            # A failed stage routes its error_message into the notes column
            # (build_benchmark_ledger_row :263-270) — the realistic free-text sink.
            "loss": {"status": "failed", "error_message": injection},
        },
    )

    ledger_path = benchmark_ledger.upsert_benchmark_ledger(
        repo_root=tmp_path,
        experiment=experiment,
        runs=[],
    )

    with Path(ledger_path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    # The dangerous notes cell is neutralized: the spreadsheet sees literal text.
    assert rows[0]["notes"] == "'" + injection
    # The dedup key (never a free-text trigger) round-trips untouched, so upsert
    # matching is unaffected.
    assert rows[0]["experiment_id"] == "exp_injection"


def test_upsert_benchmark_ledger_dedup_survives_neutralization(tmp_path: Path, monkeypatch):
    """Upserting the same experiment_id twice replaces (not appends) even though
    the notes column carries a neutralized formula trigger — confirms the write-
    boundary neutralization does not perturb the experiment_id dedup key."""
    monkeypatch.setattr(benchmark_ledger, "load_live_hf_hardware_rows", lambda: [])

    def _experiment_with_note(note: str) -> Experiment:
        return Experiment(
            experiment_id="exp_dedup",
            name="dedup",
            created_at="2026-03-22T20:00:00+00:00",
            dataset_path="repo/dataset_variant.jsonl",
            dataset_hash="hash",
            base_model_name="Qwen/Qwen3-4B",
            provider="hf_jobs",
            method="sft",
            objective="speed_cost_round",
            status="failed",
            stage_details={"loss": {"status": "failed", "error_message": note}},
        )

    benchmark_ledger.upsert_benchmark_ledger(
        repo_root=tmp_path, experiment=_experiment_with_note("=first"), runs=[]
    )
    ledger_path = benchmark_ledger.upsert_benchmark_ledger(
        repo_root=tmp_path, experiment=_experiment_with_note("=second"), runs=[]
    )

    with Path(ledger_path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    # Exactly one row (replaced, not duplicated), carrying the latest neutralized note.
    assert len(rows) == 1
    assert rows[0]["experiment_id"] == "exp_dedup"
    assert rows[0]["notes"] == "'=second"
