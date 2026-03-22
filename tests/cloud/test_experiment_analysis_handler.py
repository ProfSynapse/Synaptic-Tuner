from __future__ import annotations

import json
from argparse import Namespace

from shared.experiment_tracking.experiment import Experiment, save_experiment
from tuner.handlers.experiment_analysis_handler import ExperimentAnalysisHandler


def test_experiment_analysis_handler_reports_saved_bundle(tmp_path, capsys):
    experiment = Experiment(
        experiment_id="exp_20260322_010203",
        name="analysis-smoke",
        created_at="2026-03-22T01:02:03+00:00",
        dataset_path="repo/dataset.jsonl",
        dataset_hash="abc123",
        base_model_name="Qwen/Qwen3-4B",
        provider="hf_jobs",
        method="sft",
        objective="full_cycle",
        status="completed",
        stage_statuses={"training": "completed", "evaluation": "completed", "loss": "completed"},
        derived_outputs={},
    )
    save_experiment(experiment, base_dir=tmp_path)
    analysis_dir = tmp_path / "experiments" / experiment.experiment_id / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    summary_path = analysis_dir / "experiment_summary.json"
    summary_path.write_text(json.dumps({"eval_summary": {"passed": 4, "warned": 1, "failed": 2}}), encoding="utf-8")
    candidates_path = analysis_dir / "next_run_candidates.json"
    candidates_path.write_text(
        json.dumps(
            {
                "candidates": [
                    {"rank": 1, "hypothesis": "Increase epochs slightly."},
                    {"rank": 2, "hypothesis": "Lower learning rate by 20%."},
                ]
            }
        ),
        encoding="utf-8",
    )

    experiment.derived_outputs["experiment_summary_json"] = str(summary_path)
    experiment.derived_outputs["next_run_candidates_json"] = str(candidates_path)
    save_experiment(experiment, base_dir=tmp_path)

    handler = ExperimentAnalysisHandler(
        args=Namespace(
            json=True,
            experiment_id=experiment.experiment_id,
            base_dir=str(tmp_path),
        )
    )
    exit_code = handler.handle()

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["success"] is True
    assert payload["data"]["experiment_id"] == experiment.experiment_id
    assert payload["data"]["top_candidates"][0]["hypothesis"] == "Increase epochs slightly."
