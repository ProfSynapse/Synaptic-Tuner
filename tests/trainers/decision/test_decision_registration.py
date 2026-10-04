"""`decision` method registration: SSOT, menu label, tracking adapter, recipes.

decision models are scored by their own calibration evaluation, never served
through the causal-LM eval backends, so they stay out of those tuples (the same
asymmetry as embedding and ace_step).
"""
from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RECIPES = [
    "Trainers/recipes/decision_strands_corpus_build.yaml",
    "Trainers/recipes/decision_qwen35_2b_pointer_smoke.yaml",
    "Trainers/recipes/decision_qwen35_2b_letter_logits_smoke.yaml",
]


def test_decision_in_training_methods_ssot():
    from shared.utilities.paths import CANONICAL_OUTPUT_DIRS, TRAINING_METHODS

    assert "decision" in TRAINING_METHODS
    assert CANONICAL_OUTPUT_DIRS["decision"] == "decision_output"
    assert (REPO_ROOT / "Trainers" / "decision" / "train_decision.py").exists()


def test_decision_menu_label():
    labels = yaml.safe_load((REPO_ROOT / "Trainers" / "methods.yaml").read_text(encoding="utf-8"))
    assert "decision" in labels["method_labels"]


def test_experiment_spec_accepts_decision():
    pytest.importorskip("transformers")  # shared.experiment_tracking imports it transitively
    from shared.experiment_tracking.experiment_spec import DatasetSpec, ExperimentSpec, TrainingStageSpec

    def issues(method: str) -> list[str]:
        return ExperimentSpec(
            name="t",
            provider="local",
            method=method,
            dataset=DatasetSpec(source="hf", file="d.jsonl"),
            training=TrainingStageSpec(model_name="m"),
        ).validate()

    assert not any("unsupported method" in i for i in issues("decision"))
    assert any("unsupported method" in i for i in issues("not_a_method"))


def test_eval_backends_do_not_serve_decision():
    for rel in (
        "tuner/backends/evaluation/unsloth_backend.py",
        "tuner/backends/evaluation/mlc_backend.py",
        "tuner/backends/evaluation/llamacpp_backend.py",
    ):
        source = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert 'for method in ("sft", "kto", "grpo", "dpo"):' in source
        assert '"decision")' not in source


def test_run_record_adapter_prefers_eval_accuracy():
    pytest.importorskip("transformers")  # shared.experiment_tracking imports it transitively
    from shared.experiment_tracking.adapters import decision_lineage_to_run_record

    lineage = {
        "timestamp": "2026-10-04T12:00:00",
        "model": {"base_model": "Qwen/Qwen3.5-2B-Base"},
        "dataset": {"source": "train_v5.jsonl"},
        "hardware": {"gpu_name": "RTX 3090"},
        "results": {"final_loss": 0.8, "eval_accuracy": 0.71},
    }
    record = decision_lineage_to_run_record(lineage, "/tmp/run")
    assert record.run_type == "decision"
    assert record.primary_metric == 0.71 and record.primary_metric_name == "eval_accuracy"
    del lineage["results"]["eval_accuracy"]
    record = decision_lineage_to_run_record(lineage, "/tmp/run", cloud=True)
    assert record.run_type == "cloud_decision" and record.primary_metric_name == "final_loss"


@pytest.mark.parametrize("rel", RECIPES)
def test_recipes_compile(rel):
    from tuner.handlers.local_run_handler import LocalRunHandler

    path = REPO_ROOT / rel
    handler = LocalRunHandler(args=Namespace(json=True, job_config=str(path)))
    plan = handler._compile(path, handler._load_yaml(path))
    command = plan["command"]
    assert command[0] == "python" and command[1].startswith("Trainers/decision/")
    assert "{" not in str(plan["host_artifact_path"]), "artifact host_path templates must render"
    if "--run-timestamp" in command:
        stamp = command[command.index("--run-timestamp") + 1]
        root = command[command.index("--output-root") + 1]
        # The trainer writes <output-root>/<run-timestamp>, exactly the copied-back path.
        assert plan["host_artifact_path"] == (REPO_ROOT / root / stamp)
        config = command[command.index("--config") + 1]
        assert (REPO_ROOT / config).exists()


def test_corpus_builder_dry_run(capsys):
    sys.path.insert(0, str(REPO_ROOT / "Trainers" / "decision"))
    import build_corpus

    assert build_corpus.main(["--config", "Trainers/decision/configs/corpus_strands_v5.yaml", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "data build" in out and "--holdout emotion" in out and "-r banking77" in out
