import json
import logging
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest
import pandas as pd

from shared.experiment_tracking.experiment import Experiment, save_experiment

REPO_ROOT = Path(__file__).resolve().parents[3]
_SPEC = spec_from_file_location("compare_runs", REPO_ROOT / "Tools" / "compare_runs.py")
_MODULE = module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_MODULE)
compare_runs_main = _MODULE.main

def test_compare_runs_basic(tmp_path, fake_base_losses_path, sample_losses_a_path, sample_losses_b_path):
    """Join 3 loss files (base + 2 runs) → correct feature matrix shape and column names."""
    base_dir = tmp_path / ".tracking"
    base_dir.mkdir()
    exp_dir = base_dir / "experiments" / "exp_1"
    exp_dir.mkdir(parents=True)
    
    # Copy fixtures to expected locations
    base_dest = exp_dir / "base_losses.jsonl"
    base_dest.write_text(fake_base_losses_path.read_text())
    
    run_a_dir = base_dir / "run_A"
    run_a_dir.mkdir()
    run_a_dest = run_a_dir / "per_example_losses.jsonl"
    run_a_dest.write_text(sample_losses_a_path.read_text())
    
    run_b_dir = base_dir / "run_B"
    run_b_dir.mkdir()
    run_b_dest = run_b_dir / "per_example_losses.jsonl"
    run_b_dest.write_text(sample_losses_b_path.read_text())

    # Create experiment file
    exp = Experiment(
        experiment_id="exp_1",
        name="test_exp",
        created_at="now",
        dataset_path="a/dataset.jsonl",
        dataset_hash="hash",
        base_model_name="base-model",
        run_ids=["run_A", "run_B"],
    )
    # The path should be relative to parent directory if following code in compare_runs:
    # "Path(args.base_dir).parent / experiment.base_losses_path" -> Wait!
    # Let's ensure base_losses_path maps correctly. compare_runs has a line:
    # Path(args.base_dir).parent / experiment.base_losses_path
    # Oh! `Path(args.base_dir).parent` ... Let's just set it so it resolves properly.
    # If base_dir is `.../.tracking`, parent is `.../`. 
    # So `experiment.base_losses_path` should be `.tracking/experiments/exp_1/base_losses.jsonl`
    
    exp.base_losses_path = ".tracking/experiments/exp_1/base_losses.jsonl"
    save_experiment(exp, str(base_dir))
    
    args = SimpleNamespace(
        experiment_id="exp_1",
        base_dir=str(base_dir)
    )
    
    res = compare_runs_main(args)
    assert res == 0
    
    out_csv = exp_dir / "features.csv"
    assert out_csv.exists()
    
    df = pd.read_csv(out_csv)
    # the 3 examples match
    assert len(df) == 3
    
    expected_cols = {
        "index", "loss_base", "num_completion_tokens", "num_total_tokens", "jsonl_hash",
        "loss_run_0", "loss_delta_run_0",
        "loss_run_1", "loss_delta_run_1",
        "loss_mean", "loss_std", "loss_min", "loss_max", "delta_mean"
    }
    
    assert set(df.columns.tolist()).issuperset(expected_cols)

def test_single_run_features(tmp_path, fake_base_losses_path, sample_losses_a_path):
    """Single run -> cross-model features are NaN or computed properly."""
    base_dir = tmp_path / ".tracking"
    base_dir.mkdir()
    exp_dir = base_dir / "experiments" / "exp_1"
    exp_dir.mkdir(parents=True)
    
    base_dest = exp_dir / "base_losses.jsonl"
    base_dest.write_text(fake_base_losses_path.read_text())
    
    run_a_dir = base_dir / "run_A"
    run_a_dir.mkdir()
    run_a_dest = run_a_dir / "per_example_losses.jsonl"
    run_a_dest.write_text(sample_losses_a_path.read_text())
    
    exp = Experiment(
        experiment_id="exp_1",
        name="test_exp",
        created_at="now",
        dataset_path="a/dataset.jsonl",
        dataset_hash="hash",
        base_model_name="base-model",
        run_ids=["run_A"],
    )
    exp.base_losses_path = ".tracking/experiments/exp_1/base_losses.jsonl"
    save_experiment(exp, str(base_dir))
    
    args = SimpleNamespace(experiment_id="exp_1", base_dir=str(base_dir))
    
    res = compare_runs_main(args)
    assert res == 0
    
    out_csv = exp_dir / "features.csv"
    df = pd.read_csv(out_csv)
    assert "loss_run_0" in df.columns
    assert "loss_run_1" not in df.columns
    
    # std with one run is NaN in pandas by default
    assert df["loss_std"].isna().all()

def test_missing_run_losses_handled(tmp_path, fake_base_losses_path):
    """Test when run_ids has missing per_example_losses."""
    base_dir = tmp_path / ".tracking"
    base_dir.mkdir()
    exp_dir = base_dir / "experiments" / "exp_1"
    exp_dir.mkdir(parents=True)
    
    base_dest = exp_dir / "base_losses.jsonl"
    base_dest.write_text(fake_base_losses_path.read_text())
    
    exp = Experiment(
        experiment_id="exp_1",
        name="test_exp",
        created_at="now",
        dataset_path="a/dataset.jsonl",
        dataset_hash="hash",
        base_model_name="base-model",
        run_ids=["run_missing"],
    )
    exp.base_losses_path = ".tracking/experiments/exp_1/base_losses.jsonl"
    save_experiment(exp, str(base_dir))
    
    args = SimpleNamespace(experiment_id="exp_1", base_dir=str(base_dir))
    
    with pytest.raises(ValueError, match="No run loss data found"):
        compare_runs_main(args)
