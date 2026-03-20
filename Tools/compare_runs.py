"""
Tools/compare_runs.py

Post-hoc inference pipeline: after training completes, the trained model is run over 
each JSONL example sequentially. A comparison script joins all per-run loss files 
into a stable feature matrix (features.csv).
"""

import argparse
import sys
from pathlib import Path
import json
import logging

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.experiment_tracking import load_experiment, load_losses, save_experiment

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("Compare runs and generate feature matrix for LightGBM.")
    parser.add_argument("--experiment-id", required=True, help="Experiment ID to process")
    parser.add_argument("--base-dir", default=".tracking", help="Tracking base directory")
    return parser.parse_args()

def main(args=None):
    if args is None:
        args = parse_args()
        
    logging.basicConfig(level=logging.INFO)
    
    experiment = load_experiment(args.experiment_id, args.base_dir)
    
    if not experiment.base_losses_path:
        raise ValueError("Experiment base_losses_path is empty. Run compute-losses on base model first.")
        
    logger.info(f"Loading base losses from {experiment.base_losses_path}")
    base_losses_path = Path(args.base_dir).parent / experiment.base_losses_path
    if not base_losses_path.exists():
        raise FileNotFoundError(f"Base losses not found: {base_losses_path}")
        
    base_results = load_losses(base_losses_path)
    base_df = pd.DataFrame([loss.__dict__ for loss in base_results])
    base_df = base_df.rename(columns={"loss": "loss_base"})
    
    run_dfs = []
    for i, run_id in enumerate(experiment.run_ids):
        # find the run losses
        run_loss_path = Path(args.base_dir) / run_id / "per_example_losses.jsonl"
        if not run_loss_path.exists():
            logger.warning(f"Losses not found for run {run_id} at {run_loss_path}, skipping.")
            continue
            
        run_results = load_losses(run_loss_path)
        df = pd.DataFrame([loss.__dict__ for loss in run_results])
        df = df[["index", "loss", "num_completion_tokens"]]
        df = df.rename(columns={"loss": f"loss_run_{i}"})
        run_dfs.append((i, df))
        
    if not run_dfs:
        raise ValueError("No run loss data found. Need at least one run.")
        
    # merge
    merged_df = base_df[["index", "loss_base", "num_completion_tokens", "num_total_tokens", "jsonl_hash"]]
    
    for i, df in run_dfs:
        merged_df = merged_df.merge(df[["index", f"loss_run_{i}"]], on="index", how="left")
        merged_df[f"loss_delta_run_{i}"] = merged_df[f"loss_run_{i}"] - merged_df["loss_base"]
        
    # compute aggregations across runs
    run_cols = [f"loss_run_{i}" for i, _ in run_dfs]
    delta_cols = [f"loss_delta_run_{i}" for i, _ in run_dfs]
    
    merged_df["loss_mean"] = merged_df[run_cols].mean(axis=1)
    merged_df["loss_std"] = merged_df[run_cols].std(axis=1)
    merged_df["loss_min"] = merged_df[run_cols].min(axis=1)
    merged_df["loss_max"] = merged_df[run_cols].max(axis=1)
    merged_df["delta_mean"] = merged_df[delta_cols].mean(axis=1)
    
    out_csv = Path(args.base_dir) / "experiments" / experiment.experiment_id / "features.csv"
    merged_df.to_csv(out_csv, index=False)
    logger.info(f"Features saved to {out_csv}")
    
    experiment.features_csv_path = f"experiments/{experiment.experiment_id}/features.csv"
    save_experiment(experiment, args.base_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main())
