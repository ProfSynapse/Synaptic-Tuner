import argparse
import json
import logging
import random
import sys
from pathlib import Path

import pandas as pd

from shared.experiment_tracking.experiment import load_experiment, save_experiment
from shared.judge.judge_service import JudgeService
from shared.judge.models import JudgeConfig
from shared.judge.rubric_loader import RubricLoader
from shared.llm import create_client
from shared.cloud_artifacts import setup_logging

logger = logging.getLogger(__name__)


def read_jsonl_lines(path: Path) -> dict[int, dict]:
    """Read JSONL file into a mapping of line_index -> parsed_dict."""
    lines = {}
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            lines[idx] = json.loads(line)
    return lines


def format_example_for_judge(example: dict) -> tuple[str, str]:
    """Extract user request and assistant response from ChatML format."""
    messages = example.get("conversations", example.get("messages", []))
    
    user_req = ""
    assistant_resp = ""
    
    for m in messages:
        if m.get("role") == "user":
            user_req += m.get("content", "") + "\n"
        elif m.get("role") == "assistant":
            assistant_resp += m.get("content", "") + "\n"
            
    return user_req.strip(), assistant_resp.strip()


def process_sample(
    base_dir: Path,
    experiment_id: str,
    sample_size: int,
    rubric_key: str = "data_quality"
):
    exp = load_experiment(experiment_id, base_dir=base_dir)
    if not exp.features_csv_path:
        logger.error(f"Experiment {experiment_id} has no features_csv_path set.")
        sys.exit(1)
        
    features_path = Path(exp.features_csv_path)
    if not features_path.is_absolute():
        features_path = base_dir.parent / features_path
        
    if not features_path.exists():
        logger.error(f"Features CSV not found at {features_path}")
        sys.exit(1)
        
    df = pd.read_csv(features_path)
    if "index" not in df.columns:
        logger.error("features.csv misses 'index' column.")
        sys.exit(1)
        
    if "judge_quality" not in df.columns:
        df["judge_quality"] = pd.NA
        
    # Find candidates that haven't been judged yet
    unjudged_mask = pd.isna(df["judge_quality"])
    unjudged_indices = df[unjudged_mask]["index"].tolist()
    
    if not unjudged_indices:
        logger.info("All examples have already been judged. Nothing to do.")
        return
        
    target_count = min(sample_size, len(unjudged_indices))
    sampled_indices = sorted(random.sample(unjudged_indices, target_count))
    logger.info(f"Sampled {target_count} examples out of {len(unjudged_indices)} unjudged.")
    
    # Read the dataset
    dataset_path = Path(exp.dataset_path)
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        sys.exit(1)
        
    logger.info("Loading dataset...")
    all_lines = read_jsonl_lines(dataset_path)
    
    # Load Judge
    llm_client = create_client()
    judge_config = JudgeConfig()
    judge_service = JudgeService(llm_client, judge_config)
    rubric_loader = RubricLoader(Path("SynthChat/rubrics"))
    rubric = rubric_loader.load(rubric_key)
    
    judge_scores_path = experiment_dir / experiment_id / "judge_scores.jsonl"
    
    logger.info(f"Running LLM Judge ({llm_client.model}) using {rubric_key} rubric...")
    
    # Append to JSONL output
    with open(judge_scores_path, "a", encoding="utf-8") as out_f:
        for idx in sampled_indices:
            example = all_lines.get(idx)
            if not example:
                logger.warning(f"Index {idx} not found in dataset. Skipping.")
                continue
                
            user_req, curr_content = format_example_for_judge(example)
            
            prompt = rubric.judge_prompt.format(
                user_request=user_req,
                current_content=curr_content
            )
            
            result = judge_service.judge(prompt=prompt, rubrics=[rubric])
            
            # The schema outputs "data_quality_score" based on our rubric
            score = None
            if result.scores and len(result.scores) > 0:
                score = result.scores[0].score
                
            out_f.write(json.dumps({
                "index": idx,
                "score": score,
                "judge_quality": score,
                "raw_output": result.raw_output
            }) + "\n")
            
            # Update dataframe
            if score is not None:
                df.loc[df["index"] == idx, "judge_quality"] = score
                
            logger.info(f"Index {idx} scored: {score}")

    df.to_csv(features_path, index=False)
    logger.info(f"Updated features CSV: {features_path}")
    
    exp.judge_scores_path = f"experiments/{experiment_id}/judge_scores.jsonl"
    save_experiment(exp, experiment_dir / experiment_id)
    logger.info(f"Updated experiment {experiment_id}")


def main():
    parser = argparse.ArgumentParser(description="Sample runs and evaluate via LLM Judge.")
    parser.add_argument("--experiment-id", required=True, help="Experiment ID to process.")
    parser.add_argument("--experiment-dir", type=str, default=".tracking/experiments", help="Base tracking dir.")
    parser.add_argument("--sample-size", type=int, default=50, help="Number of samples to evaluate.")
    parser.add_argument("--rubric", type=str, default="data_quality", help="Rubric key to use.")
    
    args = parser.parse_args()
    setup_logging()
    
    process_sample(
        experiment_dir=Path(args.experiment_dir).parent / "experiments",  # fallback if they pass something else
        experiment_id=args.experiment_id,
        sample_size=args.sample_size,
        rubric_key=args.rubric
    )


if __name__ == "__main__":
    main()
