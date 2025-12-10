#!/usr/bin/env python3
"""
Parallel batch improvement - runs multiple examples concurrently.

CLOUD ONLY: Uses async/concurrent requests to OpenRouter.
"""

import sys
import json
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.llm import create_client
from improvement_engine.rubrics import load_rubric
from improvement_engine.services.judge_service import JudgeService
from improvement_engine.utils.logger import ImproveLogger
from improvement_engine.utils.yaml_loader import load_config

# Rubrics to use
JUDGE_RUBRICS = [
    "confidence_calibration",
    "context_alignment",
    "destructive_safety",
    "factuality",
    "requirements_plan",
    "tool_alignment"
]


def evaluate_single_example(args: Tuple) -> dict:
    """
    Evaluate one example with all rubrics (thread-safe).

    Args:
        args: Tuple of (line_num, example, llm_client, logger)

    Returns:
        Dict with results
    """
    line_num, example, backend, model = args

    # Create client per thread (thread-safe)
    llm_client = create_client(provider=backend, model=model)
    logger = ImproveLogger()

    logger.info(f"Line {line_num}: Starting evaluation")

    results = {
        "line": line_num,
        "rubrics": {},
        "all_passed": True
    }

    # Run all rubrics
    for rubric_name in JUDGE_RUBRICS:
        try:
            rubric = load_rubric(rubric_name)
            judge = JudgeService(rubric=rubric, llm_client=llm_client, logger=logger)
            result = judge.evaluate(example)

            results["rubrics"][rubric_name] = {
                "score": result.score,
                "passed": result.passed,
                "threshold": rubric["pass_threshold"]
            }

            if not result.passed:
                results["all_passed"] = False
                logger.warning(f"Line {line_num}: {rubric['name']} = {result.score:.2f} ❌")
            else:
                logger.success(f"Line {line_num}: {rubric['name']} = {result.score:.2f} ✓")

        except Exception as e:
            logger.error(f"Line {line_num}: {rubric_name} error - {e}")
            results["rubrics"][rubric_name] = {"error": str(e)}
            results["all_passed"] = False

    logger.info(f"Line {line_num}: Complete")
    return results


def main():
    import argparse

    # Load config for defaults
    config = load_config("config")
    batch_config = config.get("batch", {})
    default_batch_size = batch_config.get("default_size", 10)

    parser = argparse.ArgumentParser(
        description="Parallel batch evaluation (CLOUD ONLY)"
    )
    parser.add_argument("--file", required=True, help="Input JSONL file")
    parser.add_argument("--output", help="Output JSON file for results")
    parser.add_argument("--start-line", type=int, default=1, help="Start line")
    parser.add_argument("--batch-size", type=int, default=default_batch_size,
                       help=f"Number of lines to process in parallel (default: {default_batch_size})")
    parser.add_argument("--backend", default="openrouter", help="LLM backend (openrouter only)")
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--workers", type=int, default=10, help="Max concurrent workers")

    args = parser.parse_args()

    # Validate cloud-only
    if args.backend not in ["openrouter"]:
        print(f"ERROR: Parallel processing only supports cloud providers (openrouter)")
        sys.exit(1)

    logger = ImproveLogger()

    print("=" * 70)
    print("Parallel Batch Evaluation (CLOUD ONLY)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Input: {args.file}")
    print(f"  Lines: {args.start_line} to {args.start_line + args.batch_size - 1}")
    print(f"  Batch size: {args.batch_size} (processed in parallel)")
    print(f"  Workers: {args.workers} concurrent")
    print(f"  Backend: {args.backend}")
    print(f"  Model: {args.model or 'default from config'}")
    print(f"  Rubrics: {', '.join(JUDGE_RUBRICS)}")
    print()

    # Load examples
    logger.info(f"Loading {args.batch_size} examples from {args.file}...")
    examples = []
    end_line = args.start_line + args.batch_size - 1

    with open(args.file, 'r') as f:
        for i, line in enumerate(f, start=1):
            if i < args.start_line:
                continue
            if i > end_line:
                break
            examples.append((i, json.loads(line), args.backend, args.model))

    logger.success(f"Loaded {len(examples)} examples")
    print()

    # Process in parallel
    logger.info(f"Processing {len(examples)} examples with {args.workers} workers...")
    print()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(evaluate_single_example, examples))

    # Summary
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    passed_all = sum(1 for r in results if r["all_passed"])
    failed_any = len(results) - passed_all

    print(f"Total processed: {len(results)}")
    print(f"  All rubrics passed: {passed_all}")
    print(f"  Some rubrics failed: {failed_any}")
    print()

    # Per-rubric stats
    for rubric_name in JUDGE_RUBRICS:
        scores = [r["rubrics"][rubric_name]["score"]
                 for r in results
                 if rubric_name in r["rubrics"] and "score" in r["rubrics"][rubric_name]]
        passed = sum(1 for r in results
                    if rubric_name in r["rubrics"] and r["rubrics"][rubric_name].get("passed"))

        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"{rubric_name:<30} Avg: {avg_score:.2f}  Passed: {passed}/{len(results)}")

    # Save results
    if args.output:
        logger.info(f"Saving results to {args.output}...")
        with open(args.output, 'w') as f:
            json.dump({
                "config": vars(args),
                "results": results,
                "summary": {
                    "total": len(results),
                    "passed_all": passed_all,
                    "failed_any": failed_any
                }
            }, f, indent=2)
        logger.success(f"Results saved")

    print()
    logger.success("Parallel batch complete!")


if __name__ == "__main__":
    main()
