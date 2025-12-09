#!/usr/bin/env python3
"""CLI for running rubric-based judge/improver loop on datasets."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from improvement_engine.rubrics import load_rubric, list_rubrics, get_available_rubrics
from improvement_engine.services.judge_service import JudgeService
from improvement_engine.services.improver_service import ImproverService
from improvement_engine.utils.logger import ImproveLogger
from shared.llm import create_client


def process_example(example: dict, judge: JudgeService, improver: ImproverService,
                    max_retries: int, logger: ImproveLogger) -> dict:
    """Process a single example through judge/improver loop."""
    current_example = example.copy()
    attempt = 0

    while attempt <= max_retries:
        # JUDGE: Evaluate current example
        logger.info(f"  [Attempt {attempt + 1}] Evaluating...")
        judgment = judge.evaluate(current_example)

        logger.info(f"  Score: {judgment.score:.2f} (threshold: {judge.rubric['pass_threshold']:.2f})")

        # PASS? Return current version
        if judgment.passed:
            if attempt == 0:
                logger.info("  ✓ Passed on first evaluation (no changes needed)")
                return current_example  # Return original
            else:
                logger.success(f"  ✓ Fixed after {attempt} attempt(s)")
                return current_example  # Return improved

        # FAIL? Try to improve
        if attempt < max_retries:
            logger.info(f"  ✗ Failed (attempt {attempt + 1}/{max_retries})")
            logger.info(f"  Feedback: {judgment.feedback[:100]}...")

            # IMPROVER: Apply improvements
            logger.info("  Applying improvements...")
            current_example = improver.improve(current_example, judgment.feedback)
            attempt += 1
        else:
            break

    # Could not fix
    logger.warning(f"  ✗ Could not fix after {max_retries} attempts")
    return None  # Mark as failed


def main():
    parser = argparse.ArgumentParser(
        description="Run rubric-based judge/improver loop on dataset"
    )

    parser.add_argument(
        "--rubric",
        type=str,
        help="Rubric name to use (e.g., 'hallucination', 'thinking_quality')"
    )

    parser.add_argument(
        "--list-rubrics",
        action="store_true",
        help="List all available rubrics and exit"
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="lmstudio",
        choices=["lmstudio", "ollama", "openrouter"],
        help="LLM backend to use (default: lmstudio)"
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Model name (optional, backend-specific)"
    )

    parser.add_argument(
        "--input",
        type=str,
        help="Input JSONL file path"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output JSONL file path"
    )

    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="Starting line number (1-indexed)"
    )

    parser.add_argument(
        "--end",
        type=int,
        help="Ending line number (inclusive)"
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max improvement attempts for cloud providers (default: 3). Local providers use unlimited retries."
    )

    args = parser.parse_args()

    # List rubrics?
    if args.list_rubrics:
        list_rubrics()
        return

    # Validate required args
    if not args.rubric:
        print("\nError: --rubric is required")
        print("\nUse --list-rubrics to see available rubrics\n")
        parser.print_help()
        sys.exit(1)

    if not args.input or not args.output:
        print("\nError: --input and --output are required\n")
        parser.print_help()
        sys.exit(1)

    # Setup
    logger = ImproveLogger()

    logger.info(f"\n{'='*70}")
    logger.info(f"Rubric-Based Judge/Improver Loop")
    logger.info(f"{'='*70}")

    # Load rubric
    logger.info(f"\nLoading rubric: {args.rubric}")
    try:
        rubric = load_rubric(args.rubric)
        logger.success(f"✓ Loaded '{rubric['name']}'")
        logger.info(f"  Description: {rubric['description']}")
        logger.info(f"  Scope: {rubric['scope']}")
        logger.info(f"  Pass Threshold: {rubric['pass_threshold']}")
    except Exception as e:
        logger.error(f"Failed to load rubric: {e}")
        sys.exit(1)

    # Create LLM client
    logger.info(f"\nConnecting to {args.backend}...")
    try:
        llm_client = create_client(
            provider=args.backend,
            model=args.model or f"{args.backend}-default"
        )
        llm_client.test_connection()
        logger.success(f"✓ Connected to {args.backend}")
    except Exception as e:
        logger.error(f"Failed to connect to {args.backend}: {e}")
        sys.exit(1)

    # Create judge and improver services
    judge = JudgeService(rubric, llm_client, logger)
    improver = ImproverService(rubric, llm_client, logger)

    # Determine max retries (unlimited for local)
    is_local = args.backend.lower() in ["lmstudio", "ollama"]
    effective_max_retries = 999 if is_local else args.max_retries

    logger.info(f"\nMax retries per example: {'unlimited' if is_local else effective_max_retries}")

    # Load input dataset
    logger.info(f"\nLoading dataset: {args.input}")
    input_path = Path(args.input)

    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    with open(input_path, 'r', encoding='utf-8') as f:
        all_examples = [json.loads(line) for line in f if line.strip()]

    total_examples = len(all_examples)
    logger.info(f"Total examples in dataset: {total_examples}")

    # Determine range
    start_idx = args.start - 1  # Convert to 0-indexed
    end_idx = (args.end or total_examples)  # Inclusive

    examples_to_process = all_examples[start_idx:end_idx]

    # Free memory immediately after slicing
    del all_examples

    logger.info(f"Processing examples {args.start} to {end_idx}")
    logger.info(f"Examples to process: {len(examples_to_process)}")

    # Process examples
    logger.info(f"\n{'='*70}")
    logger.info("Processing Examples")
    logger.info(f"{'='*70}\n")

    passed_count = 0
    improved_count = 0
    failed_count = 0
    total_processed = 0

    # Prepare output file (create parent dirs)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Open output file in write mode (will overwrite if exists)
    with open(output_path, 'w', encoding='utf-8') as out_file:
        for i, example in enumerate(examples_to_process, start=args.start):
            logger.info(f"Example {i}:")

            result = process_example(
                example=example,
                judge=judge,
                improver=improver,
                max_retries=effective_max_retries,
                logger=logger
            )

            if result is not None:
                if result == example:
                    passed_count += 1
                else:
                    improved_count += 1
            else:
                # Keep original on failure
                result = example
                failed_count += 1

            # WRITE IMMEDIATELY after each example
            out_file.write(json.dumps(result, ensure_ascii=False) + '\n')
            out_file.flush()  # Force write to disk
            total_processed += 1

            # Free memory explicitly after each example
            del result
            del example

            logger.info("")  # Blank line between examples

    # Final save confirmation
    logger.info(f"{'='*70}")
    logger.success(f"✓ Saved {total_processed} examples to: {args.output}")

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("Summary")
    logger.info(f"{'='*70}")
    logger.info(f"Total processed: {len(results)}")
    logger.info(f"Passed (no changes): {passed_count}")
    logger.info(f"Improved: {improved_count}")
    logger.info(f"Failed to fix: {failed_count}")
    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    main()
