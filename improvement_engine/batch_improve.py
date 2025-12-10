#!/usr/bin/env python3
"""
Batch improvement with judge+improve loop.

CLOUD ONLY: OpenRouter
NOT for local providers (lmstudio, ollama) - too slow for batch processing.
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.llm import create_client
from improvement_engine.rubrics import load_rubric
from improvement_engine.services.judge_service import JudgeService
from improvement_engine.utils.logger import ImproveLogger

# Rubrics to use
JUDGE_RUBRICS = [
    "confidence_calibration",
    "context_alignment",
    "destructive_safety",
    "factuality",
    "requirements_plan",
    "tool_alignment"
]

IMPROVE_RUBRIC = "thinking_quality"  # Only rubric with improver


def improve_example(example: dict, llm_client, logger, max_iterations: int = 3):
    """
    Run full judge+improve loop on one example.

    Returns:
        Tuple of (improved_example, iteration_count, all_passed)
    """
    current = example

    for iteration in range(1, max_iterations + 1):
        logger.info(f"  Iteration {iteration}/{max_iterations}")

        # Judge with all rubrics
        all_scores = {}
        all_passed = True

        for rubric_name in JUDGE_RUBRICS:
            rubric = load_rubric(rubric_name)
            judge = JudgeService(rubric=rubric, llm_client=llm_client, logger=logger)
            result = judge.evaluate(current)

            all_scores[rubric_name] = result.score

            # DEBUG: Log full result details
            logger.debug(f"    {rubric['name']} details: {result.details}")

            if not result.passed:
                all_passed = False
                logger.warning(f"    {rubric['name']}: {result.score:.2f} (threshold: {rubric['pass_threshold']})")
                if result.feedback:
                    logger.debug(f"      Feedback: {result.feedback[:200]}")
            else:
                logger.success(f"    {rubric['name']}: {result.score:.2f} ✓")

        # If all passed, we're done
        if all_passed:
            logger.success(f"  All rubrics passed on iteration {iteration}!")
            return current, iteration, True

        # Otherwise, try to improve
        if iteration < max_iterations:
            logger.info(f"  Improving with {IMPROVE_RUBRIC}...")
            improve_rubric = load_rubric(IMPROVE_RUBRIC)

            # Judge with improve rubric to get feedback
            judge = JudgeService(rubric=improve_rubric, llm_client=llm_client, logger=logger)
            result = judge.evaluate(current)

            if not result.passed and improve_rubric.get("improver_prompt"):
                # Improve based on feedback
                improver_prompt = improve_rubric["improver_prompt"].format(
                    thinking_block=json.dumps(extract_thinking(current)),
                    feedback=result.feedback
                )

                # Get improved thinking block
                improved_thinking = llm_client.structured_output(
                    messages=[{"role": "user", "content": improver_prompt}],
                    schema=improve_rubric.get("output_schema", {})
                )

                # Replace thinking block in example
                current = replace_thinking(current, improved_thinking)
                logger.info(f"  Applied improvements")
            else:
                logger.warning(f"  No improvements available")

    logger.warning(f"  Max iterations reached without full pass")
    return current, max_iterations, False


def extract_thinking(example: dict) -> dict:
    """Extract thinking block from example."""
    for conv in example.get("conversations", []):
        if conv.get("role") == "assistant":
            content = conv.get("content", "")
            if "<thinking>" in content:
                start = content.find("<thinking>") + 10
                end = content.find("</thinking>")
                thinking_json = content[start:end].strip()
                return json.loads(thinking_json)
    return {}


def replace_thinking(example: dict, new_thinking: dict) -> dict:
    """Replace thinking block in example."""
    improved = json.loads(json.dumps(example))  # Deep copy

    for conv in improved.get("conversations", []):
        if conv.get("role") == "assistant":
            content = conv.get("content", "")
            if "<thinking>" in content:
                start = content.find("<thinking>") + 10
                end = content.find("</thinking>")
                new_content = (
                    content[:start] +
                    "\n" + json.dumps(new_thinking, indent=2) + "\n" +
                    content[end:]
                )
                conv["content"] = new_content
                break

    return improved


def main():
    # Load config for defaults
    from improvement_engine.utils.yaml_loader import load_config
    config = load_config("config")
    batch_config = config.get("batch", {})
    default_batch_size = batch_config.get("default_size", 10)
    default_max_iterations = batch_config.get("max_iterations", 3)

    parser = argparse.ArgumentParser(
        description="Batch improvement with judge+improve loop (CLOUD ONLY)"
    )
    parser.add_argument("--file", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--start-line", type=int, default=1, help="Start line (1-indexed)")
    parser.add_argument("--end-line", type=int, help="End line (1-indexed, defaults to start + batch size)")
    parser.add_argument("--batch-size", type=int, default=default_batch_size,
                        help=f"Number of lines to process (default: {default_batch_size} from config)")
    parser.add_argument("--backend", default="openrouter", help="LLM backend (openrouter only)")
    parser.add_argument("--model", help="Model name (default from config)")
    parser.add_argument("--max-iterations", type=int, default=default_max_iterations,
                        help=f"Max improve iterations per example (default: {default_max_iterations} from config)")

    args = parser.parse_args()

    # Calculate end_line from batch_size if not specified
    if not args.end_line:
        args.end_line = args.start_line + args.batch_size - 1

    # Validate cloud-only
    if args.backend not in ["openrouter"]:
        print(f"ERROR: Batch processing only supports cloud providers (openrouter)")
        print(f"       Local providers (lmstudio, ollama) are too slow for batching")
        sys.exit(1)

    logger = ImproveLogger()

    print("=" * 70)
    print("Batch Improvement - Judge+Improve Loop (CLOUD ONLY)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Input: {args.file}")
    print(f"  Output: {args.output}")
    print(f"  Lines: {args.start_line}-{args.end_line or 'end'}")
    print(f"  Backend: {args.backend}")
    print(f"  Model: {args.model or 'default from config'}")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Judge rubrics: {', '.join(JUDGE_RUBRICS)}")
    print(f"  Improve rubric: {IMPROVE_RUBRIC}")
    print()

    # Create LLM client
    logger.info("Initializing LLM client...")
    llm_client = create_client(provider=args.backend, model=args.model)

    # Load examples
    logger.info(f"Loading examples from {args.file}...")
    examples = []
    with open(args.file, 'r') as f:
        for i, line in enumerate(f, start=1):
            if i < args.start_line:
                continue
            if args.end_line and i > args.end_line:
                break
            examples.append((i, json.loads(line)))

    logger.success(f"Loaded {len(examples)} examples")
    print()

    # Process each example
    results = []
    stats = {"improved": 0, "passed_first": 0, "failed": 0}

    for line_num, example in examples:
        print("-" * 70)
        print(f"Processing Line {line_num}")
        print("-" * 70)

        improved, iterations, passed = improve_example(
            example, llm_client, logger, args.max_iterations
        )

        results.append(improved)

        if iterations == 1 and passed:
            stats["passed_first"] += 1
        elif passed:
            stats["improved"] += 1
        else:
            stats["failed"] += 1

        print()

    # Write results
    logger.info(f"Writing {len(results)} examples to {args.output}...")
    with open(args.output, 'w') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total processed: {len(examples)}")
    print(f"  Passed first try: {stats['passed_first']}")
    print(f"  Improved to pass: {stats['improved']}")
    print(f"  Failed to pass: {stats['failed']}")
    print()
    logger.success("Batch improvement complete!")


if __name__ == "__main__":
    main()
