#!/usr/bin/env python3
"""
Batch improvement with refactored architecture.

Uses ImprovementEngine for clean, config-driven improvement.

CLOUD ONLY: OpenRouter recommended
Can also work with local providers but slower.
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.llm import create_client
from shared.llm.config import LLMConfig
from improvement_engine.engine import ImprovementEngine
from improvement_engine.utils.logger import ImproveLogger


def process_file(
    input_file: Path,
    output_file: Path,
    rubric_keys: list[str],
    backend: str = "openrouter",
    max_iterations: int = 3,
    start_line: int = 1,
    end_line: int = None
):
    """
    Process a JSONL file with batch improvement.

    Args:
        input_file: Input JSONL file
        output_file: Output JSONL file
        rubric_keys: List of rubric keys to use
        backend: LLM backend (openrouter, lmstudio, ollama)
        max_iterations: Max improvement iterations per example
        start_line: Starting line number (1-indexed)
        end_line: Ending line number (1-indexed, None = all)
    """
    logger = ImproveLogger()

    # Initialize LLM client
    config = LLMConfig(backend=backend)
    llm_client = create_client(config=config)

    # Initialize engine
    rubrics_dir = Path(__file__).parent / "rubrics"
    engine = ImprovementEngine(
        llm_client=llm_client,
        rubrics_dir=rubrics_dir,
        logger=logger
    )

    # Read input
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Determine range
    if end_line is None:
        end_line = len(lines)

    start_idx = start_line - 1
    end_idx = end_line

    logger.info(f"Processing lines {start_line} to {end_line}")
    logger.info(f"Rubrics: {', '.join(rubric_keys)}")
    logger.info(f"Max iterations: {max_iterations}\n")

    # Process each example
    results = []
    stats = {"passed": 0, "failed": 0, "errors": 0}

    for i in range(start_idx, end_idx):
        if i >= len(lines):
            break

        line_num = i + 1
        logger.info(f"=== Line {line_num}/{end_idx} ===")

        try:
            example = json.loads(lines[i])

            # Run improvement
            result = engine.run(
                example=example,
                rubric_keys=rubric_keys,
                max_iterations=max_iterations
            )

            # Log result
            if result.passed:
                logger.success(f"✅ PASSED after {result.iterations} iteration(s)")
                stats["passed"] += 1
            else:
                logger.warning(f"❌ FAILED after {result.iterations} iteration(s)")
                stats["failed"] += 1

            logger.info(f"Scopes improved: {', '.join(result.scopes_improved) if result.scopes_improved else 'none'}")

            # Save improved example
            results.append(json.dumps(result.improved_example))

        except Exception as e:
            logger.error(f"Error processing line {line_num}: {e}")
            stats["errors"] += 1
            # Save original on error
            results.append(lines[i].strip())

    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for result_line in results:
            f.write(result_line + '\n')

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("BATCH IMPROVEMENT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total processed: {len(results)}")
    logger.info(f"Passed: {stats['passed']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info(f"Output: {output_file}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Batch improvement with refactored architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process entire file with default rubrics
  python batch_improve.py --file dataset.jsonl --output improved.jsonl

  # Process specific lines
  python batch_improve.py --file dataset.jsonl --output improved.jsonl \\
    --start-line 1 --end-line 100

  # Use specific rubrics
  python batch_improve.py --file dataset.jsonl --output improved.jsonl \\
    --rubrics thinking_quality,context_alignment,factuality

  # Use local LLM
  python batch_improve.py --file dataset.jsonl --output improved.jsonl \\
    --backend lmstudio
        """
    )

    parser.add_argument("--file", type=Path, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file")
    parser.add_argument(
        "--rubrics",
        default="confidence_calibration,context_alignment,destructive_safety,factuality,requirements_plan,tool_alignment",
        help="Comma-separated rubric keys (default: all main rubrics)"
    )
    parser.add_argument("--backend", default="openrouter", choices=["openrouter", "lmstudio", "ollama"])
    parser.add_argument("--max-iterations", type=int, default=3, help="Max iterations per example")
    parser.add_argument("--start-line", type=int, default=1, help="Start line (1-indexed)")
    parser.add_argument("--end-line", type=int, help="End line (1-indexed)")

    args = parser.parse_args()

    # Parse rubric keys
    rubric_keys = [k.strip() for k in args.rubrics.split(",")]

    # Run batch processing
    process_file(
        input_file=args.file,
        output_file=args.output,
        rubric_keys=rubric_keys,
        backend=args.backend,
        max_iterations=args.max_iterations,
        start_line=args.start_line,
        end_line=args.end_line
    )


if __name__ == "__main__":
    main()
