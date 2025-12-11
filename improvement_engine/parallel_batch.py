#!/usr/bin/env python3
"""
Parallel batch improvement with refactored architecture.

Uses ImprovementEngine with concurrent processing for speed.

CLOUD ONLY: Best with OpenRouter for concurrent requests.
"""

import sys
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.llm import create_client
from shared.llm.config import LLMConfig
from improvement_engine.engine import ImprovementEngine
from improvement_engine.utils.logger import ImproveLogger


def process_single_example(args: Tuple) -> Tuple[int, dict, bool]:
    """
    Process one example (thread-safe).

    Args:
        args: Tuple of (line_num, example, rubric_keys, backend, max_iterations)

    Returns:
        Tuple of (line_num, improved_example, passed)
    """
    line_num, example, rubric_keys, backend, max_iterations, rubrics_dir = args

    # Create engine per thread (thread-safe)
    default_model = {
        "lmstudio": "local-model",
        "ollama": "qwen2.5:latest",
        "openrouter": "anthropic/claude-3.5-sonnet"
    }.get(backend, "local-model")
    config = LLMConfig(provider=backend, model=default_model)
    llm_client = create_client(config=config)
    logger = ImproveLogger()

    engine = ImprovementEngine(
        llm_client=llm_client,
        rubrics_dir=rubrics_dir,
        logger=logger
    )

    logger.info(f"Line {line_num}: Starting improvement")

    try:
        # Run improvement
        result = engine.run(
            example=example,
            rubric_keys=rubric_keys,
            max_iterations=max_iterations
        )

        if result.passed:
            logger.success(f"Line {line_num}: ✅ PASSED after {result.iterations} iteration(s)")
        else:
            logger.warning(f"Line {line_num}: ❌ FAILED after {result.iterations} iteration(s)")

        return line_num, result.improved_example, result.passed

    except Exception as e:
        logger.error(f"Line {line_num}: Error - {e}")
        return line_num, example, False


def process_file_parallel(
    input_file: Path,
    output_file: Path,
    rubric_keys: list[str],
    backend: str = "openrouter",
    max_iterations: int = 3,
    max_workers: int = 4,
    start_line: int = 1,
    end_line: int = None
):
    """
    Process file with parallel workers.

    Args:
        input_file: Input JSONL file
        output_file: Output JSONL file
        rubric_keys: List of rubric keys to use
        backend: LLM backend (openrouter recommended)
        max_iterations: Max improvement iterations per example
        max_workers: Number of parallel workers
        start_line: Starting line number (1-indexed)
        end_line: Ending line number (1-indexed, None = all)
    """
    logger = ImproveLogger()

    # Validate: only cloud providers for parallel processing
    if backend in ["lmstudio", "ollama"]:
        logger.error(f"ERROR: Parallel processing does not support local providers ({backend})")
        logger.error("       Local providers cannot handle concurrent requests.")
        logger.error("       Use --backend openrouter instead.")
        raise ValueError(f"Parallel processing does not support {backend}")

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
    logger.info(f"Max iterations: {max_iterations}")
    logger.info(f"Workers: {max_workers}\n")

    # Prepare tasks
    rubrics_dir = Path(__file__).parent / "rubrics"
    tasks = []

    for i in range(start_idx, end_idx):
        if i >= len(lines):
            break

        line_num = i + 1
        try:
            example = json.loads(lines[i])
            tasks.append((line_num, example, rubric_keys, backend, max_iterations, rubrics_dir))
        except Exception as e:
            logger.error(f"Failed to parse line {line_num}: {e}")

    # Process in parallel
    results = {}
    stats = {"passed": 0, "failed": 0, "errors": 0}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_example, task): task[0] for task in tasks}

        for future in as_completed(futures):
            try:
                line_num, improved_example, passed = future.result()
                results[line_num] = improved_example

                if passed:
                    stats["passed"] += 1
                else:
                    stats["failed"] += 1

            except Exception as e:
                line_num = futures[future]
                logger.error(f"Line {line_num}: Unexpected error - {e}")
                stats["errors"] += 1
                # Use original on error
                results[line_num] = json.loads(lines[line_num - 1])

    # Write output in order
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(start_line, end_line + 1):
            if i in results:
                f.write(json.dumps(results[i]) + '\n')

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PARALLEL BATCH IMPROVEMENT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total processed: {len(results)}")
    logger.info(f"Passed: {stats['passed']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info(f"Output: {output_file}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Parallel batch improvement with refactored architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with 4 workers
  python parallel_batch.py --file dataset.jsonl --output improved.jsonl --workers 4

  # Process specific range
  python parallel_batch.py --file dataset.jsonl --output improved.jsonl \\
    --start-line 1 --end-line 100 --workers 8

  # Use specific rubrics
  python parallel_batch.py --file dataset.jsonl --output improved.jsonl \\
    --rubrics thinking_quality,context_alignment --workers 4
        """
    )

    parser.add_argument("--file", type=Path, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file")
    parser.add_argument(
        "--rubrics",
        default="confidence_calibration,context_alignment,destructive_safety,factuality,requirements_plan,tool_alignment",
        help="Comma-separated rubric keys"
    )
    parser.add_argument("--backend", default="openrouter", choices=["openrouter", "lmstudio", "ollama"])
    parser.add_argument("--max-iterations", type=int, default=3, help="Max iterations per example")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--start-line", type=int, default=1, help="Start line (1-indexed)")
    parser.add_argument("--end-line", type=int, help="End line (1-indexed)")

    args = parser.parse_args()

    # Parse rubric keys
    rubric_keys = [k.strip() for k in args.rubrics.split(",")]

    # Run parallel processing
    process_file_parallel(
        input_file=args.file,
        output_file=args.output,
        rubric_keys=rubric_keys,
        backend=args.backend,
        max_iterations=args.max_iterations,
        max_workers=args.workers,
        start_line=args.start_line,
        end_line=args.end_line
    )


if __name__ == "__main__":
    main()
