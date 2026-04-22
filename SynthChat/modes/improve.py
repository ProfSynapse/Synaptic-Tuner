"""SynthChat Improve Mode - Improve existing datasets with rubrics.

Location: SynthChat/modes/improve.py
Purpose: Implements the 'improve' CLI command. Loads an existing JSONL dataset,
         applies improvement rubrics via the ImprovementEngine, and writes
         the improved dataset to a new output file.
Usage: Called by SynthChat.run.main() when command is 'improve'.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ..engine import ImprovementEngine
from ..parallel.improve_workers import run_parallel_improvement
from ..result_writer import generate_output_path


def _parse_line_selectors(spec: str) -> List[int]:
    """Parse a comma-separated line selector spec into sorted unique line numbers."""
    selected = set()
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str.strip())
            end = int(end_str.strip())
            if start <= 0 or end <= 0:
                raise ValueError(f"Line selectors must be 1-indexed positive integers: {part}")
            if end < start:
                raise ValueError(f"Invalid descending line range: {part}")
            selected.update(range(start, end + 1))
        else:
            value = int(part)
            if value <= 0:
                raise ValueError(f"Line selectors must be 1-indexed positive integers: {part}")
            selected.add(value)
    return sorted(selected)


def _load_explicit_line_selectors(spec: str | None, line_file: str | None) -> List[int]:
    """Load explicit line selectors from CLI string and/or file."""
    selected = set()
    if spec:
        selected.update(_parse_line_selectors(spec))
    if line_file:
        path = Path(line_file)
        if not path.exists():
            raise FileNotFoundError(f"Line selector file not found: {path}")
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            selected.update(_parse_line_selectors(line))
    return sorted(selected)


def _select_examples(
    examples: Sequence[Dict],
    *,
    start_line: int | None,
    end_line: int | None,
    explicit_lines: Sequence[int],
) -> List[Tuple[int, Dict]]:
    """Return selected examples paired with original 1-indexed line numbers."""
    indexed_examples = list(enumerate(examples, start=1))
    if explicit_lines:
        explicit_set = set(explicit_lines)
        missing = [line for line in explicit_lines if line > len(examples)]
        if missing:
            raise ValueError(
                f"Requested line selectors exceed file length ({len(examples)}): {missing}"
            )
        return [(line_number, example) for line_number, example in indexed_examples if line_number in explicit_set]

    if start_line is not None or end_line is not None:
        start = start_line or 1
        end = end_line or len(examples)
        if start <= 0 or end <= 0:
            raise ValueError("Line selection is 1-indexed; start/end must be positive")
        if end < start:
            raise ValueError("--end-line must be greater than or equal to --start-line")
        return [(line_number, example) for line_number, example in indexed_examples if start <= line_number <= end]

    return indexed_examples


def improve_mode(args, *, load_settings, create_llm_client):
    """Improve existing dataset with rubrics.

    Args:
        args: Parsed CLI arguments.
        load_settings: Callable to load settings.yaml.
        create_llm_client: Callable to create LLM clients.

    Flow:
        1. Load settings and dataset
        2. Create improvement LLM client
        3. Create improvement engine
        4. Process each example
        5. Save improved dataset
    """
    print("=== SynthChat: Improve Mode ===\n")

    # Load configuration
    config_dir = Path(args.config_dir or "SynthChat/config")
    settings = load_settings(config_dir)
    rubrics_dir = Path(args.rubrics_dir or "SynthChat/rubrics")

    # Load input dataset
    if not args.input:
        print("Error: --input required for improve mode")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Load examples
    examples = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    explicit_lines = _load_explicit_line_selectors(args.lines, args.line_file)
    selected_examples = _select_examples(
        examples,
        start_line=args.start_line,
        end_line=args.end_line,
        explicit_lines=explicit_lines,
    )

    if explicit_lines:
        preview = ",".join(str(value) for value in explicit_lines[:10])
        if len(explicit_lines) > 10:
            preview += ",..."
        print(f"Processing explicit lines: {preview}")
    elif args.start_line is not None or args.end_line is not None:
        start = args.start_line or 1
        end = args.end_line or len(examples)
        print(f"Processing lines {start} to {end}")

    examples = [example for _, example in selected_examples]
    selected_line_numbers = [line_number for line_number, _ in selected_examples]

    print(f"Loaded {len(examples)} examples from {input_path}\n")

    # Determine rubrics to use
    rubrics = args.rubrics or settings["improvement"]["default_rubrics"]
    if isinstance(rubrics, str):
        rubrics = [r.strip() for r in rubrics.split(",")]

    print(f"Applying rubrics: {', '.join(rubrics)}\n")

    # Process examples
    max_iterations = args.max_iterations or settings["improvement"]["max_iterations"]
    improved_examples = []
    passed_count = 0
    failed_count = 0
    result_rows = []
    num_workers = max(1, args.workers or 1)

    if num_workers > 1:
        print(f"Using {num_workers} parallel workers\n")
        work_items = [
            (
                idx,
                example,
                rubrics,
                max_iterations,
                config_dir,
                rubrics_dir,
                settings,
                args.provider,
                args.model,
                create_llm_client,
            )
            for idx, example in enumerate(examples)
        ]
        parallel_results = run_parallel_improvement(work_items, num_workers)

        for idx, example in enumerate(examples):
            original_line_number = selected_line_numbers[idx]
            result = parallel_results[idx] if idx < len(parallel_results) else None
            if result is None:
                improved_examples.append(example)
                failed_count += 1
                result_rows.append({
                    "line_number": original_line_number,
                    "passed": False,
                    "iterations": None,
                    "final_scores": {},
                    "scopes_improved": [],
                    "error": "parallel_result_missing",
                })
                continue
            improved_examples.append(result.improved_example)
            if result.passed:
                passed_count += 1
            else:
                failed_count += 1
            result_rows.append({
                "line_number": original_line_number,
                "passed": bool(result.passed),
                "iterations": int(result.iterations),
                "final_scores": result.final_scores,
                "scopes_improved": result.scopes_improved,
            })
    else:
        # Create LLM client (CLI args override settings.yaml)
        print("Initializing improvement LLM client...")
        improve_client = create_llm_client(settings, mode="improvement",
                                           provider_override=args.provider, model_override=args.model)

        # Create improvement engine
        validation_config = config_dir / "validation.yaml"
        engine = ImprovementEngine(
            llm_client=improve_client,
            rubrics_dir=rubrics_dir,
            config_path=validation_config,
            enable_interactions=settings["logging"]["save_interactions"]
        )

        for i, example in enumerate(examples, 1):
            original_line_number = selected_line_numbers[i - 1]
            print(f"Processing {i}/{len(examples)}...")

            try:
                result = engine.run(
                    example=example,
                    rubric_keys=rubrics,
                    max_iterations=max_iterations
                )

                improved_examples.append(result.improved_example)
                if result.passed:
                    passed_count += 1
                else:
                    failed_count += 1

            except Exception as e:
                print(f"  Error: {e}")
                improved_examples.append(example)  # Keep original
                failed_count += 1
                result_rows.append({
                    "line_number": original_line_number,
                    "passed": False,
                    "iterations": None,
                    "final_scores": {},
                    "scopes_improved": [],
                    "error": str(e),
                })
                continue

            result_rows.append({
                "line_number": original_line_number,
                "passed": bool(result.passed),
                "iterations": int(result.iterations),
                "final_scores": result.final_scores,
                "scopes_improved": result.scopes_improved,
            })

    # Save results
    output_file = args.output or generate_output_path(settings, input_path)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for example in improved_examples:
            f.write(json.dumps(example) + "\n")

    report_path = Path(f"{output_file}.improve_report.json")
    report_payload = {
        "input": str(input_path),
        "output": str(output_file),
        "rubrics": rubrics,
        "max_iterations": max_iterations,
        "total_processed": len(examples),
        "passed": passed_count,
        "failed": failed_count,
        "results": result_rows,
    }
    with open(report_path, "w") as f:
        json.dump(report_payload, f, indent=2)

    # Print summary
    print(f"\n=== Summary ===")
    print(f"Total processed: {len(examples)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print(f"Output: {output_file}")
    print(f"Report: {report_path}")
