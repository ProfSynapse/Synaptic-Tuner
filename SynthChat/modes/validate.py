"""SynthChat Validate Mode - Validate datasets against rubrics.

Location: SynthChat/modes/validate.py
Purpose: Implements the 'validate' CLI command. Loads an existing JSONL dataset,
         runs validation (judge only, no improvement loop) against rubrics,
         and reports pass/fail results per example and per rubric.
Usage: Called by SynthChat.run.main() when command is 'validate'.
"""

import json
import sys
from pathlib import Path
from typing import Dict

from ..engine import ImprovementEngine
from ..services.privacy_preprocess import (
    resolve_privacy_preprocessor,
    sanitize_payload_with_metadata,
)


def validate_mode(args, *, load_settings, create_llm_client):
    """Validate dataset against rubrics without improving.

    Args:
        args: Parsed CLI arguments.
        load_settings: Callable to load settings.yaml.
        create_llm_client: Callable to create LLM clients.

    Flow:
        1. Load dataset and settings
        2. Create engine (read-only mode)
        3. Run validation on each example
        4. Report results
    """
    print("=== SynthChat: Validate Mode ===\n")

    # Load configuration
    config_dir = Path(args.config_dir or "SynthChat/config")
    settings = load_settings(config_dir)
    rubrics_dir = Path(args.rubrics_dir or "SynthChat/rubrics")
    privacy_preprocessor = resolve_privacy_preprocessor(
        config_dir=config_dir,
        settings=settings,
        apply_target="input_jsonl",
        profile_override=getattr(args, "privacy_profile", None),
    )

    # Load input dataset
    if not args.input:
        print("Error: --input required for validate mode")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Load examples
    examples = []
    with open(input_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    examples.append((line_num, json.loads(line)))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed JSON at line {line_num}: {e}")

    print(f"Loaded {len(examples)} examples from {input_path}\n")
    if privacy_preprocessor is not None:
        changed_count = 0
        sanitized_examples = []
        for line_num, example in examples:
            sanitized_example, summary = sanitize_payload_with_metadata(
                example,
                preprocessor=privacy_preprocessor,
                scope_key=f"{input_path}:{line_num}",
                metadata_field="privacy_preprocess_input",
            )
            if summary.get("changed"):
                changed_count += 1
            sanitized_examples.append((line_num, sanitized_example))
        examples = sanitized_examples
        print(
            f"Privacy input preprocessing enabled (profile={privacy_preprocessor.profile_name}, "
            f"changed={changed_count}/{len(examples)})\n"
        )

    # Create LLM client for judging (CLI args override settings.yaml)
    print("Initializing validation LLM client...")
    validate_client = create_llm_client(settings, mode="improvement",
                                        provider_override=args.provider, model_override=args.model)

    # Create improvement engine
    validation_config = config_dir / "validation.yaml"
    engine = ImprovementEngine(
        llm_client=validate_client,
        rubrics_dir=rubrics_dir,
        config_path=validation_config,
        enable_interactions=False  # No logging for validation-only
    )

    # Determine rubrics to use
    rubrics = args.rubrics or settings["improvement"]["default_rubrics"]
    if isinstance(rubrics, str):
        rubrics = [r.strip() for r in rubrics.split(",")]

    print(f"Validating against rubrics: {', '.join(rubrics)}\n")

    # Validate examples (max_iterations=1 means just judge, no improvement)
    total_passed = 0
    total_failed = 0
    failures_by_rubric = {rubric: [] for rubric in rubrics}
    failing_lines = []

    for line_num, example in examples:
        try:
            # Run validation with max_iterations=1 (judge only, no improvement loop)
            result = engine.run(
                example=example,
                rubric_keys=rubrics,
                max_iterations=1  # Just judge once, don't improve
            )

            if result.passed:
                total_passed += 1
            else:
                total_failed += 1
                failing_lines.append(line_num)

                # Track which rubrics failed
                for rubric in rubrics:
                    score = result.final_scores.get(rubric, 0.0)
                    # Load rubric to get threshold
                    rubric_config = engine.rubric_repo.get_rubric(rubric)
                    threshold = rubric_config.get("pass_threshold", 0.8)
                    if score < threshold:
                        failures_by_rubric[rubric].append(line_num)

        except Exception as e:
            print(f"  Error validating line {line_num}: {e}")
            total_failed += 1
            failing_lines.append(line_num)

    # Print summary
    total = len(examples)
    pass_rate = (total_passed / total * 100) if total > 0 else 0

    print(f"\n=== Validation Summary ===")
    print(f"Total examples: {total}")
    print(f"Passed: {total_passed} ({pass_rate:.1f}%)")
    print(f"Failed: {total_failed} ({100 - pass_rate:.1f}%)")
    if privacy_preprocessor is not None:
        print(f"Privacy profile: {privacy_preprocessor.profile_name}")

    if failing_lines:
        print(f"\nFailing lines: {', '.join(map(str, failing_lines[:20]))}")
        if len(failing_lines) > 20:
            print(f"  ... and {len(failing_lines) - 20} more")

    print(f"\n=== Failures by Rubric ===")
    for rubric, failed_lines in failures_by_rubric.items():
        if failed_lines:
            print(f"{rubric}: {len(failed_lines)} failures")
            print(f"  Lines: {', '.join(map(str, failed_lines[:10]))}")
            if len(failed_lines) > 10:
                print(f"  ... and {len(failed_lines) - 10} more")

    if total_failed > 0:
        print(f"\nRun with 'improve' mode to fix failing examples:")
        print(f"  python -m SynthChat.run improve --input {input_path} --rubrics {','.join(rubrics)}")
