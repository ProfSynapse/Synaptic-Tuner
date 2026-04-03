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
from typing import Dict

from ..engine import ImprovementEngine
from ..result_writer import generate_output_path


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

    # Apply line range if specified
    if args.start_line or args.end_line:
        start = (args.start_line or 1) - 1  # Convert to 0-indexed
        end = args.end_line or len(examples)
        examples = examples[start:end]
        print(f"Processing lines {start + 1} to {end}")

    print(f"Loaded {len(examples)} examples from {input_path}\n")

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

    for i, example in enumerate(examples, 1):
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

    # Save results
    output_file = args.output or generate_output_path(settings, input_path)
    with open(output_file, "w") as f:
        for example in improved_examples:
            f.write(json.dumps(example) + "\n")

    # Print summary
    print(f"\n=== Summary ===")
    print(f"Total processed: {len(examples)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print(f"Output: {output_file}")
