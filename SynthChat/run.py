"""SynthChat CLI - Unified entry point for dataset generation and improvement.

Location: SynthChat/run.py
Purpose: Single CLI for both "generate" and "improve" modes
Usage: python -m SynthChat.run [generate|improve|validate] [options]

Commands:
    generate - Create new dataset from scenarios
    improve  - Improve existing dataset with rubrics
    validate - Check if dataset passes rubrics (no improvement)
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from shared.llm import create_client
from .utils.yaml_loader import load_yaml
from .engine import ImprovementEngine
from .generator import SynthChatGenerator, ScenarioLoader


def load_settings(config_dir: Path) -> Dict:
    """Load settings.yaml configuration."""
    settings_path = config_dir / "settings.yaml"
    if not settings_path.exists():
        raise FileNotFoundError(f"Settings not found: {settings_path}")
    return load_yaml(settings_path)


def create_llm_client(config: Dict, mode: str = "generation"):
    """
    Create LLM client for generation or improvement.

    Args:
        config: Settings configuration
        mode: "generation" or "improvement"

    Returns:
        LLM client from shared.llm
    """
    llm_config = config["llm"].get(mode, {})
    provider = llm_config.get("provider", "lmstudio")
    model = llm_config.get("model", "local-model")

    # Create client using shared.llm factory
    client = create_client(
        provider=provider,
        model=model,
        host=llm_config.get("host"),
        port=llm_config.get("port"),
        api_key=None  # Will read from env
    )
    return client


def generate_mode(args):
    """
    Generate new dataset from scenarios.

    Flow:
        1. Load settings and scenarios
        2. Create LLM clients (generation + improvement)
        3. Create generator with improvement engine
        4. Generate examples
        5. Save results
    """
    print("=== SynthChat: Generate Mode ===\n")

    # Load configuration
    config_dir = Path(args.config_dir or "SynthChat/config")
    settings = load_settings(config_dir)

    scenarios_dir = Path(args.scenarios_dir or "SynthChat/scenarios")
    rubrics_dir = Path(args.rubrics_dir or "SynthChat/rubrics")

    # Create LLM clients
    print("Initializing LLM clients...")
    gen_client = create_llm_client(settings, mode="generation")
    improve_client = create_llm_client(settings, mode="improvement")

    # Create improvement engine
    validation_config = config_dir / "validation.yaml"
    engine = ImprovementEngine(
        llm_client=improve_client,
        rubrics_dir=rubrics_dir,
        config_path=validation_config,
        enable_interactions=settings["logging"]["save_interactions"]
    )

    # Create generator
    generator = SynthChatGenerator(
        config_dir=config_dir,
        scenarios_dir=scenarios_dir,
        rubrics_dir=rubrics_dir,
        llm_client=gen_client,
        engine=engine,
        enable_stage_validation=settings["generation"]["stage_validation"]
    )

    # Load targets
    if args.targets_file:
        with open(args.targets_file) as f:
            targets = json.load(f)
    else:
        # Use default targets from settings
        targets = settings["defaults"]["targets"]

    # Filter targets if specific scenarios requested
    if args.scenarios:
        targets = {k: v for k, v in targets.items() if k in args.scenarios}

    print(f"\nGeneration targets:")
    for scenario_key, count in targets.items():
        print(f"  {scenario_key}: {count}")
    print(f"Total: {sum(targets.values())} examples\n")

    # Generate
    max_iterations = args.max_iterations or settings["improvement"]["max_iterations"]
    results = generator.generate_batch(
        targets=targets,
        max_iterations=max_iterations,
        randomize_params=True
    )

    # Save results
    output_file = args.output or _generate_output_path(settings)
    _save_results(results, output_file, settings)

    # Print summary
    _print_summary(results, output_file)


def improve_mode(args):
    """
    Improve existing dataset with rubrics.

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

    # Create LLM client
    print("Initializing improvement LLM client...")
    improve_client = create_llm_client(settings, mode="improvement")

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
    output_file = args.output or _generate_output_path(settings, input_path)
    with open(output_file, "w") as f:
        for example in improved_examples:
            f.write(json.dumps(example) + "\n")

    # Print summary
    print(f"\n=== Summary ===")
    print(f"Total processed: {len(examples)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print(f"Output: {output_file}")


def validate_mode(args):
    """
    Validate dataset against rubrics without improving.

    Flow:
        1. Load dataset and settings
        2. Create engine (read-only mode)
        3. Run validation on each example
        4. Report results
    """
    print("=== SynthChat: Validate Mode ===\n")

    # TODO: Implement validation-only mode
    # Similar to improve but don't apply improvements
    print("Validate mode not yet implemented")
    sys.exit(1)


def _generate_output_path(settings: Dict, input_path: Optional[Path] = None) -> Path:
    """
    Generate output file path with datetime versioning.

    Args:
        settings: Settings configuration
        input_path: Input file path (for improve mode)

    Returns:
        Output file path
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if input_path:
        # Improve mode: append timestamp to input name
        stem = input_path.stem
        # Remove existing version if present (e.g., _v1.8)
        if "_v" in stem:
            stem = stem.split("_v")[0]
        return input_path.parent / f"{stem}_{timestamp}.jsonl"
    else:
        # Generate mode: use default directory
        default_dir = Path(settings["output"]["default_dir"])
        default_dir.mkdir(parents=True, exist_ok=True)
        return default_dir / f"synthchat_{timestamp}.jsonl"


def _save_results(results: List, output_file: Path, settings: Dict):
    """Save generation results to JSONL with metadata."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        # Write metadata header if enabled
        if settings["output"]["include_metadata"]:
            metadata = {
                "_meta": {
                    "synthchat_version": "1.0.0",
                    "generated_at": datetime.utcnow().isoformat(),
                    "stats": {
                        "total": len(results),
                        "passed": sum(1 for r in results if r.success),
                        "failed": sum(1 for r in results if not r.success),
                        "avg_iterations": sum(r.iterations for r in results) / len(results) if results else 0
                    }
                }
            }
            f.write(json.dumps(metadata) + "\n")

        # Write examples
        for result in results:
            f.write(json.dumps(result.example) + "\n")


def _print_summary(results: List, output_file: Path):
    """Print generation summary."""
    total = len(results)
    passed = sum(1 for r in results if r.success)
    failed = total - passed
    avg_iterations = sum(r.iterations for r in results) / total if total else 0

    print(f"\n=== Summary ===")
    print(f"Total generated: {total}")
    print(f"Passed: {passed} ({passed / total * 100:.1f}%)")
    print(f"Failed: {failed} ({failed / total * 100:.1f}%)")
    print(f"Avg iterations: {avg_iterations:.1f}")
    print(f"Output: {output_file}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SynthChat - Synthetic dataset generation and improvement"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate new dataset")
    generate_parser.add_argument("--config-dir", help="Config directory path")
    generate_parser.add_argument("--scenarios-dir", help="Scenarios directory path")
    generate_parser.add_argument("--rubrics-dir", help="Rubrics directory path")
    generate_parser.add_argument("--output", "-o", help="Output file path")
    generate_parser.add_argument("--targets-file", help="JSON file with generation targets")
    generate_parser.add_argument("--scenarios", nargs="+", help="Specific scenarios to generate")
    generate_parser.add_argument("--max-iterations", type=int, help="Max improvement iterations")

    # Improve command
    improve_parser = subparsers.add_parser("improve", help="Improve existing dataset")
    improve_parser.add_argument("--input", "-i", required=True, help="Input JSONL file")
    improve_parser.add_argument("--output", "-o", help="Output file path")
    improve_parser.add_argument("--config-dir", help="Config directory path")
    improve_parser.add_argument("--rubrics-dir", help="Rubrics directory path")
    improve_parser.add_argument("--rubrics", help="Comma-separated rubric names")
    improve_parser.add_argument("--start-line", type=int, help="Start line (1-indexed)")
    improve_parser.add_argument("--end-line", type=int, help="End line (inclusive)")
    improve_parser.add_argument("--max-iterations", type=int, help="Max improvement iterations")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate dataset (no improvement)")
    validate_parser.add_argument("--input", "-i", required=True, help="Input JSONL file")
    validate_parser.add_argument("--config-dir", help="Config directory path")
    validate_parser.add_argument("--rubrics-dir", help="Rubrics directory path")
    validate_parser.add_argument("--rubrics", help="Comma-separated rubric names")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Route to appropriate mode
    if args.command == "generate":
        generate_mode(args)
    elif args.command == "improve":
        improve_mode(args)
    elif args.command == "validate":
        validate_mode(args)


if __name__ == "__main__":
    main()
