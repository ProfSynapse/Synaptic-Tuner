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
from datetime import datetime, timezone
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from shared.llm import create_client
from .utils.yaml_loader import load_yaml
from .utils.docs_loader import DocsLoader, DocFile
from .engine import ImprovementEngine
from .generator import SynthChatGenerator, ScenarioLoader


class StreamingResultWriter:
    """Writes generation results to JSONL incrementally as they complete.

    Location: SynthChat/run.py
    Purpose: Prevents data loss during long generation runs by streaming each
    result to disk immediately instead of accumulating in memory.
    Thread-safe for use with parallel workers via threading.Lock.

    Usage:
        with StreamingResultWriter(output_file, settings) as writer:
            writer.write(result)  # Called after each example completes
    """

    def __init__(self, output_file: Path, settings: Dict):
        self._output_file = output_file
        self._settings = settings
        self._lock = threading.Lock()
        self._file = None
        self._count = 0

    def __enter__(self):
        self._output_file.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._output_file, "w")

        # Write metadata header as placeholder (will be updated at close)
        if self._settings["output"]["include_metadata"]:
            metadata = {
                "_meta": {
                    "synthchat_version": "1.0.0",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "streaming": True
                }
            }
            self._file.write(json.dumps(metadata) + "\n")
            self._file.flush()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()
        return False

    def write(self, result) -> bool:
        """Write a single GenerationResult to the output file.

        Args:
            result: GenerationResult with .example dict to serialize.

        Returns:
            True if write succeeded, False on I/O error.
        """
        try:
            line = json.dumps(result.example) + "\n"
            with self._lock:
                self._file.write(line)
                self._file.flush()
                self._count += 1
            return True
        except (IOError, OSError) as e:
            print(f"\nError writing result to {self._output_file}: {e}")
            return False

    @property
    def count(self) -> int:
        """Number of results successfully written."""
        return self._count


def load_settings(config_dir: Path) -> Dict:
    """Load settings.yaml configuration."""
    settings_path = config_dir / "settings.yaml"
    if not settings_path.exists():
        raise FileNotFoundError(f"Settings not found: {settings_path}")
    return load_yaml(settings_path)


def create_llm_client(config: Dict, mode: str = "generation",
                      provider_override: str = None, model_override: str = None):
    """
    Create LLM client for generation or improvement.

    Args:
        config: Settings configuration
        mode: "generation" or "improvement"
        provider_override: CLI override for provider (optional)
        model_override: CLI override for model (optional)

    Returns:
        LLM client from shared.llm
    """
    llm_config = config["llm"].get(mode, {})
    provider = provider_override or llm_config.get("provider", "lmstudio")
    model = model_override or llm_config.get("model", "local-model")

    # Build config defaults from settings
    config_defaults = {
        "provider": provider,
        "model": model,
        "temperature": llm_config.get("temperature", 0.7),
        "max_tokens": llm_config.get("max_tokens", 2048),
    }

    # Provider-specific config
    if provider == "unsloth":
        config_defaults["max_seq_length"] = llm_config.get("max_seq_length", 4096)
        config_defaults["load_in_4bit"] = llm_config.get("load_in_4bit", True)
        config_defaults["top_p"] = llm_config.get("top_p", 0.9)
    elif "provider_routing" in llm_config:
        config_defaults["provider_routing"] = llm_config["provider_routing"]

    # Create client using shared.llm factory
    client = create_client(config_defaults=config_defaults)
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

    # Create LLM clients (CLI args override settings.yaml)
    print("Initializing LLM clients...")
    gen_client = create_llm_client(settings, mode="generation",
                                   provider_override=args.provider, model_override=args.model)
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

    # Load docs if provided
    docs: List[DocFile] = []
    if args.docs:
        print(f"Loading docs from: {args.docs}")
        docs = DocsLoader().load(args.docs)
        print(f"Loaded {len(docs)} document(s)")
        print(f"Will generate {args.per_doc} example(s) per doc\n")

    # Generate
    max_iterations = args.max_iterations or settings["improvement"]["max_iterations"]
    args.workers = max(1, args.workers)
    num_workers = args.workers
    results = []

    # Determine output path early so we can stream results to disk
    output_file = Path(args.output) if args.output else _generate_output_path(settings)

    with StreamingResultWriter(output_file, settings) as writer:
        if docs and num_workers > 1:
            # Parallel docs-based generation with multiple workers
            print(f"Using {num_workers} parallel workers for {len(docs)} doc(s)\n")

            # Build work items: one per (doc, repetition, scenario, count) combination
            work_items = []
            task_id = 0
            for doc in docs:
                for rep in range(args.per_doc):
                    for scenario_key, count in targets.items():
                        scenario = generator.scenario_loader.get_scenario(scenario_key)
                        if not scenario:
                            print(f"Warning: Scenario not found: {scenario_key}")
                            continue
                        for _ in range(count):
                            work_items.append((
                                scenario_key, scenario, max_iterations,
                                config_dir, scenarios_dir, rubrics_dir,
                                settings, args.provider, args.model, doc, task_id
                            ))
                            task_id += 1

            results.extend(_run_parallel_generation(work_items, num_workers, writer))
        elif docs:
            # Sequential docs-based generation (single worker)
            total_docs = len(docs)
            for doc_idx, doc in enumerate(docs, 1):
                print(f"\n--- Document {doc_idx}/{total_docs}: {doc.path} ---")
                for rep in range(args.per_doc):
                    if args.per_doc > 1:
                        print(f"  Repetition {rep + 1}/{args.per_doc}")
                    batch_results = generator.generate_batch(
                        targets=targets,
                        max_iterations=max_iterations,
                        randomize_params=True,
                        doc_context=doc
                    )
                    for result in batch_results:
                        writer.write(result)
                    results.extend(batch_results)
        elif num_workers > 1:
            # Parallel generation with multiple workers (no docs)
            print(f"Using {num_workers} parallel workers\n")

            # Build work items
            work_items = []
            task_id = 0
            for scenario_key, count in targets.items():
                scenario = generator.scenario_loader.get_scenario(scenario_key)
                if not scenario:
                    print(f"Warning: Scenario not found: {scenario_key}")
                    continue
                for _ in range(count):
                    work_items.append((
                        scenario_key, scenario, max_iterations,
                        config_dir, scenarios_dir, rubrics_dir,
                        settings, args.provider, args.model, None, task_id
                    ))
                    task_id += 1

            results.extend(_run_parallel_generation(work_items, num_workers, writer))
        else:
            # Standard sequential generation (no docs)
            results = generator.generate_batch(
                targets=targets,
                max_iterations=max_iterations,
                randomize_params=True
            )
            # Stream sequential results that were accumulated by generate_batch
            for result in results:
                writer.write(result)

        print(f"\nStreamed {writer.count} examples to {output_file}")

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

    # Load configuration
    config_dir = Path(args.config_dir or "SynthChat/config")
    settings = load_settings(config_dir)
    rubrics_dir = Path(args.rubrics_dir or "SynthChat/rubrics")

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


def _run_parallel_generation(work_items: List, num_workers: int,
                             writer=None) -> List:
    """
    Execute work items in parallel using a thread pool.

    Shared execution logic for both docs-based and non-docs parallel generation.
    Handles progress tracking, error reporting, streaming to disk, and graceful
    shutdown on interrupts.

    Args:
        work_items: List of tuples to pass to _generate_single_example
        num_workers: Number of parallel worker threads
        writer: Optional StreamingResultWriter to stream results as they complete

    Returns:
        List of GenerationResult objects (successful results only),
        sorted by task_id to preserve input ordering.
    """
    if not work_items:
        print("No work items to process (check scenario names)")
        return []

    total = len(work_items)
    completed = 0
    lock = threading.Lock()
    indexed_results = []  # List of (task_id, result) for ordering

    def update_progress():
        nonlocal completed
        with lock:
            completed += 1
            pct = (completed / total * 100) if total > 0 else 0
            print(f"\rProgress: {completed}/{total} ({pct:.1f}%)", end="", flush=True)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_generate_single_example, item): item for item in work_items}

        try:
            for future in as_completed(futures):
                result, error, task_id = future.result()
                if error:
                    print(f"\n{error}")
                if result:
                    if writer:
                        writer.write(result)
                    indexed_results.append((task_id, result))
                update_progress()
        except BaseException as e:
            print(f"\nInterrupted: {e}. Shutting down workers...")
            executor.shutdown(wait=False, cancel_futures=True)
            raise

    print()  # Newline after progress

    # Sort by task_id to preserve input document order
    indexed_results.sort(key=lambda x: x[0])
    return [result for _, result in indexed_results]


def _create_worker_generator(config_dir: Path, scenarios_dir: Path, rubrics_dir: Path,
                              settings: Dict, provider: str = None, model: str = None):
    """Create a new generator instance for a worker thread."""
    gen_client = create_llm_client(settings, mode="generation",
                                   provider_override=provider, model_override=model)
    improve_client = create_llm_client(settings, mode="improvement",
                                       provider_override=provider, model_override=model)

    validation_config = config_dir / "validation.yaml"
    engine = ImprovementEngine(
        llm_client=improve_client,
        rubrics_dir=rubrics_dir,
        config_path=validation_config,
        enable_interactions=settings["logging"]["save_interactions"]
    )

    generator = SynthChatGenerator(
        config_dir=config_dir,
        scenarios_dir=scenarios_dir,
        rubrics_dir=rubrics_dir,
        llm_client=gen_client,
        engine=engine,
        enable_stage_validation=settings["generation"]["stage_validation"]
    )
    return generator


def _generate_single_example(args_tuple):
    """Worker function to generate a single example.

    Returns:
        Tuple of (result, error, task_id) where task_id preserves input ordering.
    """
    (scenario_key, scenario, max_iterations, config_dir, scenarios_dir,
     rubrics_dir, settings, provider, model, doc_context, task_id) = args_tuple

    try:
        # Each worker creates its own generator (thread-safe LLM clients)
        generator = _create_worker_generator(
            config_dir, scenarios_dir, rubrics_dir, settings, provider, model
        )

        result = generator.generate_single(
            scenario_key, scenario, max_iterations, True, doc_context
        )
        return result, None, task_id
    except Exception as e:
        return None, f"Task {task_id} error for {scenario_key}: {e}", task_id


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
    generate_parser.add_argument("--provider", help="LLM provider (overrides settings.yaml)")
    generate_parser.add_argument("--model", help="Model name (overrides settings.yaml)")
    generate_parser.add_argument("--docs", help="Path to doc file or folder (seed data for generation)")
    generate_parser.add_argument("--per-doc", type=int, default=1, help="Examples to generate per doc (default: 1)")
    generate_parser.add_argument("--workers", "-w", type=int, default=1, help="Number of parallel workers (default: 1)")

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
    improve_parser.add_argument("--provider", help="LLM provider (overrides settings.yaml)")
    improve_parser.add_argument("--model", help="Model name (overrides settings.yaml)")
    improve_parser.add_argument("--workers", "-w", type=int, default=1, help="Number of parallel workers (default: 1)")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate dataset (no improvement)")
    validate_parser.add_argument("--input", "-i", required=True, help="Input JSONL file")
    validate_parser.add_argument("--config-dir", help="Config directory path")
    validate_parser.add_argument("--rubrics-dir", help="Rubrics directory path")
    validate_parser.add_argument("--rubrics", help="Comma-separated rubric names")
    validate_parser.add_argument("--provider", help="LLM provider (overrides settings.yaml)")
    validate_parser.add_argument("--model", help="Model name (overrides settings.yaml)")

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
