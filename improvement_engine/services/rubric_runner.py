"""Refactored RubricRunner - thin CLI wrapper.

Responsibility: ONLY CLI interactions (file I/O, selection, display).
All actual work delegated to ImprovementEngine.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.llm import create_client
from shared.llm.config import LLMConfig
from improvement_engine.engine import ImprovementEngine, ImprovementResult
from improvement_engine.services.data import RubricRepository
from improvement_engine.utils.logger import ImproveLogger


class RubricRunner:
    """
    Thin CLI wrapper for ImprovementEngine.

    Responsibility: ONLY CLI concerns (SRP).
    - Command-line argument parsing
    - Interactive rubric selection
    - File I/O
    - Progress display

    Delegates ALL improvement logic to ImprovementEngine.
    """

    def __init__(
        self,
        rubrics_dir: Path,
        backend: str = "lmstudio",
        host: Optional[str] = None,
        port: Optional[int] = None,
        config_path: Optional[Path] = None
    ):
        """
        Initialize RubricRunner.

        Args:
            rubrics_dir: Directory containing rubric YAML files
            backend: LLM backend ("lmstudio", "ollama", "openrouter")
            host: LLM host (optional)
            port: LLM port (optional)
            config_path: Path to scope_config.yaml (optional)
        """
        self.logger = ImproveLogger()
        self.rubrics_dir = Path(rubrics_dir)

        # Initialize LLM client
        config = LLMConfig(backend=backend)
        if host:
            if backend == "lmstudio":
                config.lmstudio_host = host
            elif backend == "ollama":
                config.ollama_host = host
        if port and backend == "ollama":
            config.ollama_port = port

        llm_client = create_client(config=config)

        # Initialize engine (does all the real work!)
        self.engine = ImprovementEngine(
            llm_client=llm_client,
            rubrics_dir=rubrics_dir,
            config_path=config_path,
            logger=self.logger
        )

        # Access to rubric repository for selection
        self.rubric_repo = self.engine.rubric_repo

    def list_rubrics(self) -> None:
        """Display available rubrics."""
        print("\n=== Available Rubrics ===\n")

        metadata_list = self.rubric_repo.list_metadata()
        for meta in metadata_list:
            print(f"  {meta.key}")
            print(f"    Name: {meta.name}")
            print(f"    Description: {meta.description}")
            print(f"    Scope: {meta.scope}")
            print(f"    Threshold: {meta.pass_threshold}")
            print()

    def select_rubrics_interactive(self) -> List[str]:
        """Interactive checkbox-style rubric selection."""
        print("\n=== Select Rubrics ===\n")

        metadata_list = self.rubric_repo.list_metadata()
        rubric_keys = [meta.key for meta in metadata_list]

        for i, meta in enumerate(metadata_list, 1):
            print(f"  [{i}] {meta.name}")
            print(f"      {meta.description}")
            print()

        print("Enter numbers separated by commas (e.g., 1,2,3)")
        print("Or 'all' for all rubrics, 'q' to quit")

        while True:
            choice = input("\nSelect rubrics: ").strip().lower()

            if choice == 'q':
                return []

            if choice == 'all':
                return rubric_keys

            try:
                indices = [int(x.strip()) - 1 for x in choice.split(",")]
                selected = [rubric_keys[i] for i in indices if 0 <= i < len(rubric_keys)]
                if selected:
                    return selected
                print("No valid selections. Try again.")
            except (ValueError, IndexError):
                print("Invalid input. Enter numbers like: 1,2,3")

    def run_on_file(
        self,
        file_path: Path,
        output_path: Path,
        rubric_keys: List[str],
        start_line: int = 1,
        end_line: Optional[int] = None,
        max_iterations: int = 3
    ) -> None:
        """
        Run improvement on a JSONL file.

        Args:
            file_path: Input JSONL file
            output_path: Output JSONL file
            rubric_keys: List of rubric keys to use
            start_line: Starting line number (1-indexed)
            end_line: Ending line number (1-indexed, None = all)
            max_iterations: Max improvement iterations per example
        """
        # Read input file
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Determine range
        if end_line is None:
            end_line = len(lines)

        start_idx = start_line - 1  # Convert to 0-indexed
        end_idx = end_line

        print(f"\nProcessing lines {start_line} to {end_line} ({end_idx - start_idx} examples)")
        print(f"Rubrics: {', '.join(rubric_keys)}")
        print(f"Max iterations: {max_iterations}\n")

        # Process each example
        results = []
        for i in range(start_idx, end_idx):
            if i >= len(lines):
                break

            line_num = i + 1
            print(f"--- Line {line_num} ---")

            try:
                example = json.loads(lines[i])

                # Run improvement engine
                result = self.engine.run(
                    example=example,
                    rubric_keys=rubric_keys,
                    max_iterations=max_iterations
                )

                # Display result
                self._display_result(result)

                # Save improved example
                results.append(json.dumps(result.improved_example))

            except Exception as e:
                self.logger.error(f"Error processing line {line_num}: {e}")
                # Save original on error
                results.append(lines[i].strip())

        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for result_line in results:
                f.write(result_line + '\n')

        print(f"\n✅ Output written to: {output_path}")

    def run_on_example(
        self,
        example: dict,
        rubric_keys: List[str],
        max_iterations: int = 3
    ) -> ImprovementResult:
        """
        Run improvement on a single example.

        Args:
            example: Example dict
            rubric_keys: List of rubric keys to use
            max_iterations: Max improvement iterations

        Returns:
            ImprovementResult
        """
        result = self.engine.run(
            example=example,
            rubric_keys=rubric_keys,
            max_iterations=max_iterations
        )

        self._display_result(result)
        return result

    def _display_result(self, result: ImprovementResult) -> None:
        """Display improvement result."""
        if result.passed:
            print(f"✅ PASSED after {result.iterations} iteration(s)")
        else:
            print(f"❌ FAILED after {result.iterations} iteration(s)")

        print(f"Scopes improved: {', '.join(result.scopes_improved) if result.scopes_improved else 'none'}")

        print("Scores:")
        for key, score in result.final_scores.items():
            print(f"  {key}: {score:.2f}")
        print()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run rubric-based improvement on examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available rubrics
  python rubric_runner.py --list

  # Interactive mode
  python rubric_runner.py --file dataset.jsonl --output improved.jsonl

  # Specify rubrics
  python rubric_runner.py --file dataset.jsonl --output improved.jsonl \\
    --rubrics thinking_quality,context_alignment

  # Process specific lines
  python rubric_runner.py --file dataset.jsonl --output improved.jsonl \\
    --start-line 1 --end-line 10 --rubrics thinking_quality
        """
    )

    parser.add_argument("--list", action="store_true", help="List available rubrics")
    parser.add_argument("--file", type=Path, help="Input JSONL file")
    parser.add_argument("--output", type=Path, help="Output JSONL file")
    parser.add_argument("--rubrics", help="Comma-separated rubric keys")
    parser.add_argument("--start-line", type=int, default=1, help="Start line (1-indexed)")
    parser.add_argument("--end-line", type=int, help="End line (1-indexed)")
    parser.add_argument("--max-iterations", type=int, default=3, help="Max iterations")
    parser.add_argument("--backend", default="lmstudio", choices=["lmstudio", "ollama", "openrouter"])
    parser.add_argument("--host", help="LLM host")
    parser.add_argument("--port", type=int, help="LLM port")
    parser.add_argument("--config", type=Path, help="Path to scope_config.yaml")

    args = parser.parse_args()

    # Determine rubrics directory
    rubrics_dir = Path(__file__).parent.parent / "rubrics"

    # Initialize runner
    runner = RubricRunner(
        rubrics_dir=rubrics_dir,
        backend=args.backend,
        host=args.host,
        port=args.port,
        config_path=args.config
    )

    # List mode
    if args.list:
        runner.list_rubrics()
        return

    # File processing mode
    if not args.file:
        print("Error: --file required (or use --list)")
        parser.print_help()
        sys.exit(1)

    if not args.output:
        print("Error: --output required")
        parser.print_help()
        sys.exit(1)

    # Get rubric selection
    if args.rubrics:
        rubric_keys = [k.strip() for k in args.rubrics.split(",")]
    else:
        rubric_keys = runner.select_rubrics_interactive()

    if not rubric_keys:
        print("No rubrics selected. Exiting.")
        sys.exit(0)

    # Run improvement
    runner.run_on_file(
        file_path=args.file,
        output_path=args.output,
        rubric_keys=rubric_keys,
        start_line=args.start_line,
        end_line=args.end_line,
        max_iterations=args.max_iterations
    )


if __name__ == "__main__":
    main()
