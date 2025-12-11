"""Refactored RubricRunner - thin CLI wrapper.

Responsibility: ONLY CLI interactions (file I/O, selection, display).
All actual work delegated to ImprovementEngine.
"""

import argparse
import json
import sys
from datetime import datetime
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
        # Read model from config.yaml and environment variables
        from improvement_engine.utils.yaml_loader import load_config
        from shared.utilities.env import load_env_file
        import os

        # Load environment variables from .env
        load_env_file()

        config_data = load_config("config")
        llm_config_data = config_data.get("llm", {})

        # Get model based on backend
        if backend == "lmstudio":
            model = "local-model"  # LM Studio uses whatever model is loaded in UI
        elif backend == "ollama":
            model = llm_config_data.get("ollama_model", "qwen2.5:latest")
        else:  # openrouter
            model = llm_config_data.get("openrouter_model", "anthropic/claude-3.5-sonnet")

        # Get host/port from CLI args or environment variables
        lmstudio_host = host if host and backend == "lmstudio" else os.getenv("LMSTUDIO_HOST", "localhost")
        lmstudio_port = port if port and backend == "lmstudio" else int(os.getenv("LMSTUDIO_PORT", "1234"))
        ollama_host = host if host and backend == "ollama" else os.getenv("OLLAMA_HOST", "localhost")
        ollama_port = port if port and backend == "ollama" else int(os.getenv("OLLAMA_PORT", "11434"))

        config = LLMConfig(
            provider=backend,
            model=model,
            lmstudio_host=lmstudio_host,
            lmstudio_port=lmstudio_port,
            ollama_host=ollama_host,
            ollama_port=ollama_port
        )

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
        # Extract dataset name from file path for interaction logging
        # Format: parentFolder_filename (without .jsonl extension)
        file_path = Path(file_path)
        parent_folder = file_path.parent.name
        filename_stem = file_path.stem
        dataset_name = f"{parent_folder}_{filename_stem}"

        # Reinitialize engine with dataset name
        self.engine.interaction_logger.dataset_name = dataset_name
        # Update log file with dataset name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        interactions_dir = Path(self.rubrics_dir).parent / "interactions"
        self.engine.interaction_logger.log_file = interactions_dir / f"interactions_{dataset_name}_{timestamp}.jsonl"
        self.logger.info(f"Updated interaction log: {self.engine.interaction_logger.log_file}")

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

        # Open output file for incremental writing
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Collect for markdown review
        original_examples = []
        results = []

        with open(output_path, 'w', encoding='utf-8') as f_out:
            # Process each example
            for i in range(start_idx, end_idx):
                if i >= len(lines):
                    break

                line_num = i + 1
                print(f"--- Line {line_num} ---")

                try:
                    example = json.loads(lines[i])
                    original_examples.append(example)

                    # Run improvement engine
                    result = self.engine.run(
                        example=example,
                        rubric_keys=rubric_keys,
                        max_iterations=max_iterations
                    )
                    results.append(result)

                    # Display result
                    self._display_result(result)

                    # Write improved example immediately
                    f_out.write(json.dumps(result.improved_example) + '\n')
                    f_out.flush()  # Ensure it's written to disk

                except Exception as e:
                    self.logger.error(f"Error processing line {line_num}: {e}")
                    # Write original on error
                    f_out.write(lines[i].strip() + '\n')
                    f_out.flush()

        print(f"\n✅ Output written to: {output_path}")

        # Write markdown review
        if original_examples and results:
            md_path = self._write_markdown_review(output_path, original_examples, results)
            print(f"📝 Markdown review: {md_path}")

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

    def _write_markdown_review(
        self,
        output_path: Path,
        original_examples: List[dict],
        results: List[ImprovementResult]
    ) -> Path:
        """
        Write markdown review file for human inspection.

        Args:
            output_path: Path to JSONL output file
            original_examples: List of original examples
            results: List of ImprovementResult for each example

        Returns:
            Path to markdown review file
        """
        md_path = output_path.with_suffix('.review.md')

        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# Improvement Engine Review\n\n")
            f.write(f"**Output file:** `{output_path.name}`\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")

            # Summary
            passed = sum(1 for r in results if r.passed)
            f.write("## Summary\n\n")
            f.write(f"- Total examples: {len(results)}\n")
            f.write(f"- Passed: {passed}\n")
            f.write(f"- Failed: {len(results) - passed}\n\n")

            # Each example
            for idx, (orig, result) in enumerate(zip(original_examples, results), 1):
                f.write(f"---\n\n## Example {idx}\n\n")

                # Status
                status = "✅ PASSED" if result.passed else "❌ FAILED"
                f.write(f"**Status:** {status} (iterations: {result.iterations})\n\n")

                # Scores
                f.write("### Scores\n\n")
                for key, score in result.final_scores.items():
                    threshold_marker = "✓" if score >= 0.8 else "✗"
                    f.write(f"- {key}: {score:.2f} {threshold_marker}\n")
                f.write("\n")

                # Original example
                f.write("### Original Example\n\n")
                convs = orig.get("conversations", [])

                for conv in convs:
                    role = conv.get("role", "unknown").upper()
                    content = conv.get("content", "")

                    f.write(f"**{role}:**\n\n")
                    # Truncate very long content
                    if len(content) > 3000:
                        f.write(f"```\n{content[:3000]}...\n```\n\n")
                    else:
                        f.write(f"```\n{content}\n```\n\n")

                # Improved example
                f.write("### Improved Example\n\n")
                improved_convs = result.improved_example.get("conversations", [])

                for conv in improved_convs:
                    role = conv.get("role", "unknown").upper()
                    content = conv.get("content", "")

                    f.write(f"**{role}:**\n\n")
                    # Truncate very long content
                    if len(content) > 3000:
                        f.write(f"```\n{content[:3000]}...\n```\n\n")
                    else:
                        f.write(f"```\n{content}\n```\n\n")

                # Tool calls if present
                for conv in improved_convs:
                    if "tool_calls" in conv:
                        f.write("**Tool Calls:**\n\n")
                        for tc in conv["tool_calls"]:
                            func = tc.get("function", {})
                            f.write(f"- `{func.get('name', 'unknown')}`\n")
                            f.write(f"  ```json\n  {func.get('arguments', '{}')}\n  ```\n\n")

        return md_path


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
