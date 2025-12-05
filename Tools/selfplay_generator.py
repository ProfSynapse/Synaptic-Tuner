#!/usr/bin/env python3
"""
Self-Play Synthetic Data Generator for KTO Training

This module generates synthetic training data by:
1. Sampling prompts from existing sets or generating variations
2. Sending prompts to a fine-tuned model via LM Studio
3. Validating responses using the existing validator
4. Optionally executing tool calls against a real Obsidian vault via MCP
5. Collecting both correct and incorrect examples
6. Building an interleaved KTO dataset (True/False/True/False pattern)

Usage:
    # Basic usage (just validation)
    python Tools/selfplay_generator.py \
        --model claudesidian-mcp \
        --prompt-set Evaluator/prompts/tool_prompts.json \
        --output Datasets/syngen_selfplay_$(date +%Y%m%d).jsonl \
        --num-examples 1000

    # With MCP execution (requires Obsidian vault + MCP server)
    python Tools/selfplay_generator.py \
        --model claudesidian-mcp \
        --prompt-set Evaluator/prompts/tool_prompts.json \
        --output Datasets/syngen_selfplay_$(date +%Y%m%d).jsonl \
        --num-examples 1000 \
        --execute-mcp \
        --vault-path /path/to/test/vault

    # Temperature sampling for diversity
    python Tools/selfplay_generator.py \
        --model claudesidian-mcp \
        --prompt-set Evaluator/prompts/tool_prompts.json \
        --output Datasets/syngen_selfplay_$(date +%Y%m%d).jsonl \
        --num-examples 1000 \
        --temperature 0.7 \
        --num-variations 3
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from Evaluator.lmstudio_client import LMStudioClient
from Evaluator.config import LMStudioSettings
from Evaluator.prompt_sets import load_prompt_set, PromptCase
from tools.validate_syngen import validate_example, ExampleReport


@dataclass
class GenerationStats:
    """Track statistics during generation."""
    total_attempts: int = 0
    valid_examples: int = 0
    invalid_examples: int = 0
    mcp_executions: int = 0
    mcp_failures: int = 0
    errors: int = 0

    def summary(self) -> str:
        """Return a summary string."""
        return (
            f"\nGeneration Statistics:\n"
            f"  Total attempts: {self.total_attempts}\n"
            f"  Valid examples: {self.valid_examples}\n"
            f"  Invalid examples: {self.invalid_examples}\n"
            f"  MCP executions: {self.mcp_executions}\n"
            f"  MCP failures: {self.mcp_failures}\n"
            f"  Errors: {self.errors}\n"
            f"  Success rate: {self.valid_examples / max(1, self.total_attempts) * 100:.1f}%\n"
        )


class PromptVariator:
    """Generate variations of prompts for diversity."""

    def __init__(self, temperature_variations: List[float] = None):
        """Initialize with temperature variations."""
        self.temperature_variations = temperature_variations or [0.3, 0.5, 0.7, 0.9]

    def generate_variations(
        self,
        prompt_case: PromptCase,
        num_variations: int = 1
    ) -> List[Dict[str, Any]]:
        """Generate variations of a prompt case.

        Variations include:
        - Different temperatures
        - Slight rewording (future enhancement)
        - Different context values (future enhancement)

        For now, just use different temperatures.
        """
        variations = []

        for _ in range(num_variations):
            temp = random.choice(self.temperature_variations)
            variation = {
                'prompt_case': prompt_case,
                'temperature': temp,
                'top_p': 0.95,  # Standard value
            }
            variations.append(variation)

        return variations


class SelfPlayGenerator:
    """Main orchestrator for self-play data generation."""

    def __init__(
        self,
        model_name: str,
        prompt_set_path: Path,
        output_path: Path,
        num_examples: int = 1000,
        temperature: float = 0.7,
        num_variations: int = 3,
        execute_mcp: bool = False,
        vault_path: Optional[Path] = None,
        lmstudio_host: Optional[str] = None,
        lmstudio_port: Optional[int] = None,
    ):
        """Initialize the self-play generator.

        Args:
            model_name: LM Studio model name
            prompt_set_path: Path to prompt set JSON
            output_path: Output JSONL file path
            num_examples: Target number of examples to generate
            temperature: Default temperature for sampling
            num_variations: Number of variations per prompt
            execute_mcp: Whether to execute tool calls via MCP
            vault_path: Path to test Obsidian vault (if execute_mcp)
            lmstudio_host: LM Studio host (default: localhost)
            lmstudio_port: LM Studio port (default: 1234)
        """
        self.model_name = model_name
        self.prompt_set_path = prompt_set_path
        self.output_path = output_path
        self.num_examples = num_examples
        self.temperature = temperature
        self.num_variations = num_variations
        self.execute_mcp = execute_mcp
        self.vault_path = vault_path

        # Initialize LM Studio client
        settings = LMStudioSettings(
            model=model_name,
            host=lmstudio_host or "localhost",
            port=lmstudio_port or 1234,
        )
        self.client = LMStudioClient(settings=settings)

        # Initialize prompt variator
        self.variator = PromptVariator()

        # Statistics
        self.stats = GenerationStats()

        # Load prompt set
        self.prompt_cases = load_prompt_set(prompt_set_path)
        print(f"✓ Loaded {len(self.prompt_cases)} prompts from {prompt_set_path}")

        # Validate LM Studio connection
        if not self.client.is_server_running():
            raise RuntimeError("LM Studio server is not running or not accessible")
        print(f"✓ Connected to LM Studio at {settings.base_url()}")

    def generate_response(
        self,
        prompt_case: PromptCase,
        temperature: float,
        top_p: float = 0.95,
    ) -> Tuple[str, Optional[str]]:
        """Generate a response from the model.

        Returns:
            (response_text, error_message)
        """
        try:
            # Build messages
            messages = prompt_case.chat_messages()

            # Call model with temperature
            response = self.client.chat(
                messages,
                temperature=temperature,
                top_p=top_p,
            )

            return response.message, None

        except Exception as e:
            return None, str(e)

    def validate_response(
        self,
        prompt_case: PromptCase,
        response_text: str,
    ) -> Tuple[bool, ExampleReport]:
        """Validate a response using the existing validator.

        Returns:
            (is_valid, validation_report)
        """
        # Create example in ChatML format
        example = {
            "conversations": [
                {"role": "user", "content": prompt_case.question},
                {"role": "assistant", "content": response_text},
            ],
            "label": True,  # Assume desirable for now
        }

        # Add system message if present
        if prompt_case.metadata.get("system"):
            example["conversations"].insert(
                0,
                {"role": "system", "content": prompt_case.metadata["system"]}
            )

        # Validate
        report = validate_example(1, example)

        return report.is_valid, report

    def execute_mcp_tools(
        self,
        response_text: str,
    ) -> Tuple[bool, Optional[str]]:
        """Execute tool calls against MCP server (if enabled).

        This is a placeholder for MCP integration.
        In the future, this would:
        1. Extract tool calls from response
        2. Send them to MCP server
        3. Collect results
        4. Return success/failure

        Returns:
            (success, error_message)
        """
        if not self.execute_mcp:
            return True, None

        # TODO: Implement actual MCP execution
        # For now, just return success
        print("  [MCP] Execution not yet implemented (placeholder)")
        return True, None

    def collect_example(
        self,
        prompt_case: PromptCase,
        response_text: str,
        is_valid: bool,
        validation_report: ExampleReport,
    ) -> Dict[str, Any]:
        """Collect an example in ChatML format.

        Args:
            prompt_case: Original prompt case
            response_text: Model's response
            is_valid: Whether response is valid
            validation_report: Validation report

        Returns:
            Example dict in ChatML format
        """
        # Build conversations
        conversations = [
            {"role": "user", "content": prompt_case.question},
            {"role": "assistant", "content": response_text},
        ]

        # Add system message if present
        if prompt_case.metadata.get("system"):
            conversations.insert(
                0,
                {"role": "system", "content": prompt_case.metadata["system"]}
            )

        # Label based on validity
        label = is_valid

        return {
            "conversations": conversations,
            "label": label,
            "metadata": {
                "prompt_id": prompt_case.id,
                "generated_at": datetime.now().isoformat(),
                "validation_issues": [
                    {"level": issue.level, "message": issue.message}
                    for issue in validation_report.issues
                ] if not is_valid else [],
            }
        }

    def generate_dataset(self) -> None:
        """Generate the complete dataset."""
        print(f"\nGenerating {self.num_examples} examples...")
        print(f"  Model: {self.model_name}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Variations per prompt: {self.num_variations}")
        print(f"  Execute MCP: {self.execute_mcp}")
        print()

        collected_examples = []

        # Calculate how many times to cycle through prompts
        examples_per_prompt = self.num_variations
        cycles_needed = (self.num_examples + len(self.prompt_cases) * examples_per_prompt - 1) // (len(self.prompt_cases) * examples_per_prompt)

        for cycle in range(cycles_needed):
            print(f"Cycle {cycle + 1}/{cycles_needed}")

            # Shuffle prompts for variety
            shuffled_prompts = random.sample(self.prompt_cases, len(self.prompt_cases))

            for prompt_case in shuffled_prompts:
                if len(collected_examples) >= self.num_examples:
                    break

                # Generate variations
                variations = self.variator.generate_variations(
                    prompt_case,
                    num_variations=self.num_variations
                )

                for variation in variations:
                    if len(collected_examples) >= self.num_examples:
                        break

                    self.stats.total_attempts += 1

                    # Generate response
                    print(f"  [{self.stats.total_attempts}/{self.num_examples}] {prompt_case.id} (T={variation['temperature']:.1f})...", end=" ")

                    response_text, error = self.generate_response(
                        variation['prompt_case'],
                        variation['temperature'],
                        variation['top_p'],
                    )

                    if error:
                        print(f"ERROR: {error}")
                        self.stats.errors += 1
                        continue

                    # Validate response
                    is_valid, validation_report = self.validate_response(
                        prompt_case,
                        response_text,
                    )

                    # Execute MCP if enabled
                    if self.execute_mcp:
                        mcp_success, mcp_error = self.execute_mcp_tools(response_text)
                        self.stats.mcp_executions += 1
                        if not mcp_success:
                            self.stats.mcp_failures += 1
                            is_valid = False  # Mark as invalid if MCP execution failed

                    # Collect example
                    example = self.collect_example(
                        prompt_case,
                        response_text,
                        is_valid,
                        validation_report,
                    )
                    collected_examples.append(example)

                    # Update stats
                    if is_valid:
                        self.stats.valid_examples += 1
                        print("✓ VALID")
                    else:
                        self.stats.invalid_examples += 1
                        print("✗ INVALID")

                if len(collected_examples) >= self.num_examples:
                    break

        # Interleave examples for KTO training
        print("\nInterleaving examples for KTO training...")
        interleaved_examples = self._interleave_examples(collected_examples)

        # Write to output file
        print(f"Writing {len(interleaved_examples)} examples to {self.output_path}...")
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for example in interleaved_examples:
                # Remove metadata before writing (optional)
                output_example = {
                    "conversations": example["conversations"],
                    "label": example["label"],
                }
                f.write(json.dumps(output_example, ensure_ascii=False) + '\n')

        print(f"✓ Dataset written to {self.output_path}")
        print(self.stats.summary())

    def _interleave_examples(
        self,
        examples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Interleave examples in True/False/True/False pattern for KTO.

        This ensures mixed batches during training.
        """
        # Separate valid and invalid examples
        valid = [ex for ex in examples if ex["label"] is True]
        invalid = [ex for ex in examples if ex["label"] is False]

        print(f"  Valid examples: {len(valid)}")
        print(f"  Invalid examples: {len(invalid)}")

        # Balance if needed
        min_count = min(len(valid), len(invalid))
        if min_count == 0:
            print("  WARNING: No invalid examples! Cannot create interleaved dataset.")
            print("  Returning all valid examples (not suitable for KTO training)")
            return valid

        # Trim to equal sizes
        valid = valid[:min_count]
        invalid = invalid[:min_count]

        # Shuffle both
        random.shuffle(valid)
        random.shuffle(invalid)

        # Interleave: True, False, True, False, ...
        interleaved = []
        for v, i in zip(valid, invalid):
            interleaved.append(v)
            interleaved.append(i)

        print(f"  Interleaved dataset size: {len(interleaved)}")

        return interleaved


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data via self-play"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="LM Studio model name"
    )
    parser.add_argument(
        "--prompt-set",
        type=Path,
        required=True,
        help="Path to prompt set JSON file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=1000,
        help="Number of examples to generate (default: 1000)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Default temperature for sampling (default: 0.7)"
    )
    parser.add_argument(
        "--num-variations",
        type=int,
        default=3,
        help="Number of variations per prompt (default: 3)"
    )
    parser.add_argument(
        "--execute-mcp",
        action="store_true",
        help="Execute tool calls via MCP (requires vault-path)"
    )
    parser.add_argument(
        "--vault-path",
        type=Path,
        help="Path to test Obsidian vault (required if --execute-mcp)"
    )
    parser.add_argument(
        "--lmstudio-host",
        type=str,
        help="LM Studio host (default: localhost)"
    )
    parser.add_argument(
        "--lmstudio-port",
        type=int,
        help="LM Studio port (default: 1234)"
    )

    args = parser.parse_args()

    # Validate args
    if args.execute_mcp and not args.vault_path:
        parser.error("--vault-path is required when --execute-mcp is set")

    if args.output.exists():
        response = input(f"Output file '{args.output}' exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)

    # Create output directory if needed
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Initialize generator
    try:
        generator = SelfPlayGenerator(
            model_name=args.model,
            prompt_set_path=args.prompt_set,
            output_path=args.output,
            num_examples=args.num_examples,
            temperature=args.temperature,
            num_variations=args.num_variations,
            execute_mcp=args.execute_mcp,
            vault_path=args.vault_path,
            lmstudio_host=args.lmstudio_host,
            lmstudio_port=args.lmstudio_port,
        )
    except Exception as e:
        print(f"Error initializing generator: {e}", file=sys.stderr)
        sys.exit(1)

    # Generate dataset
    try:
        generator.generate_dataset()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Partial results may be saved.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during generation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
