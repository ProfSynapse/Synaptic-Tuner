"""Command-line entry point for the Evaluator.

This module provides the main CLI for running model evaluations
against Ollama or LM Studio backends.
"""
from __future__ import annotations

import argparse
import sys
from typing import List

from .cli_utils import (
    build_metadata,
    build_settings_kwargs,
    default_output_path,
    determine_exit_code,
)
from .client_factory import create_client, create_settings
from .config import (
    EvaluatorConfig,
    PromptFilter,
    expand_path,
    parse_tags,
)
from .prompt_sets import filter_prompts, load_prompt_cases
from .reporting import build_run_payload, console_summary, render_markdown, write_json
from .runner import evaluate_cases


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate tool-calling models via Ollama or LM Studio.",
        epilog="""
Backend Configuration:
  Ollama:    OLLAMA_HOST (default: 127.0.0.1), OLLAMA_PORT (default: 11434)
  LM Studio: LMSTUDIO_HOST (default: 127.0.0.1), LMSTUDIO_PORT (default: 1234)
        """,
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "lmstudio"],
        default="ollama",
        help="Backend to use for evaluation (default: ollama)",
    )
    parser.add_argument("--model", required=True, help="Model name (e.g., claudesidian-mcp)")
    parser.add_argument("--prompt-set", default="Evaluator/prompts/baseline.json", help="Path to prompt set file")
    parser.add_argument("--tags", help="Comma-separated tag filter")
    parser.add_argument("--limit", type=int, help="Max prompts to evaluate")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, help="Optional generation seed")
    parser.add_argument("--host", help="Override backend host (OLLAMA_HOST or LMSTUDIO_HOST)")
    parser.add_argument("--port", type=int, help="Override backend port (OLLAMA_PORT or LMSTUDIO_PORT)")
    parser.add_argument("--retries", type=int, default=2, help="HTTP retry attempts")
    parser.add_argument("--timeout", type=float, default=60.0, help="Request timeout (seconds)")
    parser.add_argument("--output", help="Where to write JSON results (defaults to Evaluator/results/run_<ts>.json)")
    parser.add_argument("--markdown", help="Optional Markdown summary output path")
    parser.add_argument("--dry-run", action="store_true", help="Skip backend calls (for smoke tests)")
    parser.add_argument(
        "--validate-context",
        action="store_true",
        help="Validate that model uses IDs from system prompt (requires prompts with expected_context)",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    """Main entry point for CLI evaluation."""
    args = parse_args(argv or sys.argv[1:])

    # Resolve paths
    prompt_path = expand_path(args.prompt_set)
    output_path = expand_path(args.output) if args.output else default_output_path()
    markdown_path = expand_path(args.markdown) if args.markdown else None

    # Build configuration
    prompt_filter = PromptFilter(tags=parse_tags(args.tags), limit=args.limit)
    config = EvaluatorConfig(
        prompts_path=prompt_path,
        output_path=output_path,
        save_markdown=bool(markdown_path),
        filter=prompt_filter,
        retries=args.retries,
        request_timeout=args.timeout,
        dry_run=args.dry_run,
    )
    config.validate()
    config.ensure_output_parent()
    if markdown_path:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)

    # Load and filter prompts
    cases = load_prompt_cases(config.prompts_path)
    selected_cases = filter_prompts(cases, config.filter)
    if not selected_cases:
        print("No prompts matched the provided filters.", file=sys.stderr)
        return 1

    # Get settings kwargs for host/port overrides
    settings_kwargs = build_settings_kwargs(args)

    # Create settings and client using factory (eliminates if/else chain)
    settings = create_settings(
        backend=args.backend,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        **settings_kwargs,
    )
    client = create_client(
        backend=args.backend,
        settings=settings,
        timeout=config.request_timeout,
        retries=config.retries,
    )

    # Run evaluation
    records = evaluate_cases(
        selected_cases,
        client=client,
        dry_run=config.dry_run,
        validate_context=args.validate_context,
    )

    # Build and save results
    metadata = build_metadata(config, settings, len(cases), len(selected_cases), args.backend)
    payload = build_run_payload(records, metadata=metadata)
    write_json(config.output_path, payload)
    print(console_summary(records))

    if markdown_path:
        markdown_path.write_text(render_markdown(records), encoding="utf-8")

    return determine_exit_code(records)


if __name__ == "__main__":
    raise SystemExit(main())
