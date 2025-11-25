"""Command-line entry point for the Evaluator."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union

from .config import (
    BackendType,
    EvaluatorConfig,
    LMStudioSettings,
    OllamaSettings,
    PromptFilter,
    expand_path,
    parse_tags,
)
from .lmstudio_client import LMStudioClient
from .ollama_client import OllamaClient
from .prompt_sets import filter_prompts, load_prompt_cases
from .reporting import build_run_payload, console_summary, render_markdown, write_json
from .runner import evaluate_cases


def parse_args(argv: list[str]) -> argparse.Namespace:
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    prompt_path = expand_path(args.prompt_set)
    output_path = expand_path(args.output) if args.output else default_output_path()
    markdown_path = expand_path(args.markdown) if args.markdown else None

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

    cases = load_prompt_cases(config.prompts_path)
    selected_cases = filter_prompts(cases, config.filter)
    if not selected_cases:
        print("No prompts matched the provided filters.", file=sys.stderr)
        return 1

    # Build settings kwargs from CLI overrides
    settings_kwargs = {}
    if args.host:
        settings_kwargs["host"] = args.host
    if args.port:
        settings_kwargs["port"] = args.port

    # Create backend-specific client
    if args.backend == "lmstudio":
        settings = LMStudioSettings(
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=args.seed,
            **settings_kwargs,
        )
        client = LMStudioClient(settings=settings, timeout=config.request_timeout, retries=config.retries)
    else:  # ollama
        settings = OllamaSettings(
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            seed=args.seed,
            **settings_kwargs,
        )
        client = OllamaClient(settings=settings, timeout=config.request_timeout, retries=config.retries)
    records = evaluate_cases(selected_cases, client=client, dry_run=config.dry_run)

    metadata = build_metadata(config, settings, len(cases), len(selected_cases), args.backend)
    payload = build_run_payload(records, metadata=metadata)
    write_json(config.output_path, payload)
    print(console_summary(records))

    if markdown_path:
        markdown_path.write_text(render_markdown(records), encoding="utf-8")

    # Exit code 0 if everything passed, 2 if any failures, 3 if request errors.
    any_errors = any(record.error for record in records)
    any_failures = any(not record.passed for record in records if record.error is None)
    if any_errors:
        return 3
    if any_failures:
        return 2
    return 0


def default_output_path() -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return Path(f"Evaluator/results/run_{timestamp}.json")


def build_metadata(
    config: EvaluatorConfig,
    settings: Union[OllamaSettings, LMStudioSettings],
    total_prompts: int,
    selected_prompts: int,
    backend: str,
) -> Dict[str, Any]:
    return {
        "backend": backend,
        "model": settings.model,
        "host": settings.host,
        "port": settings.port,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "max_tokens": settings.max_tokens,
        "seed": settings.seed,
        "prompt_file": str(config.prompts_path),
        "prompt_total": total_prompts,
        "prompt_selected": selected_prompts,
        "request_timeout": config.request_timeout,
        "retries": config.retries,
        "dry_run": config.dry_run,
        "tags_filter": list(config.filter.tags),
        "limit": config.filter.limit,
    }


if __name__ == "__main__":
    raise SystemExit(main())
