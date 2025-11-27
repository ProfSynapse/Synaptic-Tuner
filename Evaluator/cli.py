"""Command-line entry point for the Evaluator.

This module provides the main CLI for running model evaluations
against Ollama or LM Studio backends.
"""
from __future__ import annotations

import argparse
import json
import os
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
from .reporting import (
    build_run_payload,
    console_summary,
    render_markdown,
    write_json,
    build_evaluation_lineage,
    generate_evaluation_model_card_section,
)
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

    # Lineage and HuggingFace options
    parser.add_argument(
        "--lineage",
        help="Path to save evaluation lineage JSON (enables lineage generation)",
    )
    parser.add_argument(
        "--upload-to-hf",
        metavar="REPO_ID",
        help="Upload evaluation results to HuggingFace repo (e.g., username/model-name)",
    )
    parser.add_argument(
        "--hf-token",
        help="HuggingFace write token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--update-model-card",
        action="store_true",
        help="Update the model's README.md with evaluation results (requires --upload-to-hf)",
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
        markdown_path.write_text(render_markdown(records, args.model, str(prompt_path.name)), encoding="utf-8")

    # Generate evaluation lineage if requested
    lineage = None
    model_card_section = None
    if args.lineage or args.upload_to_hf:
        eval_config = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "backend": args.backend,
        }

        lineage = build_evaluation_lineage(
            records=records,
            model_name=args.model,
            test_suites=[str(prompt_path)],
            eval_config=eval_config,
            hardware_info={"platform": sys.platform},
        )
        model_card_section = generate_evaluation_model_card_section(lineage)

        # Save lineage to file if path provided
        if args.lineage:
            lineage_path = expand_path(args.lineage)
            lineage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(lineage_path, 'w', encoding='utf-8') as f:
                json.dump(lineage, f, indent=2)
            print(f"\n✓ Evaluation lineage saved to: {lineage_path}")

    # Upload to HuggingFace if requested
    if args.upload_to_hf:
        hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
        if not hf_token:
            print("\n❌ HuggingFace token required. Provide via --hf-token or HF_TOKEN env var")
            return 1

        try:
            from huggingface_hub import HfApi, hf_hub_download
            import tempfile
            import re

            api = HfApi()
            repo_id = args.upload_to_hf

            print(f"\n📤 Uploading evaluation results to: {repo_id}")

            # Upload evaluation lineage JSON
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(lineage, f, indent=2)
                temp_lineage_path = f.name

            api.upload_file(
                path_or_fileobj=temp_lineage_path,
                path_in_repo="evaluation_lineage.json",
                repo_id=repo_id,
                token=hf_token,
            )
            print(f"  ✓ evaluation_lineage.json uploaded")

            # Update model card if requested
            if args.update_model_card:
                try:
                    readme_path = hf_hub_download(repo_id=repo_id, filename="README.md", token=hf_token)
                    with open(readme_path, 'r', encoding='utf-8') as f:
                        existing_readme = f.read()

                    if "## Evaluation Results" in existing_readme:
                        pattern = r'## Evaluation Results.*?(?=\n## |\Z)'
                        updated_readme = re.sub(pattern, model_card_section, existing_readme, flags=re.DOTALL)
                        print("  Replacing existing evaluation section...")
                    else:
                        updated_readme = existing_readme.rstrip() + "\n\n" + model_card_section
                        print("  Adding new evaluation section...")

                    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
                        f.write(updated_readme)
                        temp_readme_path = f.name

                    api.upload_file(
                        path_or_fileobj=temp_readme_path,
                        path_in_repo="README.md",
                        repo_id=repo_id,
                        token=hf_token,
                    )
                    print(f"  ✓ README.md updated with evaluation results")

                except Exception as e:
                    print(f"  ⚠️  Could not update README: {e}")

            print(f"\n✓ Evaluation results uploaded to: https://huggingface.co/{repo_id}")

        except ImportError:
            print("\n❌ huggingface_hub not installed. Run: pip install huggingface_hub")
            return 1
        except Exception as e:
            print(f"\n❌ Upload failed: {e}")
            return 1

    return determine_exit_code(records)


if __name__ == "__main__":
    raise SystemExit(main())
