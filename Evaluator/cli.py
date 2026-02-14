"""Command-line entry point for the Evaluator.

This module provides the main CLI for running model evaluations
against Ollama or LM Studio backends.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Force UTF-8 output for Windows to handle unicode characters like ✓
if sys.platform == "win32":
    import io
    # Check if stdout/stderr are attached to a terminal or file (have buffer)
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Rich console for colored output (optional)
try:
    from rich.console import Console
    from rich.text import Text
    _console = Console()
    _RICH_AVAILABLE = True
except ImportError:
    _console = None
    _RICH_AVAILABLE = False

# Import live dashboard and UI components
try:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent))
    from shared.ui import LiveEvaluationDashboard, RICH_AVAILABLE as _SHARED_RICH
    from .ui import rich_summary, rich_failure_details, print_evaluation_header
    _DASHBOARD_AVAILABLE = True
except ImportError:
    _DASHBOARD_AVAILABLE = False
    _SHARED_RICH = False

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
from .config_loader import load_yaml_scenarios
from .reporting import (
    build_run_payload,
    console_summary,
    render_markdown,
    write_json,
    build_evaluation_lineage,
    generate_evaluation_model_card_section,
)
from .runner import evaluate_cases


def load_display_config(config_dir: Path) -> Dict[str, Any]:
    """Load display configuration from YAML."""
    display_path = config_dir / "display.yaml"
    if display_path.exists():
        with open(display_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def simplify_issue_message(msg: str, display_config: Dict[str, Any]) -> str | None:
    """Apply configured simplifications to issue messages.

    Returns None if message should be skipped entirely.
    """
    # Check skip list
    for skip_pattern in display_config.get("skip_messages", []):
        if skip_pattern in msg:
            return None

    # Apply simplifications
    for pattern, replacement in display_config.get("simplify_messages", {}).items():
        if pattern in msg:
            if replacement is None:
                # Truncate at pattern
                return msg.split(pattern)[0].strip()
            else:
                # Replace with configured message
                return replacement

    return msg


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate tool-calling models via Ollama, LM Studio, or llama.cpp.",
        epilog="""
Backend Configuration:
  Ollama:    OLLAMA_HOST (default: 127.0.0.1), OLLAMA_PORT (default: 11434)
  LM Studio: LMSTUDIO_HOST (default: 127.0.0.1), LMSTUDIO_PORT (default: 1234)
  llama.cpp: --model should be path to GGUF file (e.g., ./model.Q4_K_M.gguf)
        """,
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "lmstudio", "llamacpp", "unsloth", "openrouter", "mlc"],
        default=None,  # Auto-detect based on model path
        help="Backend to use for evaluation (auto-detects MLC models, default: ollama)",
    )
    parser.add_argument("--model", required=True, help="Model name (e.g., claudesidian-mcp)")
    parser.add_argument(
        "--config-dir",
        default="Evaluator/config",
        help="Path to YAML config directory (default: Evaluator/config)",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        help="YAML scenario file(s) to run (can specify multiple). Overrides --prompt-set",
    )
    parser.add_argument(
        "--preset",
        help="Preset from eval_run.yaml (e.g., 'quick', 'full', 'behavior_only')",
    )
    parser.add_argument("--tags", help="Comma-separated tag filter")
    parser.add_argument("--limit", type=int, help="Max prompts to evaluate")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, help="Optional generation seed")
    parser.add_argument("--host", help="Override backend host (OLLAMA_HOST or LMSTUDIO_HOST)")
    parser.add_argument("--port", type=int, help="Override backend port (OLLAMA_PORT or LMSTUDIO_PORT)")
    parser.add_argument("--mlc-port", type=int, default=8000, help="Port for MLC/WebLLM HTTP server (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser for MLC evaluation")
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
    parser.add_argument(
        "--env-backend",
        choices=["none", "local", "e2b"],
        default="none",
        help="Enable environment-backed tool execution checks (default: none)",
    )
    parser.add_argument(
        "--env-template",
        help="E2B template ID when using --env-backend e2b",
    )
    parser.add_argument(
        "--env-timeout",
        type=float,
        default=120.0,
        help="Environment command timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--env-api-key",
        help="E2B API key override (default: E2B_API_KEY env var)",
    )
    parser.add_argument(
        "--env-tool-schema",
        help="Path to tool schema YAML for environment execution (default: Evaluator/config/tool_schema.yaml)",
    )
    parser.add_argument(
        "--env-exec-config",
        help="Path to environment execution rules YAML (default: Evaluator/config/environment_execution.yaml)",
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
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Disable live dashboard, use simple text output",
    )
    return parser.parse_args(argv)


def _is_mlc_model(model_path: str) -> bool:
    """Check if the model path points to an MLC model."""
    path = Path(model_path)

    # Check for MLC indicators in the path
    if "-MLC" in str(path) or "webgpu" in str(path).lower():
        return True

    # Check if path is a directory with mlc-chat-config.json
    if path.is_dir():
        if (path / "mlc-chat-config.json").exists():
            return True
        # Check subdirectories
        for subdir in path.iterdir() if path.exists() else []:
            if subdir.is_dir() and (subdir / "mlc-chat-config.json").exists():
                return True

    return False


def main(argv: List[str] | None = None) -> int:
    """Main entry point for CLI evaluation."""
    args = parse_args(argv or sys.argv[1:])

    # Auto-detect MLC models if backend not specified
    if args.backend is None:
        if _is_mlc_model(args.model):
            args.backend = "mlc"
        else:
            args.backend = "ollama"  # Default

    # Handle MLC backend specially - requires browser-based evaluation
    if args.backend == "mlc":
        from .mlc_eval_handler import run_mlc_evaluation
        config_dir = expand_path(args.config_dir)
        return run_mlc_evaluation(
            model_path=args.model,
            config_dir=config_dir,
            port=args.mlc_port,
            open_browser=not args.no_browser,
        )

    # Resolve paths
    output_path = expand_path(args.output) if args.output else default_output_path()
    markdown_path = expand_path(args.markdown) if args.markdown else None
    config_dir = expand_path(args.config_dir)

    # Require --scenario or --preset
    if not args.scenarios and not args.preset:
        print("Error: --scenario is required. Example: --scenario behavior_prompts.yaml", file=sys.stderr)
        return 1

    # Load from YAML config system
    tag_filter = parse_tags(args.tags) if args.tags else None
    selected_cases = load_yaml_scenarios(
        config_dir=config_dir,
        scenario_files=args.scenarios,
        preset=args.preset,
        tag_filter=tag_filter,
    )

    if args.limit and len(selected_cases) > args.limit:
        selected_cases = selected_cases[:args.limit]

    # Build config for output paths
    config = EvaluatorConfig(
        prompts_path=config_dir / "scenarios",
        output_path=output_path,
        save_markdown=bool(markdown_path),
        filter=PromptFilter(tags=tag_filter, limit=args.limit),
        retries=args.retries,
        request_timeout=args.timeout,
        dry_run=args.dry_run,
    )
    prompt_path = config_dir / "scenarios"
    total_cases = len(selected_cases)

    config.ensure_output_parent()
    if markdown_path:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
    if not selected_cases:
        print("No prompts matched the provided filters.", file=sys.stderr)
        return 1

    # Load display configuration
    display_config = load_display_config(config_dir)
    labels = display_config.get("labels", {})
    colors = display_config.get("colors", {})

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

    # Determine if we should use the live dashboard
    use_dashboard = (
        _DASHBOARD_AVAILABLE
        and _SHARED_RICH
        and not args.no_dashboard
        and not args.dry_run  # Don't use dashboard for dry runs
    )

    # Dashboard-based callback for live updates
    dashboard = None
    if use_dashboard:
        dashboard = LiveEvaluationDashboard(
            title="Model Evaluation",
            total_tests=len(selected_cases),
            log_lines=5,
        )

    environment_validator = None
    if args.env_backend != "none":
        try:
            from shared.environments import EnvironmentValidator
        except ImportError as exc:
            print(f"Environment validation is unavailable: {exc}", file=sys.stderr)
            return 1

        environment_validator = EnvironmentValidator(
            backend=args.env_backend,
            e2b_template=args.env_template,
            e2b_api_key=args.env_api_key,
            timeout_seconds=args.env_timeout,
            tool_schema_path=args.env_tool_schema,
            execution_config_path=args.env_exec_config,
        )

    def on_record_dashboard(record):
        """Update dashboard with evaluation result."""
        name = record.case.case_id or "unnamed"
        latency = record.latency_s or 0.0

        # Get brief failure reason for log display
        reason = None
        if record.status in ("fail", "warn"):
            if record.error:
                reason = f"Error: {record.error[:40]}..."
            elif record.validator and record.validator.issues:
                for issue in record.validator.issues:
                    msg = simplify_issue_message(issue.message, display_config)
                    if msg:
                        reason = msg[:50] + "..." if len(msg) > 50 else msg
                        break
            elif record.environment and record.environment.issues:
                for issue in record.environment.issues:
                    msg = simplify_issue_message(issue.message, display_config)
                    if msg:
                        reason = msg[:50] + "..." if len(msg) > 50 else msg
                        break
            elif record.behavior and not record.behavior.passed:
                for issue in record.behavior.issues:
                    msg = simplify_issue_message(issue.message, display_config)
                    if msg:
                        reason = msg[:50] + "..." if len(msg) > 50 else msg
                        break

        # Check behavior results
        behavior_tested = record.behavior is not None
        behavior_passed = behavior_tested and record.behavior.passed

        dashboard.update(
            status=record.status,
            name=name,
            latency=latency,
            reason=reason,
            behavior_tested=behavior_tested,
            behavior_passed=behavior_passed,
        )

    # Fallback text-based callback (original implementation)
    def on_record_text(record):
        """Simple text output for each record."""
        name = record.case.case_id or "unnamed"
        latency = f"{record.latency_s:.2f}s" if record.latency_s else "-"

        # Get labels from config with defaults
        lbl_called = labels.get("model_called", "Model called")
        lbl_expected = labels.get("expected", "Expected")
        lbl_why = labels.get("why", "Why")
        lbl_no_tool = labels.get("no_tool_call", "(text response)")

        status = record.status
        if status == "pass":
            status_str = "✓ PASS"
        elif status == "warn":
            status_str = "⚠ WARN"
        else:
            status_str = "✗ FAIL"

        print(f"  {status_str}  {name} ({latency})")

        if status in ("fail", "warn"):
            if record.error:
                print(f"         Error: {record.error}")
            elif record.validator:
                if record.validator.tool_calls:
                    called = [tc.name for tc in record.validator.tool_calls]
                    print(f"         {lbl_called}: {', '.join(called)}")
                else:
                    print(f"         {lbl_called}: {lbl_no_tool}")

                if status == "fail":
                    expected = record.case.expected_tools or record.case.acceptable_tools
                    if expected:
                        print(f"         {lbl_expected}: {', '.join(expected)}")

                if record.validator.issues:
                    for issue in record.validator.issues:
                        msg = simplify_issue_message(issue.message, display_config)
                        if msg:
                            print(f"         {lbl_why}: {msg}")
                            break
                elif record.environment and record.environment.issues:
                    for issue in record.environment.issues:
                        msg = simplify_issue_message(issue.message, display_config)
                        if msg:
                            print(f"         {lbl_why}: {msg}")
                            break

    # Run evaluation with appropriate callback
    if use_dashboard:
        with dashboard:
            records = evaluate_cases(
                selected_cases,
                client=client,
                dry_run=config.dry_run,
                validate_context=args.validate_context,
                environment_validator=environment_validator,
                on_record=on_record_dashboard,
            )
    else:
        print(f"\nRunning {len(selected_cases)} evaluations...\n")
        records = evaluate_cases(
            selected_cases,
            client=client,
            dry_run=config.dry_run,
            validate_context=args.validate_context,
            environment_validator=environment_validator,
            on_record=on_record_text,
        )

    print()  # Blank line before summary

    # Build and save results
    metadata = build_metadata(config, settings, total_cases, len(selected_cases), args.backend)
    if environment_validator is not None:
        metadata["environment"] = {
            "backend": args.env_backend,
            "template": args.env_template,
            "timeout_seconds": args.env_timeout,
            "tool_schema_path": args.env_tool_schema,
            "execution_config_path": args.env_exec_config,
        }
    payload = build_run_payload(records, metadata=metadata)
    write_json(config.output_path, payload)
    print(f"Results saved to {config.output_path}")
    if markdown_path:
        print(f"Markdown summary saved to {markdown_path}")

    # Display summary using rich UI if available
    if _DASHBOARD_AVAILABLE and _SHARED_RICH:
        rich_summary(records)
        # Show detailed failure info
        failed_count = sum(1 for r in records if not r.passed)
        if failed_count > 0:
            rich_failure_details(records, max_display=10)
    else:
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
