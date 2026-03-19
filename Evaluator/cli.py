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
    EvalJudgeConfig,
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
from shared.cloud_eval_progress import CloudEvaluationProgressWriter, extract_record_progress


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
        description="Evaluate tool-calling models via Ollama, LM Studio, vLLM, or llama.cpp.",
        epilog="""
Backend Configuration:
  Ollama:    OLLAMA_HOST (default: 127.0.0.1), OLLAMA_PORT (default: 11434)
  LM Studio: LMSTUDIO_HOST (default: 127.0.0.1), LMSTUDIO_PORT (default: 1234)
  llama.cpp: --model should be path to GGUF file (e.g., ./model.Q4_K_M.gguf)
        """,
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "lmstudio", "vllm", "llamacpp", "unsloth", "openrouter", "mlc"],
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

    # Judge (LLM-as-judge evaluation)
    judge_group = parser.add_argument_group("Judge (LLM-as-judge evaluation)")
    judge_group.add_argument(
        "--judge",
        action="store_true",
        help="Enable LLM-as-judge evaluation alongside pattern matching",
    )
    judge_group.add_argument(
        "--judge-mode",
        choices=["and", "or", "judge_only"],
        default="and",
        help="How to combine judge and pattern-match results (default: and)",
    )
    judge_group.add_argument(
        "--judge-provider",
        choices=["openrouter", "lmstudio", "ollama"],
        help="LLM provider for judge (default: same as eval backend)",
    )
    judge_group.add_argument(
        "--judge-model",
        help="Model for judge calls (default: same as eval model)",
    )
    judge_group.add_argument(
        "--judge-rubrics",
        help="Comma-separated rubric names (e.g., tool_call_quality,response_appropriateness)",
    )
    judge_group.add_argument(
        "--judge-rubrics-dir",
        default="Evaluator/config/rubrics",
        help="Path to rubric YAML files (default: Evaluator/config/rubrics/)",
    )
    judge_group.add_argument(
        "--no-judge-log",
        action="store_true",
        help="Disable KTO interaction logging for judge calls",
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
    parser.add_argument("--progress-jsonl", help=argparse.SUPPRESS)
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

    def format_issue_for_display(message: str) -> str | None:
        return simplify_issue_message(message, display_config)

    progress_writer = None
    if args.progress_jsonl:
        progress_writer = CloudEvaluationProgressWriter(
            Path(args.progress_jsonl),
            sync_every_events=5,
        )
        progress_writer.write_metadata(
            total_tests=len(selected_cases),
            title="Cloud Evaluation",
            backend=args.backend,
            model=args.model,
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

    # Initialize judge validator if --judge enabled
    judge_validator = None
    judge_cfg = None
    if args.judge:
        try:
            from shared.llm import LLMConfig, create_client as create_llm_client
            from shared.judge import JudgeConfig, RubricLoader, InteractionLogger
            from .judge_validator import JudgeValidator

            # Populate EvalJudgeConfig from CLI args (single source of truth)
            judge_cfg = EvalJudgeConfig(
                enabled=True,
                mode=args.judge_mode,
                provider=args.judge_provider,
                model=args.judge_model,
                rubrics=parse_tags(args.judge_rubrics) if args.judge_rubrics else [],
                rubrics_dir=args.judge_rubrics_dir,
                temperature=0.3,
                max_tokens=2048,
                log_interactions=not args.no_judge_log,
            )

            # Create judge LLM client (can differ from eval backend)
            judge_llm_config = LLMConfig.from_env(env_prefix="JUDGE")
            if judge_cfg.provider:
                judge_llm_config.provider = judge_cfg.provider
            if judge_cfg.model:
                judge_llm_config.model = judge_cfg.model
            judge_llm_client = create_llm_client(config=judge_llm_config)

            # Load rubrics (M4: validate rubrics_dir is sensible)
            rubrics_dir = expand_path(judge_cfg.rubrics_dir)
            project_root = Path(__file__).parent.parent.resolve()
            if not rubrics_dir.is_relative_to(project_root):
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "Judge rubrics directory '%s' is outside the project root '%s'",
                    rubrics_dir,
                    project_root,
                )
            loader = RubricLoader(rubrics_dir)
            rubric_keys = judge_cfg.rubrics if judge_cfg.rubrics else []
            if not rubric_keys:
                rubric_keys = loader.list_available()
            if not rubric_keys:
                print(f"No rubrics found in {rubrics_dir}", file=sys.stderr)
                return 1
            rubrics = loader.load_many(rubric_keys)

            # Interaction logger for KTO training
            interaction_logger = None
            if judge_cfg.log_interactions:
                interaction_logger = InteractionLogger(
                    output_dir=Path("Evaluator/interactions"),
                    enabled=True,
                    prefix="judge",
                )

            judge_config = JudgeConfig(
                temperature=judge_cfg.temperature,
                max_tokens=judge_cfg.max_tokens,
            )
            judge_validator = JudgeValidator(
                llm_client=judge_llm_client,
                rubrics=rubrics,
                judge_config=judge_config,
                interaction_logger=interaction_logger,
                default_judge_mode=judge_cfg.mode,
                eval_model=args.model,
                judge_model=judge_cfg.model or "(same as eval)",
            )
            print(f"Judge enabled: {len(rubrics)} rubric(s) [{', '.join(rubric_keys)}], mode={judge_cfg.mode}")

        except ImportError as exc:
            print(f"Judge dependencies unavailable: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"Judge initialization failed: {exc}", file=sys.stderr)
            return 1

    def on_record_dashboard(record):
        """Update dashboard with evaluation result."""
        dashboard.update(**extract_record_progress(record, issue_formatter=format_issue_for_display))

    def on_record_progress(record):
        """Write structured progress events for cloud replay."""
        if progress_writer is None:
            return
        progress_writer.write_record(record, issue_formatter=format_issue_for_display)

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

    callbacks = [on_record_dashboard] if use_dashboard else [on_record_text]
    if progress_writer is not None:
        callbacks.append(on_record_progress)

    def on_record(record):
        for callback in callbacks:
            callback(record)

    # Run evaluation with appropriate callback
    if use_dashboard:
        with dashboard:
            records = evaluate_cases(
                selected_cases,
                client=client,
                dry_run=config.dry_run,
                validate_context=args.validate_context,
                environment_validator=environment_validator,
                judge_validator=judge_validator,
                on_record=on_record,
            )
    else:
        print(f"\nRunning {len(selected_cases)} evaluations...\n")
        records = evaluate_cases(
            selected_cases,
            client=client,
            dry_run=config.dry_run,
            validate_context=args.validate_context,
            environment_validator=environment_validator,
            judge_validator=judge_validator,
            on_record=on_record,
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

    if progress_writer is not None:
        progress_writer.write_complete()

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
            print("\n❌ HuggingFace authentication required. Set HF_TOKEN env var or provide credentials via --hf flag.")
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
