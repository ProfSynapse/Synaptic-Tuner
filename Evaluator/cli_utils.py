"""Shared CLI utilities for the Evaluator module.

This module centralizes common CLI functionality to avoid duplication
across cli.py, interactive_cli.py, and lmstudio_cli.py.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .config import (
    BaseBackendSettings,
    EvaluatorConfig,
    LMStudioSettings,
    OllamaSettings,
    VLLMSettings,
    expand_path,
)
from .enums import BackendType
from .protocols import BackendError, ModelListingClient
from .prompt_sets import PromptCase, load_prompt_cases
from .runner import EvaluationRecord


# ---------------------------------------------------------------------------
# ANSI Color Support
# ---------------------------------------------------------------------------

ANSI_CODES = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "cyan": "\033[36m",
    "yellow": "\033[33m",
    "magenta": "\033[35m",
    "green": "\033[32m",
    "red": "\033[31m",
}


def supports_ansi() -> bool:
    """Check if terminal supports ANSI color codes."""
    if os.getenv("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def color(text: str, name: str) -> str:
    """Apply ANSI color to text if supported.

    Args:
        text: Text to colorize
        name: Color name (cyan, yellow, magenta, green, red, bold)

    Returns:
        Colored text if ANSI supported, otherwise original text
    """
    if not supports_ansi():
        return text
    start = ANSI_CODES.get(name, "")
    end = ANSI_CODES["reset"] if start else ""
    return f"{start}{text}{end}"


def passfail_color(records: Sequence[EvaluationRecord]) -> str:
    """Get color based on pass/fail status of records.

    Args:
        records: Evaluation records to check

    Returns:
        Color name: "red" if any failures/errors, "green" otherwise
    """
    any_errors = any(r.error for r in records)
    any_fail = any(not r.passed for r in records if r.error is None)
    return "red" if any_errors or any_fail else "green"


# ---------------------------------------------------------------------------
# Model Selection
# ---------------------------------------------------------------------------

def select_model(client: ModelListingClient, prompt_text: str = "Select a model to evaluate:") -> Optional[str]:
    """Interactively select a model from backend server.

    Args:
        client: Any client that implements ModelListingClient protocol
        prompt_text: Text to display before model list

    Returns:
        Selected model name, or None if selection failed
    """
    try:
        models = client.list_models()
    except BackendError as exc:
        print(f"Unable to list models: {exc}", file=sys.stderr)
        return None

    if not models:
        print("Server did not return any models.", file=sys.stderr)
        return None

    if len(models) == 1:
        print(f"Using only available model: {models[0]}")
        return models[0]

    print(color(prompt_text, "magenta"))
    for idx, model in enumerate(models, start=1):
        print(f"{color(f'[{idx}]', 'yellow')} {model}")

    while True:
        choice = input("Enter a number (default 1): ").strip()
        if not choice:
            return models[0]
        try:
            index = int(choice)
        except ValueError:
            print("Please enter a valid number.", file=sys.stderr)
            continue
        if 1 <= index <= len(models):
            return models[index - 1]
        print(f"Please pick a value between 1 and {len(models)}.", file=sys.stderr)


# ---------------------------------------------------------------------------
# Settings Helpers
# ---------------------------------------------------------------------------

def build_settings_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """Build kwargs dict for settings from CLI args.

    Extracts host and port overrides if provided.

    Args:
        args: Parsed CLI arguments

    Returns:
        Dict with host/port if specified
    """
    opts: Dict[str, Any] = {}
    if hasattr(args, "host") and args.host:
        opts["host"] = args.host
    if hasattr(args, "port") and args.port is not None:
        opts["port"] = args.port
    return opts


# ---------------------------------------------------------------------------
# Output Path Generation
# ---------------------------------------------------------------------------

def default_output_path() -> Path:
    """Generate default timestamped output path for JSON results."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return Path(f"Evaluator/results/run_{timestamp}.json")


def model_output_paths(
    model_name: str,
    results_dir: Path,
    run_index: int = 0,
    total_runs: int = 1,
    suffix: str = "",
) -> Tuple[Path, Path]:
    """Generate JSON and Markdown output paths for a model evaluation.

    Args:
        model_name: Model name (will be sanitized)
        results_dir: Directory for output files
        run_index: Current run number (0-indexed)
        total_runs: Total number of runs
        suffix: Optional suffix to add to filename

    Returns:
        Tuple of (json_path, markdown_path)
    """
    # Sanitize model name for filename
    cleaned_model = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in model_name
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_name = f"{cleaned_model}_full_coverage{suffix}_{stamp}"
    json_path = results_dir / f"{base_name}.json"
    md_path = results_dir / f"{base_name}.md"

    # Add run suffix if multiple runs
    if total_runs > 1:
        json_path = json_path.parent / f"{json_path.stem}_run{run_index + 1}{json_path.suffix}"
        md_path = md_path.parent / f"{md_path.stem}_run{run_index + 1}{md_path.suffix}"

    return json_path, md_path


# ---------------------------------------------------------------------------
# Metadata Building
# ---------------------------------------------------------------------------

def build_metadata(
    config: EvaluatorConfig,
    settings: Union[OllamaSettings, LMStudioSettings, VLLMSettings],
    total_prompts: int,
    selected_prompts: int,
    backend: str,
) -> Dict[str, Any]:
    """Build metadata dict for evaluation run output.

    Args:
        config: Evaluator configuration
        settings: Backend settings
        total_prompts: Total prompts in prompt set
        selected_prompts: Number of prompts after filtering
        backend: Backend name string

    Returns:
        Metadata dict for JSON output
    """
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
        "tags_filter": list(config.filter.tags) if config.filter else [],
        "limit": config.filter.limit if config.filter else None,
    }


# ---------------------------------------------------------------------------
# Exit Code Logic
# ---------------------------------------------------------------------------

def determine_exit_code(records: Sequence[EvaluationRecord]) -> int:
    """Determine CLI exit code based on evaluation results.

    Args:
        records: Evaluation records

    Returns:
        0: All passed
        2: Some failures
        3: Request errors occurred
    """
    any_errors = any(record.error for record in records)
    any_failures = any(
        not record.passed for record in records if record.error is None
    )
    if any_errors:
        return 3
    if any_failures:
        return 2
    return 0


# ---------------------------------------------------------------------------
# Progress Printing
# ---------------------------------------------------------------------------

def print_record_progress(record: EvaluationRecord) -> None:
    """Print detailed status for a completed evaluation.

    Shows pass/fail, model output (truncated), and failure reasons.

    Args:
        record: Completed evaluation record
    """
    status, color_name = _get_record_status(record)
    label = record.case.case_id

    # Print status line
    print(f"\n{color(f'[{status}]', color_name)} {color(label, 'bold')}")

    # Print model output (truncated)
    if record.response_text is not None:
        output_preview = _format_model_output(record.response_text)
        print(f"  {color('Model output:', 'cyan')}")
        for line in output_preview.split('\n'):
            print(f"    {line}")

    # Print failure reasons
    if record.error:
        print(f"  {color('Error:', 'red')} {record.error}")
    elif not record.passed:
        _print_failure_reasons(record)


def _get_record_status(record: EvaluationRecord) -> Tuple[str, str]:
    """Get status label and color for a record.

    Args:
        record: Evaluation record

    Returns:
        Tuple of (status_label, color_name)
    """
    if record.error:
        return "ERROR", "red"
    if not record.passed:
        return "FAIL", "red"
    if record.passed:
        return "PASS", "green"
    return "SKIP", "yellow"


def _format_model_output(response_text: Any, max_length: int = 500) -> str:
    """Format model output for display, truncating if necessary.

    Args:
        response_text: Model response (string or dict)
        max_length: Maximum characters to show

    Returns:
        Formatted string for display
    """
    import json

    if response_text is None:
        return color("(no output)", "yellow")

    # Convert dict to JSON string for display
    if isinstance(response_text, dict):
        try:
            text = json.dumps(response_text, indent=2)
        except (TypeError, ValueError):
            text = str(response_text)
    else:
        text = str(response_text)

    # Clean up and truncate
    text = text.strip()
    if len(text) > max_length:
        text = text[:max_length] + color(f"... ({len(text) - max_length} more chars)", "yellow")

    return text


def _print_failure_reasons(record: EvaluationRecord) -> None:
    """Print detailed failure reasons for a failed evaluation.

    Args:
        record: Failed evaluation record
    """
    # Schema validation issues
    if record.validator and record.validator.issues:
        failed_schema_issues = [i for i in record.validator.issues if i.level == "ERROR"]
        if failed_schema_issues:
            print(f"  {color('Schema issues:', 'red')}")
            for issue in failed_schema_issues:
                print(f"    - {issue.message}")

    # Behavior validation issues
    if record.behavior and record.behavior.issues:
        failed_behavior_issues = [i for i in record.behavior.issues if not i.passed]
        if failed_behavior_issues:
            print(f"  {color('Behavior issues:', 'red')}")
            for issue in failed_behavior_issues:
                print(f"    - {color(issue.check, 'yellow')}: {issue.message}")
                if issue.expected != issue.actual:
                    print(f"      Expected: {issue.expected}")
                    print(f"      Actual: {issue.actual}")


# ---------------------------------------------------------------------------
# Banner Printing
# ---------------------------------------------------------------------------

def print_banner(title: str, subtitle: str = "") -> None:
    """Print a boxed banner.

    Args:
        title: Main title text
        subtitle: Optional subtitle text
    """
    width = max(len(title), len(subtitle)) + 10
    top = "+" + "=" * (width - 2) + "+"
    mid1 = "|" + title.center(width - 2) + "|"
    mid2 = "|" + subtitle.center(width - 2) + "|" if subtitle else ""

    lines = [top, mid1]
    if mid2:
        lines.append(mid2)
    lines.append(top)

    if supports_ansi():
        lines = [color(line, "magenta") for line in lines]
    print("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Prompt Counting Utilities
# ---------------------------------------------------------------------------

def count_prompts(path_str: str) -> int:
    """Count prompts in a JSON prompt set file.

    Args:
        path_str: Path to prompt file

    Returns:
        Number of prompts, or 0 if error
    """
    try:
        path = expand_path(path_str)
        if path.exists():
            cases = load_prompt_cases(path)
            return len(cases)
    except Exception:
        pass
    return 0


def count_behavior_patterns(path_str: str) -> int:
    """Count unique behavior patterns in a prompt set.

    Args:
        path_str: Path to prompt file

    Returns:
        Number of unique behavior pattern tags, or 0 if error
    """
    try:
        path = expand_path(path_str)
        if path.exists():
            cases = load_prompt_cases(path)
            behavior_tags = set()
            behavior_prefixes = {
                "intellectual_humility", "verification_before_action", "context_continuity",
                "error_recovery", "workspace_awareness",
                "response_patterns", "context_efficiency", "execute_prompt_usage"
            }
            for case in cases:
                for tag in (case.tags or []):
                    if tag in behavior_prefixes:
                        behavior_tags.add(tag)
            return len(behavior_tags)
    except Exception:
        pass
    return 0


# ---------------------------------------------------------------------------
# Interactive Prompts
# ---------------------------------------------------------------------------

def prompt_run_count(default: int = 1) -> int:
    """Interactively prompt user for number of runs.

    Args:
        default: Default value if user presses Enter

    Returns:
        Number of runs (>= 1)
    """
    while True:
        raw = input(f"\nHow many runs? (default {default}): ").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            print("Please enter a whole number.", file=sys.stderr)
            continue
        if value > 0:
            return value
        print("Run count must be at least 1.", file=sys.stderr)
