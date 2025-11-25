"""Minimal interactive CLI for LM Studio full-coverage evaluations."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

from .cli import build_metadata
from .config import EvaluatorConfig, LMStudioSettings, PromptFilter, expand_path
from .lmstudio_client import LMStudioClient, LMStudioError
from .lmstudio_cli import DEFAULT_PROMPT_SET, DEFAULT_RESULTS_DIR, default_output_paths
from .prompt_sets import load_prompt_cases
from .reporting import build_run_payload, console_summary, render_markdown, write_json
from .runner import evaluate_cases


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive LM Studio evaluator (full coverage).",
        epilog="Just run `python evaluator` to pick a model and how many runs to execute.",
    )
    parser.add_argument("--host", help="LM Studio host (defaults to LMSTUDIO_HOST or 127.0.0.1)")
    parser.add_argument("--port", type=int, help="LM Studio port (defaults to LMSTUDIO_PORT or 1234)")
    parser.add_argument("--model", help="Optional model ID to skip the selection prompt.")
    parser.add_argument("--runs", type=int, help="How many times to run the suite (default: ask interactively).")
    parser.add_argument("--prompt-set", default=str(DEFAULT_PROMPT_SET), help="Prompt set to use (default: full_coverage.json).")
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR), help="Directory for JSON/MD artifacts.")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds.")
    parser.add_argument("--retries", type=int, default=2, help="Retry attempts for LM Studio calls.")
    parser.add_argument("--dry-run", action="store_true", help="Skip backend calls (schema validation only).")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    _print_banner()

    # Select prompt set interactively if not provided via CLI
    if args.prompt_set == str(DEFAULT_PROMPT_SET):
        # User didn't specify, let them choose
        prompt_set_selection = _select_prompt_set()
    else:
        # User specified via --prompt-set
        prompt_set_selection = args.prompt_set

    # Handle "Run All" option
    if isinstance(prompt_set_selection, list):
        prompt_set_paths = [expand_path(p) for p in prompt_set_selection]
        run_all_suites = True
    else:
        prompt_set_paths = [expand_path(prompt_set_selection)]
        run_all_suites = False

    results_dir = expand_path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Validate all prompt paths exist
    for prompt_path in prompt_set_paths:
        if not prompt_path.exists():
            print(f"Prompt set not found: {prompt_path}", file=sys.stderr)
            return 1

    list_client = LMStudioClient(
        settings=LMStudioSettings(model="__list__", **_settings_kwargs(args)),
        timeout=args.timeout,
        retries=args.retries,
    )
    model_name = args.model or _select_model(list_client)
    if not model_name:
        return 1

    run_count = args.runs if args.runs and args.runs > 0 else _prompt_run_count()

    settings = LMStudioSettings(
        model=model_name,
        temperature=0.2,
        top_p=0.9,
        max_tokens=1024,
        seed=None,
        **_settings_kwargs(args),
    )
    client = LMStudioClient(settings=settings, timeout=args.timeout, retries=args.retries)

    any_errors = False
    any_failures = False

    # Run each test suite
    for suite_idx, prompt_path in enumerate(prompt_set_paths, 1):
        try:
            base_config = EvaluatorConfig(
                prompts_path=prompt_path,
                output_path=None,  # Set per run
                save_markdown=True,
                filter=PromptFilter(),
                retries=args.retries,
                request_timeout=args.timeout,
                dry_run=args.dry_run,
            )
            base_config.validate()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"Invalid configuration: {exc}", file=sys.stderr)
            return 1

        cases = load_prompt_cases(base_config.prompts_path)

        # Get friendly name for prompt set
        prompt_set_name = prompt_path.stem.replace('_', ' ').title()

        if run_all_suites:
            print(color(f"\n{'='*60}", "cyan"))
            print(color(f"Test Suite {suite_idx}/{len(prompt_set_paths)}: {prompt_set_name}", "cyan"))
            print(color(f"{'='*60}", "cyan"))

        print(color(f"\nRunning evaluation for: {model_name}", "cyan"))
        print(color(f"Test suite: {prompt_set_name} ({len(cases)} prompts)", "cyan"))
        print(color(f"Prompt file: {prompt_path}", "cyan"))
        print(color(f"Runs: {run_count}\n", "cyan"))

        for idx in range(run_count):
            # Generate unique output paths for each suite
            if run_all_suites:
                suite_suffix = f"_{prompt_path.stem}"
                json_path, md_path = run_output_paths_with_suffix(model_name, results_dir, idx, run_count, suite_suffix)
            else:
                json_path, md_path = run_output_paths(model_name, results_dir, idx, run_count)

            config = EvaluatorConfig(
                prompts_path=base_config.prompts_path,
                output_path=json_path,
                save_markdown=True,
                filter=base_config.filter,
                retries=base_config.retries,
                request_timeout=base_config.request_timeout,
                dry_run=base_config.dry_run,
            )
            config.ensure_output_parent()
            md_path.parent.mkdir(parents=True, exist_ok=True)

            print(color(f"--- Run {idx + 1}/{run_count} ---", "magenta"))
            records = evaluate_cases(
                cases,
                client=client,
                dry_run=config.dry_run,
                on_record=_print_record_progress,
            )

            metadata = build_metadata(config, settings, len(cases), len(cases), backend="lmstudio")
            payload = build_run_payload(records, metadata=metadata)
            write_json(config.output_path, payload)
            md_path.write_text(render_markdown(records), encoding="utf-8")

            print(color(console_summary(records), _passfail_color(records)))
            print(color(f"JSON: {json_path}", "yellow"))
            print(color(f"Markdown: {md_path}\n", "yellow"))

            if any(record.error for record in records):
                any_errors = True
            # Use record.passed which includes both schema AND behavior validation
            if any(not record.passed for record in records if record.error is None):
                any_failures = True

    if any_errors:
        return 3
    if any_failures:
        return 2
    return 0


def run_output_paths(model_name: str, results_dir: Path, run_index: int, total_runs: int) -> Tuple[Path, Path]:
    """Create per-run JSON/MD paths, suffixed when multiple runs are requested."""
    json_path, md_path = default_output_paths(model_name, results_dir)
    if total_runs > 1:
        json_path = json_path.parent / f"{json_path.stem}_run{run_index + 1}{json_path.suffix}"
        md_path = md_path.parent / f"{md_path.stem}_run{run_index + 1}{md_path.suffix}"
    return json_path, md_path


def run_output_paths_with_suffix(model_name: str, results_dir: Path, run_index: int, total_runs: int, suffix: str) -> Tuple[Path, Path]:
    """Create per-run JSON/MD paths with custom suffix for multi-suite runs."""
    json_path, md_path = default_output_paths(model_name, results_dir)
    # Insert suffix before timestamp
    stem_parts = json_path.stem.rsplit('_', 1)  # Split off timestamp
    new_stem = f"{stem_parts[0]}{suffix}_{stem_parts[1]}"
    json_path = json_path.parent / f"{new_stem}{json_path.suffix}"

    stem_parts = md_path.stem.rsplit('_', 1)
    new_stem = f"{stem_parts[0]}{suffix}_{stem_parts[1]}"
    md_path = md_path.parent / f"{new_stem}{md_path.suffix}"

    if total_runs > 1:
        json_path = json_path.parent / f"{json_path.stem}_run{run_index + 1}{json_path.suffix}"
        md_path = md_path.parent / f"{md_path.stem}_run{run_index + 1}{md_path.suffix}"
    return json_path, md_path


def _select_model(client: LMStudioClient) -> str | None:
    try:
        models = client.list_models()
    except LMStudioError as exc:
        print(f"Unable to list models from LM Studio: {exc}", file=sys.stderr)
        return None

    if not models:
        print("LM Studio did not return any models.", file=sys.stderr)
        return None

    if len(models) == 1:
        print(f"Using only available model: {models[0]}")
        return models[0]

    print(color("Select a model to evaluate:", "magenta"))
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


def _count_prompts(path_str: str) -> int:
    """Dynamically count prompts in a JSON file."""
    try:
        path = expand_path(path_str)
        if path.exists():
            cases = load_prompt_cases(path)
            return len(cases)
    except Exception:
        pass
    return 0


def _count_behavior_patterns(path_str: str) -> int:
    """Count unique behavior patterns in a prompt set by examining tags."""
    try:
        path = expand_path(path_str)
        if path.exists():
            cases = load_prompt_cases(path)
            # Extract unique behavior tags (excluding generic tags)
            behavior_tags = set()
            behavior_prefixes = {
                "intellectual_humility", "verification_before_action", "context_continuity",
                "strategic_tool_selection", "error_recovery", "workspace_awareness",
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


def _select_prompt_set() -> str | list:
    """Let user choose which test suite to run. Returns path string or list of paths for 'all'."""
    # Define paths first
    paths = {
        "1": "Evaluator/prompts/behavior_rubric.json",
        "2": "Evaluator/prompts/full_coverage.json",
        "3": "Evaluator/prompts/baseline.json",
        "4": "Evaluator/prompts/tool_combos.json",
    }

    # Dynamically count prompts
    counts = {k: _count_prompts(v) for k, v in paths.items()}
    behavior_pattern_count = _count_behavior_patterns(paths["1"])
    total_count = sum(counts.values())

    prompt_sets = {
        "1": {
            "name": "Behavior Rubric Tests",
            "path": paths["1"],
            "desc": f"{counts['1']} prompts testing {behavior_pattern_count} behavior patterns (Recommended)",
        },
        "2": {
            "name": "Full Tool Coverage",
            "path": paths["2"],
            "desc": f"{counts['2']} prompts - one test per tool",
        },
        "3": {
            "name": "Baseline Tests",
            "path": paths["3"],
            "desc": f"{counts['3']} general prompts with behavior expectations",
        },
        "4": {
            "name": "Multi-Step Workflows",
            "path": paths["4"],
            "desc": f"{counts['4']} prompts testing complex tool sequences",
        },
        "5": {
            "name": "Run All Tests",
            "path": "ALL",
            "desc": f"Run all 4 test suites sequentially ({total_count} total prompts)",
        },
    }

    print(color("\nSelect test suite:", "magenta"))
    for key in sorted(prompt_sets.keys()):
        pset = prompt_sets[key]
        print(f"{color(f'[{key}]', 'yellow')} {pset['name']}")
        print(f"     {color(pset['desc'], 'cyan')}")

    while True:
        choice = input("\nEnter a number (default 1): ").strip()
        if not choice:
            return prompt_sets["1"]["path"]
        if choice in prompt_sets:
            selected = prompt_sets[choice]
            print(color(f"Selected: {selected['name']}", "green"))
            if selected["path"] == "ALL":
                # Return all test suites
                return [
                    prompt_sets["1"]["path"],
                    prompt_sets["2"]["path"],
                    prompt_sets["3"]["path"],
                    prompt_sets["4"]["path"],
                ]
            return selected["path"]
        print("Please enter a valid option (1-5).", file=sys.stderr)


def _prompt_run_count(default: int = 1) -> int:
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


def _settings_kwargs(args: argparse.Namespace) -> dict:
    opts = {}
    if args.host:
        opts["host"] = args.host
    if args.port is not None:
        opts["port"] = args.port
    return opts


def _print_record_progress(record) -> None:
    """Emit a one-line status for each prompt as it completes."""
    status, color_name = _record_status(record)
    label = record.case.case_id
    suffix = ""
    if record.error:
        suffix = f" ({record.error})"
    elif record.validator and record.validator.issues and not record.validator.passed:
        suffix = f" ({len(record.validator.issues)} schema issue(s))"
    elif record.behavior and record.behavior.issues and not record.behavior.passed:
        # Show behavior validation failures
        failed_issues = [i for i in record.behavior.issues if not i.passed]
        suffix = f" ({len(failed_issues)} behavior issue(s))"
    print(f"{color(f'[{status}]', color_name)} {label}{suffix}")


def _record_status(record) -> tuple[str, str]:
    if record.error:
        return "ERROR", "red"
    # Use record.passed which includes both schema AND behavior validation
    if not record.passed:
        return "FAIL", "red"
    if record.passed:
        return "PASS", "green"
    # Dry-run or missing validator
    return "SKIP", "yellow"


# -- Styling helpers ------------------------------------------------------- #

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
    if os.getenv("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def color(text: str, name: str) -> str:
    if not supports_ansi():
        return text
    start = ANSI_CODES.get(name, "")
    end = ANSI_CODES["reset"] if start else ""
    return f"{start}{text}{end}"


def _print_banner() -> None:
    width = 38
    top = "+" + "=" * (width - 2) + "+"
    mid1 = "|" + "LM Studio Evaluator".center(width - 2) + "|"
    mid2 = "|" + "Interactive CLI".center(width - 2) + "|"

    lines = [top, mid1, mid2, top]
    if supports_ansi():
        lines = [color(line, "magenta") for line in lines]
    print("\n".join(lines) + "\n")


def _passfail_color(records) -> str:
    any_errors = any(r.error for r in records)
    # Use r.passed which includes both schema AND behavior validation
    any_fail = any(not r.passed for r in records if r.error is None)
    if any_errors or any_fail:
        return "red"
    return "green"


if __name__ == "__main__":
    raise SystemExit(main())
