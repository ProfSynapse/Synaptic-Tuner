"""Minimal interactive CLI for LM Studio full-coverage evaluations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Union

from .cli_utils import (
    build_metadata,
    build_settings_kwargs,
    color,
    count_behavior_patterns,
    count_prompts,
    determine_exit_code,
    model_output_paths,
    passfail_color,
    print_banner,
    print_record_progress,
    prompt_run_count,
    select_model,
)
from .config import EvaluatorConfig, LMStudioSettings, PromptFilter, expand_path
from .lmstudio_client import LMStudioClient
from .prompt_sets import load_prompt_cases
from .reporting import build_run_payload, console_summary, render_markdown, write_json
from .runner import evaluate_cases

# Default paths
DEFAULT_PROMPT_SET = Path(__file__).resolve().parent / "prompts" / "full_coverage.json"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
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
    parser.add_argument(
        "--inject-context",
        action="store_true",
        help="Inject system prompts with session/workspace IDs and validate model uses them correctly",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    """Main entry point for interactive CLI."""
    args = parse_args(argv or sys.argv[1:])

    print_banner("LM Studio Evaluator", "Interactive CLI")

    # Select prompt set interactively if not provided via CLI
    if args.prompt_set == str(DEFAULT_PROMPT_SET):
        prompt_set_selection = _select_prompt_set()
    else:
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

    # Get settings kwargs
    settings_kwargs = build_settings_kwargs(args)

    # Create client for model listing
    list_client = LMStudioClient(
        settings=LMStudioSettings(model="__list__", **settings_kwargs),
        timeout=args.timeout,
        retries=args.retries,
    )

    # Select model
    model_name = args.model or select_model(list_client)
    if not model_name:
        return 1

    run_count = args.runs if args.runs and args.runs > 0 else prompt_run_count()

    settings = LMStudioSettings(
        model=model_name,
        temperature=0.2,
        top_p=0.9,
        max_tokens=1024,
        seed=None,
        **settings_kwargs,
    )
    client = LMStudioClient(settings=settings, timeout=args.timeout, retries=args.retries)

    all_records = []

    # Run each test suite
    for suite_idx, prompt_path in enumerate(prompt_set_paths, 1):
        try:
            base_config = EvaluatorConfig(
                prompts_path=prompt_path,
                output_path=None,
                save_markdown=True,
                filter=PromptFilter(),
                retries=args.retries,
                request_timeout=args.timeout,
                dry_run=args.dry_run,
            )
            base_config.validate()
        except Exception as exc:
            print(f"Invalid configuration: {exc}", file=sys.stderr)
            return 1

        cases = load_prompt_cases(base_config.prompts_path)
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
            # Generate unique output paths
            suffix = f"_{prompt_path.stem}" if run_all_suites else ""
            json_path, md_path = model_output_paths(
                model_name, results_dir, idx, run_count, suffix=suffix
            )

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
                on_record=print_record_progress,
                inject_context=args.inject_context,
            )
            all_records.extend(records)

            metadata = build_metadata(config, settings, len(cases), len(cases), backend="lmstudio")
            payload = build_run_payload(records, metadata=metadata)
            write_json(config.output_path, payload)
            md_path.write_text(render_markdown(records), encoding="utf-8")

            print(color(console_summary(records), passfail_color(records)))
            print(color(f"JSON: {json_path}", "yellow"))
            print(color(f"Markdown: {md_path}\n", "yellow"))

    return determine_exit_code(all_records)


def _select_prompt_set() -> Union[str, List[str]]:
    """Let user choose which test suite to run.

    Returns path string or list of paths for 'all'.
    """
    paths = {
        "1": "Evaluator/prompts/behavior_rubric.json",
        "2": "Evaluator/prompts/full_coverage.json",
        "3": "Evaluator/prompts/baseline.json",
        "4": "Evaluator/prompts/tool_combos.json",
    }

    # Dynamically count prompts
    counts = {k: count_prompts(v) for k, v in paths.items()}
    behavior_pattern_count = count_behavior_patterns(paths["1"])
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
                return [
                    prompt_sets["1"]["path"],
                    prompt_sets["2"]["path"],
                    prompt_sets["3"]["path"],
                    prompt_sets["4"]["path"],
                ]
            return selected["path"]
        print("Please enter a valid option (1-5).", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
