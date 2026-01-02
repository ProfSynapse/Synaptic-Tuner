"""
Rich UI components for the Evaluator module.

Provides styled output using Rich for evaluation results, summaries, and failure analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence, Optional, Dict, Any

# Add shared to path for UI components
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check for Rich availability
try:
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.console import Console, Group
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Import shared theme if available
try:
    from shared.ui import COLORS, BOX, console
except ImportError:
    # Fallback colors
    COLORS = {
        'aqua': '#00A99D',
        'orange': '#F7931E',
        'sky': '#29ABE2',
        'cello': '#33475B',
    }
    BOX = {'star': '*'}
    console = Console() if RICH_AVAILABLE else None

# Import local modules
from .runner import EvaluationRecord
from .reporting import aggregate_stats


def rich_summary(records: Sequence[EvaluationRecord]) -> None:
    """Display a rich evaluation summary with tables and panels.

    Args:
        records: List of EvaluationRecord objects from evaluation run
    """
    if not RICH_AVAILABLE:
        # Fall back to text-based summary
        from .reporting import console_summary
        print(console_summary(records))
        return

    stats = aggregate_stats(records)

    # Overall summary panel
    summary_text = Text()
    summary_text.append(f"\nEvaluated ", style="white")
    summary_text.append(f"{stats['total']}", style=f"bold {COLORS['aqua']}")
    summary_text.append(" tests: ", style="white")
    summary_text.append(f"{stats['passed']} passed", style="bold green")
    if stats.get('warned', 0) > 0:
        summary_text.append(f", {stats['warned']} warned", style="bold yellow")
    summary_text.append(f", {stats['failed']} failed", style=f"bold {COLORS['orange']}")
    summary_text.append("\n")

    # Schema pass rate
    schema_rate = stats.get('schema_pass_rate', 0) * 100
    rate_style = COLORS['aqua'] if schema_rate >= 90 else ("yellow" if schema_rate >= 70 else COLORS['orange'])
    summary_text.append(f"Schema Pass Rate: ", style="white")
    summary_text.append(f"{schema_rate:.1f}%", style=f"bold {rate_style}")
    summary_text.append(f" ({stats['schema_passed']}/{stats['total']})", style="dim")

    # Behavior pass rate (if applicable)
    if stats.get('behavior_tested', 0) > 0:
        beh_rate = stats.get('behavior_pass_rate', 0) * 100
        beh_style = COLORS['aqua'] if beh_rate >= 90 else ("yellow" if beh_rate >= 70 else COLORS['orange'])
        summary_text.append(f"\nBehavior Pass Rate: ", style="white")
        summary_text.append(f"{beh_rate:.1f}%", style=f"bold {beh_style}")
        summary_text.append(f" ({stats['behavior_passed']}/{stats['behavior_tested']})", style="dim")

    if stats.get('request_errors', 0) > 0:
        summary_text.append(f"\n\nRequest Errors: ", style="white")
        summary_text.append(f"{stats['request_errors']}", style="bold red")

    console.print(Panel(
        summary_text,
        title=f"{BOX.get('star', '*')} Summary {BOX.get('star', '*')}",
        title_align="center",
        border_style=COLORS['aqua'],
        box=box.DOUBLE,
    ))

    # Results by tag table
    if stats.get('by_tag'):
        tag_table = Table(
            title="Results by Category",
            box=box.ROUNDED,
            border_style=COLORS['cello'],
            header_style=f"bold {COLORS['sky']}",
            show_lines=False,
        )
        tag_table.add_column("Category", style=COLORS['sky'])
        tag_table.add_column("Pass", justify="right", style="green")
        tag_table.add_column("Warn", justify="right", style="yellow")
        tag_table.add_column("Fail", justify="right", style=COLORS['orange'])
        tag_table.add_column("Total", justify="right")
        tag_table.add_column("Pass Rate", justify="right")
        if any(b.get('behavior_tested', 0) > 0 for b in stats['by_tag'].values()):
            tag_table.add_column("Behavior", justify="right")

        for tag, bucket in sorted(stats['by_tag'].items()):
            rate = bucket.get('pass_rate', 0) * 100
            rate_style = "green" if rate >= 90 else ("yellow" if rate >= 70 else COLORS['orange'])

            row = [
                tag,
                str(bucket.get('passed', 0)),
                str(bucket.get('warned', 0)),
                str(bucket.get('failed', 0)),
                str(bucket.get('total', 0)),
                Text(f"{rate:.1f}%", style=rate_style),
            ]

            # Add behavior column if applicable
            if any(b.get('behavior_tested', 0) > 0 for b in stats['by_tag'].values()):
                if bucket.get('behavior_tested', 0) > 0:
                    beh_rate = bucket.get('behavior_pass_rate', 0) * 100
                    beh_style = "green" if beh_rate >= 90 else ("yellow" if beh_rate >= 70 else COLORS['orange'])
                    row.append(Text(f"{beh_rate:.0f}%", style=beh_style))
                else:
                    row.append("-")

            tag_table.add_row(*row)

        console.print(tag_table)

    # Top failure reasons
    if stats.get('top_failure_reasons'):
        failure_lines = []
        for reason, count in stats['top_failure_reasons'][:5]:
            line = Text()
            line.append(f"  {count}× ", style=f"bold {COLORS['orange']}")
            line.append(reason, style="white")
            failure_lines.append(line)

        if failure_lines:
            console.print(Panel(
                Group(*failure_lines),
                title="Top Schema Failure Reasons",
                title_align="left",
                border_style=COLORS['orange'],
                box=box.ROUNDED,
            ))

    # Top behavior failures
    if stats.get('top_behavior_failures'):
        beh_lines = []
        for reason, count in stats['top_behavior_failures'][:5]:
            line = Text()
            line.append(f"  {count}× ", style="bold yellow")
            line.append(reason, style="white")
            beh_lines.append(line)

        if beh_lines:
            console.print(Panel(
                Group(*beh_lines),
                title="Top Behavior Failures",
                title_align="left",
                border_style="yellow",
                box=box.ROUNDED,
            ))

    console.print()


def rich_failure_details(records: Sequence[EvaluationRecord], max_display: int = 10) -> None:
    """Display detailed information about failed tests.

    Args:
        records: List of EvaluationRecord objects
        max_display: Maximum number of failures to display in detail
    """
    if not RICH_AVAILABLE:
        print("\n=== FAILURE DETAILS ===")
        for record in records:
            if not record.passed:
                print(f"\n  {record.case.case_id}")
                if record.error:
                    print(f"    Error: {record.error}")
                if record.response_text:
                    response_str = str(record.response_text)
                    if len(response_str) > 300:
                        response_str = response_str[:300] + "..."
                    print(f"    Response: {response_str}")
        return

    failed = [r for r in records if not r.passed]
    if not failed:
        return

    console.print(f"\n[bold {COLORS['orange']}]Failure Details[/bold {COLORS['orange']}] ({len(failed)} failed)\n")

    for i, record in enumerate(failed[:max_display]):
        # Build failure panel
        details = Text()

        # Question (truncated)
        question = record.case.question
        if len(question) > 100:
            question = question[:97] + "..."
        details.append("Question: ", style=COLORS['sky'])
        details.append(f"{question}\n", style="white")

        # Tags
        if record.case.tags:
            details.append("Tags: ", style=COLORS['sky'])
            details.append(f"{', '.join(record.case.tags)}\n", style="dim")

        # Error or validation details
        if record.error:
            details.append("\nError: ", style="bold red")
            details.append(f"{record.error}", style="red")
        else:
            # What was called
            if record.validator:
                if record.validator.tool_calls:
                    called = [tc.name for tc in record.validator.tool_calls]
                    details.append("\nCalled: ", style=COLORS['sky'])
                    details.append(f"{', '.join(called)}", style="white")
                else:
                    details.append("\nCalled: ", style=COLORS['sky'])
                    details.append("(text response)", style="dim")

                # What was expected
                expected = record.case.expected_tools or record.case.acceptable_tools
                if expected:
                    details.append("\nExpected: ", style=COLORS['sky'])
                    details.append(f"{', '.join(expected)}", style="green")

                # Issues
                if record.validator.issues:
                    details.append("\n\nIssues:", style=f"bold {COLORS['orange']}")
                    for issue in record.validator.issues[:3]:
                        level_style = "red" if issue.level == "error" else "yellow"
                        details.append(f"\n  [{issue.level.upper()}] ", style=level_style)
                        details.append(issue.message, style="white")

            # Behavior issues
            if record.behavior and not record.behavior.passed:
                details.append("\n\nBehavior Issues:", style="bold yellow")
                for issue in record.behavior.issues:
                    if not issue.passed:
                        details.append(f"\n  [FAIL] ", style="red")
                        details.append(f"{issue.check}: ", style=COLORS['sky'])
                        details.append(issue.message, style="white")

        # Show the actual LLM response (truncated)
        if record.response_text:
            details.append("\n\nLLM Response:", style=f"bold {COLORS['sky']}")
            response_str = str(record.response_text)
            # Truncate long responses but show enough context
            max_response_len = 500
            if len(response_str) > max_response_len:
                truncated = response_str[:max_response_len] + f"... ({len(response_str) - max_response_len} more chars)"
            else:
                truncated = response_str
            details.append(f"\n{truncated}", style="dim white")

        console.print(Panel(
            details,
            title=f"[bold]{record.case.case_id}[/bold]",
            title_align="left",
            border_style=COLORS['orange'] if record.status == "fail" else "yellow",
            box=box.ROUNDED,
        ))

    if len(failed) > max_display:
        console.print(f"  [dim]... and {len(failed) - max_display} more failures[/dim]\n")


def print_evaluation_header(
    model_name: str,
    backend: str,
    total_tests: int,
    scenario_file: str = None,
) -> None:
    """Print a styled header before evaluation starts.

    Args:
        model_name: Name of the model being evaluated
        backend: Backend type (lmstudio, ollama, etc.)
        total_tests: Number of tests to run
        scenario_file: Optional scenario file path
    """
    if not RICH_AVAILABLE:
        print(f"\n{'=' * 60}")
        print(f"  Model Evaluation")
        print(f"{'=' * 60}")
        print(f"  Model: {model_name}")
        print(f"  Backend: {backend}")
        print(f"  Tests: {total_tests}")
        if scenario_file:
            print(f"  Scenario: {scenario_file}")
        print(f"{'=' * 60}\n")
        return

    header_text = Text()
    header_text.append(f"\n  Model:    ", style=COLORS['sky'])
    header_text.append(f"{model_name}\n", style="bold white")
    header_text.append(f"  Backend:  ", style=COLORS['sky'])
    header_text.append(f"{backend}\n", style="white")
    header_text.append(f"  Tests:    ", style=COLORS['sky'])
    header_text.append(f"{total_tests}\n", style="white")
    if scenario_file:
        header_text.append(f"  Scenario: ", style=COLORS['sky'])
        header_text.append(f"{Path(scenario_file).name}", style="dim")

    title = Text()
    title.append(f" {BOX.get('star', '*')} ", style=COLORS['orange'])
    title.append("Model Evaluation", style=f"bold {COLORS['aqua']}")
    title.append(f" {BOX.get('star', '*')} ", style=COLORS['orange'])

    console.print(Panel(
        header_text,
        title=title,
        title_align="center",
        border_style=COLORS['aqua'],
        box=box.DOUBLE,
    ))
    console.print()
