"""
Synaptic Tuner UI Components

Provides styled console output with brand theming.
"""

from typing import List, Tuple

from .theme import (
    COLORS, LOGO, LOGO_SMALL, TAGLINE, BOX, STYLES,
    get_animated_logo_frame, get_static_logo,
)
from .console import (
    console,
    clear_screen,
    print_logo,
    print_header,
    print_menu,
    print_config,
    print_success,
    print_error,
    print_info,
    confirm,
    prompt,
    animated_menu,
    RICH_AVAILABLE,
    TERMINAL_MENU_AVAILABLE,
)
from .widgets import (
    spinner,
    progress_spinner,
    file_picker,
    fuzzy_menu,
    multi_select_menu,
    sparkline,
    sparkline_with_labels,
    info_panel,
    comparison_table,
    suppress_logs,
    capture_output,
    quiet_training,
)
from .dashboard import LiveDashboard, show_training_progress
from .evaluation import LiveEvaluationDashboard, rich_evaluation_summary
from .synthchat import LiveSynthChatDashboard, SynthChatMetrics
from .animations import (
    animated_main_menu as asciimatics_menu,
    play_logo_animation,
    play_training_start,
    play_training_complete,
    is_available as animations_available,
    ASCIIMATICS_AVAILABLE,
)


def print_checkbox_menu(
    options: List[Tuple[str, str, bool]],  # (key, label, default_selected)
    title: str = "Select options:",
    min_select: int = 1,
    max_select: int = None,
) -> List[str]:
    """
    Display checkbox multi-select menu using simple-term-menu.

    Args:
        options: List of (key, label, default_selected) tuples
        title: Menu title
        min_select: Minimum selections required
        max_select: Maximum selections allowed (None = unlimited)

    Returns:
        List of selected keys
    """
    try:
        from simple_term_menu import TerminalMenu

        # Build menu entries
        entries = []
        preselected = []
        for i, (key, label, selected) in enumerate(options):
            entries.append(label)
            if selected:
                preselected.append(i)

        menu = TerminalMenu(
            entries,
            title=title,
            multi_select=True,
            show_multi_select_hint=True,
            preselected_entries=preselected if preselected else None,
            multi_select_select_on_accept=False,
            multi_select_empty_ok=(min_select == 0),
        )

        selected_indices = menu.show()

        if selected_indices is None:
            return []

        # Handle single selection (returns int) vs multi (returns tuple)
        if isinstance(selected_indices, int):
            selected_indices = (selected_indices,)

        return [options[i][0] for i in selected_indices]

    except ImportError:
        # Fallback to numbered input
        print(title)
        for i, (key, label, selected) in enumerate(options, 1):
            marker = "[x]" if selected else "[ ]"
            print(f"  {i}. {marker} {label}")

        response = input("\nEnter numbers (comma-separated) or 'default': ").strip()

        if response.lower() == 'default':
            return [key for key, _, selected in options if selected]

        try:
            indices = [int(x.strip()) - 1 for x in response.split(",") if x.strip().isdigit()]
            return [options[i][0] for i in indices if 0 <= i < len(options)]
        except (ValueError, IndexError):
            return []


__all__ = [
    # Theme
    "COLORS",
    "LOGO",
    "LOGO_SMALL",
    "TAGLINE",
    "BOX",
    "STYLES",
    "get_animated_logo_frame",
    "get_static_logo",
    # Console
    "console",
    "clear_screen",
    "print_logo",
    "print_header",
    "print_menu",
    "print_config",
    "print_success",
    "print_error",
    "print_info",
    "confirm",
    "prompt",
    "animated_menu",
    "RICH_AVAILABLE",
    "TERMINAL_MENU_AVAILABLE",
    # Widgets
    "spinner",
    "progress_spinner",
    "file_picker",
    "fuzzy_menu",
    "multi_select_menu",
    "sparkline",
    "sparkline_with_labels",
    "info_panel",
    "comparison_table",
    "suppress_logs",
    "capture_output",
    "quiet_training",
    # Checkbox Menu
    "print_checkbox_menu",
    # Dashboard
    "LiveDashboard",
    "show_training_progress",
    # Evaluation
    "LiveEvaluationDashboard",
    "rich_evaluation_summary",
    # SynthChat
    "LiveSynthChatDashboard",
    "SynthChatMetrics",
    # Animations
    "asciimatics_menu",
    "play_logo_animation",
    "play_training_start",
    "play_training_complete",
    "animations_available",
    "ASCIIMATICS_AVAILABLE",
]
