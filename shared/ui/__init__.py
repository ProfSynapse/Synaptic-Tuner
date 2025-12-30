"""
Synaptic Tuner UI Components

Provides styled console output with brand theming.
"""

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
    # Dashboard
    "LiveDashboard",
    "show_training_progress",
]
