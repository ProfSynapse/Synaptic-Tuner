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
)

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
]
