"""
Synaptic Tuner Console Output

Provides styled console output functions with graceful fallback.
Follows Single Responsibility Principle - only handles display logic.
"""

import os
import sys
import threading
import time
from typing import Optional, Dict, List, Tuple

from .theme import COLORS, LOGO, LOGO_SMALL, TAGLINE, BOX, STYLES, get_animated_logo_frame, get_static_logo

# =============================================================================
# ARROW-KEY MENU SUPPORT (simple-term-menu)
# =============================================================================

TERMINAL_MENU_AVAILABLE = False
try:
    from simple_term_menu import TerminalMenu
    # Also check if we have a TTY (required for arrow-key menus)
    if sys.stdin.isatty() and sys.stdout.isatty():
        TERMINAL_MENU_AVAILABLE = True
except ImportError:
    pass

# =============================================================================
# RICH AVAILABILITY CHECK
# =============================================================================

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    from rich.prompt import Prompt, Confirm
    from rich import box
    from rich.align import Align
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


# =============================================================================
# SCREEN UTILITIES
# =============================================================================

def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


# =============================================================================
# LOGO & HEADERS
# =============================================================================

def print_logo(small: bool = False):
    """
    Print the Synaptic Tuner logo.

    Args:
        small: Use compact logo variant
    """
    if RICH_AVAILABLE:
        logo = LOGO_SMALL if small else LOGO
        console.print(logo)
        console.print(Align.center(TAGLINE))
        console.print()
    else:
        print("\n" + "=" * 60)
        print("  SYNAPTIC TUNER")
        print("  Local LLM Fine-tuning Toolkit")
        print("=" * 60 + "\n")


def print_header(title: str, subtitle: Optional[str] = None):
    """
    Print a styled section header.

    Args:
        title: Header title text
        subtitle: Optional subtitle below header
    """
    if RICH_AVAILABLE:
        header_text = Text()
        header_text.append(f" {BOX['star']} ", style=STYLES["accent"])
        header_text.append(title.upper(), style=STYLES["header"])
        header_text.append(f" {BOX['star']} ", style=STYLES["accent"])

        console.print()
        console.print(Panel(
            Align.center(header_text),
            border_style=COLORS["aqua"],
            box=box.DOUBLE,
            padding=(0, 2),
        ))
        if subtitle:
            console.print(Align.center(Text(subtitle, style=STYLES["muted"])))
        console.print()
    else:
        print()
        print("=" * 60)
        print(f"  {title}")
        if subtitle:
            print(f"  {subtitle}")
        print("=" * 60)
        print()


# =============================================================================
# MENUS
# =============================================================================

def print_menu(options: List[Tuple[str, str]], title: str = "Select an option") -> Optional[str]:
    """
    Print a styled menu and get user selection.

    Uses arrow-key navigation when simple-term-menu is available,
    falls back to numbered menu otherwise.

    Args:
        options: List of (key, description) tuples
        title: Menu title/prompt

    Returns:
        Selected option key or None if cancelled
    """
    # Try arrow-key menu first (best UX)
    if TERMINAL_MENU_AVAILABLE:
        # Build display labels
        labels = [desc for _, desc in options]
        labels.append("Exit")

        # Create menu with styling
        menu = TerminalMenu(
            labels,
            title=f"\n  {title}\n",
            cursor_index=0,
            menu_cursor=" → ",
            menu_cursor_style=("fg_cyan", "bold"),
            menu_highlight_style=("fg_cyan",),
            cycle_cursor=True,
            clear_screen=False,
            show_search_hint=True,
            search_key="/",
            quit_keys=("escape", "q"),
        )

        selected_index = menu.show()

        if selected_index is None or selected_index == len(options):
            return None
        return options[selected_index][0]

    # Fallback to Rich numbered menu
    if RICH_AVAILABLE:
        # Build menu table
        table = Table(
            show_header=False,
            box=box.SIMPLE,
            padding=(0, 2),
            border_style=COLORS["cello"],
        )
        table.add_column("Key", style=COLORS["orange"], width=4, justify="center")
        table.add_column("Option", style="white")

        for i, (key, desc) in enumerate(options, 1):
            table.add_row(f"[{i}]", desc)
        table.add_row("[0]", Text("Back / Exit", style=STYLES["muted"]))

        console.print(f"  [bold]{title}[/bold]")
        console.print()
        console.print(table)
        console.print()

        while True:
            try:
                choice = Prompt.ask(
                    f"  [{COLORS['orange']}]{BOX['arrow']}[/{COLORS['orange']}] Enter choice",
                    default="0"
                )
                if choice == "0":
                    return None
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    return options[idx][0]
            except (ValueError, IndexError):
                pass
            console.print(f"  [{COLORS['orange']}]Invalid choice. Try again.[/{COLORS['orange']}]")
    else:
        # Plain text fallback
        print(title)
        print()
        for i, (key, desc) in enumerate(options, 1):
            print(f"  [{i}] {desc}")
        print(f"  [0] Back / Exit")
        print()

        while True:
            try:
                choice = input("Enter choice: ").strip()
                if choice == "0":
                    return None
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    return options[idx][0]
            except (ValueError, IndexError):
                pass
            print("Invalid choice. Try again.")


# =============================================================================
# CONFIGURATION DISPLAY
# =============================================================================

def print_config(config: Dict[str, str], title: str = "Configuration"):
    """
    Print a configuration summary in a styled table.

    Args:
        config: Dictionary of key-value pairs
        title: Table title
    """
    if RICH_AVAILABLE:
        table = Table(
            show_header=False,
            box=box.ROUNDED,
            border_style=COLORS["cello"],
            padding=(0, 1),
            title=f"[bold]{title}[/bold]",
            title_style=STYLES["header"],
        )
        table.add_column("Key", style=COLORS["sky"])
        table.add_column("Value", style="white")

        for key, value in config.items():
            table.add_row(key, str(value))

        console.print()
        console.print(table)
        console.print()
    else:
        print()
        print(f"  {title}")
        print("  " + "-" * 40)
        for key, value in config.items():
            print(f"    {key}: {value}")
        print()


# =============================================================================
# STATUS MESSAGES
# =============================================================================

def print_success(message: str):
    """Print a success message with checkmark."""
    if RICH_AVAILABLE:
        console.print(f"  [{COLORS['aqua']}]{BOX['check']}[/{COLORS['aqua']}] {message}")
    else:
        print(f"  [OK] {message}")


def print_error(message: str):
    """Print an error message with cross."""
    if RICH_AVAILABLE:
        console.print(f"  [{COLORS['orange']}]{BOX['cross']}[/{COLORS['orange']}] {message}")
    else:
        print(f"  [ERROR] {message}")


def print_info(message: str):
    """Print an info message with bullet."""
    if RICH_AVAILABLE:
        console.print(f"  [{COLORS['sky']}]{BOX['bullet']}[/{COLORS['sky']}] {message}")
    else:
        print(f"  [INFO] {message}")


# =============================================================================
# USER INPUT
# =============================================================================

def confirm(message: str, use_arrows: bool = True) -> bool:
    """
    Ask for yes/no confirmation.

    Uses arrow-key selection when available and use_arrows=True.

    Args:
        message: Confirmation prompt
        use_arrows: Use arrow-key selection (default True)

    Returns:
        True if confirmed, False otherwise
    """
    # Arrow-key confirmation
    if use_arrows and TERMINAL_MENU_AVAILABLE:
        menu = TerminalMenu(
            ["Yes", "No"],
            title=f"\n  {message}\n",
            cursor_index=1,  # Default to "No" for safety
            menu_cursor=" → ",
            menu_cursor_style=("fg_cyan", "bold"),
            quit_keys=("escape", "q"),
        )
        selected = menu.show()
        return selected == 0  # 0 = Yes

    # Rich text confirmation
    if RICH_AVAILABLE:
        return Confirm.ask(f"  [{COLORS['purple']}]{BOX['arrow']}[/{COLORS['purple']}] {message}")

    # Plain text fallback
    response = input(f"  {message} (y/N): ").strip().lower()
    return response == "y"


def prompt(message: str, default: str = "") -> str:
    """
    Get user input with optional default.

    Args:
        message: Prompt message
        default: Default value if empty input

    Returns:
        User input string
    """
    if RICH_AVAILABLE:
        return Prompt.ask(
            f"  [{COLORS['sky']}]{BOX['arrow']}[/{COLORS['sky']}] {message}",
            default=default
        )
    else:
        if default:
            result = input(f"  {message} [{default}]: ").strip()
            return result if result else default
        return input(f"  {message}: ").strip()


# =============================================================================
# ANIMATED MENU (Main menu with bubbling beaker - asciimatics powered)
# =============================================================================

def animated_menu(
    options: List[Tuple[str, str]],
    title: str = "What would you like to do?",
    status_info: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """
    Display animated logo with interactive menu overlay using asciimatics.

    The animation loops continuously until user makes a selection.
    Menu is displayed at the bottom with arrow-key navigation.

    Args:
        options: List of (key, description) tuples
        title: Menu title/prompt
        status_info: Optional dict of status info (currently not displayed in animation)

    Returns:
        Selected option key or None if cancelled
    """
    # Import animations module (auto-installs asciimatics if needed)
    from .animations import animated_main_menu, ASCIIMATICS_AVAILABLE

    if not ASCIIMATICS_AVAILABLE:
        # If asciimatics still not available after auto-install attempt,
        # use simple arrow-key or numbered menu
        clear_screen()
        print_logo()
        if status_info:
            for key, value in status_info.items():
                print(f"  {key}: {value}")
            print()
        return print_menu(options, title)

    # Use asciimatics animated menu with looping animation
    return animated_main_menu(options, title)
