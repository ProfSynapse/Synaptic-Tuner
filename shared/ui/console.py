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
        print("  Fine-tuning for the Claudesidian MCP")
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

    Args:
        options: List of (key, description) tuples
        title: Menu title/prompt

    Returns:
        Selected option key or None if cancelled
    """
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

def confirm(message: str) -> bool:
    """
    Ask for yes/no confirmation.

    Args:
        message: Confirmation prompt

    Returns:
        True if confirmed, False otherwise
    """
    if RICH_AVAILABLE:
        return Confirm.ask(f"  [{COLORS['purple']}]{BOX['arrow']}[/{COLORS['purple']}] {message}")
    else:
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
# ANIMATED MENU (Main menu with bubbling test tube)
# =============================================================================

def animated_menu(
    options: List[Tuple[str, str]],
    title: str = "What would you like to do?",
    status_info: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """
    Display animated logo with bubbling test tube and menu.

    Animation runs until user makes a selection.

    Args:
        options: List of (key, description) tuples
        title: Menu title/prompt
        status_info: Optional dict of status info to display

    Returns:
        Selected option key or None if cancelled
    """
    if not RICH_AVAILABLE:
        # Fallback to static display
        print_logo()
        if status_info:
            for key, value in status_info.items():
                print(f"  {key}: {value}")
        return print_menu(options, title)

    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich.panel import Panel

    # Shared state for animation
    stop_animation = threading.Event()
    user_choice = [None]  # Use list to allow modification in nested function
    input_ready = threading.Event()

    def build_display(frame_num: int) -> Text:
        """Build the complete display for one frame."""
        # Get animated logo
        logo = get_animated_logo_frame(frame_num)

        # Build status section
        status_lines = []
        if status_info:
            for key, value in status_info.items():
                status_lines.append(f"  [{COLORS['cello']}]{BOX['bullet']} {key}[/{COLORS['cello']}] {value}")

        # Build menu
        menu_lines = [
            "",
            f"  [bold]{title}[/bold]",
            "",
        ]
        for i, (key, desc) in enumerate(options, 1):
            menu_lines.append(f"    [{COLORS['orange']}][{i}][/{COLORS['orange']}]  {desc}")
        menu_lines.append(f"    [{COLORS['cello']}][0][/{COLORS['cello']}]  [dim]Exit[/dim]")
        menu_lines.append("")
        menu_lines.append(f"  [{COLORS['orange']}]{BOX['arrow']}[/{COLORS['orange']}] Enter choice: ")

        # Combine all sections
        full_display = logo + "\n"
        if status_lines:
            full_display += "\n".join(status_lines) + "\n"
        full_display += "\n".join(menu_lines)

        return Text.from_markup(full_display)

    def get_input():
        """Get user input in separate thread."""
        try:
            choice = input().strip()
            user_choice[0] = choice
        except (EOFError, KeyboardInterrupt):
            user_choice[0] = "0"
        finally:
            input_ready.set()
            stop_animation.set()

    # Clear screen and start animation
    clear_screen()

    # Start input thread
    input_thread = threading.Thread(target=get_input, daemon=True)

    frame = 0
    try:
        with Live(build_display(frame), console=console, refresh_per_second=4, transient=True) as live:
            input_thread.start()

            while not stop_animation.is_set():
                frame = (frame + 1) % 4
                live.update(build_display(frame))
                time.sleep(0.25)

                # Check if input is ready
                if input_ready.is_set():
                    break

    except KeyboardInterrupt:
        return None

    # Wait for input thread to complete
    input_ready.wait(timeout=1.0)

    # Process the choice
    choice = user_choice[0]
    if choice == "0" or choice is None:
        return None

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(options):
            return options[idx][0]
    except (ValueError, IndexError):
        pass

    # Invalid choice - show error and fall back to static menu
    print_error("Invalid choice.")
    return print_menu(options, title)
