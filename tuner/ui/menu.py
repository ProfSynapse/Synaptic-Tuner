"""
Menu rendering for Synaptic Tuner.

This module provides menu display functions, delegating to Trainers/shared/ui/
when available and providing text-based fallbacks otherwise.

Location: /mnt/f/Code/Toolset-Training/tuner/ui/menu.py
Used by: All handlers for interactive menu navigation
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# Try importing from shared UI first
SHARED_UI_AVAILABLE = False
try:
    # Add Trainers/shared to path if not already there
    shared_path = Path(__file__).parent.parent.parent / "Trainers" / "shared"
    if shared_path.exists() and str(shared_path) not in sys.path:
        sys.path.insert(0, str(shared_path))

    from ui import print_menu as shared_print_menu
    from ui import animated_menu as shared_animated_menu
    from ui import BOX
    SHARED_UI_AVAILABLE = True
except ImportError:
    # Fallback BOX characters
    BOX = {
        "bullet": "•",
        "star": "★",
        "check": "✓",
        "cross": "✗",
        "arrow": "→",
        "dot": "·",
    }


def print_menu(options: List[Tuple[str, str]], title: str = "Select an option") -> Optional[str]:
    """
    Display numbered menu and get user selection.

    Delegates to Trainers/shared/ui/ if available, otherwise uses text fallback.

    Args:
        options: List of (key, description) tuples
        title: Menu title/prompt

    Returns:
        Selected option key or None if user cancelled/exited

    Example:
        choice = print_menu([
            ("sft", "Supervised Fine-Tuning"),
            ("kto", "Preference Learning"),
        ], "Select training method")

        if choice == "sft":
            # User selected SFT
            pass
    """
    if SHARED_UI_AVAILABLE:
        return shared_print_menu(options, title)

    # Fallback implementation
    print(title + "\n")
    for i, (key, desc) in enumerate(options, 1):
        print(f"  [{i}] {desc}")
    print("  [0] Back / Exit\n")

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


def animated_menu(
    options: List[Tuple[str, str]],
    title: str = "What would you like to do?",
    status_info: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """
    Show animated logo then menu (if rich available), otherwise static menu.

    Delegates to Trainers/shared/ui/ if available, otherwise uses text fallback.

    Args:
        options: List of (key, description) tuples
        title: Menu title/prompt
        status_info: Optional status information to display above menu

    Returns:
        Selected option key or None if user cancelled/exited

    Example:
        choice = animated_menu([
            ("train", "Start training"),
            ("upload", "Upload model"),
        ], "Main Menu", status_info={
            "Python": sys.executable,
            "Environment": os.environ.get("CONDA_DEFAULT_ENV", "Unknown"),
        })
    """
    if SHARED_UI_AVAILABLE:
        return shared_animated_menu(options, title, status_info)

    # Fallback implementation - just show static menu
    print("\n" + "=" * 60)
    print("  SYNAPTIC TUNER")
    print("  Fine-tuning for the Claudesidian MCP")
    print("=" * 60 + "\n")

    if status_info:
        for key, value in status_info.items():
            print(f"  {key}: {value}")
        print()

    return print_menu(options, title)
