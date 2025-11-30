"""
User input prompts for Synaptic Tuner.

This module provides functions for getting user input and displaying styled messages.
Delegates to Trainers/shared/ui/ when available, otherwise uses text fallbacks.

Location: /mnt/f/Code/Toolset-Training/tuner/ui/prompts.py
Used by: All handlers for user interaction
"""

import sys
from pathlib import Path
from typing import Dict, Optional

# Try importing from shared UI first
SHARED_UI_AVAILABLE = False
try:
    # Add Trainers/shared to path if not already there
    shared_path = Path(__file__).parent.parent.parent / "Trainers" / "shared"
    if shared_path.exists() and str(shared_path) not in sys.path:
        sys.path.insert(0, str(shared_path))

    from ui import (
        confirm as shared_confirm,
        prompt as shared_prompt,
        print_header as shared_print_header,
        print_config as shared_print_config,
        print_success as shared_print_success,
        print_error as shared_print_error,
        print_info as shared_print_info,
        BOX,
    )
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


def confirm(message: str) -> bool:
    """
    Ask for yes/no confirmation.

    Args:
        message: Confirmation prompt

    Returns:
        True if confirmed, False otherwise

    Example:
        if confirm("Start training?"):
            # User confirmed
            pass
    """
    if SHARED_UI_AVAILABLE:
        return shared_confirm(message)

    # Fallback implementation
    response = input(f"  {message} (y/N): ").strip().lower()
    return response == "y"


def prompt(message: str, default: str = "") -> str:
    """
    Get user input with optional default.

    Args:
        message: Prompt message
        default: Default value if user presses enter without input

    Returns:
        User input string

    Example:
        repo_id = prompt("Enter HuggingFace repo ID", "username/model-name")
    """
    if SHARED_UI_AVAILABLE:
        return shared_prompt(message, default)

    # Fallback implementation
    if default:
        result = input(f"  {message} [{default}]: ").strip()
        return result if result else default
    return input(f"  {message}: ").strip()


def print_header(title: str, subtitle: str = None):
    """
    Print styled header.

    Args:
        title: Header title text
        subtitle: Optional subtitle below header

    Example:
        print_header("TRAINING", "Select your platform and training method")
    """
    if SHARED_UI_AVAILABLE:
        return shared_print_header(title, subtitle)

    # Fallback implementation
    print("\n" + "=" * 60)
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print("=" * 60 + "\n")


def print_config(config: Dict[str, str], title: str = "Configuration"):
    """
    Print configuration key-value pairs in styled table.

    Args:
        config: Dictionary of configuration key-value pairs
        title: Table title

    Example:
        print_config({
            "Platform": "RTX",
            "Method": "SFT",
            "Model": "mistral-7b",
            "Batch Size": "6",
        }, "Training Configuration")
    """
    if SHARED_UI_AVAILABLE:
        return shared_print_config(config, title)

    # Fallback implementation
    print(f"\n  {title}\n  " + "-" * 40)
    for key, value in config.items():
        print(f"    {key}: {value}")
    print()


def print_success(message: str):
    """
    Print success message with checkmark.

    Args:
        message: Success message text

    Example:
        print_success("Training completed successfully")
    """
    if SHARED_UI_AVAILABLE:
        return shared_print_success(message)

    # Fallback implementation
    print(f"  {BOX.get('check', '[OK]')} {message}")


def print_error(message: str):
    """
    Print error message with cross.

    Args:
        message: Error message text

    Example:
        print_error("Training failed: CUDA out of memory")
    """
    if SHARED_UI_AVAILABLE:
        return shared_print_error(message)

    # Fallback implementation
    print(f"  {BOX.get('cross', '[ERROR]')} {message}")


def print_info(message: str):
    """
    Print info message with bullet.

    Args:
        message: Info message text

    Example:
        print_info("Loading configuration from config.yaml")
    """
    if SHARED_UI_AVAILABLE:
        return shared_print_info(message)

    # Fallback implementation
    print(f"  {BOX.get('bullet', '[INFO]')} {message}")
