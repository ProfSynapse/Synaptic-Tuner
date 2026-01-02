"""
Asciimatics-based Animations for Synaptic Tuner

Public API for terminal animations:
- animated_main_menu: Interactive menu with animated background
- play_logo_animation: Logo splash animation
- play_training_start: Training start splash
- play_training_complete: Training complete celebration
- is_available: Check if animations are available

These animations temporarily take over the terminal, then return control to Rich.
"""

import subprocess
import sys
from typing import Optional, Callable, List, Tuple


def _ensure_asciimatics_installed():
    """Auto-install asciimatics if not present."""
    try:
        import asciimatics
        return True
    except ImportError:
        print("Installing asciimatics for terminal animations...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "asciimatics", "-q"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("asciimatics installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print("Warning: Could not install asciimatics. Animations will be disabled.")
            return False


# Auto-install asciimatics if needed
_ensure_asciimatics_installed()

# Check for asciimatics availability (after potential install)
ASCIIMATICS_AVAILABLE = False
try:
    from asciimatics.screen import Screen
    from asciimatics.exceptions import ResizeScreenError, StopApplication
    ASCIIMATICS_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# PUBLIC API
# =============================================================================

def animated_main_menu(
    options: List[Tuple[str, str]],
    title: str = "What would you like to do?",
) -> Optional[str]:
    """
    Display animated logo with interactive menu overlay.

    The animation loops continuously until the user makes a selection.
    Menu is displayed at the bottom of the screen with arrow-key navigation.

    Args:
        options: List of (key, description) tuples
        title: Menu title/prompt

    Returns:
        Selected option key or None if cancelled/exit

    Example:
        choice = animated_main_menu([
            ("train", "Train a model"),
            ("upload", "Upload to HuggingFace"),
            ("eval", "Evaluate a model"),
        ])
    """
    if not ASCIIMATICS_AVAILABLE:
        return None  # Caller should fall back to Rich menu

    # Import here to avoid circular imports and ensure asciimatics is ready
    from .menu import animated_main_menu as _animated_main_menu
    return _animated_main_menu(options, title)


def play_logo_animation(duration_frames: int = 80, on_complete: Optional[Callable] = None):
    """
    Play the main menu logo animation with bubbling beaker.

    Args:
        duration_frames: How long to play (in frames, ~20fps)
        on_complete: Optional callback when animation ends

    Note: Temporarily takes over terminal, then returns control.
    """
    if not ASCIIMATICS_AVAILABLE:
        if on_complete:
            on_complete()
        return

    from .scenes import create_logo_scene

    def play(screen):
        scene = create_logo_scene(screen, duration_frames)
        if scene:
            screen.play([scene], stop_on_resize=True)

    try:
        Screen.wrapper(play)
    except (ResizeScreenError, StopApplication):
        pass

    if on_complete:
        on_complete()


def play_training_start(duration_frames: int = 50):
    """
    Play the training start animation with fire and sparks.

    Args:
        duration_frames: How long to play (in frames, ~20fps)
    """
    if not ASCIIMATICS_AVAILABLE:
        return

    from .scenes import create_training_start_scene

    def play(screen):
        scene = create_training_start_scene(screen, duration_frames)
        if scene:
            screen.play([scene], stop_on_resize=True)

    try:
        Screen.wrapper(play)
    except (ResizeScreenError, StopApplication):
        pass


def play_training_complete(simple: bool = True, duration_frames: int = 80):
    """
    Play the training complete celebration animation.

    Args:
        simple: Use shorter, simpler animation (recommended)
        duration_frames: How long to play
    """
    if not ASCIIMATICS_AVAILABLE:
        return

    from .scenes import create_simple_celebration_scene, create_celebration_scene

    def play(screen):
        if simple:
            scene = create_simple_celebration_scene(screen, duration_frames)
        else:
            scene = create_celebration_scene(screen, duration_frames)
        if scene:
            screen.play([scene], stop_on_resize=True)

    try:
        Screen.wrapper(play)
    except (ResizeScreenError, StopApplication):
        pass


def is_available() -> bool:
    """Check if asciimatics animations are available."""
    return ASCIIMATICS_AVAILABLE


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    import time

    if not ASCIIMATICS_AVAILABLE:
        print("asciimatics not installed. Install with: pip install asciimatics")
        sys.exit(1)

    print("Testing animations...")
    print("1. Logo animation (3 seconds)")
    time.sleep(1)
    play_logo_animation(duration_frames=60)

    print("\n2. Training start animation (2 seconds)")
    time.sleep(1)
    play_training_start(duration_frames=40)

    print("\n3. Training complete animation (3 seconds)")
    time.sleep(1)
    play_training_complete(simple=True, duration_frames=60)

    print("\nAll animations complete!")
