"""
Animation Scenes for Synaptic Tuner

Contains scene creation functions for:
- Logo display (non-interactive)
- Training start splash
- Training complete celebration
"""

import random
from typing import Optional

from .effects import (
    BRAND_AQUA, BRAND_PURPLE, BRAND_CELLO,
    BrandBubbles, SparkBurst,
)
from .menu import SYNAPTIC_LOGO, SYNAPTIC_WIDTH, TUNER_LOGO, TUNER_WIDTH

# Check for asciimatics availability
ASCIIMATICS_AVAILABLE = False
try:
    from asciimatics.screen import Screen
    from asciimatics.scene import Scene
    from asciimatics.effects import Print, Stars
    from asciimatics.renderers import FigletText, Fire, StaticRenderer, Rainbow
    from asciimatics.particles import (
        StarFirework, RingFirework, PalmFirework, SerpentFirework,
    )
    ASCIIMATICS_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# SCENE CREATION FUNCTIONS
# =============================================================================

def create_logo_scene(screen, duration: int = 80):
    """Create the main menu logo scene with brand-colored bubble background."""
    if not ASCIIMATICS_AVAILABLE:
        return None

    effects = []

    # === Bubble background (added first so it's behind everything) ===
    effects.append(BrandBubbles(screen, count=30))

    # === SYNAPTIC centered at top ===
    synaptic_x = (screen.width - SYNAPTIC_WIDTH) // 2
    logo_y = 2
    effects.append(Print(
        screen,
        StaticRenderer([SYNAPTIC_LOGO]),
        x=synaptic_x,
        y=logo_y,
        colour=BRAND_AQUA,  # Brand aqua #00A99D
        attr=Screen.A_BOLD,
    ))

    # === TUNER centered below SYNAPTIC ===
    tuner_x = (screen.width - TUNER_WIDTH) // 2
    tuner_y = logo_y + 7
    effects.append(Print(
        screen,
        StaticRenderer([TUNER_LOGO]),
        x=tuner_x,
        y=tuner_y,
        colour=BRAND_PURPLE,  # Brand purple #93278F
        attr=Screen.A_BOLD,
    ))

    # === Tagline centered below TUNER ===
    tagline = "Local LLM Fine-tuning Toolkit"
    tagline_y = tuner_y + 7
    effects.append(Print(
        screen,
        StaticRenderer([tagline]),
        x=(screen.width - len(tagline)) // 2,
        y=tagline_y,
        colour=BRAND_CELLO,  # Brand cello (muted)
    ))

    return Scene(effects, duration)


def create_training_start_scene(screen, duration: int = 60):
    """Create the training start splash with fire and sparks."""
    if not ASCIIMATICS_AVAILABLE:
        return None

    effects = []

    # Fire at the bottom
    effects.append(Print(
        screen,
        Fire(screen.height, screen.width, "*" * (screen.width // 2), 0.6, 40, screen.colours),
        x=screen.width // 4,
        y=screen.height - 8,
        speed=1,
        transparent=False,
    ))

    # Multiple spark bursts
    for _ in range(3):
        effects.append(SparkBurst(
            screen,
            x=random.randint(screen.width // 4, 3 * screen.width // 4),
            y=screen.height // 2,
            count=20
        ))

    # Title
    effects.append(Print(
        screen,
        FigletText("IGNITING", font='banner'),
        x=(screen.width - 50) // 2,
        y=screen.height // 3,
        colour=Screen.COLOUR_YELLOW,
        attr=Screen.A_BOLD,
    ))

    effects.append(Print(
        screen,
        FigletText("TRAINING", font='banner'),
        x=(screen.width - 50) // 2,
        y=screen.height // 3 + 5,
        colour=Screen.COLOUR_RED,
        attr=Screen.A_BOLD,
    ))

    return Scene(effects, duration)


def create_celebration_scene(screen, duration: int = 120):
    """Create the training complete celebration with fireworks."""
    if not ASCIIMATICS_AVAILABLE:
        return None

    effects = []

    # Starfield
    effects.append(Stars(screen, screen.width))

    # Multiple fireworks
    firework_types = [
        (PalmFirework, Screen.COLOUR_GREEN),
        (StarFirework, Screen.COLOUR_CYAN),
        (RingFirework, Screen.COLOUR_MAGENTA),
        (SerpentFirework, Screen.COLOUR_YELLOW),
    ]

    for i in range(15):
        firework_class, _ = random.choice(firework_types)
        effects.append(firework_class(
            screen,
            x=random.randint(3, screen.width - 4),
            y=random.randint(1, screen.height - 2),
            life_time=random.randint(20, 35),
            start_frame=random.randint(0, 80),
        ))

    # Rainbow congratulations text
    effects.append(Print(
        screen,
        Rainbow(screen, FigletText("COMPLETE!", font='banner')),
        x=(screen.width - 55) // 2,
        y=screen.height // 3,
        speed=1,
        start_frame=20,
    ))

    effects.append(Print(
        screen,
        StaticRenderer(["Training finished successfully!"]),
        x=(screen.width - 32) // 2,
        y=screen.height // 3 + 8,
        colour=Screen.COLOUR_WHITE,
        start_frame=40,
    ))

    return Scene(effects, duration)


def create_simple_celebration_scene(screen, duration: int = 80):
    """A simpler, shorter celebration for quick feedback."""
    if not ASCIIMATICS_AVAILABLE:
        return None

    effects = []

    # Central explosion burst
    centre_x = screen.width // 2
    centre_y = screen.height // 2

    effects.append(SparkBurst(screen, centre_x, centre_y, count=50))

    # A few fireworks
    for i in range(5):
        effects.append(StarFirework(
            screen,
            x=random.randint(10, screen.width - 10),
            y=random.randint(5, screen.height - 5),
            life_time=25,
            start_frame=i * 10,
        ))

    # Text
    effects.append(Print(
        screen,
        FigletText("DONE!", font='banner'),
        x=(screen.width - 30) // 2,
        y=screen.height // 2 - 3,
        colour=Screen.COLOUR_GREEN,
        attr=Screen.A_BOLD,
        start_frame=10,
    ))

    return Scene(effects, duration)
