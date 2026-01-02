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
    BRAND_AQUA, BRAND_PURPLE, BRAND_CELLO, BRAND_ORANGE,
    BrandBubbles, SparkBurst, ProgressLoader,
)
from .menu import SYNAPTIC_LOGO, SYNAPTIC_WIDTH, TUNER_LOGO, TUNER_WIDTH


# =============================================================================
# TRAINING ASCII ART (same style as SYNAPTIC/TUNER)
# =============================================================================

IGNITING_LOGO = "\n".join([
    "██╗ ██████╗ ███╗   ██╗██╗████████╗██╗███╗   ██╗ ██████╗ ",
    "██║██╔════╝ ████╗  ██║██║╚══██╔══╝██║████╗  ██║██╔════╝ ",
    "██║██║  ███╗██╔██╗ ██║██║   ██║   ██║██╔██╗ ██║██║  ███╗",
    "██║██║   ██║██║╚██╗██║██║   ██║   ██║██║╚██╗██║██║   ██║",
    "██║╚██████╔╝██║ ╚████║██║   ██║   ██║██║ ╚████║╚██████╔╝",
    "╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝   ╚═╝   ╚═╝╚═╝  ╚═══╝ ╚═════╝ ",
])
IGNITING_WIDTH = 56

TRAINING_LOGO = "\n".join([
    "████████╗██████╗  █████╗ ██╗███╗   ██╗██╗███╗   ██╗ ██████╗ ",
    "╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║██║████╗  ██║██╔════╝ ",
    "   ██║   ██████╔╝███████║██║██╔██╗ ██║██║██╔██╗ ██║██║  ███╗",
    "   ██║   ██╔══██╗██╔══██║██║██║╚██╗██║██║██║╚██╗██║██║   ██║",
    "   ██║   ██║  ██║██║  ██║██║██║ ╚████║██║██║ ╚████║╚██████╔╝",
    "   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝ ",
])
TRAINING_WIDTH = 61

COMPLETE_LOGO = "\n".join([
    " ██████╗ ██████╗ ███╗   ███╗██████╗ ██╗     ███████╗████████╗███████╗██╗",
    "██╔════╝██╔═══██╗████╗ ████║██╔══██╗██║     ██╔════╝╚══██╔══╝██╔════╝██║",
    "██║     ██║   ██║██╔████╔██║██████╔╝██║     █████╗     ██║   █████╗  ██║",
    "██║     ██║   ██║██║╚██╔╝██║██╔═══╝ ██║     ██╔══╝     ██║   ██╔══╝  ╚═╝",
    "╚██████╗╚██████╔╝██║ ╚═╝ ██║██║     ███████╗███████╗   ██║   ███████╗██╗",
    " ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝     ╚══════╝╚══════╝   ╚═╝   ╚══════╝╚═╝",
])
COMPLETE_WIDTH = 71

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
    """Create the training start splash with fireworks and brand-styled text."""
    if not ASCIIMATICS_AVAILABLE:
        return None

    effects = []

    # Starfield background
    effects.append(Stars(screen, screen.width))

    # Fireworks scattered around - start early so they're visible
    firework_types = [StarFirework, RingFirework, PalmFirework]
    for i in range(8):
        firework_class = random.choice(firework_types)
        effects.append(firework_class(
            screen,
            x=random.randint(5, screen.width - 5),
            y=random.randint(3, screen.height - 3),
            life_time=20,  # Fixed shorter lifetime
            start_frame=i * 3,  # Stagger them: 0, 3, 6, 9, 12, 15, 18, 21
        ))

    # IGNITING text (orange, brand style) - appears immediately
    igniting_x = (screen.width - IGNITING_WIDTH) // 2
    igniting_y = screen.height // 3 - 3
    effects.append(Print(
        screen,
        StaticRenderer([IGNITING_LOGO]),
        x=igniting_x,
        y=igniting_y,
        colour=BRAND_ORANGE,
        attr=Screen.A_BOLD,
        start_frame=0,
    ))

    # TRAINING text (purple, brand style) - appears immediately
    training_x = (screen.width - TRAINING_WIDTH) // 2
    training_y = igniting_y + 8
    effects.append(Print(
        screen,
        StaticRenderer([TRAINING_LOGO]),
        x=training_x,
        y=training_y,
        colour=BRAND_PURPLE,
        attr=Screen.A_BOLD,
        start_frame=0,
    ))

    # Progress loader at bottom - shows it's not stuck
    loader_y = training_y + 9
    effects.append(ProgressLoader(
        screen,
        y=loader_y,
        duration=duration,
        width=40,
        label="Initializing",
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
    firework_types = [PalmFirework, StarFirework, RingFirework, SerpentFirework]

    for i in range(15):
        firework_class = random.choice(firework_types)
        effects.append(firework_class(
            screen,
            x=random.randint(3, screen.width - 4),
            y=random.randint(1, screen.height - 2),
            life_time=random.randint(20, 35),
            start_frame=random.randint(0, 80),
        ))

    # COMPLETE! text (aqua, brand style)
    complete_x = (screen.width - COMPLETE_WIDTH) // 2
    complete_y = screen.height // 3
    effects.append(Print(
        screen,
        StaticRenderer([COMPLETE_LOGO]),
        x=complete_x,
        y=complete_y,
        colour=BRAND_AQUA,
        attr=Screen.A_BOLD,
        start_frame=20,
    ))

    # Subtitle
    subtitle = "Training finished successfully!"
    effects.append(Print(
        screen,
        StaticRenderer([subtitle]),
        x=(screen.width - len(subtitle)) // 2,
        y=complete_y + 8,
        colour=BRAND_CELLO,
        start_frame=40,
    ))

    return Scene(effects, duration)


def create_simple_celebration_scene(screen, duration: int = 80):
    """A simpler, shorter celebration for quick feedback."""
    if not ASCIIMATICS_AVAILABLE:
        return None

    effects = []

    # Starfield background
    effects.append(Stars(screen, screen.width))

    # A few fireworks
    for i in range(8):
        effects.append(StarFirework(
            screen,
            x=random.randint(10, screen.width - 10),
            y=random.randint(5, screen.height - 5),
            life_time=25,
            start_frame=i * 5,
        ))

    # COMPLETE! text (aqua, brand style)
    complete_x = (screen.width - COMPLETE_WIDTH) // 2
    complete_y = screen.height // 2 - 3
    effects.append(Print(
        screen,
        StaticRenderer([COMPLETE_LOGO]),
        x=complete_x,
        y=complete_y,
        colour=BRAND_AQUA,
        attr=Screen.A_BOLD,
        start_frame=10,
    ))

    return Scene(effects, duration)
