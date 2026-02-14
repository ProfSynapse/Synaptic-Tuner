"""
Interactive Menu for Asciimatics

Contains MenuFrame widget and animated_main_menu function.
"""

from typing import List, Tuple, Optional

from .effects import (
    BRAND_AQUA, BRAND_PURPLE, BRAND_CELLO,
    BrandBubbles,
)

# Check for asciimatics availability
ASCIIMATICS_AVAILABLE = False
try:
    from asciimatics.screen import Screen
    from asciimatics.scene import Scene
    from asciimatics.effects import Print
    from asciimatics.renderers import StaticRenderer
    from asciimatics.exceptions import ResizeScreenError, StopApplication
    from asciimatics.widgets import Frame, Layout, ListBox
    from asciimatics.event import KeyboardEvent
    ASCIIMATICS_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# LOGO CONSTANTS
# =============================================================================

SYNAPTIC_LOGO = "\n".join([
    "███████╗██╗   ██╗███╗   ██╗ █████╗ ██████╗ ████████╗██╗ ██████╗",
    "██╔════╝╚██╗ ██╔╝████╗  ██║██╔══██╗██╔══██╗╚══██╔══╝██║██╔════╝",
    "███████╗ ╚████╔╝ ██╔██╗ ██║███████║██████╔╝   ██║   ██║██║     ",
    "╚════██║  ╚██╔╝  ██║╚██╗██║██╔══██║██╔═══╝    ██║   ██║██║     ",
    "███████║   ██║   ██║ ╚████║██║  ██║██║        ██║   ██║╚██████╗",
    "╚══════╝   ╚═╝   ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝        ╚═╝   ╚═╝ ╚═════╝",
])
SYNAPTIC_WIDTH = 65

TUNER_LOGO = "\n".join([
    "████████╗██╗   ██╗███╗   ██╗███████╗██████╗ ",
    "╚══██╔══╝██║   ██║████╗  ██║██╔════╝██╔══██╗",
    "   ██║   ██║   ██║██╔██╗ ██║█████╗  ██████╔╝",
    "   ██║   ██║   ██║██║╚██╗██║██╔══╝  ██╔══██╗",
    "   ██║   ╚██████╔╝██║ ╚████║███████╗██║  ██║",
    "   ╚═╝    ╚═════╝ ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝",
])
TUNER_WIDTH = 45


# =============================================================================
# MENU FRAME
# =============================================================================

if ASCIIMATICS_AVAILABLE:

    class MenuFrame(Frame):
        """
        A transparent menu frame that overlays on the animation.
        """

        def __init__(self, screen, options, title="What would you like to do?", menu_y=None):
            # Calculate menu dimensions
            menu_height = min(len(options) + 5, screen.height // 3)
            menu_width = min(60, screen.width - 4)

            # Position: use provided y or calculate from bottom
            if menu_y is not None:
                y_pos = menu_y
            else:
                y_pos = screen.height - menu_height - 2

            super().__init__(
                screen,
                menu_height,
                menu_width,
                x=(screen.width - menu_width) // 2,
                y=y_pos,
                has_border=True,
                can_scroll=False,
                title=title,
                reduce_cpu=True,
            )

            self._options = options
            self._selected_key = None

            # Build menu items for ListBox: [(description, key), ...]
            menu_items = [(desc, key) for key, desc in options]
            menu_items.append(("Exit", None))

            # Create layout
            layout = Layout([1], fill_frame=True)
            self.add_layout(layout)

            # Add ListBox with options - first item selected by default
            self._menu = ListBox(
                menu_height - 4,
                menu_items,
                name="menu",
                add_scroll_bar=len(menu_items) > menu_height - 4,
                on_select=self._on_select,
            )
            layout.add_widget(self._menu)

            self.fix()

            # Ensure first option is selected (not Exit)
            if menu_items:
                self._menu.value = menu_items[0][1]  # Select first item's key

        def _on_select(self):
            """Handle menu selection."""
            self._selected_key = self._menu.value
            raise StopApplication("Menu selection made")

        @property
        def selected_key(self):
            return self._selected_key

        def process_event(self, event):
            # Handle Enter key to select
            if isinstance(event, KeyboardEvent):
                if event.key_code in (10, 13):  # Enter
                    self._on_select()
                    return None
                elif event.key_code == -1:  # Escape
                    self._selected_key = None
                    raise StopApplication("Menu cancelled")
            return super().process_event(event)


# =============================================================================
# MENU SCENE CREATION
# =============================================================================

def _create_menu_scene(screen, options, title="What would you like to do?"):
    """Create scene with looping animation + interactive menu."""
    effects = []

    # === Bubble background (added first so it's behind everything) ===
    effects.append(BrandBubbles(screen, count=30))

    # === SYNAPTIC centered at top ===
    synaptic_x = (screen.width - SYNAPTIC_WIDTH) // 2
    logo_y = 1
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

    # === Menu frame below tagline ===
    menu_frame = MenuFrame(screen, options, title, menu_y=tagline_y + 2)
    effects.append(menu_frame)

    # Duration -1 = loop forever until stopped
    return Scene(effects, -1), menu_frame


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

    selected_key = None

    def run_menu(screen):
        nonlocal selected_key
        while True:
            try:
                scene, menu_frame = _create_menu_scene(screen, options, title)
                # screen.play() catches StopApplication internally and returns normally,
                # so we capture selected_key after play returns, not in an except block
                screen.play([scene], stop_on_resize=True)
                selected_key = menu_frame.selected_key
                break  # Normal completion (user made selection)
            except ResizeScreenError:
                pass  # Resize detected - loop back and recreate scene

    Screen.wrapper(run_menu)

    return selected_key
