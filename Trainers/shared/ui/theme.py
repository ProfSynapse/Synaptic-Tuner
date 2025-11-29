"""
Synaptic Tuner Brand Theme

Contains all branding elements: colors, ASCII art, and visual constants.
Follows Single Responsibility Principle - only contains theme definitions.
"""

# =============================================================================
# BRAND COLORS
# =============================================================================

COLORS = {
    # Primary - Headers, success messages
    "aqua": "#00A99D",
    # RGB: (0, 169, 157) - Persian Green

    # Accent - Highlights, emphasis
    "purple": "#93278F",
    # RGB: (147, 39, 143) - Dark Purple

    # Subtle - Borders, muted text
    "cello": "#33475B",
    # RGB: (51, 71, 91) - Cello

    # Warning - Prompts, attention
    "orange": "#F7931E",
    # RGB: (247, 147, 30) - Carrot Orange

    # Info - Selections, information
    "sky": "#29ABE2",
    # RGB: (41, 171, 226) - Summer Sky

    # Light text on dark backgrounds
    "cream": "#FBF7F1",
    # RGB: (251, 247, 241) - Floral White
}


# =============================================================================
# ASCII ART LOGOS
# =============================================================================

LOGO = """
[#00A99D]
    ███████╗██╗   ██╗███╗   ██╗ █████╗ ██████╗ ████████╗██╗ ██████╗
    ██╔════╝╚██╗ ██╔╝████╗  ██║██╔══██╗██╔══██╗╚══██╔══╝██║██╔════╝
    ███████╗ ╚████╔╝ ██╔██╗ ██║███████║██████╔╝   ██║   ██║██║
    ╚════██║  ╚██╔╝  ██║╚██╗██║██╔══██║██╔═══╝    ██║   ██║██║
    ███████║   ██║   ██║ ╚████║██║  ██║██║        ██║   ██║╚██████╗
    ╚══════╝   ╚═╝   ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝        ╚═╝   ╚═╝ ╚═════╝
[/#00A99D][#93278F]
                        ████████╗██╗   ██╗███╗   ██╗███████╗██████╗
                        ╚══██╔══╝██║   ██║████╗  ██║██╔════╝██╔══██╗
                           ██║   ██║   ██║██╔██╗ ██║█████╗  ██████╔╝
                           ██║   ██║   ██║██║╚██╗██║██╔══╝  ██╔══██╗
                           ██║   ╚██████╔╝██║ ╚████║███████╗██║  ██║
                           ╚═╝    ╚═════╝ ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝
[/#93278F]"""

LOGO_SMALL = """[#00A99D]╔═╗╦ ╦╔╗╔╔═╗╔═╗╔╦╗╦╔═╗[/#00A99D] [#93278F]╔╦╗╦ ╦╔╗╔╔═╗╦═╗[/#93278F]
[#00A99D]╚═╗╚╦╝║║║╠═╣╠═╝ ║ ║║  [/#00A99D] [#93278F] ║ ║ ║║║║║╣ ╠╦╝[/#93278F]
[#00A99D]╚═╝ ╩ ╝╚╝╩ ╩╩   ╩ ╩╚═╝[/#00A99D] [#93278F] ╩ ╚═╝╝╚╝╚═╝╩╚═[/#93278F]"""

TAGLINE = "[dim]Fine-tuning for the Claudesidian MCP[/dim]"


# =============================================================================
# TEST TUBE WITH LOGO (ANIMATED FRAMES)
# =============================================================================

# Test tube parts (for composing frames)
TUBE_TOP = "[#33475B]╭───╮[/#33475B]"
TUBE_MID = "[#33475B]│[/#33475B]{inner}[#33475B]│[/#33475B]"
TUBE_LIQ = "[#33475B]│[/#33475B][#00A99D]▓▓▓[/#00A99D][#33475B]│[/#33475B]"
TUBE_BOT = "[#33475B]│[/#33475B][#00A99D]███[/#00A99D][#33475B]│[/#33475B]"
TUBE_END = "[#33475B]╰───╯[/#33475B]"

# Bubble characters with colors
B1 = "[#29ABE2]○[/#29ABE2]"  # Large bubble (sky blue)
B2 = "[#00A99D]•[/#00A99D]"  # Small bubble (aqua)

# Animation frames for bubbles (4 frames)
BUBBLE_FRAMES = [
    # Frame 0: bubbles high
    {"float2": f"  {B1} {B2}", "float1": f" {B2}  {B1}", "tube": f" {B1} "},
    # Frame 1: bubbles shift
    {"float2": f" {B2}  {B1}", "float1": f"  {B1} {B2}", "tube": f"  {B2}"},
    # Frame 2: bubbles move
    {"float2": f"   {B1}{B2}", "float1": f" {B1} {B2} ", "tube": f"{B1}  "},
    # Frame 3: bubbles cycle
    {"float2": f" {B1}  {B2}", "float1": f"  {B2}{B1} ", "tube": f" {B2} "},
]

# TUNER ASCII art lines (purple) - for combining with test tube
TUNER_LINES = [
    "[#93278F]████████╗██╗   ██╗███╗   ██╗███████╗██████╗[/#93278F]",
    "[#93278F]╚══██╔══╝██║   ██║████╗  ██║██╔════╝██╔══██╗[/#93278F]",
    "[#93278F]   ██║   ██║   ██║██╔██╗ ██║█████╗  ██████╔╝[/#93278F]",
    "[#93278F]   ██║   ██║   ██║██║╚██╗██║██╔══╝  ██╔══██╗[/#93278F]",
    "[#93278F]   ██║   ╚██████╔╝██║ ╚████║███████╗██║  ██║[/#93278F]",
    "[#93278F]   ╚═╝    ╚═════╝ ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝[/#93278F]",
]

# SYNAPTIC ASCII art (aqua) - static header
SYNAPTIC_ART = """[#00A99D]
    ███████╗██╗   ██╗███╗   ██╗ █████╗ ██████╗ ████████╗██╗ ██████╗
    ██╔════╝╚██╗ ██╔╝████╗  ██║██╔══██╗██╔══██╗╚══██╔══╝██║██╔════╝
    ███████╗ ╚████╔╝ ██╔██╗ ██║███████║██████╔╝   ██║   ██║██║
    ╚════██║  ╚██╔╝  ██║╚██╗██║██╔══██║██╔═══╝    ██║   ██║██║
    ███████║   ██║   ██║ ╚████║██║  ██║██║        ██║   ██║╚██████╗
    ╚══════╝   ╚═╝   ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝        ╚═╝   ╚═╝ ╚═════╝
[/#00A99D]"""


def get_animated_logo_frame(frame_num: int) -> str:
    """
    Generate a complete logo frame with animated test tube bubbles.

    Args:
        frame_num: Frame index (0-3, will wrap)

    Returns:
        Complete logo string for this frame
    """
    frame = BUBBLE_FRAMES[frame_num % len(BUBBLE_FRAMES)]

    # Build test tube + TUNER lines
    # Spacing: 12 leading + 5 (tube/bubbles) + 9 gap = 26 chars before TUNER
    tube_section = [
        f"             {frame['float2']}        {TUNER_LINES[0]}",
        f"            {frame['float1']}         {TUNER_LINES[1]}",
        f"            {TUBE_TOP}         {TUNER_LINES[2]}",
        f"            [#33475B]│[/#33475B]{frame['tube']}[#33475B]│[/#33475B]         {TUNER_LINES[3]}",
        f"            {TUBE_LIQ}         {TUNER_LINES[4]}",
        f"            {TUBE_BOT}         {TUNER_LINES[5]}",
        f"            {TUBE_END}",
    ]

    return SYNAPTIC_ART + "\n".join(tube_section)


def get_static_logo() -> str:
    """Get the static (non-animated) logo with test tube."""
    return get_animated_logo_frame(0)


# =============================================================================
# BOX DRAWING CHARACTERS
# =============================================================================

BOX = {
    # Corners (rounded)
    "tl": "╭",
    "tr": "╮",
    "bl": "╰",
    "br": "╯",

    # Lines
    "h": "─",
    "v": "│",

    # Decorative
    "arrow": "►",
    "bullet": "◆",
    "dot": "●",
    "check": "✓",
    "cross": "✗",
    "star": "★",
}


# =============================================================================
# RICH STYLES (conditional on rich availability)
# =============================================================================

try:
    from rich.style import Style

    STYLES = {
        "header": Style(color=COLORS["aqua"], bold=True),
        "accent": Style(color=COLORS["purple"], bold=True),
        "muted": Style(color=COLORS["cello"]),
        "warning": Style(color=COLORS["orange"]),
        "info": Style(color=COLORS["sky"]),
        "success": Style(color=COLORS["aqua"]),
    }
except ImportError:
    # Fallback when rich is not available
    STYLES = {}
