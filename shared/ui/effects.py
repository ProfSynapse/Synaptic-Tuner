"""
Custom Asciimatics Effects for Synaptic Tuner

Contains visual effect classes:
- BrandBubbles: Full-screen chaotic bubbles using brand colors
- SparkBurst: Burst of sparks for training start
"""

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from asciimatics.screen import Screen

# =============================================================================
# BRAND COLORS for 256-color terminals
# =============================================================================
# These are the closest 256-color palette matches to brand colors
BRAND_AQUA = 37       # #00A99D -> closest is 37 (dark cyan)
BRAND_PURPLE = 127    # #93278F -> closest is 127 (magenta)
BRAND_SKY = 39        # #29ABE2 -> closest is 39 (light blue)
BRAND_ORANGE = 208    # #F7931E -> closest is 208 (orange)
BRAND_CREAM = 230     # #FBF7F1 -> closest is 230 (light yellow/cream)
BRAND_CELLO = 59      # #33475B -> closest is 59 (dark blue-gray)


def _get_effect_classes():
    """
    Import and return effect classes.
    Wrapped in function to handle asciimatics not being installed.
    """
    try:
        from asciimatics.screen import Screen
        from asciimatics.effects import Effect
    except ImportError:
        return None, None

    class BrandBubbles(Effect):
        """
        Full-screen chaotic bubbles using brand colors.
        More random, more alive!
        """

        # Bubble characters - varied sizes and styles
        CHARS_SMALL = ["·", "∘", "°", "◦", "•"]
        CHARS_MED = ["○", "●", "◯", "◉"]
        CHARS_LARGE = ["⬤", "◎", "☉", "✦", "✧", "★", "☆"]
        CHARS_SPARKLE = ["✨", "⋆", "✴", "✳", "+", "*"]

        def __init__(self, screen, count=40, **kwargs):
            super().__init__(screen, **kwargs)
            self._count = count
            self._bubbles = []
            self._brand_colors = [
                BRAND_AQUA,    # #00A99D - Persian Green
                BRAND_SKY,     # #29ABE2 - Summer Sky
                BRAND_PURPLE,  # #93278F - Dark Purple
                BRAND_ORANGE,  # #F7931E - Carrot Orange
                BRAND_CREAM,   # #FBF7F1 - Floral White
            ]

        def _random_char(self):
            """Pick a random character with weighted distribution."""
            roll = random.random()
            if roll < 0.4:
                return random.choice(self.CHARS_SMALL)
            elif roll < 0.7:
                return random.choice(self.CHARS_MED)
            elif roll < 0.9:
                return random.choice(self.CHARS_LARGE)
            else:
                return random.choice(self.CHARS_SPARKLE)

        def _create_bubble(self, y=None):
            """Create a bubble with random properties."""
            if y is None:
                y = float(random.randint(0, self._screen.height - 1))

            # Random movement direction (mostly up, but some sideways)
            direction = random.uniform(-0.3, 0.3)  # Slight angle

            return {
                'x': float(random.randint(0, self._screen.width - 1)),
                'y': y,
                'char': self._random_char(),
                'speed': random.uniform(0.1, 0.8),  # More speed variation
                'drift': direction,  # Horizontal drift
                'wobble_amp': random.uniform(0.1, 0.6),  # Wobble amplitude
                'wobble_freq': random.uniform(0.1, 0.3),  # Wobble frequency
                'phase': random.uniform(0, 6.28),  # Random phase offset
                'colour': random.choice(self._brand_colors),
                'bold': random.random() < 0.4,
                'lifetime': random.randint(20, 200),  # Random lifespan
                'age': 0,
                'twinkle': random.random() < 0.2,  # Some bubbles twinkle
            }

        def reset(self):
            self._bubbles = []
            for _ in range(self._count):
                self._bubbles.append(self._create_bubble())

        def _update(self, frame_no):
            import math

            new_bubbles = []
            for bubble in self._bubbles:
                bubble['age'] += 1

                # Skip drawing sometimes for twinkle effect
                if bubble['twinkle'] and random.random() < 0.3:
                    pass  # Don't draw this frame
                else:
                    # Draw bubble
                    x, y = int(bubble['x']), int(bubble['y'])
                    if 0 <= x < self._screen.width and 0 <= y < self._screen.height:
                        attr = Screen.A_BOLD if bubble['bold'] else Screen.A_NORMAL
                        self._screen.print_at(
                            bubble['char'], x, y,
                            colour=bubble['colour'],
                            attr=attr
                        )

                # Sinusoidal wobble for more organic movement
                wobble = math.sin(bubble['age'] * bubble['wobble_freq'] + bubble['phase'])
                wobble *= bubble['wobble_amp']

                # Move with drift and wobble
                bubble['y'] -= bubble['speed']
                bubble['x'] += bubble['drift'] + wobble

                # Random speed changes
                if random.random() < 0.05:
                    bubble['speed'] += random.uniform(-0.1, 0.1)
                    bubble['speed'] = max(0.05, min(1.0, bubble['speed']))

                # Random color changes (rare)
                if random.random() < 0.02:
                    bubble['colour'] = random.choice(self._brand_colors)

                # Random character changes (rare)
                if random.random() < 0.01:
                    bubble['char'] = self._random_char()

                # Check if bubble should respawn
                should_respawn = (
                    bubble['y'] < -2 or  # Off top
                    bubble['x'] < -2 or  # Off left
                    bubble['x'] >= self._screen.width + 2 or  # Off right
                    bubble['age'] > bubble['lifetime']  # Expired
                )

                if should_respawn:
                    # Respawn at bottom with new random properties
                    new_bubbles.append(self._create_bubble(y=float(self._screen.height + random.randint(0, 5))))
                else:
                    new_bubbles.append(bubble)

            self._bubbles = new_bubbles

            # Occasionally spawn extra bubbles for bursts
            if random.random() < 0.1 and len(self._bubbles) < self._count + 10:
                self._bubbles.append(self._create_bubble(y=float(self._screen.height)))

        @property
        def stop_frame(self):
            return self._stop_frame

        @property
        def frame_update_count(self):
            return 1

    class SparkBurst(Effect):
        """
        A burst of sparks emanating from a point - for training start.
        """

        SPARK_CHARS = ["*", "✦", "✧", "⋆", "+", "·"]

        def __init__(self, screen, x, y, count=30, **kwargs):
            super().__init__(screen, **kwargs)
            self._x = x
            self._y = y
            self._count = count
            self._sparks = []

        def reset(self):
            import math
            self._sparks = []
            for _ in range(self._count):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(0.3, 1.5)
                self._sparks.append({
                    'x': float(self._x),
                    'y': float(self._y),
                    'dx': math.cos(angle) * speed,
                    'dy': math.sin(angle) * speed * 0.5,  # Slower vertical
                    'char': random.choice(self.SPARK_CHARS),
                    'life': random.randint(10, 30),
                    'age': 0,
                    'colour': random.choice([
                        Screen.COLOUR_YELLOW,
                        Screen.COLOUR_RED,
                        Screen.COLOUR_WHITE,
                    ])
                })

        def _update(self, frame_no):
            for spark in self._sparks:
                if spark['age'] < spark['life']:
                    x, y = int(spark['x']), int(spark['y'])
                    if 0 <= x < self._screen.width and 0 <= y < self._screen.height:
                        # Fade color as spark ages
                        colour = spark['colour']
                        if spark['age'] > spark['life'] * 0.7:
                            colour = Screen.COLOUR_RED
                        self._screen.print_at(spark['char'], x, y, colour=colour)

                    # Move spark
                    spark['x'] += spark['dx']
                    spark['y'] += spark['dy']
                    spark['dy'] += 0.05  # Gravity
                    spark['age'] += 1

        @property
        def stop_frame(self):
            return self._stop_frame

        @property
        def frame_update_count(self):
            return 1

    class ProgressLoader(Effect):
        """
        Animated progress bar that fills over the scene duration.
        Shows users the animation is progressing, not stuck.
        """

        def __init__(self, screen, y, duration, width=40, label="Loading", **kwargs):
            super().__init__(screen, **kwargs)
            self._y = y
            self._duration = duration
            self._width = width
            self._label = label
            self._x = (screen.width - width - len(label) - 5) // 2

        def reset(self):
            pass

        def _update(self, frame_no):
            # Calculate progress (0.0 to 1.0)
            progress = min(1.0, frame_no / max(1, self._duration - 1))
            filled = int(progress * self._width)
            empty = self._width - filled

            # Build the bar
            bar = "█" * filled + "░" * empty
            pct = int(progress * 100)

            # Draw label
            self._screen.print_at(
                f"{self._label} ",
                self._x,
                self._y,
                colour=BRAND_CELLO,
            )

            # Draw bar with brand colors
            label_len = len(self._label) + 1
            self._screen.print_at(
                "[",
                self._x + label_len,
                self._y,
                colour=BRAND_CELLO,
            )
            self._screen.print_at(
                bar[:filled],
                self._x + label_len + 1,
                self._y,
                colour=BRAND_AQUA,
                attr=Screen.A_BOLD,
            )
            self._screen.print_at(
                bar[filled:],
                self._x + label_len + 1 + filled,
                self._y,
                colour=BRAND_CELLO,
            )
            self._screen.print_at(
                f"] {pct:3d}%",
                self._x + label_len + 1 + self._width,
                self._y,
                colour=BRAND_CELLO,
            )

        @property
        def stop_frame(self):
            return self._stop_frame

        @property
        def frame_update_count(self):
            return 1

    return BrandBubbles, SparkBurst, ProgressLoader


# Export the classes (will be None if asciimatics not available)
_classes = _get_effect_classes()
if _classes:
    BrandBubbles, SparkBurst, ProgressLoader = _classes
else:
    BrandBubbles, SparkBurst, ProgressLoader = None, None, None
