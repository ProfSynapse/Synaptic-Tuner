"""
Custom Asciimatics Effects for Synaptic Tuner

Contains visual effect classes:
- BubblingFlask: Round-bottom flask with animated liquid
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
        return None, None, None

    class BubblingFlask(Effect):
        r"""
        A large Florence flask (round-bottom) rendered with # characters.

        Uses mathematically-generated circle shape compensating for terminal
        character aspect ratio (chars are ~2x taller than wide).

        Features:
        - Cream/white # outline for glass
        - Animated aqua/sky # interior for liquid with wave patterns
        - Internal bubbles rising through liquid (sky blue)
        - Bubbles escaping from neck
        """

        # Flask dimensions (Option B - radius 15 for full size)
        RADIUS = 15
        NECK_WIDTH = 5
        NECK_ROWS = 2
        LIP_WIDTH = 7
        ASPECT_RATIO = 2.0

        def __init__(self, screen, x, y, **kwargs):
            super().__init__(screen, **kwargs)
            self._x = x
            self._y = y
            self._escaped_bubbles = []
            self._internal_bubbles = []
            self._frame = 0

            # Pre-generate flask shape (list of (left_edge, right_edge) tuples)
            self._flask_rows = self._generate_flask_shape()
            self._liquid_start_row = 5  # Row where liquid begins (0-indexed from flask top)
            self._flask_height = len(self._flask_rows)

        def _generate_flask_shape(self):
            """
            Generate flask shape using circle equation.
            Returns list of (left_edge, right_edge) for each row, relative to center.
            """
            import math
            rows = []
            center_x = self.RADIUS

            def make_row_data(left, right):
                return (left, right)

            def center_row_data(width):
                left = center_x - width // 2
                right = center_x + width // 2 - (1 if width % 2 == 0 else 0)
                return make_row_data(left, right)

            # 1. Lip
            rows.append(center_row_data(self.LIP_WIDTH))

            # 2. Neck
            for _ in range(self.NECK_ROWS):
                rows.append(center_row_data(self.NECK_WIDTH))

            # 3. Circle body
            num_rows = int(self.RADIUS / self.ASPECT_RATIO)
            body_rows = []

            for row in range(num_rows * 2 + 1):
                y_normalized = (row - num_rows) / num_rows
                if abs(y_normalized) <= 1:
                    x_normalized = math.sqrt(1 - y_normalized ** 2)
                    x_offset = int(x_normalized * self.RADIUS)
                    left_edge = center_x - x_offset
                    right_edge = center_x + x_offset
                    width = right_edge - left_edge + 1
                    body_rows.append((left_edge, right_edge, width))

            # Filter tiny rows
            if body_rows:
                max_w = max(r[2] for r in body_rows)
                min_acceptable = max(self.NECK_WIDTH, int(max_w * 0.4))

                for left, right, width in body_rows:
                    if width >= min_acceptable:
                        rows.append((left, right))

            return rows

        def reset(self):
            self._escaped_bubbles = []
            self._internal_bubbles = []
            self._frame = 0

        def _get_liquid_color(self, row, col, frame):
            """Get color for liquid cell with wave animation using brand colors."""
            wave_offset = (frame // 2) % 8

            # Wave pattern moves diagonally through liquid
            heat = ((col + row * 2 + wave_offset) % 5)

            # Use brand colors: Aqua (#00A99D) and Sky (#29ABE2)
            if heat == 0:
                return BRAND_AQUA, Screen.A_BOLD
            elif heat == 1:
                return BRAND_AQUA, Screen.A_NORMAL
            elif heat == 2:
                return BRAND_SKY, Screen.A_BOLD
            elif heat == 3:
                return BRAND_SKY, Screen.A_NORMAL
            else:
                return BRAND_CREAM, Screen.A_NORMAL

        def _update(self, frame_no):
            # Use brand cream color for glass outline
            GLASS = BRAND_CREAM
            x = self._x
            y = self._y
            frame = self._frame

            # Track liquid region for bubble spawning
            liquid_left = None
            liquid_right = None
            liquid_bottom_y = None

            # =================================================================
            # DRAW FLASK ROW BY ROW
            # =================================================================
            for row_idx, (left, right) in enumerate(self._flask_rows):
                screen_y = y + row_idx
                width = right - left + 1
                is_liquid_row = row_idx >= self._liquid_start_row

                # Draw each position in this row
                for col in range(width):
                    screen_x = x + left + col
                    is_edge = (col == 0 or col == width - 1)

                    if is_edge:
                        # Glass outline - brand cream/white
                        self._screen.print_at("#", screen_x, screen_y,
                                              colour=GLASS, attr=Screen.A_BOLD)
                    elif is_liquid_row and width > 2:
                        # Interior liquid - animated aqua/sky brand colors
                        color, attr = self._get_liquid_color(row_idx, col, frame)
                        self._screen.print_at("#", screen_x, screen_y,
                                              colour=color, attr=attr)
                    # else: empty air inside (no character drawn)

                # Track liquid boundaries for bubble spawning
                if is_liquid_row and width > 2:
                    if liquid_left is None:
                        liquid_left = x + left + 1
                        liquid_right = x + right - 1
                    liquid_bottom_y = screen_y

            # =================================================================
            # INTERNAL BUBBLES (Rising through liquid)
            # =================================================================
            if liquid_left is not None and liquid_bottom_y is not None:
                # Spawn bubbles at bottom of liquid
                if frame % 4 == 0 and random.random() < 0.7:
                    bx = random.randint(liquid_left + 2, liquid_right - 2)
                    by = float(liquid_bottom_y)
                    self._internal_bubbles.append([bx, by])

            # Liquid surface Y (where bubbles escape)
            liquid_surface_y = y + self._liquid_start_row

            # Draw and update internal bubbles
            new_internal = []
            for bubble in self._internal_bubbles:
                bx, by = bubble
                screen_y = int(by)

                # Draw as bright cream/white # (stands out against liquid)
                if liquid_surface_y <= screen_y <= (liquid_bottom_y or screen_y):
                    self._screen.print_at("#", int(bx), screen_y,
                                          colour=BRAND_CREAM, attr=Screen.A_BOLD)

                # Move up with wobble
                bubble[1] -= 0.2
                bubble[0] += random.choice([-0.15, 0, 0, 0.15])

                # Transfer to escaped when reaching surface
                if bubble[1] <= liquid_surface_y:
                    # Start escaped bubble at neck position
                    neck_x = x + self.RADIUS  # Center of neck
                    self._escaped_bubbles.append([float(neck_x), float(y + 2), 0])
                elif bubble[1] >= liquid_surface_y:
                    new_internal.append(bubble)

            self._internal_bubbles = new_internal

            # =================================================================
            # ESCAPED BUBBLES (Rising out of flask neck)
            # =================================================================
            new_escaped = []
            for bubble in self._escaped_bubbles:
                bx, by, age = bubble
                screen_y = int(by)

                # Draw bubble above flask - use sky blue brand color
                if 0 <= screen_y < self._screen.height and screen_y < y:
                    if age % 3 == 0:
                        self._screen.print_at("#", int(bx), screen_y,
                                              colour=BRAND_CREAM, attr=Screen.A_BOLD)
                    elif age % 3 == 1:
                        self._screen.print_at("#", int(bx), screen_y,
                                              colour=BRAND_SKY, attr=Screen.A_BOLD)
                    else:
                        self._screen.print_at("#", int(bx), screen_y,
                                              colour=BRAND_AQUA, attr=Screen.A_NORMAL)

                # Move up with wobble
                bubble[1] -= 0.3
                bubble[0] += random.choice([-0.2, -0.1, 0, 0.1, 0.2])
                bubble[2] += 1

                # Keep bubbles that haven't risen too far
                if bubble[1] > y - 8 and bubble[2] < 40:
                    new_escaped.append(bubble)

            self._escaped_bubbles = new_escaped
            self._frame += 1

        @property
        def stop_frame(self):
            return self._stop_frame

        @property
        def frame_update_count(self):
            return 1

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

    return BubblingFlask, BrandBubbles, SparkBurst


# Export the classes (will be None if asciimatics not available)
BubblingFlask, BrandBubbles, SparkBurst = _get_effect_classes() or (None, None, None)
