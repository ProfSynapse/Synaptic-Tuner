#!/usr/bin/env python3
"""
Flask Shape Generator

Generates a visually round flask shape, accounting for terminal character aspect ratio.
Terminal chars are ~2x taller than wide, so we compensate.
"""

import math


def generate_round_flask(
    radius: int = 10,
    neck_width: int = 5,
    neck_rows: int = 3,
    lip_width: int = 7,
    aspect_ratio: float = 2.0,  # Terminal char height/width ratio
) -> list[str]:
    """
    Generate a flask that LOOKS round on screen.

    Calculates left and right edges directly from circle equation
    to ensure perfect symmetry.

    Args:
        radius: Visual radius of the sphere
        neck_width: Width of the neck
        neck_rows: Number of neck rows
        lip_width: Width of the lip
        aspect_ratio: Character height/width ratio (typically 1.8-2.2)
    """
    lines = []

    # The actual width will be 2*radius
    # The height will be radius/aspect_ratio (compressed to look round)
    max_width = radius * 2
    num_rows = int(radius / aspect_ratio)
    center_x = radius  # Center point

    def make_row(left: int, right: int) -> str:
        """Create a row with # from left to right position."""
        width = right - left + 1
        return " " * left + "#" * width

    def center_row(width: int) -> str:
        """Center a row of given width."""
        left = center_x - width // 2
        right = center_x + width // 2 - (1 if width % 2 == 0 else 0)
        return make_row(left, right)

    # 1. Lip (centered)
    lines.append(center_row(lip_width))

    # 2. Neck (centered)
    for _ in range(neck_rows):
        lines.append(center_row(neck_width))

    # 3. Circle body - calculate left/right edges directly from circle
    body_rows = []
    for row in range(num_rows * 2 + 1):
        # Convert row to Y coordinate on unit circle (-1 to 1)
        y_normalized = (row - num_rows) / num_rows

        # Circle equation: x² + y² = 1, so x = sqrt(1 - y²)
        if abs(y_normalized) <= 1:
            x_normalized = math.sqrt(1 - y_normalized**2)

            # Calculate left and right edges from center
            # Both use same x value = perfect symmetry
            x_offset = int(x_normalized * radius)
            left_edge = center_x - x_offset
            right_edge = center_x + x_offset

            width = right_edge - left_edge + 1
            body_rows.append((left_edge, right_edge, width))

    # Filter out tiny rows at edges
    max_w = max(r[2] for r in body_rows)
    min_acceptable = max(neck_width, int(max_w * 0.4))

    for left, right, width in body_rows:
        if width >= min_acceptable:
            lines.append(make_row(left, right))

    return lines


def print_flask(lines: list[str], show_annotations: bool = True):
    """Print the flask with optional annotations."""
    if not lines:
        print("No lines to print")
        return

    max_len = max(len(line.rstrip()) for line in lines)

    for i, line in enumerate(lines):
        width = len(line.strip())
        if show_annotations:
            print(f"{line:<{max_len + 2}} <- row {i}: width {width}")
        else:
            print(line)


if __name__ == "__main__":
    print("=" * 60)
    print("FLASK - Visually Round (aspect ratio corrected)")
    print("=" * 60)
    print()

    # Test different radii
    for r in [8, 10, 12, 15]:
        print(f"--- Radius {r} (width={r*2}) ---")
        print()

        lines = generate_round_flask(
            radius=r,
            neck_width=5,
            neck_rows=2,
            lip_width=7,
            aspect_ratio=2.0,
        )

        print_flask(lines, show_annotations=False)
        print()
        print(f"Rows: {len(lines)}, Max width: {max(len(l.strip()) for l in lines)}")
        print()
        print()
