"""Scale bar rendering utilities."""

import math
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def calculate_nice_scale_bar_length(
    image_width_um: float,
    target_fraction: float = 0.2,
) -> float:
    """Calculate a nice round scale bar length.

    Args:
        image_width_um: Width of image in micrometers.
        target_fraction: Target scale bar length as fraction of image width.

    Returns:
        Nice round number for scale bar length in micrometers.
    """
    target_length = image_width_um * target_fraction

    # Find order of magnitude
    if target_length <= 0:
        return 1.0

    magnitude = 10 ** math.floor(math.log10(target_length))
    normalized = target_length / magnitude

    # Pick nearest nice number: 1, 2, 5, 10
    nice_numbers = [1, 2, 5, 10]
    nice = min(nice_numbers, key=lambda x: abs(x - normalized))

    return nice * magnitude


def render_scale_bar(
    length_um: float,
    pixel_size_um: float,
    color: str = "white",
    font_size: int = 10,
    bar_height: int = 4,
    padding: int = 5,
) -> Image.Image:
    """Render a scale bar with label as a PIL Image.

    Args:
        length_um: Length of scale bar in micrometers.
        pixel_size_um: Size of one pixel in micrometers.
        color: Color of bar and text.
        font_size: Font size for label.
        bar_height: Height of the bar in pixels.
        padding: Padding around bar and text.

    Returns:
        RGBA PIL Image containing the scale bar and label.
    """
    bar_width_px = int(length_um / pixel_size_um)

    # Format label
    if length_um >= 1000:
        label = f"{length_um / 1000:.0f} mm"
    elif length_um >= 1:
        label = f"{length_um:.0f} µm"
    else:
        label = f"{length_um * 1000:.0f} nm"

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("Arial", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Measure text
    dummy_img = Image.new("RGBA", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    text_bbox = dummy_draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Calculate image size
    img_width = max(bar_width_px, text_width) + 2 * padding
    img_height = bar_height + text_height + 3 * padding

    # Create image
    img = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw bar (centered horizontally)
    bar_x = (img_width - bar_width_px) // 2
    bar_y = padding
    draw.rectangle(
        [bar_x, bar_y, bar_x + bar_width_px, bar_y + bar_height],
        fill=color,
    )

    # Draw text (centered below bar)
    text_x = (img_width - text_width) // 2
    text_y = bar_y + bar_height + padding
    draw.text((text_x, text_y), label, fill=color, font=font)

    return img
