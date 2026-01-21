"""Figure building and composition utilities."""

from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Simple colormaps as RGB tuples (will be multiplied by intensity)
COLORMAPS = {
    "gray": (1.0, 1.0, 1.0),
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "cyan": (0.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
}


def render_channel_to_rgb(
    data: np.ndarray,
    colormap: str,
    contrast_limits: Tuple[float, float],
) -> np.ndarray:
    """Render a single channel to RGB using colormap and contrast limits.

    Args:
        data: 2D numpy array of image data.
        colormap: Name of colormap to apply.
        contrast_limits: (min, max) values for contrast adjustment.

    Returns:
        3D numpy array (H, W, 3) of uint8 RGB values.
    """
    cmin, cmax = contrast_limits

    # Normalize to 0-1 based on contrast limits
    if cmax > cmin:
        normalized = (data.astype(np.float32) - cmin) / (cmax - cmin)
    else:
        normalized = np.zeros_like(data, dtype=np.float32)

    # Clip to 0-1
    normalized = np.clip(normalized, 0, 1)

    # Apply colormap
    color = COLORMAPS.get(colormap, COLORMAPS["gray"])

    rgb = np.zeros((*data.shape, 3), dtype=np.float32)
    rgb[..., 0] = normalized * color[0]
    rgb[..., 1] = normalized * color[1]
    rgb[..., 2] = normalized * color[2]

    # Convert to uint8
    return (rgb * 255).astype(np.uint8)


def create_merge_composite(channels: List[np.ndarray]) -> np.ndarray:
    """Create a merged composite from multiple RGB channel images.

    Args:
        channels: List of RGB numpy arrays (H, W, 3), all same shape.

    Returns:
        Merged RGB numpy array (H, W, 3) with additive blending.
    """
    if not channels:
        raise ValueError("At least one channel required")

    # Use float to avoid overflow during addition
    merged = np.zeros_like(channels[0], dtype=np.float32)

    for ch in channels:
        merged += ch.astype(np.float32)

    # Clip and convert back to uint8
    return np.clip(merged, 0, 255).astype(np.uint8)


def arrange_panels_in_grid(
    panels: List[np.ndarray],
    gap_fraction: float = 0.02,
    background_color: str = "black",
    columns: Optional[int] = None,
) -> np.ndarray:
    """Arrange panel images in a grid layout.

    Args:
        panels: List of RGB numpy arrays, all same shape.
        gap_fraction: Gap between panels as fraction of panel width.
        background_color: Background color for gaps ("black" or "white").
        columns: Number of columns. None = single row.

    Returns:
        RGB numpy array of the complete grid.
    """
    if not panels:
        raise ValueError("At least one panel required")

    panel_height, panel_width = panels[0].shape[:2]
    n_panels = len(panels)

    # Calculate grid dimensions
    if columns is None:
        n_cols = n_panels
        n_rows = 1
    else:
        n_cols = columns
        n_rows = (n_panels + columns - 1) // columns

    # Calculate gap in pixels
    gap_px = int(panel_width * gap_fraction)

    # Calculate total dimensions
    total_width = n_cols * panel_width + (n_cols - 1) * gap_px
    total_height = n_rows * panel_height + (n_rows - 1) * gap_px

    # Create background
    bg_value = 255 if background_color == "white" else 0
    grid = np.full((total_height, total_width, 3), bg_value, dtype=np.uint8)

    # Place panels
    for i, panel in enumerate(panels):
        row = i // n_cols
        col = i % n_cols

        y = row * (panel_height + gap_px)
        x = col * (panel_width + gap_px)

        grid[y:y + panel_height, x:x + panel_width] = panel

    return grid


def add_label_to_panel(
    panel: np.ndarray,
    label: str,
    position: str = "top-left",
    font_size: int = 12,
    color: str = "white",
    padding: int = 5,
) -> np.ndarray:
    """Add a text label to a panel image.

    Args:
        panel: RGB numpy array.
        label: Text to add.
        position: Where to place label ("top-left", "top-center", "bottom-left").
        font_size: Font size in pixels.
        color: Text color.
        padding: Padding from edge in pixels.

    Returns:
        New RGB numpy array with label added.
    """
    # Convert to PIL for text rendering
    img = Image.fromarray(panel)
    draw = ImageDraw.Draw(img)

    # Try to load font
    try:
        font = ImageFont.truetype("Arial", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Get text size
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Calculate position
    if position == "top-left":
        x, y = padding, padding
    elif position == "top-center":
        x = (panel.shape[1] - text_width) // 2
        y = padding
    elif position == "bottom-left":
        x = padding
        y = panel.shape[0] - text_height - padding
    else:
        x, y = padding, padding

    # Draw text
    draw.text((x, y), label, fill=color, font=font)

    return np.array(img)
