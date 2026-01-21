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
