"""Matplotlib figure generation for LIF channels."""

from typing import Optional, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from lif_figure.config import Config

if TYPE_CHECKING:
    from lif_figure.reader import SeriesMetadata


# Color mappings for simple named colors
COLOR_MAP = {
    "gray": (1, 1, 1),
    "red": (1, 0, 0),
    "green": (0, 1, 0),
    "blue": (0, 0, 1),
    "cyan": (0, 1, 1),
    "magenta": (1, 0, 1),
    "yellow": (1, 1, 0),
    "lime": (0, 1, 0),
}


def normalize_channel(
    data: np.ndarray,
    percentiles: Optional[tuple[float, float]] = None,
) -> np.ndarray:
    """Normalize channel data to 0-1 range.

    Args:
        data: 2D array of pixel values
        percentiles: Optional (low, high) percentiles for auto-contrast.
                     If provided, clips to these percentiles before normalizing.
                     E.g., (0.1, 99.9) clips extreme 0.1% on each end.

    Returns:
        Normalized array with values in [0, 1]
    """
    data = data.astype(np.float64)

    if percentiles is not None:
        min_val = np.percentile(data, percentiles[0])
        max_val = np.percentile(data, percentiles[1])
    else:
        min_val = data.min()
        max_val = data.max()

    if max_val == min_val:
        return np.zeros_like(data)

    normalized = (data - min_val) / (max_val - min_val)
    return np.clip(normalized, 0, 1)


def apply_colormap(data: np.ndarray, color: str) -> np.ndarray:
    """Apply a color to normalized grayscale data.

    Args:
        data: 2D array with values in [0, 1]
        color: Color name (gray, red, green, blue, cyan, magenta, yellow)

    Returns:
        3D array with shape (H, W, 3) RGB values in [0, 1]
    """
    rgb_mult = COLOR_MAP.get(color.lower(), (1, 1, 1))

    result = np.zeros((*data.shape, 3), dtype=np.float64)
    for i, mult in enumerate(rgb_mult):
        result[..., i] = data * mult

    return result


def create_merge(colored_channels: list[np.ndarray]) -> np.ndarray:
    """Create merged image by additive blending.

    Args:
        colored_channels: List of (H, W, 3) RGB arrays

    Returns:
        Merged (H, W, 3) array, clipped to [0, 1]
    """
    result = np.zeros_like(colored_channels[0])
    for ch in colored_channels:
        result = result + ch

    return np.clip(result, 0, 1)


def calculate_scale_bar(pixel_size_um: float, image_width: int) -> tuple[int, str]:
    """Calculate appropriate scale bar length.

    Args:
        pixel_size_um: Size of one pixel in micrometers
        image_width: Width of image in pixels

    Returns:
        Tuple of (bar_length_pixels, label_text)
    """
    # Target ~10-20% of image width
    target_um = pixel_size_um * image_width * 0.15

    # Nice numbers for scale bars
    nice_values = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    bar_um = min(nice_values, key=lambda x: abs(x - target_um))

    bar_pixels = int(bar_um / pixel_size_um)

    if bar_um >= 1000:
        label = f"{bar_um // 1000} mm"
    else:
        label = f"{bar_um} µm"

    return bar_pixels, label


def build_figure(
    channels: np.ndarray,
    names: list[str],
    config: Config,
    pixel_size_um: Optional[float] = None,
    metadata: Optional["SeriesMetadata"] = None,
    show_metadata: bool = True,
) -> Figure:
    """Build a figure with channel panels and merge.

    Args:
        channels: Array with shape (C, H, W)
        names: List of channel names
        config: Configuration object
        pixel_size_um: Optional pixel size for scale bar
        metadata: Optional series metadata for table
        show_metadata: Whether to show metadata table (default True)

    Returns:
        Matplotlib Figure object
    """
    n_channels = channels.shape[0]
    n_panels = n_channels + 1  # channels + merge

    # Normalize all channels
    normalized = [
        normalize_channel(channels[i], config.auto_contrast_percentiles)
        for i in range(n_channels)
    ]

    # Apply colors
    colors = [config.get_color(names[i], i) for i in range(n_channels)]
    colored = [apply_colormap(normalized[i], colors[i]) for i in range(n_channels)]

    # Create merge
    merge = create_merge(colored)

    # Determine figure size and layout
    should_show_table = show_metadata and metadata is not None
    if should_show_table:
        # Use gridspec for images + table
        fig_height = 5.5  # Extra space for table
        fig = plt.figure(figsize=(4 * n_panels, fig_height), dpi=config.dpi)
        gs = GridSpec(2, n_panels, figure=fig, height_ratios=[4, 1], hspace=0.3)
        axes = [fig.add_subplot(gs[0, i]) for i in range(n_panels)]
        table_ax = fig.add_subplot(gs[1, :])
    else:
        fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), dpi=config.dpi)
        if n_panels == 1:
            axes = [axes]
        table_ax = None

    # Set background
    fig.patch.set_facecolor(config.background)

    # Calculate scale bar if pixel size available
    scale_bar_info = None
    if pixel_size_um is not None:
        scale_bar_info = calculate_scale_bar(pixel_size_um, channels.shape[2])

    # Plot each channel (grayscale)
    for i, (ax, name, norm_ch) in enumerate(zip(axes[:-1], names, normalized)):
        ax.imshow(norm_ch, cmap="gray", vmin=0, vmax=1)
        ax.set_title(name, fontsize=config.font_size, color="white" if config.background == "black" else "black")
        ax.axis("off")
        ax.set_facecolor(config.background)

        # Add scale bar
        if scale_bar_info:
            _add_scale_bar(ax, scale_bar_info, config)

    # Plot merge
    ax_merge = axes[-1]
    ax_merge.imshow(merge)
    ax_merge.set_title("Merge", fontsize=config.font_size, color="white" if config.background == "black" else "black")
    ax_merge.axis("off")
    ax_merge.set_facecolor(config.background)

    if scale_bar_info:
        _add_scale_bar(ax_merge, scale_bar_info, config)

    # Add metadata table
    if should_show_table and table_ax is not None:
        _add_metadata_table(table_ax, names, metadata, config)
    else:
        plt.tight_layout()

    return fig


def _add_metadata_table(
    ax,
    channel_names: list[str],
    metadata: "SeriesMetadata",
    config: Config,
) -> None:
    """Add metadata table below the figure panels."""
    ax.axis("off")
    ax.set_facecolor(config.background)

    # Build table data
    headers = ["Channel", "Laser", "Power", "Detector", "Mode", "Gain", "Contrast"]
    rows = []

    # Get contrast string
    if config.auto_contrast_percentiles:
        contrast_str = f"{config.auto_contrast_percentiles[0]}-{config.auto_contrast_percentiles[1]}%"
    else:
        contrast_str = "min-max"

    # Sort laser wavelengths for matching to channels
    sorted_wavelengths = sorted(metadata.lasers.keys(), key=int)

    for i, name in enumerate(channel_names):
        # Get laser info (match by index)
        if i < len(sorted_wavelengths):
            wl = sorted_wavelengths[i]
            power = metadata.lasers[wl]
            laser_str = f"{wl}nm"
            power_str = f"{power:.0f}%" if power < 100 else f"{power:.1f}%"
        else:
            laser_str = "-"
            power_str = "-"

        # Get detector info
        if i < len(metadata.detectors):
            det = metadata.detectors[i]
            det_name = det.name
            mode_str = det.mode
            gain_str = f"{det.gain:.0f}%"
        else:
            det_name = "-"
            mode_str = "-"
            gain_str = "-"

        rows.append([name, laser_str, power_str, det_name, mode_str, gain_str, contrast_str])

    # Create table
    text_color = "white" if config.background == "black" else "black"
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(config.font_size - 2)
    table.scale(1, 1.5)  # Make rows taller

    # Style cells
    for key, cell in table.get_celld().items():
        cell.set_edgecolor(text_color)
        cell.set_facecolor(config.background)
        cell.set_text_props(color=text_color)
        # Bold header row
        if key[0] == 0:
            cell.set_text_props(weight="bold", color=text_color)


def _add_scale_bar(ax, scale_bar_info: tuple[int, str], config: Config) -> None:
    """Add scale bar to an axis."""
    bar_pixels, label = scale_bar_info

    # Position in bottom-right
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x_start = xlim[1] - bar_pixels - 10
    x_end = xlim[1] - 10
    y_pos = ylim[0] - 10  # Note: y-axis is inverted for images

    bar_color = "white" if config.background == "black" else "black"

    ax.plot([x_start, x_end], [y_pos, y_pos], color=bar_color, linewidth=config.scale_bar_height)
    ax.text(
        (x_start + x_end) / 2, y_pos + 15,
        label,
        ha="center", va="top",
        fontsize=config.font_size - 2,
        color=bar_color,
    )
