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


def _get_facecolor(background: str) -> str:
    """Convert background color to matplotlib facecolor."""
    return "none" if background.lower() == "transparent" else background


def normalize_channel(
    data: np.ndarray,
    percentiles: Optional[tuple[float, float]] = None,
    norm_range: Optional[tuple[float, float]] = None,
) -> np.ndarray:
    """Normalize channel data to 0-1 range.

    Args:
        data: 2D array of pixel values
        percentiles: Optional (low, high) percentiles for auto-contrast.
                     If provided, clips to these percentiles before normalizing.
                     E.g., (0.1, 99.9) clips extreme 0.1% on each end.
                     Ignored if norm_range is provided.
        norm_range: Optional pre-computed (min, max) range for normalization.
                    Use this for consistent normalization across Z-stacks.

    Returns:
        Normalized array with values in [0, 1]
    """
    data = data.astype(np.float64)

    if norm_range is not None:
        min_val, max_val = norm_range
    elif percentiles is not None:
        min_val = np.percentile(data, percentiles[0])
        max_val = np.percentile(data, percentiles[1])
    else:
        min_val = data.min()
        max_val = data.max()

    if max_val == min_val:
        return np.zeros_like(data)

    normalized = (data - min_val) / (max_val - min_val)
    return np.clip(normalized, 0, 1)


def compute_normalization_ranges(
    data: np.ndarray,
    percentiles: Optional[tuple[float, float]] = None,
) -> list[tuple[float, float]]:
    """Compute normalization ranges across a Z-stack for each channel.

    Args:
        data: Array with shape (Z, C, H, W)
        percentiles: Optional (low, high) percentiles for auto-contrast.

    Returns:
        List of (min, max) tuples, one per channel.
    """
    n_channels = data.shape[1]
    ranges = []

    for c in range(n_channels):
        channel_data = data[:, c, :, :]  # All Z slices for this channel
        if percentiles is not None:
            min_val = np.percentile(channel_data, percentiles[0])
            max_val = np.percentile(channel_data, percentiles[1])
        else:
            min_val = channel_data.min()
            max_val = channel_data.max()
        ranges.append((float(min_val), float(max_val)))

    return ranges


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
    normalization_ranges: Optional[list[tuple[float, float]]] = None,
) -> Figure:
    """Build a figure with channel panels and merge.

    Args:
        channels: Array with shape (C, H, W)
        names: List of channel names
        config: Configuration object
        pixel_size_um: Optional pixel size for scale bar
        metadata: Optional series metadata for table
        show_metadata: Whether to show metadata table (default True)
        normalization_ranges: Optional pre-computed (min, max) per channel.
                              Use for consistent normalization across Z-stacks.

    Returns:
        Matplotlib Figure object
    """
    n_channels = channels.shape[0]
    n_panels = n_channels + 1  # channels + merge

    # Normalize all channels
    normalized = []
    for i in range(n_channels):
        norm_range = normalization_ranges[i] if normalization_ranges else None
        normalized.append(
            normalize_channel(channels[i], config.auto_contrast_percentiles, norm_range)
        )

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
    fig.patch.set_facecolor(_get_facecolor(config.background))

    # Calculate scale bar if pixel size available
    scale_bar_info = None
    if pixel_size_um is not None:
        scale_bar_info = calculate_scale_bar(pixel_size_um, channels.shape[2])

    # Plot each channel (grayscale)
    for i, (ax, name, norm_ch) in enumerate(zip(axes[:-1], names, normalized)):
        ax.imshow(norm_ch, cmap="gray", vmin=0, vmax=1)
        ax.set_title(name, fontsize=config.font_size, color="white" if config.background == "black" else "black")
        ax.axis("off")
        ax.set_facecolor(_get_facecolor(config.background))

        # Add scale bar
        if scale_bar_info:
            _add_scale_bar(ax, scale_bar_info, config)

    # Plot merge
    ax_merge = axes[-1]
    ax_merge.imshow(merge)
    ax_merge.set_title("Merge", fontsize=config.font_size, color="white" if config.background == "black" else "black")
    ax_merge.axis("off")
    ax_merge.set_facecolor(_get_facecolor(config.background))

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
    ax.set_facecolor(_get_facecolor(config.background))

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
        cell.set_facecolor(_get_facecolor(config.background))
        cell.set_text_props(color=text_color)
        # Bold header row
        if key[0] == 0:
            cell.set_text_props(weight="bold", color=text_color)


def _add_scale_bar(ax, scale_bar_info: tuple[int, str], config: Config) -> None:
    """Add scale bar to an axis."""
    bar_pixels, label = scale_bar_info

    # Position in bottom-right, inside the image
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # For imshow, y-axis is inverted: ylim[0] is bottom (large), ylim[1] is top (small)
    # Use margin proportional to image size (3% of smaller dimension)
    img_width = xlim[1] - xlim[0]
    img_height = ylim[0] - ylim[1]  # ylim inverted for images
    margin = max(10, int(min(img_width, img_height) * 0.03))
    x_end = xlim[1] - margin
    x_start = x_end - bar_pixels
    y_bar = ylim[0] - margin  # Position bar inside from bottom

    bar_color = "white"  # Always white for visibility on microscopy images

    ax.plot([x_start, x_end], [y_bar, y_bar], color=bar_color, linewidth=config.scale_bar_height)
    # TODO: Consider a more robust alignment strategy (e.g., compute text bbox and adjust)
    ax.annotate(
        label,
        xy=(x_end, y_bar),  # Anchor at bar's right edge
        xytext=(0, 6),  # Offset upward in points
        textcoords="offset points",
        ha="right", va="bottom",
        fontsize=config.font_size - 2,
        color=bar_color,
        annotation_clip=False,
    )


def build_rows_figure(
    data: np.ndarray,
    names: list[str],
    config: Config,
    pixel_size_um: Optional[float] = None,
    z_pixel_size_um: Optional[float] = None,
    metadata: Optional["SeriesMetadata"] = None,
    show_metadata: bool = True,
    normalization_ranges: Optional[list[tuple[float, float]]] = None,
) -> Figure:
    """Build a figure with Z-slices as rows.

    Args:
        data: Array with shape (Z, C, H, W)
        names: List of channel names
        config: Configuration object
        pixel_size_um: Optional XY pixel size for scale bar
        z_pixel_size_um: Optional Z step size for position labels
        metadata: Optional series metadata for table
        show_metadata: Whether to show metadata table (default True)
        normalization_ranges: Optional pre-computed (min, max) per channel.

    Returns:
        Matplotlib Figure object
    """
    n_z = data.shape[0]
    n_channels = data.shape[1]
    n_cols = n_channels + 1  # channels + merge

    text_color = "white" if config.background == "black" else "black"

    # Figure layout
    panel_size = 2  # inches per panel
    should_show_table = show_metadata and metadata is not None
    header_ratio = 0.08  # For column titles

    if should_show_table:
        # Add extra height for header row and metadata table
        table_height_inches = 1.5  # Fixed height for table area
        fig_height = (n_z + header_ratio) * panel_size + table_height_inches + 0.5
        fig = plt.figure(figsize=(n_cols * panel_size + 1.2, fig_height), dpi=config.dpi)
        # Height ratios: header, Z rows (no table in gridspec)
        height_ratios = [header_ratio] + [1] * n_z
        gs = GridSpec(
            n_z + 1, n_cols + 1,  # +1 header, +1 col for Z labels
            figure=fig,
            height_ratios=height_ratios,
            width_ratios=[0.25] + [1] * n_cols,
            hspace=0.02,
            wspace=0.02,
            bottom=table_height_inches / fig_height + 0.02,  # Reserve space at bottom
        )
        # Create table axes separately at the bottom
        table_ax = fig.add_axes([0.15, 0.02, 0.8, table_height_inches / fig_height - 0.02])
    else:
        fig_height = (n_z + header_ratio) * panel_size
        fig = plt.figure(figsize=(n_cols * panel_size + 1.2, fig_height), dpi=config.dpi)
        height_ratios = [header_ratio] + [1] * n_z
        gs = GridSpec(
            n_z + 1, n_cols + 1,  # +1 row for header, +1 col for Z labels
            figure=fig,
            height_ratios=height_ratios,
            width_ratios=[0.25] + [1] * n_cols,
            hspace=0.02,
            wspace=0.02,
        )
        table_ax = None

    # Add column headers in the header row
    for c, name in enumerate(names):
        header_ax = fig.add_subplot(gs[0, c + 1])
        header_ax.axis("off")
        header_ax.set_facecolor(_get_facecolor(config.background))
        header_ax.text(
            0.5, 0.5, name,
            ha="center", va="center",
            fontsize=config.font_size,
            color=text_color,
            transform=header_ax.transAxes,
        )
    # Merge header
    merge_header_ax = fig.add_subplot(gs[0, n_cols])
    merge_header_ax.axis("off")
    merge_header_ax.set_facecolor(_get_facecolor(config.background))
    merge_header_ax.text(
        0.5, 0.5, "Merge",
        ha="center", va="center",
        fontsize=config.font_size,
        color=text_color,
        transform=merge_header_ax.transAxes,
    )

    fig.patch.set_facecolor(_get_facecolor(config.background))

    # Calculate scale bar
    scale_bar_info = None
    if pixel_size_um is not None:
        scale_bar_info = calculate_scale_bar(pixel_size_um, data.shape[3])

    # Get colors for channels
    colors = [config.get_color(names[i], i) for i in range(n_channels)]

    # Process each Z-slice (row) - offset by 1 for header row
    for z in range(n_z):
        row_idx = z + 1  # Offset for header row

        # Normalize channels for this slice
        normalized = []
        for c in range(n_channels):
            norm_range = normalization_ranges[c] if normalization_ranges else None
            normalized.append(
                normalize_channel(data[z, c], config.auto_contrast_percentiles, norm_range)
            )

        # Apply colors and create merge
        colored = [apply_colormap(normalized[c], colors[c]) for c in range(n_channels)]
        merge = create_merge(colored)

        # Z-position label (leftmost column)
        z_label_ax = fig.add_subplot(gs[row_idx, 0])
        z_label_ax.axis("off")
        z_label_ax.set_facecolor(_get_facecolor(config.background))

        # Calculate Z position in microns if available
        if z_pixel_size_um is not None:
            z_pos = z * z_pixel_size_um
            z_text = f"z={z_pos:.1f}µm"
        else:
            z_text = f"z={z}"

        z_label_ax.text(
            0.9, 0.5, z_text,
            ha="right", va="center",
            fontsize=config.font_size - 2,
            color=text_color,
            transform=z_label_ax.transAxes,
        )

        # Plot channels
        for c in range(n_channels):
            ax = fig.add_subplot(gs[row_idx, c + 1])
            ax.imshow(normalized[c], cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            ax.set_facecolor(_get_facecolor(config.background))

        # Plot merge
        ax_merge = fig.add_subplot(gs[row_idx, n_cols])
        ax_merge.imshow(merge)
        ax_merge.axis("off")
        ax_merge.set_facecolor(_get_facecolor(config.background))

        # Scale bar on bottom-right merge panel only
        if z == n_z - 1 and scale_bar_info:
            _add_scale_bar(ax_merge, scale_bar_info, config)

    # Add metadata table
    if should_show_table and table_ax is not None:
        _add_metadata_table(table_ax, names, metadata, config)

    return fig
