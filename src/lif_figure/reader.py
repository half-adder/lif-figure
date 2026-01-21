"""LIF file reading and Z-stack processing."""

from pathlib import Path
from typing import Optional

import numpy as np
from readlif.reader import LifFile


def parse_zstack_mode(mode_str: str) -> tuple[str, Optional[tuple[int, int]]]:
    """Parse Z-stack mode string.

    Args:
        mode_str: Mode string like 'max', 'max:5-15', 'frames', 'rows'

    Returns:
        Tuple of (mode, z_range) where z_range is None or (start, end)
    """
    if mode_str.startswith("max:"):
        range_part = mode_str[4:]
        start, end = range_part.split("-")
        return ("max", (int(start), int(end)))

    return (mode_str, None)


def apply_max_projection(
    data: np.ndarray,
    z_range: Optional[tuple[int, int]] = None
) -> np.ndarray:
    """Apply max projection along Z axis.

    Args:
        data: Array with shape (Z, C, H, W)
        z_range: Optional (start, end) indices for Z range

    Returns:
        Array with shape (C, H, W) after max projection
    """
    if z_range is not None:
        start, end = z_range
        data = data[start:end + 1]

    return np.max(data, axis=0)


def list_series(path: Path) -> list[str]:
    """List all series names in a LIF file.

    Args:
        path: Path to LIF file

    Returns:
        List of series names
    """
    lif = LifFile(str(path))
    return [info.get("name", f"Series {i}") for i, info in enumerate(lif.image_list)]


def read_series(
    path: Path,
    series_name: str,
    zstack_mode: str = "max",
) -> tuple[np.ndarray, Optional[float]]:
    """Read a single series from a LIF file.

    Args:
        path: Path to LIF file
        series_name: Name of series to read
        zstack_mode: Z-stack handling mode

    Returns:
        Tuple of (channels_array, pixel_size_um)
        channels_array has shape (C, H, W) for max projection
        or (Z, C, H, W) for frames/rows modes
    """
    lif = LifFile(str(path))
    mode, z_range = parse_zstack_mode(zstack_mode)

    # Find the series by name
    series_idx = None
    for i, info in enumerate(lif.image_list):
        if info.get("name") == series_name:
            series_idx = i
            break

    if series_idx is None:
        raise ValueError(f"Series '{series_name}' not found in {path}")

    info = lif.image_list[series_idx]
    image = lif.get_image(series_idx)

    n_channels = info.get("channels", 1)
    dims = info.get("dims", {})
    n_z = dims.get("z", 1) if isinstance(dims, dict) else 1

    # Extract pixel size
    scale_info = info.get("scale", (None,))
    pixel_size_um = scale_info[0] if scale_info and scale_info[0] else None

    # Read all frames
    frames = []
    for z in range(n_z):
        channel_data = []
        for c in range(n_channels):
            try:
                frame = image.get_frame(z=z, t=0, c=c)
                channel_data.append(np.array(frame))
            except Exception:
                # Fallback for simpler images
                if z == 0 and c == 0:
                    frame = image.get_frame()
                    channel_data.append(np.array(frame))
                break
        if channel_data:
            frames.append(np.stack(channel_data, axis=0))

    if not frames:
        raise ValueError(f"Could not read any frames from series '{series_name}'")

    # Stack Z frames: shape (Z, C, H, W)
    data = np.stack(frames, axis=0) if len(frames) > 1 else frames[0][np.newaxis, ...]

    # Apply Z-stack processing
    if mode == "max":
        data = apply_max_projection(data, z_range)
    # For 'frames' and 'rows', keep full (Z, C, H, W) shape

    return data, pixel_size_um
