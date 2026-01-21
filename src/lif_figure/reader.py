"""LIF file reading and Z-stack processing."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from readlif.reader import LifFile


@dataclass
class DetectorInfo:
    """Information about a detector channel."""
    name: str
    mode: str  # "Std" or "PC" (PhotonCounting)
    gain: float


@dataclass
class SeriesMetadata:
    """Acquisition metadata for a series."""
    lasers: dict[str, float] = field(default_factory=dict)  # wavelength -> power %
    detectors: list[DetectorInfo] = field(default_factory=list)


def extract_series_metadata(lif: LifFile, series_name: str) -> SeriesMetadata:
    """Extract acquisition metadata for a series from LIF XML.

    Args:
        lif: LifFile object
        series_name: Name of the series (may include collection path like "Collection/Series001")

    Returns:
        SeriesMetadata with laser and detector info
    """
    metadata = SeriesMetadata()
    root = lif.xml_root

    # Extract just the series name without collection path for XML matching
    # e.g., "New collection001/Series001" -> "Series001"
    xml_series_name = series_name.split('/')[-1] if '/' in series_name else series_name

    # Find the series Element
    for elem in root.iter('Element'):
        if elem.get('Name') == xml_series_name:
            # Find HardwareSetting attachment
            for attachment in elem.iter('Attachment'):
                if attachment.get('Name') == 'HardwareSetting':
                    _extract_lasers(attachment, metadata)
                    _extract_detectors(attachment, metadata)
                    break
            break

    return metadata


def _extract_lasers(attachment, metadata: SeriesMetadata) -> None:
    """Extract active laser settings from HardwareSetting."""
    # Find AOTF sections with IsChanged="1" (active during acquisition)
    for aotf in attachment.iter('Aotf'):
        if aotf.get('IsChanged') == '1':
            for laser in aotf.iter('LaserLineSetting'):
                wavelength = laser.get('LaserLine')
                intensity_str = laser.get('IntensityDev', '0')
                try:
                    intensity = float(intensity_str)
                    if intensity > 0.001 and wavelength:
                        # Store as percentage, cap display at reasonable max
                        metadata.lasers[wavelength] = intensity
                except ValueError:
                    pass


def _extract_detectors(attachment, metadata: SeriesMetadata) -> None:
    """Extract active detector settings from HardwareSetting."""
    for detector in attachment.iter('Detector'):
        if detector.get('IsActive') == '1':
            # Get detector name (fall back to index-based name)
            name = detector.get('DetectorName', '').strip()
            if not name:
                name = f"HyD{len(metadata.detectors) + 1}"

            # Get acquisition mode
            mode_name = detector.get('AcquisitionModeName', '')
            mode = "PC" if mode_name == "PhotonCounting" else "Std"

            # Get gain
            gain_str = detector.get('Gain', '0')
            try:
                gain = float(gain_str)
            except ValueError:
                gain = 0.0

            metadata.detectors.append(DetectorInfo(name=name, mode=mode, gain=gain))


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
