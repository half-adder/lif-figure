"""LIF file reader for napari."""

from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from readlif.reader import LifFile


def napari_get_reader(path: Union[str, List[str]]) -> Optional[Callable]:
    """Return a reader function if path is a LIF file.

    Args:
        path: Path to file or list of paths.

    Returns:
        Reader function or None if not a LIF file.
    """
    if isinstance(path, list):
        return None

    if not isinstance(path, str):
        return None

    if not path.lower().endswith(".lif"):
        return None

    return read_lif_file


def read_lif_file(path: str) -> List[Tuple[np.ndarray, dict, str]]:
    """Read a LIF file and return napari layer data.

    Args:
        path: Path to LIF file.

    Returns:
        List of (data, kwargs, layer_type) tuples for napari.
    """
    lif = LifFile(path)
    layers = []

    for image in lif.image_list:
        # Get image dimensions
        n_channels = image.channels

        # Read all channels
        channel_data = []
        for c in range(n_channels):
            # Get first frame (z=0, t=0) for each channel
            frame = image.get_frame(z=0, t=0, c=c)
            channel_data.append(np.array(frame))

        # Stack channels
        if len(channel_data) > 1:
            data = np.stack(channel_data, axis=0)
        else:
            data = channel_data[0]

        # Extract scale from metadata
        scale = None
        if hasattr(image, 'scale') and image.scale:
            # scale is (x, y, z) in micrometers
            if len(image.scale) >= 2:
                scale = (image.scale[1], image.scale[0])  # (y, x) for napari

        # Build layer kwargs
        kwargs = {
            "name": image.name or "LIF Image",
            "metadata": {
                "source": path,
                "pixel_size_um": image.scale[0] if image.scale else None,
            },
        }

        if scale:
            kwargs["scale"] = scale

        layers.append((data, kwargs, "image"))

    return layers
