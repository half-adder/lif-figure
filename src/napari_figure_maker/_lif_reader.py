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

    # Iterate over images using get_image() which returns LifImage objects
    for i in range(len(lif.image_list)):
        image = lif.get_image(i)

        # Get image dimensions from the info dict
        info = lif.image_list[i]
        n_channels = info.get('channels', 1)
        dims = info.get('dims')
        scale_info = info.get('scale', (None, None, None, None))
        name = info.get('name', f'Image {i}')

        # Read all channels
        channel_data = []
        for c in range(n_channels):
            # Get first frame (z=0, t=0) for each channel
            try:
                frame = image.get_frame(z=0, t=0, c=c)
                channel_data.append(np.array(frame))
            except Exception:
                # If get_frame fails, try without parameters
                try:
                    frame = image.get_frame()
                    channel_data.append(np.array(frame))
                    break  # Only one frame available
                except Exception:
                    continue

        if not channel_data:
            continue

        # Stack channels
        if len(channel_data) > 1:
            data = np.stack(channel_data, axis=0)
        else:
            data = channel_data[0]

        # Extract scale from metadata
        scale = None
        pixel_size_um = None
        if scale_info and scale_info[0] is not None:
            pixel_size_um = scale_info[0]  # x scale in um
            if scale_info[1] is not None:
                scale = (scale_info[1], scale_info[0])  # (y, x) for napari

        # Build layer kwargs
        kwargs = {
            "name": name,
            "metadata": {
                "source": path,
                "pixel_size_um": pixel_size_um,
            },
        }

        if scale:
            kwargs["scale"] = scale

        layers.append((data, kwargs, "image"))

    return layers
