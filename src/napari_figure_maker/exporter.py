"""Export functionality for napari-figure-maker."""

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def export_figure(
    figure: np.ndarray,
    path: Path,
    dpi: int = 300,
) -> None:
    """Export a figure to an image file.

    Args:
        figure: RGB numpy array of the figure.
        path: Output file path. Format determined by extension.
        dpi: Resolution in dots per inch.
    """
    path = Path(path)
    img = Image.fromarray(figure)

    # Set DPI metadata
    img.info["dpi"] = (dpi, dpi)

    # Determine format from extension
    ext = path.suffix.lower()

    if ext in [".tif", ".tiff"]:
        img.save(path, format="TIFF", dpi=(dpi, dpi))
    elif ext == ".png":
        img.save(path, format="PNG", dpi=(dpi, dpi))
    else:
        # Default to PNG
        img.save(path, format="PNG", dpi=(dpi, dpi))


def copy_to_clipboard(figure: np.ndarray) -> bool:
    """Copy figure to system clipboard.

    Args:
        figure: RGB numpy array of the figure.

    Returns:
        True if successful, False otherwise.
    """
    try:
        import io
        import subprocess
        import sys
        import tempfile

        img = Image.fromarray(figure)

        # Platform-specific clipboard handling
        if sys.platform == "darwin":  # macOS
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                img.save(f.name, format="PNG")
                subprocess.run([
                    "osascript", "-e",
                    f'set the clipboard to (read (POSIX file "{f.name}") as «class PNGf»)'
                ])
            return True
        else:
            # For other platforms, return False to indicate not implemented
            return False
    except Exception:
        return False
