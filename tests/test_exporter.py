"""Tests for export functionality."""

import numpy as np
from pathlib import Path
from PIL import Image

from napari_figure_maker.exporter import export_figure


def test_export_figure_png(tmp_path):
    """Should export figure as PNG."""
    figure = np.full((100, 200, 3), 128, dtype=np.uint8)
    output_path = tmp_path / "test_figure.png"

    export_figure(figure, output_path, dpi=300)

    assert output_path.exists()

    # Verify it's a valid PNG
    img = Image.open(output_path)
    assert img.format == "PNG"
    assert img.size == (200, 100)  # Width, Height


def test_export_figure_tiff(tmp_path):
    """Should export figure as TIFF."""
    figure = np.full((100, 200, 3), 128, dtype=np.uint8)
    output_path = tmp_path / "test_figure.tiff"

    export_figure(figure, output_path, dpi=300)

    assert output_path.exists()

    img = Image.open(output_path)
    assert img.format == "TIFF"
