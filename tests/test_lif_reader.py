"""Tests for LIF file reader."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from napari_figure_maker._lif_reader import napari_get_reader, read_lif_file


def test_napari_get_reader_accepts_lif():
    """Should return reader for .lif files."""
    reader = napari_get_reader("test.lif")
    assert reader is not None
    assert callable(reader)


def test_napari_get_reader_rejects_other():
    """Should return None for non-.lif files."""
    assert napari_get_reader("test.tiff") is None
    assert napari_get_reader("test.png") is None
    assert napari_get_reader(["multiple", "paths"]) is None


@patch("napari_figure_maker._lif_reader.LifFile")
def test_read_lif_file_returns_layer_data(mock_lif_class):
    """Should return layer data tuples from LIF file."""
    # Mock the LifFile and its contents
    mock_image = MagicMock()
    mock_image.name = "TestImage"
    mock_image.channels = 2
    mock_image.dims.x = 100
    mock_image.dims.y = 100
    mock_image.scale = (0.5, 0.5, 1.0)  # x, y, z scale in um
    mock_image.get_frame.return_value = np.zeros((100, 100), dtype=np.uint8)

    mock_lif = MagicMock()
    mock_lif.image_list = [mock_image]
    mock_lif_class.return_value = mock_lif

    result = read_lif_file("test.lif")

    assert isinstance(result, list)
    assert len(result) > 0
    # Each item should be (data, kwargs, layer_type)
    data, kwargs, layer_type = result[0]
    assert isinstance(data, np.ndarray)
    assert layer_type == "image"
