"""Tests for scale bar rendering."""

import numpy as np
from PIL import Image
from napari_figure_maker.scale_bar import calculate_nice_scale_bar_length, render_scale_bar


def test_calculate_nice_scale_bar_length_small():
    """Should pick nice round number for small images."""
    # Image is 100um wide, want bar ~20% of width = 20um
    # Should round to 20um
    length = calculate_nice_scale_bar_length(
        image_width_um=100.0,
        target_fraction=0.2,
    )
    assert length == 20.0


def test_calculate_nice_scale_bar_length_large():
    """Should pick nice round number for larger images."""
    # Image is 500um wide, want bar ~20% = 100um
    length = calculate_nice_scale_bar_length(
        image_width_um=500.0,
        target_fraction=0.2,
    )
    assert length == 100.0


def test_calculate_nice_scale_bar_length_awkward():
    """Should round to nearest nice number."""
    # Image is 73um wide, want bar ~20% = 14.6um
    # Should round to 10 or 15
    length = calculate_nice_scale_bar_length(
        image_width_um=73.0,
        target_fraction=0.2,
    )
    assert length in [10.0, 15.0, 20.0]


def test_render_scale_bar_returns_image():
    """render_scale_bar should return a PIL Image."""
    bar = render_scale_bar(
        length_um=10.0,
        pixel_size_um=0.5,
        color="white",
        font_size=10,
    )

    assert isinstance(bar, Image.Image)
    assert bar.mode == "RGBA"


def test_render_scale_bar_correct_width():
    """Scale bar image width should match physical length."""
    bar = render_scale_bar(
        length_um=10.0,
        pixel_size_um=0.5,  # 0.5um per pixel = 20 pixels for 10um
        color="white",
        font_size=10,
    )

    # Bar should be 20 pixels wide (plus some padding for text)
    assert bar.width >= 20
