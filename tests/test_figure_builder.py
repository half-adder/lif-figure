"""Tests for figure builder."""

import numpy as np
from PIL import Image
from napari_figure_maker.figure_builder import render_channel_to_rgb, create_merge_composite


def test_render_channel_to_rgb_gray():
    """Should render grayscale channel to RGB."""
    data = np.array([[0, 128], [255, 64]], dtype=np.uint8)

    rgb = render_channel_to_rgb(
        data=data,
        colormap="gray",
        contrast_limits=(0, 255),
    )

    assert isinstance(rgb, np.ndarray)
    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == np.uint8
    # Gray: R=G=B
    assert np.array_equal(rgb[0, 0], [0, 0, 0])
    assert np.array_equal(rgb[0, 1], [128, 128, 128])


def test_render_channel_to_rgb_green():
    """Should render channel with green colormap."""
    data = np.array([[0, 255]], dtype=np.uint8)

    rgb = render_channel_to_rgb(
        data=data,
        colormap="green",
        contrast_limits=(0, 255),
    )

    # Green: R=0, G=value, B=0
    assert rgb[0, 0, 1] == 0  # G at min
    assert rgb[0, 1, 1] == 255  # G at max
    assert rgb[0, 1, 0] == 0  # R should be 0
    assert rgb[0, 1, 2] == 0  # B should be 0


def test_render_channel_contrast_limits():
    """Should apply contrast limits correctly."""
    data = np.array([[50, 100, 150, 200]], dtype=np.uint8)

    rgb = render_channel_to_rgb(
        data=data,
        colormap="gray",
        contrast_limits=(100, 200),  # Map 100->0, 200->255
    )

    # Value 50 is below min, should be 0
    assert rgb[0, 0, 0] == 0
    # Value 100 is at min, should be 0
    assert rgb[0, 1, 0] == 0
    # Value 200 is at max, should be 255
    assert rgb[0, 3, 0] == 255


def test_create_merge_composite():
    """Should merge multiple RGB channels additively."""
    # Red channel
    red = np.zeros((2, 2, 3), dtype=np.uint8)
    red[0, 0] = [255, 0, 0]

    # Green channel
    green = np.zeros((2, 2, 3), dtype=np.uint8)
    green[0, 0] = [0, 255, 0]

    merged = create_merge_composite([red, green])

    # Should be yellow where both overlap
    assert merged[0, 0, 0] == 255  # R
    assert merged[0, 0, 1] == 255  # G
    assert merged[0, 0, 2] == 0    # B


def test_create_merge_composite_clipping():
    """Should clip values that exceed 255."""
    ch1 = np.full((2, 2, 3), 200, dtype=np.uint8)
    ch2 = np.full((2, 2, 3), 200, dtype=np.uint8)

    merged = create_merge_composite([ch1, ch2])

    # Should be clipped to 255, not overflow
    assert merged[0, 0, 0] == 255
