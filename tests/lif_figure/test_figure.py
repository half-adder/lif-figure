"""Tests for figure module."""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from lif_figure.figure import (
    normalize_channel,
    apply_colormap,
    create_merge,
    build_figure,
)
from lif_figure.config import Config


def test_normalize_channel():
    """normalize_channel should scale to 0-1 range."""
    data = np.array([[0, 50], [100, 200]], dtype=np.uint16)

    result = normalize_channel(data)

    assert result.min() == 0.0
    assert result.max() == 1.0
    assert result.dtype == np.float64


def test_normalize_channel_handles_constant():
    """normalize_channel should handle constant arrays."""
    data = np.array([[5, 5], [5, 5]], dtype=np.uint16)

    result = normalize_channel(data)

    assert np.all(result == 0.0)  # or 1.0, just no NaN


def test_normalize_channel_with_percentiles():
    """normalize_channel with percentiles should clip outliers."""
    # Array with outliers: 0, 1-98, 1000 (100 values total)
    data = np.array([0] + list(range(1, 99)) + [1000], dtype=np.float64).reshape(10, 10)

    # Without percentiles: 0 maps to 0, 1000 maps to 1
    result_minmax = normalize_channel(data)
    assert result_minmax.min() == 0.0
    assert result_minmax.max() == 1.0
    # Middle values get compressed
    assert result_minmax.flat[50] < 0.1  # value 50 out of 1000

    # With percentiles (5, 95): clips bottom 5% and top 5%
    result_pct = normalize_channel(data, percentiles=(5.0, 95.0))
    # Now middle values should use more of the range
    assert result_pct.flat[50] > 0.3  # value 50 now has more room


def test_normalize_channel_with_percentiles_clips():
    """normalize_channel with percentiles should clip to 0-1."""
    data = np.array([[0, 50, 100, 150, 200]], dtype=np.float64)

    # Clip to 25th-75th percentile
    result = normalize_channel(data, percentiles=(25.0, 75.0))

    # Values outside the percentile range should be clipped
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_apply_colormap_gray():
    """apply_colormap should create RGB from grayscale."""
    data = np.array([[0.0, 0.5], [1.0, 0.25]])

    result = apply_colormap(data, "gray")

    assert result.shape == (2, 2, 3)
    # Gray means R=G=B
    np.testing.assert_array_almost_equal(result[0, 0], [0, 0, 0])
    np.testing.assert_array_almost_equal(result[1, 0], [1, 1, 1])


def test_apply_colormap_blue():
    """apply_colormap with blue should only affect blue channel."""
    data = np.array([[0.0, 1.0]])

    result = apply_colormap(data, "blue")

    # Blue: (0, 0, value)
    np.testing.assert_array_almost_equal(result[0, 0], [0, 0, 0])
    np.testing.assert_array_almost_equal(result[0, 1], [0, 0, 1])


def test_create_merge():
    """create_merge should additively combine colored channels."""
    ch1 = np.array([[1.0]])  # blue
    ch2 = np.array([[1.0]])  # green

    colored = [
        apply_colormap(ch1, "blue"),
        apply_colormap(ch2, "green"),
    ]

    result = create_merge(colored)

    assert result.shape == (1, 1, 3)
    # Blue + Green = Cyan (0, 1, 1), clipped
    np.testing.assert_array_almost_equal(result[0, 0], [0, 1, 1])


def test_build_figure_returns_figure():
    """build_figure should return a matplotlib Figure."""
    channels = np.random.rand(3, 64, 64).astype(np.float32)
    names = ["DAPI", "GFP", "mCherry"]
    config = Config()

    fig = build_figure(channels, names, config, pixel_size_um=0.5)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_build_figure_panel_count():
    """build_figure should create N+1 panels (channels + merge)."""
    channels = np.random.rand(2, 64, 64).astype(np.float32)
    names = ["Ch1", "Ch2"]
    config = Config()

    fig = build_figure(channels, names, config, pixel_size_um=None)

    # Should have 3 axes: Ch1, Ch2, Merge
    assert len(fig.axes) == 3
    plt.close(fig)
