"""Tests for LIF reader module."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from lif_figure.reader import (
    list_series,
    read_series,
    apply_max_projection,
    parse_zstack_mode,
)


def test_parse_zstack_mode_max():
    """parse_zstack_mode should parse 'max' mode."""
    mode, z_range = parse_zstack_mode("max")

    assert mode == "max"
    assert z_range is None


def test_parse_zstack_mode_max_with_range():
    """parse_zstack_mode should parse 'max:5-15' format."""
    mode, z_range = parse_zstack_mode("max:5-15")

    assert mode == "max"
    assert z_range == (5, 15)


def test_parse_zstack_mode_frames():
    """parse_zstack_mode should parse 'frames' mode."""
    mode, z_range = parse_zstack_mode("frames")

    assert mode == "frames"
    assert z_range is None


def test_parse_zstack_mode_rows():
    """parse_zstack_mode should parse 'rows' mode."""
    mode, z_range = parse_zstack_mode("rows")

    assert mode == "rows"
    assert z_range is None


def test_apply_max_projection():
    """apply_max_projection should compute max along Z axis."""
    # Shape: (Z, C, H, W) = (3, 2, 4, 4)
    data = np.array([
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],      # z=0
        [[[2, 3], [4, 5]], [[6, 7], [8, 9]]],      # z=1
        [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],      # z=2
    ])

    result = apply_max_projection(data, z_range=None)

    # Should be max across z, shape (C, H, W)
    assert result.shape == (2, 2, 2)
    np.testing.assert_array_equal(result[0], [[2, 3], [4, 5]])


def test_apply_max_projection_with_range():
    """apply_max_projection should respect z_range."""
    data = np.array([
        [[[0, 0], [0, 0]]],  # z=0
        [[[5, 5], [5, 5]]],  # z=1
        [[[3, 3], [3, 3]]],  # z=2
        [[[1, 1], [1, 1]]],  # z=3
    ])

    result = apply_max_projection(data, z_range=(1, 2))

    # Max of z=1 and z=2 only
    assert result.shape == (1, 2, 2)
    np.testing.assert_array_equal(result[0], [[5, 5], [5, 5]])
