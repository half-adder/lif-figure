"""Pytest configuration and fixtures."""

import numpy as np
import pytest


@pytest.fixture
def sample_channel_data():
    """Create sample single-channel image data."""
    return np.random.randint(0, 255, (100, 100), dtype=np.uint8)


@pytest.fixture
def sample_multichannel_data():
    """Create sample 3-channel image data."""
    return [
        np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        for _ in range(3)
    ]
