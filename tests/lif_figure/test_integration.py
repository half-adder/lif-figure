"""Integration tests for lif-figure CLI."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
import numpy as np

from lif_figure.cli import main


@pytest.fixture
def mock_lif_data():
    """Create mock LIF file data."""
    # 3 channels, 64x64 pixels
    return np.random.rand(3, 64, 64).astype(np.float32)


@pytest.fixture
def mock_lif_file(tmp_path):
    """Create a fake LIF file path."""
    lif_path = tmp_path / "test.lif"
    lif_path.touch()
    return lif_path


def test_full_pipeline_with_mock(tmp_path, mock_lif_file, mock_lif_data):
    """Test full pipeline from CLI to PDF output."""
    output_dir = tmp_path / "output"

    with patch("lif_figure.cli.list_series") as mock_list, \
         patch("lif_figure.cli.read_series") as mock_read:

        mock_list.return_value = ["Sample 1", "Sample 2"]
        mock_read.return_value = (mock_lif_data, 0.5)  # pixel_size = 0.5 um

        runner = CliRunner()
        result = runner.invoke(main, [
            str(mock_lif_file),
            "--channels", "DAPI,GFP,mCherry",
            "--output", str(output_dir),
        ])

        assert result.exit_code == 0
        assert "Done" in result.output

        # Check PDFs were created
        assert (output_dir / "Sample 1.pdf").exists()
        assert (output_dir / "Sample 2.pdf").exists()


def test_series_filter_with_mock(tmp_path, mock_lif_file, mock_lif_data):
    """Test --series flag filters correctly."""
    output_dir = tmp_path / "output"

    with patch("lif_figure.cli.list_series") as mock_list, \
         patch("lif_figure.cli.read_series") as mock_read:

        mock_list.return_value = ["Sample 1", "Sample 2", "Sample 3"]
        mock_read.return_value = (mock_lif_data, 0.5)

        runner = CliRunner()
        result = runner.invoke(main, [
            str(mock_lif_file),
            "--channels", "DAPI,GFP,mCherry",
            "--series", "Sample 1,Sample 3",
            "--output", str(output_dir),
        ])

        assert result.exit_code == 0

        # Only requested series
        assert (output_dir / "Sample 1.pdf").exists()
        assert not (output_dir / "Sample 2.pdf").exists()
        assert (output_dir / "Sample 3.pdf").exists()
