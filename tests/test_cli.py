"""Tests for CLI module."""

import pytest
from pathlib import Path
from click.testing import CliRunner

from lif_figure.cli import main, sanitize_filename, parse_series_indices


def test_sanitize_filename():
    """sanitize_filename should replace invalid characters."""
    assert sanitize_filename("Normal Name") == "Normal Name"
    assert sanitize_filename("With/Slash") == "With_Slash"
    assert sanitize_filename("With\\Backslash") == "With_Backslash"
    assert sanitize_filename("With:Colon") == "With_Colon"


def test_cli_requires_channels():
    """CLI should error without --channels flag."""
    runner = CliRunner()
    result = runner.invoke(main, ["nonexistent.lif"])

    assert result.exit_code != 0
    assert "channels" in result.output.lower() or "required" in result.output.lower()


def test_cli_requires_lif_file():
    """CLI should error if file doesn't exist."""
    runner = CliRunner()
    result = runner.invoke(main, ["nonexistent.lif", "--channels", "DAPI,GFP"])

    assert result.exit_code != 0
    assert "not found" in result.output.lower() or "exist" in result.output.lower()


def test_cli_help():
    """CLI should show help."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "channels" in result.output
    assert "series" in result.output
    assert "zstack" in result.output


class TestParseSeriesIndices:
    """Tests for parse_series_indices function."""

    def test_single_index(self):
        """Single index returns list with that index."""
        assert parse_series_indices("5", 10) == [5]

    def test_single_zero(self):
        """Index 0 is valid."""
        assert parse_series_indices("0", 10) == [0]

    def test_single_last_valid(self):
        """Last valid index works."""
        assert parse_series_indices("9", 10) == [9]

    def test_range_inclusive(self):
        """Range N..M includes both endpoints."""
        assert parse_series_indices("2..5", 10) == [2, 3, 4, 5]

    def test_range_single_element(self):
        """Range N..N returns single element."""
        assert parse_series_indices("3..3", 10) == [3]

    def test_range_full(self):
        """Range covering all indices."""
        assert parse_series_indices("0..9", 10) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
