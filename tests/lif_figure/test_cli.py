"""Tests for CLI module."""

import pytest
from pathlib import Path
from click.testing import CliRunner

from lif_figure.cli import main, sanitize_filename


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
