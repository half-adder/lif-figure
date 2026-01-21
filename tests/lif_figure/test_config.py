"""Tests for config module."""

import pytest
from pathlib import Path

from lif_figure.config import Config, load_config


def test_config_defaults():
    """Config should have sensible defaults."""
    config = Config()

    assert config.colors == ["blue", "green", "red"]
    assert config.dpi == 300
    assert config.font_size == 12
    assert config.scale_bar_height == 4
    assert config.background == "black"


def test_config_color_override():
    """Config should allow color overrides by channel name."""
    config = Config(color_overrides={"DAPI": "cyan", "GFP": "lime"})

    assert config.get_color("DAPI", 0) == "cyan"
    assert config.get_color("GFP", 1) == "lime"
    assert config.get_color("mCherry", 2) == "red"  # falls back to positional


def test_load_config_returns_defaults_when_no_file():
    """load_config should return defaults when no config file exists."""
    config = load_config(None)

    assert config.dpi == 300


def test_load_config_from_yaml(tmp_path):
    """load_config should load values from YAML file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
colors:
  DAPI: cyan
dpi: 150
background: white
""")

    config = load_config(config_file)

    assert config.dpi == 150
    assert config.background == "white"
    assert config.get_color("DAPI", 0) == "cyan"
