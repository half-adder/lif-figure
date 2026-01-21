"""Tests for presets system."""

import tempfile
from pathlib import Path

from napari_figure_maker.models import ChannelConfig, FigureConfig
from napari_figure_maker.presets import save_preset, load_preset, get_preset_directory, list_presets


def test_save_and_load_preset(tmp_path):
    """Should save preset to YAML and load it back."""
    channel_configs = [
        ChannelConfig(name="DAPI", colormap="blue", label="Nuclei"),
        ChannelConfig(name="GFP", colormap="green", label="GFP"),
    ]
    figure_config = FigureConfig(dpi=300, show_merge=True)

    preset_path = tmp_path / "test_preset.yaml"

    save_preset(
        path=preset_path,
        name="Test Preset",
        channel_configs=channel_configs,
        figure_config=figure_config,
    )

    assert preset_path.exists()

    # Load it back
    loaded_name, loaded_channels, loaded_figure = load_preset(preset_path)

    assert loaded_name == "Test Preset"
    assert len(loaded_channels) == 2
    assert loaded_channels[0].name == "DAPI"
    assert loaded_channels[0].colormap == "blue"
    assert loaded_channels[1].label == "GFP"
    assert loaded_figure.dpi == 300
    assert loaded_figure.show_merge is True


def test_get_preset_directory():
    """Should return path to preset directory."""
    preset_dir = get_preset_directory()

    assert isinstance(preset_dir, Path)
    assert "napari-figure-maker" in str(preset_dir)


def test_list_presets(tmp_path, monkeypatch):
    """Should list all presets in directory."""
    # Create some preset files
    (tmp_path / "preset1.yaml").write_text("name: Preset 1\nchannels: []\nfigure: {}")
    (tmp_path / "preset2.yaml").write_text("name: Preset 2\nchannels: []\nfigure: {}")
    (tmp_path / "not_a_preset.txt").write_text("ignore me")

    # Monkeypatch the preset directory
    monkeypatch.setattr("napari_figure_maker.presets.get_preset_directory", lambda: tmp_path)

    presets = list_presets()

    assert len(presets) == 2
    assert "preset1" in [p["id"] for p in presets]
    assert "preset2" in [p["id"] for p in presets]
