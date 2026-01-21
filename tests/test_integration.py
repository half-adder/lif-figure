"""Integration tests for napari-figure-maker."""

import numpy as np
from pathlib import Path

from napari_figure_maker.models import ChannelConfig, FigureConfig
from napari_figure_maker.figure_builder import build_figure
from napari_figure_maker.exporter import export_figure
from napari_figure_maker.presets import save_preset, load_preset


def test_full_pipeline(tmp_path):
    """Test complete workflow: build figure and export."""
    # Create test data
    dapi = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    gfp = np.random.randint(0, 255, (256, 256), dtype=np.uint8)

    channels_data = [dapi, gfp]
    channel_configs = [
        ChannelConfig(name="DAPI", colormap="blue", label="Nuclei"),
        ChannelConfig(name="GFP", colormap="green", label="GFP"),
    ]
    figure_config = FigureConfig(
        dpi=300,
        show_merge=True,
        panel_gap_fraction=0.02,
        background_color="black",
    )

    # Build figure
    figure = build_figure(
        channels_data=channels_data,
        channel_configs=channel_configs,
        figure_config=figure_config,
        pixel_size_um=0.5,
    )

    assert figure is not None
    assert figure.ndim == 3
    assert figure.shape[2] == 3  # RGB

    # Export
    output_path = tmp_path / "test_figure.png"
    export_figure(figure, output_path, dpi=300)

    assert output_path.exists()


def test_preset_roundtrip(tmp_path):
    """Test saving and loading presets preserves all settings."""
    channel_configs = [
        ChannelConfig(name="Ch1", colormap="red", label="Channel 1", visible=True),
        ChannelConfig(name="Ch2", colormap="cyan", label="Channel 2", visible=False),
    ]
    figure_config = FigureConfig(
        dpi=150,
        show_merge=False,
        panel_gap_fraction=0.05,
        background_color="white",
        label_font_size=14,
        scale_bar_color="black",
    )

    preset_path = tmp_path / "roundtrip.yaml"

    # Save
    save_preset(
        path=preset_path,
        name="Roundtrip Test",
        channel_configs=channel_configs,
        figure_config=figure_config,
    )

    # Load
    name, loaded_channels, loaded_figure = load_preset(preset_path)

    # Verify
    assert name == "Roundtrip Test"
    assert len(loaded_channels) == 2
    assert loaded_channels[0].colormap == "red"
    assert loaded_channels[1].visible is False
    assert loaded_figure.dpi == 150
    assert loaded_figure.show_merge is False
    assert loaded_figure.background_color == "white"
    assert loaded_figure.label_font_size == 14


def test_figure_with_scale_bar():
    """Test figure generation with scale bar."""
    data = np.zeros((100, 100), dtype=np.uint8)
    data[40:60, 40:60] = 200  # Square in center

    figure = build_figure(
        channels_data=[data],
        channel_configs=[ChannelConfig(name="Test", colormap="gray")],
        figure_config=FigureConfig(show_merge=False),
        pixel_size_um=0.5,  # 50um total width
    )

    # Figure should have scale bar added
    assert figure is not None
    # Scale bar would be in bottom right - just verify figure was created
    assert figure.shape[0] == 100
