"""Tests for configuration data models."""

from napari_figure_maker.models import ChannelConfig, FigureConfig


def test_channel_config_defaults():
    """ChannelConfig should have sensible defaults."""
    config = ChannelConfig(name="DAPI")

    assert config.name == "DAPI"
    assert config.visible is True
    assert config.label is None  # Will use name if not set
    assert config.colormap == "gray"


def test_channel_config_custom_values():
    """ChannelConfig should accept custom values."""
    config = ChannelConfig(
        name="GFP",
        visible=True,
        label="Green",
        colormap="green",
    )

    assert config.name == "GFP"
    assert config.label == "Green"
    assert config.colormap == "green"


def test_figure_config_defaults():
    """FigureConfig should have sensible defaults."""
    config = FigureConfig()

    assert config.dpi == 300
    assert config.panel_gap_fraction == 0.02
    assert config.background_color == "black"
    assert config.show_merge is True
    assert config.scale_bar_color == "white"
    assert config.scale_bar_position == "bottom-right"
    assert config.show_scale_bar_on_merge_only is True
    assert config.label_font_size == 12
    assert config.label_position == "top-left"


def test_figure_config_custom_values():
    """FigureConfig should accept custom values."""
    config = FigureConfig(
        dpi=150,
        panel_gap_fraction=0.05,
        background_color="white",
        show_merge=False,
    )

    assert config.dpi == 150
    assert config.panel_gap_fraction == 0.05
    assert config.background_color == "white"
    assert config.show_merge is False
