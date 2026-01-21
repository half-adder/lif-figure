"""Configuration data models for napari-figure-maker."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ChannelConfig:
    """Configuration for a single channel in the figure."""

    name: str
    visible: bool = True
    label: Optional[str] = None
    colormap: str = "gray"


@dataclass
class FigureConfig:
    """Configuration for the entire figure layout and export."""

    # Export settings
    dpi: int = 300
    figure_width_inches: Optional[float] = None  # None = auto from panel size

    # Layout settings
    panel_gap_fraction: float = 0.02  # Gap as fraction of panel width
    background_color: str = "black"
    show_merge: bool = True
    grid_columns: Optional[int] = None  # None = single row

    # Scale bar settings
    scale_bar_length_um: Optional[float] = None  # None = auto
    scale_bar_color: str = "white"
    scale_bar_position: str = "bottom-right"
    scale_bar_font_size: int = 10
    show_scale_bar_on_merge_only: bool = True

    # Label settings
    label_font_size: int = 12
    label_position: str = "top-left"
    label_color: Optional[str] = None  # None = match channel colormap
