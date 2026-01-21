"""Configuration loading and defaults."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


DEFAULT_COLORS = ["blue", "green", "red"]


@dataclass
class Config:
    """Configuration for figure generation."""

    colors: list[str] = field(default_factory=lambda: DEFAULT_COLORS.copy())
    color_overrides: dict[str, str] = field(default_factory=dict)
    dpi: int = 300
    font_size: int = 12
    scale_bar_height: int = 4
    background: str = "black"

    def get_color(self, channel_name: str, index: int) -> str:
        """Get color for a channel, checking overrides first."""
        if channel_name in self.color_overrides:
            return self.color_overrides[channel_name]
        if index < len(self.colors):
            return self.colors[index]
        return "gray"


def load_config(config_path: Optional[Path]) -> Config:
    """Load config from YAML file or return defaults."""
    if config_path is None:
        return Config()

    if not config_path.exists():
        return Config()

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    color_overrides = data.get("colors", {})

    return Config(
        color_overrides=color_overrides,
        dpi=data.get("dpi", 300),
        font_size=data.get("font_size", 12),
        scale_bar_height=data.get("scale_bar_height", 4),
        background=data.get("background", "black"),
    )
