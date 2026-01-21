"""Preset management for napari-figure-maker."""

import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from napari_figure_maker.models import ChannelConfig, FigureConfig


def save_preset(
    path: Path,
    name: str,
    channel_configs: List[ChannelConfig],
    figure_config: FigureConfig,
) -> None:
    """Save a preset to a YAML file.

    Args:
        path: File path to save to.
        name: Human-readable name for the preset.
        channel_configs: Channel configurations to save.
        figure_config: Figure configuration to save.
    """
    preset_data = {
        "name": name,
        "channels": [asdict(c) for c in channel_configs],
        "figure": asdict(figure_config),
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(preset_data, f, default_flow_style=False)


def load_preset(path: Path) -> Tuple[str, List[ChannelConfig], FigureConfig]:
    """Load a preset from a YAML file.

    Args:
        path: File path to load from.

    Returns:
        Tuple of (name, channel_configs, figure_config).
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    name = data.get("name", "Unnamed Preset")

    channel_configs = [
        ChannelConfig(**ch) for ch in data.get("channels", [])
    ]

    figure_config = FigureConfig(**data.get("figure", {}))

    return name, channel_configs, figure_config


def get_preset_directory() -> Path:
    """Get the directory where presets are stored.

    Returns:
        Path to preset directory (creates if doesn't exist).
    """
    # Use XDG config directory on Linux, appropriate locations on other platforms
    if os.name == "nt":  # Windows
        base = Path(os.environ.get("APPDATA", "~"))
    elif os.name == "posix" and "darwin" in os.uname().sysname.lower():  # macOS
        base = Path.home() / "Library" / "Application Support"
    else:  # Linux and others
        base = Path(os.environ.get("XDG_CONFIG_HOME", "~/.config"))

    preset_dir = base.expanduser() / "napari-figure-maker" / "presets"
    preset_dir.mkdir(parents=True, exist_ok=True)

    return preset_dir


def list_presets() -> List[Dict]:
    """List all available presets.

    Returns:
        List of dicts with 'id', 'name', and 'path' for each preset.
    """
    preset_dir = get_preset_directory()
    presets = []

    for path in preset_dir.glob("*.yaml"):
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            presets.append({
                "id": path.stem,
                "name": data.get("name", path.stem),
                "path": path,
            })
        except Exception:
            # Skip invalid files
            pass

    return presets
