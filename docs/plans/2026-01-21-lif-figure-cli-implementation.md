# LIF Figure CLI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a CLI tool that generates publication-ready figure panels (PDF) from Leica LIF microscopy files.

**Architecture:** CLI entry point parses args and loads config, reader extracts channels from LIF with Z-stack processing, figure module builds matplotlib figures with labels and scale bars, CLI saves PDFs to output directory.

**Tech Stack:** Python 3.13, click (CLI), readlif (LIF parsing), matplotlib (figures), numpy (arrays), pyyaml (config)

---

## Task 1: Project Restructuring

Create the new `lif_figure` package alongside existing code. We'll keep the napari plugin code but add the CLI as a separate package.

**Files:**
- Create: `src/lif_figure/__init__.py`
- Create: `src/lif_figure/py.typed`
- Modify: `pyproject.toml` - add CLI entry point

**Step 1: Create package directory and init**

```bash
mkdir -p src/lif_figure
```

Create `src/lif_figure/__init__.py`:
```python
"""LIF Figure CLI - Generate publication-ready figure panels from LIF files."""

__version__ = "0.1.0"
```

**Step 2: Create py.typed marker**

Create `src/lif_figure/py.typed`:
```
# Marker file for PEP 561
```

**Step 3: Update pyproject.toml with CLI entry point**

Add to `pyproject.toml` after the napari entry-points section:

```toml
[project.scripts]
lif-figure = "lif_figure.cli:main"
```

Update dependencies to include matplotlib if not present:
```toml
dependencies = [
    "numpy",
    "pillow",
    "pyyaml",
    "readlif",
    "click>=8.0",
    "matplotlib>=3.5",
]
```

**Step 4: Sync dependencies**

Run: `uv sync`
Expected: Dependencies install successfully

**Step 5: Commit**

```bash
git add -A
git commit -m "chore: add lif_figure package structure with CLI entry point"
```

---

## Task 2: Config Module

Handle config file loading and defaults.

**Files:**
- Create: `src/lif_figure/config.py`
- Create: `tests/lif_figure/__init__.py`
- Create: `tests/lif_figure/test_config.py`

**Step 1: Create test directory**

```bash
mkdir -p tests/lif_figure
touch tests/lif_figure/__init__.py
```

**Step 2: Write failing tests for config**

Create `tests/lif_figure/test_config.py`:
```python
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
```

**Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/lif_figure/test_config.py -v`
Expected: FAIL with "No module named 'lif_figure.config'"

**Step 4: Implement config module**

Create `src/lif_figure/config.py`:
```python
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
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/lif_figure/test_config.py -v`
Expected: All 4 tests PASS

**Step 6: Commit**

```bash
git add -A
git commit -m "feat(lif-figure): add config module with YAML loading"
```

---

## Task 3: Reader Module - Basic LIF Reading

Read LIF files and extract channel data.

**Files:**
- Create: `src/lif_figure/reader.py`
- Create: `tests/lif_figure/test_reader.py`

**Step 1: Write failing tests for reader**

Create `tests/lif_figure/test_reader.py`:
```python
"""Tests for LIF reader module."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from lif_figure.reader import (
    list_series,
    read_series,
    apply_max_projection,
    parse_zstack_mode,
)


def test_parse_zstack_mode_max():
    """parse_zstack_mode should parse 'max' mode."""
    mode, z_range = parse_zstack_mode("max")

    assert mode == "max"
    assert z_range is None


def test_parse_zstack_mode_max_with_range():
    """parse_zstack_mode should parse 'max:5-15' format."""
    mode, z_range = parse_zstack_mode("max:5-15")

    assert mode == "max"
    assert z_range == (5, 15)


def test_parse_zstack_mode_frames():
    """parse_zstack_mode should parse 'frames' mode."""
    mode, z_range = parse_zstack_mode("frames")

    assert mode == "frames"
    assert z_range is None


def test_parse_zstack_mode_rows():
    """parse_zstack_mode should parse 'rows' mode."""
    mode, z_range = parse_zstack_mode("rows")

    assert mode == "rows"
    assert z_range is None


def test_apply_max_projection():
    """apply_max_projection should compute max along Z axis."""
    # Shape: (Z, C, H, W) = (3, 2, 4, 4)
    data = np.array([
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],      # z=0
        [[[2, 3], [4, 5]], [[6, 7], [8, 9]]],      # z=1
        [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],      # z=2
    ])

    result = apply_max_projection(data, z_range=None)

    # Should be max across z, shape (C, H, W)
    assert result.shape == (2, 2, 2)
    np.testing.assert_array_equal(result[0], [[2, 3], [4, 5]])


def test_apply_max_projection_with_range():
    """apply_max_projection should respect z_range."""
    data = np.array([
        [[[0, 0], [0, 0]]],  # z=0
        [[[5, 5], [5, 5]]],  # z=1
        [[[3, 3], [3, 3]]],  # z=2
        [[[1, 1], [1, 1]]],  # z=3
    ])

    result = apply_max_projection(data, z_range=(1, 2))

    # Max of z=1 and z=2 only
    assert result.shape == (1, 2, 2)
    np.testing.assert_array_equal(result[0], [[5, 5], [5, 5]])
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/lif_figure/test_reader.py -v`
Expected: FAIL with "No module named 'lif_figure.reader'"

**Step 3: Implement reader module (parsing and projection)**

Create `src/lif_figure/reader.py`:
```python
"""LIF file reading and Z-stack processing."""

from pathlib import Path
from typing import Optional

import numpy as np
from readlif.reader import LifFile


def parse_zstack_mode(mode_str: str) -> tuple[str, Optional[tuple[int, int]]]:
    """Parse Z-stack mode string.

    Args:
        mode_str: Mode string like 'max', 'max:5-15', 'frames', 'rows'

    Returns:
        Tuple of (mode, z_range) where z_range is None or (start, end)
    """
    if mode_str.startswith("max:"):
        range_part = mode_str[4:]
        start, end = range_part.split("-")
        return ("max", (int(start), int(end)))

    return (mode_str, None)


def apply_max_projection(
    data: np.ndarray,
    z_range: Optional[tuple[int, int]] = None
) -> np.ndarray:
    """Apply max projection along Z axis.

    Args:
        data: Array with shape (Z, C, H, W)
        z_range: Optional (start, end) indices for Z range

    Returns:
        Array with shape (C, H, W) after max projection
    """
    if z_range is not None:
        start, end = z_range
        data = data[start:end + 1]

    return np.max(data, axis=0)


def list_series(path: Path) -> list[str]:
    """List all series names in a LIF file.

    Args:
        path: Path to LIF file

    Returns:
        List of series names
    """
    lif = LifFile(str(path))
    return [info.get("name", f"Series {i}") for i, info in enumerate(lif.image_list)]


def read_series(
    path: Path,
    series_name: str,
    zstack_mode: str = "max",
) -> tuple[np.ndarray, Optional[float]]:
    """Read a single series from a LIF file.

    Args:
        path: Path to LIF file
        series_name: Name of series to read
        zstack_mode: Z-stack handling mode

    Returns:
        Tuple of (channels_array, pixel_size_um)
        channels_array has shape (C, H, W) for max projection
        or (Z, C, H, W) for frames/rows modes
    """
    lif = LifFile(str(path))
    mode, z_range = parse_zstack_mode(zstack_mode)

    # Find the series by name
    series_idx = None
    for i, info in enumerate(lif.image_list):
        if info.get("name") == series_name:
            series_idx = i
            break

    if series_idx is None:
        raise ValueError(f"Series '{series_name}' not found in {path}")

    info = lif.image_list[series_idx]
    image = lif.get_image(series_idx)

    n_channels = info.get("channels", 1)
    dims = info.get("dims", {})
    n_z = dims.get("z", 1) if isinstance(dims, dict) else 1

    # Extract pixel size
    scale_info = info.get("scale", (None,))
    pixel_size_um = scale_info[0] if scale_info and scale_info[0] else None

    # Read all frames
    frames = []
    for z in range(n_z):
        channel_data = []
        for c in range(n_channels):
            try:
                frame = image.get_frame(z=z, t=0, c=c)
                channel_data.append(np.array(frame))
            except Exception:
                # Fallback for simpler images
                if z == 0 and c == 0:
                    frame = image.get_frame()
                    channel_data.append(np.array(frame))
                break
        if channel_data:
            frames.append(np.stack(channel_data, axis=0))

    if not frames:
        raise ValueError(f"Could not read any frames from series '{series_name}'")

    # Stack Z frames: shape (Z, C, H, W)
    data = np.stack(frames, axis=0) if len(frames) > 1 else frames[0][np.newaxis, ...]

    # Apply Z-stack processing
    if mode == "max":
        data = apply_max_projection(data, z_range)
    # For 'frames' and 'rows', keep full (Z, C, H, W) shape

    return data, pixel_size_um
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/lif_figure/test_reader.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(lif-figure): add reader module with Z-stack handling"
```

---

## Task 4: Figure Module - Panel Building

Build matplotlib figures with labeled panels and scale bars.

**Files:**
- Create: `src/lif_figure/figure.py`
- Create: `tests/lif_figure/test_figure.py`

**Step 1: Write failing tests for figure module**

Create `tests/lif_figure/test_figure.py`:
```python
"""Tests for figure module."""

import pytest
import numpy as np
import matplotlib.pyplot as plt

from lif_figure.figure import (
    normalize_channel,
    apply_colormap,
    create_merge,
    build_figure,
)
from lif_figure.config import Config


def test_normalize_channel():
    """normalize_channel should scale to 0-1 range."""
    data = np.array([[0, 50], [100, 200]], dtype=np.uint16)

    result = normalize_channel(data)

    assert result.min() == 0.0
    assert result.max() == 1.0
    assert result.dtype == np.float64


def test_normalize_channel_handles_constant():
    """normalize_channel should handle constant arrays."""
    data = np.array([[5, 5], [5, 5]], dtype=np.uint16)

    result = normalize_channel(data)

    assert np.all(result == 0.0)  # or 1.0, just no NaN


def test_apply_colormap_gray():
    """apply_colormap should create RGB from grayscale."""
    data = np.array([[0.0, 0.5], [1.0, 0.25]])

    result = apply_colormap(data, "gray")

    assert result.shape == (2, 2, 3)
    # Gray means R=G=B
    np.testing.assert_array_almost_equal(result[0, 0], [0, 0, 0])
    np.testing.assert_array_almost_equal(result[1, 0], [1, 1, 1])


def test_apply_colormap_blue():
    """apply_colormap with blue should only affect blue channel."""
    data = np.array([[0.0, 1.0]])

    result = apply_colormap(data, "blue")

    # Blue: (0, 0, value)
    np.testing.assert_array_almost_equal(result[0, 0], [0, 0, 0])
    np.testing.assert_array_almost_equal(result[0, 1], [0, 0, 1])


def test_create_merge():
    """create_merge should additively combine colored channels."""
    ch1 = np.array([[1.0]])  # blue
    ch2 = np.array([[1.0]])  # green

    colored = [
        apply_colormap(ch1, "blue"),
        apply_colormap(ch2, "green"),
    ]

    result = create_merge(colored)

    assert result.shape == (1, 1, 3)
    # Blue + Green = Cyan (0, 1, 1), clipped
    np.testing.assert_array_almost_equal(result[0, 0], [0, 1, 1])


def test_build_figure_returns_figure():
    """build_figure should return a matplotlib Figure."""
    channels = np.random.rand(3, 64, 64).astype(np.float32)
    names = ["DAPI", "GFP", "mCherry"]
    config = Config()

    fig = build_figure(channels, names, config, pixel_size_um=0.5)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_build_figure_panel_count():
    """build_figure should create N+1 panels (channels + merge)."""
    channels = np.random.rand(2, 64, 64).astype(np.float32)
    names = ["Ch1", "Ch2"]
    config = Config()

    fig = build_figure(channels, names, config, pixel_size_um=None)

    # Should have 3 axes: Ch1, Ch2, Merge
    assert len(fig.axes) == 3
    plt.close(fig)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/lif_figure/test_figure.py -v`
Expected: FAIL with "No module named 'lif_figure.figure'"

**Step 3: Implement figure module**

Create `src/lif_figure/figure.py`:
```python
"""Matplotlib figure generation for LIF channels."""

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from lif_figure.config import Config


# Color mappings for simple named colors
COLOR_MAP = {
    "gray": (1, 1, 1),
    "red": (1, 0, 0),
    "green": (0, 1, 0),
    "blue": (0, 0, 1),
    "cyan": (0, 1, 1),
    "magenta": (1, 0, 1),
    "yellow": (1, 1, 0),
    "lime": (0, 1, 0),
}


def normalize_channel(data: np.ndarray) -> np.ndarray:
    """Normalize channel data to 0-1 range.

    Args:
        data: 2D array of pixel values

    Returns:
        Normalized array with values in [0, 1]
    """
    data = data.astype(np.float64)
    min_val = data.min()
    max_val = data.max()

    if max_val == min_val:
        return np.zeros_like(data)

    return (data - min_val) / (max_val - min_val)


def apply_colormap(data: np.ndarray, color: str) -> np.ndarray:
    """Apply a color to normalized grayscale data.

    Args:
        data: 2D array with values in [0, 1]
        color: Color name (gray, red, green, blue, cyan, magenta, yellow)

    Returns:
        3D array with shape (H, W, 3) RGB values in [0, 1]
    """
    rgb_mult = COLOR_MAP.get(color.lower(), (1, 1, 1))

    result = np.zeros((*data.shape, 3), dtype=np.float64)
    for i, mult in enumerate(rgb_mult):
        result[..., i] = data * mult

    return result


def create_merge(colored_channels: list[np.ndarray]) -> np.ndarray:
    """Create merged image by additive blending.

    Args:
        colored_channels: List of (H, W, 3) RGB arrays

    Returns:
        Merged (H, W, 3) array, clipped to [0, 1]
    """
    result = np.zeros_like(colored_channels[0])
    for ch in colored_channels:
        result = result + ch

    return np.clip(result, 0, 1)


def calculate_scale_bar(pixel_size_um: float, image_width: int) -> tuple[int, str]:
    """Calculate appropriate scale bar length.

    Args:
        pixel_size_um: Size of one pixel in micrometers
        image_width: Width of image in pixels

    Returns:
        Tuple of (bar_length_pixels, label_text)
    """
    # Target ~10-20% of image width
    target_um = pixel_size_um * image_width * 0.15

    # Nice numbers for scale bars
    nice_values = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    bar_um = min(nice_values, key=lambda x: abs(x - target_um))

    bar_pixels = int(bar_um / pixel_size_um)

    if bar_um >= 1000:
        label = f"{bar_um // 1000} mm"
    else:
        label = f"{bar_um} µm"

    return bar_pixels, label


def build_figure(
    channels: np.ndarray,
    names: list[str],
    config: Config,
    pixel_size_um: Optional[float] = None,
) -> Figure:
    """Build a figure with channel panels and merge.

    Args:
        channels: Array with shape (C, H, W)
        names: List of channel names
        config: Configuration object
        pixel_size_um: Optional pixel size for scale bar

    Returns:
        Matplotlib Figure object
    """
    n_channels = channels.shape[0]
    n_panels = n_channels + 1  # channels + merge

    # Normalize all channels
    normalized = [normalize_channel(channels[i]) for i in range(n_channels)]

    # Apply colors
    colors = [config.get_color(names[i], i) for i in range(n_channels)]
    colored = [apply_colormap(normalized[i], colors[i]) for i in range(n_channels)]

    # Create merge
    merge = create_merge(colored)

    # Create figure
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), dpi=config.dpi)
    if n_panels == 1:
        axes = [axes]

    # Set background
    fig.patch.set_facecolor(config.background)

    # Calculate scale bar if pixel size available
    scale_bar_info = None
    if pixel_size_um is not None:
        scale_bar_info = calculate_scale_bar(pixel_size_um, channels.shape[2])

    # Plot each channel (grayscale)
    for i, (ax, name, norm_ch) in enumerate(zip(axes[:-1], names, normalized)):
        ax.imshow(norm_ch, cmap="gray", vmin=0, vmax=1)
        ax.set_title(name, fontsize=config.font_size, color="white" if config.background == "black" else "black")
        ax.axis("off")
        ax.set_facecolor(config.background)

        # Add scale bar
        if scale_bar_info:
            _add_scale_bar(ax, scale_bar_info, config)

    # Plot merge
    ax_merge = axes[-1]
    ax_merge.imshow(merge)
    ax_merge.set_title("Merge", fontsize=config.font_size, color="white" if config.background == "black" else "black")
    ax_merge.axis("off")
    ax_merge.set_facecolor(config.background)

    if scale_bar_info:
        _add_scale_bar(ax_merge, scale_bar_info, config)

    plt.tight_layout()
    return fig


def _add_scale_bar(ax, scale_bar_info: tuple[int, str], config: Config) -> None:
    """Add scale bar to an axis."""
    bar_pixels, label = scale_bar_info

    # Position in bottom-right
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x_start = xlim[1] - bar_pixels - 10
    x_end = xlim[1] - 10
    y_pos = ylim[0] - 10  # Note: y-axis is inverted for images

    bar_color = "white" if config.background == "black" else "black"

    ax.plot([x_start, x_end], [y_pos, y_pos], color=bar_color, linewidth=config.scale_bar_height)
    ax.text(
        (x_start + x_end) / 2, y_pos + 15,
        label,
        ha="center", va="top",
        fontsize=config.font_size - 2,
        color=bar_color,
    )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/lif_figure/test_figure.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(lif-figure): add figure module with panel building"
```

---

## Task 5: CLI Module

Command-line interface with click.

**Files:**
- Create: `src/lif_figure/cli.py`
- Create: `tests/lif_figure/test_cli.py`

**Step 1: Write failing tests for CLI**

Create `tests/lif_figure/test_cli.py`:
```python
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
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/lif_figure/test_cli.py -v`
Expected: FAIL with "No module named 'lif_figure.cli'"

**Step 3: Implement CLI module**

Create `src/lif_figure/cli.py`:
```python
"""Command-line interface for lif-figure."""

import re
import sys
from pathlib import Path
from typing import Optional

import click
import matplotlib.pyplot as plt

from lif_figure.config import load_config
from lif_figure.reader import list_series, read_series
from lif_figure.figure import build_figure


def sanitize_filename(name: str) -> str:
    """Sanitize a string for use as filename."""
    return re.sub(r'[/\\:*?"<>|]', "_", name)


@click.command()
@click.argument("input_file", type=click.Path(exists=False))
@click.option(
    "--channels", "-c",
    required=True,
    help="Channel names, comma-separated (e.g., 'DAPI,GFP,mCherry')",
)
@click.option(
    "--series", "-s",
    default=None,
    help="Series to process, comma-separated (default: all)",
)
@click.option(
    "--output", "-o",
    default="./figures",
    type=click.Path(),
    help="Output directory (default: ./figures)",
)
@click.option(
    "--zstack", "-z",
    default="max",
    help="Z-stack mode: max, max:START-END, frames, rows (default: max)",
)
@click.option(
    "--config",
    default=None,
    type=click.Path(exists=True),
    help="Optional YAML config file",
)
def main(
    input_file: str,
    channels: str,
    series: Optional[str],
    output: str,
    zstack: str,
    config: Optional[str],
) -> None:
    """Generate publication-ready figure panels from LIF files.

    Each series produces a PDF with grayscale channel panels and a color merge.
    """
    input_path = Path(input_file)
    output_path = Path(output)
    config_path = Path(config) if config else None

    # Validate input file
    if not input_path.exists():
        click.echo(f"Error: File not found: {input_file}", err=True)
        sys.exit(1)

    if not input_path.suffix.lower() == ".lif":
        click.echo(f"Error: Not a LIF file: {input_file}", err=True)
        sys.exit(1)

    # Parse channel names
    channel_names = [c.strip() for c in channels.split(",")]

    # Load config
    cfg = load_config(config_path)

    # Auto-detect config in current directory
    if config_path is None:
        auto_config = Path("lif-figure.yaml")
        if auto_config.exists():
            cfg = load_config(auto_config)

    # Get series to process
    all_series = list_series(input_path)

    if series:
        series_names = [s.strip() for s in series.split(",")]
        # Validate series names
        for name in series_names:
            if name not in all_series:
                click.echo(f"Error: Series '{name}' not found in {input_file}", err=True)
                click.echo(f"Available series: {', '.join(all_series)}", err=True)
                sys.exit(1)
    else:
        series_names = all_series

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    click.echo(f"Processing {input_path.name}")

    # Process each series
    for i, series_name in enumerate(series_names, 1):
        click.echo(f"  Series {i} of {len(series_names)}: \"{series_name}\"", nl=False)

        try:
            data, pixel_size_um = read_series(input_path, series_name, zstack)

            # Validate channel count
            if data.ndim == 3:  # (C, H, W)
                n_channels = data.shape[0]
            else:  # (Z, C, H, W) for frames/rows mode
                n_channels = data.shape[1]

            if n_channels != len(channel_names):
                click.echo(f" [SKIP: {n_channels} channels, expected {len(channel_names)}]")
                continue

            # Handle different Z-stack modes
            if data.ndim == 3:
                # Single figure (max projection)
                fig = build_figure(data, channel_names, cfg, pixel_size_um)

                safe_name = sanitize_filename(series_name)
                output_file = output_path / f"{safe_name}.pdf"
                fig.savefig(output_file, format="pdf", bbox_inches="tight", facecolor=cfg.background)
                plt.close(fig)

                click.echo(f" → {output_file}")

            elif zstack == "frames":
                # Separate PDF per Z frame
                subdir = output_path / sanitize_filename(series_name)
                subdir.mkdir(exist_ok=True)

                for z in range(data.shape[0]):
                    fig = build_figure(data[z], channel_names, cfg, pixel_size_um)
                    output_file = subdir / f"z{z:02d}.pdf"
                    fig.savefig(output_file, format="pdf", bbox_inches="tight", facecolor=cfg.background)
                    plt.close(fig)

                click.echo(f" → {subdir}/ ({data.shape[0]} frames)")

            elif zstack == "rows":
                # All Z frames as rows in single PDF
                # TODO: Implement multi-row figure
                click.echo(" [SKIP: rows mode not yet implemented]")
                continue

        except Exception as e:
            click.echo(f" [ERROR: {e}]")
            continue

    click.echo(f"Done. Figures saved to {output_path}/")


if __name__ == "__main__":
    main()
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/lif_figure/test_cli.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(lif-figure): add CLI with click"
```

---

## Task 6: Integration Test with Mock LIF

Test the full pipeline with mocked LIF reading.

**Files:**
- Create: `tests/lif_figure/test_integration.py`

**Step 1: Write integration test**

Create `tests/lif_figure/test_integration.py`:
```python
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
```

**Step 2: Run integration tests**

Run: `uv run pytest tests/lif_figure/test_integration.py -v`
Expected: All 2 tests PASS

**Step 3: Run full test suite**

Run: `uv run pytest tests/lif_figure/ -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add -A
git commit -m "test(lif-figure): add integration tests with mocked LIF"
```

---

## Task 7: Verify CLI Entry Point

Test that the installed CLI command works.

**Files:** None (verification only)

**Step 1: Sync and verify CLI is installed**

Run: `uv sync`
Run: `uv run lif-figure --help`

Expected output should include:
```
Usage: lif-figure [OPTIONS] INPUT_FILE

  Generate publication-ready figure panels from LIF files.

Options:
  -c, --channels TEXT  Channel names, comma-separated...
  -s, --series TEXT    Series to process...
  ...
```

**Step 2: Run all tests to confirm nothing broke**

Run: `uv run pytest tests/lif_figure/ -v`
Expected: All tests PASS

**Step 3: Final commit if any changes**

```bash
git status
# If pyproject.toml changed during uv sync:
git add pyproject.toml uv.lock
git commit -m "chore: update lockfile"
```

---

## Summary

After completing all tasks, you will have:

1. **lif_figure package** - New CLI tool alongside existing napari plugin
2. **config module** - YAML config loading with sensible defaults
3. **reader module** - LIF reading with Z-stack modes (max, frames, rows)
4. **figure module** - Matplotlib figure generation with labels and scale bars
5. **cli module** - Click-based CLI with all options from design
6. **Full test coverage** - Unit and integration tests

The CLI will be available as `lif-figure` command after `uv sync`.
