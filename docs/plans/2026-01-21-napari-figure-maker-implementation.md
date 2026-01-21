# napari-figure-maker Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a napari plugin that creates publication-ready multichannel figure panels from microscopy images.

**Architecture:** Napari serves as the image viewer and source of truth for contrast/colormap settings. The plugin adds a dock widget that captures the current view state, composites channels into a panel grid using numpy/Pillow, and exports at configurable DPI with scale bars and labels.

**Tech Stack:** Python, napari, Pillow, numpy, readlif, PyYAML, pytest

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/napari_figure_maker/__init__.py`
- Create: `src/napari_figure_maker/napari.yaml`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

**Step 1: Initialize uv project**

Run: `cd /Users/sean/code/napari-figure-maker && uv init --lib --name napari-figure-maker`
Expected: Creates basic pyproject.toml and src directory

**Step 2: Create pyproject.toml (overwrite uv default)**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "napari-figure-maker"
version = "0.1.0"
description = "Create publication-ready multichannel figure panels from microscopy images"
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
dependencies = [
    "napari>=0.4.18",
    "numpy",
    "pillow",
    "pyyaml",
    "readlif",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-qt",
]

[project.entry-points."napari.manifest"]
napari-figure-maker = "napari_figure_maker:napari.yaml"

[tool.hatch.build.targets.wheel]
packages = ["src/napari_figure_maker"]
```

**Step 3: Sync dependencies with uv**

Run: `cd /Users/sean/code/napari-figure-maker && uv sync --all-extras`
Expected: Creates uv.lock and installs all dependencies

**Step 4: Create package __init__.py**

```python
"""napari-figure-maker: Create multichannel figure panels from microscopy images."""

__version__ = "0.1.0"
```

**Step 5: Create napari.yaml manifest**

```yaml
name: napari-figure-maker
display_name: Figure Maker
contributions:
  widgets:
    - command: napari_figure_maker.make_figure_widget
      display_name: Figure Maker
```

**Step 6: Create test infrastructure**

`tests/__init__.py`: empty file

`tests/conftest.py`:
```python
"""Pytest configuration and fixtures."""

import numpy as np
import pytest


@pytest.fixture
def sample_channel_data():
    """Create sample single-channel image data."""
    return np.random.randint(0, 255, (100, 100), dtype=np.uint8)


@pytest.fixture
def sample_multichannel_data():
    """Create sample 3-channel image data."""
    return [
        np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        for _ in range(3)
    ]
```

**Step 7: Verify project structure**

Run: `ls -la src/napari_figure_maker/`
Expected: `__init__.py`, `napari.yaml`

**Step 8: Commit**

```bash
git add -A
git commit -m "chore: initial project scaffolding with napari plugin manifest"
```

**Note:** All subsequent test commands use `uv run pytest` to run within the uv-managed environment.

---

## Task 2: Configuration Data Models

**Files:**
- Create: `src/napari_figure_maker/models.py`
- Create: `tests/test_models.py`

**Step 1: Write the failing test for ChannelConfig**

`tests/test_models.py`:
```python
"""Tests for configuration data models."""

from napari_figure_maker.models import ChannelConfig


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
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/sean/code/napari-figure-maker && uv run pytest tests/test_models.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'napari_figure_maker.models'"

**Step 3: Write minimal ChannelConfig implementation**

`src/napari_figure_maker/models.py`:
```python
"""Configuration data models for napari-figure-maker."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChannelConfig:
    """Configuration for a single channel in the figure."""

    name: str
    visible: bool = True
    label: Optional[str] = None
    colormap: str = "gray"
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/sean/code/napari-figure-maker && uv run pytest tests/test_models.py -v`
Expected: PASS

**Step 5: Write failing test for FigureConfig**

Add to `tests/test_models.py`:
```python
from napari_figure_maker.models import ChannelConfig, FigureConfig


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
```

**Step 6: Run test to verify it fails**

Run: `cd /Users/sean/code/napari-figure-maker && uv run pytest tests/test_models.py::test_figure_config_defaults -v`
Expected: FAIL with "cannot import name 'FigureConfig'"

**Step 7: Write FigureConfig implementation**

Add to `src/napari_figure_maker/models.py`:
```python
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
```

**Step 8: Run tests to verify they pass**

Run: `cd /Users/sean/code/napari-figure-maker && uv run pytest tests/test_models.py -v`
Expected: All PASS

**Step 9: Commit**

```bash
git add -A
git commit -m "feat: add ChannelConfig and FigureConfig data models"
```

---

## Task 3: Scale Bar Rendering

**Files:**
- Create: `src/napari_figure_maker/scale_bar.py`
- Create: `tests/test_scale_bar.py`

**Step 1: Write failing test for scale bar length calculation**

`tests/test_scale_bar.py`:
```python
"""Tests for scale bar rendering."""

import numpy as np
from napari_figure_maker.scale_bar import calculate_nice_scale_bar_length


def test_calculate_nice_scale_bar_length_small():
    """Should pick nice round number for small images."""
    # Image is 100um wide, want bar ~20% of width = 20um
    # Should round to 20um
    length = calculate_nice_scale_bar_length(
        image_width_um=100.0,
        target_fraction=0.2,
    )
    assert length == 20.0


def test_calculate_nice_scale_bar_length_large():
    """Should pick nice round number for larger images."""
    # Image is 500um wide, want bar ~20% = 100um
    length = calculate_nice_scale_bar_length(
        image_width_um=500.0,
        target_fraction=0.2,
    )
    assert length == 100.0


def test_calculate_nice_scale_bar_length_awkward():
    """Should round to nearest nice number."""
    # Image is 73um wide, want bar ~20% = 14.6um
    # Should round to 10 or 15
    length = calculate_nice_scale_bar_length(
        image_width_um=73.0,
        target_fraction=0.2,
    )
    assert length in [10.0, 15.0, 20.0]
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/sean/code/napari-figure-maker && uv run pytest tests/test_scale_bar.py::test_calculate_nice_scale_bar_length_small -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement scale bar length calculation**

`src/napari_figure_maker/scale_bar.py`:
```python
"""Scale bar rendering utilities."""

import math
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def calculate_nice_scale_bar_length(
    image_width_um: float,
    target_fraction: float = 0.2,
) -> float:
    """Calculate a nice round scale bar length.

    Args:
        image_width_um: Width of image in micrometers.
        target_fraction: Target scale bar length as fraction of image width.

    Returns:
        Nice round number for scale bar length in micrometers.
    """
    target_length = image_width_um * target_fraction

    # Find order of magnitude
    if target_length <= 0:
        return 1.0

    magnitude = 10 ** math.floor(math.log10(target_length))
    normalized = target_length / magnitude

    # Pick nearest nice number: 1, 2, 5, 10
    nice_numbers = [1, 2, 5, 10]
    nice = min(nice_numbers, key=lambda x: abs(x - normalized))

    return nice * magnitude
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/sean/code/napari-figure-maker && uv run pytest tests/test_scale_bar.py -v`
Expected: All PASS

**Step 5: Write failing test for scale bar rendering**

Add to `tests/test_scale_bar.py`:
```python
from PIL import Image
from napari_figure_maker.scale_bar import render_scale_bar


def test_render_scale_bar_returns_image():
    """render_scale_bar should return a PIL Image."""
    bar = render_scale_bar(
        length_um=10.0,
        pixel_size_um=0.5,
        color="white",
        font_size=10,
    )

    assert isinstance(bar, Image.Image)
    assert bar.mode == "RGBA"


def test_render_scale_bar_correct_width():
    """Scale bar image width should match physical length."""
    bar = render_scale_bar(
        length_um=10.0,
        pixel_size_um=0.5,  # 0.5um per pixel = 20 pixels for 10um
        color="white",
        font_size=10,
    )

    # Bar should be 20 pixels wide (plus some padding for text)
    assert bar.width >= 20
```

**Step 6: Run test to verify it fails**

Run: `cd /Users/sean/code/napari-figure-maker && uv run pytest tests/test_scale_bar.py::test_render_scale_bar_returns_image -v`
Expected: FAIL with "cannot import name 'render_scale_bar'"

**Step 7: Implement scale bar rendering**

Add to `src/napari_figure_maker/scale_bar.py`:
```python
def render_scale_bar(
    length_um: float,
    pixel_size_um: float,
    color: str = "white",
    font_size: int = 10,
    bar_height: int = 4,
    padding: int = 5,
) -> Image.Image:
    """Render a scale bar with label as a PIL Image.

    Args:
        length_um: Length of scale bar in micrometers.
        pixel_size_um: Size of one pixel in micrometers.
        color: Color of bar and text.
        font_size: Font size for label.
        bar_height: Height of the bar in pixels.
        padding: Padding around bar and text.

    Returns:
        RGBA PIL Image containing the scale bar and label.
    """
    bar_width_px = int(length_um / pixel_size_um)

    # Format label
    if length_um >= 1000:
        label = f"{length_um / 1000:.0f} mm"
    elif length_um >= 1:
        label = f"{length_um:.0f} µm"
    else:
        label = f"{length_um * 1000:.0f} nm"

    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("Arial", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Measure text
    dummy_img = Image.new("RGBA", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    text_bbox = dummy_draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Calculate image size
    img_width = max(bar_width_px, text_width) + 2 * padding
    img_height = bar_height + text_height + 3 * padding

    # Create image
    img = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw bar (centered horizontally)
    bar_x = (img_width - bar_width_px) // 2
    bar_y = padding
    draw.rectangle(
        [bar_x, bar_y, bar_x + bar_width_px, bar_y + bar_height],
        fill=color,
    )

    # Draw text (centered below bar)
    text_x = (img_width - text_width) // 2
    text_y = bar_y + bar_height + padding
    draw.text((text_x, text_y), label, fill=color, font=font)

    return img
```

**Step 8: Run tests to verify they pass**

Run: `cd /Users/sean/code/napari-figure-maker && uv run pytest tests/test_scale_bar.py -v`
Expected: All PASS

**Step 9: Commit**

```bash
git add -A
git commit -m "feat: add scale bar calculation and rendering"
```

---

## Task 4: Figure Builder Core

**Files:**
- Create: `src/napari_figure_maker/figure_builder.py`
- Create: `tests/test_figure_builder.py`

**Step 1: Write failing test for channel rendering**

`tests/test_figure_builder.py`:
```python
"""Tests for figure builder."""

import numpy as np
from PIL import Image
from napari_figure_maker.figure_builder import render_channel_to_rgb


def test_render_channel_to_rgb_gray():
    """Should render grayscale channel to RGB."""
    data = np.array([[0, 128], [255, 64]], dtype=np.uint8)

    rgb = render_channel_to_rgb(
        data=data,
        colormap="gray",
        contrast_limits=(0, 255),
    )

    assert isinstance(rgb, np.ndarray)
    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == np.uint8
    # Gray: R=G=B
    assert np.array_equal(rgb[0, 0], [0, 0, 0])
    assert np.array_equal(rgb[0, 1], [128, 128, 128])


def test_render_channel_to_rgb_green():
    """Should render channel with green colormap."""
    data = np.array([[0, 255]], dtype=np.uint8)

    rgb = render_channel_to_rgb(
        data=data,
        colormap="green",
        contrast_limits=(0, 255),
    )

    # Green: R=0, G=value, B=0
    assert rgb[0, 0, 1] == 0  # G at min
    assert rgb[0, 1, 1] == 255  # G at max
    assert rgb[0, 1, 0] == 0  # R should be 0
    assert rgb[0, 1, 2] == 0  # B should be 0


def test_render_channel_contrast_limits():
    """Should apply contrast limits correctly."""
    data = np.array([[50, 100, 150, 200]], dtype=np.uint8)

    rgb = render_channel_to_rgb(
        data=data,
        colormap="gray",
        contrast_limits=(100, 200),  # Map 100->0, 200->255
    )

    # Value 50 is below min, should be 0
    assert rgb[0, 0, 0] == 0
    # Value 100 is at min, should be 0
    assert rgb[0, 1, 0] == 0
    # Value 200 is at max, should be 255
    assert rgb[0, 3, 0] == 255
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/sean/code/napari-figure-maker && uv run pytest tests/test_figure_builder.py::test_render_channel_to_rgb_gray -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement channel rendering**

`src/napari_figure_maker/figure_builder.py`:
```python
"""Figure building and composition utilities."""

from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Simple colormaps as RGB tuples (will be multiplied by intensity)
COLORMAPS = {
    "gray": (1.0, 1.0, 1.0),
    "red": (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
    "cyan": (0.0, 1.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
    "yellow": (1.0, 1.0, 0.0),
}


def render_channel_to_rgb(
    data: np.ndarray,
    colormap: str,
    contrast_limits: Tuple[float, float],
) -> np.ndarray:
    """Render a single channel to RGB using colormap and contrast limits.

    Args:
        data: 2D numpy array of image data.
        colormap: Name of colormap to apply.
        contrast_limits: (min, max) values for contrast adjustment.

    Returns:
        3D numpy array (H, W, 3) of uint8 RGB values.
    """
    cmin, cmax = contrast_limits

    # Normalize to 0-1 based on contrast limits
    if cmax > cmin:
        normalized = (data.astype(np.float32) - cmin) / (cmax - cmin)
    else:
        normalized = np.zeros_like(data, dtype=np.float32)

    # Clip to 0-1
    normalized = np.clip(normalized, 0, 1)

    # Apply colormap
    color = COLORMAPS.get(colormap, COLORMAPS["gray"])

    rgb = np.zeros((*data.shape, 3), dtype=np.float32)
    rgb[..., 0] = normalized * color[0]
    rgb[..., 1] = normalized * color[1]
    rgb[..., 2] = normalized * color[2]

    # Convert to uint8
    return (rgb * 255).astype(np.uint8)
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/sean/code/napari-figure-maker && uv run pytest tests/test_figure_builder.py -v`
Expected: All PASS

**Step 5: Write failing test for merge composite**

Add to `tests/test_figure_builder.py`:
```python
from napari_figure_maker.figure_builder import create_merge_composite


def test_create_merge_composite():
    """Should merge multiple RGB channels additively."""
    # Red channel
    red = np.zeros((2, 2, 3), dtype=np.uint8)
    red[0, 0] = [255, 0, 0]

    # Green channel
    green = np.zeros((2, 2, 3), dtype=np.uint8)
    green[0, 0] = [0, 255, 0]

    merged = create_merge_composite([red, green])

    # Should be yellow where both overlap
    assert merged[0, 0, 0] == 255  # R
    assert merged[0, 0, 1] == 255  # G
    assert merged[0, 0, 2] == 0    # B


def test_create_merge_composite_clipping():
    """Should clip values that exceed 255."""
    ch1 = np.full((2, 2, 3), 200, dtype=np.uint8)
    ch2 = np.full((2, 2, 3), 200, dtype=np.uint8)

    merged = create_merge_composite([ch1, ch2])

    # Should be clipped to 255, not overflow
    assert merged[0, 0, 0] == 255
```

**Step 6: Run test to verify it fails**

Run: `cd /Users/sean/code/napari-figure-maker && uv run pytest tests/test_figure_builder.py::test_create_merge_composite -v`
Expected: FAIL with "cannot import name 'create_merge_composite'"

**Step 7: Implement merge composite**

Add to `src/napari_figure_maker/figure_builder.py`:
```python
def create_merge_composite(channels: List[np.ndarray]) -> np.ndarray:
    """Create a merged composite from multiple RGB channel images.

    Args:
        channels: List of RGB numpy arrays (H, W, 3), all same shape.

    Returns:
        Merged RGB numpy array (H, W, 3) with additive blending.
    """
    if not channels:
        raise ValueError("At least one channel required")

    # Use float to avoid overflow during addition
    merged = np.zeros_like(channels[0], dtype=np.float32)

    for ch in channels:
        merged += ch.astype(np.float32)

    # Clip and convert back to uint8
    return np.clip(merged, 0, 255).astype(np.uint8)
```

**Step 8: Run tests to verify they pass**

Run: `cd /Users/sean/code/napari-figure-maker && uv run pytest tests/test_figure_builder.py -v`
Expected: All PASS

**Step 9: Commit**

```bash
git add -A
git commit -m "feat: add channel rendering and merge composite functions"
```

---

## Task 5: Panel Grid Layout

**Files:**
- Modify: `src/napari_figure_maker/figure_builder.py`
- Modify: `tests/test_figure_builder.py`

**Step 1: Write failing test for panel grid**

Add to `tests/test_figure_builder.py`:
```python
from napari_figure_maker.figure_builder import arrange_panels_in_grid


def test_arrange_panels_single_row():
    """Should arrange panels in a single row by default."""
    panels = [
        np.zeros((100, 100, 3), dtype=np.uint8),
        np.full((100, 100, 3), 128, dtype=np.uint8),
        np.full((100, 100, 3), 255, dtype=np.uint8),
    ]

    result = arrange_panels_in_grid(
        panels=panels,
        gap_fraction=0.0,  # No gap for easier testing
        background_color="black",
    )

    # Should be 100 high, 300 wide (3 panels)
    assert result.shape == (100, 300, 3)
    # First panel is black
    assert result[50, 50, 0] == 0
    # Second panel is gray
    assert result[50, 150, 0] == 128
    # Third panel is white
    assert result[50, 250, 0] == 255


def test_arrange_panels_with_gap():
    """Should add gaps between panels."""
    panels = [
        np.zeros((100, 100, 3), dtype=np.uint8),
        np.zeros((100, 100, 3), dtype=np.uint8),
    ]

    result = arrange_panels_in_grid(
        panels=panels,
        gap_fraction=0.1,  # 10% of panel width = 10px gap
        background_color="white",
    )

    # Width: 100 + 10 + 100 = 210
    assert result.shape == (100, 210, 3)
    # Gap should be white (background)
    assert result[50, 105, 0] == 255
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/sean/code/napari-figure-maker && uv run pytest tests/test_figure_builder.py::test_arrange_panels_single_row -v`
Expected: FAIL with "cannot import name 'arrange_panels_in_grid'"

**Step 3: Implement panel grid layout**

Add to `src/napari_figure_maker/figure_builder.py`:
```python
def arrange_panels_in_grid(
    panels: List[np.ndarray],
    gap_fraction: float = 0.02,
    background_color: str = "black",
    columns: Optional[int] = None,
) -> np.ndarray:
    """Arrange panel images in a grid layout.

    Args:
        panels: List of RGB numpy arrays, all same shape.
        gap_fraction: Gap between panels as fraction of panel width.
        background_color: Background color for gaps ("black" or "white").
        columns: Number of columns. None = single row.

    Returns:
        RGB numpy array of the complete grid.
    """
    if not panels:
        raise ValueError("At least one panel required")

    panel_height, panel_width = panels[0].shape[:2]
    n_panels = len(panels)

    # Calculate grid dimensions
    if columns is None:
        n_cols = n_panels
        n_rows = 1
    else:
        n_cols = columns
        n_rows = (n_panels + columns - 1) // columns

    # Calculate gap in pixels
    gap_px = int(panel_width * gap_fraction)

    # Calculate total dimensions
    total_width = n_cols * panel_width + (n_cols - 1) * gap_px
    total_height = n_rows * panel_height + (n_rows - 1) * gap_px

    # Create background
    bg_value = 255 if background_color == "white" else 0
    grid = np.full((total_height, total_width, 3), bg_value, dtype=np.uint8)

    # Place panels
    for i, panel in enumerate(panels):
        row = i // n_cols
        col = i % n_cols

        y = row * (panel_height + gap_px)
        x = col * (panel_width + gap_px)

        grid[y:y + panel_height, x:x + panel_width] = panel

    return grid
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/sean/code/napari-figure-maker && uv run pytest tests/test_figure_builder.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add panel grid layout function"
```

---

## Task 6: Add Labels to Panels

**Files:**
- Modify: `src/napari_figure_maker/figure_builder.py`
- Modify: `tests/test_figure_builder.py`

**Step 1: Write failing test for label rendering**

Add to `tests/test_figure_builder.py`:
```python
from PIL import Image
from napari_figure_maker.figure_builder import add_label_to_panel


def test_add_label_to_panel():
    """Should add text label to panel."""
    panel = np.zeros((100, 100, 3), dtype=np.uint8)

    labeled = add_label_to_panel(
        panel=panel,
        label="DAPI",
        position="top-left",
        font_size=12,
        color="white",
    )

    assert isinstance(labeled, np.ndarray)
    assert labeled.shape == panel.shape
    # Top-left corner should no longer be pure black (has text)
    # Check a small region where text should be
    top_left_region = labeled[5:20, 5:50]
    assert top_left_region.max() > 0  # Some white pixels from text
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/sean/code/napari-figure-maker && uv run pytest tests/test_figure_builder.py::test_add_label_to_panel -v`
Expected: FAIL with "cannot import name 'add_label_to_panel'"

**Step 3: Implement label rendering**

Add to `src/napari_figure_maker/figure_builder.py`:
```python
def add_label_to_panel(
    panel: np.ndarray,
    label: str,
    position: str = "top-left",
    font_size: int = 12,
    color: str = "white",
    padding: int = 5,
) -> np.ndarray:
    """Add a text label to a panel image.

    Args:
        panel: RGB numpy array.
        label: Text to add.
        position: Where to place label ("top-left", "top-center", "bottom-left").
        font_size: Font size in pixels.
        color: Text color.
        padding: Padding from edge in pixels.

    Returns:
        New RGB numpy array with label added.
    """
    # Convert to PIL for text rendering
    img = Image.fromarray(panel)
    draw = ImageDraw.Draw(img)

    # Try to load font
    try:
        font = ImageFont.truetype("Arial", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Get text size
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Calculate position
    if position == "top-left":
        x, y = padding, padding
    elif position == "top-center":
        x = (panel.shape[1] - text_width) // 2
        y = padding
    elif position == "bottom-left":
        x = padding
        y = panel.shape[0] - text_height - padding
    else:
        x, y = padding, padding

    # Draw text
    draw.text((x, y), label, fill=color, font=font)

    return np.array(img)
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/sean/code/napari-figure-maker && uv run pytest tests/test_figure_builder.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add text label rendering for panels"
```

---

## Task 7: Complete Figure Builder

**Files:**
- Modify: `src/napari_figure_maker/figure_builder.py`
- Modify: `tests/test_figure_builder.py`

**Step 1: Write failing test for build_figure**

Add to `tests/test_figure_builder.py`:
```python
from napari_figure_maker.figure_builder import build_figure
from napari_figure_maker.models import ChannelConfig, FigureConfig


def test_build_figure_basic():
    """Should build a complete figure with channels and merge."""
    # Two simple channels
    channels_data = [
        np.full((50, 50), 100, dtype=np.uint8),  # Channel 1
        np.full((50, 50), 200, dtype=np.uint8),  # Channel 2
    ]

    channel_configs = [
        ChannelConfig(name="Ch1", colormap="green"),
        ChannelConfig(name="Ch2", colormap="magenta"),
    ]

    figure_config = FigureConfig(
        show_merge=True,
        panel_gap_fraction=0.0,
    )

    result = build_figure(
        channels_data=channels_data,
        channel_configs=channel_configs,
        figure_config=figure_config,
        pixel_size_um=None,  # No scale bar
    )

    assert isinstance(result, np.ndarray)
    # 3 panels: Ch1, Ch2, Merge
    assert result.shape[1] == 50 * 3  # Width = 3 panels
    assert result.shape[0] == 50  # Height = 1 panel


def test_build_figure_no_merge():
    """Should build figure without merge panel."""
    channels_data = [
        np.full((50, 50), 100, dtype=np.uint8),
        np.full((50, 50), 200, dtype=np.uint8),
    ]

    channel_configs = [
        ChannelConfig(name="Ch1", colormap="green"),
        ChannelConfig(name="Ch2", colormap="magenta"),
    ]

    figure_config = FigureConfig(
        show_merge=False,
        panel_gap_fraction=0.0,
    )

    result = build_figure(
        channels_data=channels_data,
        channel_configs=channel_configs,
        figure_config=figure_config,
        pixel_size_um=None,
    )

    # 2 panels only (no merge)
    assert result.shape[1] == 50 * 2
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/sean/code/napari-figure-maker && uv run pytest tests/test_figure_builder.py::test_build_figure_basic -v`
Expected: FAIL with "cannot import name 'build_figure'"

**Step 3: Implement build_figure**

Add to `src/napari_figure_maker/figure_builder.py`:
```python
from napari_figure_maker.models import ChannelConfig, FigureConfig
from napari_figure_maker.scale_bar import calculate_nice_scale_bar_length, render_scale_bar


def build_figure(
    channels_data: List[np.ndarray],
    channel_configs: List[ChannelConfig],
    figure_config: FigureConfig,
    pixel_size_um: Optional[float] = None,
    contrast_limits: Optional[List[Tuple[float, float]]] = None,
) -> np.ndarray:
    """Build a complete figure from channel data.

    Args:
        channels_data: List of 2D numpy arrays, one per channel.
        channel_configs: Configuration for each channel.
        figure_config: Overall figure configuration.
        pixel_size_um: Pixel size in micrometers (for scale bar).
        contrast_limits: Contrast limits per channel. If None, uses data min/max.

    Returns:
        RGB numpy array of the complete figure.
    """
    if len(channels_data) != len(channel_configs):
        raise ValueError("channels_data and channel_configs must have same length")

    # Default contrast limits
    if contrast_limits is None:
        contrast_limits = [(d.min(), d.max()) for d in channels_data]

    # Render each channel to RGB
    rgb_channels = []
    for data, config, clim in zip(channels_data, channel_configs, contrast_limits):
        if config.visible:
            rgb = render_channel_to_rgb(data, config.colormap, clim)
            rgb_channels.append(rgb)

    # Build panels list
    panels = []
    visible_configs = [c for c in channel_configs if c.visible]

    for rgb, config in zip(rgb_channels, visible_configs):
        # Add label if configured
        label = config.label or config.name
        panel = add_label_to_panel(
            rgb,
            label=label,
            position=figure_config.label_position,
            font_size=figure_config.label_font_size,
            color=figure_config.label_color or "white",
        )
        panels.append(panel)

    # Create merge if requested
    if figure_config.show_merge and len(rgb_channels) > 1:
        merge = create_merge_composite(rgb_channels)
        merge = add_label_to_panel(
            merge,
            label="Merge",
            position=figure_config.label_position,
            font_size=figure_config.label_font_size,
            color=figure_config.label_color or "white",
        )
        panels.append(merge)

    # Arrange in grid
    grid = arrange_panels_in_grid(
        panels,
        gap_fraction=figure_config.panel_gap_fraction,
        background_color=figure_config.background_color,
        columns=figure_config.grid_columns,
    )

    # Add scale bar if pixel size is known
    if pixel_size_um is not None:
        grid = _add_scale_bar_to_figure(grid, pixel_size_um, figure_config)

    return grid


def _add_scale_bar_to_figure(
    figure: np.ndarray,
    pixel_size_um: float,
    config: FigureConfig,
) -> np.ndarray:
    """Add scale bar to bottom-right of figure."""
    image_width_um = figure.shape[1] * pixel_size_um

    # Calculate nice scale bar length
    if config.scale_bar_length_um is not None:
        bar_length = config.scale_bar_length_um
    else:
        bar_length = calculate_nice_scale_bar_length(image_width_um)

    # Render scale bar
    bar_img = render_scale_bar(
        length_um=bar_length,
        pixel_size_um=pixel_size_um,
        color=config.scale_bar_color,
        font_size=config.scale_bar_font_size,
    )

    # Convert figure to PIL
    fig_pil = Image.fromarray(figure)

    # Calculate position (bottom-right with padding)
    padding = 10
    x = figure.shape[1] - bar_img.width - padding
    y = figure.shape[0] - bar_img.height - padding

    # Paste scale bar (with alpha)
    fig_pil.paste(bar_img, (x, y), bar_img)

    return np.array(fig_pil)
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/sean/code/napari-figure-maker && uv run pytest tests/test_figure_builder.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add complete build_figure function with merge and scale bar"
```

---

## Task 8: Presets System

**Files:**
- Create: `src/napari_figure_maker/presets.py`
- Create: `tests/test_presets.py`

**Step 1: Write failing test for preset save/load**

`tests/test_presets.py`:
```python
"""Tests for presets system."""

import tempfile
from pathlib import Path

from napari_figure_maker.models import ChannelConfig, FigureConfig
from napari_figure_maker.presets import save_preset, load_preset


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
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/sean/code/napari-figure-maker && uv run pytest tests/test_presets.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement preset save/load**

`src/napari_figure_maker/presets.py`:
```python
"""Preset management for napari-figure-maker."""

from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

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
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/sean/code/napari-figure-maker && uv run pytest tests/test_presets.py -v`
Expected: All PASS

**Step 5: Write failing test for preset directory management**

Add to `tests/test_presets.py`:
```python
from napari_figure_maker.presets import get_preset_directory, list_presets


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
```

**Step 6: Run test to verify it fails**

Run: `cd /Users/sean/code/napari-figure-maker && uv run pytest tests/test_presets.py::test_get_preset_directory -v`
Expected: FAIL with "cannot import name 'get_preset_directory'"

**Step 7: Implement preset directory functions**

Add to `src/napari_figure_maker/presets.py`:
```python
import os
from typing import Dict, List


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
```

**Step 8: Run tests to verify they pass**

Run: `cd /Users/sean/code/napari-figure-maker && uv run pytest tests/test_presets.py -v`
Expected: All PASS

**Step 9: Commit**

```bash
git add -A
git commit -m "feat: add preset save/load and directory management"
```

---

## Task 9: Export Functions

**Files:**
- Create: `src/napari_figure_maker/exporter.py`
- Create: `tests/test_exporter.py`

**Step 1: Write failing test for PNG export**

`tests/test_exporter.py`:
```python
"""Tests for export functionality."""

import numpy as np
from pathlib import Path
from PIL import Image

from napari_figure_maker.exporter import export_figure


def test_export_figure_png(tmp_path):
    """Should export figure as PNG."""
    figure = np.full((100, 200, 3), 128, dtype=np.uint8)
    output_path = tmp_path / "test_figure.png"

    export_figure(figure, output_path, dpi=300)

    assert output_path.exists()

    # Verify it's a valid PNG
    img = Image.open(output_path)
    assert img.format == "PNG"
    assert img.size == (200, 100)  # Width, Height


def test_export_figure_tiff(tmp_path):
    """Should export figure as TIFF."""
    figure = np.full((100, 200, 3), 128, dtype=np.uint8)
    output_path = tmp_path / "test_figure.tiff"

    export_figure(figure, output_path, dpi=300)

    assert output_path.exists()

    img = Image.open(output_path)
    assert img.format == "TIFF"
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/sean/code/napari-figure-maker && uv run pytest tests/test_exporter.py::test_export_figure_png -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Implement export function**

`src/napari_figure_maker/exporter.py`:
```python
"""Export functionality for napari-figure-maker."""

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def export_figure(
    figure: np.ndarray,
    path: Path,
    dpi: int = 300,
) -> None:
    """Export a figure to an image file.

    Args:
        figure: RGB numpy array of the figure.
        path: Output file path. Format determined by extension.
        dpi: Resolution in dots per inch.
    """
    path = Path(path)
    img = Image.fromarray(figure)

    # Set DPI metadata
    img.info["dpi"] = (dpi, dpi)

    # Determine format from extension
    ext = path.suffix.lower()

    if ext in [".tif", ".tiff"]:
        img.save(path, format="TIFF", dpi=(dpi, dpi))
    elif ext == ".png":
        # PNG uses pixels per meter, convert from DPI
        ppm = int(dpi / 0.0254)
        img.save(path, format="PNG", dpi=(dpi, dpi), pnginfo=None)
        # Re-save with proper metadata using PIL's approach
        img.save(path, format="PNG", dpi=(dpi, dpi))
    else:
        # Default to PNG
        img.save(path, format="PNG", dpi=(dpi, dpi))


def copy_to_clipboard(figure: np.ndarray) -> bool:
    """Copy figure to system clipboard.

    Args:
        figure: RGB numpy array of the figure.

    Returns:
        True if successful, False otherwise.
    """
    try:
        from PIL import Image
        import io

        img = Image.fromarray(figure)

        # Platform-specific clipboard handling
        import sys
        if sys.platform == "darwin":  # macOS
            import subprocess
            # Convert to PNG bytes
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)

            # Use pbcopy with TIFF (macOS clipboard prefers TIFF)
            process = subprocess.Popen(
                ["osascript", "-e", 'set the clipboard to (read (POSIX file "/dev/stdin") as TIFF picture)'],
                stdin=subprocess.PIPE,
            )
            # Actually, simpler approach - save to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                img.save(f.name, format="PNG")
                subprocess.run([
                    "osascript", "-e",
                    f'set the clipboard to (read (POSIX file "{f.name}") as «class PNGf»)'
                ])
            return True
        else:
            # For other platforms, try pyperclip or similar
            # For now, return False to indicate not implemented
            return False
    except Exception:
        return False
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/sean/code/napari-figure-maker && uv run pytest tests/test_exporter.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add figure export to PNG and TIFF"
```

---

## Task 10: LIF Reader (napari plugin)

**Files:**
- Create: `src/napari_figure_maker/lif_reader.py`
- Create: `tests/test_lif_reader.py`
- Modify: `src/napari_figure_maker/napari.yaml`

**Step 1: Write failing test for LIF series listing**

`tests/test_lif_reader.py`:
```python
"""Tests for LIF file reading."""

import pytest
from pathlib import Path

# Note: These tests require a sample LIF file
# We'll create a mock for unit testing

from napari_figure_maker.lif_reader import get_lif_series_info


def test_get_lif_series_info_returns_list():
    """Should return list of series info dicts."""
    # This test will be skipped if no sample LIF file available
    # For now, test the function signature with a mock
    pytest.skip("Requires sample LIF file - integration test")
```

**Step 2: Implement LIF reader**

`src/napari_figure_maker/lif_reader.py`:
```python
"""LIF file reading utilities for napari-figure-maker."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


def get_lif_series_info(path: Path) -> List[Dict[str, Any]]:
    """Get information about all series in a LIF file.

    Args:
        path: Path to LIF file.

    Returns:
        List of dicts with series metadata (name, dimensions, channels, etc.)
    """
    from readlif.reader import LifFile

    lif = LifFile(str(path))
    series_info = []

    for i, image in enumerate(lif.get_iter_image()):
        info = {
            "index": i,
            "name": image.name,
            "dims": image.dims,  # NamedTuple with x, y, z, t, m
            "channels": image.channels,
            "bit_depth": image.bit_depth,
            "scale": image.scale,  # Tuple of (x, y, z) scale in meters
        }
        series_info.append(info)

    return series_info


def read_lif_series(
    path: Path,
    series_index: int = 0,
    z_index: Optional[int] = None,
    t_index: Optional[int] = None,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Read a series from a LIF file.

    Args:
        path: Path to LIF file.
        series_index: Which series to read.
        z_index: Specific Z slice (None = middle slice if Z stack).
        t_index: Specific timepoint (None = first timepoint).

    Returns:
        Tuple of (list of channel arrays, metadata dict).
    """
    from readlif.reader import LifFile

    lif = LifFile(str(path))
    images = list(lif.get_iter_image())

    if series_index >= len(images):
        raise IndexError(f"Series index {series_index} out of range (file has {len(images)} series)")

    image = images[series_index]

    # Handle Z selection
    z_size = image.dims.z
    if z_index is None and z_size > 1:
        z_index = z_size // 2  # Middle slice
    elif z_index is None:
        z_index = 0

    # Handle T selection
    t_size = image.dims.t
    if t_index is None:
        t_index = 0

    # Read each channel
    channels = []
    for c in range(image.channels):
        # readlif uses get_frame(z, t, c) ordering
        frame = image.get_frame(z=z_index, t=t_index, c=c)
        channels.append(np.array(frame))

    # Build metadata
    metadata = {
        "name": image.name,
        "pixel_size_um": image.scale[0] * 1e6 if image.scale else None,  # Convert m to um
        "z_index": z_index,
        "t_index": t_index,
        "channel_names": [f"Channel {i}" for i in range(image.channels)],  # LIF doesn't always have names
    }

    return channels, metadata


def napari_get_reader(path):
    """napari reader plugin hook.

    Args:
        path: Path to file.

    Returns:
        Reader function if this is a LIF file, None otherwise.
    """
    if isinstance(path, str):
        path = Path(path)

    if isinstance(path, Path) and path.suffix.lower() == ".lif":
        return lif_reader_function

    return None


def lif_reader_function(path):
    """Read a LIF file and return napari layer data.

    Args:
        path: Path to LIF file.

    Returns:
        List of layer data tuples for napari.
    """
    path = Path(path)
    channels, metadata = read_lif_series(path, series_index=0)

    layers = []
    pixel_size = metadata.get("pixel_size_um")

    for i, channel_data in enumerate(channels):
        name = f"{metadata['name']} - Ch{i}"

        layer_kwargs = {
            "name": name,
            "metadata": {
                "pixel_size_um": pixel_size,
                "source_file": str(path),
                "channel_index": i,
            },
        }

        # Add scale if available
        if pixel_size:
            layer_kwargs["scale"] = (pixel_size, pixel_size)

        layers.append((channel_data, layer_kwargs, "image"))

    return layers
```

**Step 3: Update napari.yaml to register reader**

Update `src/napari_figure_maker/napari.yaml`:
```yaml
name: napari-figure-maker
display_name: Figure Maker
contributions:
  commands:
    - id: napari_figure_maker.make_figure_widget
      python_name: napari_figure_maker._widget:FigureMakerWidget
      title: Open Figure Maker
    - id: napari_figure_maker.read_lif
      python_name: napari_figure_maker.lif_reader:napari_get_reader
      title: Read LIF file
  widgets:
    - command: napari_figure_maker.make_figure_widget
      display_name: Figure Maker
  readers:
    - command: napari_figure_maker.read_lif
      accepts_directories: false
      filename_patterns:
        - "*.lif"
```

**Step 4: Commit**

```bash
git add -A
git commit -m "feat: add LIF file reader with napari plugin integration"
```

---

## Task 11: Main Widget UI

**Files:**
- Create: `src/napari_figure_maker/_widget.py`
- Create: `tests/test_widget.py`

**Step 1: Write basic widget test**

`tests/test_widget.py`:
```python
"""Tests for the napari widget."""

import pytest

# Widget tests require napari and Qt, which need special handling
pytest.importorskip("napari")


def test_widget_creates(make_napari_viewer):
    """Widget should instantiate without error."""
    from napari_figure_maker._widget import FigureMakerWidget

    viewer = make_napari_viewer()
    widget = FigureMakerWidget(viewer)

    assert widget is not None
```

**Step 2: Implement the widget**

`src/napari_figure_maker/_widget.py`:
```python
"""Main napari widget for figure generation."""

from pathlib import Path
from typing import List, Optional

import numpy as np
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QSpinBox,
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QScrollArea,
    QLineEdit,
)
from qtpy.QtCore import Qt

from napari_figure_maker.models import ChannelConfig, FigureConfig
from napari_figure_maker.figure_builder import build_figure
from napari_figure_maker.exporter import export_figure, copy_to_clipboard
from napari_figure_maker.presets import list_presets, load_preset, save_preset, get_preset_directory


class ChannelConfigWidget(QWidget):
    """Widget for configuring a single channel."""

    COLORMAPS = ["gray", "red", "green", "blue", "cyan", "magenta", "yellow"]

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.channel_name = name
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Visibility checkbox
        self.visible_cb = QCheckBox()
        self.visible_cb.setChecked(True)
        layout.addWidget(self.visible_cb)

        # Channel name label
        self.name_label = QLabel(self.channel_name)
        self.name_label.setMinimumWidth(80)
        layout.addWidget(self.name_label)

        # Custom label input
        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("Label (optional)")
        self.label_edit.setMaximumWidth(100)
        layout.addWidget(self.label_edit)

        # Colormap selector
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(self.COLORMAPS)
        layout.addWidget(self.colormap_combo)

        layout.addStretch()

    def get_config(self) -> ChannelConfig:
        """Get the current channel configuration."""
        label = self.label_edit.text() or None
        return ChannelConfig(
            name=self.channel_name,
            visible=self.visible_cb.isChecked(),
            label=label,
            colormap=self.colormap_combo.currentText(),
        )

    def set_config(self, config: ChannelConfig):
        """Set the channel configuration."""
        self.visible_cb.setChecked(config.visible)
        self.label_edit.setText(config.label or "")
        idx = self.colormap_combo.findText(config.colormap)
        if idx >= 0:
            self.colormap_combo.setCurrentIndex(idx)


class FigureMakerWidget(QWidget):
    """Main widget for generating figures from napari layers."""

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.channel_widgets: List[ChannelConfigWidget] = []

        self._setup_ui()
        self._connect_signals()
        self._refresh_channels()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # --- Channels Section ---
        channels_group = QGroupBox("Channels")
        channels_layout = QVBoxLayout(channels_group)

        self.channels_container = QWidget()
        self.channels_layout = QVBoxLayout(self.channels_container)
        self.channels_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidget(self.channels_container)
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(150)
        channels_layout.addWidget(scroll)

        refresh_btn = QPushButton("Refresh Channels")
        refresh_btn.clicked.connect(self._refresh_channels)
        channels_layout.addWidget(refresh_btn)

        layout.addWidget(channels_group)

        # --- Figure Settings ---
        settings_group = QGroupBox("Figure Settings")
        settings_layout = QVBoxLayout(settings_group)

        # DPI
        dpi_layout = QHBoxLayout()
        dpi_layout.addWidget(QLabel("DPI:"))
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(300)
        dpi_layout.addWidget(self.dpi_spin)
        dpi_layout.addStretch()
        settings_layout.addLayout(dpi_layout)

        # Show merge
        self.show_merge_cb = QCheckBox("Include merge panel")
        self.show_merge_cb.setChecked(True)
        settings_layout.addWidget(self.show_merge_cb)

        # Scale bar
        self.show_scale_bar_cb = QCheckBox("Show scale bar")
        self.show_scale_bar_cb.setChecked(True)
        settings_layout.addWidget(self.show_scale_bar_cb)

        layout.addWidget(settings_group)

        # --- Presets ---
        presets_group = QGroupBox("Presets")
        presets_layout = QHBoxLayout(presets_group)

        self.preset_combo = QComboBox()
        self._refresh_presets()
        presets_layout.addWidget(self.preset_combo)

        load_preset_btn = QPushButton("Load")
        load_preset_btn.clicked.connect(self._load_preset)
        presets_layout.addWidget(load_preset_btn)

        save_preset_btn = QPushButton("Save")
        save_preset_btn.clicked.connect(self._save_preset)
        presets_layout.addWidget(save_preset_btn)

        layout.addWidget(presets_group)

        # --- Export Buttons ---
        export_layout = QHBoxLayout()

        self.export_btn = QPushButton("Export Figure...")
        self.export_btn.clicked.connect(self._export_figure)
        export_layout.addWidget(self.export_btn)

        self.clipboard_btn = QPushButton("Copy to Clipboard")
        self.clipboard_btn.clicked.connect(self._copy_to_clipboard)
        export_layout.addWidget(self.clipboard_btn)

        layout.addLayout(export_layout)

        # Quick export
        self.quick_export_btn = QPushButton("Quick Export (Last Settings)")
        self.quick_export_btn.clicked.connect(self._quick_export)
        layout.addWidget(self.quick_export_btn)

        layout.addStretch()

    def _connect_signals(self):
        """Connect to napari viewer signals."""
        self.viewer.layers.events.inserted.connect(self._refresh_channels)
        self.viewer.layers.events.removed.connect(self._refresh_channels)

    def _refresh_channels(self, event=None):
        """Refresh the channel list from viewer layers."""
        # Clear existing
        for w in self.channel_widgets:
            w.deleteLater()
        self.channel_widgets.clear()

        # Add widget for each image layer
        for layer in self.viewer.layers:
            if layer._type_string == "image":
                widget = ChannelConfigWidget(layer.name)
                self.channels_layout.addWidget(widget)
                self.channel_widgets.append(widget)

    def _refresh_presets(self):
        """Refresh the presets dropdown."""
        self.preset_combo.clear()
        self.preset_combo.addItem("(No preset)")

        for preset in list_presets():
            self.preset_combo.addItem(preset["name"], preset["path"])

    def _get_figure_config(self) -> FigureConfig:
        """Get current figure configuration from UI."""
        return FigureConfig(
            dpi=self.dpi_spin.value(),
            show_merge=self.show_merge_cb.isChecked(),
        )

    def _get_channel_configs(self) -> List[ChannelConfig]:
        """Get channel configurations from UI."""
        return [w.get_config() for w in self.channel_widgets]

    def _build_figure(self) -> Optional[np.ndarray]:
        """Build figure from current viewer state."""
        channel_configs = self._get_channel_configs()
        figure_config = self._get_figure_config()

        # Get data and contrast limits from napari layers
        channels_data = []
        contrast_limits = []
        pixel_size_um = None

        for layer in self.viewer.layers:
            if layer._type_string == "image":
                # Get 2D slice (handle nD data)
                data = layer.data
                if data.ndim > 2:
                    # Get current slice
                    slices = [slice(None)] * data.ndim
                    for i, idx in enumerate(self.viewer.dims.current_step[:-2]):
                        slices[i] = idx
                    data = data[tuple(slices)]

                channels_data.append(data)
                contrast_limits.append(layer.contrast_limits)

                # Try to get pixel size from metadata
                if pixel_size_um is None and hasattr(layer, "metadata"):
                    pixel_size_um = layer.metadata.get("pixel_size_um")

        if not channels_data:
            return None

        return build_figure(
            channels_data=channels_data,
            channel_configs=channel_configs,
            figure_config=figure_config,
            pixel_size_um=pixel_size_um if self.show_scale_bar_cb.isChecked() else None,
            contrast_limits=contrast_limits,
        )

    def _export_figure(self):
        """Export figure with file dialog."""
        figure = self._build_figure()
        if figure is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Figure",
            str(Path.home() / "figure.png"),
            "PNG (*.png);;TIFF (*.tiff *.tif);;All Files (*)",
        )

        if path:
            export_figure(figure, Path(path), dpi=self.dpi_spin.value())
            self._last_export_path = Path(path)

    def _quick_export(self):
        """Export using last settings."""
        if not hasattr(self, "_last_export_path"):
            self._export_figure()
            return

        figure = self._build_figure()
        if figure is None:
            return

        export_figure(figure, self._last_export_path, dpi=self.dpi_spin.value())

    def _copy_to_clipboard(self):
        """Copy figure to clipboard."""
        figure = self._build_figure()
        if figure is None:
            return

        copy_to_clipboard(figure)

    def _load_preset(self):
        """Load selected preset."""
        path = self.preset_combo.currentData()
        if path is None:
            return

        name, channel_configs, figure_config = load_preset(path)

        # Apply figure config
        self.dpi_spin.setValue(figure_config.dpi)
        self.show_merge_cb.setChecked(figure_config.show_merge)

        # Apply channel configs (match by name or position)
        for i, widget in enumerate(self.channel_widgets):
            # Try to find matching config by name
            matching = [c for c in channel_configs if c.name == widget.channel_name]
            if matching:
                widget.set_config(matching[0])
            elif i < len(channel_configs):
                # Fall back to position
                widget.set_config(channel_configs[i])

    def _save_preset(self):
        """Save current settings as preset."""
        from qtpy.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if not ok or not name:
            return

        # Sanitize filename
        filename = "".join(c for c in name if c.isalnum() or c in " -_").strip()
        filename = filename.replace(" ", "_").lower()

        path = get_preset_directory() / f"{filename}.yaml"

        save_preset(
            path=path,
            name=name,
            channel_configs=self._get_channel_configs(),
            figure_config=self._get_figure_config(),
        )

        self._refresh_presets()
```

**Step 3: Update conftest.py for widget testing**

Add to `tests/conftest.py`:
```python
@pytest.fixture
def make_napari_viewer():
    """Fixture to create napari viewers for testing."""
    pytest.importorskip("napari")
    from napari import Viewer

    viewers = []

    def _make_viewer(**kwargs):
        viewer = Viewer(**kwargs, show=False)
        viewers.append(viewer)
        return viewer

    yield _make_viewer

    for v in viewers:
        v.close()
```

**Step 4: Commit**

```bash
git add -A
git commit -m "feat: add FigureMakerWidget with full UI"
```

---

## Task 12: Integration Testing & Polish

**Files:**
- Create: `tests/test_integration.py`
- Create: `README.md`

**Step 1: Write integration test**

`tests/test_integration.py`:
```python
"""Integration tests for napari-figure-maker."""

import numpy as np
import pytest
from pathlib import Path

from napari_figure_maker.models import ChannelConfig, FigureConfig
from napari_figure_maker.figure_builder import build_figure
from napari_figure_maker.exporter import export_figure


def test_full_pipeline(tmp_path):
    """Test the full pipeline from data to exported figure."""
    # Create sample 3-channel data
    channels_data = [
        np.random.randint(0, 255, (256, 256), dtype=np.uint8),
        np.random.randint(0, 255, (256, 256), dtype=np.uint8),
        np.random.randint(0, 255, (256, 256), dtype=np.uint8),
    ]

    channel_configs = [
        ChannelConfig(name="DAPI", colormap="blue", label="Nuclei"),
        ChannelConfig(name="GFP", colormap="green", label="GFP"),
        ChannelConfig(name="RFP", colormap="magenta", label="RFP"),
    ]

    figure_config = FigureConfig(
        dpi=300,
        show_merge=True,
        panel_gap_fraction=0.02,
    )

    # Build figure
    figure = build_figure(
        channels_data=channels_data,
        channel_configs=channel_configs,
        figure_config=figure_config,
        pixel_size_um=0.5,
    )

    # Verify figure dimensions (3 channels + merge = 4 panels)
    assert figure.shape[0] == 256  # Height
    expected_width = 256 * 4 + int(256 * 0.02) * 3  # 4 panels + 3 gaps
    assert abs(figure.shape[1] - expected_width) < 5  # Allow small rounding

    # Export
    output_path = tmp_path / "test_output.png"
    export_figure(figure, output_path, dpi=300)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_minimal_figure(tmp_path):
    """Test with minimal configuration."""
    channels_data = [
        np.full((100, 100), 128, dtype=np.uint8),
    ]

    channel_configs = [
        ChannelConfig(name="Ch1"),
    ]

    figure_config = FigureConfig(show_merge=False)

    figure = build_figure(
        channels_data=channels_data,
        channel_configs=channel_configs,
        figure_config=figure_config,
    )

    # Single panel, no merge
    assert figure.shape == (100, 100, 3)
```

**Step 2: Run integration tests**

Run: `cd /Users/sean/code/napari-figure-maker && uv run pytest tests/test_integration.py -v`
Expected: All PASS

**Step 3: Create README**

`README.md`:
```markdown
# napari-figure-maker

A napari plugin for creating publication-ready multichannel figure panels from microscopy images.

## Features

- **Use napari as the source of truth** - Adjust contrast, colors, and Z-slices in napari, then export exactly what you see
- **Automatic panel layout** - Individual channels arranged in a grid with merged composite
- **Scale bars** - Auto-calculated from image metadata with customizable appearance
- **Channel labels** - Automatic or custom text labels on each panel
- **Presets** - Save and reuse your favorite configurations
- **Quick export** - One-click export for rapid QC
- **LIF support** - Read Leica LIF files directly

## Installation

```bash
pip install napari-figure-maker
```

Or for development:

```bash
git clone https://github.com/yourusername/napari-figure-maker
cd napari-figure-maker
pip install -e ".[dev]"
```

## Usage

1. Open your image in napari (supports LIF files directly, or any napari-supported format)
2. Adjust contrast and colormaps for each channel using napari's layer controls
3. Open the Figure Maker widget from `Plugins > Figure Maker`
4. Configure channels (visibility, labels, colors)
5. Click "Export Figure" or "Copy to Clipboard"

## Presets

Save your channel configurations as presets to reuse across experiments:

1. Set up your channels how you like
2. Click "Save" in the Presets section
3. Give it a name
4. Load it later from the dropdown

Presets are saved as YAML files in:
- macOS: `~/Library/Application Support/napari-figure-maker/presets/`
- Linux: `~/.config/napari-figure-maker/presets/`
- Windows: `%APPDATA%/napari-figure-maker/presets/`

## License

MIT
```

**Step 4: Run all tests**

Run: `cd /Users/sean/code/napari-figure-maker && uv run pytest -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add -A
git commit -m "feat: add integration tests and README"
```

---

## Summary

This plan creates a fully functional napari plugin with:

1. **Data models** (Task 2) - `ChannelConfig` and `FigureConfig` dataclasses
2. **Scale bar rendering** (Task 3) - Calculate nice lengths, render with Pillow
3. **Figure building** (Tasks 4-7) - Channel rendering, merge composite, grid layout, labels
4. **Presets** (Task 8) - YAML save/load with directory management
5. **Export** (Task 9) - PNG/TIFF export with DPI settings
6. **LIF reader** (Task 10) - napari reader plugin for Leica files
7. **Widget UI** (Task 11) - Full Qt widget with all controls
8. **Integration tests** (Task 12) - End-to-end testing and documentation

Each task follows TDD: write failing test → implement → verify passing → commit.
