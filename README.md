# napari-figure-maker

Create publication-ready multichannel figure panels from microscopy images in napari.

## Features

- **Multichannel Figure Panels**: Arrange fluorescence channels side-by-side with merge
- **Automatic Scale Bars**: Calculate and render scale bars with nice round numbers
- **Channel Labels**: Add customizable labels to each panel
- **Colormap Support**: Built-in colormaps (gray, red, green, blue, cyan, magenta, yellow)
- **Preset System**: Save and load figure configurations
- **LIF File Support**: Read Leica LIF files directly into napari
- **Export Options**: Save figures as PNG or TIFF with proper DPI

## Installation

```bash
pip install napari-figure-maker
```

Or for development:

```bash
git clone https://github.com/yourusername/napari-figure-maker.git
cd napari-figure-maker
uv sync --all-extras
```

## Usage

### In napari

1. Open napari and load your image layers
2. Open the Figure Maker widget: `Plugins > Figure Maker`
3. Select channels from the list
4. Configure options (merge, DPI, background color)
5. Click "Export Figure..." to save

### Programmatic Usage

```python
import numpy as np
from napari_figure_maker.models import ChannelConfig, FigureConfig
from napari_figure_maker.figure_builder import build_figure
from napari_figure_maker.exporter import export_figure

# Your channel data
dapi = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
gfp = np.random.randint(0, 255, (512, 512), dtype=np.uint8)

# Configure channels
channel_configs = [
    ChannelConfig(name="DAPI", colormap="blue", label="Nuclei"),
    ChannelConfig(name="GFP", colormap="green", label="GFP"),
]

# Configure figure
figure_config = FigureConfig(
    dpi=300,
    show_merge=True,
    panel_gap_fraction=0.02,
    background_color="black",
)

# Build and export
figure = build_figure(
    channels_data=[dapi, gfp],
    channel_configs=channel_configs,
    figure_config=figure_config,
    pixel_size_um=0.5,  # For scale bar
)

export_figure(figure, "my_figure.png", dpi=300)
```

### Using Presets

```python
from napari_figure_maker.presets import save_preset, load_preset

# Save your configuration
save_preset(
    path="my_preset.yaml",
    name="Standard 2-Channel",
    channel_configs=channel_configs,
    figure_config=figure_config,
)

# Load it later
name, channels, figure = load_preset("my_preset.yaml")
```

## Configuration Options

### ChannelConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| name | str | required | Channel identifier |
| visible | bool | True | Include in figure |
| label | str | None | Display label (uses name if None) |
| colormap | str | "gray" | Color lookup table |

### FigureConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| dpi | int | 300 | Export resolution |
| panel_gap_fraction | float | 0.02 | Gap as fraction of panel width |
| background_color | str | "black" | Gap/background color |
| show_merge | bool | True | Include merge panel |
| scale_bar_color | str | "white" | Scale bar color |
| label_font_size | int | 12 | Channel label font size |
| label_position | str | "top-left" | Label placement |

## License

MIT
