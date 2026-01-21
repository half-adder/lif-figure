# LIF Figure CLI Design

## Overview

A CLI tool for generating publication-ready figure panels from Leica LIF microscopy files. Each series produces a PDF with grayscale individual channels and a color-merged panel.

## CLI Interface

```bash
lif-figure INPUT_FILE [OPTIONS]
```

### Arguments

- `INPUT_FILE` - Path to .lif file (required)

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--series "Name1,Name2"` | Series to process | All series |
| `--channels "DAPI,GFP,mCherry"` | Channel names, comma-separated | **Required** |
| `--output DIR` | Output directory | `./figures` |
| `--zstack MODE` | Z-stack handling mode | `max` |
| `--config FILE` | Optional YAML config | None |

### Z-Stack Modes

- `max` - Max projection of all Z frames
- `max:5-15` - Max projection of frames 5-15 (0-indexed)
- `frames` - Each Z frame as separate PDF in subdirectory
- `rows` - All Z frames as rows in single PDF

### Example Usage

```bash
# Basic usage - all series, default settings
lif-figure experiment.lif --channels "DAPI,GFP,mCherry"

# Specific series with custom output
lif-figure experiment.lif --channels "DAPI,GFP,mCherry" --series "Control,Treatment" --output ./results

# Z-stack with specific range
lif-figure experiment.lif --channels "DAPI,GFP,mCherry" --zstack max:10-20
```

## Output Structure

### Directory Layout

```
./figures/
├── Series_001.pdf
├── Series_002.pdf
├── My Sample Name.pdf
└── zstack_series/           # only when --zstack frames
    ├── Series_003_z00.pdf
    ├── Series_003_z01.pdf
    └── ...
```

### Figure Layout

**Standard 4-panel row:**
```
┌─────────┬─────────┬─────────┬─────────┐
│  DAPI   │   GFP   │ mCherry │  Merge  │
│ (gray)  │ (gray)  │ (gray)  │ (color) │
│         │         │         │         │
│    ────┤│    ────┤│    ────┤│    ────┤
│   10 µm │   10 µm │   10 µm │   10 µm │
└─────────┴─────────┴─────────┴─────────┘
```

- Labels: top-left of each panel (channel name, "Merge" for merged)
- Scale bars: bottom-right of each panel (automatic from LIF metadata)
- Individual channels: grayscale
- Merge panel: color (blue/green/red by default)

**Z-stack rows mode:**
```
┌─────────┬─────────┬─────────┬─────────┐
│ DAPI    │ GFP     │ mCherry │ Merge   │  Z=0
│ Z=0     │ Z=0     │ Z=0     │ Z=0     │
└─────────┴─────────┴─────────┴─────────┘
┌─────────┬─────────┬─────────┬─────────┐
│ DAPI    │ GFP     │ mCherry │ Merge   │  Z=1
│ Z=1     │ Z=1     │ Z=1     │ Z=1     │
└─────────┴─────────┴─────────┴─────────┘
```

## Config File (Optional)

Auto-detected as `lif-figure.yaml` in current directory, or specified with `--config`.

```yaml
# Channel color overrides (default: blue, green, red positional)
colors:
  DAPI: blue
  GFP: green
  mCherry: magenta

# Figure styling
dpi: 300
font_size: 12
scale_bar_height: 4  # pixels
background: black    # or white
```

**Precedence:** CLI flags > config file > defaults

## Module Structure

```
lif_figure/
├── __init__.py
├── cli.py          # argument parsing, main entry point
├── reader.py       # LIF file reading, Z-stack handling
├── figure.py       # matplotlib figure generation
└── config.py       # config file loading, defaults
```

## Data Flow

1. `cli.py` parses arguments, loads optional config file
2. `reader.py` opens LIF file, extracts requested series/channels as numpy arrays, applies Z-stack processing
3. `figure.py` receives arrays + metadata, builds 4-panel matplotlib figure with labels and scale bars
4. `cli.py` saves each figure as PDF to output directory

## Key Functions

```python
# reader.py
def read_lif_series(
    path: str,
    series_names: list[str] | None,
    zstack_mode: str
) -> dict[str, tuple[np.ndarray, float | None]]:
    """Returns {series_name: (channels_array, pixel_size_um)}"""

# figure.py
def build_figure(
    channels: np.ndarray,
    names: list[str],
    colors: list[str],
    pixel_size_um: float | None
) -> matplotlib.figure.Figure:
    """Creates 4-panel figure with labels and scale bars"""

# cli.py
def save_figure(fig: Figure, output_path: Path) -> None:
    """Saves figure as PDF"""
```

## Error Handling

### Errors (exit immediately)

- File not found or invalid LIF format
- `--channels` flag not provided
- Series name not found in file
- Channel count mismatch (e.g., file has 2 channels but 3 names given)

### Warnings (continue processing)

- Series has no Z dimension but `--zstack` specified → process normally
- Missing pixel size metadata → skip scale bar with warning
- Invalid filename characters in series name → sanitize automatically

## Progress Output

```
Processing experiment.lif
  Series 1 of 5: "Control Sample" → figures/Control Sample.pdf
  Series 2 of 5: "Treatment A" → figures/Treatment A.pdf
  ...
Done. 5 figures saved to ./figures/
```

## Dependencies

- `readlif` - LIF file parsing
- `matplotlib` - Figure generation and PDF output
- `numpy` - Array operations
- `click` - CLI argument parsing
- `pyyaml` - Config file parsing
