# lif-figure

Generate publication-ready figure panels from Leica LIF microscopy files.

## Features

- **Multichannel Figure Panels**: Grayscale channel panels with color merge
- **Automatic Scale Bars**: Calculated from LIF metadata with nice round numbers
- **Channel Labels**: Customizable labels for each panel
- **Z-Stack Support**: Max projection, individual frames, or range selection
- **Auto-Contrast**: Optional percentile-based intensity normalization
- **Metadata Table**: Acquisition parameters displayed below figures
- **PDF Output**: Vector graphics suitable for publication

## Installation

[uv](https://docs.astral.sh/uv/) is the recommended way to install and run lif-figure.

### Run without installing

```bash
uvx lif-figure input.lif --channels "DAPI,GFP,mCherry"
```

### Install as a tool

```bash
uv tool install lif-figure
lif-figure input.lif --channels "DAPI,GFP,mCherry"
```

### For development

```bash
git clone https://github.com/half-adder/lif-figure.git
cd lif-figure
uv sync
uv run lif-figure input.lif --channels "DAPI,GFP,mCherry"
```

## Usage

```bash
lif-figure input.lif --channels "DAPI,GFP,mCherry"
```

This processes all series in the LIF file and outputs PDF figures to `./figures/`.

**Important:** All series in the LIF file are expected to contain the same channels in the same order. The `--channels` argument defines the labels applied to each channel by index (first name = channel 0, second = channel 1, etc.). Series with a different number of channels will be skipped.

### Options

```bash
lif-figure input.lif \
    --channels "DAPI,GFP,mCherry" \
    --series "Image1,Image2" \
    --output ./output \
    --zstack max \
    --auto-contrast \
    --config lif-figure.yaml
```

| Option | Short | Description |
|--------|-------|-------------|
| `--channels` | `-c` | Channel names, comma-separated (required) |
| `--series` | `-s` | Series to process, comma-separated (default: all) |
| `--output` | `-o` | Output directory (default: `./figures`) |
| `--zstack` | `-z` | Z-stack mode: `max`, `max:START-END`, `frames` (default: `max`) |
| `--config` | | YAML config file path |
| `--auto-contrast` | `-a` | Enable auto-contrast (optional: `LOW,HIGH` percentiles) |
| `--no-metadata` | | Hide acquisition metadata table |

### Z-Stack Modes

- `max` - Maximum intensity projection across all Z slices
- `max:5-15` - Max projection of Z slices 5 through 15
- `frames` - Output each Z slice as a separate PDF

### Configuration File

Create `lif-figure.yaml` in your working directory (auto-detected) or specify with `--config`:

```yaml
# Channel color overrides
colors:
  DAPI: blue
  GFP: green
  mCherry: red

# Figure settings
dpi: 300
font_size: 12
scale_bar_height: 4
background: black

# Auto-contrast percentiles (optional)
auto_contrast_percentiles: [0.1, 99.9]
```

## Output

Each series produces a PDF with:
- Individual grayscale panels for each channel (labeled)
- Color merge panel
- Scale bar with measurement
- Acquisition metadata table (optional)

## License

MIT
