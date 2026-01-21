# napari-figure-maker Design

A napari plugin for creating multichannel figure panels from microscopy images.

## Problem

Creating publication-ready figures from confocal microscopy data (LIF files, etc.) requires:
- Browsing complex file structures (multiple series, Z-stacks)
- Adjusting contrast and pseudocolors per channel
- Arranging channels as side-by-side panels with a merge
- Adding scale bars and labels
- Exporting at publication-quality DPI

Current workflow involves multiple tools (Fiji, Illustrator) and manual repetition.

## Solution

A napari plugin that uses the viewer as the source of truth for image display, then exports the current view as a formatted figure panel.

## Core Workflow

1. **Open image** in napari (LIF via `readlif`, or any napari-supported format)
2. **Browse and adjust** using napari's native UI:
   - Select series from multi-series files via dock widget
   - Navigate Z-slices with slider
   - Adjust contrast limits per channel
   - Assign colormaps/LUTs
3. **Generate figure** via the Figure Maker dock widget:
   - Captures current channel layers and display settings
   - Renders each channel as a panel
   - Creates merged composite
   - Arranges in grid layout
   - Adds scale bar and labels
4. **Export** to PNG/TIFF at specified DPI

Key principle: No separate adjustment UI. Whatever you see in napari is what appears in the figure.

## Figure Layout & Customization

### Panel Arrangement

Default layout: individual channels in order, merge on the right.

```
[Ch1] [Ch2] [Ch3] [Merge]
```

Configurable options:
- **Grid shape** - Force specific arrangement (e.g., 2x2)
- **Channel order** - Drag to reorder, or hide channels
- **Merge position** - Right side, bottom-right, or omit

### Scale Bar

- Auto-calculated from image metadata (pixel size)
- Configurable: length (auto or specified), color (white/black/auto), position, font size
- Display on merge only, or all panels

### Channel Labels

- Default: pull from image metadata channel names
- Override with custom text
- Position: top-left of panel, or header row above
- Optional: color matches channel LUT

### Spacing & Style

- Gap between panels (default ~2% of panel width)
- Optional border around panels
- Background color (black default, white option)

## Presets System

### Saving Presets

Save current figure settings as named presets:
- "3-channel IF" - DAPI/blue, GFP/green, RFP/magenta
- "Live imaging" - GFP/green, mCherry/red, minimal style

Presets store:
- LUT assignments per channel position/name
- Label text
- Scale bar settings
- Layout options
- DPI/size settings

Presets do NOT store contrast limits (those come from current napari view).

### Applying Presets

Channel matching when loading preset:
1. Match by channel name if available (e.g., "DAPI" → "DAPI")
2. Fall back to position (channel 0 → first preset slot)
3. Manual reassignment available

### Storage

- Location: `~/.config/napari-figure-maker/presets/`
- Format: YAML files (portable, shareable)
- Last-used settings remembered between sessions

### Quick Export

One-click export using last settings or default preset for rapid QC.

## Export Options

### File Formats

- **PNG** - Default, good compression, universal
- **TIFF** - Lossless, higher bit depth for archival
- **SVG** - Vector format, labels/scale bar remain editable

### Resolution & Sizing

- **DPI**: 300 (publication), 150 (draft), or custom
- **Figure width**: Specify in inches/cm, height auto-calculated
- **Panel size**: Alternative—specify individual panel dimensions

### Naming & Batch

- Default: `{source_name}_{series}_figure.png`
- Configurable output directory
- Batch export for multiple series

### Clipboard

Copy to clipboard at screen resolution for quick paste into slides/docs.

### Reproducibility

Optional sidecar file (YAML) with exact settings and contrast limits used.

## Technical Design

### Dependencies

- `napari` - Core viewer
- `readlif` - LIF file reading (pure Python, no Java)
- `pillow` - Figure composition, text rendering
- `numpy` - Array manipulation (already a napari dependency)

No matplotlib required. Optional `aicsimageio[bioformats]` for broader format support.

### Plugin Structure

```
napari-figure-maker/
├── src/napari_figure_maker/
│   ├── __init__.py
│   ├── _widget.py          # Main dock widget UI
│   ├── figure_builder.py   # Panel layout & composition
│   ├── presets.py          # Preset load/save/apply
│   ├── scale_bar.py        # Scale bar rendering
│   ├── exporter.py         # File export handling
│   └── utils.py            # Helpers
├── tests/
├── pyproject.toml
├── napari.yaml             # Plugin manifest
└── README.md
```

### Rendering Pipeline

1. **Get channel data from napari**: Use `layer.data` with current contrast limits and colormap applied
2. **Composite grid**: Pure numpy to arrange channel arrays with spacing
3. **Add annotations**: Pillow `ImageDraw` for scale bar and text labels
4. **Export**: Pillow for PNG/TIFF, optional SVG generation

### napari Integration

- Registers dock widget via `napari.yaml` manifest
- Accesses viewer layers, contrast limits, colormaps through napari API
- Listens for layer changes to update channel list
- Uses napari's native layer system (no custom viewers)

## Future Considerations

Not in initial scope, but possible extensions:
- Support for additional formats beyond LIF (CZI, ND2, etc.)
- ROI cropping before export
- Multi-figure layouts (multiple images in one figure)
- Batch processing CLI mode

## Success Criteria

- Open LIF file, adjust contrast, export figure in under 1 minute
- Presets work across different experiments with same channel setup
- Output meets publication requirements (300 DPI, scale bar, labels)
- No Java/bioformats dependency for basic LIF support
