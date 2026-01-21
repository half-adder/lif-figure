# Metadata Table Feature Design

## Overview

Add a table below figure panels showing per-channel acquisition settings extracted from the LIF file metadata.

## Table Format

| Channel | Laser | Power | Detector | Mode | Gain | Contrast |
|---------|-------|-------|----------|------|------|----------|
| DAPI    | 405nm | 100%  | HyD1     | Std  | 100% | min-max  |
| GFP     | 488nm | 44%   | HyD2     | PC   | 10%  | 5-95%    |
| ALFA    | 552nm | 65%   | HyD3     | PC   | 10%  | 5-95%    |

- **Contrast column:** Shows "min-max" for standard normalization, or percentiles (e.g., "5-95%") when `--auto-contrast` is used

## Data Extraction

For each series, extract from the `HardwareSetting` XML attachment:

### Lasers
- Find active AOTF sections (`IsChanged="1"`)
- Get `LaserLineSetting` entries with non-zero `IntensityDev`
- Store wavelength → power mapping

### Detectors
- Find `Detector` elements with `IsActive="1"`
- Extract:
  - Detector name/index
  - `AcquisitionModeName` (empty = Standard, "PhotonCounting" = PC)
  - `Gain` value

### Channel Mapping
- Match channels to detectors by index order (channel 0 → first active detector, etc.)

## CLI Interface

- Metadata table shown by default
- `--no-metadata` flag to hide it

## Implementation

### Files to Modify

1. **`reader.py`** — Add `extract_series_metadata()` function:
   ```python
   @dataclass
   class DetectorInfo:
       name: str
       mode: str  # "Std" or "PC"
       gain: float

   @dataclass
   class SeriesMetadata:
       lasers: dict[str, float]  # wavelength -> power %
       detectors: list[DetectorInfo]  # per-channel detector info
   ```

2. **`figure.py`** — Update `build_figure()` to:
   - Accept optional `SeriesMetadata` and contrast percentiles
   - Add a table subplot below the image panels
   - Render the per-channel metadata table

3. **`cli.py`** — Add `--no-metadata` flag, pass metadata through the pipeline
