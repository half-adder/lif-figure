# Metadata Extraction Logic

This document describes how lif-figure extracts acquisition metadata from Leica LIF files.

## LIF File Structure

LIF files contain an XML metadata section that describes the acquisition settings. The relevant structure for metadata extraction is:

```
Element (Name="SeriesXXX")
├── MultiBand (Channel, LeftWorld, RightWorld, ...)
└── Attachment (Name="HardwareSetting")
    ├── Aotf (IsChanged="1")
    │   └── LaserLineSetting (LaserLine, IntensityDev)
    └── Detector (Name, Channel, Gain, AcquisitionModeName, ...)
```

## Laser Extraction

Lasers are extracted from `Aotf` elements within the `HardwareSetting` attachment.

### Logic

1. Find all `Aotf` elements with `IsChanged="1"` (indicates active during acquisition)
2. For each `LaserLineSetting` child:
   - Extract `LaserLine` (wavelength in nm)
   - Extract `IntensityDev` (power as percentage)
   - Only include lasers with intensity > 0.001%

### Matching to Channels

Lasers are matched to channels by sorting wavelengths numerically. This assumes:
- Shorter wavelengths (e.g., 405nm for DAPI) correspond to earlier channels
- Longer wavelengths (e.g., 594nm for mCherry) correspond to later channels

This is a heuristic that works for typical multi-channel confocal setups.

## Detector Extraction

Detectors are extracted using a two-step process:

### Step 1: Identify Active Channels via MultiBand

`MultiBand` elements define spectral detection windows for each channel. The key insight is that inactive channels have narrow placeholder ranges (~5nm), while active channels have meaningful detection ranges (>10nm).

```
MultiBand elements example:
  Ch1: 380-385nm (width=5nm)   → INACTIVE (placeholder)
  Ch2: 410-483nm (width=73nm)  → ACTIVE
  Ch3: 486-491nm (width=5nm)   → INACTIVE (placeholder)
  Ch4: 493-547nm (width=54nm)  → ACTIVE
  Ch5: 580-757nm (width=177nm) → ACTIVE
```

This gives us the exact detector channel numbers used for imaging (e.g., [2, 4, 5]).

### Step 2: Match Channels to Detector Metadata

For each active channel number from MultiBand:

1. Find the `Detector` element with matching `Channel` attribute
2. Only consider detectors with `Gain` attribute (complete acquisition settings)
3. Extract metadata:
   - **Name**: e.g., "HyD 2", "PMT 1"
   - **Mode**:
     - HyD detectors: "PhotonCounting" → "PC", "PhotonIntegration" → "Std"
     - PMT detectors: "-" (no mode concept)
   - **Gain**: as percentage

### Why MultiBand?

Previous approaches (filtering by detector type or Gain > 0) were unreliable:
- Users can mix HyD and PMT detectors in the same acquisition
- Detectors may have leftover gain settings from previous experiments
- More detectors may have non-zero gain than there are image channels

MultiBand detection ranges reliably indicate which channels actually captured image data.

### Fallback Behavior

If no MultiBand elements are found, the code falls back to:
1. Find all detectors with Gain > 0
2. Exclude channel 100 (PMT Trans - transmitted light)
3. Sort by channel number
4. Use first N detectors matching image channel count

### Channel Mapping

Image channels (0, 1, 2, ...) map to detector channels by sort order:

| Image Channel | Detector Channel (sorted) |
|--------------|---------------------------|
| 0            | Lowest active channel     |
| 1            | Next active channel       |
| 2            | Next active channel       |

For example, if active channels are [2, 4, 5]:
- Image channel 0 → Detector channel 2
- Image channel 1 → Detector channel 4
- Image channel 2 → Detector channel 5

## Metadata Table Output

The extracted metadata is displayed in a table below the figure:

| Column    | Source                          |
|-----------|--------------------------------|
| Channel   | User-provided channel names    |
| Laser     | Sorted laser wavelengths       |
| Power     | Laser IntensityDev percentage  |
| Detector  | Detector name (HyD or PMT)     |
| Mode      | PC, Std, or - (for PMT)        |
| Gain      | Detector gain percentage       |
| Contrast  | Auto-contrast percentiles used |

## Limitations

- Laser-to-channel matching is heuristic (by wavelength order)
- Does not extract spectral detection ranges for display
- Does not handle sequential scanning with different settings per sequence
