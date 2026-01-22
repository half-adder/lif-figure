# Metadata Extraction Logic

This document describes how lif-figure extracts acquisition metadata from Leica LIF files.

## LIF File Structure

LIF files contain an XML metadata section that describes the acquisition settings. The relevant structure for metadata extraction is:

```
Element (Name="SeriesXXX")
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

Detectors are extracted from `Detector` elements within the `HardwareSetting` attachment.

### Logic

1. Find all `Detector` elements that:
   - Have a `Name` starting with "HyD" (hybrid detectors used in confocal)
   - Have a `Gain` attribute (indicates complete acquisition settings)

2. For each detector:
   - Extract `Channel` number for ordering
   - Extract `Name` (e.g., "HyD 2", "HyD 4")
   - Extract `AcquisitionModeName`:
     - "PhotonCounting" → "PC"
     - "PhotonIntegration" → "Std"
     - Other/missing → "Std"
   - Extract `Gain` as percentage

3. Deduplicate by channel number (XML may contain duplicate entries)

4. Sort by channel number to match image channel order

### Why Filter by HyD and Gain?

The LIF XML contains multiple `Detector` entries:
- Some are PMT detectors (not used for fluorescence channels)
- Some entries lack acquisition settings (just detector definitions)
- Some are duplicates from different XML contexts

Filtering to HyD detectors with `Gain` attributes ensures we get:
- Only the detectors actually used for image acquisition
- Complete acquisition settings (mode, gain)

### Channel Mapping

Image channels (0, 1, 2, ...) map to detector channels by sort order:

| Image Channel | Detector Channel (sorted) |
|--------------|---------------------------|
| 0            | Lowest channel number     |
| 1            | Next channel number       |
| 2            | Next channel number       |

For example, if HyD detectors use channels 2, 4, 5:
- Image channel 0 → HyD channel 2
- Image channel 1 → HyD channel 4
- Image channel 2 → HyD channel 5

## Metadata Table Output

The extracted metadata is displayed in a table below the figure:

| Column    | Source                          |
|-----------|--------------------------------|
| Channel   | User-provided channel names    |
| Laser     | Sorted laser wavelengths       |
| Power     | Laser IntensityDev percentage  |
| Detector  | HyD detector name              |
| Mode      | PC or Std                      |
| Gain      | Detector gain percentage       |
| Contrast  | Auto-contrast percentiles used |

## Limitations

- Assumes HyD detectors are used (common for modern Leica confocal)
- Laser-to-channel matching is heuristic (by wavelength order)
- Does not extract spectral detection ranges
- Does not handle sequential scanning with different settings per sequence
