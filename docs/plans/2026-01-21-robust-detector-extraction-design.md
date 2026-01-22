# Robust Detector Extraction Design

## Problem

The current detector extraction logic assumes HyD detectors and filters by `Gain > 0`. This fails when:

1. PMT detectors are used instead of (or mixed with) HyD
2. Detectors have leftover gain settings from previous experiments
3. More detectors have non-zero gain than there are image channels

## Solution

Use **MultiBand detection ranges** to identify which detector channels are actually active, then match those to Detector elements for metadata.

## Algorithm

### Step 1: Identify Active Channels via MultiBand

MultiBand elements define spectral detection windows for each channel. Inactive channels have narrow placeholder ranges (~5nm), while active channels have meaningful detection ranges (>10nm).

```python
def get_active_channels(element) -> list[int]:
    active = []
    seen = set()

    for mb in element.iter('MultiBand'):
        ch = mb.get('Channel')
        if ch not in seen:
            seen.add(ch)
            left = float(mb.get('LeftWorld', 0))
            right = float(mb.get('RightWorld', 0))
            width = right - left
            if width > 10:  # Active detection window
                active.append(int(ch))

    return sorted(active)
```

### Step 2: Match Channels to Detector Metadata

For each active channel number, find the corresponding Detector element:

```python
def get_detector_info(attachment, active_channels) -> list[DetectorInfo]:
    # Build map of channel -> detector info
    detector_map = {}

    for det in attachment.iter('Detector'):
        ch = det.get('Channel')
        if ch and det.get('Gain'):  # Has complete acquisition info
            channel = int(ch)
            if channel in active_channels:
                name = det.get('Name', '')
                mode = get_mode(det)
                gain = float(det.get('Gain', 0))
                detector_map[channel] = DetectorInfo(name, mode, gain)

    # Return in channel order
    return [detector_map.get(ch) for ch in active_channels]

def get_mode(detector) -> str:
    name = detector.get('Name', '')
    if name.startswith('PMT'):
        return '-'  # PMT has no mode concept

    mode_name = detector.get('AcquisitionModeName', '')
    if mode_name == 'PhotonCounting':
        return 'PC'
    elif mode_name == 'PhotonIntegration':
        return 'Std'
    return 'Std'
```

### Step 3: Edge Cases

1. **No MultiBand elements**: Fall back to current logic (Gain > 0, sorted by channel, take first N)
2. **MultiBand count ≠ image channels**: Log warning, proceed with MultiBand-based detection
3. **Detector not found**: Show "-" for that channel's metadata
4. **Duplicate entries**: Use entry with Gain attribute (complete acquisition info)

## Metadata Table Output

| Detector Type | Mode Column |
|--------------|-------------|
| HyD          | "PC" or "Std" |
| PMT          | "-" |

## Validation

Tested against:
- `2024-09-06 gfp-scm young and old embryos 488.lif` (PMT-only, 2 channels)
- `2026-01-20 ALFA Flag Antibody Optimization.lif` (HyD, 3 channels, Series018)

Both correctly identify active detector channels using MultiBand width > 10nm threshold.

## Files to Modify

- `src/lif_figure/reader.py`: Update `_extract_detectors()` and add `_get_active_channels()`
- `docs/metadata-extraction.md`: Update documentation
