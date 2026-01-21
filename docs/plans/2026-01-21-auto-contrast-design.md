# Auto-Contrast Feature Design

## Overview

Add percentile-based auto-contrast to improve visibility of details in microscopy images by handling both bright outliers (hot pixels) and weak/dim signals.

## Behavior

Auto-contrast uses **percentile-based clipping**. Instead of stretching min→max to 0→1, it:
1. Finds the pixel value at the low percentile (e.g., 0.1th)
2. Finds the pixel value at the high percentile (e.g., 99.9th)
3. Clips values outside that range
4. Stretches the clipped range to 0→1

This ignores hot pixels (bright outliers) and boosts weak signals by not letting dark background dominate.

**Default percentiles:** 0.1 and 99.9 (clips extreme 0.1% on each end)

## Interface

### CLI Flag

- `--auto-contrast` or `-a` — enables with defaults (0.1, 99.9)
- `--auto-contrast 0.5,99.5` — enables with custom percentiles

### Config File (`lif-figure.yaml`)

```yaml
auto_contrast_percentiles: [0.1, 99.9]
```

When `--auto-contrast` is used without values, it checks the config file for custom percentiles, falling back to defaults if not specified.

### Precedence

CLI values > config file > defaults (0.1, 99.9)

## Implementation

### Files to Modify

1. **`config.py`** — Add `auto_contrast_percentiles: tuple[float, float]` field with default `(0.1, 99.9)`

2. **`figure.py`** — Update `normalize_channel()` to accept optional percentiles parameter. When provided, use `np.percentile()` to compute clip bounds instead of min/max

3. **`cli.py`** — Add `--auto-contrast/-a` option that accepts optional comma-separated values. Pass to config/figure as appropriate

### No Changes Needed

- `reader.py`
- Existing tests (current behavior preserved when flag not used)

### New Tests

- Verify percentile clipping works correctly
- Verify CLI flag parsing with and without values
