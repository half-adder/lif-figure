# Series Index Validation Design

## Overview

Comprehensive validation for the `--series-index` (`-si`) option, including a syntax change from `-` to `..` for range separators to support negative indices.

## Syntax Specification

**Range separator: `..` (replaces `-`)**

| Syntax | Meaning | Example (10 series) |
|--------|---------|---------------------|
| `N` | Single index | `5` → [5] |
| `N..M` | Range inclusive | `2..5` → [2,3,4,5] |
| `-N` | Nth from end | `-1` → [9], `-3` → [7] |
| `..N` | Start through N | `..2` → [0,1,2] |
| `N..` | N through end | `7..` → [7,8,9] |
| `..-N` | Start through Nth from end | `..-2` → [0..8] |
| `-N..` | Nth from end through end | `-3..` → [7,8,9] |
| Mixed | Combine with commas | `0..2,5,-1` → [0,1,2,5,9] |

**Whitespace:** Stripped around `,` and `..` (lenient handling)

## Validation Rules

**Errors (terse messages):**

| Condition | Example | Error Message |
|-----------|---------|---------------|
| Inverted range | `5..2` | `Invalid range: 5..2 (start > end)` |
| Duplicate index | `1,1` or `1..3,2` | `Duplicate index: 2` |
| Out of bounds | `15` (10 series) | `Index 15 out of range (0-9)` |
| Empty part | `1,,3` or `,1` | `Empty index in specification` |
| Non-numeric | `foo` or `1..bar` | `Invalid index: 'foo'` |
| Float value | `1.5` | `Invalid index: '1.5'` |
| Malformed range | `1..2..3` | `Invalid range syntax: '1..2..3'` |
| Empty result | (edge case) | `No indices specified` |
| Negative out of bounds | `-20` (10 series) | `Index -20 out of range (-10 to 9)` |

**Validation order:**

1. Syntax validation (parsing)
2. Resolve negative indices to positive
3. Check bounds
4. Check for duplicates
5. Return sorted list

## Implementation

Replace `parse_series_indices` function in `cli.py:34-50`:

1. Split on commas, strip whitespace from each part
2. For each part, determine type:
   - Empty → error
   - Contains `..` → parse as range
   - Else → parse as single index
3. Range parsing (`..` present):
   - Split on `..` (max 2 parts, else error)
   - Handle open-ended: `..N`, `N..`, `N..M`
   - Parse start/end as integers (error if non-numeric)
   - Resolve negative indices: `-1` → `max_index - 1`
   - Validate start ≤ end after resolution (error if inverted)
   - Expand to list of indices
4. Single index parsing:
   - Parse as integer (error if non-numeric)
   - Resolve negative indices
   - Validate bounds
5. Duplicate check: Track seen indices, error on repeat
6. Return sorted list

## Test Cases

**Valid inputs (10 series, indices 0-9):**

| Input | Expected Output |
|-------|-----------------|
| `5` | [5] |
| `0..2` | [0, 1, 2] |
| `-1` | [9] |
| `-3..-1` | [7, 8, 9] |
| `..2` | [0, 1, 2] |
| `7..` | [7, 8, 9] |
| `0..2,5,8..` | [0, 1, 2, 5, 8, 9] |
| `0, 2 , 4` | [0, 2, 4] |
| `" 1 .. 3 "` | [1, 2, 3] |

**Invalid inputs (should error):**

| Input | Error |
|-------|-------|
| `5..2` | Inverted range |
| `-1..-3` | Inverted range (9..7 after resolution) |
| `1,1` | Duplicate |
| `0..2,1` | Duplicate |
| `15` | Out of bounds |
| `-20` | Negative out of bounds |
| `foo` | Non-numeric |
| `1.5` | Float |
| `1,,3` | Empty part |
| `1..2..3` | Malformed range |
| `` | No indices |

## Documentation Updates

1. **CLI help text** (`cli.py:68`): Change example to `'0..2,5,8..'`, add negative index note
2. **README**: Update `--series-index` examples
3. **Changelog**: Note breaking change

## Breaking Changes

- Range syntax changed from `N-M` to `N..M`
- Added negative index support (`-1` = last)
- Added open-ended ranges (`..N`, `N..`)
- Strict validation: duplicates and inverted ranges now error
