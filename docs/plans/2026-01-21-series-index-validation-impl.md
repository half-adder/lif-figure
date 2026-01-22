# Series Index Validation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace `parse_series_indices` with comprehensive validation, new `..` range syntax, negative index support, and open-ended ranges.

**Architecture:** Single function rewrite with helper for resolving negative indices. TDD approach - write failing tests first, then implement. Breaking change to range syntax.

**Tech Stack:** Python, pytest, Click CLI

---

## Task 1: Test Valid Single Indices

**Files:**
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**

Add to `tests/test_cli.py`:

```python
from lif_figure.cli import parse_series_indices


class TestParseSeriesIndices:
    """Tests for parse_series_indices function."""

    def test_single_index(self):
        """Single index returns list with that index."""
        assert parse_series_indices("5", 10) == [5]

    def test_single_zero(self):
        """Index 0 is valid."""
        assert parse_series_indices("0", 10) == [0]

    def test_single_last_valid(self):
        """Last valid index works."""
        assert parse_series_indices("9", 10) == [9]
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py::TestParseSeriesIndices -v`
Expected: PASS (existing function handles these)

**Step 3: Commit test**

```bash
git add tests/test_cli.py
git commit -m "test: add single index tests for parse_series_indices"
```

---

## Task 2: Test Valid Range Syntax (New `..` Separator)

**Files:**
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**

Add to `TestParseSeriesIndices` class:

```python
    def test_range_inclusive(self):
        """Range N..M includes both endpoints."""
        assert parse_series_indices("2..5", 10) == [2, 3, 4, 5]

    def test_range_single_element(self):
        """Range N..N returns single element."""
        assert parse_series_indices("3..3", 10) == [3]

    def test_range_full(self):
        """Range covering all indices."""
        assert parse_series_indices("0..9", 10) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py::TestParseSeriesIndices::test_range_inclusive -v`
Expected: FAIL (old function uses `-` not `..`)

**Step 3: Commit failing tests**

```bash
git add tests/test_cli.py
git commit -m "test: add range syntax tests using .. separator"
```

---

## Task 3: Test Negative Indices

**Files:**
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**

Add to `TestParseSeriesIndices` class:

```python
    def test_negative_last(self):
        """Negative -1 means last index."""
        assert parse_series_indices("-1", 10) == [9]

    def test_negative_third_from_end(self):
        """Negative -3 means third from end."""
        assert parse_series_indices("-3", 10) == [7]

    def test_negative_range(self):
        """Range with negative indices."""
        assert parse_series_indices("-3..-1", 10) == [7, 8, 9]

    def test_mixed_positive_negative_range(self):
        """Range from positive to negative."""
        assert parse_series_indices("5..-1", 10) == [5, 6, 7, 8, 9]
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py::TestParseSeriesIndices::test_negative_last -v`
Expected: FAIL (negative indices not supported)

**Step 3: Commit failing tests**

```bash
git add tests/test_cli.py
git commit -m "test: add negative index tests"
```

---

## Task 4: Test Open-Ended Ranges

**Files:**
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**

Add to `TestParseSeriesIndices` class:

```python
    def test_open_start(self):
        """..N means 0 through N."""
        assert parse_series_indices("..2", 10) == [0, 1, 2]

    def test_open_end(self):
        """N.. means N through last."""
        assert parse_series_indices("7..", 10) == [7, 8, 9]

    def test_open_start_negative_end(self):
        """..-2 means 0 through second-to-last."""
        assert parse_series_indices("..-2", 10) == [0, 1, 2, 3, 4, 5, 6, 7, 8]

    def test_open_negative_start(self):
        """-3.. means third-from-end through last."""
        assert parse_series_indices("-3..", 10) == [7, 8, 9]
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py::TestParseSeriesIndices::test_open_start -v`
Expected: FAIL (open ranges not supported)

**Step 3: Commit failing tests**

```bash
git add tests/test_cli.py
git commit -m "test: add open-ended range tests"
```

---

## Task 5: Test Comma-Separated Mixed Syntax

**Files:**
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**

Add to `TestParseSeriesIndices` class:

```python
    def test_multiple_singles(self):
        """Comma-separated single indices."""
        assert parse_series_indices("1,3,5", 10) == [1, 3, 5]

    def test_mixed_singles_and_ranges(self):
        """Mix of singles and ranges."""
        assert parse_series_indices("0..2,5,8..", 10) == [0, 1, 2, 5, 8, 9]

    def test_sorted_output(self):
        """Output is always sorted."""
        assert parse_series_indices("5,1,3", 10) == [1, 3, 5]
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py::TestParseSeriesIndices::test_mixed_singles_and_ranges -v`
Expected: FAIL (new syntax)

**Step 3: Commit failing tests**

```bash
git add tests/test_cli.py
git commit -m "test: add comma-separated mixed syntax tests"
```

---

## Task 6: Test Whitespace Handling

**Files:**
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**

Add to `TestParseSeriesIndices` class:

```python
    def test_whitespace_around_commas(self):
        """Whitespace around commas is stripped."""
        assert parse_series_indices("1 , 3 , 5", 10) == [1, 3, 5]

    def test_whitespace_around_dots(self):
        """Whitespace around .. is stripped."""
        assert parse_series_indices("1 .. 3", 10) == [1, 2, 3]

    def test_leading_trailing_whitespace(self):
        """Leading/trailing whitespace stripped."""
        assert parse_series_indices("  2..4  ", 10) == [2, 3, 4]
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py::TestParseSeriesIndices::test_whitespace_around_dots -v`
Expected: FAIL (new syntax)

**Step 3: Commit failing tests**

```bash
git add tests/test_cli.py
git commit -m "test: add whitespace handling tests"
```

---

## Task 7: Test Error - Inverted Range

**Files:**
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**

Add to `TestParseSeriesIndices` class:

```python
    def test_error_inverted_range(self):
        """Inverted range raises error."""
        with pytest.raises(ValueError, match=r"Invalid range: 5\.\.2 \(start > end\)"):
            parse_series_indices("5..2", 10)

    def test_error_inverted_after_resolution(self):
        """Inverted range after negative resolution raises error."""
        # -1 = 9, -3 = 7, so -1..-3 is 9..7 which is inverted
        with pytest.raises(ValueError, match=r"start > end"):
            parse_series_indices("-1..-3", 10)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py::TestParseSeriesIndices::test_error_inverted_range -v`
Expected: FAIL (currently silently produces empty range)

**Step 3: Commit failing tests**

```bash
git add tests/test_cli.py
git commit -m "test: add inverted range error tests"
```

---

## Task 8: Test Error - Duplicates

**Files:**
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**

Add to `TestParseSeriesIndices` class:

```python
    def test_error_duplicate_single(self):
        """Duplicate single index raises error."""
        with pytest.raises(ValueError, match=r"Duplicate index: 1"):
            parse_series_indices("1,1", 10)

    def test_error_duplicate_from_range(self):
        """Index appearing in range and separately raises error."""
        with pytest.raises(ValueError, match=r"Duplicate index: 2"):
            parse_series_indices("1..3,2", 10)

    def test_error_overlapping_ranges(self):
        """Overlapping ranges raise error."""
        with pytest.raises(ValueError, match=r"Duplicate index"):
            parse_series_indices("1..5,3..7", 10)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py::TestParseSeriesIndices::test_error_duplicate_single -v`
Expected: FAIL (currently dedupes silently)

**Step 3: Commit failing tests**

```bash
git add tests/test_cli.py
git commit -m "test: add duplicate index error tests"
```

---

## Task 9: Test Error - Out of Bounds

**Files:**
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**

Add to `TestParseSeriesIndices` class:

```python
    def test_error_out_of_bounds_positive(self):
        """Index >= max_index raises error."""
        with pytest.raises(ValueError, match=r"Index 15 out of range \(0-9\)"):
            parse_series_indices("15", 10)

    def test_error_out_of_bounds_negative(self):
        """Negative index too large raises error."""
        with pytest.raises(ValueError, match=r"Index -20 out of range \(-10 to 9\)"):
            parse_series_indices("-20", 10)

    def test_error_range_end_out_of_bounds(self):
        """Range end out of bounds raises error."""
        with pytest.raises(ValueError, match=r"out of range"):
            parse_series_indices("5..15", 10)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py::TestParseSeriesIndices::test_error_out_of_bounds_negative -v`
Expected: FAIL (negative bounds message different)

**Step 3: Commit failing tests**

```bash
git add tests/test_cli.py
git commit -m "test: add out of bounds error tests"
```

---

## Task 10: Test Error - Malformed Input

**Files:**
- Modify: `tests/test_cli.py`

**Step 1: Write the failing tests**

Add to `TestParseSeriesIndices` class:

```python
    def test_error_empty_string(self):
        """Empty string raises error."""
        with pytest.raises(ValueError, match=r"No indices specified"):
            parse_series_indices("", 10)

    def test_error_empty_part(self):
        """Empty part between commas raises error."""
        with pytest.raises(ValueError, match=r"Empty index in specification"):
            parse_series_indices("1,,3", 10)

    def test_error_leading_comma(self):
        """Leading comma raises error."""
        with pytest.raises(ValueError, match=r"Empty index in specification"):
            parse_series_indices(",1,3", 10)

    def test_error_non_numeric(self):
        """Non-numeric value raises error."""
        with pytest.raises(ValueError, match=r"Invalid index: 'foo'"):
            parse_series_indices("foo", 10)

    def test_error_float(self):
        """Float value raises error."""
        with pytest.raises(ValueError, match=r"Invalid index: '1\.5'"):
            parse_series_indices("1.5", 10)

    def test_error_malformed_range(self):
        """Multiple .. in range raises error."""
        with pytest.raises(ValueError, match=r"Invalid range syntax: '1\.\.2\.\.3'"):
            parse_series_indices("1..2..3", 10)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py::TestParseSeriesIndices::test_error_empty_string -v`
Expected: FAIL (different error handling)

**Step 3: Commit failing tests**

```bash
git add tests/test_cli.py
git commit -m "test: add malformed input error tests"
```

---

## Task 11: Implement New parse_series_indices Function

**Files:**
- Modify: `src/lif_figure/cli.py:34-50`

**Step 1: Replace the function**

Replace `parse_series_indices` in `src/lif_figure/cli.py`:

```python
def parse_series_indices(spec: str, max_index: int) -> list[int]:
    """Parse index specification like '0..2,5,8..' into [0,1,2,5,8,9].

    Supports:
    - Single indices: 5
    - Ranges: 2..5 (inclusive)
    - Negative indices: -1 (last), -3 (third from end)
    - Open ranges: ..3 (start to 3), 5.. (5 to end)
    - Mixed: 0..2,5,-1

    Raises ValueError for invalid input.
    """
    spec = spec.strip()
    if not spec:
        raise ValueError("No indices specified")

    def resolve_index(idx_str: str, context: str = "") -> int:
        """Parse and resolve a single index (handles negatives)."""
        idx_str = idx_str.strip()
        try:
            idx = int(idx_str)
        except ValueError:
            raise ValueError(f"Invalid index: '{idx_str}'")

        # Resolve negative indices
        if idx < 0:
            resolved = max_index + idx
            if resolved < 0:
                raise ValueError(f"Index {idx} out of range (-{max_index} to {max_index - 1})")
            return resolved
        else:
            if idx >= max_index:
                raise ValueError(f"Index {idx} out of range (0-{max_index - 1})")
            return idx

    indices = []
    seen = set()

    for part in spec.split(","):
        part = part.strip()

        if not part:
            raise ValueError("Empty index in specification")

        if ".." in part:
            # Range syntax
            segments = part.split("..")
            if len(segments) != 2:
                raise ValueError(f"Invalid range syntax: '{part}'")

            start_str, end_str = segments
            start_str = start_str.strip()
            end_str = end_str.strip()

            # Handle open-ended ranges
            if start_str == "":
                start = 0
            else:
                start = resolve_index(start_str)

            if end_str == "":
                end = max_index - 1
            else:
                end = resolve_index(end_str)

            # Check for inverted range
            if start > end:
                raise ValueError(f"Invalid range: {part.strip()} (start > end)")

            # Add range indices, checking for duplicates
            for i in range(start, end + 1):
                if i in seen:
                    raise ValueError(f"Duplicate index: {i}")
                seen.add(i)
                indices.append(i)
        else:
            # Single index
            idx = resolve_index(part)
            if idx in seen:
                raise ValueError(f"Duplicate index: {idx}")
            seen.add(idx)
            indices.append(idx)

    return sorted(indices)
```

**Step 2: Run all tests**

Run: `pytest tests/test_cli.py::TestParseSeriesIndices -v`
Expected: All PASS

**Step 3: Commit implementation**

```bash
git add src/lif_figure/cli.py
git commit -m "feat: rewrite parse_series_indices with .. syntax and validation

BREAKING CHANGE: Range syntax changed from N-M to N..M
- Support negative indices (-1 = last)
- Support open-ended ranges (..N, N..)
- Error on inverted ranges, duplicates, malformed input"
```

---

## Task 12: Update CLI Help Text

**Files:**
- Modify: `src/lif_figure/cli.py:66-69`

**Step 1: Update the help string**

Change the `--series-index` option help text:

```python
@click.option(
    "--series-index", "-si",
    default=None,
    help="Series indices (0-indexed): single (5), range (2..5), negative (-1=last), open (3.., ..3). Comma-separated.",
)
```

**Step 2: Run CLI help to verify**

Run: `lif-figure --help`
Expected: Updated help text shows `..` syntax

**Step 3: Commit**

```bash
git add src/lif_figure/cli.py
git commit -m "docs: update --series-index help text for new syntax"
```

---

## Task 13: Update README

**Files:**
- Modify: `README.md`

**Step 1: Find and update the series-index documentation**

Find line with `--series-index` in the options table and update:

```markdown
| `--series-index` | `-si` | Series indices: `5`, `2..5`, `-1`, `3..`, `..3` (0-indexed) |
```

**Step 2: Verify README renders correctly**

Read the README and check the table formatting.

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: update README for new --series-index syntax"
```

---

## Task 14: Run Full Test Suite

**Files:** None (verification only)

**Step 1: Run all tests**

Run: `pytest -v`
Expected: All tests pass

**Step 2: Run linter**

Run: `ruff check src/ tests/`
Expected: No errors

**Step 3: Manual smoke test**

If you have a LIF file available, test:
```bash
lif-figure sample.lif -c DAPI,GFP -si "0..2,-1"
```

---

## Summary

| Task | Description | Tests |
|------|-------------|-------|
| 1-6 | Valid input tests | 15 tests |
| 7-10 | Error case tests | 14 tests |
| 11 | Implementation | - |
| 12-13 | Documentation | - |
| 14 | Verification | - |

Total: ~29 new tests for `parse_series_indices`.
