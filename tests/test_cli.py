"""Tests for CLI module."""

import pytest
from click.testing import CliRunner

from lif_figure.cli import main, sanitize_filename, parse_series_indices


def test_sanitize_filename():
    """sanitize_filename should replace invalid characters."""
    assert sanitize_filename("Normal Name") == "Normal Name"
    assert sanitize_filename("With/Slash") == "With_Slash"
    assert sanitize_filename("With\\Backslash") == "With_Backslash"
    assert sanitize_filename("With:Colon") == "With_Colon"


def test_cli_requires_channels():
    """CLI should error without --channels flag."""
    runner = CliRunner()
    result = runner.invoke(main, ["nonexistent.lif"])

    assert result.exit_code != 0
    assert "channels" in result.output.lower() or "required" in result.output.lower()


def test_cli_requires_lif_file():
    """CLI should error if file doesn't exist."""
    runner = CliRunner()
    result = runner.invoke(main, ["nonexistent.lif", "--channels", "DAPI,GFP"])

    assert result.exit_code != 0
    assert "not found" in result.output.lower() or "exist" in result.output.lower()


def test_cli_help():
    """CLI should show help."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "channels" in result.output
    assert "series" in result.output
    assert "zstack" in result.output


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

    def test_range_inclusive(self):
        """Range N..M includes both endpoints."""
        assert parse_series_indices("2..5", 10) == [2, 3, 4, 5]

    def test_range_single_element(self):
        """Range N..N returns single element."""
        assert parse_series_indices("3..3", 10) == [3]

    def test_range_full(self):
        """Range covering all indices."""
        assert parse_series_indices("0..9", 10) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

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

    def test_multiple_singles(self):
        """Comma-separated single indices."""
        assert parse_series_indices("1,3,5", 10) == [1, 3, 5]

    def test_mixed_singles_and_ranges(self):
        """Mix of singles and ranges."""
        assert parse_series_indices("0..2,5,8..", 10) == [0, 1, 2, 5, 8, 9]

    def test_sorted_output(self):
        """Output is always sorted."""
        assert parse_series_indices("5,1,3", 10) == [1, 3, 5]

    def test_whitespace_around_commas(self):
        """Whitespace around commas is stripped."""
        assert parse_series_indices("1 , 3 , 5", 10) == [1, 3, 5]

    def test_whitespace_around_dots(self):
        """Whitespace around .. is stripped."""
        assert parse_series_indices("1 .. 3", 10) == [1, 2, 3]

    def test_leading_trailing_whitespace(self):
        """Leading/trailing whitespace stripped."""
        assert parse_series_indices("  2..4  ", 10) == [2, 3, 4]

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

    def test_error_inverted_range(self):
        """Inverted range raises error."""
        with pytest.raises(ValueError, match=r"Invalid range: 5\.\.2 \(start > end\)"):
            parse_series_indices("5..2", 10)

    def test_error_inverted_after_resolution(self):
        """Inverted range after negative resolution raises error."""
        # -1 = 9, -3 = 7, so -1..-3 is 9..7 which is inverted
        with pytest.raises(ValueError, match=r"start > end"):
            parse_series_indices("-1..-3", 10)

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
