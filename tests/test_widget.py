"""Tests for the main widget."""

import pytest
from unittest.mock import MagicMock

from napari_figure_maker._widget import FigureMakerWidget


def test_widget_creates(qtbot):
    """Widget should instantiate without napari viewer."""
    widget = FigureMakerWidget(napari_viewer=None)
    qtbot.addWidget(widget)

    assert widget is not None


def test_widget_has_export_button(qtbot):
    """Widget should have an export button."""
    widget = FigureMakerWidget(napari_viewer=None)
    qtbot.addWidget(widget)

    # Should have export button
    assert hasattr(widget, "export_btn")


def test_widget_has_layer_combo(qtbot):
    """Widget should have a layer selection dropdown."""
    widget = FigureMakerWidget(napari_viewer=None)
    qtbot.addWidget(widget)

    assert hasattr(widget, "layer_combo")


def test_widget_has_channel_table(qtbot):
    """Widget should have a channel configuration table."""
    widget = FigureMakerWidget(napari_viewer=None)
    qtbot.addWidget(widget)

    assert hasattr(widget, "channel_table")
