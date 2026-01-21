"""Main widget for napari-figure-maker."""

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import numpy as np
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from napari_figure_maker.exporter import export_figure
from napari_figure_maker.figure_builder import build_figure
from napari_figure_maker.models import ChannelConfig, FigureConfig

if TYPE_CHECKING:
    import napari


class FigureMakerWidget(QWidget):
    """Main widget for creating figure panels from napari layers."""

    def __init__(self, napari_viewer: Optional["napari.Viewer"] = None):
        super().__init__()
        self.viewer = napari_viewer
        self._setup_ui()

    def _setup_ui(self):
        """Set up the widget UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Channel selection section
        channel_group = QGroupBox("Channels")
        channel_layout = QVBoxLayout()
        channel_group.setLayout(channel_layout)

        self.channel_list = QListWidget()
        self.channel_list.setSelectionMode(QListWidget.MultiSelection)
        channel_layout.addWidget(self.channel_list)

        refresh_btn = QPushButton("Refresh Layers")
        refresh_btn.clicked.connect(self._refresh_layers)
        channel_layout.addWidget(refresh_btn)

        layout.addWidget(channel_group)

        # Figure options section
        options_group = QGroupBox("Figure Options")
        options_layout = QVBoxLayout()
        options_group.setLayout(options_layout)

        # Show merge checkbox
        self.show_merge_cb = QCheckBox("Show Merge Panel")
        self.show_merge_cb.setChecked(True)
        options_layout.addWidget(self.show_merge_cb)

        # DPI selection
        dpi_layout = QHBoxLayout()
        dpi_layout.addWidget(QLabel("DPI:"))
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(300)
        dpi_layout.addWidget(self.dpi_spin)
        options_layout.addLayout(dpi_layout)

        # Background color
        bg_layout = QHBoxLayout()
        bg_layout.addWidget(QLabel("Background:"))
        self.bg_combo = QComboBox()
        self.bg_combo.addItems(["black", "white"])
        bg_layout.addWidget(self.bg_combo)
        options_layout.addLayout(bg_layout)

        layout.addWidget(options_group)

        # Export section
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout()
        export_group.setLayout(export_layout)

        self.export_btn = QPushButton("Export Figure...")
        self.export_btn.clicked.connect(self._export_figure)
        export_layout.addWidget(self.export_btn)

        self.preview_btn = QPushButton("Preview")
        self.preview_btn.clicked.connect(self._preview_figure)
        export_layout.addWidget(self.preview_btn)

        layout.addWidget(export_group)

        # Add stretch at bottom
        layout.addStretch()

        # Initial refresh
        self._refresh_layers()

    def _refresh_layers(self):
        """Refresh the channel list from napari layers."""
        self.channel_list.clear()

        if self.viewer is None:
            return

        for layer in self.viewer.layers:
            if hasattr(layer, "data") and isinstance(layer.data, np.ndarray):
                item = QListWidgetItem(layer.name)
                item.setData(256, layer)  # Store layer reference
                self.channel_list.addItem(item)
                item.setSelected(True)

    def _get_selected_layers(self) -> List:
        """Get currently selected layers."""
        layers = []
        for item in self.channel_list.selectedItems():
            layer = item.data(256)
            if layer is not None:
                layers.append(layer)
        return layers

    def _build_current_figure(self) -> Optional[np.ndarray]:
        """Build figure from current selections."""
        layers = self._get_selected_layers()
        if not layers:
            return None

        # Build channel configs from layers
        channel_configs = []
        channels_data = []

        for layer in layers:
            # Get layer data (handle multichannel)
            data = layer.data
            if data.ndim == 3 and data.shape[0] <= 4:
                # Multichannel - use first channel for now
                data = data[0]
            elif data.ndim > 2:
                # Take a 2D slice
                data = data[tuple([0] * (data.ndim - 2) + [slice(None), slice(None)])]

            channels_data.append(data)

            # Get colormap name
            colormap = "gray"
            if hasattr(layer, "colormap") and hasattr(layer.colormap, "name"):
                colormap = layer.colormap.name

            channel_configs.append(ChannelConfig(
                name=layer.name,
                colormap=colormap,
                visible=True,
            ))

        # Build figure config
        figure_config = FigureConfig(
            dpi=self.dpi_spin.value(),
            show_merge=self.show_merge_cb.isChecked(),
            background_color=self.bg_combo.currentText(),
        )

        # Get pixel size if available
        pixel_size_um = None
        if layers and hasattr(layers[0], "metadata"):
            pixel_size_um = layers[0].metadata.get("pixel_size_um")

        return build_figure(
            channels_data=channels_data,
            channel_configs=channel_configs,
            figure_config=figure_config,
            pixel_size_um=pixel_size_um,
        )

    def _export_figure(self):
        """Export the current figure."""
        figure = self._build_current_figure()
        if figure is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Figure",
            "",
            "PNG Files (*.png);;TIFF Files (*.tiff *.tif);;All Files (*)",
        )

        if path:
            export_figure(figure, Path(path), dpi=self.dpi_spin.value())

    def _preview_figure(self):
        """Show a preview of the figure."""
        figure = self._build_current_figure()
        if figure is None:
            return

        if self.viewer is not None:
            # Add as a new layer
            self.viewer.add_image(figure, name="Figure Preview", rgb=True)
