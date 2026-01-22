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
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qtpy.QtCore import Qt

from napari_figure_maker.exporter import export_figure
from napari_figure_maker.figure_builder import build_figure, COLORMAPS
from napari_figure_maker.models import ChannelConfig, FigureConfig

if TYPE_CHECKING:
    import napari


class FigureMakerWidget(QWidget):
    """Main widget for creating figure panels from napari layers."""

    def __init__(self, napari_viewer: Optional["napari.Viewer"] = None):
        super().__init__()
        self.viewer = napari_viewer
        self._current_layer = None
        self._setup_ui()

    def _setup_ui(self):
        """Set up the widget UI."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Layer selection section
        layer_group = QGroupBox("Select Image Series")
        layer_layout = QVBoxLayout()
        layer_group.setLayout(layer_layout)

        self.layer_combo = QComboBox()
        self.layer_combo.currentIndexChanged.connect(self._on_layer_selected)
        layer_layout.addWidget(self.layer_combo)

        refresh_btn = QPushButton("Refresh Layers")
        refresh_btn.clicked.connect(self._refresh_layers)
        layer_layout.addWidget(refresh_btn)

        layout.addWidget(layer_group)

        # Channel configuration section
        channel_group = QGroupBox("Channels")
        channel_layout = QVBoxLayout()
        channel_group.setLayout(channel_layout)

        # Table: Channel index | Label | Colormap | Visible
        self.channel_table = QTableWidget()
        self.channel_table.setColumnCount(4)
        self.channel_table.setHorizontalHeaderLabels(["Ch", "Label", "Colormap", "Visible"])
        self.channel_table.horizontalHeader().setStretchLastSection(True)
        self.channel_table.setColumnWidth(0, 30)
        self.channel_table.setColumnWidth(1, 80)
        self.channel_table.setColumnWidth(2, 80)
        self.channel_table.setColumnWidth(3, 50)
        channel_layout.addWidget(self.channel_table)

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
        """Refresh the layer dropdown from napari layers."""
        self.layer_combo.clear()
        self.layer_combo.addItem("-- Select a layer --", None)

        if self.viewer is None:
            # Try to get current viewer as fallback
            try:
                import napari
                self.viewer = napari.current_viewer()
            except Exception:
                pass

        if self.viewer is None:
            return

        for layer in self.viewer.layers:
            if hasattr(layer, "data") and isinstance(layer.data, np.ndarray):
                self.layer_combo.addItem(layer.name, layer)

    def _on_layer_selected(self, index: int):
        """Handle layer selection change."""
        layer = self.layer_combo.itemData(index)
        self._current_layer = layer
        self._update_channel_table()

    def _update_channel_table(self):
        """Update channel table based on selected layer."""
        self.channel_table.setRowCount(0)

        if self._current_layer is None:
            return

        data = self._current_layer.data

        # Detect number of channels
        # Assume first dimension is channels if shape is (C, H, W) where C is small
        if data.ndim == 3 and data.shape[0] <= 10:
            n_channels = data.shape[0]
        elif data.ndim == 2:
            n_channels = 1
        else:
            # For higher dimensions, take first axis as channels
            n_channels = data.shape[0] if data.shape[0] <= 10 else 1

        # Default colormaps for common channel counts
        default_colormaps = ["blue", "green", "red", "cyan", "magenta", "yellow", "gray"]

        self.channel_table.setRowCount(n_channels)

        for i in range(n_channels):
            # Channel number
            ch_item = QTableWidgetItem(str(i + 1))
            ch_item.setFlags(ch_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.channel_table.setItem(i, 0, ch_item)

            # Label (editable)
            label_item = QTableWidgetItem(f"Ch{i + 1}")
            self.channel_table.setItem(i, 1, label_item)

            # Colormap dropdown
            cmap_combo = QComboBox()
            cmap_combo.addItems(list(COLORMAPS.keys()))
            if i < len(default_colormaps):
                cmap_combo.setCurrentText(default_colormaps[i])
            self.channel_table.setCellWidget(i, 2, cmap_combo)

            # Visible checkbox
            visible_cb = QCheckBox()
            visible_cb.setChecked(True)
            self.channel_table.setCellWidget(i, 3, visible_cb)

    def _get_channel_configs(self) -> List[ChannelConfig]:
        """Get channel configurations from table."""
        configs = []
        for i in range(self.channel_table.rowCount()):
            label = self.channel_table.item(i, 1).text()
            cmap_combo = self.channel_table.cellWidget(i, 2)
            visible_cb = self.channel_table.cellWidget(i, 3)

            configs.append(ChannelConfig(
                name=f"Ch{i + 1}",
                label=label,
                colormap=cmap_combo.currentText(),
                visible=visible_cb.isChecked(),
            ))
        return configs

    def _build_current_figure(self) -> Optional[np.ndarray]:
        """Build figure from current selections."""
        if self._current_layer is None:
            return None

        data = self._current_layer.data
        channel_configs = self._get_channel_configs()

        # Split channels from the layer data
        if data.ndim == 3 and data.shape[0] <= 10:
            # Shape is (C, H, W)
            channels_data = [data[i] for i in range(data.shape[0])]
        elif data.ndim == 2:
            # Single channel
            channels_data = [data]
        else:
            # Take 2D slices
            channels_data = [data[i] for i in range(min(data.shape[0], len(channel_configs)))]

        # Build figure config
        figure_config = FigureConfig(
            dpi=self.dpi_spin.value(),
            show_merge=self.show_merge_cb.isChecked(),
            background_color=self.bg_combo.currentText(),
        )

        # Get pixel size if available
        pixel_size_um = None
        if hasattr(self._current_layer, "metadata"):
            pixel_size_um = self._current_layer.metadata.get("pixel_size_um")

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
