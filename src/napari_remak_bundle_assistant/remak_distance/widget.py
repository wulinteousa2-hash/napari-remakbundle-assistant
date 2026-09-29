"""Qt widget for Remak-to-nerve-edge morphometry."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np
from napari.layers import Image, Labels, Shapes
from qtpy.QtCore import QObject, QThread, Qt, Signal, Slot
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .measurements import (
    MeasurementCancelledError,
    MeasurementResult,
    MeasurementValidationError,
    RemakMeasurement,
    measure_centroids_to_shapes_boundary,
)

_LINES_NAME = "Remak Centroid–Nerve Boundary Lines"
_POINTS_NAME = "Remak Centroid–Nerve Boundary Points"
_LEGACY_LINES_NAME = "Remak–Nerve Distance Lines"
_LEGACY_POINTS_NAME = "Remak–Nerve Nearest Points"
_MANUAL_LINES_NAME = "Manual Remak–Boundary Lines"
_MANUAL_POINTS_NAME = "Manual Remak–Boundary Points"
_QC_ROLE = "napari-remak-bundle-assistant:distance-qc"
_MANUAL_ROLE = "napari-remak-bundle-assistant:manual-distance"


@dataclass(frozen=True)
class _ManualMeasurement:
    measurement_id: int
    remak_id: int
    remak_y: float
    remak_x: float
    boundary_y: float
    boundary_x: float
    distance_source_px: float
    distance_physical: float


@dataclass(frozen=True)
class _ShapesBoundarySource:
    """Thread-safe vector snapshot of the nerve Shapes boundary."""

    data: tuple[np.ndarray, ...]
    shape_types: tuple[str, ...]


class _MeasurementWorker(QObject):
    """Run the chunked scientific calculation outside Qt's UI thread."""

    progress = Signal(int, int, str)
    succeeded = Signal(object)
    failed = Signal(object)
    cancelled = Signal()

    def __init__(
        self,
        labels: Any,
        nerve_boundary: _ShapesBoundarySource,
        scale: tuple[float, float],
        *,
        include_normalized_distance: bool,
    ) -> None:
        super().__init__()
        self._labels = labels
        self._nerve_boundary = nerve_boundary
        self._scale = scale
        self._include_normalized_distance = include_normalized_distance
        self._cancel_event = Event()

    def cancel(self) -> None:
        """Request cancellation at the next safe checkpoint."""
        self._cancel_event.set()

    @Slot()
    def run(self) -> None:
        try:
            result = measure_centroids_to_shapes_boundary(
                self._labels,
                self._nerve_boundary.data,
                self._nerve_boundary.shape_types,
                scale=self._scale,
                include_normalized_distance=self._include_normalized_distance,
                progress_callback=self.progress.emit,
                is_cancelled=self._cancel_event.is_set,
            )
        except MeasurementCancelledError:
            self.cancelled.emit()
        except Exception as error:  # Relay worker failures safely to the UI thread.
            self.failed.emit(error)
        else:
            self.succeeded.emit(result)


class _NumericItem(QTableWidgetItem):
    """A numeric table item that sorts by its numeric value."""

    def __init__(self, value: int | float, decimals: int | None = None) -> None:
        if isinstance(value, int):
            text = str(value)
        elif decimals is None:
            text = f"{value:.8g}"
        else:
            text = f"{value:.{decimals}f}"
        super().__init__(text)
        self.setData(Qt.ItemDataRole.UserRole, float(value))

    def __lt__(self, other: QTableWidgetItem) -> bool:
        own = self.data(Qt.ItemDataRole.UserRole)
        theirs = other.data(Qt.ItemDataRole.UserRole)
        if own is not None and theirs is not None:
            return float(own) < float(theirs)
        return super().__lt__(other)


class RemakDistanceWidget(QWidget):
    """Measure curated Remak centroids to a vector nerve boundary."""

    def __init__(self, napari_viewer: Any) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self._result: MeasurementResult | None = None
        self._row_measurements: dict[int, RemakMeasurement] = {}
        self._manual_measurements: list[_ManualMeasurement] = []
        self._manual_remak: Labels | None = None
        self._manual_nerve: Shapes | None = None
        self._manual_points_layer: Any | None = None
        self._manual_lines_layer: Any | None = None
        self._manual_updating = False
        self._measurement_thread: QThread | None = None
        self._measurement_worker: _MeasurementWorker | None = None
        self._calibration_layer: Any | None = None
        self._build_ui()
        self._connect_layer_events()
        self.refresh_layers()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        introduction = QLabel(
            "Measure each curated Remak bundle centroid to the closest point on "
            "the vector outer nerve boundary. Source layers are never modified."
        )
        introduction.setWordWrap(True)
        layout.addWidget(introduction)

        self.inputs_group = QGroupBox("Inputs")
        form = QFormLayout(self.inputs_group)
        self.image_combo = QComboBox()
        self.image_combo.setToolTip(
            "Optional reference image. Its name is included in exported CSV files."
        )
        self.remak_combo = QComboBox()
        self.remak_combo.setToolTip(
            "Labels layer with 0 as background and one unique positive integer "
            "for every Remak bundle."
        )
        self.remak_combo.currentIndexChanged.connect(
            lambda _index: self._update_calibration_text()
        )
        self.nerve_combo = QComboBox()
        self.nerve_combo.setToolTip(
            "Closed Shapes layer defining the outer nerve boundary."
        )
        form.addRow("Reference image (optional)", self.image_combo)
        form.addRow("Remak bundle labels", self.remak_combo)
        form.addRow("Nerve boundary (Shapes)", self.nerve_combo)

        label_rule = QLabel(
            "Each disconnected 4-connected region is measured separately, even "
            "when regions share a label value; label 0 is ignored."
        )
        label_rule.setWordWrap(True)
        form.addRow("", label_rule)

        refresh = QPushButton("Refresh layers")
        refresh.clicked.connect(self.refresh_layers)
        form.addRow("", refresh)
        layout.addWidget(self.inputs_group)

        self.options_group = QGroupBox("Automatic measurement output")
        option_layout = QVBoxLayout(self.options_group)
        self.lines_check = QCheckBox("Create distance lines")
        self.lines_check.setChecked(True)
        self.lines_check.setToolTip(
            "Connect each Remak centroid to its closest point on the nerve boundary."
        )
        self.points_check = QCheckBox("Create centroid/boundary markers")
        self.normalized_check = QCheckBox("Include normalized distance")
        self.normalized_check.setChecked(True)
        option_layout.addWidget(self.lines_check)
        option_layout.addWidget(self.points_check)
        option_layout.addWidget(self.normalized_check)
        self.manual_group = QGroupBox("Manual label-to-boundary measurement")
        manual_layout = QVBoxLayout(self.manual_group)
        manual_note = QLabel(
            "Uses the selected Remak Labels and Shapes nerve boundary—never the "
            "reference image. For each record, click inside a positive Remak label "
            "first, then click the corresponding outer Shapes boundary."
        )
        manual_note.setWordWrap(True)
        manual_layout.addWidget(manual_note)
        manual_actions = QHBoxLayout()
        self.manual_start_button = QPushButton("Start manual point pairs")
        self.manual_start_button.clicked.connect(self.start_manual_measurement)
        self.manual_undo_button = QPushButton("Undo last pair")
        self.manual_undo_button.setEnabled(False)
        self.manual_undo_button.clicked.connect(self.undo_manual_pair)
        self.manual_finish_button = QPushButton("Finish manual session")
        self.manual_finish_button.setEnabled(False)
        self.manual_finish_button.clicked.connect(self.finish_manual_measurement)
        manual_actions.addWidget(self.manual_start_button)
        manual_actions.addWidget(self.manual_undo_button)
        manual_actions.addWidget(self.manual_finish_button)
        manual_layout.addLayout(manual_actions)
        self.manual_instruction = QLabel(
            "Manual session inactive. Select Remak Labels and a Shapes nerve ROI."
        )
        self.manual_instruction.setWordWrap(True)
        manual_layout.addWidget(self.manual_instruction)
        layout.addWidget(self.manual_group)
        layout.addWidget(self.options_group)

        buttons = QHBoxLayout()
        self.measure_button = QPushButton("Run automatic measurement")
        self.measure_button.setToolTip(
            "Measure each curated label centroid to the closest vector boundary point."
        )
        self.measure_button.clicked.connect(self.measure)
        self.clear_button = QPushButton("Clear results")
        self.clear_button.clicked.connect(self.clear_results)
        self.export_button = QPushButton("Export CSV")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_csv)
        buttons.addWidget(self.measure_button)
        buttons.addWidget(self.clear_button)
        buttons.addWidget(self.export_button)
        layout.addLayout(buttons)

        self.calibration_widget = QWidget()
        calibration_layout = QHBoxLayout(self.calibration_widget)
        calibration_layout.setContentsMargins(0, 0, 0, 0)
        self.unit_label = QLabel("Physical calibration: —")
        calibration_layout.addWidget(self.unit_label)
        calibration_layout.addStretch(1)
        calibration_layout.addWidget(QLabel("Pixel size Y"))
        self.pixel_size_y = QDoubleSpinBox()
        self.pixel_size_y.setDecimals(9)
        self.pixel_size_y.setRange(0.000000001, 1_000_000_000.0)
        self.pixel_size_y.setValue(1.0)
        self.pixel_size_y.setKeyboardTracking(False)
        self.pixel_size_y.setToolTip("Physical size of one pixel along Y.")
        calibration_layout.addWidget(self.pixel_size_y)
        calibration_layout.addWidget(QLabel("X"))
        self.pixel_size_x = QDoubleSpinBox()
        self.pixel_size_x.setDecimals(9)
        self.pixel_size_x.setRange(0.000000001, 1_000_000_000.0)
        self.pixel_size_x.setValue(1.0)
        self.pixel_size_x.setKeyboardTracking(False)
        self.pixel_size_x.setToolTip("Physical size of one pixel along X.")
        calibration_layout.addWidget(self.pixel_size_x)
        calibration_layout.addWidget(QLabel("Unit"))
        self.pixel_unit_edit = QLineEdit("pixels")
        self.pixel_unit_edit.setMaximumWidth(80)
        self.pixel_unit_edit.setPlaceholderText("µm")
        self.pixel_unit_edit.setToolTip("Examples: µm, nm, mm, or pixels.")
        calibration_layout.addWidget(self.pixel_unit_edit)
        self.apply_calibration_button = QPushButton("Apply / update results")
        self.apply_calibration_button.setToolTip(
            "Use this calibration without modifying the source layer. Existing "
            "automatic results are recalculated."
        )
        self.apply_calibration_button.clicked.connect(self._apply_calibration)
        calibration_layout.addWidget(self.apply_calibration_button)
        self.status_label = QLabel("Choose the Remak Labels layer and nerve ROI.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.calibration_widget)
        layout.addWidget(self.status_label)

        self.progress_group = QGroupBox("Quantification progress")
        progress_layout = QVBoxLayout(self.progress_group)
        self.progress_label = QLabel(
            "Step 1 of 7 — Validating Labels and Shapes inputs …"
        )
        self.progress_label.setWordWrap(True)
        progress_layout.addWidget(self.progress_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar)
        progress_buttons = QHBoxLayout()
        progress_buttons.addStretch(1)
        self.cancel_button = QPushButton("Cancel measurement")
        self.cancel_button.clicked.connect(self._request_measurement_cancel)
        progress_buttons.addWidget(self.cancel_button)
        progress_layout.addLayout(progress_buttons)
        self.progress_group.setVisible(False)
        layout.addWidget(self.progress_group)

        self.table = QTableWidget(0, 11)
        self.table.setHorizontalHeaderLabels(
            [
                "Measurement",
                "Remak ID",
                "Component",
                "Centroid distance px",
                "Centroid distance physical",
                "Area px",
                "Area physical",
                "Centroid Y",
                "Centroid X",
                "Boundary Y",
                "Boundary X",
            ]
        )
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(
            """
            QTableWidget {
                background-color: #262930;
                alternate-background-color: #343842;
                color: #f0f0f0;
                gridline-color: #59606b;
            }
            QTableWidget::item {
                color: #f0f0f0;
                padding: 2px;
            }
            QTableWidget::item:selected {
                background-color: #2563a6;
                color: #ffffff;
            }
            QHeaderView::section {
                background-color: #454b55;
                color: #ffffff;
                border: 1px solid #59606b;
                padding: 3px;
            }
            """
        )
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.itemDoubleClicked.connect(self._center_on_result)
        self.table.setToolTip(
            "Sort by any column. Double-click a row to center that Remak bundle."
        )
        layout.addWidget(self.table, 1)

    def _connect_layer_events(self) -> None:
        for event_name in ("inserted", "removed", "reordered"):
            emitter = getattr(self.viewer.layers.events, event_name, None)
            if emitter is not None:
                emitter.connect(self.refresh_layers)

    @staticmethod
    def _selected_layer(combo: QComboBox) -> Any | None:
        return combo.currentData()

    @staticmethod
    def _restore_combo(
        combo: QComboBox,
        layers: list[Any],
        *,
        optional: bool = False,
    ) -> None:
        previous = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        if optional:
            combo.addItem("None", None)
        for layer in layers:
            combo.addItem(layer.name, layer)
        if previous is not None:
            index = combo.findData(previous)
            if index >= 0:
                combo.setCurrentIndex(index)
        combo.blockSignals(False)

    def refresh_layers(self, event: Any | None = None) -> None:
        """Refresh layer selectors while preserving valid selections."""
        del event
        layers = list(self.viewer.layers)
        self._restore_combo(
            self.image_combo,
            [layer for layer in layers if isinstance(layer, Image)],
            optional=True,
        )
        label_layers = [layer for layer in layers if isinstance(layer, Labels)]
        roi_layers = [layer for layer in layers if isinstance(layer, Shapes)]
        self._restore_combo(self.remak_combo, label_layers)
        self._restore_combo(self.nerve_combo, roi_layers)
        self._update_calibration_text()

    def _update_calibration_text(self) -> None:
        layer = self._selected_layer(self.remak_combo)
        if layer is None:
            self.unit_label.setText("Physical calibration: —")
            self.calibration_widget.setEnabled(False)
            self._calibration_layer = None
            return
        self.calibration_widget.setEnabled(True)
        if layer is not self._calibration_layer:
            scale = tuple(float(value) for value in layer.scale[-2:])
            unit = self._physical_unit(layer)
            self.pixel_size_y.setValue(scale[0])
            self.pixel_size_x.setValue(scale[1])
            self.pixel_unit_edit.setText(unit)
            self._calibration_layer = layer
        scale = self._measurement_scale()
        unit = self._measurement_unit()
        if unit in {"pixel", "pixels"} and np.allclose(scale, (1.0, 1.0)):
            self.unit_label.setText("Distance units: pixels")
        else:
            self.unit_label.setText(
                f"Distance calibration: y={scale[0]:g}, x={scale[1]:g} "
                f"{unit} per image pixel"
            )

    def _measurement_scale(self) -> tuple[float, float]:
        return (float(self.pixel_size_y.value()), float(self.pixel_size_x.value()))

    def _measurement_unit(self) -> str:
        return self.pixel_unit_edit.text().strip() or "physical units"

    def _apply_calibration(self) -> None:
        """Apply the editable calibration and refresh existing measurements."""
        self._update_calibration_text()
        if self._measurement_thread is not None:
            return
        if self._result is not None:
            self.measure()
            return
        if self._manual_points_layer is not None:
            self._manual_points_changed()
            self.status_label.setText(
                "Pixel calibration applied; manual physical distances updated."
            )
            return
        self.status_label.setText(
            "Pixel calibration applied. It will be used by the next measurement."
        )

    @staticmethod
    def _physical_unit(layer: Any) -> str:
        units = getattr(layer, "units", None)
        if units is None:
            return "layer units"
        try:
            last_two = [str(value) for value in units[-2:]]
        except (TypeError, IndexError):
            return str(units)
        if last_two[0] == last_two[1]:
            return last_two[0]
        return f"({last_two[0]}, {last_two[1]})"

    def _nerve_boundary(
        self,
        layer: Any,
        remak_layer: Any,
    ) -> _ShapesBoundarySource:
        if isinstance(layer, Shapes):
            if len(layer.data) == 0:
                raise MeasurementValidationError("The nerve Shapes layer is empty.")
            if layer.ndim != 2:
                raise MeasurementValidationError("The nerve Shapes layer must be 2D.")
            shape_types = [str(value) for value in layer.shape_type]
            unsupported = sorted(
                {
                    value
                    for value in shape_types
                    if value not in {"polygon", "rectangle", "ellipse"}
                }
            )
            if unsupported:
                raise MeasurementValidationError(
                    "The nerve ROI must be a closed polygon, rectangle, or ellipse; "
                    "unsupported shape type: " + ", ".join(unsupported)
                )

            # Convert Shapes data through world space onto the Remak data grid.
            transformed_data: list[np.ndarray] = []
            for vertices in layer.data:
                transformed_vertices = []
                for vertex in np.asarray(vertices, dtype=float):
                    world = layer.data_to_world(tuple(vertex))
                    remak_data = remak_layer.world_to_data(world)
                    transformed_vertices.append(np.asarray(remak_data)[-2:])
                transformed_data.append(np.asarray(transformed_vertices, dtype=float))
            return _ShapesBoundarySource(
                data=tuple(transformed_data),
                shape_types=tuple(shape_types),
            )
        raise MeasurementValidationError(
            "Automatic centroid measurement requires a closed Shapes nerve boundary."
        )

    def measure(self) -> None:
        """Validate layers and start a responsive background measurement."""
        remak = self._selected_layer(self.remak_combo)
        nerve = self._selected_layer(self.nerve_combo)
        if remak is None or nerve is None:
            self._show_error("Choose both a Remak Labels layer and a nerve ROI.")
            return
        if remak is nerve:
            self._show_error("The Remak layer and nerve ROI must be different layers.")
            return

        self._update_calibration_text()
        self.clear_results()
        self.measure_button.setEnabled(False)
        self.status_label.setText("Measuring centroids to the vector boundary …")
        try:
            labels = remak.data
            if len(labels.shape) != 2:
                raise MeasurementValidationError("The Remak Labels layer must be 2D.")
            nerve_boundary = self._nerve_boundary(nerve, remak)
        except (MeasurementValidationError, MemoryError, ValueError) as error:
            self.measure_button.setEnabled(True)
            self._show_error(str(error))
            return

        self.progress_label.setText(
            "Step 1 of 7 — Validating Labels and Shapes inputs …"
        )
        self.progress_bar.setRange(0, 0)
        self.cancel_button.setEnabled(True)
        self.status_label.setVisible(False)
        self.calibration_widget.setEnabled(False)
        self.progress_group.setVisible(True)
        self.inputs_group.setEnabled(False)
        self.options_group.setEnabled(False)
        self.clear_button.setEnabled(False)
        QApplication.processEvents()

        thread = QThread()
        worker = _MeasurementWorker(
            labels,
            nerve_boundary,
            self._measurement_scale(),
            include_normalized_distance=self.normalized_check.isChecked(),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._measurement_progressed)
        worker.succeeded.connect(self._measurement_succeeded)
        worker.failed.connect(self._measurement_failed)
        worker.cancelled.connect(self._measurement_cancelled)
        for signal in (worker.succeeded, worker.failed, worker.cancelled):
            signal.connect(thread.quit)
            signal.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._measurement_thread_finished)
        self._measurement_thread = thread
        self._measurement_worker = worker
        thread.start()

    def start_manual_measurement(self) -> None:
        """Start point pairs using the selected Labels and Shapes layers."""
        remak = self._selected_layer(self.remak_combo)
        nerve = self._selected_layer(self.nerve_combo)
        if not isinstance(remak, Labels):
            self._show_error("Choose a Remak Labels layer for manual measurement.")
            return
        if not isinstance(nerve, Shapes):
            self._show_error("Choose a Shapes layer as the manual nerve boundary.")
            return
        if remak.ndim != 2 or nerve.ndim != 2 or len(nerve.data) == 0:
            self._show_error("Manual measurement requires non-empty 2D layers.")
            return
        unsupported = sorted(
            {
                str(value)
                for value in nerve.shape_type
                if str(value) not in {"polygon", "rectangle", "ellipse"}
            }
        )
        if unsupported:
            self._show_error(
                "Manual nerve boundaries must be polygons, rectangles, or "
                "ellipses; unsupported: " + ", ".join(unsupported)
            )
            return
        if self._measurement_thread is not None:
            self._show_error("Wait for the current background operation to finish.")
            return

        self.clear_results()
        self._manual_remak = remak
        self._manual_nerve = nerve
        scale = tuple(float(value) for value in remak.scale[-2:])
        translate = tuple(float(value) for value in remak.translate[-2:])
        self._manual_points_layer = self.viewer.add_points(
            np.empty((0, 2), dtype=float),
            ndim=2,
            name=_MANUAL_POINTS_NAME,
            size=12,
            face_color="yellow",
            border_color="black",
            scale=scale,
            translate=translate,
            metadata={_MANUAL_ROLE: True},
        )
        self._manual_points_layer.events.data.connect(self._manual_points_changed)
        self._manual_points_layer.mode = "add"
        self.viewer.layers.selection.active = self._manual_points_layer
        self.manual_start_button.setEnabled(False)
        self.manual_finish_button.setEnabled(True)
        self.manual_undo_button.setEnabled(False)
        self.manual_instruction.setText(
            "Click 1 inside a positive Remak label. Click 2 on the outer Shapes "
            "boundary. "
            "Continue with another Remak center and boundary pair."
        )
        self.status_label.setText(
            "Manual measurement active: click a Remak bundle center."
        )

    def _manual_points_changed(self, event: Any | None = None) -> None:
        del event
        if self._manual_updating or self._manual_points_layer is None:
            return
        remak = self._manual_remak
        if remak is None:
            return
        points = np.asarray(self._manual_points_layer.data, dtype=float)
        if points.size == 0:
            points = np.empty((0, 2), dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            return

        if points.shape[0] > 0 and points.shape[0] % 2 == 0:
            snapped = self._snap_to_manual_boundary(points[-1])
            if snapped is None:
                message = "The selected Shapes layer has no measurable boundary."
                self.manual_instruction.setText(message)
                self.status_label.setText(message)
                return
            if not np.allclose(points[-1], snapped):
                points = points.copy()
                points[-1] = snapped
                self._manual_updating = True
                try:
                    self._manual_points_layer.data = points
                finally:
                    self._manual_updating = False

        labels = remak.data
        if points.shape[0] % 2:
            latest = points[-1]
            row = int(round(float(latest[0])))
            column = int(round(float(latest[1])))
            valid = (
                0 <= row < labels.shape[0]
                and 0 <= column < labels.shape[1]
                and int(np.asarray(labels[row, column]).item()) > 0
            )
            if not valid:
                self._manual_updating = True
                try:
                    self._manual_points_layer.data = points[:-1]
                finally:
                    self._manual_updating = False
                message = "First point must be inside a positive Remak label."
                self.manual_instruction.setText(message)
                self.status_label.setText(message)
                return

        measurements: list[_ManualMeasurement] = []
        lines: list[np.ndarray] = []
        for index in range(points.shape[0] // 2):
            remak_point = points[index * 2]
            boundary_point = points[index * 2 + 1]
            row = int(round(float(remak_point[0])))
            column = int(round(float(remak_point[1])))
            if not (0 <= row < labels.shape[0] and 0 <= column < labels.shape[1]):
                continue
            remak_id = int(np.asarray(labels[row, column]).item())
            if remak_id <= 0:
                continue
            delta = boundary_point - remak_point
            scale_y, scale_x = self._measurement_scale()
            measurements.append(
                _ManualMeasurement(
                    measurement_id=index + 1,
                    remak_id=remak_id,
                    remak_y=float(remak_point[0]),
                    remak_x=float(remak_point[1]),
                    boundary_y=float(boundary_point[0]),
                    boundary_x=float(boundary_point[1]),
                    distance_source_px=float(np.hypot(delta[0], delta[1])),
                    distance_physical=float(
                        np.hypot(delta[0] * scale_y, delta[1] * scale_x)
                    ),
                )
            )
            lines.append(np.asarray([remak_point, boundary_point], dtype=float))

        self._manual_measurements = measurements
        self._populate_manual_table()
        self._update_manual_lines(lines)
        self.manual_undo_button.setEnabled(points.shape[0] > 0)
        self.export_button.setEnabled(bool(measurements))
        if points.shape[0] % 2:
            message = (
                f"Manual measurement {len(measurements) + 1}: now click the "
                "outer nerve boundary."
            )
        else:
            message = (
                f"Recorded {len(measurements)} manual measurement(s). "
                "Click the next Remak bundle center."
            )
        self.manual_instruction.setText(message)
        self.status_label.setText(message)

    def _snap_to_manual_boundary(self, point: np.ndarray) -> np.ndarray | None:
        """Return the closest point on the selected Shapes boundary."""
        remak = self._manual_remak
        nerve = self._manual_nerve
        if remak is None or nerve is None:
            return None
        best_point: np.ndarray | None = None
        best_squared = np.inf
        for raw_vertices, raw_type in zip(nerve.data, nerve.shape_type, strict=True):
            transformed = []
            for vertex in np.asarray(raw_vertices, dtype=float):
                world = nerve.data_to_world(tuple(vertex))
                transformed.append(np.asarray(remak.world_to_data(world))[-2:])
            vertices = np.asarray(transformed, dtype=float)
            shape_type = str(raw_type)
            if shape_type == "ellipse" and len(vertices) >= 4:
                center = np.mean(vertices[:4], axis=0)
                axis_a = (vertices[1] - vertices[0]) / 2.0
                axis_b = (vertices[3] - vertices[0]) / 2.0
                angles = np.linspace(0.0, 2.0 * np.pi, 257)
                vertices = (
                    center
                    + np.cos(angles)[:, None] * axis_a
                    + np.sin(angles)[:, None] * axis_b
                )
            elif len(vertices) >= 2:
                vertices = np.vstack([vertices, vertices[0]])
            else:
                continue

            starts = vertices[:-1]
            vectors = vertices[1:] - starts
            lengths_squared = np.sum(vectors * vectors, axis=1)
            usable = lengths_squared > 0
            if not np.any(usable):
                continue
            starts = starts[usable]
            vectors = vectors[usable]
            lengths_squared = lengths_squared[usable]
            fractions = np.clip(
                np.sum((point - starts) * vectors, axis=1) / lengths_squared,
                0.0,
                1.0,
            )
            candidates = starts + fractions[:, None] * vectors
            squared = np.sum((candidates - point) ** 2, axis=1)
            index = int(np.argmin(squared))
            if squared[index] < best_squared:
                best_squared = float(squared[index])
                best_point = candidates[index]
        return best_point

    def _populate_manual_table(self) -> None:
        unit = self._measurement_unit()
        headers = [
            "Measurement",
            "Remak ID",
            "Distance px",
            f"Distance ({unit})",
            "Remak Y",
            "Remak X",
            "Boundary Y",
            "Boundary X",
        ]
        self.table.setSortingEnabled(False)
        self.table.clear()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(self._manual_measurements))
        self._row_measurements.clear()
        for row, item in enumerate(self._manual_measurements):
            values = [
                item.measurement_id,
                item.remak_id,
                item.distance_source_px,
                item.distance_physical,
                item.remak_y,
                item.remak_x,
                item.boundary_y,
                item.boundary_x,
            ]
            for column, value in enumerate(values):
                self.table.setItem(row, column, _NumericItem(value))
        self.table.resizeColumnsToContents()
        self.table.setSortingEnabled(True)

    def _update_manual_lines(self, lines: list[np.ndarray]) -> None:
        if self._manual_lines_layer is not None:
            if self._manual_lines_layer in self.viewer.layers:
                if lines:
                    self._manual_lines_layer.data = lines
                else:
                    self.viewer.layers.remove(self._manual_lines_layer)
                    self._manual_lines_layer = None
            else:
                self._manual_lines_layer = None
        if lines and self._manual_lines_layer is None and self._manual_remak is not None:
            self._manual_lines_layer = self.viewer.add_shapes(
                lines,
                shape_type="line",
                name=_MANUAL_LINES_NAME,
                edge_color="cyan",
                edge_width=3,
                scale=tuple(float(value) for value in self._manual_remak.scale[-2:]),
                translate=tuple(
                    float(value) for value in self._manual_remak.translate[-2:]
                ),
                metadata={_MANUAL_ROLE: True},
            )
            self._manual_lines_layer.editable = False

        # Adding or updating a Shapes layer can make it the active layer. Keep
        # point placement active so the user can immediately enter the next
        # Remak/boundary pair without manually reselecting the Points layer.
        if (
            self._manual_points_layer is not None
            and self.manual_finish_button.isEnabled()
        ):
            self._manual_points_layer.mode = "add"
            self.viewer.layers.selection.active = self._manual_points_layer

    def undo_manual_pair(self) -> None:
        """Remove the incomplete point or most recently completed pair."""
        if self._manual_points_layer is None:
            return
        points = np.asarray(self._manual_points_layer.data, dtype=float)
        remove_count = 1 if len(points) % 2 else min(2, len(points))
        if remove_count:
            self._manual_points_layer.data = points[:-remove_count]

    def finish_manual_measurement(self) -> None:
        """Stop point placement while retaining the recorded layers and table."""
        if self._manual_points_layer is not None:
            points = np.asarray(self._manual_points_layer.data)
            if len(points) % 2:
                self._manual_points_layer.data = points[:-1]
            self._manual_points_layer.mode = "select"
        self.manual_start_button.setEnabled(True)
        self.manual_finish_button.setEnabled(False)
        self.manual_undo_button.setEnabled(bool(self._manual_measurements))
        self.manual_instruction.setText(
            f"Manual session finished with {len(self._manual_measurements)} "
            "measurement(s)."
        )
        self.status_label.setText(self.manual_instruction.text())

    @Slot(int, int, str)
    def _measurement_progressed(
        self, completed: int, total: int, message: str
    ) -> None:
        if total > 0:
            if self.progress_bar.maximum() != total:
                self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(completed)
        else:
            self.progress_bar.setRange(0, 0)
        self.progress_label.setText(message)

    @Slot(object)
    def _measurement_succeeded(self, result: MeasurementResult) -> None:
        self._result = result
        try:
            self._measurement_progressed(
                0,
                0,
                "Step 7 of 7 — Populating the table and QC layers …",
            )
            self._populate_table(result)
            remak = self._selected_layer(self.remak_combo)
            if remak is not None:
                self._update_qc_layers(remak, result)
        except Exception as error:
            self._finish_measurement_ui()
            self._show_error(
                "Measurements were calculated, but displaying the results failed: "
                f"{error}"
            )
            return
        self._finish_measurement_ui()
        self.export_button.setEnabled(True)
        self.status_label.setText(
            f"Measured {len(result.measurements)} connected Remak component(s). "
            "Double-click a result to locate it."
        )

    @Slot(object)
    def _measurement_failed(self, error: Exception) -> None:
        self._finish_measurement_ui()
        self._show_error(str(error))

    @Slot()
    def _measurement_cancelled(self) -> None:
        self._finish_measurement_ui()
        self.status_label.setText("Measurement cancelled.")

    def _request_measurement_cancel(self) -> None:
        if self._measurement_worker is not None:
            self._measurement_worker.cancel()
        self.cancel_button.setEnabled(False)
        message = "Cancelling after the current label chunk …"
        self.progress_label.setText(message)

    def _finish_measurement_ui(self) -> None:
        self.progress_group.setVisible(False)
        self.status_label.setVisible(True)
        self.calibration_widget.setEnabled(True)
        self.inputs_group.setEnabled(True)
        self.options_group.setEnabled(True)
        self.clear_button.setEnabled(True)
        self.measure_button.setEnabled(True)

    @Slot()
    def _measurement_thread_finished(self) -> None:
        self._measurement_thread = None
        self._measurement_worker = None

    def _show_error(self, message: str) -> None:
        self.status_label.setText("Measurement stopped: " + message)
        QMessageBox.warning(self, "Remak–Nerve Distance", message)

    def _populate_table(self, result: MeasurementResult) -> None:
        include_normalized = self.normalized_check.isChecked()
        unit = self._measurement_unit()
        headers = [
            "Measurement",
            "Remak ID",
            "Component",
            "Centroid distance px",
            f"Centroid distance ({unit})",
            "Area px",
            f"Area ({unit}²)",
            "Centroid Y",
            "Centroid X",
            "Boundary Y",
            "Boundary X",
        ]
        if include_normalized:
            headers.append("Normalized distance")

        self.table.setSortingEnabled(False)
        self.table.clear()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(result.measurements))
        self._row_measurements.clear()

        for row, item in enumerate(result.measurements):
            values: list[int | float] = [
                item.measurement_id or row + 1,
                item.remak_id,
                item.component_index,
                item.min_distance_px,
                item.min_distance_physical,
                item.remak_area_px,
                item.remak_area_physical,
                item.remak_centroid_y,
                item.remak_centroid_x,
                item.nearest_nerve_y,
                item.nearest_nerve_x,
            ]
            if include_normalized and item.normalized_distance is not None:
                values.append(item.normalized_distance)
            for column, value in enumerate(values):
                table_item = _NumericItem(value)
                measurement_id = item.measurement_id or row + 1
                table_item.setData(Qt.ItemDataRole.UserRole + 1, measurement_id)
                self.table.setItem(row, column, table_item)
            self._row_measurements[item.measurement_id or row + 1] = item

        self.table.resizeColumnsToContents()
        self.table.setSortingEnabled(True)

    def _center_on_result(self, table_item: QTableWidgetItem) -> None:
        measurement_id = table_item.data(Qt.ItemDataRole.UserRole + 1)
        measurement = self._row_measurements.get(int(measurement_id))
        remak = self._selected_layer(self.remak_combo)
        if measurement is None or remak is None:
            return
        point = (measurement.remak_centroid_y, measurement.remak_centroid_x)
        world = remak.data_to_world(point)
        self.viewer.camera.center = tuple(world[-2:])
        self.viewer.layers.selection.active = remak

    def _remove_qc_layers(self) -> None:
        reserved_names = (
            _LINES_NAME,
            _POINTS_NAME,
            _LEGACY_LINES_NAME,
            _LEGACY_POINTS_NAME,
        )
        for layer in list(self.viewer.layers):
            metadata = getattr(layer, "metadata", {})
            name = str(getattr(layer, "name", ""))
            restored_snapshot = any(
                name == base
                or (
                    name.startswith(base + " [")
                    and name.endswith("]")
                    and name[len(base) + 2 : -1].isdigit()
                )
                for base in reserved_names
            )
            if metadata.get(_QC_ROLE, False) or restored_snapshot:
                self.viewer.layers.remove(layer)

    def _update_qc_layers(self, remak: Any, result: MeasurementResult) -> None:
        self._remove_qc_layers()
        scale = tuple(float(value) for value in remak.scale[-2:])
        translate = tuple(float(value) for value in remak.translate[-2:])

        if self.lines_check.isChecked():
            lines = [
                np.asarray(
                    [
                        [item.nearest_remak_y, item.nearest_remak_x],
                        [item.nearest_nerve_y, item.nearest_nerve_x],
                    ],
                    dtype=float,
                )
                for item in result.measurements
            ]
            self.viewer.add_shapes(
                lines,
                shape_type="line",
                name=_LINES_NAME,
                edge_color="cyan",
                edge_width=2,
                scale=scale,
                translate=translate,
                features={
                    "measurement_id": [
                        item.measurement_id for item in result.measurements
                    ],
                    "remak_id": [item.remak_id for item in result.measurements],
                    "component": [
                        item.component_index for item in result.measurements
                    ],
                },
                metadata={_QC_ROLE: True},
            )

        if self.points_check.isChecked():
            points: list[list[float]] = []
            remak_ids: list[int] = []
            measurement_ids: list[int | None] = []
            component_indices: list[int] = []
            endpoint_types: list[str] = []
            for item in result.measurements:
                points.extend(
                    [
                        [item.nearest_remak_y, item.nearest_remak_x],
                        [item.nearest_nerve_y, item.nearest_nerve_x],
                    ]
                )
                remak_ids.extend([item.remak_id, item.remak_id])
                measurement_ids.extend(
                    [item.measurement_id, item.measurement_id]
                )
                component_indices.extend(
                    [item.component_index, item.component_index]
                )
                endpoint_types.extend(["remak", "nerve"])
            self.viewer.add_points(
                np.asarray(points, dtype=float),
                name=_POINTS_NAME,
                size=8,
                face_color="yellow",
                border_color="black",
                scale=scale,
                translate=translate,
                features={
                    "measurement_id": measurement_ids,
                    "remak_id": remak_ids,
                    "component": component_indices,
                    "endpoint": endpoint_types,
                },
                metadata={_QC_ROLE: True},
            )

    def clear_results(self) -> None:
        """Clear derived results and plugin-created visual QC layers."""
        self._result = None
        self._row_measurements.clear()
        self._manual_measurements.clear()
        self.table.clearContents()
        self.table.setRowCount(0)
        self.export_button.setEnabled(False)
        self._remove_qc_layers()
        self._manual_updating = True
        try:
            if self._manual_points_layer is not None:
                try:
                    self._manual_points_layer.events.data.disconnect(
                        self._manual_points_changed
                    )
                except (TypeError, ValueError):
                    pass
            for layer in list(self.viewer.layers):
                if getattr(layer, "metadata", {}).get(_MANUAL_ROLE, False):
                    self.viewer.layers.remove(layer)
        finally:
            self._manual_updating = False
        self._manual_points_layer = None
        self._manual_lines_layer = None
        self._manual_remak = None
        self._manual_nerve = None
        self.manual_start_button.setEnabled(True)
        self.manual_finish_button.setEnabled(False)
        self.manual_undo_button.setEnabled(False)
        self.manual_instruction.setText(
            "Manual session inactive. Select Remak Labels and a Shapes nerve ROI."
        )
        self.status_label.setText("Results cleared. Source layers were not modified.")

    def export_csv(self) -> None:
        """Export per-bundle measurements with repeated image-level metadata."""
        if self._result is None and not self._manual_measurements:
            self._show_error("Create measurements before exporting a CSV file.")
            return
        manual = bool(self._manual_measurements) and self._result is None
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Remak–nerve distances",
            (
                "manual_remak_nerve_distances.csv"
                if manual
                else "remak_nerve_distances.csv"
            ),
            "CSV files (*.csv)",
        )
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"

        image = self._selected_layer(self.image_combo)
        image_name = image.name if image is not None else ""
        remak = self._selected_layer(self.remak_combo)
        unit = self._measurement_unit()
        if manual:
            manual_remak = self._manual_remak or remak
            manual_nerve = self._manual_nerve
            assert manual_remak is not None
            common = {
                "measurement_mode": "manual_point_pair",
                "remak_layer": manual_remak.name,
                "nerve_shapes_layer": (
                    manual_nerve.name if manual_nerve is not None else ""
                ),
                "pixel_size_y": self.pixel_size_y.value(),
                "pixel_size_x": self.pixel_size_x.value(),
                "physical_unit": unit,
            }
            rows = [
                {
                    **common,
                    "measurement_id": item.measurement_id,
                    "remak_id": item.remak_id,
                    "distance_source_px": item.distance_source_px,
                    "distance_physical": item.distance_physical,
                    "remak_y": item.remak_y,
                    "remak_x": item.remak_x,
                    "boundary_y": item.boundary_y,
                    "boundary_x": item.boundary_x,
                }
                for item in self._manual_measurements
            ]
        else:
            assert self._result is not None
            common = {
                "measurement_mode": "automatic_centroid_to_vector_boundary",
                "image_name": image_name,
                "nerve_area_px": self._result.nerve_area_px,
                "nerve_area_physical": self._result.nerve_area_physical,
                "pixel_size_y": self._result.pixel_size_y,
                "pixel_size_x": self._result.pixel_size_x,
                "physical_unit": unit,
                "number_of_remak_bundles": len(self._result.measurements),
            }
            rows = [
                {**common, **item.as_record()}
                for item in self._result.measurements
            ]
        if not rows:
            self._show_error("There are no measurements to export.")
            return
        try:
            with Path(path).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        except OSError as error:
            self._show_error(f"Could not write CSV: {error}")
            return
        self.status_label.setText(f"Exported {len(rows)} row(s) to {path}")
