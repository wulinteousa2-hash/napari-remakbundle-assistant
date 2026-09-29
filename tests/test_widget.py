"""Integration tests for layer handling and visual QC output."""

from __future__ import annotations

import numpy as np
import pytest

napari = pytest.importorskip("napari")

from napari_remak_bundle_assistant.remak_distance.widget import (  # noqa: E402
    RemakDistanceWidget,
)


def test_widget_measures_shapes_boundary_and_creates_exact_line(
    make_napari_viewer, qtbot
) -> None:
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((11, 11)), name="EM image")
    remak_data = np.zeros((11, 11), dtype=np.uint8)
    remak_data[5, 5] = 7
    remak = viewer.add_labels(remak_data, name="Remak bundles", scale=(2, 1))
    nerve = viewer.add_shapes(
        [np.asarray([[0, 0], [0, 10], [10, 10], [10, 0]])],
        shape_type="polygon",
        name="Nerve ROI",
        scale=(2, 1),
    )
    widget = RemakDistanceWidget(viewer)
    qtbot.addWidget(widget)

    widget.remak_combo.setCurrentIndex(widget.remak_combo.findData(remak))
    widget.nerve_combo.setCurrentIndex(widget.nerve_combo.findData(nerve))
    widget.measure()

    qtbot.waitUntil(lambda: widget._result is not None, timeout=5000)
    qtbot.waitUntil(lambda: widget._measurement_thread is None, timeout=5000)

    assert widget.table.rowCount() == 1
    assert "Remak Centroid–Nerve Boundary Lines" in viewer.layers
    line = viewer.layers["Remak Centroid–Nerve Boundary Lines"].data[0]
    item = widget._result.measurements[0]
    np.testing.assert_array_equal(
        line,
        [
            [item.nearest_remak_y, item.nearest_remak_x],
            [item.nearest_nerve_y, item.nearest_nerve_x],
        ],
    )


def test_shapes_roi_is_accepted(make_napari_viewer, qtbot) -> None:
    viewer = make_napari_viewer()
    remak_data = np.zeros((10, 10), dtype=np.uint8)
    remak_data[5, 5] = 1
    remak = viewer.add_labels(remak_data, name="Remak bundles")
    shapes = viewer.add_shapes(
        [np.asarray([[1, 1], [1, 8], [8, 8], [8, 1]])],
        shape_type="polygon",
        name="Nerve outline",
        scale=(1.1, 1.1),
        translate=(-0.5, -0.5),
    )
    widget = RemakDistanceWidget(viewer)
    qtbot.addWidget(widget)
    widget.remak_combo.setCurrentIndex(widget.remak_combo.findData(remak))
    widget.nerve_combo.setCurrentIndex(widget.nerve_combo.findData(shapes))

    widget.measure()

    qtbot.waitUntil(lambda: widget._result is not None, timeout=5000)
    qtbot.waitUntil(lambda: widget._measurement_thread is None, timeout=5000)

    assert widget._result is not None
    assert len(widget._result.measurements) == 1
    assert not widget.status_label.isHidden()
    assert widget.progress_group.isHidden()


def test_rerun_replaces_restored_qc_layers_without_metadata(
    make_napari_viewer, qtbot
) -> None:
    viewer = make_napari_viewer()
    labels = np.zeros((10, 10), dtype=np.uint8)
    labels[5, 5] = 1
    remak = viewer.add_labels(labels, name="Remak bundles")
    boundary = viewer.add_shapes(
        [np.asarray([[1, 1], [1, 8], [8, 8], [8, 1]])],
        shape_type="polygon",
        name="Nerve boundary",
    )
    old_line = [np.asarray([[5, 5], [1, 5]], dtype=float)]
    viewer.add_shapes(
        old_line,
        shape_type="line",
        name="Remak Centroid–Nerve Boundary Lines",
    )
    viewer.add_shapes(
        old_line,
        shape_type="line",
        name="Remak Centroid–Nerve Boundary Lines [1]",
    )
    viewer.add_points(
        np.asarray([[5, 5]], dtype=float),
        name="Remak Centroid–Nerve Boundary Points",
    )
    widget = RemakDistanceWidget(viewer)
    qtbot.addWidget(widget)
    widget.remak_combo.setCurrentIndex(widget.remak_combo.findData(remak))
    widget.nerve_combo.setCurrentIndex(widget.nerve_combo.findData(boundary))

    widget.measure()
    qtbot.waitUntil(lambda: widget._measurement_thread is None, timeout=5000)

    automatic_lines = [
        layer
        for layer in viewer.layers
        if layer.name.startswith("Remak Centroid–Nerve Boundary Lines")
    ]
    automatic_points = [
        layer
        for layer in viewer.layers
        if layer.name.startswith("Remak Centroid–Nerve Boundary Points")
    ]
    assert len(automatic_lines) == 1
    assert automatic_lines[0].name == "Remak Centroid–Nerve Boundary Lines"
    assert automatic_points == []


def test_automatic_progress_does_not_duplicate_the_step_in_status(
    make_napari_viewer, qtbot
) -> None:
    viewer = make_napari_viewer()
    labels = np.zeros((200, 200), dtype=np.uint8)
    labels[100, 100] = 1
    remak = viewer.add_labels(labels, name="Remak bundles")
    nerve = viewer.add_shapes(
        [np.asarray([[0, 0], [0, 199], [199, 199], [199, 0]])],
        shape_type="polygon",
        name="Nerve boundary",
    )
    widget = RemakDistanceWidget(viewer)
    qtbot.addWidget(widget)
    widget.show()
    widget.remak_combo.setCurrentIndex(widget.remak_combo.findData(remak))
    widget.nerve_combo.setCurrentIndex(widget.nerve_combo.findData(nerve))

    widget.measure()

    assert widget.status_label.isHidden()
    assert not widget.progress_group.isHidden()
    qtbot.waitUntil(lambda: widget._measurement_thread is None, timeout=5000)
    assert not widget.status_label.isHidden()
    assert widget.progress_group.isHidden()


def test_custom_pixel_size_updates_automatic_physical_results(
    make_napari_viewer, qtbot
) -> None:
    viewer = make_napari_viewer()
    labels = np.zeros((11, 11), dtype=np.uint8)
    labels[5, 5] = 1
    remak = viewer.add_labels(labels, name="Remak bundles")
    nerve = viewer.add_shapes(
        [np.asarray([[0, 0], [0, 10], [10, 10], [10, 0]])],
        shape_type="polygon",
        name="Nerve boundary",
    )
    widget = RemakDistanceWidget(viewer)
    qtbot.addWidget(widget)
    widget.remak_combo.setCurrentIndex(widget.remak_combo.findData(remak))
    widget.nerve_combo.setCurrentIndex(widget.nerve_combo.findData(nerve))

    widget.measure()
    qtbot.waitUntil(lambda: widget._measurement_thread is None, timeout=5000)
    assert widget._result is not None
    assert widget._result.measurements[0].min_distance_physical == pytest.approx(5.0)

    widget.pixel_size_y.setValue(2.0)
    widget.pixel_size_x.setValue(3.0)
    widget.pixel_unit_edit.setText("µm")

    widget.apply_calibration_button.click()

    qtbot.waitUntil(lambda: widget._measurement_thread is None, timeout=5000)
    assert widget._result is not None
    item = widget._result.measurements[0]
    assert item.min_distance_px == pytest.approx(5.0)
    assert item.min_distance_physical == pytest.approx(10.0)
    assert widget.table.horizontalHeaderItem(4).text() == "Centroid distance (µm)"
    assert widget.table.horizontalHeaderItem(6).text() == "Area (µm²)"


def test_manual_point_pair_creates_line_and_table_record(
    make_napari_viewer, qtbot
) -> None:
    viewer = make_napari_viewer()
    labels = np.zeros((10, 10), dtype=np.uint8)
    labels[1:4, 1:4] = 7
    labels[6:8, 1:3] = 8
    remak = viewer.add_labels(labels, name="Curated Remak", scale=(2.0, 3.0))
    nerve = viewer.add_shapes(
        [np.asarray([[0, 0], [0, 9], [9, 9], [9, 0]])],
        shape_type="polygon",
        name="Outer nerve boundary",
        scale=(2.0, 3.0),
    )
    widget = RemakDistanceWidget(viewer)
    qtbot.addWidget(widget)
    widget.remak_combo.setCurrentIndex(widget.remak_combo.findData(remak))
    widget.nerve_combo.setCurrentIndex(widget.nerve_combo.findData(nerve))

    widget.start_manual_measurement()
    widget._manual_points_layer.data = np.asarray([[2.0, 2.0], [5.0, 6.0]])

    assert widget.table.rowCount() == 1
    assert len(widget._manual_measurements) == 1
    item = widget._manual_measurements[0]
    assert item.remak_id == 7
    assert item.boundary_x == pytest.approx(9.0)
    assert item.distance_source_px == pytest.approx(np.hypot(3.0, 7.0))
    assert item.distance_physical == pytest.approx(np.hypot(6.0, 21.0))
    assert "Manual Remak–Boundary Lines" in viewer.layers
    assert widget.export_button.isEnabled()
    assert viewer.layers.selection.active is widget._manual_points_layer
    assert widget._manual_points_layer.mode == "add"

    widget._manual_points_layer.data = np.vstack(
        [widget._manual_points_layer.data, [[6.0, 2.0], [8.0, 5.0]]]
    )

    assert widget.table.rowCount() == 2
    assert [item.remak_id for item in widget._manual_measurements] == [7, 8]
    assert len(viewer.layers["Manual Remak–Boundary Lines"].data) == 2
    assert viewer.layers.selection.active is widget._manual_points_layer
    assert widget._manual_points_layer.mode == "add"


def test_manual_first_point_must_hit_positive_remak_label(
    make_napari_viewer, qtbot
) -> None:
    viewer = make_napari_viewer()
    labels = np.zeros((10, 10), dtype=np.uint8)
    labels[4:6, 4:6] = 3
    remak = viewer.add_labels(labels, name="Curated Remak")
    nerve = viewer.add_shapes(
        [np.asarray([[0, 0], [0, 9], [9, 9], [9, 0]])],
        shape_type="polygon",
        name="Outer nerve boundary",
    )
    widget = RemakDistanceWidget(viewer)
    qtbot.addWidget(widget)
    widget.remak_combo.setCurrentIndex(widget.remak_combo.findData(remak))
    widget.nerve_combo.setCurrentIndex(widget.nerve_combo.findData(nerve))

    widget.start_manual_measurement()
    widget._manual_points_layer.data = np.asarray([[1.0, 1.0]])

    assert len(widget._manual_points_layer.data) == 0
    assert "positive Remak label" in widget.status_label.text()
