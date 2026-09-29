"""Synthetic-geometry tests for Remak-to-nerve distance measurement."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage as ndi

from napari_remak_bundle_assistant.remak_distance.measurements import (
    MeasurementValidationError,
    measure_centroids_to_shapes_boundary,
    measure_remak_to_nerve_distance,
)


def test_chunked_centroids_project_to_vector_boundary() -> None:
    labels = np.zeros((10, 10), dtype=np.uint16)
    labels[4, 4:7] = 7
    boundary = (np.asarray([[0, 0], [0, 9], [9, 9], [9, 0]]),)
    updates: list[tuple[int, int, str]] = []

    result = measure_centroids_to_shapes_boundary(
        labels,
        boundary,
        ("polygon",),
        chunk_rows=2,
        progress_callback=lambda completed, total, message: updates.append(
            (completed, total, message)
        ),
    )

    item = result.measurements[0]
    assert item.remak_id == 7
    assert (item.remak_centroid_y, item.remak_centroid_x) == pytest.approx((4, 5))
    assert item.min_distance_px == pytest.approx(4)
    assert (item.nearest_nerve_y, item.nearest_nerve_x) == pytest.approx((0, 5))
    assert item.remak_area_px == 3
    assert any("label rows" in message for _, _, message in updates)


def test_same_label_disconnected_fragments_form_separate_components() -> None:
    labels = np.zeros((9, 9), dtype=np.uint8)
    labels[2, 2] = 3
    labels[6, 6] = 3
    boundary = (np.asarray([[0, 0], [0, 8], [8, 8], [8, 0]]),)

    result = measure_centroids_to_shapes_boundary(
        labels, boundary, ("polygon",), chunk_rows=3
    )

    assert len(result.measurements) == 2
    assert [item.remak_id for item in result.measurements] == [3, 3]
    assert [item.component_index for item in result.measurements] == [1, 2]
    assert [item.remak_area_px for item in result.measurements] == [1, 1]
    assert [
        (item.remak_centroid_y, item.remak_centroid_x)
        for item in result.measurements
    ] == pytest.approx([(2, 2), (6, 6)])


def test_component_crossing_a_chunk_seam_is_merged() -> None:
    labels = np.zeros((8, 8), dtype=np.uint8)
    labels[2:6, 3:5] = 9
    boundary = (np.asarray([[0, 0], [0, 7], [7, 7], [7, 0]]),)

    result = measure_centroids_to_shapes_boundary(
        labels, boundary, ("polygon",), chunk_rows=2
    )

    assert len(result.measurements) == 1
    assert result.measurements[0].remak_id == 9
    assert result.measurements[0].remak_area_px == 8
    assert result.measurements[0].component_index == 1


def test_vector_projection_respects_anisotropic_physical_scale() -> None:
    labels = np.zeros((21, 11), dtype=np.uint8)
    labels[10, 5] = 1
    boundary = (np.asarray([[0, 0], [0, 10], [20, 10], [20, 0]]),)

    result = measure_centroids_to_shapes_boundary(
        labels, boundary, ("polygon",), scale=(0.5, 2.0)
    )

    item = result.measurements[0]
    assert item.min_distance_physical == pytest.approx(5)
    assert item.min_distance_px == pytest.approx(10)
    assert item.nearest_nerve_y == pytest.approx(
        0
    ) or item.nearest_nerve_y == pytest.approx(20)
    assert item.nearest_nerve_x == pytest.approx(5)


def test_chunked_engine_never_materializes_the_complete_label_layer() -> None:
    array = np.zeros((7, 8), dtype=np.uint8)
    array[3, 4] = 2

    class SliceOnlyLabels:
        shape = array.shape

        def __getitem__(self, key):
            return array[key]

        def __array__(self):
            raise AssertionError("complete layer must not be materialized")

    boundary = (np.asarray([[0, 0], [0, 7], [6, 7], [6, 0]]),)
    result = measure_centroids_to_shapes_boundary(
        SliceOnlyLabels(), boundary, ("polygon",), chunk_rows=2
    )

    assert result.measurements[0].remak_id == 2


def _circular_nerve(shape=(41, 41), center=(20, 20), radius=15):
    yy, xx = np.indices(shape)
    return (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius**2


def test_circular_nerve_single_remak() -> None:
    nerve = _circular_nerve()
    labels = np.zeros(nerve.shape, dtype=np.uint16)
    labels[20, 20] = 1

    result = measure_remak_to_nerve_distance(labels, nerve)

    measurement = result.measurements[0]
    assert measurement.remak_id == 1
    # The raster perimeter contains (6, 19), sqrt(14**2 + 1**2) pixels away.
    expected = np.sqrt(197.0)
    assert measurement.min_distance_px == pytest.approx(expected)
    assert measurement.min_distance_physical == pytest.approx(expected)


def test_multiple_remak_labels() -> None:
    nerve = np.ones((11, 11), dtype=bool)
    labels = np.zeros(nerve.shape, dtype=np.int32)
    labels[5, 5] = 3
    labels[2, 5] = 8

    result = measure_remak_to_nerve_distance(labels, nerve)

    assert [item.remak_id for item in result.measurements] == [3, 8]
    distances = {item.remak_id: item.min_distance_px for item in result.measurements}
    assert distances == pytest.approx({3: 5.0, 8: 2.0})


def test_progress_reports_each_completed_label() -> None:
    nerve = np.ones((11, 11), dtype=bool)
    labels = np.zeros(nerve.shape, dtype=np.int32)
    labels[5, 5] = 3
    labels[2, 5] = 8
    updates: list[tuple[int, int, str]] = []

    measure_remak_to_nerve_distance(
        labels,
        nerve,
        progress_callback=lambda completed, total, message: updates.append(
            (completed, total, message)
        ),
    )

    determinate = [update for update in updates if update[1] > 0]
    assert [(completed, total) for completed, total, _ in determinate] == [
        (0, 2),
        (1, 2),
        (2, 2),
    ]
    assert "ID 8" in determinate[-1][2]


def test_touching_outer_boundary_is_zero() -> None:
    nerve = _circular_nerve()
    labels = np.zeros(nerve.shape, dtype=np.int16)
    labels[5, 20] = 1

    result = measure_remak_to_nerve_distance(labels, nerve)

    measurement = result.measurements[0]
    assert measurement.min_distance_px == 0
    assert measurement.min_distance_physical == 0
    assert (measurement.nearest_nerve_y, measurement.nearest_nerve_x) == (5, 20)


def test_anisotropic_scale_selects_physical_nearest_boundary() -> None:
    nerve = np.ones((21, 11), dtype=bool)
    labels = np.zeros(nerve.shape, dtype=np.uint8)
    labels[10, 5] = 1

    result = measure_remak_to_nerve_distance(labels, nerve, scale=(0.5, 2.0))

    measurement = result.measurements[0]
    assert measurement.min_distance_px == pytest.approx(5.0)
    assert measurement.min_distance_physical == pytest.approx(5.0)
    assert measurement.nearest_nerve_y in (0, 20)
    assert measurement.nearest_nerve_x == 5
    assert measurement.remak_area_physical == pytest.approx(1.0)


def test_remak_outside_roi_reports_ids() -> None:
    nerve = np.zeros((10, 10), dtype=bool)
    nerve[2:8, 2:8] = True
    labels = np.zeros(nerve.shape, dtype=np.int32)
    labels[1, 5] = 12

    with pytest.raises(MeasurementValidationError, match="12"):
        measure_remak_to_nerve_distance(labels, nerve)


def test_empty_nerve_mask_is_invalid() -> None:
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[5, 5] = 1

    with pytest.raises(MeasurementValidationError, match="nerve ROI is empty"):
        measure_remak_to_nerve_distance(labels, np.zeros_like(labels))


def test_empty_remak_layer_is_invalid() -> None:
    labels = np.zeros((10, 10), dtype=np.int32)

    with pytest.raises(MeasurementValidationError, match="no positive labels"):
        measure_remak_to_nerve_distance(labels, np.ones_like(labels))


def test_internal_nerve_hole_is_not_treated_as_outer_boundary() -> None:
    nerve = np.ones((15, 15), dtype=bool)
    nerve[7, 7] = False
    labels = np.zeros(nerve.shape, dtype=np.uint8)
    labels[7, 6] = 1

    result = measure_remak_to_nerve_distance(labels, nerve)

    assert result.measurements[0].min_distance_px == pytest.approx(6.0)


def test_optional_normalized_distance_and_metadata() -> None:
    nerve = np.ones((11, 11), dtype=bool)
    labels = np.zeros(nerve.shape, dtype=np.uint8)
    labels[5, 5] = 1

    result = measure_remak_to_nerve_distance(
        labels,
        nerve,
        scale=(2.0, 3.0),
        include_centroid_distance=True,
        include_normalized_distance=True,
    )

    measurement = result.measurements[0]
    assert result.nerve_area_px == 121
    assert result.nerve_area_physical == pytest.approx(726.0)
    assert measurement.centroid_to_nerve_distance is not None
    assert measurement.normalized_distance == pytest.approx(
        measurement.min_distance_physical / np.sqrt(726.0 / np.pi)
    )


def test_centroid_distance_matches_interpolated_edt_without_full_distance_array() -> None:
    nerve = np.ones((11, 11), dtype=bool)
    labels = np.zeros(nerve.shape, dtype=np.uint8)
    labels[5, 4] = 1
    labels[6, 5] = 1
    scale = (2.0, 3.0)

    result = measure_remak_to_nerve_distance(
        labels,
        nerve,
        scale=scale,
        include_centroid_distance=True,
    )

    boundary = nerve & ~ndi.binary_erosion(
        nerve,
        structure=ndi.generate_binary_structure(2, 1),
        border_value=0,
    )
    distance = ndi.distance_transform_edt(~boundary, sampling=scale)
    expected = ndi.map_coordinates(
        distance,
        [[5.5], [4.5]],
        order=1,
        mode="nearest",
    )[0]
    assert result.measurements[0].centroid_to_nerve_distance == pytest.approx(expected)


@pytest.mark.parametrize(
    "labels, nerve, message",
    [
        (np.zeros((3, 4), dtype=int), np.ones((4, 3)), "identical"),
        (np.ones((3, 3), dtype=float), np.ones((3, 3)), "integer"),
        (np.ones((3, 3, 1), dtype=int), np.ones((3, 3, 1)), "2D"),
    ],
)
def test_invalid_array_inputs(labels, nerve, message) -> None:
    with pytest.raises(MeasurementValidationError, match=message):
        measure_remak_to_nerve_distance(labels, nerve)
