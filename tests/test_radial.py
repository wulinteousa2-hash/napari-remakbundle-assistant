import numpy as np
import pytest

from napari_remak_bundle_assistant.remak_distance.measurements import (
    MeasurementResult, RemakMeasurement,
)
from napari_remak_bundle_assistant.remak_distance.radial import measure_radial_distances
from napari_remak_bundle_assistant.remak_distance.radial import measure_radial_shapes


def test_shapes_area_centroid_and_exact_intersection():
    # Extra collinear vertex would bias a simple vertex average.
    polygon = np.array([[0, 0], [0, 5], [0, 20], [20, 20], [20, 0]])
    result = measure_radial_shapes(result_at((10, 14), (2, 2)), (polygon,), ("polygon",))
    r = result.radial_records[0]
    assert r["nerve_centroid_y"] == pytest.approx(10)
    assert r["nerve_centroid_x"] == pytest.approx(10)
    assert r["radial_boundary_x"] == pytest.approx(20)
    assert r["radial_ab_physical"] == pytest.approx(8)
    assert r["radial_bc_physical"] == pytest.approx(12)


def test_shapes_concave_first_exit_and_invalid_path():
    polygon = np.array([[0, 0], [0, 19], [12, 19], [12, 23],
                        [0, 23], [0, 31], [21, 31], [21, 0]])
    r = measure_radial_shapes(result_at((9, 17)), (polygon,), ("polygon",)).radial_records[0]
    assert r["radial_status"] == "valid"
    assert r["radial_boundary_x"] == pytest.approx(19)
    r = measure_radial_shapes(result_at((9, 26)), (polygon,), ("polygon",)).radial_records[0]
    assert r["radial_status"] == "A_to_B_not_inside_nerve"


def result_at(b, scale=(1, 1)):
    item = RemakMeasurement(7, 3, 3, 1, 1, *b, *b, 0, 0, measurement_id=1)
    return MeasurementResult((item,), 100, 100, *scale)


def test_circle_exact_ray_and_legacy_values():
    y, x = np.indices((41, 41))
    mask = (y - 20)**2 + (x - 20)**2 <= 15**2
    original = result_at((20, 30), (2, 3))
    measured = measure_radial_distances(original, mask)
    r = measured.radial_records[0]
    assert measured.measurements == original.measurements
    assert original.radial_records == ()
    assert r["radial_ab_physical"] == pytest.approx(30)
    assert r["radial_bc_physical"] == pytest.approx(16.5)
    assert r["radial_ac_physical"] == pytest.approx(46.5)
    assert r["radial_normalized_position"] == pytest.approx(10 / 15.5)


def test_diagonal_collinearity_anisotropic_distance():
    mask = np.ones((21, 21), bool)
    r = measure_radial_distances(result_at((12, 14), (2, 3)), mask).radial_records[0]
    a = np.array([r["nerve_centroid_y"], r["nerve_centroid_x"]])
    c = np.array([r["radial_boundary_y"], r["radial_boundary_x"]])
    b = np.array([12, 14])
    assert (c - a)[0] * (b - a)[1] == pytest.approx((c - a)[1] * (b - a)[0])
    assert r["radial_ab_physical"] == pytest.approx(np.hypot(4, 12))
    assert r["radial_ac_physical"] == pytest.approx(
        r["radial_ab_physical"] + r["radial_bc_physical"])


def test_irregular_mask_first_exit_and_invalid_reentry():
    mask = np.ones((21, 31), bool)
    mask[:12, 19:23] = False  # indentation: ray exits and later reenters
    r = measure_radial_distances(result_at((9, 17)), mask).radial_records[0]
    assert r["radial_status"] == "valid"
    assert r["radial_boundary_x"] == pytest.approx(18.5)
    r = measure_radial_distances(result_at((9, 26)), mask).radial_records[0]
    assert r["radial_status"] == "A_to_B_not_inside_nerve"
    assert r["radial_bc_physical"] is None


def test_undefined_direction_and_centroid_outside():
    mask = np.ones((21, 21), bool)
    assert measure_radial_distances(result_at((10, 10)), mask).radial_records[0][
        "radial_status"] == "undefined_ray_A_equals_B"
    mask[4:17, 4:17] = False
    r = measure_radial_distances(result_at((2, 10)), mask).radial_records[0]
    assert r["radial_status"] == "A_to_B_not_inside_nerve"


def test_hole_beyond_b_is_not_outer_boundary():
    mask = np.ones((21, 31), bool)
    mask[9:12, 22] = False
    mask[9:12, 8] = False
    r = measure_radial_distances(result_at((10, 19)), mask).radial_records[0]
    assert r["radial_boundary_x"] == pytest.approx(30.5)
