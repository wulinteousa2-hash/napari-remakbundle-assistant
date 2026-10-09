"""Exact ray traversal of pixel cells for independent radial measurements."""

from dataclasses import replace

import numpy as np
from scipy import ndimage as ndi

from .measurements import (
    MeasurementCancelledError,
    MeasurementResult,
    MeasurementValidationError,
    _boundary_polyline,
)


def measure_radial_shapes(result, boundaries, shape_types, *, is_cancelled=None):
    """Measure against one closed vector region, without allocating a mask.

    A is the enclosed area centroid. Ellipses use the same sampled polyline
    as the existing shortest-distance engine.
    """
    from skimage.measure import points_in_poly

    if len(boundaries) != 1:
        raise MeasurementValidationError(
            "Radial Shapes analysis requires one closed nerve shape; "
            "use a Labels mask for a region composed of multiple shapes."
        )
    polygon = _boundary_polyline(boundaries[0], shape_types[0])
    starts, ends = polygon[:-1], polygon[1:]
    cross = starts[:, 0] * ends[:, 1] - ends[:, 0] * starts[:, 1]
    signed_area2 = cross.sum()
    if abs(signed_area2) < 1e-12:
        raise MeasurementValidationError("The radial nerve shape has zero enclosed area.")
    a = ((starts + ends) * cross[:, None]).sum(axis=0) / (3 * signed_area2)
    edges = ends - starts
    scale = np.array([result.pixel_size_y, result.pixel_size_x])
    records = []
    for item in result.measurements:
        if is_cancelled is not None and is_cancelled():
            raise MeasurementCancelledError("Measurement cancelled.")
        b = np.array([item.remak_centroid_y, item.remak_centroid_x])
        direction = b - a
        record = dict(radial_ab_physical=None, radial_bc_physical=None,
                      radial_ac_physical=None, radial_normalized_position=None,
                      radial_status="valid", nerve_centroid_y=float(a[0]),
                      nerve_centroid_x=float(a[1]), radial_boundary_y=None,
                      radial_boundary_x=None)
        records.append(record)
        if np.linalg.norm(direction) < 1e-12:
            record["radial_status"] = "undefined_ray_A_equals_B"
            continue
        delta = starts - a
        denominator = direction[0] * edges[:, 1] - direction[1] * edges[:, 0]
        usable = np.abs(denominator) > 1e-12
        t = np.zeros(len(edges))
        u = t.copy()
        t[usable] = (delta[usable, 0] * edges[usable, 1]
                     - delta[usable, 1] * edges[usable, 0]) / denominator[usable]
        u[usable] = (delta[usable, 0] * direction[1]
                     - delta[usable, 1] * direction[0]) / denominator[usable]
        hits = t[usable & (t >= 0) & (u >= -1e-10) & (u <= 1 + 1e-10)]
        knots = np.unique(np.concatenate(([0., 1.], hits)))
        knots = np.append(knots, knots[-1] + 1)
        mids = a + ((knots[:-1] + knots[1:]) / 2)[:, None] * direction
        inside = points_in_poly(mids, polygon)
        if not points_in_poly(np.array([a, b]), polygon).all() or np.any(
            ~inside[knots[:-1] < 1]
        ):
            record["radial_status"] = "A_to_B_not_inside_nerve"
            continue
        exits = np.flatnonzero((knots[:-1] >= 1) & ~inside)
        if not len(exits) or knots[exits[0]] <= 1:
            record["radial_status"] = "no_outward_intersection_beyond_B"
            continue
        tc = float(knots[exits[0]])
        c = a + tc * direction
        ab = float(np.linalg.norm(direction * scale))
        record.update(radial_ab_physical=ab, radial_bc_physical=(tc - 1) * ab,
                      radial_ac_physical=tc * ab, radial_normalized_position=1 / tc,
                      radial_boundary_y=float(c[0]), radial_boundary_x=float(c[1]))
    return replace(result, radial_records=tuple(records))


def measure_radial_distances(result: MeasurementResult, nerve_mask, *,
                             is_cancelled=None) -> MeasurementResult:
    """Reuse measured B centroids; intersect A→B rays with outer pixel-cell edges.

    Holes are excluded when validating A–B, but filled for finding the outer
    boundary C2. All intervals between grid crossings are checked, so narrow
    excursions outside a concave mask cannot be skipped by sampling.
    """
    mask = np.asarray(nerve_mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        raise MeasurementValidationError("Radial analysis needs a nonempty 2D nerve mask.")
    a = np.asarray(ndi.center_of_mass(mask))
    outer = ndi.binary_fill_holes(mask)
    scale = np.array([result.pixel_size_y, result.pixel_size_x])
    records = []
    for item in result.measurements:
        if is_cancelled is not None and is_cancelled():
            raise MeasurementCancelledError("Measurement cancelled.")
        b = np.array([item.remak_centroid_y, item.remak_centroid_x])
        direction = b - a
        record = dict(radial_ab_physical=None, radial_bc_physical=None,
                      radial_ac_physical=None, radial_normalized_position=None,
                      radial_status="valid", nerve_centroid_y=float(a[0]),
                      nerve_centroid_x=float(a[1]), radial_boundary_y=None,
                      radial_boundary_x=None)
        records.append(record)
        if np.linalg.norm(direction) < 1e-12:
            record["radial_status"] = "undefined_ray_A_equals_B"
            continue
        crossings = [0.0, 1.0]
        for axis, size in enumerate(mask.shape):
            if direction[axis] != 0:
                ts = (np.arange(size + 1) - 0.5 - a[axis]) / direction[axis]
                crossings.extend(ts[ts > 0].tolist())
        ts = np.unique(crossings)
        # Include an interval beyond the image, where the ray must be outside.
        ts = np.append(ts, ts[-1] + 1)
        midpoints = a + ((ts[:-1] + ts[1:]) / 2)[:, None] * direction
        cells = np.floor(midpoints + 0.5).astype(int)
        in_image = np.all((cells >= 0) & (cells < mask.shape), axis=1)
        inside = np.zeros(len(cells), dtype=bool)
        filled_inside = inside.copy()
        inside[in_image] = mask[tuple(cells[in_image].T)]
        filled_inside[in_image] = outer[tuple(cells[in_image].T)]
        b_cell = np.floor(b + 0.5).astype(int)
        b_inside = (np.all(b_cell >= 0) and np.all(b_cell < mask.shape)
                    and mask[tuple(b_cell)])
        if not b_inside or np.any(~inside[ts[:-1] < 1]):
            record["radial_status"] = "A_to_B_not_inside_nerve"
            continue
        exits = np.flatnonzero((ts[:-1] >= 1) & ~filled_inside)
        if not len(exits):
            record["radial_status"] = "no_outward_intersection"
            continue
        t = float(ts[exits[0]])
        if t <= 1:
            record["radial_status"] = "B_on_outer_boundary"
            continue
        c = a + t * direction
        ab = float(np.linalg.norm(direction * scale))
        record.update(radial_ab_physical=ab, radial_bc_physical=(t - 1) * ab,
                      radial_ac_physical=t * ab, radial_normalized_position=1 / t,
                      radial_boundary_y=float(c[0]), radial_boundary_x=float(c[1]))
    return replace(result, radial_records=tuple(records))
