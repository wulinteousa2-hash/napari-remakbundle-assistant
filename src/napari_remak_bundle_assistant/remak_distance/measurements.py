"""Scientific calculations for Remak-to-nerve-edge distance."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from math import pi, sqrt
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label as connected_components


class MeasurementValidationError(ValueError):
    """Raised when input masks cannot produce valid measurements."""


class MeasurementCancelledError(RuntimeError):
    """Raised when a running measurement is cancelled by the caller."""


ProgressCallback = Callable[[int, int, str], None]
CancellationCallback = Callable[[], bool]


@dataclass(frozen=True)
class RemakMeasurement:
    """Measurements for one uniquely labelled Remak bundle."""

    remak_id: int
    min_distance_px: float
    min_distance_physical: float
    remak_area_px: int
    remak_area_physical: float
    remak_centroid_y: float
    remak_centroid_x: float
    nearest_remak_y: float
    nearest_remak_x: float
    nearest_nerve_y: float
    nearest_nerve_x: float
    centroid_to_nerve_distance: float | None = None
    normalized_distance: float | None = None
    measurement_id: int | None = None
    component_index: int = 1

    def as_record(self) -> dict[str, Any]:
        """Return a CSV/table-friendly record."""
        return asdict(self)


@dataclass(frozen=True)
class MeasurementResult:
    """Per-object measurements plus image-level metadata."""

    measurements: tuple[RemakMeasurement, ...]
    nerve_area_px: int
    nerve_area_physical: float
    pixel_size_y: float
    pixel_size_x: float

    def as_records(self) -> list[dict[str, Any]]:
        """Return one dictionary per Remak bundle."""
        return [item.as_record() for item in self.measurements]


def _validated_inputs(
    remak_labels: np.ndarray,
    nerve_mask: np.ndarray,
    scale: tuple[float, float],
    report: ProgressCallback,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[float, float],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    labels = np.asarray(remak_labels)
    nerve_data = np.asarray(nerve_mask)

    if labels.ndim != 2 or nerve_data.ndim != 2:
        raise MeasurementValidationError(
            "Remak labels and the nerve ROI must both be 2D."
        )
    if labels.shape != nerve_data.shape:
        raise MeasurementValidationError(
            "Remak labels and the nerve ROI must have identical spatial dimensions."
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise MeasurementValidationError(
            "The Remak layer must contain integer labels (0, 1, 2, ...)."
        )
    if labels.size and int(np.min(labels)) < 0:
        raise MeasurementValidationError("Remak label values cannot be negative.")
    try:
        scale_y, scale_x = (float(scale[0]), float(scale[1]))
    except (IndexError, TypeError, ValueError) as error:
        raise MeasurementValidationError(
            "Scale must contain two positive values: (scale_y, scale_x)."
        ) from error
    if not np.isfinite((scale_y, scale_x)).all() or scale_y <= 0 or scale_x <= 0:
        raise MeasurementValidationError("Pixel scale values must be finite and positive.")

    report(0, 0, "Step 2 of 10 — Preparing binary masks …")
    nerve = np.asarray(nerve_data, dtype=bool)
    if not np.any(nerve):
        raise MeasurementValidationError("The nerve ROI is empty.")

    report(0, 0, "Step 3 of 10 — Identifying Remak bundles …")
    # Work only with foreground coordinates from here on.  Besides avoiding a
    # full-size temporary in several checks, this lets the measurement pass
    # visit all Remak pixels once instead of scanning the entire image once per
    # label (which is especially costly for large EM images).
    ys, xs = np.nonzero(labels)
    point_labels = labels[ys, xs]
    ids = np.unique(point_labels)
    if ids.size == 0:
        raise MeasurementValidationError(
            "The Remak layer contains no positive labels to measure."
        )

    outside_ids = np.unique(point_labels[~nerve[ys, xs]])
    if outside_ids.size:
        listed = ", ".join(str(int(value)) for value in outside_ids[:12])
        suffix = " ..." if outside_ids.size > 12 else ""
        raise MeasurementValidationError(
            "Remak bundles extend outside the nerve ROI: " + listed + suffix
        )

    return labels, nerve, (scale_y, scale_x), ids, ys, xs


def _outer_boundary(nerve_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a hole-filled nerve ROI and its outer, one-pixel boundary."""
    filled = ndi.binary_fill_holes(nerve_mask)
    eroded = ndi.binary_erosion(
        filled,
        structure=ndi.generate_binary_structure(2, 1),
        border_value=0,
    )
    boundary = filled & ~eroded
    if not np.any(boundary):
        raise MeasurementValidationError("The nerve ROI has no measurable boundary.")
    return filled, boundary


def _boundary_polyline(vertices: np.ndarray, shape_type: str) -> np.ndarray:
    """Return a closed polyline, sampling ellipses without rasterization."""
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise MeasurementValidationError(
            "Every nerve boundary must contain finite 2D vertices."
        )
    if shape_type == "ellipse":
        if len(points) < 4:
            raise MeasurementValidationError(
                "An ellipse nerve boundary must contain four control vertices."
            )
        center = np.mean(points[:4], axis=0)
        axis_a = (points[1] - points[0]) / 2.0
        axis_b = (points[3] - points[0]) / 2.0
        angles = np.linspace(0.0, 2.0 * np.pi, 1025)
        return (
            center
            + np.cos(angles)[:, None] * axis_a
            + np.sin(angles)[:, None] * axis_b
        )
    if shape_type not in {"polygon", "rectangle"}:
        raise MeasurementValidationError(
            "The automatic nerve boundary must be a polygon, rectangle, or ellipse."
        )
    if len(points) < 3:
        raise MeasurementValidationError(
            "The automatic nerve boundary must contain at least three vertices."
        )
    return np.vstack([points, points[0]])


def _nearest_polyline_point(
    point: np.ndarray, polylines: tuple[np.ndarray, ...]
) -> tuple[np.ndarray, float]:
    best_point: np.ndarray | None = None
    best_squared = np.inf
    for vertices in polylines:
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
    if best_point is None:
        raise MeasurementValidationError("The Shapes ROI has no measurable boundary.")
    return best_point, sqrt(best_squared)


class _UnionFind:
    """Merge component fragments that cross consecutive row chunks."""

    def __init__(self) -> None:
        self.parent = [0]
        self.rank = [0]

    def add(self) -> int:
        index = len(self.parent)
        self.parent.append(index)
        self.rank.append(0)
        return index

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def join(self, first: int, second: int) -> None:
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first == root_second:
            return
        if self.rank[root_first] < self.rank[root_second]:
            root_first, root_second = root_second, root_first
        self.parent[root_second] = root_first
        if self.rank[root_first] == self.rank[root_second]:
            self.rank[root_first] += 1


def measure_centroids_to_shapes_boundary(
    remak_labels: Any,
    boundaries: tuple[np.ndarray, ...],
    shape_types: tuple[str, ...],
    scale: tuple[float, float] = (1.0, 1.0),
    *,
    include_normalized_distance: bool = False,
    chunk_rows: int = 256,
    progress_callback: ProgressCallback | None = None,
    is_cancelled: CancellationCallback | None = None,
) -> MeasurementResult:
    """Measure label centroids to a vector Shapes boundary with bounded memory.

    The Labels source is read once in row chunks. Disconnected islands are
    retained as separate 4-connected components even when they share a label
    value. The Shapes boundary is never rasterized and no image-sized Euclidean
    distance transform is allocated.
    """

    def report(completed: int, total: int, message: str) -> None:
        if progress_callback is not None:
            progress_callback(completed, total, message)

    def check_cancelled() -> None:
        if is_cancelled is not None and is_cancelled():
            raise MeasurementCancelledError("Measurement cancelled.")

    report(0, 0, "Step 1 of 7 — Validating Labels and Shapes inputs …")
    shape = tuple(int(value) for value in remak_labels.shape)
    if len(shape) != 2 or min(shape) < 1:
        raise MeasurementValidationError(
            "The Remak Labels layer must be non-empty and 2D."
        )
    if not boundaries or len(boundaries) != len(shape_types):
        raise MeasurementValidationError("The Shapes nerve boundary is empty.")
    try:
        scale_y, scale_x = float(scale[0]), float(scale[1])
    except (IndexError, TypeError, ValueError) as error:
        raise MeasurementValidationError(
            "Scale must contain two positive values: (scale_y, scale_x)."
        ) from error
    if not np.isfinite((scale_y, scale_x)).all() or min(scale_y, scale_x) <= 0:
        raise MeasurementValidationError(
            "Pixel scale values must be finite and positive."
        )
    if chunk_rows < 1:
        raise MeasurementValidationError("Chunk height must be positive.")
    # Validate the small vector input before starting the potentially long
    # Labels scan. It remains in data coordinates until physical projection.
    data_polylines = tuple(
        _boundary_polyline(vertices, shape_type)
        for vertices, shape_type in zip(boundaries, shape_types, strict=True)
    )

    # Cap each chunk by pixel count as well as row count. On a 60,000-pixel-wide
    # layer this keeps temporary foreground indices and reduction arrays in the
    # tens of MB instead of allowing a 256-row chunk to grow beyond 100 MB.
    rows_per_chunk = min(chunk_rows, max(1, 2_000_000 // shape[1]))
    union = _UnionFind()
    fragment_labels = [0]
    fragment_counts = [0.0]
    fragment_sums_y = [0.0]
    fragment_sums_x = [0.0]
    previous_bottom_ids: np.ndarray | None = None
    previous_bottom_values: np.ndarray | None = None
    chunks = (shape[0] + rows_per_chunk - 1) // rows_per_chunk
    report(
        0,
        chunks,
        f"Step 2 of 7 — Finding connected components across "
        f"{shape[0]:,} label rows …",
    )
    for chunk_index, start_y in enumerate(range(0, shape[0], rows_per_chunk)):
        check_cancelled()
        stop_y = min(shape[0], start_y + rows_per_chunk)
        block = np.asarray(remak_labels[start_y:stop_y, :])
        if block.ndim != 2 or not np.issubdtype(block.dtype, np.integer):
            raise MeasurementValidationError(
                "The Remak layer must contain 2D integer labels."
            )
        if block.size and int(np.min(block)) < 0:
            raise MeasurementValidationError("Remak label values cannot be negative.")
        local = connected_components(block, background=0, connectivity=1)
        local_count = int(local.max())
        local_to_global = np.zeros(local_count + 1, dtype=np.int64)
        if local_count:
            flat_indices = np.flatnonzero(local)
            local_ids = local.ravel()[flat_indices]
            local_y, local_x = np.divmod(flat_indices, shape[1])
            counts = np.bincount(local_ids, minlength=local_count + 1)
            sums_y = np.bincount(
                local_ids,
                weights=local_y + start_y,
                minlength=local_count + 1,
            )
            sums_x = np.bincount(
                local_ids,
                weights=local_x,
                minlength=local_count + 1,
            )
            label_values = np.zeros(local_count + 1, dtype=block.dtype)
            np.maximum.at(
                label_values,
                local_ids,
                block.ravel()[flat_indices],
            )
            for local_id in range(1, local_count + 1):
                provisional = union.add()
                local_to_global[local_id] = provisional
                fragment_labels.append(int(label_values[local_id]))
                fragment_counts.append(float(counts[local_id]))
                fragment_sums_y.append(float(sums_y[local_id]))
                fragment_sums_x.append(float(sums_x[local_id]))

        current_top_ids = local_to_global[local[0]]
        if previous_bottom_ids is not None and previous_bottom_values is not None:
            touching = (
                (previous_bottom_ids > 0)
                & (current_top_ids > 0)
                & (previous_bottom_values == block[0])
            )
            if np.any(touching):
                pairs = np.unique(
                    np.column_stack(
                        (previous_bottom_ids[touching], current_top_ids[touching])
                    ),
                    axis=0,
                )
                for first, second in pairs:
                    union.join(int(first), int(second))
        previous_bottom_ids = local_to_global[local[-1]].copy()
        previous_bottom_values = block[-1].copy()
        report(
            chunk_index + 1,
            chunks,
            f"Step 2 of 7 — Indexed components through row {stop_y:,} "
            f"of {shape[0]:,} …",
        )

    if len(fragment_labels) == 1:
        raise MeasurementValidationError(
            "The Remak layer contains no positive labels to measure."
        )
    check_cancelled()
    totals: dict[int, list[float]] = {}
    for provisional in range(1, len(fragment_labels)):
        root = union.find(provisional)
        aggregate = totals.setdefault(
            root,
            [float(fragment_labels[provisional]), 0.0, 0.0, 0.0],
        )
        aggregate[1] += fragment_counts[provisional]
        aggregate[2] += fragment_sums_y[provisional]
        aggregate[3] += fragment_sums_x[provisional]
    report(
        0,
        0,
        f"Step 3 of 7 — Calculating {len(totals)} component centroids …",
    )

    sampling = np.asarray([scale_y, scale_x], dtype=float)
    report(0, 0, "Step 4 of 7 — Preparing the vector nerve boundary …")
    physical_polylines = tuple(polyline * sampling for polyline in data_polylines)
    nerve_area_physical = float(
        sum(
            abs(
                np.sum(
                    polyline[:-1, 1] * polyline[1:, 0]
                    - polyline[1:, 1] * polyline[:-1, 0]
                )
            )
            / 2.0
            for polyline in physical_polylines
        )
    )
    equivalent_radius = (
        sqrt(nerve_area_physical / pi) if nerve_area_physical > 0 else None
    )

    measurements: list[RemakMeasurement] = []
    ordered_components = sorted(
        totals.items(),
        key=lambda item: (
            int(item[1][0]),
            item[1][2] / item[1][1],
            item[1][3] / item[1][1],
        ),
    )
    component_number_by_label: dict[int, int] = {}
    total_components = len(ordered_components)
    for index, (_root, values) in enumerate(ordered_components):
        check_cancelled()
        remak_id = int(values[0])
        count, sum_y, sum_x = values[1:]
        component_index = component_number_by_label.get(remak_id, 0) + 1
        component_number_by_label[remak_id] = component_index
        centroid = np.asarray([sum_y / count, sum_x / count], dtype=float)
        centroid_physical = centroid * sampling
        nearest_physical, distance_physical = _nearest_polyline_point(
            centroid_physical, physical_polylines
        )
        nearest_data = nearest_physical / sampling
        distance_px = float(np.linalg.norm(nearest_data - centroid))
        normalized = (
            distance_physical / equivalent_radius
            if include_normalized_distance and equivalent_radius
            else None
        )
        area_px = int(count)
        measurements.append(
            RemakMeasurement(
                remak_id=remak_id,
                min_distance_px=distance_px,
                min_distance_physical=distance_physical,
                remak_area_px=area_px,
                remak_area_physical=float(area_px * scale_y * scale_x),
                remak_centroid_y=float(centroid[0]),
                remak_centroid_x=float(centroid[1]),
                nearest_remak_y=float(centroid[0]),
                nearest_remak_x=float(centroid[1]),
                nearest_nerve_y=float(nearest_data[0]),
                nearest_nerve_x=float(nearest_data[1]),
                centroid_to_nerve_distance=distance_physical,
                normalized_distance=normalized,
                measurement_id=index + 1,
                component_index=component_index,
            )
        )
        report(
            index + 1,
            total_components,
            f"Step 5 of 7 — Projecting centroid {index + 1} of "
            f"{total_components} (label {remak_id}, component "
            f"{component_index}) …",
        )

    nerve_area_px = int(round(nerve_area_physical / (scale_y * scale_x)))
    report(0, 0, "Step 6 of 7 — Finalizing centroid measurements …")
    return MeasurementResult(
        measurements=tuple(measurements),
        nerve_area_px=nerve_area_px,
        nerve_area_physical=nerve_area_physical,
        pixel_size_y=scale_y,
        pixel_size_x=scale_x,
    )


def measure_remak_to_nerve_distance(
    remak_labels: np.ndarray,
    nerve_mask: np.ndarray,
    scale: tuple[float, float] = (1.0, 1.0),
    *,
    include_centroid_distance: bool = False,
    include_normalized_distance: bool = False,
    progress_callback: ProgressCallback | None = None,
    is_cancelled: CancellationCallback | None = None,
) -> MeasurementResult:
    """Measure minimum bundle-edge-to-outer-nerve-edge distances.

    Distances are measured between pixel centers. The nerve ROI is hole-filled
    before its outer boundary is extracted, so internal holes are not mistaken
    for the nerve perimeter. A bundle occupying a boundary pixel has distance
    zero. ``sampling=scale`` makes the primary EDT anisotropy-aware.
    """
    def report(completed: int, total: int, message: str) -> None:
        if progress_callback is not None:
            progress_callback(completed, total, message)

    def check_cancelled() -> None:
        if is_cancelled is not None and is_cancelled():
            raise MeasurementCancelledError("Measurement cancelled.")

    report(0, 0, "Step 1 of 10 — Validating selected layers and calibration …")
    labels, nerve, sampling, ids, ys, xs = _validated_inputs(
        remak_labels, nerve_mask, scale, report
    )
    check_cancelled()
    report(
        0,
        0,
        f"Step 4 of 10 — Building the outer nerve boundary "
        f"({ids.size} bundles found) …",
    )
    filled_nerve, boundary = _outer_boundary(nerve)

    nerve_area_px = int(np.count_nonzero(filled_nerve))
    pixel_area = sampling[0] * sampling[1]
    nerve_area_physical = float(nerve_area_px * pixel_area)
    equivalent_radius = sqrt(nerve_area_physical / pi)

    # Assign every foreground pixel to its compact, sorted label index.  All
    # later reductions are over these foreground arrays, never over the full
    # image once per object.
    group = np.searchsorted(ids, labels[ys, xs])
    count = np.bincount(group, minlength=ids.size)
    centroid_y_values = np.bincount(group, weights=ys, minlength=ids.size) / count
    centroid_x_values = np.bincount(group, weights=xs, minlength=ids.size) / count
    order = np.argsort(group, kind="stable")
    offsets = np.concatenate(([0], np.cumsum(count)))

    check_cancelled()
    report(0, 0, "Step 5 of 10 — Computing physical-distance transform …")
    nearest_indices = ndi.distance_transform_edt(
        ~boundary,
        sampling=sampling,
        return_distances=False,
        return_indices=True,
    )
    check_cancelled()

    nearest_y_values = nearest_indices[0, ys, xs]
    nearest_x_values = nearest_indices[1, ys, xs]
    physical_values = np.hypot(
        (ys - nearest_y_values) * sampling[0],
        (xs - nearest_x_values) * sampling[1],
    )
    centroid_distances: np.ndarray | None = None
    if include_centroid_distance:
        # Reproduce bilinear interpolation of the EDT at each centroid without
        # retaining a second full-image float64 distance array.  Distances at
        # the four neighboring pixels are derived directly from the nearest
        # boundary indices already in memory.
        y0 = np.floor(centroid_y_values).astype(np.intp)
        x0 = np.floor(centroid_x_values).astype(np.intp)
        y1 = np.minimum(y0 + 1, labels.shape[0] - 1)
        x1 = np.minimum(x0 + 1, labels.shape[1] - 1)
        weight_y = centroid_y_values - y0
        weight_x = centroid_x_values - x0

        def distances_at(query_y: np.ndarray, query_x: np.ndarray) -> np.ndarray:
            nearest_y = nearest_indices[0, query_y, query_x]
            nearest_x = nearest_indices[1, query_y, query_x]
            return np.hypot(
                (query_y - nearest_y) * sampling[0],
                (query_x - nearest_x) * sampling[1],
            )

        top = (
            distances_at(y0, x0) * (1.0 - weight_x)
            + distances_at(y0, x1) * weight_x
        )
        bottom = (
            distances_at(y1, x0) * (1.0 - weight_x)
            + distances_at(y1, x1) * weight_x
        )
        centroid_distances = top * (1.0 - weight_y) + bottom * weight_y
    del nearest_indices

    report(0, 0, "Step 6 of 10 — Computing distances in image pixels …")
    if np.isclose(sampling[0], sampling[1]):
        pixel_values = None
    else:
        nearest_pixel_indices = ndi.distance_transform_edt(
            ~boundary,
            return_distances=False,
            return_indices=True,
        )
        check_cancelled()
        pixel_values = np.hypot(
            ys - nearest_pixel_indices[0, ys, xs],
            xs - nearest_pixel_indices[1, ys, xs],
        )
        del nearest_pixel_indices

    selected_indices = np.empty(ids.size, dtype=np.intp)
    min_physical_values = np.empty(ids.size, dtype=float)
    min_pixel_values = np.empty(ids.size, dtype=float)
    total = int(ids.size)
    report(0, total, f"Step 7 of 10 — Measuring bundles: 0 of {total} …")
    for index, raw_id in enumerate(ids):
        check_cancelled()
        candidates = order[offsets[index] : offsets[index + 1]]
        selected = candidates[int(np.argmin(physical_values[candidates]))]
        selected_indices[index] = selected
        min_physical_values[index] = physical_values[selected]
        if pixel_values is None:
            min_pixel_values[index] = physical_values[selected] / sampling[0]
        else:
            min_pixel_values[index] = np.min(pixel_values[candidates])
        report(
            index + 1,
            total,
            f"Step 7 of 10 — Measuring bundle {index + 1} of {total} "
            f"(ID {int(raw_id)}) …",
        )
    del physical_values, pixel_values

    measurements: list[RemakMeasurement] = []
    report(0, 0, "Step 8 of 10 — Finalizing optional and normalized metrics …")
    for index, raw_id in enumerate(ids):
        remak_id = int(raw_id)
        selected = int(selected_indices[index])
        remak_y = int(ys[selected])
        remak_x = int(xs[selected])
        nerve_y = int(nearest_y_values[selected])
        nerve_x = int(nearest_x_values[selected])
        min_physical = float(min_physical_values[index])
        min_px = float(min_pixel_values[index])
        centroid_y = float(centroid_y_values[index])
        centroid_x = float(centroid_x_values[index])
        centroid_distance = (
            None
            if centroid_distances is None
            else float(centroid_distances[index])
        )

        normalized: float | None = None
        if include_normalized_distance:
            normalized = min_physical / equivalent_radius

        area_px = int(count[index])
        measurements.append(
            RemakMeasurement(
                remak_id=remak_id,
                min_distance_px=min_px,
                min_distance_physical=min_physical,
                remak_area_px=area_px,
                remak_area_physical=float(area_px * pixel_area),
                remak_centroid_y=centroid_y,
                remak_centroid_x=centroid_x,
                nearest_remak_y=remak_y,
                nearest_remak_x=remak_x,
                nearest_nerve_y=nerve_y,
                nearest_nerve_x=nerve_x,
                centroid_to_nerve_distance=centroid_distance,
                normalized_distance=normalized,
            )
        )
    return MeasurementResult(
        measurements=tuple(measurements),
        nerve_area_px=nerve_area_px,
        nerve_area_physical=nerve_area_physical,
        pixel_size_y=sampling[0],
        pixel_size_x=sampling[1],
    )
