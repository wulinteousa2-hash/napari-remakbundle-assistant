# napari-remak-bundle-assistant

Research-oriented napari tools for measuring the distance from each curated
Remak bundle's **centroid** to the nearest point on the vector outer boundary
of a peripheral nerve cross-section.

The plugin supports both automatic component-centroid analysis and a
lightweight manual workflow. Automatic analysis gives every disconnected
4-connected region its own row, including regions that share a source label
value. Manual measurement reads only the curated label beneath the user's
first click and records a second click on the Shapes nerve boundary; it does
not rasterize or scan the full image.

## Inputs

Open three layers in napari:

1. a reference EM `Image` layer (optional for measurement, used for context
   and CSV metadata),
2. a curated Remak `Labels` layer in which `0` is background; disconnected
   regions are measured separately even when they share a positive label ID,
   and
3. a closed polygon, rectangle, or ellipse `Shapes` layer tracing the outer
   nerve boundary.

The Shapes ROI is transformed from its own coordinates through world space to
the Remak Labels grid, so matching layer scale and translation manually is not
required. It remains vector geometry: automatic measurement does not rasterize
it. The plugin never modifies the source image or segmentation layers.

## Workflow

1. Start napari and open **Plugins → napari-remak-bundle-assistant →
   Remak–Nerve Distance**.
2. Choose the reference image, Remak bundle labels, and nerve ROI.
3. Confirm or edit **Pixel size Y**, **Pixel size X**, and **Unit**. These are
   initialized from the Labels layer but do not modify that source layer.
4. Select optional normalized measurements and visual QC outputs.
5. Click **Run automatic measurement**.
6. Sort the results by any table column. Double-click a row to center that
   bundle in the viewer.
7. Click **Export CSV** for downstream analysis in Python, R, Prism, or Excel.

Use **Apply / update results** after changing the calibration. Existing
automatic results are recalculated because anisotropic pixel sizes can change
which boundary point is physically closest. Existing manual physical distances
are updated immediately. Pixel sizes and the entered unit are included in CSV
exports.

The QC line for each bundle joins its calculated centroid to the closest point
on the Shapes boundary. Optional endpoint markers identify those coordinates.

### Manual label-to-boundary workflow

1. Select the curated Remak `Labels` layer and a `Shapes` nerve boundary.
2. Click **Start manual point pairs**.
3. Click inside a positive Remak label. The label ID is read from that single
   pixel.
4. Click near the desired location on the visible outer Shapes boundary. The
   endpoint snaps to the closest polygon, rectangle, or ellipse boundary.
5. Repeat the two clicks for other bundles, or use **Undo last pair**.
6. Finish the session and export the table to CSV.

Each pair produces a cyan QC line and a table row containing the Remak ID,
pixel and calibrated physical distance, and both endpoint coordinates. The
reference image is not an analysis input in this mode.

## Measurement definition

For a Remak bundle \(R\), its centroid \(c_R\), and the vector outer nerve
boundary \(\partial N\):

\[
d=\min_{n\in\partial N}\|c_R-n\|
\]

The Labels layer is scanned once in small row chunks. Four-connected regions
are tracked across chunk seams, and each disconnected component receives its
own measurement row, centroid, and QC line. The original label value and a
within-label component number remain in the table and CSV. Each centroid is
projected onto the nearest vector boundary segment. No full-size binary mask or
Euclidean distance map is allocated. Projection is performed in calibrated
physical coordinates, so anisotropic napari scale is respected. Pixel and area
measurements are also reported. The optional normalized metric is:

\[
d_{norm}=d/\sqrt{A_{nerve}/\pi}
\]

where nerve area and distance use the same physical calibration.

## Validation and reproducibility

Analysis stops with a clear message when:

- either input is not 2D,
- Remak data are not integer labels,
- the Shapes boundary or Remak layer is empty, or
- the boundary is not a supported closed shape.

CSV rows include the image name, nerve area, pixel sizes, physical unit, total
bundle count, object ID, distances, areas, centroid, and both nearest-point
coordinates.

## Installation for development

```bash
cd /home/wteox/Projects/napari/napari-remakbundle-assistant
python -m pip install -e '.[test]'
pytest
```

## Package structure

```text
src/napari_remak_bundle_assistant/remak_distance/
├── measurements.py   # napari-independent scientific engine
└── widget.py         # layer selection, table, QC layers, CSV export
tests/
└── test_measurements.py
```

## License

MIT
