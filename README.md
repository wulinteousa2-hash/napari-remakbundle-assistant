# napari-remak-bundle-assistant

**Version 1.0.1** · [Changelog](CHANGELOG.md)

Research-oriented napari tools for measuring the distance from each curated
Remak bundle's **centroid** to the nearest point on the vector outer boundary
of a peripheral nerve cross-section.

Optional radial analysis measures along the nerve-center-to-bundle direction.
The **Analysis** tab provides per-image summaries and charts, with PNG, PDF,
and SVG export. The whole plugin panel and long bar charts are scrollable.

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

## How the automatic measurement works

The automatic measurement uses the geometric center of each segmented Remak
bundle and the nearest location on the traced outer nerve boundary:

```text
segmented Remak bundle                         outer nerve boundary
      █████                                               /
    ███ ● ███  ───────── shortest distance ────────────  ×
      █████                                             /
          centroid                         nearest boundary point
```

1. **Identify each bundle.** Pixels with the same label that share an edge are
   treated as one connected component. Separate regions are measured
   independently, even if they happen to have the same label value.
2. **Calculate the centroid.** Every pixel in the component is given equal
   weight. The software averages all pixel-center Y coordinates and all
   pixel-center X coordinates. The resulting point is the bundle's centroid,
   or geometric center. For a strongly curved or concave bundle, this point
   can fall outside the labeled pixels; this is a normal property of a
   centroid.
3. **Represent the nerve outline as a vector boundary.** Polygon and rectangle
   vertices are joined by straight segments. An ellipse is represented by a
   finely sampled closed curve. The outline remains continuous vector geometry
   rather than being converted into a pixel-thick boundary.
4. **Find the nearest boundary point.** The centroid is projected onto every
   boundary segment. A projection that would fall beyond a segment is limited
   to that segment's endpoint. The candidate with the smallest distance is
   selected as the nearest point on the nerve boundary.
5. **Apply the image calibration.** Pixel size in Y and X is applied before
   comparing distances. This makes the result physically correct when pixels
   are not square. The result table reports the calibrated distance, and the
   optional QC line displays the exact centroid-to-boundary measurement.

This is a **centroid-to-boundary** measurement, not a measurement from the
edge of the Remak bundle. The same rule is applied consistently to every
component, making results reproducible across an image set.

## Workflow

1. Start napari and open **Plugins → napari-remak-bundle-assistant →
   Remak–Nerve Distance**.
2. Choose the reference image, Remak bundle labels, and nerve ROI.
3. Confirm or edit **Pixel size Y**, **Pixel size X**, and **Unit**. These are
   initialized from the Labels layer but do not modify that source layer.
4. Select optional normalized measurements, radial analysis, and visual outputs.
5. Click **Run automatic measurement**.
6. Sort the **Measurements** table or double-click a row to locate a bundle.
7. Open **Analysis** to see summaries and charts; use **Save chart** to export.
8. Click **Export CSV** for downstream analysis in Python, R, Prism, or Excel.

Scroll the plugin panel to reach controls and results in a short window. Long
bar charts have their own scrollbar so all displayed bundle labels remain accessible.

### Per-image analysis

After **Run automatic measurement** completes or **Finish manual session** is
clicked, open the **Analysis** tab beside **Measurements**. It shows the number
of observations, median, quartiles, interquartile range, and minimum–maximum.
Choose a distance histogram or a sorted bar chart identifying each bundle and
measurement. With valid radial results, normalized radial position and
shortest-versus-radial distance charts are also available. Invalid radial rows
are counted and excluded from radial charts; their existing shortest distances
remain in the distance summary.

Charts use the completed result rows, without reading image pixels or creating
napari layers. The plotting canvas is created when Analysis is first opened.
For large tables the bar chart displays the 60 longest distances; statistics
and histogram counts include all measurements. Histogram tick marks are limited
to 2,000 ordered observations. Manual observations are identified as a possible
selected subset, and summaries make no claims about biological significance.
**Save chart** exports the current plot as PNG, PDF, or SVG. New measurements
replace the previous analysis; clearing results clears the analysis too.

### Optional Radial Distance Analysis

Enable **Radial Distance Analysis**. By default it reuses the selected
**Nerve boundary (Shapes)**; no painting or extra mask is needed. The
**Radial nerve region (Shapes / Labels)** selector also accepts an explicit
Shapes layer or a nerve segmentation Labels layer. For Shapes, A is the
centroid of the enclosed area, and C₂ lies on the vector boundary. One closed
nerve shape is required; ellipses use the existing sampled vector outline.
For Labels, the layer must share the Remak grid and transform, and all positive
pixels define the nerve. The existing shortest B–C₁ analysis stays unchanged.

A is the nerve region centroid; B is reused from the existing bundle measurement.
C₂ is the first outward intersection beyond B on the straight A→B ray. The
independent radial Shapes layer shows AB in orange and BC₂ in magenta. Distances
use the existing Y/X calibration; enter pixel sizes in µm and set Unit to µm
for micrometer output. The table and CSV add AB, BC₂, AC₂, and AB/AC₂. CSV names
are `radial_ab_physical`, `radial_bc_physical`, `radial_ac_physical`, and
`radial_normalized_position`, with centroid/intersection coordinates and
`radial_status`. Existing columns retain their names and meanings.

Normalized radial position is **AB/AC₂**: 0 represents the nerve center and 1
represents the boundary. The complementary ratio BC₂/AC₂ equals 1 − AB/AC₂.
This radial position differs from the existing optional normalized shortest
distance, which is preserved unchanged.

With a Shapes source, A is calculated from polygon area, not by averaging
boundary vertices. C₂ is found by ray–segment intersection, not nearest-point
search. Radial calculations run in the background directly on the vector
boundary, without creating a Labels layer or allocating a full-image nerve
mask. Their cost depends on boundary vertices and bundle count. The Remak
Labels source is still scanned in row chunks to obtain B.

For Labels inputs, the mask boundary follows pixel-cell edges, half a pixel
from pixel centers.
Every grid-crossing interval is checked: A–B must stay inside the actual nerve
mask. Internal holes are filled only when finding the outer boundary C₂.
Concave boundaries use the first exit beyond B, even if the ray later reenters.
Invalid paths, centroids outside the nerve, and coincident A/B are flagged;
their radial distances are blank and no radial lines are drawn. This mask
boundary can differ slightly from the independently traced Shapes boundary C₁.

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

## Technical measurement definition

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
git clone https://github.com/wulinteousa2-hash/napari-remakbundle-assistant.git
cd napari-remakbundle-assistant
python -m pip install -e '.[test]'
pytest
```

## Package structure

```text
src/napari_remak_bundle_assistant/remak_distance/
├── measurements.py   # napari-independent scientific engine
├── radial.py         # independent mask and vector radial calculations
├── analysis.py       # descriptive summaries, charts, figure export
└── widget.py         # scrollable GUI, table, QC layers, CSV export
tests/
├── test_measurements.py
├── test_radial.py
├── test_analysis.py
└── test_widget.py
```

## License

MIT
