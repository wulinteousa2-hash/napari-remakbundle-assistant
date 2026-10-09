# Changelog

## 1.0.1 — 2026-10-09

### Added

- Independent **Radial Distance Analysis**, reusing existing Remak centroids B
  and calculating nerve centroid A and the outward ray intersection C₂.
- Calibrated AB, BC₂, AC₂, and normalized radial position AB/AC₂, with additional
  CSV fields for measurements, coordinates, and geometry status.
- Shapes and Labels nerve-region inputs. The selected Shapes boundary is reused
  by default, without painting a mask or creating an intermediate Labels layer.
- Separate radial Shapes visualization: AB in orange and BC₂ in magenta.
- Validation of the A–B interior path and first outward crossing beyond B for
  irregular boundaries. Invalid geometry is flagged with blank radial distances.
- **Analysis** tab after automatic measurement or a finished manual session:
  count, median, quartiles, IQR, range, distance distribution, bundle bars,
  radial position distribution, and shortest-versus-radial distance scatterplot.
- PNG, PDF, and SVG chart export. Charts use completed result rows and are
  created on demand; bar charts show the 60 longest distances for large tables.

### Changed

- Made the entire plugin panel scrollable, with separate scrolling for long
  bar charts to keep results accessible in smaller windows.
- Added Matplotlib as an explicit dependency for the Analysis tab.
- Updated package version metadata to 1.0.1.

### Compatibility and validation

- Preserved existing shortest B–C₁ calculations, existing normalized-distance
  semantics, manual point-pair measurements, and existing CSV column names.
- Shapes radial calculations use vector geometry directly; they do not allocate
  an image-sized nerve mask. Labels radial calculations use pixel-cell edges.
- All 40 tests passed, covering legacy measurements, radial geometry and
  calibration, chart statistics and export, GUI workflows, and panel scrolling.

## 1.0.0

- Initial release with automatic centroid-to-boundary measurements, manual
  point-pair measurement, physical calibration, visual QC, and CSV export.
