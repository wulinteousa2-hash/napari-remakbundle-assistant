"""Descriptive per-image charts from completed measurements only."""

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QVBoxLayout, QWidget,
)


def describe(values, unit):
    """Describe observations without assuming a population distribution."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return "No measurements."
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return (f"n = {len(values)}; median {median:.4g} {unit}; "
            f"Q1–Q3 {q1:.4g}–{q3:.4g} {unit} (IQR {q3-q1:.4g}); "
            f"range {values.min():.4g}–{values.max():.4g} {unit}.")


class AnalysisWidget(QWidget):
    """Keep a result snapshot; allocate the plotting canvas only on demand."""

    def __init__(self):
        super().__init__()
        self.figure = self.canvas = None
        self.distances = np.array([])
        self.labels = []
        self.radial = []
        self.unit = ""
        self.title = ""
        self.dirty = True
        self.layout = QVBoxLayout(self)
        self.summary = QLabel("Complete an automatic measurement or finish a manual session.")
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.TextFormat.PlainText)
        self.layout.addWidget(self.summary)
        controls = QHBoxLayout()
        self.chart = QComboBox()
        self.chart.currentIndexChanged.connect(self.render)
        controls.addWidget(self.chart)
        self.save_button = QPushButton("Save chart")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_chart)
        controls.addWidget(self.save_button)
        self.layout.addLayout(controls)
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.layout.addWidget(self.note)
        self.layout.addStretch(1)

    def set_results(self, distances=(), labels=(), radial=(), *, unit="", title="",
                    manual=False):
        self.distances = np.asarray(distances, dtype=float).copy()
        self.labels, self.radial = list(labels), list(radial)
        self.unit, self.title = unit, title
        self.dirty = True
        self.note.clear()
        text = f"{title}\n{describe(self.distances, unit)}"
        if manual:
            text += "\nManual point-pair observations; these may be a selected subset of bundles."
        else:
            text += "\nAutomatic shortest centroid-to-boundary distances; each connected component is one observation."
        valid = [r for r in self.radial if r["radial_status"] == "valid"]
        if self.radial:
            text += (f"\nRadial: {len(valid)} valid, {len(self.radial)-len(valid)} invalid "
                     "(excluded from radial charts).")
        if valid:
            text += "\nNormalized radial position: " + describe(
                [r["radial_normalized_position"] for r in valid], "")
            text += " Center = 0; boundary = 1."
        self.summary.setText(text if len(self.distances) else "No completed measurements.")
        previous = self.chart.currentText()
        self.chart.blockSignals(True)
        self.chart.clear()
        if len(self.distances):
            self.chart.addItems(["Distance distribution", "Distance by bundle"])
            if valid:
                self.chart.addItems(["Radial position distribution", "Shortest versus radial distance"])
        index = self.chart.findText(previous)
        self.chart.setCurrentIndex(max(0, index))
        self.chart.blockSignals(False)
        self.save_button.setEnabled(False)
        if self.figure is not None:
            self.figure.clear()
            self.canvas.draw_idle()
        if self.isVisible():
            self.render()

    def render(self, *_):
        if not len(self.distances):
            return
        if self.figure is None:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            self.figure = Figure(figsize=(7, 5), constrained_layout=True)
            self.canvas = FigureCanvasQTAgg(self.figure)
            self.canvas.setMinimumHeight(300)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(self.canvas)
            scroll.setMinimumHeight(300)
            self.layout.insertWidget(3, scroll, 1)
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        kind = self.chart.currentText()
        height = (max(350, min(60, len(self.distances)) * 15 + 120)
                  if kind == "Distance by bundle" else 350)
        self.canvas.setMinimumHeight(height)
        self.figure.set_size_inches(7, height / 100)
        note = "Descriptive results for this image; no biological significance test is implied."
        if kind == "Distance distribution":
            ax.hist(self.distances, bins=min(40, max(1, int(np.sqrt(len(self.distances))))),
                    color="#258baf", edgecolor="white")
            # Bound artist size for very large component sets.
            points = np.sort(self.distances)
            points = points[np.linspace(0, len(points)-1, min(2000, len(points))).astype(int)]
            ax.plot(points, np.zeros(len(points)), "|", color="#17324d", alpha=.5)
            ax.axvline(np.median(self.distances), color="#d46724", label="Median")
            ax.legend()
            ax.set(xlabel=f"Distance ({self.unit})", ylabel="Observations")
            if len(self.distances) > 2000:
                note += " Tick marks show 2,000 ordered observations; histogram uses all results."
        elif kind == "Distance by bundle":
            order = np.argsort(self.distances)[-60:]
            ax.barh(np.arange(len(order)), self.distances[order], color="#258baf")
            ax.set_yticks(np.arange(len(order)), [self.labels[i] for i in order], fontsize=7)
            ax.set(xlabel=f"Distance ({self.unit})", ylabel="Bundle / measurement")
            if len(self.distances) > 60:
                note += " Showing the 60 longest distances; summary and distribution use all results."
        elif kind == "Radial position distribution":
            positions = [r["radial_normalized_position"] for r in self.radial
                         if r["radial_status"] == "valid"]
            ax.hist(positions, bins=np.linspace(0, 1, 11), color="#b952a0", edgecolor="white")
            ax.set(xlim=(0, 1), xlabel="AB / AC₂ (0 = center, 1 = boundary)", ylabel="Bundles")
        else:
            indices = [i for i, r in enumerate(self.radial) if r["radial_status"] == "valid"]
            x = self.distances[indices]
            y = np.array([self.radial[i]["radial_bc_physical"] for i in indices])
            ax.scatter(x, y, s=16, alpha=.55, rasterized=True, color="#b952a0")
            upper = max(float(x.max()), float(y.max()), 1e-9)
            ax.plot([0, upper], [0, upper], "--", color="gray", label="Equal distances")
            ax.legend()
            ax.set(xlabel=f"Shortest BC₁ ({self.unit})", ylabel=f"Radial BC₂ ({self.unit})")
            note += " Differences reflect direction and may also reflect different boundary sources."
        ax.set_title(f"{self.title}\n{kind}", fontsize=10)
        self.note.setText(note)
        self.canvas.draw_idle()
        self.save_button.setEnabled(True)
        self.dirty = False

    def save_chart(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save analysis chart", "remak_analysis.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )
        if not path or self.figure is None:
            return
        from pathlib import Path
        if Path(path).suffix.lower() not in {".png", ".pdf", ".svg"}:
            path += ".png"
        try:
            self.figure.savefig(path, dpi=200)
        except OSError as error:
            self.note.setText(f"Could not save chart: {error}")
        else:
            self.note.setText(f"Saved chart to {path}")
