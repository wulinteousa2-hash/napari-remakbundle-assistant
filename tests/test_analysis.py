"""Analysis summaries, plotting, exclusions, and exported figures."""

import numpy as np
import pytest
from qtpy.QtWidgets import QFileDialog

from napari_remak_bundle_assistant.remak_distance.analysis import AnalysisWidget, describe


def test_summary_statistics():
    text = describe([1, 2, 3, 4, 5], "µm")
    assert "median 3 µm" in text
    assert "Q1–Q3 2–4 µm (IQR 2)" in text
    assert "range 1–5 µm" in text
    assert describe([], "µm") == "No measurements."


def test_lazy_charts_radial_exclusion_and_export(qtbot, monkeypatch, tmp_path):
    widget = AnalysisWidget()
    qtbot.addWidget(widget)
    radial = [
        dict(radial_status="valid", radial_normalized_position=.8, radial_bc_physical=4),
        dict(radial_status="A_to_B_not_inside_nerve", radial_normalized_position=None,
             radial_bc_physical=None),
    ]
    widget.set_results([2, 6], ["7.1 (#1)", "7.2 (#2)"], radial, unit="µm", title="Image A")
    assert widget.figure is None  # Measurement completion does not allocate a plot.
    assert "1 valid, 1 invalid" in widget.summary.text()
    assert "median 4 µm" in widget.summary.text()
    widget.render()
    assert widget.save_button.isEnabled()
    assert widget.figure.axes[0].get_xlabel() == "Distance (µm)"
    widget.chart.setCurrentText("Distance by bundle")
    assert len(widget.figure.axes[0].patches) == 2
    widget.chart.setCurrentText("Radial position distribution")
    assert sum(p.get_height() for p in widget.figure.axes[0].patches) == 1
    widget.chart.setCurrentText("Shortest versus radial distance")
    offsets = widget.figure.axes[0].collections[0].get_offsets()
    np.testing.assert_allclose(offsets, [[2, 4]])
    path = tmp_path / "chart.png"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), ""))
    widget.save_chart()
    assert path.read_bytes().startswith(b"\x89PNG")
    widget.set_results()
    assert widget.chart.count() == 0
    assert not widget.save_button.isEnabled()
    assert widget.figure.axes == []


def test_manual_and_large_bar_chart(qtbot):
    widget = AnalysisWidget()
    qtbot.addWidget(widget)
    widget.set_results(np.arange(100), [str(i) for i in range(100)],
                       unit="nm", title="Manual image", manual=True)
    assert "selected subset" in widget.summary.text()
    assert widget.chart.count() == 2
    widget.chart.setCurrentText("Distance by bundle")
    assert len(widget.figure.axes[0].patches) == 60
    assert "60 longest" in widget.note.text()
    assert "n = 100" in widget.summary.text()


@pytest.mark.parametrize("values", [[0], [3, 3, 3]])
def test_single_or_constant_distribution(qtbot, values):
    widget = AnalysisWidget()
    qtbot.addWidget(widget)
    widget.set_results(values, [str(i) for i in range(len(values))], unit="µm")
    widget.render()
    assert sum(p.get_height() for p in widget.figure.axes[0].patches) == len(values)
