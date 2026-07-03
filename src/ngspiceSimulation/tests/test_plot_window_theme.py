"""Regression tests for the apply_theme re-entrancy guard.

setStyleSheet() synchronously dispatches a QEvent.PaletteChange. Before the
guard, changeEvent re-called apply_theme on that event, which called
setStyleSheet again — an unbounded recursion that blew Python's recursion
limit (surfacing as a RecursionError deep in a canvas sizeHint/numpy
reduction) or, with cheaper frames, spun long enough to freeze the GUI.
Constructing the widget at all exercises the loop, since __init__ calls
apply_theme.
"""
import os

import numpy as np
import pytest

from PyQt6.QtCore import QEvent

from ngspiceSimulation.plot_window import plotWindow


def _write_tran_project(d: str, npts: int = 32, nnodes: int = 2) -> None:
    with open(os.path.join(d, "analysis"), "w") as f:
        f.write(".tran 1e-6 1e-3 0\n")
    t = np.linspace(0, 1e-3, npts)
    cols = [np.sin(2 * np.pi * (50 + 10 * k) * t) for k in range(nnodes)]
    names = [f"net{k}" for k in range(nnodes)]
    with open(os.path.join(d, "plot_data_v.txt"), "w") as f:
        f.write("Index   time   " + "   ".join(names) + "\n")
        f.write("-" * 40 + "\n")
        M = np.column_stack([np.arange(npts), t] + cols)
        np.savetxt(f, M, fmt=["%d"] + ["%.9g"] * (nnodes + 1), delimiter="\t")
    open(os.path.join(d, "plot_data_i.txt"), "w").close()


@pytest.fixture
def plot_window(qapp, tmp_path):
    _write_tran_project(str(tmp_path))
    w = plotWindow(str(tmp_path), "proj")
    yield w
    w.close()


def test_construction_does_not_recurse(plot_window):
    # Reaching here means __init__ -> apply_theme -> setStyleSheet ->
    # PaletteChange -> ... did not recurse into a RecursionError.
    assert plot_window._applying_theme is False


def test_palette_change_reapplies_theme_once(qapp, plot_window, monkeypatch):
    """A single PaletteChange must run the theme body exactly once.

    Without the guard the self-induced PaletteChange from setStyleSheet would
    re-enter the body cascade-style; the guard collapses it to one pass.
    """
    calls = {"n": 0}
    orig = plotWindow._apply_theme_impl

    def counting_impl(self):
        calls["n"] += 1
        return orig(self)

    monkeypatch.setattr(plotWindow, "_apply_theme_impl", counting_impl)
    qapp.sendEvent(plot_window, QEvent(QEvent.Type.PaletteChange))
    assert calls["n"] == 1
    assert plot_window._applying_theme is False
