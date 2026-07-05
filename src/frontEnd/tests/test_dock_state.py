"""Regression tests for area-02 F9/F10.

F9(a): the dock registry + naming counter are per-instance state, not module
globals -- two DockAreas no longer share one ``dock`` dict / ``count``.

F10: a tab is matched to its dock by an *exact* windowTitle, so closing
``Simulation-RLC-2`` can never take out ``Simulation-RLC-21`` (the old
``startswith`` prefix match) and tab text is never elided.
"""
from PyQt6 import QtCore, QtWidgets

from frontEnd.DockArea import DockArea


class _RecordingDock(QtWidgets.QDockWidget):
    def __init__(self, title):
        super().__init__(title)
        self.setObjectName(title)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.close_events = 0

    def closeEvent(self, event):
        self.close_events += 1
        super().closeEvent(event)


def test_dock_registry_and_counter_are_per_instance(qapp):
    a = DockArea()
    b = DockArea()
    try:
        # Independent registries -- each seeded only with its own Welcome dock.
        assert a._docks is not b._docks
        assert 'Welcome' in a._docks and 'Welcome' in b._docks
        assert a._docks['Welcome'] is not b._docks['Welcome']

        # Mutating one instance's counter must not touch the other's.
        a._count = 99
        assert b._count == 1
    finally:
        a.deleteLater()
        b.deleteLater()
        qapp.processEvents()


def test_tab_close_matches_exact_title_not_prefix(qapp):
    da = DockArea()
    try:
        # Two docks whose titles share a prefix -- the classic wrong-close case.
        short = _RecordingDock('Simulation-RLC-2')
        long = _RecordingDock('Simulation-RLC-21')
        for d in (short, long):
            da.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea, d)
            d.setVisible(True)

        # Build a tab bar whose tab 0 reads the *shorter* title.
        tab_bar = QtWidgets.QTabBar()
        tab_bar.addTab('Simulation-RLC-2')

        da.handle_tab_close(0, tab_bar)

        # Only the exact-title dock is destroyed; the prefix sibling survives.
        assert short.close_events == 1
        assert long.close_events == 0
        assert long.windowTitle() == 'Simulation-RLC-21'
    finally:
        da.deleteLater()
        qapp.processEvents()


def test_tab_bars_do_not_elide(qapp):
    da = DockArea()
    try:
        # Tabify two docks so a dock-area tab bar exists, then arm close buttons.
        d = _RecordingDock('Simulation-Long-Name-1')
        da.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea, d)
        da.tabifyDockWidget(da._docks['Welcome'], d)

        for tb in da.findChildren(QtWidgets.QTabBar):
            if isinstance(tb.parent(), QtWidgets.QTabWidget):
                continue
            assert tb.elideMode() == QtCore.Qt.TextElideMode.ElideNone
    finally:
        da.deleteLater()
        qapp.processEvents()
