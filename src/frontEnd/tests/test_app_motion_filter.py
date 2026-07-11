"""P1.2: the two app-wide event filters merged into one, gated on event type.

PopupMotionFilter (menu/combo/tree rounding) and EffectShowRefreshFilter
(drop-shadow revalidate on Show) were both installed application-wide, so every
event in the process crossed C++->Python twice and ran isinstance checks. They
are now one AppWideMotionFilter that bails on anything but Polish/Show first.

These pin that the gate short-circuits and that BOTH merged behaviours survive.
"""
from PyQt6 import QtCore, QtWidgets

from frontEnd import motion

_WA_TRANSLUCENT = QtCore.Qt.WidgetAttribute.WA_TranslucentBackground


def test_gate_ignores_non_polish_show_events(qapp):
    filt = motion.AppWideMotionFilter(None)
    menu = QtWidgets.QMenu()
    # A Timer event (stand-in for the mouse/paint/timer flood) must be dropped
    # without touching the object.
    ev = QtCore.QEvent(QtCore.QEvent.Type.Timer)
    assert filt.eventFilter(menu, ev) is False
    assert not menu.testAttribute(_WA_TRANSLUCENT)


def test_polish_still_rounds_menus(qapp):
    filt = motion.AppWideMotionFilter(None)
    menu = QtWidgets.QMenu()
    filt.eventFilter(menu, QtCore.QEvent(QtCore.QEvent.Type.Polish))
    assert menu.testAttribute(_WA_TRANSLUCENT)


def test_show_revalidates_drop_shadow_without_error(qapp):
    filt = motion.AppWideMotionFilter(None)
    w = QtWidgets.QWidget()
    eff = QtWidgets.QGraphicsDropShadowEffect(w)
    w.setGraphicsEffect(eff)
    # Merged filter schedules a deferred off/on toggle on Show; flush the
    # singleShot and confirm the effect ends enabled (net no-op, no crash).
    filt.eventFilter(w, QtCore.QEvent(QtCore.QEvent.Type.Show))
    QtWidgets.QApplication.processEvents()
    QtWidgets.QApplication.sendPostedEvents(None, QtCore.QEvent.Type.Timer)
    assert eff.isEnabled()


def test_install_app_motion_registers_single_filter(qapp):
    motion.install_app_motion(qapp)
    assert isinstance(
        qapp._esim_app_motion_filter, motion.AppWideMotionFilter)
