# =========================================================================
#             FILE: FullScreen.py
#
#      DESCRIPTION: A small, contextual "go fullscreen" control for a docked
#                   panel. It is meant to live in the panel's OWN header
#                   (the Makerchip flow strip, the Plotting toolbar, the
#                   KicadToNgspice tab corner) -- never a global toolbar -- so
#                   the affordance sits where the user is actually working and
#                   reads as "fullscreen THIS, then dock it back".
#
#                   On activation it reparents the host dock's content into a
#                   frameless top-level window and shows it *truly* fullscreen
#                   (the whole screen, not merely the app's work area). The
#                   same button flips to an exit affordance; Esc and F11 also
#                   exit. Docking back returns the content to its QDockWidget,
#                   which keeps its original tab slot.
#
#  ORGANIZATION: eSim Team at FOSSEE, IIT Bombay
# =========================================================================
from PyQt6 import QtCore, QtGui, QtWidgets, sip

try:
    from frontEnd.icon_paths import dock_back_icon, fullscreen_icon
except Exception:  # pragma: no cover - flat sys.path (script / harness run)
    from icon_paths import dock_back_icon, fullscreen_icon


class FullScreenToggle(QtWidgets.QToolButton):
    """Per-panel fullscreen toggle. Drop one into any panel's header; it finds
    its host QDockWidget at click time, so no wiring is needed."""

    _ICON_PX = 14

    def __init__(self, parent=None):
        super().__init__(parent)
        self._win = None
        self._dock = None
        self._content = None
        self._full = False
        self._icon_refresh_pending = False
        self.setAutoRaise(True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setIconSize(QtCore.QSize(self._ICON_PX, self._ICON_PX))
        self.setStyleSheet(
            "QToolButton { border:none; background:transparent; padding:2px 6px; }")
        self._set_state(full=False)
        self.clicked.connect(self.toggle)

    # ------------------------------------------------------------------ #
    def _set_state(self, full):
        # eSim's own SVG pair, not SP_TitleBarMaxButton/SP_TitleBarNormalButton:
        # the standard pixmaps resolve to the platform's title-bar glyphs, so
        # this button was a Windows chrome square on one OS and an icon-theme
        # arrow on another -- exactly what icon_paths exists to end.
        self._full = bool(full)
        self.setIcon(dock_back_icon(self._ICON_PX) if full
                     else fullscreen_icon(self._ICON_PX))
        self.setToolTip("Exit fullscreen  (Esc)" if full
                        else "Fullscreen this panel  (Esc to exit)")

    def changeEvent(self, event):
        # icon_paths bakes the theme's foreground INTO the rasterised SVG, so
        # an icon built under dark keeps painting near-white after a switch to
        # light. Re-render on the palette change; the state is unaffected.
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.Type.PaletteChange:
            self._schedule_icon_refresh()

    def _schedule_icon_refresh(self):
        """Re-render on the NEXT tick, never inside the handler.

        PaletteChange is delivered from inside the style polish that
        setStyleSheet/setPalette is running. Rasterising an SVG and calling
        setIcon() there re-enters that polish, which re-delivers PaletteChange
        -- with one of these toggles in every docked panel that is unbounded
        recursion, i.e. a C-stack overflow (an access violation on Windows),
        not a slow repaint. Deferring also coalesces the burst of events a
        single theme toggle produces into one re-render.
        """
        if self._icon_refresh_pending:
            return
        self._icon_refresh_pending = True
        QtCore.QTimer.singleShot(0, self._refresh_icon)

    def _refresh_icon(self):
        self._icon_refresh_pending = False
        # The panel can be torn down between the event and the tick.
        if sip.isdeleted(self):
            return
        self._set_state(self._full)

    def _resolve_host(self):
        """Walk up to the enclosing QDockWidget; the widget just beneath it is
        the content to reparent."""
        content = self
        node = self.parentWidget()
        while node is not None:
            if isinstance(node, QtWidgets.QDockWidget):
                return node, content
            content = node
            node = node.parentWidget()
        return None, None

    # ------------------------------------------------------------------ #
    def toggle(self):
        if self._win is not None:
            self._exit()
        else:
            self._enter()

    def _enter(self):
        dock, content = self._resolve_host()
        if dock is None or content is None:
            return
        self._dock, self._content = dock, content

        win = QtWidgets.QWidget()
        win.setWindowTitle(dock.windowTitle())
        lay = QtWidgets.QVBoxLayout(win)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(content)          # reparents content out of the dock

        # Esc / F11 (and an external Alt+F4) all route through the close path.
        for key in ("Escape", "F11"):
            sc = QtGui.QShortcut(QtGui.QKeySequence(key), win)
            sc.activated.connect(win.close)
        win.closeEvent = self._make_close_handler()

        self._win = win
        self._set_state(full=True)
        win.showFullScreen()

    def _make_close_handler(self):
        def _on_close(event):
            if self._win is not None:
                self._win = None
                dock, content = self._dock, self._content
                # The host dock can be destroyed WHILE its panel is fullscreen
                # (Close Project / closing the now-empty tab). The content is
                # safe -- it was reparented into the fullscreen window, not the
                # dock -- but the QDockWidget wrapper is gone, so setWidget/show/
                # raise_ on it would RuntimeError inside this closeEvent.
                dock_alive = dock is not None and not sip.isdeleted(dock)
                if dock_alive and content is not None:
                    # Back into the dock -- it kept its tab slot all along.
                    dock.setWidget(content)
                    dock.show()
                    dock.raise_()
                elif content is not None and not sip.isdeleted(content):
                    # Dock is gone: drop the orphaned panel rather than leaving
                    # it floating parentless for the rest of the session.
                    content.setParent(None)
                    content.deleteLater()
                self._dock = None
                self._content = None
                self._set_state(full=False)
            event.accept()
        return _on_close

    def _exit(self):
        if self._win is not None:
            self._win.close()
