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
from frontEnd.icon_paths import fullscreen_icon, dock_back_icon


class FullScreenToggle(QtWidgets.QToolButton):
    """Per-panel fullscreen toggle. Drop one into any panel's header; it finds
    its host QDockWidget at click time, so no wiring is needed."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._win = None
        self._dock = None
        self._content = None
        self.setAutoRaise(True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        # Icon-only left users hunting for an unlabelled glyph. Show the label
        # beside the icon so the control reads as "[⛶ Fullscreen]" -- the
        # affordance is now self-explanatory instead of relying on a tooltip.
        self.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.setStyleSheet(
            "QToolButton { border:none; background:transparent; padding:2px 8px; }")
        self._set_state(full=False)
        self.clicked.connect(self.toggle)

    # ------------------------------------------------------------------ #
    def _set_state(self, full):
        # Purpose-built, theme-aware SVGs (icon_paths) rather than the OS
        # SP_TitleBar* pixmap, which rasterised mushy and read as a window
        # control, not "fullscreen this panel".
        if full:
            self.setIcon(dock_back_icon())
            self.setText("Exit Fullscreen")
            self.setToolTip("Exit fullscreen  (Esc)")
        else:
            self.setIcon(fullscreen_icon())
            self.setText("Fullscreen")
            self.setToolTip("Fullscreen this panel  (Esc to exit)")

    def changeEvent(self, event):
        # The SVG icon colour is baked in at build time, so a live light/dark
        # theme switch would leave the glyph the old theme's colour. Rebuild it
        # (state preserved: _win set == currently fullscreen) on PaletteChange.
        if event.type() == QtCore.QEvent.Type.PaletteChange:
            self._set_state(full=self._win is not None)
        super().changeEvent(event)

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
            # Keep a LOCAL reference to the fullscreen window for the whole
            # handler. ``content`` is a child of ``win``; if the only reference
            # to ``win`` (``self._win``) were dropped first, sip would delete
            # win -- and content with it -- so the setWidget below touched a
            # freed QWidget (RuntimeError) and left the dock's tab blank. The
            # local ref keeps win, hence content, alive until we reparent.
            win = self._win
            if win is not None:
                self._win = None      # re-entrant close is now a no-op
                dock, content = self._dock, self._content
                self._dock = None
                self._content = None

                content_alive = (content is not None
                                 and not sip.isdeleted(content))
                # The host dock can be destroyed WHILE its panel is fullscreen
                # (Close Project / closing the now-empty tab), so verify it too
                # before setWidget/show/raise_ on a possibly-gone QDockWidget.
                dock_alive = dock is not None and not sip.isdeleted(dock)
                if dock_alive and content_alive:
                    # Reparent BEFORE win dies -- back into the dock, which kept
                    # its tab slot all along.
                    dock.setWidget(content)
                    dock.show()
                    dock.raise_()
                elif content_alive:
                    # Dock is gone: drop the orphaned panel rather than leaving
                    # it floating parentless for the rest of the session.
                    content.setParent(None)
                    content.deleteLater()

                self._set_state(full=False)
                # Reap the now-empty fullscreen wrapper. deleteLater defers the
                # delete until after this closeEvent unwinds, so win is never
                # deleted from inside its own event.
                win.deleteLater()
            event.accept()
        return _on_close

    def _exit(self):
        if self._win is not None:
            self._win.close()
