"""Custom rounded tooltip for the eSim 'Aurora' design system.

Qt's native ``QToolTip`` is a top-level window whose corners are NOT clipped by
a stylesheet ``border-radius`` — the radius only draws an inset rounded border
*inside* a square window, leaving the squared outer corners visible. That is
the "ugly square blue box" the QSS ``QToolTip`` rule could never fully fix.

This module replaces it application-wide with a frameless, translucent,
painted-rounded card that honours the active theme and carries a soft drop
shadow. Install once via :func:`install_tooltips`; an app-level event filter
then intercepts every ``QEvent.ToolTip``, suppresses the native tip and shows
ours instead.

No project imports — safe to import in headless contexts.
"""
from PyQt6 import QtCore, QtGui, QtWidgets


def _is_dark(app):
    try:
        return app.palette().color(
            QtGui.QPalette.ColorRole.Window).lightness() < 128
    except Exception:
        return True


def _tip_colors(app):
    """(background, text, border) as QSS-ready strings, accent-tinted border.

    Both themes use a dark navy card with light text — a dark tooltip over a
    light UI is a deliberate, legible contrast, and matches the prior intent of
    the QSS ``QToolTip`` rule."""
    dark = _is_dark(app)
    acc = app.palette().color(QtGui.QPalette.ColorRole.Highlight)
    if not acc.isValid():
        acc = QtGui.QColor("#53D7FF" if dark else "#0077A8")
    bg = "rgba(13,23,40,0.98)" if dark else "rgba(16,27,46,0.98)"
    text = "#EAF3FF"
    border = "rgba(%d,%d,%d,0.42)" % (acc.red(), acc.green(), acc.blue())
    return bg, text, border


class AuroraToolTip(QtWidgets.QWidget):
    """Frameless translucent tooltip with a painted rounded card + shadow.

    A transparent outer widget reserves ``_PAD`` px of breathing room on every
    side so the inner card's drop shadow has somewhere to fall; the visible card
    is the inner ``QFrame`` whose rounded corners are genuine (the window is
    translucent, so its own square corners are invisible)."""

    _PAD = 16  # transparent margin around the card for the shadow blur

    def __init__(self):
        super().__init__(None)
        self.setWindowFlags(
            QtCore.Qt.WindowType.ToolTip
            | QtCore.Qt.WindowType.FramelessWindowHint
        )
        self.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        self._card = QtWidgets.QFrame(self)
        self._card.setObjectName("auroraTipCard")
        self._label = QtWidgets.QLabel(self._card)
        self._label.setTextFormat(QtCore.Qt.TextFormat.AutoText)
        self._label.setWordWrap(True)
        self._label.setMaximumWidth(380)

        card_lay = QtWidgets.QVBoxLayout(self._card)
        card_lay.setContentsMargins(13, 9, 13, 10)
        card_lay.addWidget(self._label)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(self._PAD, self._PAD, self._PAD, self._PAD)
        outer.addWidget(self._card)

        shadow = QtWidgets.QGraphicsDropShadowEffect(self._card)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 8)
        shadow.setColor(QtGui.QColor(0, 0, 0, 160))
        self._card.setGraphicsEffect(shadow)

        self._hide_timer = QtCore.QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide)

    def show_text(self, text, global_pos, app):
        bg, fg, border = _tip_colors(app)
        self._card.setStyleSheet(
            "QFrame#auroraTipCard{"
            "background:%s;border:1px solid %s;border-radius:11px;}" % (bg, border)
        )
        self._label.setStyleSheet(
            "color:%s;font-size:12px;font-weight:500;"
            "background:transparent;border:none;" % fg
        )
        self._label.setText(text)
        self.adjustSize()
        self._place(global_pos)
        self.show()
        self.raise_()
        # Safety-net auto-hide; the Leave/press handlers normally dismiss first.
        self._hide_timer.start(min(15000, 4000 + len(text) * 55))

    def _place(self, gpos):
        screen = (QtWidgets.QApplication.screenAt(gpos)
                  or QtWidgets.QApplication.primaryScreen())
        geo = screen.availableGeometry()
        sz = self.size()
        pad = self._PAD
        # Offset the visible card from the cursor; the transparent pad cancels
        # so the card (not the window) sits at the intended offset.
        x = gpos.x() + 16 - pad
        y = gpos.y() + 22 - pad
        # Flip above the cursor if it would overflow the bottom edge.
        if y + sz.height() - pad > geo.bottom():
            y = gpos.y() - sz.height() + pad - 6
        x = max(geo.left() - pad, min(x, geo.right() - sz.width() + pad))
        y = max(geo.top() - pad, min(y, geo.bottom() - sz.height() + pad))
        self.move(x, y)


class _ToolTipFilter(QtCore.QObject):
    """App-level filter: show the Aurora tooltip, suppress the native one."""

    _DISMISS = frozenset((
        QtCore.QEvent.Type.Leave,
        QtCore.QEvent.Type.WindowDeactivate,
        QtCore.QEvent.Type.FocusOut,
        QtCore.QEvent.Type.Wheel,
        QtCore.QEvent.Type.MouseButtonPress,
        QtCore.QEvent.Type.KeyPress,
    ))

    def __init__(self, app):
        super().__init__(app)
        self._app = app
        self._tip = None

    def _ensure(self):
        if self._tip is None:
            self._tip = AuroraToolTip()
        return self._tip

    def _hide(self):
        if self._tip is not None and self._tip.isVisible():
            self._tip.hide()

    @staticmethod
    def _item_tip(global_pos):
        """ToolTipRole of the item-view row under ``global_pos`` (or "")."""
        w = QtWidgets.QApplication.widgetAt(global_pos)
        view = w
        while view is not None and not isinstance(
                view, QtWidgets.QAbstractItemView):
            view = view.parent()
        if not isinstance(view, QtWidgets.QAbstractItemView):
            return ""
        pos = view.viewport().mapFromGlobal(global_pos)
        idx = view.indexAt(pos)
        if not idx.isValid():
            return ""
        data = idx.data(QtCore.Qt.ItemDataRole.ToolTipRole)
        return data or ""

    def eventFilter(self, obj, event):
        et = event.type()
        if et == QtCore.QEvent.Type.ToolTip:
            text = ""
            if isinstance(obj, QtWidgets.QWidget):
                text = obj.toolTip()
            if not text:
                # Item-view tips (e.g. the project tree) live on the item, not
                # the widget: read the hovered index's ToolTipRole so those get
                # the Aurora card too instead of the native square box.
                text = self._item_tip(event.globalPos())
            if not text:
                # Fall back to whatever bare widget sits under the cursor.
                w = QtWidgets.QApplication.widgetAt(event.globalPos())
                if isinstance(w, QtWidgets.QWidget):
                    text = w.toolTip()
            if text:
                self._ensure().show_text(text, event.globalPos(), self._app)
                return True   # suppress the native square tooltip
            self._hide()
            return False
        if et in self._DISMISS and self._tip is not None:
            # Never dismiss on events delivered to the tooltip itself.
            if obj is not self._tip and not (
                isinstance(obj, QtWidgets.QWidget)
                and self._tip.isAncestorOf(obj)
            ):
                self._hide()
        return False


def install_tooltips(app):
    """Install the Aurora tooltip filter on ``app`` (idempotent)."""
    if getattr(app, "_esim_tooltip_filter", None) is not None:
        return app._esim_tooltip_filter
    filt = _ToolTipFilter(app)
    app._esim_tooltip_filter = filt
    app.installEventFilter(filt)
    return filt
