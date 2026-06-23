"""Named elevation scale — replaces the scattered per-widget shadow magic
numbers across motion.py / Welcome.py / ProjectExplorer.py / dialogs.py.

One scale, two alpha tracks (dark vs light), and light-mode shadows are
tinted blue-grey so they read as soft ambient occlusion on white instead
of an invisible black smudge.

Native only: QGraphicsDropShadowEffect.
"""
from PyQt6 import QtWidgets, QtGui

try:
    from frontEnd import tokens
except Exception:  # pragma: no cover - import when run as a script
    import tokens

# level: (blur, dx, dy, alpha_dark, alpha_light)
_SCALE = {
    "e1": (16, 0, 3,  60,  30),   # resting buttons / list rows
    "e2": (24, 0, 6,  90,  42),   # cards, tree, inputs, panels
    "e3": (34, 0, 10, 120, 58),   # docks, toolbars
    "e4": (46, 0, 16, 150, 72),   # dialogs, popped-out windows
    "e5": (60, 0, 22, 180, 92),   # modal / About / Preferences
}


def is_dark(widget) -> bool:
    return widget.palette().color(
        QtGui.QPalette.ColorRole.Window).lightness() < 128


def elevate(widget, level="e2", tint=None):
    """Apply (or update) a layered drop shadow at a named elevation.

    ``tint`` optionally colours the shadow with an accent for a 'glow'.
    Light mode otherwise uses a blue-grey tinted shadow (premium ambient
    occlusion) rather than pure black.
    """
    if level not in _SCALE:
        level = "e2"
    blur, dx, dy, ad, al = _SCALE[level]
    eff = widget.graphicsEffect()
    if not isinstance(eff, QtWidgets.QGraphicsDropShadowEffect):
        eff = QtWidgets.QGraphicsDropShadowEffect(widget)
        widget.setGraphicsEffect(eff)
    eff.setBlurRadius(blur)
    eff.setOffset(dx, dy)
    dark = is_dark(widget)
    if tint:
        c = QtGui.QColor(tint)
    else:
        r, g, b = tokens.theme(dark)["shadow_rgb"]
        c = QtGui.QColor(r, g, b)
    c.setAlpha(ad if dark else al)
    eff.setColor(c)
    return eff


def clear(widget):
    """Remove any drop shadow."""
    eff = widget.graphicsEffect()
    if isinstance(eff, QtWidgets.QGraphicsDropShadowEffect):
        widget.setGraphicsEffect(None)


def refresh_effects(root):
    """Re-validate cached QGraphicsEffect renders within ``root`` + children.

    A QGraphicsDropShadowEffect caches a pixmap of its source. After a
    hide->show (QStackedWidget page switch, tab change) — especially while a
    window is maximized/fullscreen, where the expose/repaint path differs —
    or a theme swap, that cache can stay stale so the widget paints blank
    until a hover marks it dirty. Toggling each effect off/on forces an
    immediate clean repaint.
    """
    if root is None:
        return
    try:
        targets = [root] + root.findChildren(QtWidgets.QWidget)
        for w in targets:
            eff = w.graphicsEffect()
            if eff is not None and eff.isEnabled():
                eff.setEnabled(False)
                eff.setEnabled(True)
        root.update()
    except Exception:
        pass
