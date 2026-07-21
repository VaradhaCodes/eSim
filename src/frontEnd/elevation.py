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

#: Stamped on every widget ``elevate`` touches, so a theme change can find
#: what this module painted and re-tint it (see ``retint``).
LEVEL_PROP = "_esim_elevation"

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


def spec(level="e2"):
    """The raw ``(blur, dx, dy, alpha_dark, alpha_light)`` row for a level."""
    return _SCALE.get(level, _SCALE["e2"])


def shadow_color(widget, level="e2", tint=None) -> QtGui.QColor:
    """Exactly the colour ``elevate`` would paint on ``widget`` at ``level``.

    Exposed so a painter that animates its own shadow (the Welcome tiles fade
    theirs out into an accent glow) can start from the elevation the widget
    rests at, instead of re-inventing a black one beside this table.
    """
    _, _, _, ad, al = spec(level)
    dark = is_dark(widget)
    if tint:
        c = QtGui.QColor(tint)
    else:
        r, g, b = tokens.theme(dark)["shadow_rgb"]
        c = QtGui.QColor(r, g, b)
    c.setAlpha(ad if dark else al)
    return c


def elevate(widget, level="e2", tint=None, offset=None):
    """Apply (or update) a layered drop shadow at a named elevation.

    ``tint`` optionally colours the shadow with an accent for a 'glow'.
    Light mode otherwise uses a blue-grey tinted shadow (premium ambient
    occlusion) rather than pure black.

    ``offset`` overrides only the direction ``(dx, dy)``, keeping the level's
    blur, alpha and colour. It exists for surfaces whose shadow must be aimed
    rather than dropped — the two toolbars meeting at the inverted-L joint are
    the only case (see ``motion.apply_toolbar_depth``).
    """
    blur, dx, dy, _, _ = spec(level)
    if offset is not None:
        dx, dy = offset
    eff = widget.graphicsEffect()
    if not isinstance(eff, QtWidgets.QGraphicsDropShadowEffect):
        eff = QtWidgets.QGraphicsDropShadowEffect(widget)
        widget.setGraphicsEffect(eff)
    eff.setBlurRadius(blur)
    eff.setOffset(dx, dy)
    eff.setColor(shadow_color(widget, level, tint))
    # Only untinted shadows are tagged: a tint is a caller's own colour and
    # retint() must not overwrite it.
    if not tint:
        widget.setProperty(LEVEL_PROP, level)
    return eff


def retint(widget):
    """Re-colour ``widget``'s shadow for the theme it is in NOW, if it was
    elevated by this module. Returns True when a shadow was re-coloured.

    ``elevate`` bakes the colour at call time, so without this every shadow
    keeps its old theme's tint until the widget happens to be re-elevated —
    which for a toolbar or a project tree is 'never'. The level is read back
    off the widget rather than tracked in a registry so a destroyed widget
    cannot strand an entry.
    """
    level = widget.property(LEVEL_PROP)
    if not level:
        return False
    eff = widget.graphicsEffect()
    if not isinstance(eff, QtWidgets.QGraphicsDropShadowEffect):
        return False
    eff.setColor(shadow_color(widget, level))
    return True


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
