"""Live, reusable pieces of the Aurora design language.

* mono_family      — the portable monospace family every editor/console
                     resolves through, so one machine's missing font does not
                     silently change the face on that platform alone.
* accent_color     — the active theme's accent as a QColor, for painters:
                     a QSS rule can name a token, a QPainter cannot.
* HoverSurfaceMixin— a surface that lifts under the cursor: an animated
                     0->1 progress a painter can read, plus a drop shadow that
                     fades from its resting elevation into an accent glow.

Everything in here has a consumer; that is the rule for this module. It used
to also carry a painted GradientLabel / AuroraHeroFrame and a whole custom
dock-drag stack (DockTitleBar, DockDropOverlay, FloatingDockHost,
RailDragGrip) that nothing ever instantiated — deleted in the UI audit's P3
pass. The dock-drag work encoded four hard-won Wayland/DnD findings; those
are written up in ``DockArea.apply_fullscreen_feature``, and the code itself
is in ``git log -- src/frontEnd/widgets.py``.

Depends only on PyQt6 plus the design-system leaves (``tokens``/``elevation``).
"""
from PyQt6 import QtCore, QtGui, QtWidgets

try:
    from frontEnd import elevation, tokens
except Exception:  # pragma: no cover
    import elevation
    import tokens


def _is_dark(widget) -> bool:
    return widget.palette().color(
        QtGui.QPalette.ColorRole.Window).lightness() < 128


def accent_color(widget, alpha=255, key="accent") -> QtGui.QColor:
    """The active theme's ``key`` token as a QColor with ``alpha``.

    For painters: a QSS rule can name a token, a QPainter cannot, and the
    literal it would otherwise carry is the same literal that goes stale the
    day the palette is retuned.
    """
    c = QtGui.QColor(tokens.theme(_is_dark(widget))[key])
    c.setAlpha(alpha)
    return c


def _lerp_color(a: QtGui.QColor, b: QtGui.QColor, t: float) -> QtGui.QColor:
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return QtGui.QColor(
        int(a.red() + (b.red() - a.red()) * t),
        int(a.green() + (b.green() - a.green()) * t),
        int(a.blue() + (b.blue() - a.blue()) * t),
        int(a.alpha() + (b.alpha() - a.alpha()) * t),
    )


class HoverSurfaceMixin:
    """Cursor-lift behaviour for a card-like surface.

    Mix in front of a QWidget subclass, call :meth:`_init_hover_anim` from
    ``__init__``, and read ``self._hover_progress`` (0..1) in ``paintEvent``.

    The shadow half only engages if the widget already carries a drop shadow;
    it interpolates from EXACTLY what ``elevation.elevate(w, REST_LEVEL)``
    painted — resting colour, alpha and geometry all come off the scale — to an
    accent glow. Nothing here re-invents a black shadow beside that table,
    which is what made the old copy invisible in light mode: it faded from
    black-alpha-48 on a #F3F7FC page.
    """

    #: the elevation the surface rests at; override per class if needed
    REST_LEVEL = "e2"
    #: extra blur, and the glow's alpha per theme, at full hover
    HOVER_BLUR_LIFT = 20
    HOVER_GLOW_ALPHA = (160, 120)      # dark, light

    def _init_hover_anim(self, duration=None):
        self._hover_progress = 0.0
        self._hover_anim = QtCore.QPropertyAnimation(
            self, b"hoverProgress", self)
        self._hover_anim.setDuration(
            tokens.DUR["base"] if duration is None else duration)
        self._hover_anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)

    def hover_glow_color(self) -> QtGui.QColor:
        dark = _is_dark(self)
        return accent_color(self, self.HOVER_GLOW_ALPHA[0 if dark else 1])

    def getHoverProgress(self):
        return self._hover_progress

    def setHoverProgress(self, value):
        self._hover_progress = float(value)
        self.update()

        eff = self.graphicsEffect()
        if not isinstance(eff, QtWidgets.QGraphicsDropShadowEffect):
            return
        blur, _, dy, _, _ = elevation.spec(self.REST_LEVEL)
        rest = elevation.shadow_color(self, self.REST_LEVEL)
        eff.setColor(_lerp_color(rest, self.hover_glow_color(), value))
        eff.setBlurRadius(blur + int(self.HOVER_BLUR_LIFT * value))
        # The card rises toward the cursor, so its shadow tucks back under it.
        eff.setOffset(0, dy - int(dy * value))

    hoverProgress = QtCore.pyqtProperty(
        float, fget=getHoverProgress, fset=setHoverProgress)

    def enterEvent(self, event):
        self._animate_hover(1.0)
        if hasattr(super(), 'enterEvent'):
            super().enterEvent(event)

    def leaveEvent(self, event):
        self._animate_hover(0.0)
        if hasattr(super(), 'leaveEvent'):
            super().leaveEvent(event)

    def _animate_hover(self, end):
        self._hover_anim.stop()
        self._hover_anim.setStartValue(self._hover_progress)
        self._hover_anim.setEndValue(end)
        self._hover_anim.start()


_MONO_CACHE = None


#: Preference chain, head-aligned with what the .qss sheets DECLARE
#: ("JetBrains Mono", "Cascadia Mono", "Consolas", monospace) so a console
#: styled by the sheet and an editor styled in Python resolve to one face.
MONO_PREFS = ["JetBrains Mono", "Cascadia Mono", "Cascadia Code",
              "DejaVu Sans Mono", "Menlo", "Consolas", "Liberation Mono",
              "Courier New"]


def mono_family() -> str:
    """First installed cross-platform monospace family.

    'Consolas' is Windows-only; relying on it makes the editors render with a
    silent platform fallback on Linux/macOS. This picks the best available
    from a portable preference chain so the default looks identical anywhere.
    """
    global _MONO_CACHE
    if _MONO_CACHE:
        return _MONO_CACHE
    try:
        fams = set(QtGui.QFontDatabase.families())
    except Exception:
        fams = set()
    for f in MONO_PREFS:
        if f in fams:
            _MONO_CACHE = f
            return f
    # Deliberately NOT cached: with no QGuiApplication yet, families() is
    # empty, and caching that would pin every later caller to the generic.
    return "monospace"
