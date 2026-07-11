import os
import weakref
from PyQt6 import QtCore, QtGui, QtWidgets

# Motion (button glows + per-button drop shadows) defaults OFF on Windows, where
# widget graphics effects are CPU-blurred with no compositor help and are the
# biggest structural drag; ON elsewhere so the effect stays discoverable. The
# Preferences toggle overrides this default either way.
_MOTION_DEFAULT = (os.name != "nt")

# Every TactileButtonFilter registers here so a theme change can stop all of
# their running glow animations before the global re-polish toggles the very
# graphics effects those animations are driving (a freed effect touched by a
# live animation = use-after-free → segfault). WeakSet so transient dialogs'
# filters don't leak.
_TACTILE_FILTERS = weakref.WeakSet()


def stop_all_glow():
    """Stop every button glow animation across all installed filters. Call
    before re-applying the theme (setStyleSheet + effect toggling)."""
    for f in list(_TACTILE_FILTERS):
        try:
            f.stop_all_glow()
        except RuntimeError:
            pass


def is_dark_widget(widget):
    return widget.palette().color(QtGui.QPalette.ColorRole.Window).lightness() < 128


def accent_qcolor(widget):
    c = widget.palette().color(QtGui.QPalette.ColorRole.Highlight)
    if not c.isValid():
        c = QtGui.QColor("#53D7FF" if is_dark_widget(widget) else "#0077A8")
    return c


def set_shadow(widget, blur=24, x=0, y=8, alpha=70, color=None):
    eff = widget.graphicsEffect()
    if not isinstance(eff, QtWidgets.QGraphicsDropShadowEffect):
        eff = QtWidgets.QGraphicsDropShadowEffect(widget)
        widget.setGraphicsEffect(eff)
    eff.setBlurRadius(blur)
    eff.setOffset(x, y)
    c = QtGui.QColor(color) if color else QtGui.QColor(0, 0, 0)
    c.setAlpha(alpha)
    eff.setColor(c)
    return eff


def apply_panel_depth(widget, blur=28, y=8, alpha=76):
    return set_shadow(widget, blur=blur, y=y, alpha=alpha)


def apply_popup_depth(widget):
    return set_shadow(widget, blur=42, y=14, alpha=128)


def apply_accent_glow(widget, blur=22, alpha=74):
    return set_shadow(widget, blur=blur, y=0, alpha=alpha, color=accent_qcolor(widget))


def make_menu_rounded(menu):
    """Give a QMenu genuinely transparent corners.

    The menu QSS uses ``border-radius`` to round the frame, but a QMenu's
    top-level window is an opaque rectangle by default, so the four corners
    OUTSIDE the rounded frame paint as solid (black) squares — the "black
    corners" gap between the rounding and the squared window. Marking the
    window translucent makes those corners transparent so only the rounded
    frame shows. Set this on a freshly-created menu BEFORE it is first shown
    (the native window must be created with the translucent attribute), so
    call it right after ``QMenu(...)`` / ``addMenu(...)``.
    """
    if menu is None:
        return menu
    try:
        menu.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
    except Exception:
        pass
    return menu


def apply_menu_rounded_mask(menu, radius=14):
    """Clip a QMenu's window to a rounded-rect region so its corners are round
    on EVERY surface.

    ``WA_TranslucentBackground`` only produces transparent corners when the
    popup's surface has an alpha channel. Over a QWebEngineView (Makerchip /
    User Manual) or the plotting widgets that surface has no alpha, so the
    squared window's corners paint solid black around the QSS ``border-radius``.
    A mask physically removes those corner pixels from the window — no alpha or
    compositor needed — so the rounding shows everywhere. Re-applied on each
    Show because a QMenu sizes to its content (the mask must match its size).
    The radius matches the QMenu QSS ``border-radius`` (14px).
    """
    if menu is None:
        return
    try:
        size = menu.size()
        if size.width() <= 0 or size.height() <= 0:
            return
        bmp = QtGui.QBitmap(size)
        bmp.fill(QtCore.Qt.GlobalColor.color0)          # color0 = clipped out
        p = QtGui.QPainter(bmp)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setPen(QtCore.Qt.GlobalColor.color1)          # color1 = kept
        p.setBrush(QtCore.Qt.GlobalColor.color1)
        p.drawRoundedRect(0, 0, size.width(), size.height(), radius, radius)
        p.end()
        menu.setMask(QtGui.QRegion(bmp))
    except Exception:
        pass


class TactileButtonFilter(QtCore.QObject):
    """Button hover/press feedback via shadow glow only.
    Position animation removed — conflicts with Qt layouts inside toolbars.
    Press depth comes from QSS padding-top/padding-bottom shift instead."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._glow_anims = {}
        _TACTILE_FILTERS.add(self)

    def stop_all_glow(self):
        for anim in list(self._glow_anims.values()):
            try:
                anim.stop()
            except RuntimeError:
                pass
        self._glow_anims.clear()

    def eventFilter(self, obj, event):
        if not isinstance(obj, (QtWidgets.QPushButton, QtWidgets.QToolButton)) or not obj.isEnabled():
            return False
        if obj.property("noMotion"):
            return False

        is_dock = bool(obj.property("dockPopButton"))
        et = event.type()

        if et == QtCore.QEvent.Type.Enter:
            obj.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            if not is_dock:
                self._animate_glow(obj, 18, 4, 84)
        elif et == QtCore.QEvent.Type.Leave:
            if not is_dock:
                self._animate_glow(obj, 18, 4, 14)
        elif et == QtCore.QEvent.Type.MouseButtonPress and event.button() == QtCore.Qt.MouseButton.LeftButton:
            if not is_dock:
                self._animate_glow(obj, 14, 1, 60)
        elif et == QtCore.QEvent.Type.MouseButtonRelease:
            if not is_dock:
                self._animate_glow(obj, 18, 4, 84 if obj.underMouse() else 14)
        return False

    def _animate_glow(self, obj, blur, y, alpha):
        """Smoothly transition the drop-shadow glow color on a button."""
        # The previous animation was started DeleteWhenStopped, so once it has
        # finished Qt has freed its C++ object — calling stop() on the stale
        # Python ref then raises "wrapped C/C++ object … deleted". That storm of
        # exceptions fired during the global re-polish a theme change triggers
        # and could touch a freed graphics effect (use-after-free → segfault).
        # Guard the stop and drop the stale entry.
        old = self._glow_anims.pop(obj, None)
        if old is not None:
            try:
                old.stop()
            except RuntimeError:
                pass
        eff = obj.graphicsEffect()
        if not isinstance(eff, QtWidgets.QGraphicsDropShadowEffect):
            return
        acc = accent_qcolor(obj)
        try:
            current_alpha = eff.color().alpha()
        except RuntimeError:
            return

        anim = QtCore.QVariantAnimation(obj)
        anim.setDuration(140)
        anim.setStartValue(current_alpha / 255.0)
        anim.setEndValue(alpha / 255.0)
        anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)

        def _apply_alpha(val):
            # The effect can be torn down mid-animation (theme re-polish / dock
            # close); never touch a freed C++ effect.
            try:
                new_c = QtGui.QColor(acc)
                new_c.setAlpha(int(val * 255))
                eff.setColor(new_c)
            except RuntimeError:
                pass
        anim.valueChanged.connect(_apply_alpha)
        # Drop our reference the instant it stops, so a later stop() can never
        # reach a DeleteWhenStopped-freed animation.
        anim.finished.connect(lambda o=obj: self._glow_anims.pop(o, None))
        self._glow_anims[obj] = anim
        anim.start(QtCore.QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)

        try:
            eff.setBlurRadius(blur)
            eff.setOffset(0, y)
        except RuntimeError:
            pass


_motion_enabled_cache = None


def motion_enabled():
    """True when animated button glows are enabled (default on).

    Hover/press glows drive continuous repaints + a per-button drop-shadow.
    They're on by default so the effect is discoverable, with a Preferences
    toggle to turn them off (reduce motion / save power). The prefs file is
    read once and cached (this is called on every dock open and every button
    wire-up); Preferences save invalidates the cache via
    :func:`invalidate_motion_cache`. Reading the file keeps this module free of
    an Appconfig import; an absent/unreadable file falls back to the default.
    """
    global _motion_enabled_cache
    if _motion_enabled_cache is not None:
        return _motion_enabled_cache
    import json
    from configuration import paths
    try:
        path = paths.esim_config_path("preferences.json")
        with open(path) as fh:
            val = bool(json.load(fh).get("enable_motion", _MOTION_DEFAULT))
    except Exception:
        val = _MOTION_DEFAULT
    _motion_enabled_cache = val
    return val


def invalidate_motion_cache():
    """Drop the cached enable_motion value so the next read re-hits the file.

    Call after Preferences writes preferences.json."""
    global _motion_enabled_cache
    _motion_enabled_cache = None


def install_button_motion(root):
    # Single gate for every call site: when motion is off, install nothing so
    # no glow animations or extra graphics effects are ever created.
    if not motion_enabled():
        return
    # Reuse ONE filter per root across repeated calls. install_button_motion
    # runs on every dock open; a fresh TactileButtonFilter each time would
    # STACK on every button (installEventFilter with a new object appends, it
    # never replaces), so after N dock opens each button carried N filters and
    # every hover fired N glow animations -- the "gets laggy over a session"
    # signature, since the DockArea lives for the whole app lifetime.
    filt = getattr(root, "_esim_press_motion_filter", None)
    if filt is None:
        filt = TactileButtonFilter(root)
        root._esim_press_motion_filter = filt
    for w in root.findChildren((QtWidgets.QPushButton, QtWidgets.QToolButton)):
        if isinstance(w.parent(), QtWidgets.QToolBar):
            continue
        # Idempotent per button: one already wired in a previous pass keeps its
        # single filter and its one base shadow -- skip it so neither stacks.
        if w.property("_esim_motion_installed"):
            continue
        w.setProperty("_esim_motion_installed", True)
        w.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        w.installEventFilter(filt)
        # noMotion opts a button fully out: no glow animation AND no base shadow
        # effect. Small chrome controls (e.g. the verifier's move arrows) use it
        # so the neon halo never bleeds over their neighbours.
        if not w.property("dockPopButton") and not w.property("noMotion"):
            eff = w.graphicsEffect()
            if eff is None:
                eff = QtWidgets.QGraphicsDropShadowEffect(w)
                eff.setBlurRadius(18)
                eff.setOffset(0, 4)
                # Base neon glow so it has the shadow by default
                c = accent_qcolor(w)
                c.setAlpha(14)
                eff.setColor(c)
                w.setGraphicsEffect(eff)


class TabMotionFilter(QtCore.QObject):
    """Cursor feedback for tab bars: hand cursor, glow on press."""

    def eventFilter(self, obj, event):
        if not isinstance(obj, QtWidgets.QTabBar):
            return False
        if event.type() == QtCore.QEvent.Type.Enter:
            obj.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        elif event.type() == QtCore.QEvent.Type.MouseButtonPress:
            obj.setCursor(QtCore.Qt.CursorShape.ClosedHandCursor)
            apply_accent_glow(obj, blur=20, alpha=60)
        elif event.type() in (QtCore.QEvent.Type.MouseButtonRelease, QtCore.QEvent.Type.Leave):
            obj.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            if event.type() == QtCore.QEvent.Type.Leave:
                eff = obj.graphicsEffect()
                if isinstance(eff, QtWidgets.QGraphicsDropShadowEffect):
                    c = eff.color()
                    c.setAlpha(0)
                    eff.setColor(c)
        return False


def install_tab_kinetics(root):
    filt = TabMotionFilter(root)
    root._esim_tab_kinetic_filter = filt
    for bar in root.findChildren(QtWidgets.QTabBar):
        bar.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        bar.installEventFilter(filt)


class PopupMotionFilter(QtCore.QObject):
    """Make every QMenu translucent so QSS border-radius corners are genuine.

    No opacity fade animation — animating ``windowOpacity`` on a QMenu can
    cause the native window to re-create mid-show, which generates a second
    Show event and makes the menu flash open/closed every other click.
    The rounding alone (WA_TranslucentBackground + QSS border-radius) is
    enough for the polished look.
    """

    def eventFilter(self, obj, event):
        et = event.type()

        # Cheap gate FIRST: this filter only reacts to Polish and Show. As an
        # app-wide filter it sees every event in the process (MouseMove, Paint,
        # Timer, ...), each crossing C++->Python; bail on the ~99% we ignore
        # before doing any isinstance / type-name work.
        if et not in (QtCore.QEvent.Type.Polish, QtCore.QEvent.Type.Show):
            return False

        # Smooth expand/collapse for every tree (project tree, device-model
        # grouping, etc.). Set once, when the view is first polished — Qt has a
        # built-in animated transition; we just have to opt in.
        if isinstance(obj, QtWidgets.QTreeView) and et == QtCore.QEvent.Type.Polish:
            try:
                obj.setAnimated(True)
            except Exception:
                pass
            return False

        # ComboBox dropdowns share the menu problem: the popup *container* is a
        # square top-level window, so the rounded QSS on the inner item-view used
        # to sit on sharp black corners. Same fix as menus -- translucent surface
        # before the native window exists, plus a rounded mask on show for the
        # no-compositor / over-QWebEngineView case. Radius matches the item-view
        # QSS (10px).
        if type(obj).__name__ == "QComboBoxPrivateContainer":
            if et == QtCore.QEvent.Type.Polish:
                if not obj.testAttribute(
                        QtCore.Qt.WidgetAttribute.WA_TranslucentBackground):
                    obj.setAttribute(
                        QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
            elif et == QtCore.QEvent.Type.Show:
                apply_menu_rounded_mask(obj, radius=10)
                QtCore.QTimer.singleShot(
                    0, lambda c=obj: apply_menu_rounded_mask(c, radius=10))
            return False

        if isinstance(obj, QtWidgets.QMenu):
            if et == QtCore.QEvent.Type.Polish:
                # Set translucency BEFORE the native popup window is created.
                # A post-show attribute change (or setMask) is ignored once the
                # window is realized as an opaque rectangle — which is why menus
                # built by QWebEngineView (Makerchip) / the plotting module kept
                # black corners while our own menus (rounded at creation) did
                # not. Polish fires before window creation, so the alpha-rounded
                # corners now work for every menu, on any compositor (incl.
                # Wayland, where window-shape masks are not honored at all).
                if not obj.testAttribute(
                        QtCore.Qt.WidgetAttribute.WA_TranslucentBackground):
                    obj.setAttribute(
                        QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
            elif et == QtCore.QEvent.Type.Show:
                # X11-without-compositor fallback: clip the corners with a
                # rounded mask for any surface that still lacks alpha. Harmless
                # where translucency already rounded them; does NOT recreate the
                # native window (so no every-other-click flash). Re-applied next
                # tick so the mask matches the menu's final laid-out size.
                apply_menu_rounded_mask(obj)
                QtCore.QTimer.singleShot(0, lambda m=obj: apply_menu_rounded_mask(m))
        return False


def install_popup_motion(app):
    filt = PopupMotionFilter(app)
    app._esim_popup_motion_filter = filt
    app.installEventFilter(filt)


class EffectShowRefreshFilter(QtCore.QObject):
    """Re-validate a widget's drop-shadow cache whenever it is shown.

    A QGraphicsDropShadowEffect caches a render of its source. When a widget
    is hidden then shown again (QStackedWidget page switch, tab change) — most
    visibly while a window is maximized/fullscreen, where the expose/repaint
    path differs — that cache can stay stale and the widget paints blank until
    a hover marks it dirty. Toggling the effect off/on on Show forces a clean
    repaint. Installed application-wide so every effect-bearing widget is
    covered without per-site wiring.
    """

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Type.Show and isinstance(obj, QtWidgets.QWidget):
            eff = obj.graphicsEffect()
            if isinstance(eff, QtWidgets.QGraphicsDropShadowEffect) and eff.isEnabled():
                # Defer the toggle to the next tick: a Show can be delivered
                # mid-paint, and toggling the effect synchronously would force
                # a reentrant repaint ("paint device that is being painted").
                QtCore.QTimer.singleShot(0, lambda e=eff: self._revalidate(e))
        return False

    @staticmethod
    def _revalidate(eff):
        try:
            if eff.isEnabled():
                eff.setEnabled(False)
                eff.setEnabled(True)
        except RuntimeError:
            # underlying effect/widget was deleted before the tick fired
            pass


def install_effect_refresh(app):
    filt = EffectShowRefreshFilter(app)
    app._esim_effect_refresh_filter = filt
    app.installEventFilter(filt)


class AppWideMotionFilter(PopupMotionFilter):
    """The single app-wide filter: PopupMotionFilter's popup/tree rounding PLUS
    EffectShowRefreshFilter's drop-shadow revalidation on Show.

    Merged into one object so each of the process's events crosses the
    C++->Python boundary once, not twice (two separate app-wide filters each
    ran an eventFilter per event). The Polish/Show gate inherited from
    PopupMotionFilter still short-circuits everything else first.
    """

    def eventFilter(self, obj, event):
        et = event.type()
        # Same cheap gate as the base; keep it here so the added Show work below
        # is never reached for the events we ignore.
        if et not in (QtCore.QEvent.Type.Polish, QtCore.QEvent.Type.Show):
            return False
        # Popup rounding + tree animation (base handles Polish and Show).
        super().eventFilter(obj, event)
        # Effect revalidation is Show-only.
        if et == QtCore.QEvent.Type.Show and isinstance(obj, QtWidgets.QWidget):
            eff = obj.graphicsEffect()
            if isinstance(eff, QtWidgets.QGraphicsDropShadowEffect) \
                    and eff.isEnabled():
                QtCore.QTimer.singleShot(
                    0, lambda e=eff: EffectShowRefreshFilter._revalidate(e))
        return False


def install_app_motion(app):
    """Install the single merged app-wide motion filter (replaces the old
    install_popup_motion + install_effect_refresh pair)."""
    filt = AppWideMotionFilter(app)
    app._esim_app_motion_filter = filt
    app.installEventFilter(filt)


def install_context_menu_motion(root):
    filt = PopupMotionFilter(root)
    root._esim_context_motion_filter = filt
    root.installEventFilter(filt)


def apply_toolbar_depth(window):
    """Floating-toolbar shadows on both bars for 3D inverted-L.

    Both toolbars share the same opaque fill (QSS), so they read as one
    piece. The left rail is added AFTER the top bar, giving it higher
    z-order — its opaque background paints OVER the top bar's downward
    shadow at the joint, hiding the seam. The left rail's own shadow is
    pushed slightly DOWN (y=6) so its blur does not bleed upward into
    the corner.
    """
    specs = {
        "topToolbar": (34, 0, 5, 82),
        "leftToolBar": (24, 4, 6, 72),
    }
    for name, (blur, x, y, alpha) in specs.items():
        tb = window.findChild(QtWidgets.QToolBar, name)
        if not tb:
            continue
        eff = QtWidgets.QGraphicsDropShadowEffect(tb)
        eff.setBlurRadius(blur)
        eff.setOffset(x, y)
        eff.setColor(QtGui.QColor(0, 0, 0, alpha))
        tb.setGraphicsEffect(eff)


def install_menu_depth(root):
    """Pre-install depth shadows on menu bars for consistent 3D presence."""
    for menu in root.findChildren(QtWidgets.QMenu):
        eff = QtWidgets.QGraphicsDropShadowEffect(menu)
        eff.setBlurRadius(34)
        eff.setOffset(0, 12)
        eff.setColor(QtGui.QColor(0, 0, 0, 120))
        menu.setGraphicsEffect(eff)
