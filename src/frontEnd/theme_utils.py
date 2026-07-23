import os
import sys
import json
import ctypes
from PyQt6 import QtGui, QtCore, QtWidgets
from configuration import paths

ACCENT_TOKENS = {
    "dark": ["#53D7FF", "#8BEAFF", "#18A8D8", "#0E7490", "#7CE3FF", "#1CB8E8", "#3B82F6", "#165982", "#1E88E5"],
    "light": ["#0077A8", "#00A4DC", "#005E86"],
}

SECONDARY_TOKENS = {
    "dark": ["#050812", "#070C18", "#070B14", "#111827"],
    "light": ["#F3F7FC", "#EEF4FB", "#F6F8FC", "#f4f5f7"],
}

INTERNAL_TOKENS = {
    "dark": ["#0E1728", "#08111F", "#09111F", "#0B1220", "#111B2D", "#101827", "#151F32", "#1A2740", "#0A1020"],
    "light": ["#FFFFFF", "#FBFDFF", "#F8FBFF", "#FBFCFE", "#EEF4FA", "#EEF3FA"],
}


# Whether the last apply_theme() resolved to dark. Read by the Show-time
# titlebar hook in motion.py so windows created AFTER a theme apply still get
# the right caption color.
_CURRENT_DARK = False


def current_theme_is_dark():
    return _CURRENT_DARK


def apply_titlebar_theme(window, is_dark=None):
    """Windows-only: color the native titlebar to match the active theme.

    Qt never touches the titlebar — it is drawn by DWM, which follows the OS
    accent/light setting, so a dark-themed eSim kept a light (or accent-
    colored) caption bar that no stylesheet can reach. On Ubuntu the window
    manager themes the decoration itself, which is why this mismatch never
    shows there. DwmSetWindowAttribute is the only way in:

      20 DWMWA_USE_IMMERSIVE_DARK_MODE — dark caption (Win10 1809+)
      35 DWMWA_CAPTION_COLOR           — exact caption color (Win11+)
      36 DWMWA_TEXT_COLOR              — caption text color  (Win11+)

    The color attributes fail harmlessly (E_INVALIDARG) on Win10; the
    immersive flag alone still gets a dark bar there. Safe to call repeatedly
    and on any top-level widget; no-op off Windows or before the native
    window exists.
    """
    if sys.platform != "win32" or window is None or not window.isWindow():
        return
    if is_dark is None:
        is_dark = _CURRENT_DARK
    try:
        # Never force native-window creation here: winId() on a not-yet-shown
        # widget realizes the window early, which would defeat attributes that
        # must be set pre-creation (WA_TranslucentBackground) and re-open the
        # first-show flash. Windows without a handle get themed by the Show
        # hook in motion.py once they actually appear.
        if window.windowHandle() is None:
            return
        hwnd = int(window.winId())
        if not hwnd:
            return
        dwm = ctypes.windll.dwmapi
        dark_flag = ctypes.c_int(1 if is_dark else 0)
        dwm.DwmSetWindowAttribute(
            ctypes.c_void_p(hwnd), 20,
            ctypes.byref(dark_flag), ctypes.sizeof(dark_flag))

        # COLORREF is 0x00BBGGRR; match the QPalette Window/WindowText pair
        # so the caption reads as part of the app surface.
        def colorref(hex_color):
            c = QtGui.QColor(hex_color)
            return ctypes.c_uint32(
                (c.blue() << 16) | (c.green() << 8) | c.red())

        caption = colorref("#050812" if is_dark else "#F3F7FC")
        text = colorref("#F8FBFF" if is_dark else "#142033")
        dwm.DwmSetWindowAttribute(
            ctypes.c_void_p(hwnd), 35,
            ctypes.byref(caption), ctypes.sizeof(caption))
        dwm.DwmSetWindowAttribute(
            ctypes.c_void_p(hwnd), 36,
            ctypes.byref(text), ctypes.sizeof(text))
    except Exception:
        pass


def apply_round_corners(window):
    """Windows-only: have DWM round the window's corners in the compositor.

      33 DWMWA_WINDOW_CORNER_PREFERENCE / 2 DWMWCP_ROUND (Win11+)

    Qt cannot round a popup by itself here. Under the Fusion base style the
    popup's native window is never marked layered (WS_EX_LAYERED stays clear),
    so its raster backing store has nowhere to put alpha: everything outside
    the QSS border-radius flushes as OPAQUE BLACK. WA_TranslucentBackground
    therefore buys no transparency on this path and actively costs a black
    corner, and clipping those pixels off with setMask() only trades the black
    for a hard staircase -- a QRegion is 1-bit, so it cannot hold the partial
    coverage a smooth curve needs, and Qt scales the logical-pixel mask up to
    device pixels (x1.75 at 175% display scaling), which coarsens the steps
    further. That staircase is what the rounded menus actually looked like.

    DWM has the alpha the backing store lacks. It clips and antialiases the
    window after Qt has painted, so the corners come out smooth with no mask
    and no translucency. Requires the native window to exist -- call it on
    Show, not on create. Fails harmlessly (E_INVALIDARG) on Win10, which has
    no rounded corners to ask for.
    """
    if sys.platform != "win32" or window is None:
        return
    try:
        if window.windowHandle() is None:
            return
        hwnd = int(window.winId())
        if not hwnd:
            return
        pref = ctypes.c_int(2)  # DWMWCP_ROUND
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            ctypes.c_void_p(hwnd), 33,
            ctypes.byref(pref), ctypes.sizeof(pref))
    except Exception:
        pass


def replace_tokens(qss, tokens, value):
    for token in tokens:
        qss = qss.replace(token, value)
    return qss


class ComboPopupStyle(QtWidgets.QProxyStyle):
    """Fusion with the macOS-flavoured combo popup turned off.

    Fusion answers SH_ComboBox_Popup = 1, which makes QComboBox treat its
    popup as a *menu*: QComboBoxPrivateContainer paints PE_PanelMenu (an
    opaque square panel with a 1px border) behind the item view, insets the
    view by the menu's vertical margins, and positions the popup so the
    current item lands on top of the combo instead of dropping below it. Our
    sheet then draws its own rounded border on the view inside all that, so
    every dropdown reads as a rounded card floating in a square one, opening
    in the wrong place.

    Answering 0 puts QComboBox back on the plain drop-down path: no menu
    panel, no margins, popup anchored under the combo. The container is still
    an opaque top-level window, so it would show square corners behind the
    view's border-radius -- polish() makes it translucent and frameless, which
    leaves the item view as the only thing that paints.
    """

    _POLISH_FLAG = "_esim_combo_popup_polished"

    def styleHint(self, hint, option=None, widget=None, returnData=None):
        if hint == QtWidgets.QStyle.StyleHint.SH_ComboBox_Popup:
            return 0
        if hint == QtWidgets.QStyle.StyleHint.SH_ComboBox_PopupFrameStyle:
            return int(QtWidgets.QFrame.Shape.NoFrame)
        return super().styleHint(hint, option, widget, returnData)

    def polish(self, target):
        # QStyle::polish is overloaded on QWidget/QApplication/QPalette and
        # PyQt routes all three here; the QPalette one has to return its
        # argument or Qt reads a null palette back.
        if isinstance(target, QtGui.QPalette):
            return super().polish(target)

        # setWindowFlag() reparents the container, which re-polishes it, so
        # this would recurse without the flag.
        if (isinstance(target, QtWidgets.QWidget)
                and target.metaObject().className()
                == "QComboBoxPrivateContainer"
                and not target.property(self._POLISH_FLAG)):
            target.setProperty(self._POLISH_FLAG, True)
            target.setWindowFlag(
                QtCore.Qt.WindowType.FramelessWindowHint, True)
            target.setWindowFlag(
                QtCore.Qt.WindowType.NoDropShadowWindowHint, True)
            target.setAttribute(
                QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)

        super().polish(target)


# Process-lifetime cache of fully-resolved stylesheets, keyed by everything
# that affects the output: (qss file, accent, secondary, internal, zoom). The
# build step reads the file, rewrites image urls, runs three token-replace
# passes, an rgba() recolor regex and a per-metric px-scale regex over a large
# sheet -- doing that on every theme toggle (the user flipping Light/Dark back
# and forth, or the OS colorScheme signal firing) is pure waste when the inputs
# repeat. There are only a handful of distinct combinations in a session, so an
# unbounded dict here stays tiny.
_QSS_CACHE = {}


def build_qss(qss_name, is_dark, accent_color, secondary_color,
              internal_bg_color, zoom_level):
    """Return the fully-resolved stylesheet string for the given inputs,
    memoized. Pure function of its arguments (plus the on-disk .qss, which does
    not change during a run), so it is safe to cache for the process lifetime."""
    key = (qss_name, accent_color, secondary_color, internal_bg_color,
           zoom_level)
    cached = _QSS_CACHE.get(key)
    if cached is not None:
        return cached

    qss_path = os.path.join(os.path.dirname(__file__), qss_name)
    if not os.path.exists(qss_path):
        _QSS_CACHE[key] = ""
        return ""
    with open(qss_path, 'r') as f:
        qss_content = f.read()

    # Resolve the relative ``url("images/...")`` references in the QSS to an
    # absolute path next to this module. Previously these only worked because
    # the launcher does ``cd src/frontEnd`` first; from any other working
    # directory (or a frozen PyInstaller build) the dock/tab icons silently
    # failed to load. This makes them CWD-independent.
    img_dir = os.path.join(os.path.dirname(__file__), 'images')
    img_dir = img_dir.replace(os.sep, '/')
    qss_content = qss_content.replace('url("images/', 'url("%s/' % img_dir)

    mode_key = "dark" if is_dark else "light"

    if accent_color != "default":
        qss_content = replace_tokens(
            qss_content, ACCENT_TOKENS[mode_key], accent_color)
        # Also recolor the rgba() glows/borders so the accent fully propagates.
        qss_content = recolor_accent_rgba(qss_content, mode_key, accent_color)

    if secondary_color != "system":
        qss_content = replace_tokens(
            qss_content, SECONDARY_TOKENS[mode_key], secondary_color)

    if internal_bg_color != "system":
        qss_content = replace_tokens(
            qss_content, INTERNAL_TOKENS[mode_key], internal_bg_color)

    if zoom_level != 100:
        scale = zoom_level / 100.0
        import re

        def scale_px(match):
            val = float(match.group(1))
            # Keep 1-2px hairline borders crisp; only scale real metrics.
            if val <= 2:
                return match.group(0)
            return f"{int(round(val * scale))}px"
        qss_content = re.sub(r'(\d+(?:\.\d+)?)px', scale_px, qss_content)

    _QSS_CACHE[key] = qss_content
    return qss_content


def recolor_accent_rgba(qss, mode_key, accent_hex):
    """Recolor every ``rgba(<default-accent>, a)`` glow to the chosen accent.

    The old token-replace step only swapped a handful of *solid* hexes, so a
    custom accent left ~200 hard-coded ``rgba(83,215,255,a)`` (dark) /
    ``rgba(0,119,168,a)`` (light) borders, hovers and glows the original cyan.
    This recolors all of them so the accent truly propagates across the UI.
    """
    try:
        from frontEnd import tokens as _tok
    except Exception:
        import tokens as _tok
    import re
    r0, g0, b0 = _tok.DEFAULT_ACCENT_RGB[mode_key]
    nr, ng, nb = _tok.hex_to_rgb(accent_hex)
    if (r0, g0, b0) == (nr, ng, nb):
        return qss
    pattern = re.compile(
        r"rgba\(\s*%d\s*,\s*%d\s*,\s*%d\s*," % (r0, g0, b0))
    return pattern.sub("rgba(%d,%d,%d," % (nr, ng, nb), qss)


# The UI's px metrics were hand-tuned against a workspace this many *logical*
# pixels tall and wide -- i.e. the screen size at which zoom_level 100 is the
# right answer. Derived from the one hand-calibrated data point we have: a
# 1646x1029 logical workspace (a 2880x1800 panel at 175% Windows scaling) was
# tuned by eye to 90%, so 1029 / 0.90 ~= 1150 and 1646 / 0.90 ~= 1830.
_DESIGN_WORKSPACE_H = 1150
_DESIGN_WORKSPACE_W = 1830

# Never auto-pick a value the user would immediately have to undo. Below 60%
# the menu and status text stop being comfortably readable, and above 150% we
# would be overriding the deliberate "I want a small UI" choice the user
# already expressed in their OS display settings.
_CALIBRATION_FLOOR = 60
_CALIBRATION_CEILING = 150


def calibrate_default_zoom(screen=None):
    """Pick a sensible first-run zoom for the screen eSim is starting on.

    Qt has already divided the OS scale factor out of these numbers, so what
    we read is the *logical* workspace: a 1920x1080 panel at 150% Windows
    scaling reports 1280x720, and genuinely has only that much room for UI.
    eSim's chrome (menu bar, top toolbar, left rail, dock tabs, status bar) is
    a fixed logical-pixel cost, so on a short workspace it eats a far larger
    share of the window. That -- not the panel's resolution or its DPI -- is
    why the same build reads as "great at 60%" on one machine and "great at
    90%" on another, and why a single hard-coded default cannot serve both.

    Height is the binding constraint on every 16:9 / 16:10 display; the width
    term only bites on unusually narrow screens (e.g. 1280x1024), where the
    left rail would otherwise crowd the canvas.
    """
    if screen is None:
        screen = QtGui.QGuiApplication.primaryScreen()
    if screen is None:
        return 100
    # availableGeometry, not geometry: the taskbar/dock is space eSim will
    # never get, and a machine with less of it should start smaller.
    avail = screen.availableGeometry()
    if avail.height() <= 0 or avail.width() <= 0:
        return 100
    ratio = min(avail.height() / _DESIGN_WORKSPACE_H,
                avail.width() / _DESIGN_WORKSPACE_W)
    # Land on the same 10% grid the -/+ buttons step through, so the value the
    # user sees in the pill is one they could have dialled in themselves.
    zoom = int(round(ratio * 10)) * 10
    return max(_CALIBRATION_FLOOR, min(_CALIBRATION_CEILING, zoom))


def ensure_zoom_calibrated(screen=None):
    """Seed zoom_level from the screen on first run, then never touch it again.

    Once the key exists -- because we wrote it here, or because the user
    touched the zoom pill -- it is the user's preference and is left alone,
    including if they later move the window to a different monitor. Returns
    the zoom level now in effect.
    """
    prefs = get_preferences()
    existing = prefs.get("zoom_level")
    if isinstance(existing, int) and 50 <= existing <= 300:
        return existing
    zoom = calibrate_default_zoom(screen)
    prefs["zoom_level"] = zoom
    try:
        paths.write_json_atomic(
            paths.esim_config_path("preferences.json"), prefs)
    except Exception:
        # A read-only config dir must never block startup -- the calibrated
        # value still applies to this session, we just recompute next launch.
        pass
    return zoom


def get_preferences(user_home=None):
    prefs = {"theme_mode": "System", "accent_color": "default", "secondary_accent_color": "system"}
    try:
        path = (os.path.join(user_home, ".esim", "preferences.json")
                if user_home else paths.esim_config_path("preferences.json"))
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
                prefs.update(data)
    except Exception as e:
        print("Error loading preferences: ", str(e))
    return prefs


def _refresh_graphics_effects(app):
    """Invalidate cached QGraphicsEffect renders after a theme change.

    A QGraphicsDropShadowEffect keeps a cached pixmap of its source widget.
    A stylesheet/palette swap repaints the widget but leaves the effect cache
    stale, so the widget (e.g. a themed button) can render blank until a hover
    forces it dirty. Toggling enabled off/on re-validates the source and
    triggers an immediate repaint. Cheap: theme changes are rare.
    """
    from PyQt6 import QtWidgets
    for tlw in app.topLevelWidgets():
        try:
            targets = [tlw] + tlw.findChildren(QtWidgets.QWidget)
            for w in targets:
                eff = w.graphicsEffect()
                if eff is not None and eff.isEnabled():
                    eff.setEnabled(False)
                    eff.setEnabled(True)
            tlw.update()
        except Exception:
            pass


def system_is_dark():
    """True when the OS prefers a dark color scheme.

    QStyleHints.colorScheme() only exists on Qt >= 6.5; Ubuntu 24.04 LTS ships
    Qt 6.4, where calling it raises AttributeError (which used to silently
    disable all theming at startup and crash the theme toggle). Fall back to
    GNOME's color-scheme setting, then to palette lightness.
    """
    hints = QtGui.QGuiApplication.styleHints()
    if hasattr(hints, "colorScheme"):
        return hints.colorScheme() == QtCore.Qt.ColorScheme.Dark
    try:
        import subprocess
        out = subprocess.run(
            ["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"],
            capture_output=True, text=True, timeout=2).stdout
        if "dark" in out.lower():
            return True
        if out.strip():
            return False
    except Exception:
        pass
    win = QtGui.QGuiApplication.palette().color(QtGui.QPalette.ColorRole.Window)
    return win.isValid() and win.lightness() < 128


def apply_theme(app):
    prefs = get_preferences()
    theme_mode = prefs.get("theme_mode", "System")
    accent_color = prefs.get("accent_color", "default")
    secondary_color = prefs.get("secondary_accent_color", "system")
    internal_bg_color = prefs.get("internal_bg_color", "system")

    # Stop every button glow animation before we re-polish: setStyleSheet +
    # _refresh_graphics_effects() below toggle the very QGraphicsDropShadowEffect
    # objects a live animation drives, and a freed effect touched by a running
    # animation is a use-after-free that segfaults during the theme change.
    try:
        from frontEnd import motion
        motion.stop_all_glow()
    except Exception:
        pass

    is_dark = False
    if theme_mode == "Dark":
        is_dark = True
    elif theme_mode == "Light":
        is_dark = False
    else:
        is_dark = system_is_dark()

    global _CURRENT_DARK
    _CURRENT_DARK = is_dark

    if is_dark:
        qss_name = 'style_dark.qss'
    else:
        qss_name = 'style_light.qss'

    # Build (or fetch from cache) the fully-resolved sheet. Toggling theme back
    # and forth now re-reads nothing and re-runs no regex once each combination
    # has been seen once.
    zoom_level = prefs.get("zoom_level", 100)
    qss_content = build_qss(qss_name, is_dark, accent_color, secondary_color,
                            internal_bg_color, zoom_level)

    # Install the palette BEFORE the stylesheet. QApplication.setPalette()
    # propagates unreliably while an app stylesheet is active (documented Qt
    # caveat: style sheets and setPalette don't mix), so widgets whose QSS
    # rules leave the background to the palette (e.g. the dock-area backdrop
    # behind Welcome) kept the *previous* theme's palette when the palette was
    # set after the sheet — light mode showed a dark dock. Setting the palette
    # first means the full repolish that setStyleSheet() triggers resolves
    # every widget against the new palette in one pass. (The old code got away
    # with sheet-then-palette only because the unconditional setStyle("Fusion")
    # afterwards forced a second full repolish.)
    if is_dark:
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#050812"))
        palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#F8FBFF"))
        palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#08111F"))
        palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor("#0E1728"))
        palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor("#0E1728"))
        palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor("#F8FBFF"))
        palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor("#F8FBFF"))
        palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#0E1728"))
        palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("#F8FBFF"))
        palette.setColor(QtGui.QPalette.ColorRole.Link, QtGui.QColor("#53D7FF"))
        palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(accent_color if accent_color != "default" else "#53D7FF"))
        palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#FFFFFF"))
        palette.setColor(QtGui.QPalette.ColorRole.Mid, QtGui.QColor("#1D2B45"))
        palette.setColor(QtGui.QPalette.ColorRole.Midlight, QtGui.QColor("#30415F"))
        palette.setColor(QtGui.QPalette.ColorRole.PlaceholderText, QtGui.QColor("#94A8C3"))
        app.setPalette(palette)
    else:
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#F3F7FC"))
        palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#142033"))
        palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#FBFDFF"))
        palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor("#EDF4FA"))
        palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor("#142033"))
        palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor("#F8FBFF"))
        palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor("#142033"))
        palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#FFFFFF"))
        palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("#142033"))
        palette.setColor(QtGui.QPalette.ColorRole.Link, QtGui.QColor("#0077A8"))
        palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(accent_color if accent_color != "default" else "#0077A8"))
        palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#FFFFFF"))
        palette.setColor(QtGui.QPalette.ColorRole.Mid, QtGui.QColor("#D6E1EE"))
        palette.setColor(QtGui.QPalette.ColorRole.Midlight, QtGui.QColor("#AFC0D3"))
        palette.setColor(QtGui.QPalette.ColorRole.PlaceholderText, QtGui.QColor("#6B7F99"))
        app.setPalette(palette)

    app.setStyleSheet(qss_content)

    # Set the base widget style exactly once per application. setStyle() is
    # not a no-op when the style is already Fusion: every call constructs a
    # fresh QStyle and re-polishes every widget in the process — a second
    # full-app repolish on top of the one setStyleSheet() above already did.
    # On a populated session (docks + editors + plot windows, thousands of
    # widgets) that redundant pass alone costs ~0.7s of the theme-toggle
    # freeze. The active style never changes after startup, so gate it.
    # (Can't compare app.style().objectName(): with an app stylesheet
    # installed the active style is a QStyleSheetStyle whose name is "".)
    #
    # setStyle() takes ownership and deletes the outgoing style, so the proxy
    # has to be constructed at the call rather than cached and reused; the
    # gate makes that a single instance living as long as the QApplication.
    if not getattr(app, "_esim_base_style_set", False):
        app.setStyle(ComboPopupStyle("Fusion"))
        app._esim_base_style_set = True

    # Re-theming (setStyleSheet + setPalette) does NOT invalidate the cached
    # render of a QGraphicsDropShadowEffect, so every widget that carries one
    # (all motion-installed buttons, toolbars, the project tree, Welcome cards,
    # Verilog panels) can paint blank after a theme switch until a hover forces
    # a repaint. Toggling each effect off/on marks its source dirty.
    #
    # Run it now AND again on the next event-loop tick: per-widget changeEvent
    # handlers (SpiceEditor / ProjectExplorer) re-apply their own stylesheets in
    # response to the QEvent.PaletteChange that this setPalette posts, which can
    # land *after* the synchronous pass and re-stale their effect — the deferred
    # pass mops that up.
    _refresh_graphics_effects(app)
    QtCore.QTimer.singleShot(0, lambda: _refresh_graphics_effects(app))

    from frontEnd.icon_paths import workspace_icon, timeline_icon, help_icon, dev_docs_icon, settings_icon, home_icon
    for widget in app.topLevelWidgets():
        # Keep every open window's native titlebar in step with the theme
        # (windows shown later are handled by the Show hook in motion.py).
        apply_titlebar_theme(widget, is_dark)
        if hasattr(widget, 'home_action'):
            widget.home_action.setIcon(home_icon())
        if hasattr(widget, 'wrkspce'):
            widget.wrkspce.setIcon(workspace_icon())
        if hasattr(widget, 'timeline_action'):
            widget.timeline_action.setIcon(timeline_icon())
        if hasattr(widget, 'helpfile'):
            widget.helpfile.setIcon(help_icon())
        if hasattr(widget, 'devdocs'):
            widget.devdocs.setIcon(dev_docs_icon())
        if hasattr(widget, 'preferences_action'):
            widget.preferences_action.setIcon(settings_icon())
