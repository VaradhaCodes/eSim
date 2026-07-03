import os
import json
from PyQt6 import QtGui, QtCore

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


def replace_tokens(qss, tokens, value):
    for token in tokens:
        qss = qss.replace(token, value)
    return qss


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


def get_preferences(user_home):
    prefs = {"theme_mode": "System", "accent_color": "default", "secondary_accent_color": "system"}
    try:
        path = os.path.join(user_home, ".esim", "preferences.json")
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


def apply_theme(app):
    if os.name == 'nt':
        user_home = os.path.join('library', 'config')
    else:
        user_home = os.path.expanduser('~')

    prefs = get_preferences(user_home)
    theme_mode = prefs.get("theme_mode", "System")
    accent_color = prefs.get("accent_color", "default")
    secondary_color = prefs.get("secondary_accent_color", "system")
    internal_bg_color = prefs.get("internal_bg_color", "system")

    scheme = QtGui.QGuiApplication.styleHints().colorScheme()

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
        is_dark = (scheme == QtCore.Qt.ColorScheme.Dark)

    if is_dark:
        qss_name = 'style_dark.qss'
    else:
        qss_name = 'style_light.qss'

    qss_path = os.path.join(os.path.dirname(__file__), qss_name)
    qss_content = ""
    if os.path.exists(qss_path):
        with open(qss_path, 'r') as f:
            qss_content = f.read()

        # Resolve the relative ``url("images/...")`` references in the QSS to an
        # absolute path next to this module. Previously these only worked
        # because the launcher does ``cd src/frontEnd`` first; from any other
        # working directory (or a frozen PyInstaller build) the dock/tab icons
        # silently failed to load. This makes them CWD-independent.
        img_dir = os.path.join(os.path.dirname(__file__), 'images')
        img_dir = img_dir.replace(os.sep, '/')
        qss_content = qss_content.replace('url("images/', 'url("%s/' % img_dir)

        mode_key = "dark" if is_dark else "light"

        if accent_color != "default":
            qss_content = replace_tokens(qss_content, ACCENT_TOKENS[mode_key], accent_color)
            # Also recolor the rgba() glows/borders so the accent fully propagates.
            qss_content = recolor_accent_rgba(qss_content, mode_key, accent_color)

        if secondary_color != "system":
            qss_content = replace_tokens(qss_content, SECONDARY_TOKENS[mode_key], secondary_color)

        if internal_bg_color != "system":
            qss_content = replace_tokens(qss_content, INTERNAL_TOKENS[mode_key], internal_bg_color)

        zoom_level = prefs.get("zoom_level", 100)
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

    app.setStyleSheet(qss_content)

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

    app.setStyle("Fusion")

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
