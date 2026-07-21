# ngspiceSimulation/_palette.py
"""Theme-aware color palette shared by every plotting widget.

The plotting surface has three independent color sources:

  1. The active light/dark theme (chosen by the user in preferences.json).
  2. The user-selected accent color (also in preferences.json).
  3. The trace color table (``constants.VIBRANT_COLOR_PALETTE``) which is data,
     not chrome, and stays the same across themes so existing user configs
     keep the colors they expect.

This module exposes ``current_palette()`` — a dict of named tokens — that the
plotting widgets read at construction time to build their stylesheets and to
feed matplotlib's ``rcParams``. The palette is a flat string-only dict so the
plotting stylesheet f-string can interpolate it directly without converting to
``QColor`` first.

The module never imports from frontEnd/ — it's a leaf helper, so the
backend plotting tree stays importable in headless tests.
"""
from __future__ import annotations

import os
import json
from typing import Any, Dict, Optional

from PyQt6 import QtCore, QtGui, QtWidgets


# Fallback defaults if the runtime can't reach the live QApplication.
#
# These are the Aurora palette (frontEnd/tokens.py), hand-copied on purpose:
# this module must not import frontEnd (leaf rule, see the module docstring).
# ``tests/test_palette_tokens_match.py`` asserts the copies stay in step with
# tokens.py, so drift shows up as a test failure instead of two design
# languages sitting side by side in the same window.
#
# The trace colors (``constants.VIBRANT_COLOR_PALETTE``) and the cursor marker
# hues below are data identity, not chrome — they stay put across a retint so
# saved user configs keep the colors they expect.
_LIGHT_DEFAULTS: Dict[str, Any] = {
    "is_dark": False,
    # Surfaces — tokens LIGHT bg / surface / surface_2
    "bg":            "#F3F7FC",
    "surface":       "#FFFFFF",
    "panel":         "#FFFFFF",
    "panel_alt":     "#F6F9FD",
    # Borders / dividers — tokens stroke, then the QPalette Mid/Midlight tones
    "border":        "#DCE6F1",
    "border_strong": "#AFC0D3",
    "divider":       "#DCE6F1",
    "spine_separator": "#AFC0D3",
    # Text — tokens text / text_muted / text_subtle
    "text":          "#142033",
    "text_muted":    "#6B7F99",
    "text_subtle":   "#9AAABE",
    # Brand — tokens accent / accent_hi / accent_lo
    "primary":       "#0077A8",
    "primary_hover": "#00A4DC",
    "primary_pressed": "#005E86",
    # Overlays — the accent tints the app sheet uses for the same states
    "hover_overlay":   "rgba(0,119,168,0.07)",
    "pressed_overlay": "rgba(0,119,168,0.24)",
    "selection_bg":    "rgba(0,119,168,0.16)",
    "selection_text":  "#142033",
    # Plotting axes/legend/grid — the axes are a card on the window backdrop
    "axes_face":     "#FFFFFF",
    "axes_edge":     "#DCE6F1",
    "label_color":   "#142033",
    "tick_color":    "#6B7F99",
    "grid_color":    "#DCE6F1",
    "legend_face":   "#FFFFFF",
    "legend_edge":   "#DCE6F1",
    "stats_text":    "#405168",   # tokens text_dim
    "info_text":     "#9AAABE",
    # Cursor UI
    "cursor1":       "#e53935",
    "cursor2":       "#1976d2",
    "cursor_delta":  "#e65100",
    "cursor_chrome": "#9AAABE",   # "@" / undefined-state labels
    "cursor_dim":    "#6B7F99",   # dimmed trace names in rows
    "cursor_disabled": "#AFC0D3",  # "not set" / "—" placeholders
}

_DARK_DEFAULTS: Dict[str, Any] = {
    "is_dark": True,
    # Surfaces — tokens DARK bg / surface / surface_2
    "bg":            "#050812",
    "surface":       "#0E1728",
    "panel":         "#0E1728",
    "panel_alt":     "#121E33",
    # Borders / dividers — tokens stroke, then the QPalette Mid/Midlight tones
    "border":        "#1D2B45",
    "border_strong": "#30415F",
    "divider":       "#1D2B45",
    "spine_separator": "#30415F",
    # Text — tokens text / text_muted / text_subtle
    "text":          "#F8FBFF",
    "text_muted":    "#94A8C3",
    "text_subtle":   "#5F728D",
    # Brand — tokens accent / accent_hi / accent_lo
    # (primary is replaced by the accent color when the user has set one)
    "primary":       "#53D7FF",
    "primary_hover": "#8BEAFF",
    "primary_pressed": "#18A8D8",
    # Overlays — the accent tints the app sheet uses for the same states
    "hover_overlay":   "rgba(83,215,255,0.12)",
    "pressed_overlay": "rgba(83,215,255,0.28)",
    "selection_bg":    "rgba(83,215,255,0.20)",
    "selection_text":  "#F8FBFF",
    # Plotting axes/legend/grid — the axes are a card on the window backdrop
    "axes_face":     "#0E1728",
    "axes_edge":     "#1D2B45",
    "label_color":   "#F8FBFF",
    "tick_color":    "#94A8C3",
    "grid_color":    "#1D2B45",
    "legend_face":   "#0E1728",
    "legend_edge":   "#1D2B45",
    "stats_text":    "#D3DEEF",   # tokens text_dim
    "info_text":     "#5F728D",
    # Cursor UI — brighter variants for dark mode contrast.
    "cursor1":       "#ef5350",
    "cursor2":       "#42a5f5",
    "cursor_delta":  "#ffb74d",
    "cursor_chrome": "#5F728D",
    "cursor_dim":    "#94A8C3",
    "cursor_disabled": "#30415F",
}


def _read_pref(pref: str, default: str) -> str:
    """Read a single key from ``~/.esim/preferences.json``."""
    try:
        home = os.path.expanduser("~")
        path = os.path.join(home, ".esim", "preferences.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            v = data.get(pref)
            if v:
                return str(v)
    except Exception:
        pass
    return default


def _detect_is_dark(app: Optional[QtWidgets.QApplication]) -> bool:
    """Decide whether we're rendering in dark mode.

    Priority: explicit ``theme_mode`` pref → live palette lightness → system
    colorScheme(). The palette branch is what catches a user overriding the
    palette programmatically (e.g. by a toggle that calls setPalette).
    """
    mode = _read_pref("theme_mode", "System")
    if mode == "Dark":
        return True
    if mode == "Light":
        return False

    if app is not None:
        try:
            win = app.palette().color(QtGui.QPalette.ColorRole.Window)
            if win.isValid() and win.lightness() < 128:
                return True
        except Exception:
            pass
        try:
            # colorScheme() needs Qt >= 6.5; on Ubuntu 24.04 (Qt 6.4) the
            # palette-lightness branch above is the only system signal.
            hints = QtGui.QGuiApplication.styleHints()
            if hasattr(hints, "colorScheme"):
                return hints.colorScheme() == QtCore.Qt.ColorScheme.Dark
        except Exception:
            pass
        return False
    return False


def current_palette(app: Optional[QtWidgets.QApplication] = None) -> Dict[str, Any]:
    """Return the theme-aware color dict for the plotting module.

    Reads preferences.json for ``theme_mode``, ``accent_color``,
    ``secondary_accent_color`` and ``internal_bg_color``. If the live app
    already has a palette set (after ``apply_theme()`` runs), we honor that
    by deriving ``text``/``bg``/``surface`` from the active palette so the
    widgets line up with the rest of eSim.
    """
    app = app or QtWidgets.QApplication.instance()
    dark = _detect_is_dark(app)
    palette: Dict[str, Any] = dict(_DARK_DEFAULTS if dark else _LIGHT_DEFAULTS)

    # Honor the accent color the user picked, if any.
    accent_pref = _read_pref("accent_color", "default")
    if accent_pref and accent_pref != "default":
        palette["primary"] = accent_pref

    # Match surfaces with whatever apply_theme wired into the global palette,
    # so opening a plotting window mid-session aligns with the rest of eSim.
    if app is not None:
        try:
            q_pal = app.palette()
            palette["bg"]       = q_pal.color(QtGui.QPalette.ColorRole.Base).name().upper() \
                                  if dark else q_pal.color(QtGui.QPalette.ColorRole.Base).name()
            palette["panel"]    = q_pal.color(QtGui.QPalette.ColorRole.Base).name()
            palette["surface"]  = q_pal.color(QtGui.QPalette.ColorRole.Window).name()
            palette["text"]     = q_pal.color(QtGui.QPalette.ColorRole.Text).name()
            palette["text_muted"] = q_pal.color(QtGui.QPalette.ColorRole.PlaceholderText).name() \
                                    if hasattr(QtGui.QPalette.ColorRole, "PlaceholderText") else palette["text_muted"]
        except Exception:
            # Stay with the static defaults if the palette isn't ready yet.
            pass

    return palette


def matplotlib_rc_overrides(palette: Dict[str, Any]) -> Dict[str, Any]:
    """Translate the plotting palette to a matplotlib ``rcParams`` update dict.

    Returned dict can be splatted into ``plt.rcParams.update(...)``. Only the
    keys that affect the plotting surface are included; theme/blur/anim keys
    stay at matplotlib's defaults.
    """
    return {
        "figure.facecolor":      palette["bg"],
        "axes.facecolor":        palette["axes_face"],
        "axes.edgecolor":        palette["axes_edge"],
        "axes.labelcolor":       palette["label_color"],
        "xtick.color":           palette["tick_color"],
        "ytick.color":           palette["tick_color"],
        "text.color":            palette["label_color"],
        "grid.color":            palette["grid_color"],
        "legend.facecolor":      palette["legend_face"],
        "legend.edgecolor":      palette["legend_edge"],
        "legend.labelcolor":     palette["label_color"],
        "savefig.facecolor":     palette["bg"],
        "savefig.edgecolor":     palette["bg"],
    }
