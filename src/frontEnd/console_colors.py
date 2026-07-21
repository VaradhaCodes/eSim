"""Semantic colours for eSim's text consoles, one per theme.

Six widgets in this tree paint log output with colour — the Verilog verifier
console, the d_cosim log, the NGSpice simulation console, the NgVeri terminal
and the two banner paths in ModelGeneration. Each of them had grown its own
palette of light-web literals (GitHub greys, pure ``#000000``, neon
``#00FF00``), which is invisible or garish once the app flips to dark. This
module is the single source those six now read.

Two rules shaped the values:

1. **Semantics, not hues.** A caller asks for ``ok`` / ``error`` / ``head``,
   never for a green. Which green is a theme decision made here once.
2. **Measured against the real console background, not the window.** eSim's
   consoles sit on ``surface`` (``#0E1728`` dark / ``#FFFFFF`` light) or on the
   slightly deeper simulation card (``#08111F`` / ``#FBFDFF``). Every value
   below clears WCAG AA (4.5:1) on both of its theme's backgrounds;
   ``audit_harness/verify_ui_s3.py`` re-computes that and fails if a retint
   breaks it.

Where a token is used verbatim it is named in the comment. Three light values
deliberately leave the LIGHT token set, for the same reason ``STYLE_LIGHT``'s
InfoBar does: ``success #059669`` and ``warning #D97706`` are tuned to sit
*next to* light body text as accents, and land at 3.8:1 and 3.2:1 when they
*are* the body text on white; ``text_muted #6B7F99`` reaches only 4.1:1, which
is the ~4.0:1 the whole light theme accepts for muted chrome but not enough for
a log line someone has to read. Each takes the next step down its own ramp.
"""
from frontEnd import tokens

# Semantic role → what it marks in a console
#
#   info    tool chatter: paths, progress, "loading…"     (console body tone)
#   detail  low-priority diagnostics, one tier dimmer
#   ok      outcomes that mean "you're good"
#   warn    cancellations, notes, hints
#   error   failures, stderr
#   head    section headings / phase banners
#   output  raw tool stdout, at full contrast
_DARK = {
    "info":   tokens.DARK["text_dim"],      # #D3DEEF — the console's own colour
    "detail": tokens.DARK["text_muted"],    # #94A8C3
    "ok":     tokens.DARK["success"],       # #42E6A4
    "warn":   tokens.DARK["warning"],       # #FACC15
    "error":  tokens.DARK["danger"],        # #FB7185
    "head":   tokens.DARK["accent"],        # #53D7FF
    "output": tokens.DARK["text"],          # #F8FBFF
}

_LIGHT = {
    "info":   tokens.LIGHT["text_dim"],     # #405168 — the console's own colour
    "detail": "#5A6E89",                    # the light sheet's darker grey;
                                            # LIGHT.text_muted is only 4.1:1
    "ok":     "#047857",                    # emerald-700; LIGHT.success is 3.8:1
    "warn":   "#B45309",                    # amber-700;   LIGHT.warning is 3.2:1
    "error":  tokens.LIGHT["danger"],       # #E11D48
    "head":   tokens.LIGHT["accent"],       # #0077A8
    "output": tokens.LIGHT["text"],         # #142033
}

# Banner sizing. The old status banners were 26px / 25pt — four times body
# size, which reads as shouting rather than hierarchy. One step of emphasis is
# enough when the colour already carries the meaning.
BANNER_PX = 16
BANNER_WEIGHT = 700
# Qt's rich text clamps font-weight at 900 (Black); several call sites asked
# for 1000, which silently clamped while implying a weight that does not exist.
HEAVY_WEIGHT = 800


def console_colors(is_dark: bool) -> dict:
    """Semantic console palette for one theme. Pure — no Qt, no I/O."""
    return dict(_DARK if is_dark else _LIGHT)


def console_is_dark(app=None) -> bool:
    """Resolve the active theme for console output.

    ``apply_theme`` installs a QPalette whose Window colour *is* the resolved
    eSim theme (not the OS one), so the palette is the authoritative signal and
    matches what every other painter in the tree reads. Falls back to the flag
    ``apply_theme`` records, then to light. Never raises: the loggers that call
    this run on worker threads and in headless tests where there is no
    QApplication at all.
    """
    try:
        from PyQt6 import QtGui, QtWidgets
        app = app or QtWidgets.QApplication.instance()
        if app is not None:
            win = app.palette().color(QtGui.QPalette.ColorRole.Window)
            if win.isValid():
                return win.lightness() < 128
    except Exception:
        pass
    try:
        from frontEnd.theme_utils import current_theme_is_dark
        return bool(current_theme_is_dark())
    except Exception:
        return False


def current_console_colors(app=None) -> dict:
    """Semantic console palette for the theme that is live right now.

    Cheap enough to call per emitted line, which is what the HTML consoles do —
    they bake the colour into the document, so resolving late is the only way a
    line written after a theme flip comes out in the new theme.
    """
    return console_colors(console_is_dark(app))
