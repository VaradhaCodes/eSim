"""Single source of truth for the eSim 'Aurora' design system.

Nothing else in the codebase should hard-code a brand hex string. Python
painters import THEME/accent helpers from here; ``theme_utils`` uses the
accent values to propagate a custom accent through the whole stylesheet
(including the rgba() glows that the old token-replace step missed).

The .qss sheets are the visually-tuned truth: every value below is the one
the sheets (and the QPalette blocks in ``theme_utils.apply_theme``) actually
paint, so a painter reading a token lands on the same pixel as a rule reading
the literal. Receipts are in the comments; keep them in step when retuning a
sheet.

Background ladder, deepest first — both themes stack the same way:

    bg_sunken   the workspace floor the dock cards sit on
                (``QDockWidget`` / ``::title`` / ``#dockTitleBar``)
    bg          the window itself — ``QMainWindow``/``QDialog``/``QFrame``,
                and ``QPalette.Window``
    bg_raise    one notch above the window (chrome strips, e.g. #waveHeader)
    surface     a card/panel floating on the window
    surface_2   a raised row or a disabled control inside a card
    surface_3   the topmost tint (hover fills inside a card)

No external dependencies — pure stdlib + values.
"""

# ── Per-theme palette ────────────────────────────────────────────────
DARK = {
    "bg_sunken":     "#05070F",
    "bg":            "#050812",
    "bg_raise":      "#0A1020",
    "surface":       "#0E1728",
    "surface_2":     "#121E33",
    "surface_3":     "#17243B",
    "stroke":        "#1D2B45",
    "text":          "#F8FBFF",
    "text_dim":      "#D3DEEF",   # menubar/toolbutton/console body text
    "text_muted":    "#94A8C3",   # QLabel[cssClass="muted"], status bar
    "text_subtle":   "#5F728D",
    "text_invert":   "#03121C",
    "accent":        "#53D7FF",   # primary accent (cyan)
    "accent_hi":     "#8BEAFF",
    "accent_lo":     "#18A8D8",
    "accent_2":      "#9B7CFF",   # violet companion
    "accent_2_hi":   "#C4B5FD",
    "success":       "#42E6A4",
    "warning":       "#FACC15",
    "danger":        "#FB7185",
    "danger_lo":     "#E11D48",
    "sel_bg":        "#0E7490",
    "sel_fg":        "#FFFFFF",
    "shadow_rgb":    (0, 0, 0),
}

LIGHT = {
    "bg_sunken":     "#EEF3FA",
    "bg":            "#F3F7FC",
    "bg_raise":      "#F8FBFF",
    "surface":       "#FFFFFF",
    "surface_2":     "#F6F9FD",
    "surface_3":     "#FFFFFF",
    "stroke":        "#DCE6F1",
    "text":          "#142033",
    "text_dim":      "#405168",   # menubar/toolbutton/console body text
    "text_muted":    "#6B7F99",   # QLabel[cssClass="muted"], status bar
    "text_subtle":   "#9AAABE",
    "text_invert":   "#FFFFFF",
    "accent":        "#0077A8",
    "accent_hi":     "#00A4DC",
    "accent_lo":     "#005E86",
    "accent_2":      "#6D5DF6",
    "accent_2_hi":   "#8B7CF8",
    "success":       "#059669",
    "warning":       "#D97706",
    "danger":        "#E11D48",
    "danger_lo":     "#BE123C",
    "sel_bg":        "#0077A8",
    "sel_fg":        "#FFFFFF",
    # blue-grey tinted shadow so light-mode depth reads as soft ambient
    # occlusion instead of an invisible/muddy black.
    "shadow_rgb":    (27, 42, 65),
}

# The default accent's literal rgb, as authored in the .qss files. Used by
# theme_utils to find-and-recolor every rgba(...) glow when the user picks
# a custom accent.
DEFAULT_ACCENT_RGB = {
    "dark":  (83, 215, 255),    # #53D7FF
    "light": (0, 119, 168),     # #0077A8
}

# Shape & rhythm (theme-independent)
RADIUS = {"sm": 8, "md": 12, "lg": 16, "xl": 20, "pill": 999}
SPACE = {"xs": 4, "sm": 8, "md": 12, "lg": 16, "xl": 24, "xxl": 32}
DUR = {"fast": 130, "base": 180, "slow": 240}  # ms


def theme(is_dark: bool) -> dict:
    return DARK if is_dark else LIGHT


def hex_to_rgb(hexv: str):
    hexv = hexv.lstrip("#")
    return int(hexv[0:2], 16), int(hexv[2:4], 16), int(hexv[4:6], 16)
