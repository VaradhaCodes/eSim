"""The plotting palette must not drift away from the Aurora design tokens.

``ngspiceSimulation/_palette.py`` deliberately never imports ``frontEnd`` — it
is a leaf helper so the plotting tree stays importable headless. The price is
a hand-copied palette, and before this test the copy had rotted into a second
design language (Tailwind greys ``#1F2937``/``#6B7280``/``#165982`` next to an
app painted in Aurora), which is only visible to someone with both windows
open side by side.

The plotting module keeps its own richer vocabulary (axes/legend/cursor roles
that ``tokens.py`` has no opinion about) — only the keys that genuinely name
the same thing are pinned here. A retint of either side now fails a test
instead of rotting quietly.
"""
import pytest

from frontEnd import tokens
from ngspiceSimulation import _palette


# plotting-palette key -> tokens key. Same role, same pixel, both themes.
SHARED_KEYS = {
    "bg":              "bg",
    "surface":         "surface",
    "panel":           "surface",
    "panel_alt":       "surface_2",
    "border":          "stroke",
    "divider":         "stroke",
    "text":            "text",
    "text_muted":      "text_muted",
    "text_subtle":     "text_subtle",
    "primary":         "accent",
    "primary_hover":   "accent_hi",
    "primary_pressed": "accent_lo",
    "label_color":     "text",
    "tick_color":      "text_muted",
    "axes_face":       "surface",
    "legend_face":     "surface",
    "axes_edge":       "stroke",
    "grid_color":      "stroke",
    "legend_edge":     "stroke",
    "stats_text":      "text_dim",
}


@pytest.mark.parametrize("dark", [True, False], ids=["dark", "light"])
@pytest.mark.parametrize("pal_key,tok_key", sorted(SHARED_KEYS.items()))
def test_shared_key_matches_tokens(dark, pal_key, tok_key):
    defaults = _palette._DARK_DEFAULTS if dark else _palette._LIGHT_DEFAULTS
    expected = tokens.theme(dark)[tok_key]
    assert defaults[pal_key].upper() == expected.upper(), (
        f"{'dark' if dark else 'light'} palette['{pal_key}'] drifted from "
        f"tokens['{tok_key}']"
    )


@pytest.mark.parametrize("dark", [True, False], ids=["dark", "light"])
def test_both_themes_define_the_same_keys(dark):
    # A key added to one theme only is the other half of the drift problem:
    # current_palette() would KeyError (or silently fall back) in one theme.
    assert set(_palette._DARK_DEFAULTS) == set(_palette._LIGHT_DEFAULTS)
    assert _palette._DARK_DEFAULTS["is_dark"] is True
    assert _palette._LIGHT_DEFAULTS["is_dark"] is False


def test_no_stray_named_colors():
    """Every non-``is_dark`` value is a hex or an rgba() string.

    Guards against a well-meaning ``"gray"`` / ``"white"`` creeping back in:
    matplotlib accepts those, Qt stylesheets accept those, and neither is a
    palette value anyone can retint.
    """
    for name, defaults in (("dark", _palette._DARK_DEFAULTS),
                           ("light", _palette._LIGHT_DEFAULTS)):
        for key, value in defaults.items():
            if key == "is_dark":
                continue
            assert isinstance(value, str), f"{name}['{key}'] is not a string"
            assert value.startswith("#") or value.startswith("rgba("), (
                f"{name}['{key}'] = {value!r} is not a hex or rgba() color")
