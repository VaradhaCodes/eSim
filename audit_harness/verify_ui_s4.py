"""UI_AUDIT session S4 verifier — 2.1, 2.3, 2.4, 2.5, 2.9, 2.2 (Python half),
C7, C5.

Runs offscreen; no eSim project, ngspice or QScintilla needed.

    QT_QPA_PLATFORM=offscreen python audit_harness/verify_ui_s4.py

Each check prints PASS/FAIL and the script exits non-zero if any fail.

Note on fonts: this box exposes no families to offscreen Qt, so every check
about the FACE asserts on the requested family string (which QFont/QSS both
preserve) or on resolver identity, never on what a rasteriser picked.
"""
import io
import os
import re
import sys
import tokenize

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
for _p in (_SRC, os.path.join(_SRC, "frontEnd")):
    # frontEnd/ too: Application.py imports its sibling `pathmagic` flat.
    if _p not in sys.path:
        sys.path.insert(0, _p)

from PyQt6 import QtCore, QtGui, QtWidgets                          # noqa: E402

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

from frontEnd import elevation, theme_utils, tokens                 # noqa: E402

RESULTS = []


def check(name, fn):
    try:
        fn()
    except AssertionError as exc:
        RESULTS.append((name, False, str(exc)))
    except Exception as exc:                       # noqa: BLE001
        RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
    else:
        RESULTS.append((name, True, ""))


def _src(rel):
    return open(os.path.join(_SRC, rel), encoding="utf-8").read()


def _code(rel):
    """File contents with comments and docstrings stripped.

    Every fix in this session replaced a literal with a token lookup and left a
    comment naming the literal it removed, so a naive grep for '#53D7FF' finds
    the explanation rather than the defect. Only executable text counts.
    """
    body = _src(rel)
    out = []
    prev_type = None
    try:
        for tok in tokenize.generate_tokens(io.StringIO(body).readline):
            if tok.type == tokenize.COMMENT:
                continue
            if (tok.type == tokenize.STRING
                    and prev_type in (None, tokenize.INDENT, tokenize.NEWLINE,
                                      tokenize.NL, tokenize.DEDENT)):
                continue          # docstring
            out.append(tok.string)
            prev_type = tok.type
    except tokenize.TokenError:
        return body
    return "\n".join(out)


# ── WCAG contrast, computed the same way S2/S3 did ──────────────────────
def _lin(c):
    c /= 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _lum(rgb):
    r, g, b = (_lin(v) for v in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _hex_rgb(h):
    return tokens.hex_to_rgb(h)


def _contrast(fg_hex, bg_hex):
    a, b = _lum(_hex_rgb(fg_hex)), _lum(_hex_rgb(bg_hex))
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def _over(fg_hex, alpha, bg_hex):
    """Composite `fg` at `alpha` over `bg`; returns a hex string."""
    f, b = _hex_rgb(fg_hex), _hex_rgb(bg_hex)
    return "#%02X%02X%02X" % tuple(
        int(round(f[i] * alpha + b[i] * (1 - alpha))) for i in range(3))


# ── theme plumbing ──────────────────────────────────────────────────────
def _set_theme(dark):
    """Put the application into `dark`/`light` the way apply_theme does, as far
    as anything reading QPalette.Window can tell."""
    pal = QtGui.QPalette(APP.palette())
    t = tokens.theme(dark)
    pal.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(t["bg"]))
    pal.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(t["accent"]))
    APP.setPalette(pal)
    APP.processEvents()


def _shadow(w):
    eff = w.graphicsEffect()
    assert isinstance(eff, QtWidgets.QGraphicsDropShadowEffect), \
        f"{w} carries {eff!r}, not a drop shadow"
    return eff


def _rgba(c):
    return (c.red(), c.green(), c.blue(), c.alpha())


# ════════════════════════════════════════════════════════════════════════
# 2.1 — the elevation system is actually used
# ════════════════════════════════════════════════════════════════════════
_SHADOW_OWNERS = {
    # the two modules allowed to construct a drop shadow from scratch
    "frontEnd/elevation.py", "frontEnd/motion.py",
}
_SHADOW_CONSUMERS = [
    "browser/Welcome.py", "frontEnd/tooltips.py", "codeEditor/EditorWindow.py",
    "frontEnd/ProjectExplorer.py", "maker/VerilogVerifier.py",
]


def p21_no_consumer_hand_rolls_a_shadow():
    for rel in _SHADOW_CONSUMERS:
        body = _code(rel)
        assert "QGraphicsDropShadowEffect(" not in body, \
            f"{rel} still builds its own drop shadow instead of elevating"


def p21_no_black_shadow_literals_remain():
    """The defect: every hand-rolled shadow was pure black, which at these
    alphas is invisible on the light theme's #F3F7FC window."""
    for rel in _SHADOW_CONSUMERS + ["frontEnd/motion.py"]:
        body = _code(rel)
        assert not re.search(r"QColor\(\s*0\s*,\s*0\s*,\s*0", body), \
            f"{rel} still paints a literal black shadow"


def p21_light_shadows_are_tinted_and_dark_ones_are_not():
    w = QtWidgets.QWidget()
    for dark in (True, False):
        _set_theme(dark)
        elevation.elevate(w, "e2")
        want_rgb = tokens.theme(dark)["shadow_rgb"]
        _, _, _, ad, al = elevation.spec("e2")
        got = _rgba(_shadow(w).color())
        assert got == (*want_rgb, ad if dark else al), \
            f"dark={dark}: shadow {got}, expected {(*want_rgb, ad if dark else al)}"
    # and the whole point: light is NOT black
    assert tokens.LIGHT["shadow_rgb"] != (0, 0, 0)


def p21_panel_and_popup_land_on_the_scale():
    from frontEnd import motion
    _set_theme(True)
    for fn, level in ((motion.apply_panel_depth, "e2"),
                      (motion.apply_popup_depth, "e4")):
        w = QtWidgets.QWidget()
        fn(w)
        blur, dx, dy, ad, _ = elevation.spec(level)
        eff = _shadow(w)
        assert eff.blurRadius() == blur and eff.yOffset() == dy, \
            f"{fn.__name__} is not {level}: blur {eff.blurRadius()}, y {eff.yOffset()}"
        assert eff.color().alpha() == ad, \
            f"{fn.__name__} alpha {eff.color().alpha()}, expected {ad}"
        assert w.property(elevation.LEVEL_PROP) == level


def p21_message_boxes_are_popup_depth():
    from frontEnd import dialogs
    _set_theme(True)
    msg = QtWidgets.QMessageBox()
    dialogs._prepare_msg(msg, "info")
    assert msg.property(elevation.LEVEL_PROP) == "e4", \
        "message boxes no longer sit at e4"


def p21_toolbars_keep_their_seam_offsets():
    """Blur/alpha/colour come off the scale; only the DIRECTION stays local —
    the left rail's shadow is aimed so it does not bleed up into the joint."""
    from frontEnd import motion
    _set_theme(False)
    win = QtWidgets.QMainWindow()
    bars = {}
    for name in ("topToolbar", "leftToolBar"):
        tb = QtWidgets.QToolBar(win)
        tb.setObjectName(name)
        win.addToolBar(tb)
        bars[name] = tb
    motion.apply_toolbar_depth(win)
    blur, _, _, _, al = elevation.spec("e3")
    for name, want_off in (("topToolbar", (0, 5)), ("leftToolBar", (4, 6))):
        eff = _shadow(bars[name])
        got = (int(eff.xOffset()), int(eff.yOffset()))
        assert got == want_off, f"{name} offset {got}, expected {want_off}"
        assert eff.blurRadius() == blur, \
            f"{name} blur {eff.blurRadius()} is off the e3 scale ({blur})"
        assert _rgba(eff.color()) == (*tokens.LIGHT["shadow_rgb"], al), \
            f"{name} is not painting the light theme's tint"


def p21_set_shadow_defaults_to_the_theme_tint():
    """The one-move fix: every remaining set_shadow caller inherits the tint."""
    from frontEnd import motion
    _set_theme(False)
    w = QtWidgets.QWidget()
    motion.set_shadow(w, blur=10, alpha=50)
    assert _rgba(_shadow(w).color())[:3] == tokens.LIGHT["shadow_rgb"], \
        "set_shadow still defaults to black"
    # an explicit colour still wins (apply_accent_glow depends on it)
    motion.set_shadow(w, blur=10, alpha=50, color="#FF00FF")
    assert _rgba(_shadow(w).color())[:3] == (255, 0, 255)


def p21_elevated_shadows_retint_on_a_theme_change():
    _set_theme(True)
    w = QtWidgets.QWidget()
    elevation.elevate(w, "e3")
    before = _rgba(_shadow(w).color())
    _set_theme(False)
    assert _rgba(_shadow(w).color()) == before, \
        "sanity: the colour is baked in, it cannot change on its own"
    assert elevation.retint(w) is True
    after = _rgba(_shadow(w).color())
    assert after != before and after[:3] == tokens.LIGHT["shadow_rgb"], \
        f"retint left {after}"
    # a shadow this module did not paint is left alone
    other = QtWidgets.QWidget()
    from frontEnd import motion
    motion.set_shadow(other, color="#123456")
    assert elevation.retint(other) is False


def p21_theme_sweep_retints_what_it_walks():
    """_refresh_graphics_effects already visits every widget of every window;
    the re-tint has to ride along or nothing on screen ever changes tone."""
    _set_theme(True)
    win = QtWidgets.QWidget()
    child = QtWidgets.QWidget(win)
    elevation.elevate(child, "e2")
    win.show()
    APP.processEvents()
    assert _rgba(_shadow(child).color())[:3] == tokens.DARK["shadow_rgb"]
    _set_theme(False)
    theme_utils._refresh_graphics_effects(APP)
    assert _rgba(_shadow(child).color())[:3] == tokens.LIGHT["shadow_rgb"], \
        "apply_theme's effect sweep does not re-tint elevated shadows"
    win.close()


def p21_tooltip_card_has_room_for_its_shadow():
    from frontEnd import tooltips
    blur, _, dy, _, _ = elevation.spec("e3")
    reach = blur / 2.0 + dy
    assert tooltips.AuroraToolTip._PAD >= reach, \
        (f"tooltip pad {tooltips.AuroraToolTip._PAD} < e3's reach {reach}: "
         "the shadow would be clipped square by the window edge")


def p21_tooltip_reelevates_on_show():
    """The tip window is built once and reused all session, so the only place
    it can pick up a new theme's tint is show_text."""
    body = _src("frontEnd/tooltips.py")
    show = body.split("def show_text", 1)[1].split("\n    def ", 1)[0]
    assert "elevate(" in show, "show_text never re-elevates the card"


# ════════════════════════════════════════════════════════════════════════
# 2.2 (Python half) + C5 — Welcome owns its hover, from tokens
# ════════════════════════════════════════════════════════════════════════
def p21_welcome_tiles_are_elevated_on_show():
    """The tiles ARE on the scale — just not from __init__.

    Elevating during construction asks an unshown, unpolished widget which
    theme it is in, and on this tree that palette read on this path tips a
    latent fault in the style repolish: a full-suite run then dies with an
    access violation inside apply_theme. Receipt: the SAME crash reproduces on
    an otherwise-pristine tree with nothing but an inert
    ``widget.palette().color(Window).lightness()`` added to the old
    ``_apply_tile_shadow``, so it is not this session's doing — but it is this
    session's problem, and the deferred path is the correct design anyway.
    """
    from browser import Welcome
    _set_theme(False)
    page = Welcome.Welcome()
    tiles = page.findChildren(Welcome.ToolCard) + \
        page.findChildren(Welcome.HeroBanner)
    assert len(tiles) >= 13, f"only found {len(tiles)} tiles"
    assert all(t.graphicsEffect() is None for t in tiles), \
        "a tile was elevated during construction"
    page.show()
    APP.processEvents()
    for t in tiles:
        want = "e3" if isinstance(t, Welcome.HeroBanner) else \
            Welcome.ToolCard.REST_LEVEL
        assert t.property(elevation.LEVEL_PROP) == want, \
            f"{type(t).__name__} is at {t.property(elevation.LEVEL_PROP)}, not {want}"
        assert _rgba(_shadow(t).color())[:3] == tokens.LIGHT["shadow_rgb"], \
            f"{type(t).__name__} is not carrying the light theme's tint"
    page.close()


def p22_welcome_has_no_colour_literals():
    body = _code("browser/Welcome.py")
    hexes = re.findall(r"#[0-9A-Fa-f]{6}\b", body)
    rgbs = re.findall(r"QColor\(\s*\d+\s*,\s*\d+\s*,\s*\d+", body)
    assert not hexes and not rgbs, \
        f"Welcome.py still hardcodes {hexes or rgbs}"


def p22_dead_gradientlabel_import_is_gone():
    body = _src("browser/Welcome.py")
    assert "GradientLabel" not in body, \
        "Welcome.py still imports GradientLabel, which it never instantiates"


def p22_hover_wash_differs_between_themes():
    """The finding: the wash was dark-theme cyan, painted on the light theme's
    white card too. Rendered and measured against the composite each theme's
    accent must produce over the SAME neutral base."""
    from browser import Welcome
    alpha = 42 / 255.0
    washed = {}
    for dark in (True, False):
        _set_theme(dark)
        card = Welcome.ToolCard("t", "", "d", "x", lambda *_: None)
        card.resize(240, 96)
        card.show()
        APP.processEvents()

        def px(progress, c=card):
            c.setHoverProgress(progress)
            APP.processEvents()
            return c.grab().toImage().pixelColor(120, 48).name().upper()

        # The backdrop is whatever this theme paints; the wash is measured as
        # the difference the hover makes ON it, so nothing has to be assumed.
        base, washed[dark] = px(0.0), px(1.0)
        want = _hex_rgb(_over(tokens.theme(dark)["accent"], alpha, base))
        got = _hex_rgb(washed[dark])
        off = max(abs(got[i] - want[i]) for i in range(3))
        assert off <= 3, \
            (f"dark={dark}: hover paints {washed[dark]} over {base}; this "
             f"theme's accent would give {want} (off by {off})")
        card.close()
    assert washed[True] != washed[False], \
        "the hover wash lands on the same pixel in both themes"


def p22_hero_orb_uses_both_accent_tokens():
    from browser import Welcome
    _set_theme(True)
    hero = Welcome.HeroBanner()
    hero.resize(600, 160)
    hero.show()
    APP.processEvents()
    img = hero.grab().toImage()
    corner = img.pixelColor(img.width() - 24, 18)
    assert corner.isValid() and corner.name() != "#000000", "hero orb did not paint"
    hero.close()


def pc5_mixin_moved_into_the_widget_module():
    from frontEnd import widgets
    assert hasattr(widgets, "HoverSurfaceMixin"), \
        "C5: the mixin did not move into widgets.py"
    assert "class HoverSurfaceMixin" not in _src("browser/Welcome.py"), \
        "C5: Welcome.py still defines its own copy"
    assert "HoverSurfaceMixin" in _src("browser/Welcome.py"), \
        "Welcome no longer uses the mixin at all"


def pc5_hover_lerps_from_the_resting_elevation():
    """The old copy faded from black-alpha-48 — a value that existed nowhere
    else. It must start from exactly what elevate() painted and end on accent."""
    from browser import Welcome
    _set_theme(False)
    card = Welcome.ToolCard("t", "", "d", "x", lambda *_: None)
    elevation.elevate(card, Welcome.ToolCard.REST_LEVEL)
    rest = _rgba(elevation.shadow_color(card, Welcome.ToolCard.REST_LEVEL))

    card.setHoverProgress(0.0)
    assert _rgba(_shadow(card).color()) == rest, \
        f"at rest the shadow is {_rgba(_shadow(card).color())}, not the e2 tint {rest}"

    card.setHoverProgress(1.0)
    glow = _rgba(card.hover_glow_color())
    assert _rgba(_shadow(card).color()) == glow, "full hover is not the accent glow"
    assert glow[:3] == _hex_rgb(tokens.LIGHT["accent"]), \
        "the light glow is not the light accent"

    card.setHoverProgress(0.5)
    mid = _rgba(_shadow(card).color())
    for i in range(4):
        lo, hi = sorted((rest[i], glow[i]))
        assert lo <= mid[i] <= hi, f"channel {i} left the rest->glow ramp"
    # and the card rises: its shadow tucks in as the hover completes
    card.setHoverProgress(0.0)
    y0 = _shadow(card).yOffset()
    card.setHoverProgress(1.0)
    assert _shadow(card).yOffset() < y0


# ════════════════════════════════════════════════════════════════════════
# 2.3 — FlowNavigator reads tokens instead of copying them
# ════════════════════════════════════════════════════════════════════════
_FLOW_ALLOWED_NON_TOKEN = {"#92400E"}     # documented amber-800, see below


def _pill_tokens_for(dark):
    from maker import FlowNavigator

    class _Probe(FlowNavigator.FlowNavigator):
        def __init__(self):                     # no Qt build, no NGHDL import
            pass

        def _is_dark(self):
            return dark

    return _Probe()._pill_tokens()


def p23_pill_values_are_all_real_tokens():
    for dark in (True, False):
        t = tokens.theme(dark)
        allowed = {v.upper() for v in t.values() if isinstance(v, str)}
        for key, val in _pill_tokens_for(dark).items():
            for h in re.findall(r"#[0-9A-Fa-f]{6}", val):
                assert h.upper() in allowed or h.upper() in _FLOW_ALLOWED_NON_TOKEN, \
                    f"dark={dark}: {key}={h} is not a token in this theme"


def p23_pill_rgba_is_built_from_token_rgb():
    for dark in (True, False):
        t = tokens.theme(dark)
        pills = _pill_tokens_for(dark)
        for key, token in (("stage_checked_bg", "accent"), ("reload_bg", "warning")):
            got = tuple(int(n) for n in re.findall(r"\d+", pills[key])[:3])
            assert got == tokens.hex_to_rgb(t[token]), \
                f"dark={dark}: {key} rgb {got} is not {token}"


def p23_source_carries_no_copied_palette():
    body = _code("maker/FlowNavigator.py")
    hexes = {h.upper() for h in re.findall(r"#[0-9A-Fa-f]{6}\b", body)}
    assert hexes <= _FLOW_ALLOWED_NON_TOKEN, \
        f"FlowNavigator still hand-copies {sorted(hexes - _FLOW_ALLOWED_NON_TOKEN)}"


def p23_pill_shape_is_unchanged():
    """_apply_pill_theme was deliberately not touched, so the dict must keep
    every key it reads."""
    body = _src("maker/FlowNavigator.py")
    used = set(re.findall(r"t\['([a-z_]+)'\]", body))
    for dark in (True, False):
        assert used <= set(_pill_tokens_for(dark)), \
            f"dark={dark}: missing keys {sorted(used - set(_pill_tokens_for(dark)))}"


def p23_reload_banner_text_is_legible():
    """The one non-token value, and the reason for it: the light bar puts text
    on a 10% warning wash over the strip, where #D97706 is 3.2:1."""
    for dark in (True, False):
        t = tokens.theme(dark)
        pills = _pill_tokens_for(dark)
        strip = pills["bar_bg"]
        wash = _over(t["warning"], 0.10, strip)
        ratio = _contrast(pills["reload_fg"], wash)
        assert ratio >= 4.5, \
            f"dark={dark}: reload text {pills['reload_fg']} on {wash} is {ratio:.2f}:1"


def p23_strip_sits_where_the_theme_wants_it():
    """Dark lifts the chrome strip off the window, light leaves it flat — the
    asymmetry S1 recorded. A 'fix' to bg_raise in light shifts the header."""
    assert _pill_tokens_for(True)["bar_bg"] == tokens.DARK["bg_raise"]
    assert _pill_tokens_for(False)["bar_bg"] == tokens.LIGHT["bg"]


# ════════════════════════════════════════════════════════════════════════
# 2.4 — the About surfaces read tokens
# ════════════════════════════════════════════════════════════════════════
_ABOUT_KEYS = {"dark", "page", "header", "chip", "chip_border", "title",
               "muted", "subtle", "accent", "link", "sep", "pill_bg", "pill_fg"}


def _about_for(dark):
    from frontEnd import dialogs
    host = QtWidgets.QWidget()
    pal = QtGui.QPalette(host.palette())
    pal.setColor(QtGui.QPalette.ColorRole.Window,
                 QtGui.QColor(tokens.theme(dark)["bg"]))
    host.setPalette(pal)
    return dialogs._about_palette(host)


def p24_about_keys_are_unchanged():
    """Both consumers index this dict by name; the finding was about where the
    values come from, not what the shape is."""
    for dark in (True, False):
        assert set(_about_for(dark)) == _ABOUT_KEYS, \
            f"dark={dark}: shape changed to {sorted(_about_for(dark))}"


def p24_about_values_are_all_real_tokens():
    for dark in (True, False):
        allowed = {v.upper() for v in tokens.theme(dark).values()
                   if isinstance(v, str)}
        for key, val in _about_for(dark).items():
            if not isinstance(val, str):
                continue
            for h in re.findall(r"#[0-9A-Fa-f]{6}", val):
                assert h.upper() in allowed, \
                    f"dark={dark}: {key}={h} is not a token in this theme"


def p24_about_rgba_is_built_from_token_rgb():
    for dark in (True, False):
        t, c = tokens.theme(dark), _about_for(dark)
        assert tuple(int(n) for n in re.findall(r"\d+", c["pill_bg"])[:3]) \
            == tokens.hex_to_rgb(t["accent"]), "pill tint is not the accent"
        for key in ("sep", "chip_border"):
            assert tuple(int(n) for n in re.findall(r"\d+", c[key])[:3]) \
                == tokens.hex_to_rgb(t["text"]), f"{key} is not built from text"


def p24_source_carries_no_copied_palette():
    body = _code("frontEnd/dialogs.py")
    hexes = {h.upper() for h in re.findall(r"#[0-9A-Fa-f]{6}\b", body)}
    assert not hexes, f"dialogs.py still hand-copies {sorted(hexes)}"


def p24_about_text_is_legible_on_its_own_page():
    for dark in (True, False):
        c = _about_for(dark)
        for key, floor in (("title", 4.5), ("link", 4.5), ("muted", 4.0)):
            ratio = _contrast(c[key], c["page"])
            assert ratio >= floor, \
                f"dark={dark}: {key} {c[key]} on {c['page']} is {ratio:.2f}:1"


def p24_no_off_scale_font_weight():
    """S2 collapsed the sheets to 600/700/800; this file's inline sheets were
    not in that sweep and still asked for 650."""
    body = _src("frontEnd/dialogs.py")
    weights = {int(w) for w in re.findall(r"font-weight:\s*(\d+)", body)}
    assert weights <= {600, 700, 800}, f"dialogs.py uses weights {sorted(weights)}"


# ════════════════════════════════════════════════════════════════════════
# 2.5 — FullScreenToggle uses eSim's own icons
# ════════════════════════════════════════════════════════════════════════
def p25_no_platform_standard_pixmaps():
    body = _code("frontEnd/FullScreen.py")
    assert "SP_TitleBar" not in body and "standardIcon" not in body, \
        "FullScreen.py still asks the platform style for its glyphs"


def p25_both_states_carry_a_real_icon():
    from frontEnd.FullScreen import FullScreenToggle
    _set_theme(True)
    btn = FullScreenToggle()
    seen = {}
    for full in (False, True):
        btn._set_state(full=full)
        icon = btn.icon()
        assert not icon.isNull(), f"full={full}: no icon"
        img = icon.pixmap(64, 64).toImage()
        assert not img.isNull() and img.constBits() is not None
        seen[full] = img
        assert btn._full is full, "the state is not tracked for the re-tint"
    assert seen[False] != seen[True], "both states draw the same glyph"


def p25_icon_retints_on_a_palette_change():
    """icon_paths bakes the foreground into the raster, so a switch to light
    leaves a near-white glyph on a near-white toolbar without this."""
    from frontEnd.FullScreen import FullScreenToggle
    _set_theme(True)
    btn = FullScreenToggle()
    dark_img = btn.icon().pixmap(64, 64).toImage()
    _set_theme(False)
    QtWidgets.QApplication.sendEvent(
        btn, QtCore.QEvent(QtCore.QEvent.Type.PaletteChange))
    APP.processEvents()                       # the re-render is deferred
    light_img = btn.icon().pixmap(64, 64).toImage()
    assert dark_img != light_img, \
        "the icon did not re-render on PaletteChange"


def p25_retint_never_runs_inside_the_handler():
    """A setIcon() inside PaletteChange re-enters the polish that delivered it;
    with one toggle per docked panel that is a C-stack overflow, and it took
    the whole test suite down when this was first written."""
    from frontEnd.FullScreen import FullScreenToggle
    body = _src("frontEnd/FullScreen.py")
    handler = body.split("def changeEvent", 1)[1].split("\n    def ", 1)[0]
    assert "_set_state" not in handler, \
        "changeEvent re-renders synchronously — that is the recursion"
    assert "singleShot" in body, "the refresh is not deferred to the next tick"

    _set_theme(True)
    btn = FullScreenToggle()
    calls = []
    real = btn._set_state
    btn._set_state = lambda full, _r=real, _c=calls: (_c.append(full), _r(full))[1]
    for _ in range(5):                        # a real toggle emits a burst
        QtWidgets.QApplication.sendEvent(
            btn, QtCore.QEvent(QtCore.QEvent.Type.PaletteChange))
    assert calls == [], "a re-render happened synchronously"
    APP.processEvents()
    assert len(calls) == 1, f"the burst coalesced into {len(calls)} re-renders"


# ════════════════════════════════════════════════════════════════════════
# 2.9 — one font stack, and it actually reaches the widget
# ════════════════════════════════════════════════════════════════════════
_FONT_FILES = ["maker/VerilogVerifier.py", "codeEditor/PlainEditor.py",
               "maker/ToolchainCheck.py"]


def p29_no_platform_only_font_families():
    for rel in _FONT_FILES:
        body = _code(rel)
        for bad in ('"Segoe UI"', "'Segoe UI'", '"Consolas"', "'Consolas'",
                    '"Monospace"', "'Monospace'"):
            assert bad not in body, f"{rel} still names {bad} directly"


def p29_one_resolver_feeds_every_editor():
    from codeEditor import theme
    from frontEnd import widgets
    fake = {"Cascadia Code", "Cascadia Mono", "JetBrains Mono", "Consolas"}
    real_families = QtGui.QFontDatabase.families
    real_cache = widgets._MONO_CACHE
    try:
        QtGui.QFontDatabase.families = staticmethod(lambda *a, **k: sorted(fake))
        widgets._MONO_CACHE = None
        want = widgets.mono_family()
        assert want == "JetBrains Mono", \
            f"mono_family picked {want}; the sheets declare JetBrains Mono first"
        assert theme.editor_font().family() == want, \
            "the editor resolves a different family than the rest of the app"
    finally:
        QtGui.QFontDatabase.families = real_families
        widgets._MONO_CACHE = real_cache


def p29_missing_font_db_is_not_cached():
    """Before a QGuiApplication exists families() is empty; caching that would
    pin every later caller to the generic."""
    from frontEnd import widgets
    real_families = QtGui.QFontDatabase.families
    real_cache = widgets._MONO_CACHE
    try:
        QtGui.QFontDatabase.families = staticmethod(lambda *a, **k: [])
        widgets._MONO_CACHE = None
        assert widgets.mono_family() == "monospace"
        assert widgets._MONO_CACHE is None, "the empty result was cached"
    finally:
        QtGui.QFontDatabase.families = real_families
        widgets._MONO_CACHE = real_cache


def p29_sheet_owns_the_verifier_console_font():
    """The deleted setFont was dead weight, and this is the receipt: with an
    app sheet installed the SHEET wins, so the console has been rendering the
    sheet's stack all along and the Python font only misled readers."""
    APP.setStyleSheet(theme_utils.build_qss(
        "style_dark.qss", True, "default", "system", "system", 100))
    try:
        console = QtWidgets.QTextEdit()
        console.setObjectName("verilogConsole")
        console.show()
        APP.processEvents()
        f = console.font()
        assert f.family() == "JetBrains Mono", \
            f"#verilogConsole resolved {f.family()!r}, not the sheet's stack"
        assert f.pixelSize() == 12, f"size {f.pixelSize()}px is not the sheet's"
        console.close()

        label = QtWidgets.QLabel("Module Hierarchy")
        label.setObjectName("verilogSidebarTitle")
        label.show()
        APP.processEvents()
        lf = label.font()
        assert lf.pixelSize() == 11 and lf.weight() == 800, \
            f"#verilogSidebarTitle resolved {lf.pixelSize()}px/{lf.weight()}"
        label.close()
    finally:
        APP.setStyleSheet("")


def p29_mono_survives_the_app_sheet_where_it_must():
    """The live half of the finding. A QSS-styled text view CANNOT hold a mono
    face through setFont — the app sheet's QWidget font rule beats it — so the
    toolchain report and the fallback editor were rendering in Inter."""
    from codeEditor import theme
    APP.setStyleSheet(theme_utils.build_qss(
        "style_dark.qss", True, "default", "system", "system", 100))
    try:
        want = theme.editor_font(11)
        loser = QtWidgets.QPlainTextEdit()
        loser.setFont(want)
        loser.show()
        APP.processEvents()
        assert loser.font().family() != want.family(), \
            "premise check failed: setFont held, so the local sheet is pointless"

        winner = QtWidgets.QPlainTextEdit()
        winner.setFont(want)
        winner.setStyleSheet(theme.mono_font_css(want))
        winner.show()
        APP.processEvents()
        assert winner.font().family() == want.family(), \
            f"the local sheet lost too: {winner.font().family()!r}"
        winner.close()
        loser.close()
    finally:
        APP.setStyleSheet("")


def p29_both_live_sites_carry_that_sheet():
    for rel in ("codeEditor/PlainEditor.py", "maker/ToolchainCheck.py"):
        assert "mono_font_css" in _code(rel), \
            f"{rel} sets a mono font that the app sheet will overrule"


def p29_fallback_editor_renders_mono():
    import tempfile
    from codeEditor import theme
    from codeEditor.PlainEditor import PlainEditor
    APP.setStyleSheet(theme_utils.build_qss(
        "style_dark.qss", True, "default", "system", "system", 100))
    fd, path = tempfile.mkstemp(suffix=".cir")
    os.close(fd)
    try:
        open(path, "w", encoding="utf-8").write("* netlist\nV1 1 0 5\n")
        ed = PlainEditor(path)
        ed.show()
        APP.processEvents()
        assert ed.font().family() == theme.editor_font().family(), \
            f"fallback editor renders in {ed.font().family()!r}"
        ed.close()
    finally:
        APP.setStyleSheet("")
        os.unlink(path)


# ════════════════════════════════════════════════════════════════════════
# C7 — the About dialog tracks its content instead of clipping it
# ════════════════════════════════════════════════════════════════════════
_ABOUT_HOSTS = []      # keeps the parents alive; a dead parent takes the dialog


def _built_about(zoom):
    """Build the real About dialog and capture it instead of exec()ing."""
    from frontEnd import dialogs
    captured = {}
    real_exec = QtWidgets.QDialog.exec

    def fake_exec(self):
        captured["dlg"] = self
        return 0

    host = QtWidgets.QWidget()
    _ABOUT_HOSTS.append(host)
    APP.setStyleSheet(theme_utils.build_qss(
        "style_dark.qss", True, "default", "system", "system", zoom))
    QtWidgets.QDialog.exec = fake_exec
    try:
        dialogs.show_about_dialog(host)
    finally:
        QtWidgets.QDialog.exec = real_exec
        APP.setStyleSheet("")
    dlg = captured["dlg"]
    _ABOUT_HOSTS.append(dlg)
    return dlg


def pc7_about_is_no_longer_a_fixed_frame():
    dlg = _built_about(100)
    assert (dlg.minimumWidth(), dlg.minimumHeight()) == (440, 500), \
        "the floor moved; 440x500 was the shipped size"
    assert dlg.maximumHeight() > 500, \
        "the dialog is still pinned to a fixed height"


def pc7_about_grows_with_the_zoom_level():
    small = _built_about(100)
    large = _built_about(200)
    assert large.sizeHint().height() > small.sizeHint().height(), \
        "sanity: a 200% sheet must ask for more room"
    assert large.height() >= large.sizeHint().height(), \
        (f"at 200% the frame is {large.height()}px for "
         f"{large.sizeHint().height()}px of content — the credits clip")


for label, fn in [
    ("2.1  no consumer builds its own shadow", p21_no_consumer_hand_rolls_a_shadow),
    ("2.1  no black shadow literals left",   p21_no_black_shadow_literals_remain),
    ("2.1  light tinted / dark not",
     p21_light_shadows_are_tinted_and_dark_ones_are_not),
    ("2.1  panel=e2, popup=e4",              p21_panel_and_popup_land_on_the_scale),
    ("2.1  message boxes sit at e4",         p21_message_boxes_are_popup_depth),
    ("2.1  toolbars keep the seam offsets",  p21_toolbars_keep_their_seam_offsets),
    ("2.1  set_shadow defaults to the tint",
     p21_set_shadow_defaults_to_the_theme_tint),
    ("2.1  shadows re-tint on theme change",
     p21_elevated_shadows_retint_on_a_theme_change),
    ("2.1  the theme sweep drives it",       p21_theme_sweep_retints_what_it_walks),
    ("2.1  tooltip pad clears e3's reach",   p21_tooltip_card_has_room_for_its_shadow),
    ("2.1  tooltip re-elevates on show",     p21_tooltip_reelevates_on_show),
    ("2.1  Welcome tiles elevate on show",   p21_welcome_tiles_are_elevated_on_show),
    ("2.2  Welcome has no colour literals",  p22_welcome_has_no_colour_literals),
    ("2.2  dead GradientLabel import gone",  p22_dead_gradientlabel_import_is_gone),
    ("2.2  hover wash differs per theme",    p22_hover_wash_differs_between_themes),
    ("2.2  hero orb still paints",           p22_hero_orb_uses_both_accent_tokens),
    ("C5   mixin lives in widgets.py",       pc5_mixin_moved_into_the_widget_module),
    ("C5   hover lerps from e2 to accent",
     pc5_hover_lerps_from_the_resting_elevation),
    ("2.3  pill values are real tokens",     p23_pill_values_are_all_real_tokens),
    ("2.3  pill rgba built from token rgb",  p23_pill_rgba_is_built_from_token_rgb),
    ("2.3  no copied palette in source",     p23_source_carries_no_copied_palette),
    ("2.3  dict shape unchanged",            p23_pill_shape_is_unchanged),
    ("2.3  reload banner text is legible",   p23_reload_banner_text_is_legible),
    ("2.3  strip height is per-theme",       p23_strip_sits_where_the_theme_wants_it),
    ("2.4  About keys unchanged",            p24_about_keys_are_unchanged),
    ("2.4  About values are real tokens",    p24_about_values_are_all_real_tokens),
    ("2.4  About rgba from token rgb",       p24_about_rgba_is_built_from_token_rgb),
    ("2.4  no copied palette in dialogs",    p24_source_carries_no_copied_palette),
    ("2.4  About text is legible",           p24_about_text_is_legible_on_its_own_page),
    ("2.4  no off-scale font weight",        p24_no_off_scale_font_weight),
    ("2.5  no platform standard pixmaps",    p25_no_platform_standard_pixmaps),
    ("2.5  both states carry an icon",       p25_both_states_carry_a_real_icon),
    ("2.5  icon re-tints on PaletteChange",  p25_icon_retints_on_a_palette_change),
    ("2.5  re-tint is deferred, not nested",
     p25_retint_never_runs_inside_the_handler),
    ("2.9  no platform-only families",       p29_no_platform_only_font_families),
    ("2.9  one resolver for every editor",   p29_one_resolver_feeds_every_editor),
    ("2.9  empty font DB is not cached",     p29_missing_font_db_is_not_cached),
    ("2.9  sheet owns the console font",     p29_sheet_owns_the_verifier_console_font),
    ("2.9  mono survives the app sheet",
     p29_mono_survives_the_app_sheet_where_it_must),
    ("2.9  both live sites carry the sheet", p29_both_live_sites_carry_that_sheet),
    ("2.9  fallback editor renders mono",    p29_fallback_editor_renders_mono),
    ("C7   About is not a fixed frame",      pc7_about_is_no_longer_a_fixed_frame),
    ("C7   About grows with the zoom",       pc7_about_grows_with_the_zoom_level),
]:
    check(label, fn)

failed = [r for r in RESULTS if not r[1]]
for label, ok, msg in RESULTS:
    print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"\n        {msg}" if msg else ""))
print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
sys.exit(1 if failed else 0)
