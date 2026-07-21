"""UI_AUDIT session S2 verifier — 1.6, 2.7, 2.8, 2.2 (QSS half), 1.7.

Runs offscreen; no eSim project, ngspice or QScintilla needed.

    QT_QPA_PLATFORM=offscreen python audit_harness/verify_ui_s2.py

Each check prints PASS/FAIL and the script exits non-zero if any fail.
"""
import os
import re
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from PyQt6 import QtCore, QtWidgets                                 # noqa: E402

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

_QSS_DIR = os.path.join(_SRC, "frontEnd")
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


# ── shared QSS parsing ──────────────────────────────────────────────────
def _sheet(name):
    return open(os.path.join(_QSS_DIR, name), encoding="utf-8").read()


def _blocks(body):
    """[(selector, [(prop, value), ...]), ...] in source order, comments out."""
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    out = []
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", body):
        sel = " ".join(m.group(1).split())
        props = []
        for decl in m.group(2).split(";"):
            decl = decl.strip()
            if not decl:
                continue
            key, _, val = decl.partition(":")
            props.append((key.strip(), " ".join(val.split())))
        out.append((sel, props))
    return out


_COLORISH = re.compile(
    r"#[0-9A-Fa-f]{3,8}|rgba?\(|qlineargradient|transparent|url\(")


# ── 1.6: the two app sheets are structurally identical ──────────────────
def p16_structural_diff_is_empty():
    """The audit's stated goal: the diff returns nothing but palette values."""
    dark, light = _blocks(_sheet("style_dark.qss")), _blocks(_sheet("style_light.qss"))
    dsel = [s for s, _ in dark]
    lsel = [s for s, _ in light]

    assert dsel == lsel, (
        "selector sets/order diverged; only in dark: "
        f"{[s for s in dsel if s not in lsel]}; only in light: "
        f"{[s for s in lsel if s not in dsel]}")

    for (sel, dp), (_, lp) in zip(dark, light, strict=True):
        dk = [k for k, _ in dp]
        lk = [k for k, _ in lp]
        assert dk == lk, f"{sel}: dark keys {dk} vs light keys {lk}"
        # Metrics and font values must be theme-independent; only colors may
        # differ between the sheets.
        for (k, dv), (_, lv) in zip(dp, lp, strict=True):
            if dv == lv or _COLORISH.search(dv) or _COLORISH.search(lv):
                continue
            raise AssertionError(
                f"{sel} :: {k} differs but is not a color — "
                f"dark {dv!r} light {lv!r}")


def p16_italic_disabled_state_is_gone():
    """Light's italic :disabled + its `font-style: normal` counter-patch."""
    dark, light = _sheet("style_dark.qss"), _sheet("style_light.qss")
    assert dark.count("font-style") == light.count("font-style"), (
        f"font-style count still asymmetric: dark {dark.count('font-style')} "
        f"vs light {light.count('font-style')}")
    for name, body in (("dark", dark), ("light", light)):
        for sel, props in _blocks(body):
            if not sel.startswith("QPushButton"):
                continue
            keys = [k for k, _ in props]
            assert "font-style" not in keys, \
                f"{name}: {sel} still carries font-style"


def p16_dead_dark_only_rule_removed():
    assert 'heroGradient' not in _sheet("style_dark.qss"), \
        "the dark-only heroGradient rule is back"


# ── 2.7: three-step weight scale ────────────────────────────────────────
def p27_weight_scale_is_three_steps():
    for name in ("style_dark.qss", "style_light.qss"):
        weights = re.findall(r"font-weight:\s*(\d+)", _sheet(name))
        stray = sorted({w for w in weights if w not in ("600", "700", "800")})
        assert not stray, f"{name}: off-scale weights {stray}"


def p27_weight_distribution_mirrors():
    from collections import Counter
    d = Counter(re.findall(r"font-weight:\s*(\d+)", _sheet("style_dark.qss")))
    lt = Counter(re.findall(r"font-weight:\s*(\d+)", _sheet("style_light.qss")))
    assert d == lt, f"weight histograms differ: dark {dict(d)} light {dict(lt)}"


# ── 2.8: messageKind actually paints ────────────────────────────────────
_KINDS = {
    "error":    ("#FB7185", "#E11D48"),
    "warning":  ("#FACC15", "#D97706"),
    "info":     ("#53D7FF", "#0077A8"),
    "question": ("#9B7CFF", "#6D5DF6"),
}


def p28_selectors_exist_in_both_sheets():
    for name in ("style_dark.qss", "style_light.qss"):
        body = _sheet(name)
        for kind in _KINDS:
            sel = f'QMessageBox#esimMessageBox[messageKind="{kind}"]'
            assert sel in body, f"{name}: no rule for {sel}"


def p28_every_kind_dialogs_emits_is_styled():
    """The QSS must cover exactly the kinds _prepare_msg can set."""
    src = open(os.path.join(_SRC, "frontEnd", "dialogs.py"),
               encoding="utf-8").read()
    emitted = set(re.findall(r'_prepare_msg\([^,]+,\s*"([a-z]+)"\)', src))
    assert emitted == set(_KINDS), \
        f"dialogs.py emits {sorted(emitted)}, sheets style {sorted(_KINDS)}"


def p28_stripe_renders_in_both_themes():
    from frontEnd import dialogs, theme_utils
    for qss_name, dark in (("style_dark.qss", True), ("style_light.qss", False)):
        APP.setStyleSheet(theme_utils.build_qss(
            qss_name, dark, "default", "system", "system", 100))
        for kind, (dk, lt) in _KINDS.items():
            want = (dk if dark else lt).upper()
            msg = QtWidgets.QMessageBox()
            msg.setText("Severity spine check.")
            dialogs._prepare_msg(msg, kind)
            msg.show()
            APP.processEvents()
            img = msg.grab().toImage()
            row = img.height() // 2
            edge = [img.pixelColor(x, row).name().upper() for x in (0, 1, 2)]
            assert edge == [want] * 3, \
                f"{qss_name} {kind}: left edge {edge}, expected 3px of {want}"
            msg.close()
            msg.deleteLater()
    APP.setStyleSheet("")


# ── 2.2 (QSS half): one hover system, not two ───────────────────────────
def p22_no_qss_hover_on_welcome_card():
    for name in ("style_dark.qss", "style_light.qss"):
        sels = [s for s, _ in _blocks(_sheet(name)) if "welcomeCard" in s]
        assert sels == ["QFrame#welcomeCard"], \
            f"{name}: welcomeCard rules are {sels}, expected the resting one only"


def p22_painter_still_owns_hover():
    """Deleting the QSS half must not have left the card with no hover at all."""
    from browser.Welcome import ToolCard
    assert "paintEvent" in vars(ToolCard), "ToolCard lost its hover painter"
    assert hasattr(ToolCard, "hoverProgress"), "the hover animation property is gone"


# ── 1.7: the editor chrome is Aurora in BOTH themes ─────────────────────
def _editor_sheets():
    src = open(os.path.join(_SRC, "codeEditor", "EditorWindow.py"),
               encoding="utf-8").read()
    grab = lambda n: re.search(n + r'\s*=\s*"""(.*?)"""', src, re.S).group(1)  # noqa: E731
    return grab("STYLE_LIGHT"), grab("STYLE_DARK")


def p17_editor_sheets_are_mirrors():
    light, dark = (_blocks(b) for b in _editor_sheets())
    lsel = [s for s, _ in light]
    dsel = [s for s, _ in dark]
    assert lsel == dsel, (
        f"only light: {[s for s in lsel if s not in dsel]}; "
        f"only dark: {[s for s in dsel if s not in lsel]}")
    for (sel, lp), (_, dp) in zip(light, dark, strict=True):
        lk = [k for k, _ in lp]
        dk = [k for k, _ in dp]
        assert lk == dk, f"{sel}: light keys {lk} vs dark keys {dk}"


def p17_light_chrome_is_aurora():
    from frontEnd import tokens
    light, _ = _editor_sheets()
    # amber-800/900, the LIGHT warning ramp extended two steps darker so body
    # text clears 4.5:1 on the warm InfoBar tint; documented above the sheets.
    allowed = {v.upper() for v in tokens.LIGHT.values() if isinstance(v, str)}
    allowed |= {"#92400E", "#78350F"}
    stray = sorted({h.upper() for h in re.findall(r"#[0-9A-Fa-f]{6}", light)
                    if h.upper() not in allowed})
    assert not stray, f"non-Aurora hexes in STYLE_LIGHT: {stray}"

    triples = set(re.findall(r"rgba\((\d+,\d+,\d+)", light))
    want = {
        ",".join(str(c) for c in tokens.hex_to_rgb(tokens.LIGHT["accent"])),
        ",".join(str(c) for c in tokens.hex_to_rgb(tokens.LIGHT["danger"])),
        ",".join(str(c) for c in tokens.hex_to_rgb(tokens.LIGHT["warning"])),
        "255,255,255",
    }
    assert triples <= want, f"unexpected rgba bases in STYLE_LIGHT: {triples - want}"


def _relative_luminance(hexv):
    def channel(c):
        c /= 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    h = hexv.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def _contrast(fg, bg):
    a, b = _relative_luminance(fg), _relative_luminance(bg)
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)


def _over(rgba, base):
    r, g, b, alpha = rgba
    h = base.lstrip("#")
    br, bg, bb = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#%02X%02X%02X" % (
        round(alpha * r + (1 - alpha) * br),
        round(alpha * g + (1 - alpha) * bg),
        round(alpha * b + (1 - alpha) * bb))


def p17_light_tinted_surfaces_are_legible():
    """The values this sheet invents must clear WCAG AA on their own tint.

    Only the roles that sit on a *coloured wash* are pinned here — the muted
    text tier (`text_muted`) lands at ~4.0 across the whole light theme and is
    a token-level decision, not this sheet's to make.
    """
    from frontEnd import tokens
    warn = tokens.hex_to_rgb(tokens.LIGHT["warning"])
    danger = tokens.hex_to_rgb(tokens.LIGHT["danger"])
    editor_bg = tokens.LIGHT["bg"]

    info_bg = _over((*warn, 0.10), editor_bg)
    action_bg = _over((*warn, 0.16), info_bg)
    close_bg = _over((*danger, 0.14), tokens.LIGHT["surface"])

    for role, fg, bg in (
        ("infoBar title",   "#78350F", info_bg),
        ("infoBar message", "#92400E", info_bg),
        ("infoAction label", "#78350F", action_bg),
        ("infoClose glyph", "#92400E", info_bg),
        ("findClose hover", tokens.LIGHT["danger_lo"], close_bg),
    ):
        got = _contrast(fg, bg)
        assert got >= 4.5, f"{role}: {fg} on {bg} is only {got:.2f}:1"


def p17_github_palette_is_gone():
    light, _ = _editor_sheets()
    for stale in ("#F6F8FA", "#0366D6", "#D0D7DE", "#E1E4E8", "#57606A",
                  "#41484F", "#1F2328", "#E7ECF1", "#DDEBFB", "#9CC4F0",
                  "#E1604D", "#FCE5C0", "#C9A227", "#5C4405"):
        assert stale not in light.upper(), f"GitHub-era literal {stale} survives"


def p17_editor_window_paints_aurora():
    from codeEditor.EditorWindow import STYLE_DARK, STYLE_LIGHT
    from frontEnd import tokens
    for sheet, want in ((STYLE_LIGHT, tokens.LIGHT["bg"]),
                        (STYLE_DARK, tokens.DARK["bg"])):
        win = QtWidgets.QMainWindow()
        central = QtWidgets.QWidget()
        central.setObjectName("editorCentral")
        win.setCentralWidget(central)
        win.resize(320, 200)
        win.setStyleSheet(sheet)
        win.show()
        APP.processEvents()
        got = central.palette().color(
            central.backgroundRole()).name().upper()
        assert got == want.upper(), f"#editorCentral painted {got}, want {want}"
        win.close()
        win.deleteLater()


# ── every sheet touched this session still parses ───────────────────────
def all_sheets_parse_clean():
    from codeEditor.EditorWindow import STYLE_DARK, STYLE_LIGHT
    from frontEnd import theme_utils

    seen = []
    old = QtCore.qInstallMessageHandler(
        lambda mode, ctx, text: seen.append(text))
    try:
        probe = QtWidgets.QWidget()
        for sheet in (STYLE_LIGHT, STYLE_DARK):
            probe.setStyleSheet(sheet)
            APP.processEvents()
        probe.deleteLater()
        for qss_name, dark in (("style_dark.qss", True),
                               ("style_light.qss", False)):
            APP.setStyleSheet(theme_utils.build_qss(
                qss_name, dark, "default", "system", "system", 100))
            APP.processEvents()
        APP.setStyleSheet("")
    finally:
        QtCore.qInstallMessageHandler(old)
    bad = [t for t in seen if "stylesheet" in t.lower()]
    assert not bad, f"Qt rejected a sheet: {bad}"


for label, fn in [
    ("1.6  sheet structural diff is empty",   p16_structural_diff_is_empty),
    ("1.6  italic disabled state gone",       p16_italic_disabled_state_is_gone),
    ("1.6  dark-only dead rule removed",      p16_dead_dark_only_rule_removed),
    ("2.7  weights are 600/700/800 only",     p27_weight_scale_is_three_steps),
    ("2.7  weight histograms mirror",         p27_weight_distribution_mirrors),
    ("2.8  messageKind rules in both sheets", p28_selectors_exist_in_both_sheets),
    ("2.8  every emitted kind is styled",     p28_every_kind_dialogs_emits_is_styled),
    ("2.8  stripe renders in both themes",    p28_stripe_renders_in_both_themes),
    ("2.2  no QSS hover on welcomeCard",      p22_no_qss_hover_on_welcome_card),
    ("2.2  painter still owns hover",         p22_painter_still_owns_hover),
    ("1.7  editor sheets are mirrors",        p17_editor_sheets_are_mirrors),
    ("1.7  light chrome is Aurora-only",      p17_light_chrome_is_aurora),
    ("1.7  tinted surfaces clear WCAG AA",    p17_light_tinted_surfaces_are_legible),
    ("1.7  GitHub palette is gone",           p17_github_palette_is_gone),
    ("1.7  editor window paints Aurora bg",   p17_editor_window_paints_aurora),
    ("--   every touched sheet parses",       all_sheets_parse_clean),
]:
    check(label, fn)

failed = [r for r in RESULTS if not r[1]]
for label, ok, msg in RESULTS:
    print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"\n        {msg}" if msg else ""))
print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
sys.exit(1 if failed else 0)
