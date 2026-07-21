"""UI_AUDIT session S1 verifier — C2, C1 (_palette retint), 1.1, 1.2, 1.4, 1.5.

Runs offscreen; no eSim project or ngspice needed.

    QT_QPA_PLATFORM=offscreen python audit_harness/verify_ui_s1.py

Each check prints PASS/FAIL and the script exits non-zero if any fail.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from PyQt6 import QtWidgets                                        # noqa: E402

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

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


# ── C2: tokens.py agrees with the sheets it claims to be the source of ──
def c2_tokens_match_sheets():
    from frontEnd import tokens
    qss_dir = os.path.join(_SRC, "frontEnd")
    dark = open(os.path.join(qss_dir, "style_dark.qss"),
                encoding="utf-8").read().upper()
    light = open(os.path.join(qss_dir, "style_light.qss"),
                 encoding="utf-8").read().upper()

    # bg/text tiers must be values the sheet actually paints, not near-misses.
    for key in ("bg_sunken", "bg", "bg_raise", "surface", "text",
                "text_dim", "text_muted", "text_subtle", "accent", "stroke"):
        assert tokens.DARK[key].upper() in dark, \
            f"DARK[{key}]={tokens.DARK[key]} appears nowhere in style_dark.qss"
        assert tokens.LIGHT[key].upper() in light, \
            f"LIGHT[{key}]={tokens.LIGHT[key]} appears nowhere in style_light.qss"

    # The window backdrop is the one value theme_utils also pushes into
    # QPalette.Window — that pair drifting is what C2 was about.
    assert tokens.DARK["bg"] == "#050812", tokens.DARK["bg"]
    assert tokens.LIGHT["bg"] == "#F3F7FC", tokens.LIGHT["bg"]
    assert tokens.DARK["text"] == "#F8FBFF", tokens.DARK["text"]


def c2_palette_blocks_agree():
    """theme_utils' QPalette literals now equal the tokens they mirror."""
    src = open(os.path.join(_SRC, "frontEnd", "theme_utils.py"),
               encoding="utf-8").read()
    from frontEnd import tokens
    for hexv in (tokens.DARK["bg"], tokens.DARK["text"],
                 tokens.DARK["text_muted"], tokens.LIGHT["bg"],
                 tokens.LIGHT["text"], tokens.LIGHT["text_muted"]):
        assert hexv in src, f"{hexv} missing from theme_utils palette blocks"


# ── C1 (half): the plotting palette is Aurora, not Tailwind ─────────────
def c1_palette_is_aurora():
    from ngspiceSimulation import _palette
    from frontEnd import tokens
    for dark, defaults in ((True, _palette._DARK_DEFAULTS),
                           (False, _palette._LIGHT_DEFAULTS)):
        t = tokens.theme(dark)
        for pal_key, tok_key in (("text", "text"), ("text_muted", "text_muted"),
                                 ("primary", "accent"), ("bg", "bg"),
                                 ("surface", "surface"), ("border", "stroke")):
            assert defaults[pal_key].upper() == t[tok_key].upper(), \
                f"{'dark' if dark else 'light'} {pal_key} != tokens {tok_key}"
    # The Tailwind greys the audit called out are gone for good.
    for stale in ("#1F2937", "#6B7280", "#165982", "#9CA3AF", "#E5E7EB"):
        assert stale not in str(_palette._LIGHT_DEFAULTS), stale
        assert stale not in str(_palette._DARK_DEFAULTS), stale


def c1_matplotlib_overrides_resolve():
    from ngspiceSimulation import _palette
    rc = _palette.matplotlib_rc_overrides(_palette._DARK_DEFAULTS)
    assert rc["figure.facecolor"] == "#050812", rc["figure.facecolor"]
    assert rc["axes.facecolor"] == "#0E1728", rc["axes.facecolor"]
    rc = _palette.matplotlib_rc_overrides(_palette._LIGHT_DEFAULTS)
    assert rc["figure.facecolor"] == "#F3F7FC", rc["figure.facecolor"]
    assert rc["axes.facecolor"] == "#FFFFFF", rc["axes.facecolor"]


# ── 1.1: the delegate no longer emits an unparseable sheet ──────────────
def p11_delegate_has_no_local_sheet():
    from PyQt6 import QtWidgets
    from modelEditor.ModelEditor import ParameterEditDelegate

    assert not hasattr(ParameterEditDelegate, "_is_dark"), \
        "_is_dark survived; it only existed to feed the deleted sheet"
    # createEditor must be Qt's, not an override that re-adds a sheet.
    assert "createEditor" not in vars(ParameterEditDelegate), \
        "createEditor override is back"

    table = QtWidgets.QTableWidget(1, 1)
    table.setItem(0, 0, QtWidgets.QTableWidgetItem("1k"))
    delegate = ParameterEditDelegate(table)
    table.setItemDelegate(delegate)
    idx = table.model().index(0, 0)
    opt = QtWidgets.QStyleOptionViewItem()
    editor = delegate.createEditor(table, opt, idx)
    assert isinstance(editor, QtWidgets.QLineEdit), type(editor)
    assert editor.styleSheet() == "", repr(editor.styleSheet())
    editor.deleteLater()
    table.deleteLater()


# ── 1.2: TerminalUi ships no widget-level sheets ────────────────────────
def p12_terminalui_has_no_widget_sheets():
    ui = open(os.path.join(_SRC, "frontEnd", "TerminalUi.ui"),
              encoding="utf-8").read()
    assert "styleSheet" not in ui, "a styleSheet property is back in the .ui"
    assert "rgb(36, 31, 49)" not in ui and "rgb(54,158,225)" not in ui
    # The console still carries the objectName the app sheet targets.
    assert 'name="simulationConsole"' in ui


def p12_console_reads_the_app_sheet():
    from PyQt6 import QtWidgets, QtCore
    from frontEnd import theme_utils

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from frontEnd.TerminalUi import TerminalUi

    for qss_name, dark, want_bg in (("style_dark.qss", True, "#08111F"),
                                    ("style_light.qss", False, "#FBFDFF")):
        app.setStyleSheet(theme_utils.build_qss(
            qss_name, dark, "default", "system", "system", 100))
        term = TerminalUi(QtCore.QProcess(), [])
        try:
            assert term.simulationConsole.styleSheet() == "", \
                "console carries a widget-level sheet again"
            assert term.progressBar.styleSheet() == "", \
                "progress bar carries a widget-level sheet again"
            term.show()
            app.processEvents()
            got = term.simulationConsole.palette().color(
                term.simulationConsole.backgroundRole()).name().upper()
            assert got == want_bg.upper(), \
                f"{qss_name}: console background {got}, expected {want_bg}"
        finally:
            term.close()
            term.deleteLater()
    app.setStyleSheet("")


# ── 1.4: no legacy gray group boxes left in the converter screens ───────
def p14_no_gray_groupboxes():
    ktn = os.path.join(_SRC, "kicadtoNgspice")
    for fn in sorted(os.listdir(ktn)):
        if not fn.endswith(".py"):
            continue
        body = open(os.path.join(ktn, fn), encoding="utf-8").read()
        assert "1px solid gray" not in body, f"{fn} still styles a gray box"
        assert "setStyleSheet" not in body, f"{fn} re-applies a local sheet"


def p14_groupbox_rule_still_themes_them():
    from PyQt6 import QtWidgets
    from frontEnd import theme_utils
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    for qss_name, dark in (("style_dark.qss", True), ("style_light.qss", False)):
        app.setStyleSheet(theme_utils.build_qss(
            qss_name, dark, "default", "system", "system", 100))
        box = QtWidgets.QGroupBox("Analysis")
        box.setLayout(QtWidgets.QGridLayout())
        box.show()
        app.processEvents()
        # The global rule reserves the title strip via margin-top; a box with
        # no local sheet must still get non-zero contents margins from QSS.
        top = box.contentsMargins().top()
        assert top > 0, f"{qss_name}: group box has no themed title inset"
        box.close()
        box.deleteLater()
    app.setStyleSheet("")


# ── 1.5: exactly one rule for the hidden dock title-bar buttons ─────────
def p15_single_dock_button_rule():
    import re
    qss_dir = os.path.join(_SRC, "frontEnd")
    for name in ("style_dark.qss", "style_light.qss"):
        body = open(os.path.join(qss_dir, name), encoding="utf-8").read()
        hits = re.findall(r"QDockWidget::(?:close|float)-button", body)
        assert len(hits) == 2, \
            f"{name}: {len(hits)} dock-button selectors, expected the 2 in " \
            "the single hiding rule"
        assert "dock_fullscreen" not in body, \
            f"{name}: still assigns an image to a 0x0 button"


def p15_sheets_stayed_parallel():
    """Both sheets must have had the same block removed (audit 1.6 goal)."""
    import re
    qss_dir = os.path.join(_SRC, "frontEnd")

    def selectors(name):
        body = open(os.path.join(qss_dir, name), encoding="utf-8").read()
        body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
        return [s.strip() for s in re.findall(r"([^{}]+)\{", body)]

    dark, light = selectors("style_dark.qss"), selectors("style_light.qss")
    only_dark = [s for s in dark if s not in light and "QDockWidget" in s]
    only_light = [s for s in light if s not in dark and "QDockWidget" in s]
    assert not only_dark and not only_light, (only_dark, only_light)


for label, fn in [
    ("C2   tokens match the sheets",           c2_tokens_match_sheets),
    ("C2   theme_utils palette agrees",        c2_palette_blocks_agree),
    ("C1   plotting palette is Aurora",        c1_palette_is_aurora),
    ("C1   matplotlib rc overrides resolve",   c1_matplotlib_overrides_resolve),
    ("1.1  delegate emits no local sheet",     p11_delegate_has_no_local_sheet),
    ("1.2  .ui carries no widget sheets",      p12_terminalui_has_no_widget_sheets),
    ("1.2  console themed by app sheet",       p12_console_reads_the_app_sheet),
    ("1.4  no gray group boxes left",          p14_no_gray_groupboxes),
    ("1.4  global QGroupBox rule applies",     p14_groupbox_rule_still_themes_them),
    ("1.5  one dock-button rule per sheet",    p15_single_dock_button_rule),
    ("1.5  sheets stayed parallel",            p15_sheets_stayed_parallel),
]:
    check(label, fn)

failed = [r for r in RESULTS if not r[1]]
for label, ok, msg in RESULTS:
    print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"\n        {msg}" if msg else ""))
print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
sys.exit(1 if failed else 0)
