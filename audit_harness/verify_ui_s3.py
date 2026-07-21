"""UI_AUDIT session S3 verifier — 1.9, 1.3, 1.8, 1.10, 2.6.

Runs offscreen; no eSim project, ngspice, matplotlib canvas or QScintilla
needed. The plotting checks exercise the mixins directly against a stub host
carrying a real palette, so nothing has to load a .raw file.

    QT_QPA_PLATFORM=offscreen python audit_harness/verify_ui_s3.py

Each check prints PASS/FAIL and the script exits non-zero if any fail.
"""
import os
import re
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
for _p in (_SRC, os.path.join(_SRC, "frontEnd")):
    # frontEnd/ too: Application.py imports its sibling `pathmagic` flat.
    if _p not in sys.path:
        sys.path.insert(0, _p)

from PyQt6 import QtGui, QtWidgets                                  # noqa: E402

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


def _src(rel):
    return open(os.path.join(_SRC, rel), encoding="utf-8").read()


def _code(rel):
    """File contents with comments and docstrings stripped.

    Every fix in this session replaced a literal with a token lookup and left a
    comment naming the literal it removed, so a naive grep for '#00FF00' finds
    the explanation rather than the defect. Only executable text counts.
    """
    import io
    import tokenize
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
            if tok.type not in (tokenize.NL, tokenize.NEWLINE):
                prev_type = tok.type
            else:
                prev_type = tok.type
    except tokenize.TokenError:
        return body
    return "\n".join(out)


# ── WCAG contrast, computed the same way S2 did ─────────────────────────
def _lin(c):
    c /= 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _lum(hexv):
    r, g, b = (int(hexv.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def _contrast(fg, bg):
    a, b = _lum(fg), _lum(bg)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


# ── 1.9: one console palette, measured, and actually adopted ────────────
# The backgrounds each console really paints on, from the two app sheets.
_CONSOLE_BACKDROPS = {
    True:  ["#0E1728",   # QTextEdit / verilogBottomContainer::pane (maker)
            "#08111F"],  # QPlainTextEdit#simulationConsole
    False: ["#FFFFFF",   # QTextEdit (maker)
            "#FBFDFF"],  # QPlainTextEdit#simulationConsole
}


def p19_console_colors_clear_wcag_aa():
    """Every semantic colour is legible on every console it can land on."""
    from frontEnd.console_colors import console_colors
    worst = []
    for dark, backdrops in _CONSOLE_BACKDROPS.items():
        colors = console_colors(dark)
        for level, hexv in colors.items():
            for bg in backdrops:
                ratio = _contrast(hexv, bg)
                if ratio < 4.5:
                    worst.append(
                        f"{'dark' if dark else 'light'}.{level} {hexv} on "
                        f"{bg} = {ratio:.2f}:1")
    assert not worst, "below WCAG AA: " + "; ".join(worst)


def p19_both_themes_define_the_same_levels():
    from frontEnd.console_colors import console_colors
    d, li = console_colors(True), console_colors(False)
    assert set(d) == set(li), f"level sets differ: {set(d) ^ set(li)}"
    same = [k for k in d if d[k].upper() == li[k].upper()]
    assert not same, (
        f"{same} is the same colour in both themes — either it is theme-"
        "independent (say so) or one side was not retuned")


def p19_no_console_keeps_a_hardcoded_literal():
    """The six files the finding names carry no colour literals any more."""
    targets = {
        "maker/VerilogVerifier.py":       ["#57606A", "#1A7F37", "#CF222E",
                                           "#9A6700", "#0969DA", "#24292E"],
        "maker/CosimLogger.py":           ["#00AA00", "#E07B00", "#FF0000",
                                           "#666666", "#B30086", "#000000"],
        "ngspiceSimulation/NgspiceWidget.py": ["#00ff00", "#ff3333"],
        "frontEnd/TerminalUi.py":         ["#FF8624"],
        "maker/NgVeri.py":                ["#00FF00", "#00AA00", "#FF0000"],
        "maker/ModelGeneration.py":       ["#ff0000", "#008000", "#0000FF"],
    }
    bad = []
    for rel, literals in targets.items():
        code = _code(rel).upper()
        bad += [f"{rel}:{lit}" for lit in literals if lit.upper() in code]
    assert not bad, "still hardcoded: " + ", ".join(bad)


def p19_no_console_asks_for_an_impossible_weight():
    """Qt clamps rich-text font-weight at 900; 1000 implied a weight that does
    not exist and silently rendered as Black."""
    bad = []
    for rel in ("maker/VerilogVerifier.py", "maker/CosimLogger.py",
                "maker/NgVeri.py", "maker/ModelGeneration.py",
                "frontEnd/TerminalUi.py",
                "ngspiceSimulation/NgspiceWidget.py"):
        for m in re.finditer(r"font-weight:\s*(\d+)", _src(rel)):
            if int(m.group(1)) > 900:
                bad.append(f"{rel}:{m.group(1)}")
    assert not bad, "weight > 900: " + ", ".join(bad)


def p19_banners_are_no_longer_shouting():
    from frontEnd.console_colors import BANNER_PX
    assert 14 <= BANNER_PX <= 16, BANNER_PX
    bad = []
    for rel in ("frontEnd/TerminalUi.py", "ngspiceSimulation/NgspiceWidget.py",
                "maker/NgVeri.py", "maker/ModelGeneration.py"):
        for m in re.finditer(r"font-size:\s*(\d+)(px|pt)", _src(rel)):
            size_px = int(m.group(1)) * (1 if m.group(2) == "px" else 4 / 3)
            if size_px > 20:
                bad.append(f"{rel}:{m.group(0)}")
    assert not bad, "banner still oversized: " + ", ".join(bad)


def p19_emitted_html_tracks_the_live_theme():
    """The real regression: a line written after a theme flip must come out in
    the NEW theme. Colours are baked into the document, so this only works if
    the value is resolved at emit time, not at import time."""
    from frontEnd.console_colors import console_colors
    from maker.CosimLogger import CosimLog

    out = []
    log = CosimLog(sink=out.append)
    seen = {}
    for dark in (True, False):
        pal = QtGui.QPalette()
        pal.setColor(QtGui.QPalette.ColorRole.Window,
                     QtGui.QColor("#050812" if dark else "#F3F7FC"))
        APP.setPalette(pal)
        out.clear()
        log.error("boom")
        seen[dark] = out[0]
        want = console_colors(dark)["error"]
        assert want in out[0], (
            f"{'dark' if dark else 'light'} emit used {out[0]!r}, want {want}")
    assert seen[True] != seen[False], "same HTML in both themes"


def p19_verifier_recolours_its_backlog():
    """A theme toggle must repaint lines already in the console, otherwise a
    mid-session flip leaves a scrollback of the old theme's colours."""
    from frontEnd.console_colors import console_colors
    from maker import VerilogVerifier as VV

    console = QtWidgets.QTextEdit()

    class _Host:
        _LOG_COLORS = VV.VerilogVerifier._LOG_COLORS
        _append_console = VV.VerilogVerifier._append_console
        _retheme_console = VV.VerilogVerifier._retheme_console

    host = _Host()
    host.console = console

    def _set(dark):
        pal = QtGui.QPalette()
        pal.setColor(QtGui.QPalette.ColorRole.Window,
                     QtGui.QColor("#050812" if dark else "#F3F7FC"))
        APP.setPalette(pal)

    levels = sorted(console_colors(True))
    _set(True)
    for level in levels:
        host._append_console(f"line {level}", host._LOG_COLORS[level])
    dark_seen = _fragment_colors(console)
    assert dark_seen == {v.upper() for v in console_colors(True).values()}, \
        f"dark console wrote {dark_seen}"

    _set(False)
    host._retheme_console()
    light_seen = _fragment_colors(console)
    assert light_seen == {v.upper() for v in console_colors(False).values()}, \
        f"after retheme console holds {light_seen}"
    assert console.toPlainText().count("line ") == len(levels), \
        "text was disturbed"
    console.deleteLater()


def _fragment_colors(edit):
    doc = edit.document()
    seen = set()
    block = doc.begin()
    while block.isValid():
        it = block.begin()
        while not it.atEnd():
            frag = it.fragment()
            it += 1
            if frag.isValid() and frag.length():
                seen.add(frag.charFormat().foreground().color().name().upper())
        block = block.next()
    return seen


# ── 1.3: the simulation status dot follows the theme ────────────────────
def p13_status_dot_uses_the_active_theme():
    from frontEnd import tokens
    from frontEnd.Application import Application

    dot_sheets = {}
    for dark in (True, False):
        import frontEnd.theme_utils as tu
        tu._CURRENT_DARK = dark
        host = type("H", (), {
            "_set_sim_status": Application._set_sim_status,
            "_retint_sim_status": Application._retint_sim_status,
        })()
        host.sim_status_dot = QtWidgets.QLabel("●")
        for state, key in (("idle", "text_subtle"), ("running", "warning"),
                           ("ok", "success"), ("failed", "danger")):
            host._set_sim_status(state)
            want = tokens.theme(dark)[key]
            got = host.sim_status_dot.styleSheet()
            assert want in got, (
                f"{'dark' if dark else 'light'} {state}: sheet {got!r} "
                f"lacks {want}")
        dot_sheets[dark] = host.sim_status_dot.styleSheet()
        host.sim_status_dot.deleteLater()
    assert dot_sheets[True] != dot_sheets[False], (
        "the dot paints identically in both themes — the light branch is dead")


def p13_status_dot_retints_on_theme_change():
    """The dot carries a widget-level sheet, which re-styling cannot reach; the
    only way back is re-running _set_sim_status with the state on screen."""
    from frontEnd import tokens
    from frontEnd.Application import Application
    import frontEnd.theme_utils as tu

    host = type("H", (), {
        "_set_sim_status": Application._set_sim_status,
        "_retint_sim_status": Application._retint_sim_status,
    })()
    host.sim_status_dot = QtWidgets.QLabel("●")

    tu._CURRENT_DARK = True
    host._set_sim_status("ok")
    assert tokens.DARK["success"] in host.sim_status_dot.styleSheet()

    tu._CURRENT_DARK = False
    host._retint_sim_status()
    sheet = host.sim_status_dot.styleSheet()
    assert tokens.LIGHT["success"] in sheet, (
        f"still dark after retint: {sheet!r}")
    assert tokens.DARK["success"] not in sheet
    host.sim_status_dot.deleteLater()


def p13_theme_apply_calls_the_retint():
    """The hook has to be wired, not merely present."""
    body = _src("frontEnd/theme_utils.py")
    assert "_retint_sim_status" in body, (
        "apply_theme never calls _retint_sim_status — the dot would only "
        "re-tint on the next simulation")


# ── plotting: a stub host that owns a real palette ──────────────────────
def _plot_host(dark):
    """Minimal object carrying the mixin methods and a real _palette."""
    from ngspiceSimulation import _palette as pal_mod
    from ngspiceSimulation._cursor_mixin import _CursorMixin
    from ngspiceSimulation._list_mixin import _ListMixin
    from ngspiceSimulation._func_trace_mixin import _FuncTraceMixin

    class _Host(_CursorMixin, _ListMixin, _FuncTraceMixin):
        def __init__(self):
            self._palette = dict(pal_mod._DARK_DEFAULTS if dark
                                 else pal_mod._LIGHT_DEFAULTS)
            self.color_palette = ["#E53935", "#1E88E5", "#43A047"]
            self.cursor_positions = []
            self._func_traces = []
            self._func_visible = []
            self.traces = {}
    return _Host()


# ── 1.8: trace-colour popups are themed ─────────────────────────────────
def p18_color_menus_have_no_light_only_literals():
    for rel in ("ngspiceSimulation/_list_mixin.py",
                "ngspiceSimulation/_func_trace_mixin.py"):
        code = _code(rel)
        for lit in ("#FFFFFF", "#E0E0E0", "#212121", "#757575", "#9E9E9E"):
            assert lit not in code.upper(), f"{rel} still hardcodes {lit}"


def p18_popup_chrome_differs_between_themes():
    from PyQt6.QtWidgets import QMenu
    seen = {}
    for dark in (True, False):
        host = _plot_host(dark)
        menu = QMenu()
        host.populate_color_menu(menu, [])
        widget = menu.actions()[0].defaultWidget()
        sheet = widget.styleSheet()
        assert host._palette["panel"] in sheet, sheet
        swatch = widget.findChildren(QtWidgets.QPushButton)[0]
        assert host._palette["border_strong"] in swatch.styleSheet()
        assert host._palette["text"] in swatch.styleSheet()
        seen[dark] = sheet
        menu.deleteLater()
    assert seen[True] != seen[False], "popup paints the same in both themes"


def p18_hidden_rows_use_the_dim_tier():
    """A hidden trace is dimmed, not erased: the old #757575/#9E9E9E pair was
    tuned for a white list and read as noise on the dark panel."""
    from ngspiceSimulation.trace import Trace
    for dark in (True, False):
        host = _plot_host(dark)
        host.waveform_list = QtWidgets.QListWidget()
        host.traces[0] = Trace(index=0, name="v(out)", color="#E53935",
                               thickness=1.0, style="-")
        host.traces[0].visible = False
        item = QtWidgets.QListWidgetItem()
        item.setData(0x0100, 0)          # Qt.ItemDataRole.UserRole
        host.waveform_list.addItem(item)
        host.update_list_item_appearance(item, 0)
        row = host.waveform_list.itemWidget(item)
        label = row.findChildren(QtWidgets.QLabel)[1]
        assert host._palette["cursor_dim"] in label.styleSheet(), (
            f"{'dark' if dark else 'light'} hidden row: {label.styleSheet()!r}")
        ratio = _contrast(host._palette["cursor_dim"],
                          host._palette["panel"])
        assert ratio >= 3.0, (
            f"hidden row at {ratio:.2f}:1 on the panel is not dim, it is gone")
        host.waveform_list.deleteLater()


# ── 1.10 / 2.6: cursor readouts and painted chrome ──────────────────────
def p110_cursor_html_has_no_hardcoded_greys():
    code = _code("ngspiceSimulation/_cursor_mixin.py")
    for lit in ("#333", "#555", "#999", "#aaa", "'red'", "'blue'"):
        assert lit not in code, f"_cursor_mixin still uses {lit}"
    win = _code("ngspiceSimulation/plot_window.py")
    for lit in ("#e53935", "#1976d2", "#e65100", "#aaa", "#444444"):
        assert lit not in win, f"plot_window still uses {lit}"


def p110_readouts_are_legible_on_the_plot_panel():
    """The finding in one number: #333 on the dark panel was 1.4:1."""
    from ngspiceSimulation import _palette as pal_mod
    for defaults in (pal_mod._DARK_DEFAULTS, pal_mod._LIGHT_DEFAULTS):
        bg = defaults["panel"]
        for key, floor in (("stats_text", 4.5), ("cursor_dim", 4.0)):
            ratio = _contrast(defaults[key], bg)
            assert ratio >= floor, (
                f"{'dark' if defaults['is_dark'] else 'light'} {key} "
                f"{defaults[key]} on {bg} = {ratio:.2f}:1 (want {floor})")


def p110_no_readout_paints_text_with_a_border_tone():
    """cursor_disabled reads as a text role but holds border_strong (1.75:1 on
    the dark panel). Nothing in the readout or the trace list may use it."""
    for rel in ("ngspiceSimulation/_cursor_mixin.py",
                "ngspiceSimulation/_list_mixin.py",
                "ngspiceSimulation/plot_window.py"):
        code = _code(rel)
        for key in ("cursor_disabled", "cursor_chrome"):
            assert key not in code, (
                f"{rel} paints text with {key} — see the reasoning in "
                "_CursorMixin's docstring")


def p110_readout_html_is_built_from_the_palette():
    for dark in (True, False):
        host = _plot_host(dark)
        p = host._palette
        head = host._cursor_head_html(0, "1.234 ms")
        assert p["cursor1"] in head and p["cursor_dim"] in head \
            and p["stats_text"] in head, head
        delta = host._delta_html(1.0, 1.0, "ms")
        assert p["cursor_delta"] in delta and p["stats_text"] in delta, delta
        placeholder = host._cursor_placeholder_html(1)
        assert p["cursor2"] in placeholder \
            and p["cursor_dim"] in placeholder, placeholder


def p110_readouts_differ_between_themes():
    d, li = _plot_host(True), _plot_host(False)
    assert d._cursor_head_html(0, "x") != li._cursor_head_html(0, "x")
    assert d._delta_html(1, 1, "s") != li._delta_html(1, 1, "s")
    assert d._cursor_placeholder_html(None) != li._cursor_placeholder_html(None)


def p26_cursor_line_matches_its_own_readout():
    """The drawn axvline and the label announcing it are one colour. They were
    'red'/'blue' vs #e53935/#1976d2 — two different reds for one cursor."""
    for dark in (True, False):
        host = _plot_host(dark)
        for n in (0, 1):
            hue = host._cursor_hue(n)
            assert hue == host._palette["cursor1" if n == 0 else "cursor2"]
            assert hue in host._cursor_head_html(n, "1")


def p26_plot_window_retints_what_it_paints():
    """_apply_theme_impl has to drive the hand-painted surfaces; a stylesheet
    swap reaches none of them."""
    body = _src("ngspiceSimulation/plot_window.py")
    assert "_retint_painted_chrome" in body
    impl = body.split("def _apply_theme_impl", 1)[1]
    assert "_retint_painted_chrome()" in impl.split("def _retint", 1)[0], (
        "_apply_theme_impl never calls the re-tint")
    for name in ("refresh_list_theme", "retint_cursor_readouts",
                 "_make_focus_icon"):
        assert name in body, f"{name} not wired into the re-tint"


def p26_matplotlib_chrome_is_themed():
    code = _code("ngspiceSimulation/_render_mixin.py")
    for lit in ("'white'", "#E0E0E0", "#757575", "#444444", "#BDBDBD"):
        assert lit not in code, f"_render_mixin still hardcodes {lit}"
    for key in ("legend_face", "legend_edge", "stats_text",
                "spine_separator", "text_muted"):
        assert key in code, f"{key} never adopted"


def p26_retint_survives_a_bare_host():
    """_retint_painted_chrome runs from __init__ too, before the later widgets
    exist; each step is guarded so a partial window cannot raise."""
    from ngspiceSimulation.plot_window import plotWindow

    class _Bare:
        _retint_painted_chrome = plotWindow._retint_painted_chrome

        def refresh_list_theme(self):
            raise RuntimeError("no list yet")

        def retint_cursor_readouts(self):
            raise RuntimeError("no labels yet")

    _Bare()._retint_painted_chrome()          # must not raise


for label, fn in [
    ("1.9  console colours clear WCAG AA",   p19_console_colors_clear_wcag_aa),
    ("1.9  both themes define same levels",  p19_both_themes_define_the_same_levels),
    ("1.9  no console keeps a literal",      p19_no_console_keeps_a_hardcoded_literal),
    ("1.9  no font-weight above 900",
     p19_no_console_asks_for_an_impossible_weight),
    ("1.9  banners are not shouting",        p19_banners_are_no_longer_shouting),
    ("1.9  emitted HTML tracks the theme",   p19_emitted_html_tracks_the_live_theme),
    ("1.9  verifier recolours its backlog",  p19_verifier_recolours_its_backlog),
    ("1.3  status dot uses active theme",    p13_status_dot_uses_the_active_theme),
    ("1.3  status dot retints on toggle",    p13_status_dot_retints_on_theme_change),
    ("1.3  apply_theme calls the retint",    p13_theme_apply_calls_the_retint),
    ("1.8  colour menus have no literals",
     p18_color_menus_have_no_light_only_literals),
    ("1.8  popup chrome is per-theme",       p18_popup_chrome_differs_between_themes),
    ("1.8  hidden rows use the dim tier",    p18_hidden_rows_use_the_dim_tier),
    ("1.10 cursor HTML has no greys",        p110_cursor_html_has_no_hardcoded_greys),
    ("1.10 readouts legible on the panel",
     p110_readouts_are_legible_on_the_plot_panel),
    ("1.10 no text in a stroke-only tone",
     p110_no_readout_paints_text_with_a_border_tone),
    ("1.10 readout HTML from the palette",
     p110_readout_html_is_built_from_the_palette),
    ("1.10 readouts differ between themes",  p110_readouts_differ_between_themes),
    ("2.6  cursor line matches its readout", p26_cursor_line_matches_its_own_readout),
    ("2.6  plot window retints its paint",   p26_plot_window_retints_what_it_paints),
    ("2.6  matplotlib chrome is themed",     p26_matplotlib_chrome_is_themed),
    ("2.6  retint survives a bare host",     p26_retint_survives_a_bare_host),
]:
    check(label, fn)

failed = [r for r in RESULTS if not r[1]]
for label, ok, msg in RESULTS:
    print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"\n        {msg}" if msg else ""))
print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
sys.exit(1 if failed else 0)
