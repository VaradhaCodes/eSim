"""UI_AUDIT session S5 verifier — 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, C3.

Runs offscreen; no eSim project, ngspice or QScintilla needed.

    QT_QPA_PLATFORM=offscreen python audit_harness/verify_ui_s5.py

Each check prints PASS/FAIL and the script exits non-zero if any fail.

This session is deletion, so most checks assert an absence — and an absence is
the easiest thing in the world to assert vacuously. Three of them are therefore
written as CLOSED LOOPS over the whole tree rather than as greps for the names
the audit happened to list:

  * every objectName / attribute selector in either sheet must have a setter in
    some .py/.ui file (this is what found the 40th dead rule, #verilogNoWaveform,
    which the audit's table missed);
  * every image file must be referenced AND every referenced image must exist;
  * every public name in widgets.py and every icon factory must have a consumer.

A future dead rule fails the guard on the day it is written, which is the point
of doing P3 once.
"""
import ast
import os
import re
import sys
import glob

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
_FE = os.path.join(_SRC, "frontEnd")
_IMG = os.path.join(_FE, "images")
for _p in (_SRC, _FE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from PyQt6 import QtCore, QtGui, QtWidgets                          # noqa: E402

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


# ── sources ─────────────────────────────────────────────────────────────
def _src(rel):
    return open(os.path.join(_SRC, rel), encoding="utf-8").read()


def _sheet(name):
    return open(os.path.join(_FE, name), encoding="utf-8").read()


SHEETS = {n: _sheet(n) for n in ("style_dark.qss", "style_light.qss")}


def _uncommented(body):
    return re.sub(r"/\*.*?\*/", "", body, flags=re.S)


def _blocks(body):
    """[(selector, [(prop, value), ...]), ...] in source order, comments out."""
    out = []
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", _uncommented(body)):
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


def _selector_text(body):
    """Only the selector halves — so `#F8FBFF` in a declaration is not read as
    an objectName, which is the trap in scanning a stylesheet with a regex."""
    return " , ".join(s for s, _ in _blocks(body))


_TREE_SOURCE = None


def _tree_source():
    """Every .py and .ui in src/, concatenated. Setters are string literals."""
    global _TREE_SOURCE
    if _TREE_SOURCE is None:
        buf = []
        for pat in ("**/*.py", "**/*.ui"):
            for p in glob.glob(os.path.join(_SRC, pat), recursive=True):
                buf.append(open(p, encoding="utf-8", errors="ignore").read())
        _TREE_SOURCE = "\n".join(buf)
    return _TREE_SOURCE


def _literal_in_tree(value):
    src = _tree_source()
    return f'"{value}"' in src or f"'{value}'" in src


def _fn_node(rel, name):
    for node in ast.walk(ast.parse(_src(rel))):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == name:
            return node
    raise AssertionError(f"{rel}: no function named {name}")


# ── 3.1 dead QSS rule groups ────────────────────────────────────────────
DEAD_SELECTOR_TOKENS = [
    'cssClass="labeledIcon"', 'cssClass="gradientTitle"', "heroGradient",
    'cssClass="error"', 'cssClass="warning"', 'cssClass="success"',
    'cssClass="swatch"', 'cssClass="colorPick"', 'cssClass="previewStrip"',
    "preferencesPreviewStrip", "themePreview", "preferencesTabs",
    "editorFindBar", "spiceEditorToolbar", "spiceEditorStatus",
    "verilogCodeEditor", "dockPopButton", "isPoppedOut", "isCloseBtn",
    "dockFloatHost", "dockTitleBar", "dockCardTitle", "dockDropOverlay",
    "dockToolChrome", "verilogNoWaveform",
]


def p31_dead_selector_groups_are_gone():
    """Every group the audit's table named, in BOTH sheets."""
    for name, body in SHEETS.items():
        for tok in DEAD_SELECTOR_TOKENS:
            assert tok not in body, f"{name} still carries {tok}"


def p31_every_objectname_selector_has_a_setter():
    """Closed loop: an `#name` rule with no setObjectName("name") is dead.

    This is the guard the audit's table was a hand-written approximation of.
    """
    body = " , ".join(_selector_text(b) for b in SHEETS.values())
    names = sorted(set(re.findall(r"#([A-Za-z_][A-Za-z0-9_]*)", body)))
    assert names, "selector scan found nothing — the parser broke"
    orphans = [n for n in names if not _literal_in_tree(n)]
    assert not orphans, f"objectName rules with no setter: {orphans}"


def p31_every_attribute_selector_has_a_setter():
    """Same loop for `[prop="value"]` rules, minus the ones Qt owns itself."""
    qt_owned = {"true", "false"}          # e.g. [flat="true"] on a Qt subcontrol
    body = " , ".join(_selector_text(b) for b in SHEETS.values())
    pairs = set(re.findall(r'\[([A-Za-z_][A-Za-z0-9_]*)="([^"]+)"\]', body))
    assert pairs, "attribute-selector scan found nothing"
    orphans = sorted({f"[{k}={v}]" for k, v in pairs
                      if v not in qt_owned and not _literal_in_tree(v)})
    assert not orphans, f"attribute rules with no setter: {orphans}"


def p31_structural_diff_is_still_empty():
    """S2's success criterion has to survive a session of deletions: the two
    sheets must have been trimmed in mirrored ranges."""
    dark, light = _blocks(SHEETS["style_dark.qss"]), _blocks(SHEETS["style_light.qss"])
    dsel = [s for s, _ in dark]
    lsel = [s for s, _ in light]
    assert dsel == lsel, (
        "selector order diverged; only in dark: "
        f"{[s for s in dsel if s not in lsel]}; only in light: "
        f"{[s for s in lsel if s not in dsel]}")
    for (sel, dp), (_, lp) in zip(dark, light, strict=True):
        dk = [k for k, _ in dp]
        lk = [k for k, _ in lp]
        assert dk == lk, f"{sel}: dark keys {dk} vs light keys {lk}"


def p31_both_sheets_still_parse():
    """Qt itself, not a regex: a mis-balanced brace from a cut would show here."""
    from frontEnd import theme_utils
    seen = []
    old = QtCore.qInstallMessageHandler(
        lambda mode, ctx, text: seen.append(text))
    try:
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


def p31_live_dock_card_rule_survived():
    """The deletion ran through the dock section; #dockCard must NOT be in it."""
    for name, body in SHEETS.items():
        sels = [s for s, _ in _blocks(body)]
        assert "QFrame#dockCard" in sels, f"{name} lost the live dock card rule"
    assert 'setObjectName("dockCard")' in _src("frontEnd/DockArea.py"), \
        "DockArea no longer sets the objectName the surviving rule targets"


# ── 3.2 widgets.py ──────────────────────────────────────────────────────
DEAD_WIDGET_NAMES = ["GradientLabel", "AuroraHeroFrame", "RailDragGrip",
                     "DockTitleBar", "DockDropOverlay", "FloatingDockHost",
                     "_is_wayland"]


def p32_dead_widget_classes_are_gone():
    body = _src("frontEnd/widgets.py")
    for n in DEAD_WIDGET_NAMES:
        assert f"class {n}" not in body and f"def {n}" not in body, \
            f"widgets.py still defines {n}"
    # and nothing anywhere still names them in code (docstrings may, and do)
    for n in DEAD_WIDGET_NAMES:
        hits = [p for p in glob.glob(os.path.join(_SRC, "**/*.py"), recursive=True)
                if re.search(rf"^\s*(from|import).*\b{n}\b",
                             open(p, encoding="utf-8", errors="ignore").read(),
                             re.M)]
        assert not hits, f"{n} is still imported by {hits}"


def p32_module_exports_only_live_names():
    """Closed loop: every public top-level name in widgets.py has a consumer."""
    tree = ast.parse(_src("frontEnd/widgets.py"))
    public = [n.name for n in tree.body
              if isinstance(n, (ast.FunctionDef, ast.ClassDef))
              and not n.name.startswith("_")]
    assert public, "widgets.py exports nothing — the parse broke"
    others = ""
    for p in glob.glob(os.path.join(_SRC, "**/*.py"), recursive=True):
        if os.path.basename(p) == "widgets.py":
            continue
        others += open(p, encoding="utf-8", errors="ignore").read()
    dead = [n for n in public if n not in others]
    assert not dead, f"widgets.py exports with no consumer: {dead}"


def p32_live_widget_api_still_works():
    """The three survivors are not just present, they still function."""
    from frontEnd import widgets, tokens

    fam = widgets.mono_family()
    assert isinstance(fam, str) and fam, "mono_family returned nothing"

    probe = QtWidgets.QWidget()
    pal = probe.palette()
    pal.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#050812"))
    probe.setPalette(pal)
    got = widgets.accent_color(probe, 128)
    assert got.name().upper() == tokens.DARK["accent"], \
        f"accent_color resolved {got.name()} on a dark palette"
    assert got.alpha() == 128

    class Card(widgets.HoverSurfaceMixin, QtWidgets.QFrame):
        def __init__(self):
            super().__init__()
            self._init_hover_anim()

    c = Card()
    c.setHoverProgress(1.0)
    assert c.getHoverProgress() == 1.0, "hover progress no longer round-trips"


def p32_wayland_findings_survived_the_deletion():
    """The dock-drag code went; its four hard-won findings had to stay."""
    doc = _src("frontEnd/DockArea.py")
    for needle in ("startSystemMove", "Wayland", "setFloating",
                   "singleShot", "QDrag", "widgets.py"):
        assert needle in doc, \
            f"DockArea lost the dock-drag write-up (missing {needle!r})"
    body = _src("frontEnd/widgets.py")
    assert "git log" in body, \
        "widgets.py does not point at where the deleted code lives"


# ── 3.3 motion.py ───────────────────────────────────────────────────────
def p33_dead_installers_are_gone():
    body = _src("frontEnd/motion.py")
    for n in ("install_popup_motion", "install_effect_refresh",
              "install_menu_depth"):
        assert f"def {n}" not in body, f"motion.py still defines {n}"


def p33_revalidate_survives_and_is_used():
    """The audit's one carve-out: keep the class for _revalidate's sake."""
    from frontEnd import motion
    assert hasattr(motion.EffectShowRefreshFilter, "_revalidate"), \
        "the helper the merged filter calls is gone"
    src = _src("frontEnd/motion.py")
    assert "EffectShowRefreshFilter._revalidate" in src, \
        "AppWideMotionFilter no longer calls the helper it kept alive"
    assert "def eventFilter" not in \
        src[src.index("class EffectShowRefreshFilter"):
            src.index("class AppWideMotionFilter")], \
        "the un-installable eventFilter is still there"


def p33_dockpopbutton_is_gone_tree_wide():
    for p in glob.glob(os.path.join(_SRC, "**/*.py"), recursive=True) + \
            glob.glob(os.path.join(_FE, "*.qss")):
        body = open(p, encoding="utf-8", errors="ignore").read()
        code = "\n".join(ln for ln in body.splitlines()
                         if not ln.strip().startswith("#"))
        assert "dockPopButton" not in code, f"{os.path.basename(p)} still reads it"


def p33_the_property_no_longer_suppresses_a_glow():
    """Behavioural: with the branch gone, dockPopButton must be inert.

    Before, this exact button rested at alpha 0 because the property was
    consulted; a default button rests lit.
    """
    from frontEnd import motion
    btn = QtWidgets.QPushButton("go")
    btn.setDefault(True)
    lit = motion.rest_alpha(btn)
    assert lit > 0, "a default button should rest lit"
    btn.setProperty("dockPopButton", True)
    assert motion.rest_alpha(btn) == lit, \
        "the dead property still changes the resting alpha"
    btn.setProperty("noMotion", True)
    assert motion.rest_alpha(btn) == 0, "noMotion (the live opt-out) stopped working"


def p33_plain_buttons_still_take_the_glow_path():
    """The `if not is_dock:` guards wrapped every branch — removing them must
    leave the branches themselves intact."""
    from frontEnd import motion
    filt = motion.TactileButtonFilter()
    btn = QtWidgets.QPushButton("go")
    btn.show()
    filt.eventFilter(btn, QtCore.QEvent(QtCore.QEvent.Type.Enter))
    assert btn in filt._glow_anims, "Enter no longer starts a glow animation"
    assert isinstance(btn.graphicsEffect(),
                      QtWidgets.QGraphicsDropShadowEffect), \
        "Enter no longer builds the drop shadow"
    filt.stop_all_glow()
    btn.deleteLater()


# ── 3.4 icon_paths.py ───────────────────────────────────────────────────
DEAD_ICONS = {"backup_icon": "_BACKUP_SVG", "close_proj_icon": "_CLOSE_PROJ_SVG",
              "copy_icon": "_COPY_SVG", "close_icon": "_CLOSE_SVG"}


def p34_dead_icon_factories_are_gone():
    body = _src("frontEnd/icon_paths.py")
    for fn, const in DEAD_ICONS.items():
        assert f"def {fn}" not in body, f"icon_paths.py still defines {fn}"
        assert const not in body, f"{const} was orphaned rather than deleted"


def p34_every_surviving_factory_has_a_consumer():
    """Closed loop, the same shape as the QSS one."""
    tree = ast.parse(_src("frontEnd/icon_paths.py"))
    factories = [n.name for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name.endswith("_icon")
                 and not n.name.startswith("_")]
    assert len(factories) >= 10, f"only found {factories}"
    others = ""
    for p in glob.glob(os.path.join(_SRC, "**/*.py"), recursive=True):
        if os.path.basename(p) == "icon_paths.py":
            continue
        others += open(p, encoding="utf-8", errors="ignore").read()
    dead = [f for f in factories if f not in others]
    assert not dead, f"icon factories with no consumer: {dead}"


def p34_every_surviving_factory_still_renders():
    """A deleted SVG constant that another factory shared would show up here as
    an empty icon, not as an ImportError."""
    from frontEnd import icon_paths
    tree = ast.parse(_src("frontEnd/icon_paths.py"))
    for n in tree.body:
        if not (isinstance(n, ast.FunctionDef) and n.name.endswith("_icon")
                and not n.name.startswith("_")):
            continue                       # _svg_icon is the builder, not one
        icon = getattr(icon_paths, n.name)()
        assert not icon.isNull(), f"{n.name}() produced a null icon"
        pm = icon.pixmap(16, 16)
        assert not pm.isNull() and pm.width() > 0, f"{n.name}() rasterised to nothing"


# ── 3.5 orphaned assets ─────────────────────────────────────────────────
DELETED_ASSETS = [
    "text_find_dark.svg", "text_find_light.svg",
    "text_save_dark.svg", "text_save_light.svg",
    "text_save_as_dark.svg", "text_save_as_light.svg",
    "text_wrap_dark.svg", "text_wrap_light.svg",
    "dock_pop_dark.svg", "dock_pop_dark_hover.svg",
    "dock_pop_light.svg", "dock_pop_light_hover.svg",
    "dock_fullscreen_dark.svg", "dock_fullscreen_dark_hover.svg",
    "dock_fullscreen_light.svg", "dock_fullscreen_light_hover.svg",
]


def p35_orphan_assets_are_gone():
    for f in DELETED_ASSETS:
        assert not os.path.exists(os.path.join(_IMG, f)), f"{f} is still on disk"


def p35_every_referenced_image_exists():
    """Forward direction: a url() the sheets name must resolve, or the control
    renders with no glyph at runtime and nothing warns."""
    missing = []
    for name, body in SHEETS.items():
        for ref in re.findall(r'url\("images/([^"]+)"\)', body):
            if not os.path.exists(os.path.join(_IMG, ref)):
                missing.append(f"{name} -> {ref}")
    assert not missing, f"stylesheet references a deleted image: {missing}"


def p35_no_image_file_is_unreferenced():
    """Reverse direction: this is the check that makes 3.5 stay done."""
    refs = "\n".join(SHEETS.values()) + _tree_source()
    orphans = [os.path.basename(p)
               for p in sorted(glob.glob(os.path.join(_IMG, "*.svg")))
               if os.path.basename(p) not in refs]
    assert not orphans, f"unreferenced image assets: {orphans}"


# ── 3.6 hygiene ─────────────────────────────────────────────────────────
def p36_commented_out_widget_block_is_gone():
    body = _src("frontEnd/Application.py")
    assert "self.soc" not in body, "the commented-out SoC block is still there"
    assert "showSoCRelease" not in body, "its dead connect survived"
    commented_sheets = [ln for ln in body.splitlines()
                        if ln.strip().startswith("#") and "setStyleSheet" in ln]
    assert not commented_sheets, \
        f"commented-out stylesheet lines remain: {commented_sheets}"


def p36_preferences_imports_sit_at_the_top():
    """`import json as _json` lived below the class that used it."""
    tree = ast.parse(_src("frontEnd/PreferencesDialog.py"))
    first_def = next((i for i, n in enumerate(tree.body)
                      if isinstance(n, (ast.ClassDef, ast.FunctionDef))), None)
    assert first_def is not None
    late = [n for n in tree.body[first_def:]
            if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not late, f"{len(late)} module-level import(s) still below the class"
    body = _src("frontEnd/PreferencesDialog.py")
    for n in ("json_load(", "json_dump(", "import json as _json", "_json."):
        assert n not in body, f"the {n} wrapper survived"
    assert "import json" in body and "json.load(" in body, \
        "the wrappers went but the plain calls did not replace them"


def p36_every_preferences_write_is_atomic():
    """reject()'s second write was a bare open(w) — the one path that runs while
    the user is closing the app."""
    tree = ast.parse(_src("frontEnd/PreferencesDialog.py"))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == "open"):
            continue
        mode = ""
        if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
            mode = str(node.args[1].value)
        mode += "".join(str(k.value.value) for k in node.keywords
                        if k.arg == "mode" and isinstance(k.value, ast.Constant))
        assert "w" not in mode and "a" not in mode, \
            f"PreferencesDialog line {node.lineno} still opens for writing"
    reject = _fn_node("frontEnd/PreferencesDialog.py", "reject")
    calls = [n.func.attr for n in ast.walk(reject)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert "write_json_atomic" in calls, \
        "reject() no longer persists the revert at all"


def p36_theme_comment_names_handlers_that_exist():
    """The stale comment named SpiceEditor, which has not existed for a while.
    Whatever replaced it has to be true, or the comment rots again."""
    body = _src("frontEnd/theme_utils.py")
    assert "SpiceEditor" not in body, "theme_utils still names SpiceEditor"
    i = body.index("Run it now AND again on the next event-loop tick")
    comment = body[i:i + 500]
    named = [n for n in ("CodeEditor", "EditorWindow", "FullScreenToggle",
                         "FlowNavigator", "VerilogVerifier", "plotWindow")
             if n in comment]
    assert len(named) >= 3, f"the replacement comment names too little: {named}"
    have = set()
    for p in glob.glob(os.path.join(_SRC, "**/*.py"), recursive=True):
        text = open(p, encoding="utf-8", errors="ignore").read()
        if "def changeEvent" not in text:
            continue
        for n in named:
            if re.search(rf"class {n}\b", text):
                have.add(n)
    missing = [n for n in named if n not in have]
    assert not missing, \
        f"comment names classes with no changeEvent handler: {missing}"


# ── C3 the accent engine is vestigial, and now says so ──────────────────
def pc3_the_write_lock_is_documented():
    import frontEnd.tokens as tokens
    doc = (tokens.__doc__ or "")
    for needle in ("accent_color", "_collect_prefs", "preferences.json"):
        assert needle in doc, \
            f"tokens.py docstring does not explain the write-lock ({needle})"
    tu = _src("frontEnd/theme_utils.py")
    head = tu[:tu.index("SECONDARY_TOKENS")]
    assert "tokens.py" in head, \
        "ACCENT_TOKENS carries no pointer to the explanation"


def pc3_the_docstring_matches_the_code():
    """The note is only true while _collect_prefs really pins all three keys.
    Re-enable accent picking and this fails, so the doc gets fixed with it."""
    node = _fn_node("frontEnd/PreferencesDialog.py", "_collect_prefs")
    ret = next(n for n in ast.walk(node) if isinstance(n, ast.Return))
    assert isinstance(ret.value, ast.Dict)
    pinned = {}
    for k, v in zip(ret.value.keys, ret.value.values, strict=True):
        if isinstance(k, ast.Constant) and isinstance(v, ast.Constant):
            pinned[k.value] = v.value
    for key, sentinel in (("accent_color", "default"),
                          ("secondary_accent_color", "system"),
                          ("internal_bg_color", "system")):
        assert pinned.get(key) == sentinel, \
            f"{key} is no longer pinned to {sentinel!r} — tokens.py now lies"


def pc3_the_engine_still_works_for_an_old_prefs_file():
    """Kept, not stripped: a preferences.json from an older build must still
    resolve rather than crash the theme build."""
    from frontEnd import theme_utils
    out = theme_utils.build_qss("style_dark.qss", True, "#FF00AA",
                                "system", "system", 100)
    assert "#FF00AA" in out.upper(), \
        "a custom accent no longer reaches the sheet"
    assert "rgba(255,0,170" in out.replace(" ", ""), \
        "recolor_accent_rgba no longer retints the glows"


for label, fn in [
    ("3.1  named dead groups are gone",      p31_dead_selector_groups_are_gone),
    ("3.1  every #name rule has a setter",
     p31_every_objectname_selector_has_a_setter),
    ("3.1  every [prop=v] rule has a setter",
     p31_every_attribute_selector_has_a_setter),
    ("3.1  structural diff still empty",     p31_structural_diff_is_still_empty),
    ("3.1  both sheets still parse in Qt",   p31_both_sheets_still_parse),
    ("3.1  live #dockCard rule survived",    p31_live_dock_card_rule_survived),
    ("3.2  dead widget classes are gone",    p32_dead_widget_classes_are_gone),
    ("3.2  every export has a consumer",     p32_module_exports_only_live_names),
    ("3.2  survivors still function",        p32_live_widget_api_still_works),
    ("3.2  Wayland findings were preserved",
     p32_wayland_findings_survived_the_deletion),
    ("3.3  dead installers are gone",        p33_dead_installers_are_gone),
    ("3.3  _revalidate kept and still used", p33_revalidate_survives_and_is_used),
    ("3.3  dockPopButton gone tree-wide",    p33_dockpopbutton_is_gone_tree_wide),
    ("3.3  the property is inert now",
     p33_the_property_no_longer_suppresses_a_glow),
    ("3.3  plain buttons still glow",
     p33_plain_buttons_still_take_the_glow_path),
    ("3.4  dead icon factories are gone",    p34_dead_icon_factories_are_gone),
    ("3.4  every factory has a consumer",
     p34_every_surviving_factory_has_a_consumer),
    ("3.4  every factory still renders",     p34_every_surviving_factory_still_renders),
    ("3.5  orphan assets are gone",          p35_orphan_assets_are_gone),
    ("3.5  every referenced image exists",   p35_every_referenced_image_exists),
    ("3.5  no image file is unreferenced",   p35_no_image_file_is_unreferenced),
    ("3.6  commented-out block is gone",     p36_commented_out_widget_block_is_gone),
    ("3.6  Preferences imports at the top",  p36_preferences_imports_sit_at_the_top),
    ("3.6  every prefs write is atomic",     p36_every_preferences_write_is_atomic),
    ("3.6  theme comment names real handlers",
     p36_theme_comment_names_handlers_that_exist),
    ("C3   write-lock is documented",        pc3_the_write_lock_is_documented),
    ("C3   the doc matches the code",        pc3_the_docstring_matches_the_code),
    ("C3   old prefs files still resolve",
     pc3_the_engine_still_works_for_an_old_prefs_file),
]:
    check(label, fn)

failed = [r for r in RESULTS if not r[1]]
for label, ok, msg in RESULTS:
    print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"\n        {msg}" if msg else ""))
print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
sys.exit(1 if failed else 0)
