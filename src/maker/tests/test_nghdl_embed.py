"""Regression guards for embedding the NGHDL GUI as a Makerchip tab.

The NGHDL window (``nghdl/src/ngspice_ghdl.py``) doubles as a standalone app
*and* as a tab inside eSim's Makerchip dock. When embedded it must never:

  * call ``sys.exit()`` -- that would terminate all of eSim; and
  * leave eSim's working directory changed.

It must also construct as a plain ``QWidget`` so it can live in a tab, and a
broken / missing NGHDL install must degrade to a placeholder rather than break
the Makerchip dock.

If one of these tests fails, fix the embedding in ``ngspice_ghdl.py`` /
``makerchip.py`` rather than relaxing the guard.
"""
import ast
import os

import pytest

_HERE = os.path.dirname(__file__)
_NGHDL_SRC = os.path.abspath(
    os.path.join(_HERE, "..", "..", "..", "nghdl", "src"))
_NGHDL_GUI = os.path.join(_NGHDL_SRC, "ngspice_ghdl.py")
_MAKERCHIP = os.path.join(_HERE, "..", "makerchip.py")
_APPLICATION = os.path.join(_HERE, "..", "..", "frontEnd", "Application.py")


def _read(path):
    with open(path) as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# AST guard: no unguarded sys.exit() in the NGHDL GUI
# ---------------------------------------------------------------------------

def _sys_exit_calls(tree):
    """Yield (Call node, ancestors) for every ``sys.exit(...)`` in the tree."""
    def walk(node, ancestors):
        for child in ast.iter_child_nodes(node):
            if (isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == "exit"
                    and isinstance(child.func.value, ast.Name)
                    and child.func.value.id == "sys"):
                yield child, ancestors
            yield from walk(child, ancestors + [node])
    yield from walk(tree, [])


def _guarded_by_embedded(ancestors):
    """True if some ancestor ``if`` statement tests ``self.embedded``."""
    for node in ancestors:
        if isinstance(node, ast.If):
            for sub in ast.walk(node.test):
                if isinstance(sub, ast.Attribute) and sub.attr == "embedded":
                    return True
    return False


def _in_main(ancestors):
    """True if the call lives inside the standalone ``main()`` entry point."""
    return any(isinstance(n, ast.FunctionDef) and n.name == "main"
               for n in ancestors)


def test_no_unguarded_sys_exit_in_nghdl_gui():
    tree = ast.parse(_read(_NGHDL_GUI))
    offenders = [call.lineno for call, ancestors in _sys_exit_calls(tree)
                 if not (_in_main(ancestors) or _guarded_by_embedded(ancestors))]
    assert not offenders, (
        "Unguarded sys.exit() at line(s) %s in ngspice_ghdl.py -- in embedded "
        "mode this terminates all of eSim. Guard with `if not self.embedded`."
        % offenders)


# ---------------------------------------------------------------------------
# Source-level wiring guards (cheap, no heavy imports)
# ---------------------------------------------------------------------------

def test_makerchip_wires_nghdl_tab():
    src = _read(_MAKERCHIP)
    assert '"NGHDL"' in src, "makerchip.py must add an 'NGHDL' tab"
    assert "Mainwindow(embedded=True)" in src, \
        "NGHDL tab must construct the GUI in embedded mode"
    assert "nghdlPlaceholder" in src, \
        "a guarded placeholder fallback must exist"


def test_makerchip_nghdl_build_is_guarded():
    """The lazy NGHDL build must be wrapped in try/except so a broken install
    cannot take down the Makerchip dock."""
    tree = ast.parse(_read(_MAKERCHIP))
    build = next((n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "buildNgHdlTab"),
                 None)
    assert build is not None, "makerchip.py must define buildNgHdlTab"
    assert any(isinstance(n, ast.Try) for n in ast.walk(build)), \
        "buildNgHdlTab must guard construction with try/except"


def test_toolbar_no_standalone_nghdl():
    app_src = _read(_APPLICATION)
    assert "def open_nghdl" not in app_src, \
        "standalone open_nghdl handler should be removed"
    assert "addAction(self.nghdl)" not in app_src, \
        "standalone NGHDL toolbar action should be removed"


# ---------------------------------------------------------------------------
# Functional guard (needs a configured NGHDL install + headless Qt)
# ---------------------------------------------------------------------------

def _have_config():
    cfg = os.path.join(os.path.expanduser("~"), ".nghdl", "config.ini")
    return os.path.isfile(cfg) and os.path.getsize(cfg) > 0


@pytest.mark.skipif(not _have_config(),
                    reason="NGHDL not installed (~/.nghdl/config.ini absent)")
def test_embedded_mainwindow_is_safe_widget(qapp):
    import sys
    if _NGHDL_SRC not in sys.path:
        sys.path.append(_NGHDL_SRC)
    from PyQt6 import QtWidgets
    import Appconfig
    from ngspice_ghdl import Mainwindow

    cwd_before = os.getcwd()
    win = Mainwindow(embedded=True)
    try:
        assert isinstance(win, QtWidgets.QWidget)
        # Embedded mode behaves like `nghdl -e` (creates the KiCad symbol).
        assert Appconfig.Appconfig.esimFlag == 1
        # The Exit button must not be placed in the embedded layout.
        assert win.exitbtn.parent() is None
        # Constructing the tab must not move eSim's working directory.
        assert os.getcwd() == cwd_before
    finally:
        win.deleteLater()
