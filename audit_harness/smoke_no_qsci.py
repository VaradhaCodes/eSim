"""Import-smoke for R2-1: eSim must still start with QScintilla missing.

Blocks ``PyQt6.Qsci`` at the import hook level (the closest in-repo stand-in
for a machine where the module is absent or its .pyd fails to load), then
imports the whole startup chain and builds the fallback editor. Exit 0 = the
app can reach its window without QScintilla.
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SRC = os.path.join(REPO, "src")

ISO = tempfile.mkdtemp(prefix="esim_smoke_home_")
os.environ["HOME"] = ISO
os.environ["USERPROFILE"] = ISO
os.environ["APPDATA"] = os.path.join(ISO, "AppData", "Roaming")
os.environ["LOCALAPPDATA"] = os.path.join(ISO, "AppData", "Local")
os.makedirs(os.environ["APPDATA"], exist_ok=True)
os.makedirs(os.environ["LOCALAPPDATA"], exist_ok=True)
os.environ["QT_QPA_PLATFORM"] = "offscreen"

sys.path.insert(0, SRC)
# Application.py does a bare `import pathmagic`, which lives beside it -- the
# launcher runs from that directory, so mirror it here.
sys.path.insert(0, os.path.join(SRC, "frontEnd"))


class _BlockQsci:
    """Meta-path finder that makes PyQt6.Qsci look uninstalled."""

    def find_module(self, name, path=None):
        return None

    def find_spec(self, name, path=None, target=None):
        if name == "PyQt6.Qsci" or name.startswith("PyQt6.Qsci."):
            raise ImportError("No module named 'PyQt6.Qsci' (blocked by smoke)")
        return None


sys.meta_path.insert(0, _BlockQsci())
for mod in [m for m in sys.modules if m.startswith("PyQt6.Qsci")]:
    del sys.modules[mod]

failures = []

try:
    from PyQt6.Qsci import QsciScintilla  # noqa: F401
    failures.append("Qsci import was NOT blocked -- smoke is meaningless")
except ImportError:
    pass

from PyQt6 import QtWidgets  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

for name in ("codeEditor.lexers", "codeEditor.theme", "codeEditor.PlainEditor",
             "codeEditor.EditorWindow", "frontEnd.ProjectExplorer",
             "frontEnd.Application"):
    try:
        __import__(name)
        print("[OK]     import %s" % name)
    except Exception as exc:
        failures.append("import %s -> %r" % (name, exc))
        print("[FAIL]   import %s -> %r" % (name, exc))

# The fallback editor must actually build and report its language.
try:
    from codeEditor.PlainEditor import PlainEditor
    from codeEditor import lexers
    probe = os.path.join(ISO, "probe.cir")
    with open(probe, "w") as handle:
        handle.write("* probe\nr1 1 0 1k\n.end\n")
    editor = PlainEditor(probe)
    assert editor.language() == "SPICE", editor.language()
    assert lexers.make_lexer(probe, None) is None
    print("[OK]     PlainEditor builds, language=%s, make_lexer=None"
          % editor.language())
except Exception as exc:
    failures.append("PlainEditor -> %r" % (exc,))
    print("[FAIL]   PlainEditor -> %r" % (exc,))

print("\nRESULT: %s" % ("FAIL (%d)" % len(failures) if failures else "PASS"))
for line in failures:
    print("  - " + line)
sys.exit(1 if failures else 0)
