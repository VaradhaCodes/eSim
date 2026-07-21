"""Round 3 fuzz harness: NgspicetoModelica via ModelicaUI.callConverter
(item 1c). Verifies the converter's try/except+finally actually contains
every parser crash class found by static read, and that CWD is restored.
"""
import os
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SRC = os.path.join(REPO, "src")

ISO = tempfile.mkdtemp(prefix="esim_fuzz_home_")
os.environ["HOME"] = ISO
os.environ["USERPROFILE"] = ISO
os.environ["APPDATA"] = os.path.join(ISO, "AppData", "Roaming")
os.environ["LOCALAPPDATA"] = os.path.join(ISO, "AppData", "Local")
os.makedirs(os.environ["APPDATA"], exist_ok=True)
os.makedirs(os.environ["LOCALAPPDATA"], exist_ok=True)
os.environ["QT_QPA_PLATFORM"] = "offscreen"

sys.path.insert(0, SRC)

from PyQt6 import QtWidgets  # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

from configuration import Dialogs  # noqa: E402

DIALOGS = []


class _StubBox:
    def __init__(self):
        self._title = ""

    def setModal(self, *_):
        pass

    def setWindowTitle(self, t):
        self._title = t

    def setTextFormat(self, *_):
        pass

    def setText(self, t):
        DIALOGS.append(("box", self._title, str(t)[:160]))

    def showMessage(self, m):
        DIALOGS.append(("error", self._title, str(m)[:160]))

    def exec(self):
        return 0


Dialogs.make_message_box = lambda parent=None, *a, **k: _StubBox()
Dialogs.make_error_message = lambda parent=None: _StubBox()

from ngspicetoModelica import ModelicaUI  # noqa: E402
from configuration.Appconfig import Appconfig  # noqa: E402

# Seed the map JSON the way a healthy install's config.ini would.
Appconfig.modelica_map_json = os.path.join(
    REPO, "library", "ngspicetoModelica", "Mapping.json")

CASES = [
    ("A. valid RC netlist", """* rc circuit
r1 n1 n2 1k
c1 n2 0 1u
v1 n1 0 dc 5
.tran 1m 10m
.end
"""),
    ("B. leading + continuation first line", "+ 1 2 3\nr1 1 0 1k\n.end\n"),
    ("C. .include without extension", ".include foo\nr1 1 0 1k\n.end\n"),
    ("D. .model line too short", ".model q1\nr1 1 0 1k\n.end\n"),
    ("E. .model param without =", ".model m1 npn(bf)\nr1 1 0 1k\n.end\n"),
    ("F. f-source with 3 fields", "f1 1 2\nr1 1 0 1k\n.end\n"),
    ("G. x line with missing .sub file", "x1 1 2 missing_sub\nr1 1 0 1k\n.end\n"),
    ("H. v pulse with too few args", "v1 1 0 pulse(0 5)\nr1 1 0 1k\n.end\n"),
    ("I. r line with single node", "r1 1\n.end\n"),
    ("J. q with unknown model ref", "q1 1 2 3 nomodel\n.end\n"),
    ("K. empty file", ""),
    ("L. only a comment", "* nothing\n"),
]


def main():
    fails = 0
    for name, body in CASES:
        DIALOGS.clear()
        d = tempfile.mkdtemp(prefix="fuzzmo_")
        cir = os.path.join(d, "proj.cir.out")
        with open(cir, "w") as f:
            f.write(body)
        cwd = os.getcwd()
        try:
            w = ModelicaUI.OpenModelicaEditor(d)
            w.ngspiceNetlist = cir
            w.callConverter()
            kind = DIALOGS[-1] if DIALOGS else ("none", "", "")
            print("[PASS] %s :: dialog=%s %.90r" % (name, kind[0], kind[2]))
        except Exception:
            fails += 1
            tb = traceback.format_exc().strip().splitlines()
            site = next((ln.strip() for ln in reversed(tb)
                         if 'File "' in ln), "?")
            print("[RAISE] %s\n        %s\n        %s" % (name, tb[-1], site))
        if os.getcwd() != cwd:
            print("        [CWD STRANDED] -> %s" % os.getcwd())
            os.chdir(cwd)
    print("\n%d/%d raised" % (fails, len(CASES)))


if __name__ == "__main__":
    main()
