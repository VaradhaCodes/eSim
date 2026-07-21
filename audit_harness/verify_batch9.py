"""Batch 9 verification: dead-restore / false-success cleanup.

Empirically certifies the fixes that turned "looks fine, does nothing" code
into code that actually works (or fails honestly):

  R2-2      Microcontroller prev-values restore now runs (was dead: bare
            undefined ``root`` -> caught NameError every field).
  R3-7      ngmo empty/comment-only netlist -> honest error + NO .mo written
            (was a false "successfully converted" + junk file).
  R3-13 b   CodeEditor.reload survives the file being deleted underneath it.
  R3-13 c   DeviceModel.GenerateSOCbutton closes its file handles (with).
  R3-13 d   Source/Model/Microcontroller pre-bind root=None (no unbound shape).

Offscreen, isolated HOME/APPDATA, this machine only. Run: python verify_batch9.py
"""
import ast
import os
import sys
import tempfile
import traceback
from xml.etree import ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SRC = os.path.join(REPO, "src")

ISO = tempfile.mkdtemp(prefix="esim_b9_home_")
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

PASS, FAIL = [], []


def ok(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print("[%s] %s%s" % ("PASS" if cond else "FAIL", name,
                         ("  -- " + detail) if detail else ""))


# ---------------------------------------------------------------------------
# R2-2 -- Microcontroller previous-values restore is now LIVE.
# The restore loop keys a saved value off (child.tag == line[3],
# child.text == line[2]) and writes child[i].text into the widget. Build the
# minimal modelList line + matching XML and assert the widget picks the value
# up. Before the fix the loop iterated an undefined ``root`` -> NameError
# swallowed -> the field kept its empty default.
# ---------------------------------------------------------------------------
def test_r2_2_restore():
    from kicadtoNgspice import Microcontroller

    saved_xml = os.path.join(ISO, "proj_Previous_Values.xml")
    root = ET.Element("root")
    mc = ET.SubElement(root, "microcontroller")
    field = ET.SubElement(mc, "matchtag")   # child.tag == line[3]
    field.text = "matchtext"                # child.text == line[2]
    ET.SubElement(field, "v").text = "RESTORED_VALUE"   # child[0].text
    ET.ElementTree(root).write(saved_xml)

    # Route the widget's prev-values lookup at our crafted file.
    Microcontroller.previous_values_path = lambda _k: saved_xml

    # 8-field modelList line; scalar param branch. Indices the code touches:
    # line[5] group title, line[7] param dict, line[2]/line[3] restore keys.
    line = [0, "n1", "matchtext", "matchtag", "4", "MyMCU", "6",
            {"pk": "Param Label"}]
    w = Microcontroller.Microcontroller(["dummy"], [line], "proj.cir")

    restored = [v.text() for v in w.obj_trac.microcontroller_var.values()]
    ok("R2-2 microcontroller restore fires",
       "RESTORED_VALUE" in restored,
       "widget values=%r" % restored)


# ---------------------------------------------------------------------------
# R3-7 -- empty netlist is an honest error, and NO .mo is written.
# ---------------------------------------------------------------------------
def test_r3_7_no_false_success():
    from configuration import Dialogs
    from configuration.Appconfig import Appconfig
    from ngspicetoModelica import ModelicaUI

    shown = []

    class _Box:
        def setModal(self, *_): pass
        def setWindowTitle(self, *_): pass
        def setTextFormat(self, *_): pass
        def setText(self, t): shown.append(("box", str(t)))
        def showMessage(self, m): shown.append(("error", str(m)))
        def exec(self): pass

    Dialogs.make_message_box = lambda parent=None, *a, **k: _Box()
    Dialogs.make_error_message = lambda parent=None: _Box()
    Appconfig.modelica_map_json = os.path.join(
        REPO, "library", "ngspicetoModelica", "Mapping.json")

    d = tempfile.mkdtemp(prefix="b9_ngmo_")
    cir = os.path.join(d, "proj.cir.out")
    with open(cir, "w") as f:
        f.write("* just a comment\n")           # comment-only == empty circuit
    w = ModelicaUI.OpenModelicaEditor(d)
    w.ngspiceNetlist = cir
    w.callConverter()

    last = shown[-1] if shown else ("none", "")
    mo_written = os.path.exists(os.path.join(d, "proj.mo"))
    ok("R3-7 empty netlist -> error dialog (not success)",
       last[0] == "error" and "no circuit elements" in last[1].lower(),
       "dialog=%r" % (last,))
    ok("R3-7 empty netlist writes NO .mo", not mo_written,
       "proj.mo exists=%s" % mo_written)


# ---------------------------------------------------------------------------
# R3-13 b -- CodeEditor.reload survives the file vanishing under it.
# ---------------------------------------------------------------------------
def test_r3_13b_reload_deleted():
    from codeEditor.CodeEditor import CodeEditor

    d = tempfile.mkdtemp(prefix="b9_ce_")
    fp = os.path.join(d, "gen.v")
    with open(fp, "w") as f:
        f.write("line one\nline two\n")
    ed = CodeEditor(fp)
    os.remove(fp)                                # external delete
    raised = None
    try:
        ed.reload()                              # must not blow up
    except Exception as e:                        # noqa: BLE001
        raised = e
    ok("R3-13b reload on deleted file does not raise", raised is None,
       repr(raised) if raised else "buffer kept: %r" % ed.text()[:20])


# ---------------------------------------------------------------------------
# R3-13 c/d -- static source guarantees (handles closed, root pre-bound).
# ---------------------------------------------------------------------------
def _src(rel):
    with open(os.path.join(SRC, rel), encoding="utf-8") as f:
        return f.read()


def test_r3_13c_soc_handles():
    body = _src("kicadtoNgspice/DeviceModel.py")
    # No bare raw opens left in the SOC generator; both go through `with`.
    ok("R3-13c GenerateSOCbutton uses `with open` (no leaked handles)",
       "with open(os.path.join(projpath, filename)) as analysisfile:" in body
       and "with open(parsed_path, 'w') as parsedfile:" in body
       and "parsedfile = open(" not in body
       and "analysisfile = open(" not in body)


def test_r3_13d_root_prebound():
    for rel in ("kicadtoNgspice/Source.py",
                "kicadtoNgspice/Model.py",
                "kicadtoNgspice/Microcontroller.py"):
        tree = ast.parse(_src(rel))
        # Every __init__ that restores prev-values must bind a plain `root`
        # name (root = ...) at statement level so it is never unbound.
        binds = any(
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "root"
                    for t in n.targets)
            for n in ast.walk(tree))
        # ...and no leftover assignment to self.root for the restore node.
        self_root = "self.root =" in _src(rel)
        ok("R3-13d %s pre-binds local root" % os.path.basename(rel),
           binds and not self_root,
           "root-bind=%s self.root=%s" % (binds, self_root))


def main():
    for t in (test_r2_2_restore, test_r3_7_no_false_success,
              test_r3_13b_reload_deleted, test_r3_13c_soc_handles,
              test_r3_13d_root_prebound):
        try:
            t()
        except Exception:                          # noqa: BLE001
            FAIL.append(t.__name__)
            print("[FAIL] %s (harness raised)" % t.__name__)
            traceback.print_exc()
    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
