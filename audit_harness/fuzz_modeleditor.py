"""Round 3 fuzz harness: ModelEditor against fuzzed .lib/.xml (item 1b).

Redirects paths.library_path to a temp tree so the repo library is never
written. Patches file dialogs / input dialogs to canned returns.
"""
import os
import shutil
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

# fake library root (deviceModelLibrary layout) in temp
LIBROOT = tempfile.mkdtemp(prefix="esim_fuzz_lib_")
DML = os.path.join(LIBROOT, "deviceModelLibrary")
for sub in ("User Libraries", "Diode", "MOS", "JFET", "IGBT", "Misc",
            "Transistor", "Templates"):
    os.makedirs(os.path.join(DML, sub), exist_ok=True)
# real templates are useful for the New flow
REAL_TPL = os.path.join(SRC, "..", "library", "deviceModelLibrary", "Templates")
REAL_TPL = os.path.abspath(REAL_TPL)
if os.path.isdir(REAL_TPL):
    for f in os.listdir(REAL_TPL):
        shutil.copy2(os.path.join(REAL_TPL, f),
                     os.path.join(DML, "Templates", f))

from configuration import paths, Dialogs  # noqa: E402

_orig_library_path = paths.library_path


def _fake_library_path(name):
    return os.path.join(LIBROOT, name)


paths.library_path = _fake_library_path

DIALOGS = []
for _k in ("information", "critical", "warning", "question"):
    def _mk(kind):
        def f(parent, title, text, *a, **kw):
            DIALOGS.append((kind, str(title), str(text)[:150]))
        return f
    setattr(Dialogs, _k, _mk(_k))

from modelEditor import ModelEditor  # noqa: E402

# canned dialog returns, set per scenario
NEXT_OPEN = [""]
NEXT_TEXTS = []


def _fake_getopen(*a, **k):
    return (NEXT_OPEN[0], "*.lib")


def _fake_gettext(*a, **k):
    if NEXT_TEXTS:
        return NEXT_TEXTS.pop(0)
    return ("", False)


QtWidgets.QFileDialog.getOpenFileName = staticmethod(_fake_getopen)
QtWidgets.QInputDialog.getText = staticmethod(_fake_gettext)

RESULTS = []


def scenario(name):
    def deco(fn):
        RESULTS.append((name, fn))
        return fn
    return deco


def make_lib(body, name="fuzz.lib"):
    d = tempfile.mkdtemp(prefix="fuzzlib_")
    p = os.path.join(d, name)
    with open(p, "w") as f:
        f.write(body)
    return p


def fresh():
    DIALOGS.clear()
    return ModelEditor.ModelEditorclass()


@scenario("A. upload well-formed diode .lib")
def s_a():
    w = fresh()
    NEXT_OPEN[0] = make_lib(
        ".MODEL 1N4148 D(IS=4.352E-9 N=1.906 BV=110 RS=0.6458)\n")
    NEXT_TEXTS.append(("fuzz_ok_model", True))
    w.converttoxml()
    assert os.path.exists(os.path.join(DML, "User Libraries",
                                       "fuzz_ok_model.lib"))
    return "uploaded ok, dialogs=%r" % (DIALOGS[:2],)


@scenario("B. upload .lib with NO .model line")
def s_b():
    w = fresh()
    NEXT_OPEN[0] = make_lib("* just a comment\n* nothing else\n")
    NEXT_TEXTS.append(("fuzz_nomodel", True))
    w.converttoxml()
    return "survived, dialogs=%r" % (DIALOGS[:2],)


@scenario("C. upload EMPTY .lib")
def s_c():
    w = fresh()
    NEXT_OPEN[0] = make_lib("")
    NEXT_TEXTS.append(("fuzz_empty", True))
    w.converttoxml()
    return "survived, dialogs=%r" % (DIALOGS[:2],)


@scenario("D. upload .lib where .model is the LAST token")
def s_d():
    w = fresh()
    NEXT_OPEN[0] = make_lib(".model")
    NEXT_TEXTS.append(("fuzz_last", True))
    w.converttoxml()
    return "survived, dialogs=%r" % (DIALOGS[:2],)


@scenario("E. upload .lib whose params are not valid XML tag names")
def s_e():
    w = fresh()
    NEXT_OPEN[0] = make_lib(".MODEL bad D(1abc=5 x:y=2)\n")
    NEXT_TEXTS.append(("fuzz_badtag", True))
    w.converttoxml()
    return "survived, dialogs=%r" % (DIALOGS[:2],)


@scenario("F. upload .lib, then library name with illegal filename chars")
def s_f():
    w = fresh()
    NEXT_OPEN[0] = make_lib(".MODEL ok D(IS=1e-9)\n")
    NEXT_TEXTS.append(('bad"name<>|', True))
    w.converttoxml()
    return "survived, dialogs=%r" % (DIALOGS[:2],)


@scenario("G. Edit a .lib whose sibling .xml is MISSING")
def s_g():
    w = fresh()
    lib = make_lib(".MODEL solo D(IS=1e-9)\n", "solo.lib")
    NEXT_OPEN[0] = lib
    w.openedit()
    return "survived (swallowed?), dialogs=%r" % (DIALOGS[:2],)


@scenario("H. Edit a .lib whose .xml exists but lacks ref_model, then Save")
def s_h():
    w = fresh()
    lib = make_lib(".MODEL noref D(IS=1e-9)\n", "noref.lib")
    xml = os.path.splitext(lib)[0] + ".xml"
    with open(xml, "w") as f:
        f.write("<library><model_name>D</model_name>"
                "<param><IS>1e-9</IS></param></library>")
    NEXT_OPEN[0] = lib
    w.openedit()
    w.savemodelfile()
    return "survived, dialogs=%r" % (DIALOGS[:3],)


@scenario("I. Edit valid lib -> New -> Save WITHOUT device type (txtfile)")
def s_i():
    w = fresh()
    lib = make_lib(".MODEL seq D(IS=1e-9)\n", "seq.lib")
    xml = os.path.splitext(lib)[0] + ".xml"
    with open(xml, "w") as f:
        f.write("<library><model_name>D</model_name><ref_model>seq</ref_model>"
                "<param><IS>1e-9</IS></param></library>")
    NEXT_OPEN[0] = lib
    w.openedit()                     # table mounted, savebtn enabled
    NEXT_TEXTS.append(("newmodel", True))
    w.opennew()                      # newflag=1, radios enabled, none checked
    w.savemodelfile()                # createXML with no radio checked
    return "survived, dialogs=%r" % (DIALOGS[:3],)


@scenario("J. Edit a .lib whose .xml is MALFORMED (truncated)")
def s_j():
    w = fresh()
    lib = make_lib(".MODEL trunc D(IS=1e-9)\n", "trunc.lib")
    xml = os.path.splitext(lib)[0] + ".xml"
    with open(xml, "w") as f:
        f.write("<library><param>")
    NEXT_OPEN[0] = lib
    w.openedit()
    return "survived, dialogs=%r" % (DIALOGS[:2],)


def main():
    print("libroot:", LIBROOT)
    failures = 0
    for name, fn in RESULTS:
        DIALOGS.clear()
        NEXT_TEXTS.clear()
        cwd_before = os.getcwd()
        try:
            note = fn()
            print("[PASS] %s :: %s" % (name, note))
        except Exception:
            failures += 1
            tb = traceback.format_exc().strip().splitlines()
            site = next((ln.strip() for ln in reversed(tb)
                         if 'File "' in ln), "?")
            print("[RAISE] %s\n        %s\n        %s" % (name, tb[-1], site))
        cwd_after = os.getcwd()
        if cwd_after != cwd_before:
            print("        [CWD STRANDED] %s -> %s" % (cwd_before, cwd_after))
            os.chdir(cwd_before)
    print("\n%d/%d scenarios raised" % (failures, len(RESULTS)))


if __name__ == "__main__":
    main()
