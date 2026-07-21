"""Round 3 fuzz harness: subcircuit flows (item 1d).
uploadSub full flow + validateSubcir/validateSub direct + openSub edge.
Library writes redirected to temp; WorkerThread patched (no real eeschema).
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

LIBROOT = tempfile.mkdtemp(prefix="esim_fuzz_sublib_")
os.makedirs(os.path.join(LIBROOT, "SubcircuitLibrary"), exist_ok=True)

from configuration import paths, Dialogs  # noqa: E402

paths.library_path = lambda *parts: os.path.join(LIBROOT, *parts)

DIALOGS = []
for _k in ("information", "critical", "warning"):
    def _mk(kind):
        def f(parent, title, text, *a, **kw):
            DIALOGS.append((kind, str(text)[:120]))
        return f
    setattr(Dialogs, _k, _mk(_k))

STARTED_CMDS = []

from projManagement import Worker  # noqa: E402


class _FakeThread:
    def __init__(self, cmd):
        STARTED_CMDS.append(cmd)

    def start(self):
        pass


Worker.WorkerThread = _FakeThread

from projManagement.Validation import Validation  # noqa: E402
from subcircuit import uploadSub, openSub  # noqa: E402
# openSub imported WorkerThread directly — patch its module binding too.
openSub.WorkerThread = _FakeThread

NEXT_OPEN = [""]
QtWidgets.QFileDialog.getOpenFileName = staticmethod(
    lambda *a, **k: (NEXT_OPEN[0], "*.sub"))
QtWidgets.QFileDialog.getExistingDirectory = staticmethod(
    lambda *a, **k: NEXT_OPEN[0])

V = Validation()
RESULTS = []


def scenario(name):
    def deco(fn):
        RESULTS.append((name, fn))
        return fn
    return deco


def make_sub(body, name="frag.sub", binary=False):
    d = tempfile.mkdtemp(prefix="fuzzsub_")
    p = os.path.join(d, name)
    with open(p, "wb" if binary else "w") as f:
        f.write(body)
    return p


@scenario("A. upload valid .sub (happy path)")
def s_a():
    p = make_sub(".subckt ok_sub 1 2 3\nr1 1 2 1k\n.ends ok_sub\n",
                 "ok_sub.sub")
    NEXT_OPEN[0] = p
    uploadSub.UploadSub().upload()
    dest = os.path.join(LIBROOT, "SubcircuitLibrary", "ok_sub", "ok_sub.sub")
    assert os.path.exists(dest), "should be copied"
    return "copied ok"


@scenario("B. upload .sub with UTF-16/binary garbage content")
def s_b():
    p = make_sub(b"\xff\xfe\x00\x01BINARY\x00garbage\x9c",
                 "bin_sub.sub", binary=True)
    NEXT_OPEN[0] = p
    uploadSub.UploadSub().upload()
    return "survived, dialogs=%r" % (DIALOGS[:2],)


@scenario("C. validateSubcir on file deleted after selection (M10 race)")
def s_c():
    p = make_sub(".subckt x 1 2\n.ends x\n", "x.sub")
    os.remove(p)
    V.validateSubcir(p, "x")
    return "survived"


@scenario("D. validateSub on dir whose .sub vanishes mid-check")
def s_d():
    d = tempfile.mkdtemp(prefix="fuzzsubdir_")
    p = os.path.join(d, "y.sub")
    with open(p, "w") as f:
        f.write(".subckt y 1 2\n.ends y\n")
    import projManagement.Validation as VV
    real_exists = os.path.exists

    def racing_exists(path):
        r = real_exists(path)
        if path.endswith("y.sub") and r:
            os.remove(path)   # vanish between the check and the open
        return r
    VV.os.path.exists = racing_exists
    try:
        V.validateSub(d, 2)
    finally:
        VV.os.path.exists = real_exists
    return "survived"


@scenario("E. openSub on folder containing ZERO .sub files")
def s_e():
    d = tempfile.mkdtemp(prefix="fuzzempty_")
    NEXT_OPEN[0] = d
    STARTED_CMDS.clear()
    openSub.openSub().body()
    return "cmd=%r" % (STARTED_CMDS,)


@scenario("F. upload .sub whose .subckt name mismatches filename")
def s_f():
    p = make_sub(".subckt other 1 2\n.ends other\n", "mismatch.sub")
    NEXT_OPEN[0] = p
    uploadSub.UploadSub().upload()
    return "survived, dialogs=%r" % (DIALOGS[:2],)


@scenario("G. upload valid .sub twice (dir exists second time)")
def s_g():
    body = ".subckt twice 1 2\n.ends twice\n"
    p = make_sub(body, "twice.sub")
    NEXT_OPEN[0] = p
    uploadSub.UploadSub().upload()
    uploadSub.UploadSub().upload()
    return "survived, dialogs=%r" % (DIALOGS[:2],)


def main():
    fails = 0
    for name, fn in RESULTS:
        DIALOGS.clear()
        try:
            note = fn()
            print("[PASS] %s :: %s" % (name, note))
        except Exception:
            fails += 1
            tb = traceback.format_exc().strip().splitlines()
            site = next((ln.strip() for ln in reversed(tb)
                         if 'File "' in ln), "?")
            print("[RAISE] %s\n        %s\n        %s" % (name, tb[-1], site))
    print("\n%d/%d raised" % (fails, len(RESULTS)))


if __name__ == "__main__":
    main()
