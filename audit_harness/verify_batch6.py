"""Batch 6 verification harness (file-I/O race hardening + HIGH rename).

Proves, offscreen on this machine, the fixes for:
  M5  the four schematic/library converters degrade a missing source file and a
      failed workspace copytree into a dialog instead of an excepthook crash.
  M10 validateSub / validateSubcir survive a .sub that vanishes / is unreadable
      (already gated by fuzz_subcircuit.py; re-checked here directly).
  H6  renameProject's revert path guards every os.rename -- a revert that hits
      the same lock no longer escapes; the project is reported as a mixed state
      instead of half-renamed into the excepthook.
  R3-13a  the dead Model.add_hex_btn (wired to a nonexistent self.addHex) is gone.

Each test is independent; a heavy/failing one can't sink the rest.
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SRC = os.path.join(REPO, "src")

ISO = tempfile.mkdtemp(prefix="esim_b6_home_")
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

# Record every dialog instead of showing it (offscreen exec() would block).
CRITICALS = []
Dialogs.critical = lambda parent, title, text, *a, **k: CRITICALS.append(
    (str(title), str(text)))


class FakeBox:
    """Stand-in for the QMessageBox that convert()'s success path builds."""
    def __init__(self):
        self.text = ""
        self.icon = None

    def setIcon(self, i):
        self.icon = i

    def setWindowTitle(self, t):
        pass

    def setText(self, t):
        self.text = t

    def setStandardButtons(self, b):
        pass

    def exec(self):
        return 0


# --------------------------------------------------------------------------- #
def test_m5_missing_source_is_a_dialog():
    """convert() on a path that isn't there -> 'File not found' dialog, no raise."""
    from converter.pspiceToKicad import PspiceConverter
    from converter.ltspiceToKicad import LTspiceConverter
    from converter.libConverter import PspiceLibConverter
    from converter.LtspiceLibConverter import LTspiceLibConverter

    cases = [
        (PspiceConverter, "ghost.sch"),
        (LTspiceConverter, "ghost.asc"),
        (PspiceLibConverter, "ghost.slb"),
        (LTspiceLibConverter, "ghost.asy"),
    ]
    for cls, name in cases:
        CRITICALS.clear()
        gone = os.path.join(tempfile.gettempdir(), "b6_no_such_%d_%s" % (
            os.getpid(), name))
        cls(parent=None).convert(gone)          # pre-fix: FileNotFoundError
        assert CRITICALS and CRITICALS[-1][0] == "File not found", (
            "%s: missing source did not raise the File-not-found dialog (%r)"
            % (cls.__name__, CRITICALS))
    return "4/4 converters: missing source -> dialog, no crash"


def test_m5_copytree_failure_is_a_dialog():
    """Parser succeeds but the workspace copy fails -> warning, output kept.

    Batch 10's L6 moved the workspace copy off the GUI thread into the async
    ``_on_convert_done`` slot (BackgroundJob), so ``convert()`` now returns
    before the copy runs. Drive ``_on_convert_done`` directly with a rc=0
    result -- that is where the M5 copytree guard lives in the merged code."""
    from converter import pspiceToKicad

    src = tempfile.mkdtemp(prefix="b6_m5src_")
    ws = tempfile.mkdtemp(prefix="b6_m5ws_")

    orig_copy = pspiceToKicad.shutil.copytree
    orig_mkbox = Dialogs.make_message_box
    box = FakeBox()

    def _boom(*a, **k):
        raise OSError("workspace locked by a sync client")
    pspiceToKicad.shutil.copytree = _boom
    Dialogs.make_message_box = lambda *a, **k: box
    conv = pspiceToKicad.PspiceConverter(parent=None)
    conv.get_workspace_directory = lambda: ws
    try:
        # rc=0 -> success path; copytree (_boom) fires inside the M5 guard.
        conv._on_convert_done(src, "demo", (0, "", ""))
    finally:
        pspiceToKicad.shutil.copytree = orig_copy
        Dialogs.make_message_box = orig_mkbox

    assert "could not be copied" in box.text, (
        "copytree failure not reported to the user: %r" % box.text)
    return "copytree OSError -> warning dialog, conversion not discarded"


# --------------------------------------------------------------------------- #
def test_m10_validate_survive_vanished_file():
    """validateSub / validateSubcir on a missing anchor -> codes, not a crash."""
    from projManagement.Validation import Validation
    v = Validation()

    empty_dir = tempfile.mkdtemp(prefix="b6_m10dir_")      # no .sub inside
    assert v.validateSub(empty_dir, 2) == "DIREC", \
        "validateSub on a dir with no readable .sub must return DIREC"

    gone = os.path.join(tempfile.gettempdir(), "b6_m10_%d.sub" % os.getpid())
    assert v.validateSubcir(gone, "x") is False, \
        "validateSubcir on a missing file must return False, not raise"
    return "validateSub->DIREC, validateSubcir->False (no excepthook)"


# --------------------------------------------------------------------------- #
def test_h6_revert_failure_is_reported_not_crashed():
    """A rename that fails, then whose revert also fails, -> mixed-state dialog."""
    from frontEnd.ProjectExplorer import ProjectExplorer

    stem = "proj"
    parent = tempfile.mkdtemp(prefix="b6_h6_")
    projectPath = os.path.join(parent, stem)
    os.makedirs(projectPath)
    for fn in (stem + ".proj", stem + ".cir"):
        open(os.path.join(projectPath, fn), "w").close()
    newName = "renamed"
    updatedPath = os.path.join(parent, newName)

    px = ProjectExplorer()
    px.treewidget.setColumnCount(2)
    item = QtWidgets.QTreeWidgetItem()
    item.setText(0, stem)
    item.setData(0, px.STEM_ROLE, stem)
    item.setText(1, projectPath)
    px.treewidget.addTopLevelItem(item)
    px.treewidget.setCurrentItem(item)

    px.obj_appconfig.project_explorer = {
        projectPath: [stem + ".proj", stem + ".cir"]}
    px.refreshProject = lambda *a, **k: True          # skip the tree rebuild
    px._repointSchematic = lambda *a, **k: None
    QtWidgets.QInputDialog.getText = staticmethod(
        lambda *a, **k: (newName, True))

    real_rename = os.rename
    state = {"n": 0}

    def flaky_rename(a, b):
        state["n"] += 1
        if state["n"] == 1:
            return real_rename(a, b)     # folder forward rename succeeds
        raise OSError("locked by KiCad")  # every file / revert rename fails

    CRITICALS.clear()
    os.rename = flaky_rename
    try:
        px.renameProject()               # pre-fix: 2nd OSError -> excepthook
    finally:
        os.rename = real_rename

    assert CRITICALS, "no dialog surfaced from a failed rename+revert"
    assert "mixed state" in CRITICALS[-1][1], (
        "half-renamed project not reported as a mixed state: %r"
        % (CRITICALS[-1],))
    # The forward folder rename really happened; the revert of it failed, so
    # the folder is left at the new path -- exactly the mixed state we report.
    assert os.path.isdir(updatedPath), "folder-forward rename did not run"
    return "revert OSError contained; mixed state reported, no excepthook"


# --------------------------------------------------------------------------- #
def test_r3_13a_dead_add_hex_btn_removed():
    """Model.add_hex_btn (wired to a nonexistent self.addHex) is gone."""
    from kicadtoNgspice.Model import Model
    assert "add_hex_btn" not in Model.__dict__, \
        "dead add_hex_btn is still defined on Model"
    assert not hasattr(Model, "addHex"), \
        "Model still exposes addHex (the method add_hex_btn wrongly wired)"
    return "add_hex_btn removed; no dangling self.addHex reference"


# --------------------------------------------------------------------------- #
TESTS = [
    ("M5 missing source -> dialog", test_m5_missing_source_is_a_dialog),
    ("M5 copytree failure -> dialog", test_m5_copytree_failure_is_a_dialog),
    ("M10 validate vanished-file", test_m10_validate_survive_vanished_file),
    ("H6 rename revert contained", test_h6_revert_failure_is_reported_not_crashed),
    ("R3-13a dead method removed", test_r3_13a_dead_add_hex_btn_removed),
]

if __name__ == "__main__":
    ok = 0
    for name, fn in TESTS:
        try:
            detail = fn()
            print("[PASS] %s -- %s" % (name, detail), flush=True)
            ok += 1
        except Exception as e:
            import traceback
            print("[FAIL] %s -- %r" % (name, e), flush=True)
            traceback.print_exc()
    print("\n%d/%d passed" % (ok, len(TESTS)), flush=True)
    sys.exit(0 if ok == len(TESTS) else 1)
