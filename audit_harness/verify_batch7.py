"""Batch 7 verification harness (startup + optional-dependency resilience).

Proves, offscreen on this machine, the fixes for:
  H7   a corrupt workspace.txt check-token no longer aborts startup -- both at
       the source (read_workspace clamps) and at the crash line (Workspace's
       Qt.CheckState() can't throw).
  M8   with watchdog PRESENT the external-edit watch still starts (the guard did
       not break the normal path). The watchdog-ABSENT path is proven by the
       dedicated smoke_no_watchdog.py.
  R2-7 pspiceToKicad no longer imports frontEnd.ProjectExplorer (which dragged
       the whole editor/Qsci chain into the Schematic Converter).

Each test is independent; a failing one can't sink the rest.
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SRC = os.path.join(REPO, "src")

ISO = tempfile.mkdtemp(prefix="esim_b7_home_")
os.environ["HOME"] = ISO
os.environ["USERPROFILE"] = ISO
os.environ["APPDATA"] = os.path.join(ISO, "AppData", "Roaming")
os.environ["LOCALAPPDATA"] = os.path.join(ISO, "AppData", "Local")
os.makedirs(os.environ["APPDATA"], exist_ok=True)
os.makedirs(os.environ["LOCALAPPDATA"], exist_ok=True)
os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, SRC)

from PyQt6 import QtCore, QtWidgets  # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

failures = []


def _check(name, cond, detail=""):
    if cond:
        print("[OK]     %s" % name)
    else:
        failures.append("%s %s" % (name, detail))
        print("[FAIL]   %s %s" % (name, detail))


def _write_workspace(line):
    cfg = os.path.join(ISO, ".esim")
    os.makedirs(cfg, exist_ok=True)
    with open(os.path.join(cfg, "workspace.txt"), "w", encoding="utf-8") as fh:
        fh.write(line)


# --------------------------------------------------------------------------- #
def test_h7_read_workspace_clamps():
    """A corrupt/out-of-range check token is clamped to '0'; path is preserved.
    A valid '0'/'2' token passes through untouched."""
    from configuration import paths
    path = os.path.join(ISO, "ws with space")
    cases = {
        "5 %s" % path: ("0", path),      # numeric but out of the CheckState set
        "x %s" % path: ("0", path),      # non-numeric -- old int() ValueError
        "-1 %s" % path: ("0", path),
        "0 %s" % path: ("0", path),      # valid Unchecked, unchanged
        "2 %s" % path: ("2", path),      # valid Checked, unchanged
    }
    for line, (want_check, want_path) in cases.items():
        _write_workspace(line)
        got_check, got_path = paths.read_workspace()
        _check("H7 read_workspace(%r)" % line.split(" ", 1)[0],
               got_check == want_check and got_path == want_path,
               "-> (%r, %r), wanted (%r, %r)"
               % (got_check, got_path, want_check, want_path))


def test_h7_dialog_builds_with_bad_check():
    """Even if workspace_check somehow holds a bad value, building the Workspace
    dialog must not raise (defense-in-depth at the Qt.CheckState crash line)."""
    from configuration.Appconfig import Appconfig
    from frontEnd import Workspace
    Appconfig.load_workspace()               # seed home / paths
    for bad in ("5", "x", "-1", "999"):
        Appconfig.workspace_check = bad
        try:
            dlg = Workspace.Workspace()
            state = dlg.chkbox.checkState()
            dlg.deleteLater()
            _check("H7 Workspace() with workspace_check=%r" % bad,
                   state == QtCore.Qt.CheckState.Unchecked,
                   "-> checkState=%r" % state)
        except Exception as exc:
            _check("H7 Workspace() with workspace_check=%r" % bad, False,
                   "-> raised %r" % (exc,))
    Appconfig.workspace_check = "0"


def test_m8_watch_starts_when_watchdog_present():
    """With watchdog installed the observer must still start (guard is a no-op
    on the happy path) and close cleanly."""
    from maker import DesignBus
    _check("M8 _HAS_WATCHDOG is True here", DesignBus._HAS_WATCHDOG is True)
    _check("M8 _DiskWatchHandler is a real class",
           DesignBus._DiskWatchHandler is not None)
    bus = DesignBus.DesignBus(0)
    target = os.path.join(ISO, "design.v")
    with open(target, "w", encoding="utf-8") as fh:
        fh.write("module m; endmodule\n")
    bus._path = target                       # set directly, skip Maker mirror
    try:
        bus._start_watch()
        _check("M8 observer started with watchdog present",
               bus._observer is not None)
    finally:
        bus.close()
    _check("M8 close() stops and clears the observer", bus._observer is None)


def test_r2_7_pspice_drops_projectexplorer():
    """The dead ProjectExplorer import is gone from pspiceToKicad, and the
    module imports without it."""
    src_file = os.path.join(SRC, "converter", "pspiceToKicad.py")
    with open(src_file, encoding="utf-8") as fh:
        text = fh.read()
    _check("R2-7 no ProjectExplorer reference in pspiceToKicad.py source",
           "ProjectExplorer" not in text)
    try:
        from converter import pspiceToKicad  # noqa: F401
        _check("R2-7 converter.pspiceToKicad imports cleanly", True)
    except Exception as exc:
        _check("R2-7 converter.pspiceToKicad imports cleanly", False,
               "-> %r" % (exc,))


for test in (test_h7_read_workspace_clamps,
             test_h7_dialog_builds_with_bad_check,
             test_m8_watch_starts_when_watchdog_present,
             test_r2_7_pspice_drops_projectexplorer):
    try:
        test()
    except Exception as exc:
        failures.append("%s crashed: %r" % (test.__name__, exc))
        print("[FAIL]   %s crashed: %r" % (test.__name__, exc))

print("\nRESULT: %s" % ("FAIL (%d)" % len(failures) if failures else "PASS"))
for line in failures:
    print("  - " + line)
sys.exit(1 if failures else 0)
