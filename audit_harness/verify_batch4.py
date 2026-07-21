"""Batch 4 verification harness (project/dock state + thread teardown).

Proves, offscreen on this machine, the fixes for:
  H1  kicadToNgspiceEditor uses the CAPTURED project; None -> info dialog, no crash
  H2  FullScreen exit after the host dock was destroyed -> no RuntimeError
  H4  the Model Creation dock is registered in dock_dict (Close Project reaps it)
  R3-2 closing that dock stops the DesignBus watchdog thread (no per-cycle leak)
  R3-12 dock destruction joins the verifier's worker thread (no "QThread destroyed
        while running") -- via the destroyed-slot teardown that closeEvent skips.

Each test is independent; a heavy/failing one can't sink the rest. Thread tests
compare against the count AFTER the widget is built (web-engine / framework
threads already up), so only the observer/worker delta is measured.
"""
import gc
import os
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SRC = os.path.join(REPO, "src")

ISO = tempfile.mkdtemp(prefix="esim_b4_home_")
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


def _threads():
    return threading.active_count()


def _flush_deletes():
    """Force the DeferredDelete a real event loop would flush, so destroyed
    fires now (a bare processEvents() never deletes deleteLater'd widgets)."""
    _app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    _app.processEvents()


# --------------------------------------------------------------------------- #
def test_h4_r3_2_dock_reaped_and_thread_freed():
    """makerchip dock is in dock_dict AND Close Project frees the watch thread."""
    from configuration.Appconfig import Appconfig
    from frontEnd.DockArea import DockArea

    proj = tempfile.mkdtemp(prefix="b4_maker_")
    Appconfig.current_project["ProjectName"] = proj
    Appconfig.dock_dict[proj] = []

    da = DockArea()
    da.makerchip()

    dock = da._tool_docks.get(("model-creation", proj))
    assert dock is not None, "Model Creation dock not registered in _tool_docks"

    registered = any(d is dock for d in Appconfig.dock_dict.get(proj, []))
    assert registered, "H4: dock NOT in dock_dict -> Close Project would skip it"

    # Start the watchdog observer (the leaking resource) by giving the bus a file.
    bus = dock._tool_widget.flow.bus
    mid = _threads()                       # widget fully built; web threads up
    bus.save_to_disk(os.path.join(proj, "m.v"))
    time.sleep(0.2)
    peak = _threads()
    assert peak > mid, "observer thread did not start (nothing to prove)"

    da.closeDock()                         # the exact Close Project path
    _flush_deletes()
    gc.collect()
    time.sleep(0.5)
    gc.collect()
    end = _threads()

    from PyQt6 import sip
    assert sip.isdeleted(dock), "dock survived Close Project"
    assert end <= mid, (
        "R3-2: watch thread leaked across Close Project "
        "(mid=%d peak=%d end=%d)" % (mid, peak, end))
    return "dock reaped; observer freed (mid=%d peak=%d end=%d)" % (mid, peak, end)


# --------------------------------------------------------------------------- #
def test_r3_12_factory_joins_and_reaps():
    """The destroyed-slot teardown cancels + joins a running job and reaps tmp."""
    from maker.VerilogVerifier import _make_verifier_teardown

    calls = {"cancel": 0, "wait": 0}

    class FakeJob:
        def isRunning(self):
            return True

        def wait(self, ms):
            calls["wait"] += 1
            return True

    class FakeCancel:
        def cancel(self):
            calls["cancel"] += 1

    tmp = tempfile.mkdtemp(prefix="b4_tmp_")
    state = {"job": FakeJob(), "cancel": FakeCancel(), "tmpdir": tmp}
    _make_verifier_teardown(state)()

    assert calls["cancel"] == 1, "cancel token not fired"
    assert calls["wait"] == 1, "worker thread not joined"
    assert not os.path.isdir(tmp), "tmpdir not reaped"
    assert state["job"] is None and state["tmpdir"] is None, "state not cleared"
    return "join+cancel+reap+clear all fired"


def test_r3_12_destroyed_triggers_teardown():
    """Destroying the real verifier fires the teardown (joins a live job)."""
    from maker.VerilogVerifier import VerilogVerifier

    calls = {"wait": 0}

    class FakeJob:
        def isRunning(self):
            return True

        def wait(self, ms):
            calls["wait"] += 1
            return True

    v = VerilogVerifier()
    v._teardown_state["job"] = FakeJob()
    v._teardown_state["cancel"] = None
    v.setParent(None)
    v.deleteLater()
    _flush_deletes()
    assert calls["wait"] == 1, (
        "destroyed did not run the worker join (wait=%d)" % calls["wait"])
    return "destroyed -> worker joined"


# --------------------------------------------------------------------------- #
def test_h2_fullscreen_exit_after_dock_destroyed():
    """Exiting fullscreen after the host dock died must not RuntimeError."""
    from frontEnd.FullScreen import FullScreenToggle
    from PyQt6 import sip

    win = QtWidgets.QMainWindow()
    dock = QtWidgets.QDockWidget("Panel")
    content = QtWidgets.QWidget()
    header = QtWidgets.QWidget()
    hl = QtWidgets.QHBoxLayout(header)
    toggle = FullScreenToggle()
    hl.addWidget(toggle)
    cl = QtWidgets.QVBoxLayout(content)
    cl.addWidget(header)
    dock.setWidget(content)
    win.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea, dock)

    toggle._enter()                        # reparents content into fullscreen win
    assert toggle._win is not None, "did not enter fullscreen"

    # Simulate Close Project destroying the now-empty dock while fullscreen.
    dock.setParent(None)
    dock.deleteLater()
    _flush_deletes()
    assert sip.isdeleted(dock), "dock not destroyed by the test"

    # Pre-fix this raised RuntimeError (setWidget on a deleted QDockWidget).
    toggle._exit()
    _flush_deletes()
    assert toggle._win is None and toggle._dock is None, "state not reset"
    return "fullscreen exit survived a destroyed host dock"


# --------------------------------------------------------------------------- #
def test_h1_converter_none_project_guard():
    """kicadToNgspiceEditor with no captured/live project -> info dialog, no crash."""
    from configuration import Dialogs
    from configuration.Appconfig import Appconfig
    from frontEnd.DockArea import DockArea

    fired = {"info": 0}
    orig = Dialogs.information
    Dialogs.information = lambda *a, **k: fired.__setitem__("info", fired["info"] + 1)
    try:
        Appconfig.current_project["ProjectName"] = None
        da = DockArea()
        da.kicadToNgspiceEditor("x.cir", projDir=None)  # nothing captured, none live
    finally:
        Dialogs.information = orig
    assert fired["info"] == 1, "no info dialog on closed-project convert"
    return "None project -> info dialog, no TypeError"


# --------------------------------------------------------------------------- #
def test_m4_refreshproject_guards():
    """Vanished folder -> False (openProject then bails); no-node -> no crash."""
    from configuration import Dialogs
    from frontEnd.ProjectExplorer import ProjectExplorer

    px = ProjectExplorer()
    orig = Dialogs.critical
    Dialogs.critical = lambda *a, **k: None
    try:
        missing = os.path.join(tempfile.gettempdir(),
                               "b4_no_such_%d" % os.getpid())
        assert px.refreshProject(missing) is False, \
            "vanished folder must return False (so openProject aborts)"
        # Existing folder but NO tree node and NO selection: pre-fix this hit
        # None.childCount() -> AttributeError.
        good = tempfile.mkdtemp(prefix="b4_m4_")
        open(os.path.join(good, "a.proj"), "w").close()
        assert px.refreshProject(good) is True, \
            "no-node refresh must return True, not AttributeError"
    finally:
        Dialogs.critical = orig
    return "vanished->False, no-node->True (parentnode None-guarded)"


def test_m7_newproject_rollback():
    """A write/registry failure mid-create rolls back the half-made project."""
    import projManagement.newProject as npmod
    from projManagement.newProject import NewProjectInfo
    from configuration import Dialogs
    from configuration.Appconfig import Appconfig

    ws = tempfile.mkdtemp(prefix="b4_ws_")
    npi = NewProjectInfo()
    npi.obj_appconfig.default_workspace['workspace'] = ws
    before = Appconfig.current_project.get('ProjectName')

    def _boom(*a, **k):
        raise OSError("simulated read-only workspace")

    orig_save, orig_crit = npmod.save_project_explorer, Dialogs.critical
    npmod.save_project_explorer = _boom          # fails INSIDE the guarded region
    Dialogs.critical = lambda *a, **k: None
    projname = "b4proj"
    try:
        r1, r2 = npi.createProject(projname)
    finally:
        npmod.save_project_explorer, Dialogs.critical = orig_save, orig_crit

    projdir = os.path.join(ws, projname)
    assert (r1, r2) == (None, None), "failed create must return (None, None)"
    assert not os.path.isdir(projdir), \
        "half-created folder not rolled back: %s" % projdir
    assert Appconfig.current_project.get('ProjectName') == before, \
        "must not switch current_project to the failed project"
    return "folder rolled back; current_project unchanged"


# --------------------------------------------------------------------------- #
TESTS = [
    ("H4/R3-2 dock reaped + thread freed", test_h4_r3_2_dock_reaped_and_thread_freed),
    ("R3-12 teardown factory join/reap", test_r3_12_factory_joins_and_reaps),
    ("R3-12 destroyed triggers teardown", test_r3_12_destroyed_triggers_teardown),
    ("H2 fullscreen exit vs dead dock", test_h2_fullscreen_exit_after_dock_destroyed),
    ("H1 converter None-project guard", test_h1_converter_none_project_guard),
    ("M4 refreshProject guards", test_m4_refreshproject_guards),
    ("M7 newProject rollback", test_m7_newproject_rollback),
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
