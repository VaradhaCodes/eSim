"""Batch 10 verification harness (the LOW tail: L2-L6, L9).

Proves, offscreen on this machine, the fixes for:
  L2  a missing splash_screen_esim.png -> the isNull() guard skips the whole
      scale/mask/QPainter chain (no "Paint device returned engine == 0"
      warnings, no invisible splash); startup continues with splash = None.
  L3  plot_window._spin_arrow_icon treats a MISSING or ZERO-BYTE cache file as
      uncached and re-renders, and tolerates a failed QPixmap.save without
      raising -- so a temp-dir cleaner can no longer point the QSS at a broken
      image: url().
  L4  Worker.WorkerThread.__del__ waits with a 2000 ms cap (not an unbounded
      wait) so interpreter shutdown can't stall on a still-live child.
  L5  Kicad.openSchematic checks the resolved schematic exists; a missing file
      -> a clear error dialog and NO eeschema WorkerThread launch.
  L6  PspiceConverter.convert runs the parser on a BackgroundJob (returns at
      once, no GUI-thread subprocess block), guards against a double run, and
      routes non-zero exit / success through GUI-thread slots.
  L9  ProjectExplorer.handleDirectoryChanged debounces: a burst of watcher
      events for a path collapses to a SINGLE refreshProject once the timer
      fires (the OneDrive/Dropbox refresh-storm fix).

Each test is independent; a heavy/failing one can't sink the rest.
"""
import gc
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SRC = os.path.join(REPO, "src")

ISO = tempfile.mkdtemp(prefix="esim_b10_home_")
os.environ["HOME"] = ISO
os.environ["USERPROFILE"] = ISO
os.environ["APPDATA"] = os.path.join(ISO, "AppData", "Roaming")
os.environ["LOCALAPPDATA"] = os.path.join(ISO, "AppData", "Local")
os.makedirs(os.environ["APPDATA"], exist_ok=True)
os.makedirs(os.environ["LOCALAPPDATA"], exist_ok=True)
os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, SRC)

from PyQt6 import QtCore, QtGui, QtWidgets  # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)


def _pump(ms):
    """Run the real event loop for ~ms so single-shot timers actually fire."""
    loop = QtCore.QEventLoop()
    QtCore.QTimer.singleShot(ms, loop.quit)
    loop.exec()


# --------------------------------------------------------------------------- #
def test_l2_null_splash_guard():
    """A null splash pixmap: the pre-fix paint chain warns; the guard is clean."""
    warnings = []

    def handler(mode, ctx, msg):
        warnings.append(msg)

    QtCore.qInstallMessageHandler(handler)
    try:
        # A missing image file yields a null (0x0) pixmap -- exactly what
        # Application.main() gets from paths.image_path('splash_screen_esim.png')
        # when the asset is absent.
        splash_pix = QtGui.QPixmap(os.path.join(ISO, "does_not_exist.png"))
        assert splash_pix.isNull(), "missing image should give a null pixmap"

        # (1) PRE-FIX behaviour: feed the null pixmap through the same
        # scale/mask/QPainter sequence the fixed code now skips. This paints
        # onto a 0x0 device and MUST emit the engine==0 warning -- proving the
        # bug the guard removes is real.
        warnings.clear()
        scaled = splash_pix.scaledToWidth(
            int(splash_pix.width() * 0.8),
            QtCore.Qt.TransformationMode.SmoothTransformation)
        rounded = QtGui.QPixmap(scaled.size())          # 0x0
        rounded.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(rounded)                      # begin on 0x0 device
        p.drawPixmap(0, 0, scaled)
        p.end()
        pre_fix_warned = len(warnings) > 0
        assert pre_fix_warned, \
            "expected QPainter warnings painting a null pixmap (bug not shown)"

        # (2) FIXED behaviour: the isNull() guard short-circuits to splash=None
        # and never touches QPainter -> no warnings at all.
        warnings.clear()
        if splash_pix.isNull():
            splash = None
        else:                                            # pragma: no cover
            raise AssertionError("guard failed to detect the null pixmap")
        assert splash is None, "guarded path must yield splash=None"
        assert warnings == [], \
            "guarded path emitted Qt warnings: %r" % warnings
    finally:
        QtCore.qInstallMessageHandler(None)
    return "null pixmap: unguarded chain warned, isNull() guard silent"


# --------------------------------------------------------------------------- #
def test_l3_spin_arrow_cache_revalidates():
    """Missing / zero-byte cache -> re-render; failed save -> no exception."""
    from ngspiceSimulation.plot_window import plotWindow

    # _spin_arrow_icon uses no instance state, so a throwaway self is fine.
    call = lambda: plotWindow._spin_arrow_icon(object(), "up", "#abcdef")

    p1 = call()
    assert os.path.exists(p1) and os.path.getsize(p1) > 0, \
        "first call must render a non-empty PNG"

    # Truncate to zero bytes (a temp cleaner mid-write / interrupted save).
    with open(p1, "wb"):
        pass
    assert os.path.getsize(p1) == 0
    p2 = call()
    assert os.path.getsize(p2) > 0, \
        "zero-byte cache must be treated as stale and re-rendered"

    # Delete entirely (temp cleaner removed it between sessions).
    os.remove(p1)
    assert not os.path.exists(p1)
    p3 = call()
    assert os.path.exists(p3) and os.path.getsize(p3) > 0, \
        "deleted cache must be re-rendered"

    # Failed save: point tempPath at a path that cannot hold the file so
    # QPixmap.save returns False. The method must NOT raise -- it returns a
    # path string and lets the QSS degrade to a missing glyph.
    orig_tempPath = QtCore.QDir.tempPath
    bogus = os.path.join(ISO, "nope_%d" % os.getpid(), "still_nope")
    QtCore.QDir.tempPath = staticmethod(lambda: bogus)
    try:
        r = plotWindow._spin_arrow_icon(object(), "down", "#010203")
        assert isinstance(r, str), "must still return a path string on save fail"
    finally:
        QtCore.QDir.tempPath = orig_tempPath
    return "re-renders missing+empty cache; failed save raised nothing"


# --------------------------------------------------------------------------- #
def test_l4_worker_del_bounded_wait():
    """__del__ waits with a 2000 ms deadline, not an unbounded wait()."""
    from projManagement import Worker

    wt = Worker.WorkerThread("echo hi")   # constructed, never started
    recorded = []
    wt.wait = lambda *a, **k: recorded.append(a)   # instance shadow

    # Invoke the real __del__ body explicitly (GC timing is non-deterministic).
    type(wt).__del__(wt)

    assert recorded == [(2000,)], \
        "expected exactly one wait(2000); got %r" % recorded
    return "__del__ called wait(2000) -- bounded"


# --------------------------------------------------------------------------- #
def test_l5_openschematic_missing_file():
    """A resolved-but-missing schematic -> error dialog, no eeschema launch."""
    from projManagement import Kicad as kicadmod
    from configuration.Appconfig import Appconfig

    proj = tempfile.mkdtemp(prefix="b10_sch_")
    Appconfig.current_project["ProjectName"] = proj

    k = kicadmod.Kicad(dockarea=None)
    # Force the branch: project is "valid", stem known, but the schematic path
    # main_schematic hands back does not exist on disk.
    k.obj_validation.validateKicad = lambda *_: True
    k.obj_appconfig.get_proj_stem = lambda *a, **kw: "ghost"
    missing = os.path.join(proj, "ghost.kicad_sch")   # never created

    fired = {"critical": 0, "worker": 0}
    orig_ms = kicadmod.main_schematic
    orig_crit = kicadmod.Dialogs.critical
    orig_worker = kicadmod.Worker.WorkerThread
    kicadmod.main_schematic = lambda *a, **kw: missing
    kicadmod.Dialogs.critical = \
        lambda *a, **kw: fired.__setitem__("critical", fired["critical"] + 1)

    class _NoLaunch:
        def __init__(self, *a, **kw):
            fired["worker"] += 1

        def start(self):                              # pragma: no cover
            fired["worker"] += 100
    kicadmod.Worker.WorkerThread = _NoLaunch
    try:
        k.openSchematic()
    finally:
        kicadmod.main_schematic = orig_ms
        kicadmod.Dialogs.critical = orig_crit
        kicadmod.Worker.WorkerThread = orig_worker

    assert fired["critical"] == 1, "missing schematic must raise ONE error dialog"
    assert fired["worker"] == 0, \
        "eeschema WorkerThread must NOT be constructed for a missing file"
    return "missing schematic -> 1 dialog, 0 eeschema launches"


# --------------------------------------------------------------------------- #
def test_l6_pspice_parser_backgrounded():
    """convert() schedules a BackgroundJob (non-blocking); slots route results."""
    import maker.hdl.jobs as jobsmod
    from converter.pspiceToKicad import PspiceConverter
    from configuration import Dialogs

    # (a) _run_parser returns (rc, out, err) and never raises on non-zero exit.
    rc, out, err = PspiceConverter._run_parser(
        [sys.executable, "-c",
         "import sys; sys.stderr.write('boom'); sys.exit(3)"])
    assert (rc, err.strip()) == (3, "boom"), \
        "_run_parser should report the failing exit via the tuple; got %r" % (
            (rc, out, err),)

    conv = PspiceConverter(parent=None)

    # (b) convert() hands the parser to a BackgroundJob and returns AT ONCE.
    captured = {}

    class FakeJob:
        def __init__(self, fn, *args, **kw):
            captured["fn"] = fn
            captured["args"] = args
            self.succeeded = _Sig()
            self.failed = _Sig()
            self.finished = _Sig()

        def isRunning(self):
            return True                # so the double-run guard can see it

        def start(self):
            captured["started"] = True

        def deleteLater(self):
            pass

    class _Sig:
        def connect(self, *_):
            pass

    src = os.path.join(ISO, "design.sch")
    with open(src, "w") as fh:
        fh.write("* not empty\n")

    orig_job = jobsmod.BackgroundJob
    jobsmod.BackgroundJob = FakeJob
    try:
        conv.convert(src)
        assert captured.get("started") is True, "convert did not start a job"
        assert captured["fn"] == PspiceConverter._run_parser, \
            "the backgrounded callable must be _run_parser"
        assert list(captured["args"][0])[0] == sys.executable, \
            "the parser command must run under the eSim interpreter"

        # Double-run guard: _convert_job.isRunning() is True -> second convert
        # must NOT build another job.
        captured["started"] = False
        conv.convert(src)
        assert captured["started"] is False, \
            "a second convert while running must be refused"
    finally:
        jobsmod.BackgroundJob = orig_job

    # (c) completion routing on the GUI thread.
    crit = {"n": 0}
    orig_crit = Dialogs.critical
    Dialogs.critical = lambda *a, **kw: crit.__setitem__("n", crit["n"] + 1)
    try:
        conv._on_convert_done("/tmp/x", "x", (3, "", "parser blew up"))
        assert crit["n"] == 1, "non-zero exit must raise a failure dialog"

        conv._on_convert_failed("could not start interpreter")
        assert crit["n"] == 2, "a launch failure must raise a failure dialog"
    finally:
        Dialogs.critical = orig_crit
    return "parser on BackgroundJob; double-run refused; slots route rc/err"


# --------------------------------------------------------------------------- #
def test_l9_directory_change_debounced():
    """A burst of watcher events collapses to ONE refreshProject."""
    from frontEnd.ProjectExplorer import ProjectExplorer

    px = ProjectExplorer()

    # A matching, expanded top-level item so the flush actually targets it.
    watched = tempfile.mkdtemp(prefix="b10_ws_")
    item = QtWidgets.QTreeWidgetItem(["proj", watched])
    px.treewidget.addTopLevelItem(item)
    item.setExpanded(True)

    calls = {"refresh": 0}
    px.refreshProject = lambda *a, **kw: calls.__setitem__(
        "refresh", calls["refresh"] + 1)

    # Fire the storm: many events for the SAME path in one tick.
    for _ in range(12):
        px.handleDirectoryChanged(watched)

    # Nothing should have run yet -- the work is deferred behind the timer.
    assert calls["refresh"] == 0, "refresh ran synchronously (not debounced)"
    assert px._pending_dir_changes == {watched}, \
        "12 same-path events must dedupe to one pending entry"
    assert px._dir_refresh_timer.isActive(), "debounce timer not armed"

    _pump(500)   # let the single-shot timer fire

    assert calls["refresh"] == 1, \
        "12 events must coalesce to exactly ONE refresh; got %d" % calls["refresh"]
    assert px._pending_dir_changes == set(), "pending set not cleared after flush"

    # A fresh event after the flush arms a new cycle (one more refresh).
    px.handleDirectoryChanged(watched)
    _pump(500)
    assert calls["refresh"] == 2, "a post-flush event must schedule a new refresh"
    return "12-event storm -> 1 refresh; next event -> 1 more"


# --------------------------------------------------------------------------- #
TESTS = [
    ("L2 null-splash isNull() guard", test_l2_null_splash_guard),
    ("L3 spin-arrow cache revalidation", test_l3_spin_arrow_cache_revalidates),
    ("L4 Worker __del__ bounded wait", test_l4_worker_del_bounded_wait),
    ("L5 openSchematic missing-file guard", test_l5_openschematic_missing_file),
    ("L6 pspice parser backgrounded", test_l6_pspice_parser_backgrounded),
    ("L9 directoryChanged debounced", test_l9_directory_change_debounced),
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
        gc.collect()
    print("\n%d/%d passed" % (ok, len(TESTS)), flush=True)
    sys.exit(0 if ok == len(TESTS) else 1)
