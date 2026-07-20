"""Round 3 process-lifecycle chaos harness (item 2).

Drives NgspiceWidget through a stub ngspice that (a) exits non-zero,
(b) hangs, (c) prints garbage bytes, (d) dies mid-output, plus
(e) FailedToStart. Counts finish_simulation invocations, sim_end_signal
emissions and dialogs — certifies the B3/H3 claims on the QProcess matrix.
"""
import os
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SRC = os.path.join(REPO, "src")
STUB = os.path.join(HERE, "chaos_stub.py")

ISO = tempfile.mkdtemp(prefix="esim_chaos_home_")
os.environ["HOME"] = ISO
os.environ["USERPROFILE"] = ISO
os.environ["APPDATA"] = os.path.join(ISO, "AppData", "Roaming")
os.environ["LOCALAPPDATA"] = os.path.join(ISO, "AppData", "Local")
os.makedirs(os.environ["APPDATA"], exist_ok=True)
os.makedirs(os.environ["LOCALAPPDATA"], exist_ok=True)
os.environ["QT_QPA_PLATFORM"] = "offscreen"

sys.path.insert(0, SRC)

from PyQt6 import QtWidgets, QtCore  # noqa: E402
from PyQt6.QtCore import pyqtSignal  # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

from configuration import Dialogs  # noqa: E402

DIALOGS = []


class _StubBox:
    def __init__(self):
        self._t = ""

    def setModal(self, *_):
        pass

    def setIcon(self, *_):
        pass

    def setWindowTitle(self, t):
        self._t = t

    def setText(self, t):
        DIALOGS.append(("box", str(t)[:100]))

    def setStandardButtons(self, *_):
        pass

    def showMessage(self, m):
        DIALOGS.append(("error", str(m)[:100]))

    def exec(self):
        return 0


Dialogs.make_message_box = lambda parent=None, *a, **k: _StubBox()
Dialogs.make_error_message = lambda parent=None: _StubBox()
Dialogs.warning = lambda parent, title, text, *a, **k: DIALOGS.append(
    ("warning", str(text)[:100]))

from configuration.Appconfig import Appconfig  # noqa: E402
from maker import CosimConfig  # noqa: E402
from ngspiceSimulation import NgspiceWidget as NW  # noqa: E402

PROJ = tempfile.mkdtemp(prefix="chaosproj_")
NETLIST = os.path.join(PROJ, "proj.cir.out")
with open(NETLIST, "w") as f:
    f.write("* chaos netlist\nr1 1 0 1k\n.tran 1m 2m\n.end\n")

Appconfig.current_project["ProjectName"] = PROJ
Appconfig.proc_dict[PROJ] = []


class SimEnd(QtCore.QObject):
    sig = pyqtSignal(object, object)


CALLS = {"finish": 0, "emit": 0}

_orig_finish = NW.NgspiceWidget.finish_simulation


def counting_finish(self, *a, **k):
    CALLS["finish"] += 1
    return _orig_finish(self, *a, **k)


NW.NgspiceWidget.finish_simulation = counting_finish


def run_case(name, mode, missing_bin=False, close_mid_run=False,
             wait_s=8.0):
    print("[START] %s" % name, flush=True)
    DIALOGS.clear()
    CALLS["finish"] = 0
    CALLS["emit"] = 0

    if missing_bin:
        CosimConfig.ngspice_binary = (
            lambda: os.path.join(PROJ, "no_such_ngspice.exe"))
        NW.NgspiceWidget._prepare_ngspice_arguments = (
            lambda self, netlist: ["-b", netlist])
    else:
        CosimConfig.ngspice_binary = lambda: sys.executable
        NW.NgspiceWidget._prepare_ngspice_arguments = (
            lambda self, netlist, _m=mode: [STUB, _m, netlist])

    holder = SimEnd()
    results = []
    holder.sig.connect(lambda st, code: (CALLS.__setitem__(
        "emit", CALLS["emit"] + 1), results.append((st, code))))

    try:
        print("[CONSTRUCT] building widget", flush=True)
        w = NW.NgspiceWidget(NETLIST, holder.sig, plotFlag=False)
        print("[CONSTRUCT] built", flush=True)
    except Exception:
        print("[RAISE-CONSTRUCT] %s\n%s" % (name, traceback.format_exc()))
        return

    deadline = time.monotonic() + wait_s
    closed = False
    while time.monotonic() < deadline:
        _app.processEvents()
        if close_mid_run and not closed and time.monotonic() > (
                deadline - wait_s + 1.5):
            # simulate the user clicking the tab's X mid-run: destroy the dock
            w.setParent(None)
            # deleteLater like _destroy_dock; then flush the DeferredDelete
            # event so the widget is REALLY destroyed. A bare processEvents()
            # never processes DeferredDelete, so the pre-fix harness left the
            # widget alive and only *labelled* it "widget-destroyed" -- the
            # top-level Qt event loop in the real app does flush it, which is
            # exactly when NgspiceWidget.destroyed must fire the completion
            # signal (B3).
            w.deleteLater()
            _app.processEvents()
            _app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
            _app.processEvents()
            closed = True
        if not close_mid_run and CALLS["emit"] > 0:
            # allow a grace period to catch DOUBLE emissions
            grace = time.monotonic() + 1.0
            while time.monotonic() < grace:
                _app.processEvents()
            break
        time.sleep(0.02)

    state = None
    try:
        state = w.process.state() if not closed else "widget-destroyed"
    except RuntimeError:
        state = "c++ deleted"

    # Kill leftover child if any (hang case)
    try:
        if not closed and w.process.state() != \
                QtCore.QProcess.ProcessState.NotRunning:
            w.process.kill()
            w.process.waitForFinished(3000)
    except RuntimeError:
        pass

    print("[CASE] %-38s finish_calls=%d emits=%d results=%s dialogs=%s "
          "procstate=%s"
          % (name, CALLS["finish"], CALLS["emit"],
             [(str(s).split('.')[-1], c) for s, c in results],
             [d[0] for d in DIALOGS], state), flush=True)


run_case("A. instant non-zero exit", "fail")
run_case("B. hard crash (abort)", "crash")
run_case("C. garbage bytes both channels", "garbage")
run_case("D. dies mid multibyte output", "midoutput")
run_case("E. binary missing (FailedToStart)", None, missing_bin=True)
run_case("F. clean success exit", "ok")
run_case("G. hang + user closes dock mid-run (B3)", "hang",
         close_mid_run=True, wait_s=5.0)
print("done")
