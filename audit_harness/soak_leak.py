"""Round 3 soak/leak harness (item 3).

Focus targets that reproduce only under repetition:
 - H4: a DesignBus watch abandoned WITHOUT close() (the makerchip dock is never
   registered in dock_dict, so Close Project never calls DesignBus.close()) —
   does the watchdog observer thread leak?
 - control: the same bus with a proper close() — threads must return to baseline.
 - NgspiceWidget process-registry growth across many runs (the _unregister_process
   claim) using the chaos stub.
"""
import os
import sys
import tempfile
import threading
import time
import gc

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SRC = os.path.join(REPO, "src")
STUB = os.path.join(HERE, "chaos_stub.py")

ISO = tempfile.mkdtemp(prefix="esim_soak_home_")
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


def thread_count():
    return threading.active_count()


def soak_designbus_no_close(n=30):
    from maker.DesignBus import DesignBus
    base = thread_count()
    peak = base
    for i in range(n):
        d = tempfile.mkdtemp(prefix="soakbus_")
        bus = DesignBus(0)
        bus.set_content("module m(); endmodule\n")
        bus.save_to_disk(os.path.join(d, "m%d.v" % i))  # starts observer thread
        # ABANDON without close() — exactly the H4 leak path
        del bus
        gc.collect()
        _app.processEvents()
        peak = max(peak, thread_count())
    time.sleep(0.5)
    gc.collect()
    return base, peak, thread_count()


def soak_designbus_with_close(n=30):
    from maker.DesignBus import DesignBus
    base = thread_count()
    peak = base
    for i in range(n):
        d = tempfile.mkdtemp(prefix="soakbus2_")
        bus = DesignBus(0)
        bus.set_content("module m(); endmodule\n")
        bus.save_to_disk(os.path.join(d, "m%d.v" % i))
        bus.close()   # proper teardown
        del bus
        gc.collect()
        _app.processEvents()
        peak = max(peak, thread_count())
    time.sleep(0.5)
    return base, peak, thread_count()


def soak_ngspice_registry(n=25):
    from configuration import Dialogs
    Dialogs.make_message_box = lambda parent=None, *a, **k: _Silent()
    Dialogs.make_error_message = lambda parent=None: _Silent()
    from configuration.Appconfig import Appconfig
    from maker import CosimConfig
    from ngspiceSimulation import NgspiceWidget as NW

    PROJ = tempfile.mkdtemp(prefix="soakproj_")
    NET = os.path.join(PROJ, "p.cir.out")
    with open(NET, "w") as f:
        f.write("* n\nr1 1 0 1k\n.end\n")
    Appconfig.current_project["ProjectName"] = PROJ
    Appconfig.proc_dict[PROJ] = []
    CosimConfig.ngspice_binary = lambda: sys.executable
    NW.NgspiceWidget._prepare_ngspice_arguments = (
        lambda self, netlist: [STUB, "ok", netlist])

    class SimEnd(QtCore.QObject):
        sig = pyqtSignal(object, object)

    for i in range(n):
        holder = SimEnd()
        done = {"v": False}
        holder.sig.connect(lambda *a: done.__setitem__("v", True))
        w = NW.NgspiceWidget(NET, holder.sig, plotFlag=False)
        deadline = time.time() + 6
        while time.time() < deadline and not done["v"]:
            _app.processEvents()
            time.sleep(0.01)
        w.setParent(None)
        w.deleteLater()
        _app.processEvents()
    return len(Appconfig.process_obj), len(Appconfig.proc_dict.get(PROJ, []))


class _Silent:
    def __getattr__(self, _):
        return lambda *a, **k: None


if __name__ == "__main__":
    print("== H4: DesignBus abandoned WITHOUT close() ==", flush=True)
    b, p, e = soak_designbus_no_close()
    print("   threads base=%d peak=%d end=%d  (leak=%d)" % (b, p, e, e - b),
          flush=True)

    print("== control: DesignBus WITH close() ==", flush=True)
    b, p, e = soak_designbus_with_close()
    print("   threads base=%d peak=%d end=%d  (leak=%d)" % (b, p, e, e - b),
          flush=True)

    print("== NgspiceWidget process-registry after 25 runs ==", flush=True)
    po, pd = soak_ngspice_registry()
    print("   process_obj=%d proc_dict[proj]=%d  (0 == no leak)" % (po, pd),
          flush=True)
    print("done", flush=True)
