"""Probe: why does the process registry retain entries after finished runs?"""
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SRC = os.path.join(REPO, "src")
STUB = os.path.join(HERE, "chaos_stub.py")

ISO = tempfile.mkdtemp(prefix="esim_probe_home_")
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


class _Silent:
    def __getattr__(self, _):
        return lambda *a, **k: None


from configuration import Dialogs  # noqa: E402
Dialogs.make_message_box = lambda parent=None, *a, **k: _Silent()
Dialogs.make_error_message = lambda parent=None: _Silent()
from configuration.Appconfig import Appconfig  # noqa: E402
from maker import CosimConfig  # noqa: E402
from ngspiceSimulation import NgspiceWidget as NW  # noqa: E402

PROJ = tempfile.mkdtemp(prefix="probeproj_")
NET = os.path.join(PROJ, "p.cir.out")
with open(NET, "w") as f:
    f.write("* n\nr1 1 0 1k\n.end\n")
Appconfig.current_project["ProjectName"] = PROJ
Appconfig.proc_dict[PROJ] = []
CosimConfig.ngspice_binary = lambda: sys.executable
NW.NgspiceWidget._prepare_ngspice_arguments = (
    lambda self, netlist: [STUB, "ok", netlist])

orig_unreg = NW.NgspiceWidget._unregister_process
orig_finish = NW.NgspiceWidget.finish_simulation

calls = {"finish": 0, "unreg": 0}


def spy_finish(self, *a, **k):
    calls["finish"] += 1
    return orig_finish(self, *a, **k)


def spy_unreg(self):
    calls["unreg"] += 1
    before = (len(Appconfig.process_obj), len(Appconfig.proc_dict[PROJ]))
    r = orig_unreg(self)
    after = (len(Appconfig.process_obj), len(Appconfig.proc_dict[PROJ]))
    print("   unreg: before=%s after=%s  self.process in obj: %s" %
          (before, after, self.process in Appconfig.process_obj), flush=True)
    return r


NW.NgspiceWidget.finish_simulation = spy_finish
NW.NgspiceWidget._unregister_process = spy_unreg


class SimEnd(QtCore.QObject):
    sig = pyqtSignal(object, object)


for i in range(3):
    holder = SimEnd()
    done = {"v": False}
    holder.sig.connect(lambda *a: done.__setitem__("v", True))
    w = NW.NgspiceWidget(NET, holder.sig, plotFlag=False)
    deadline = time.time() + 6
    while time.time() < deadline and not done["v"]:
        _app.processEvents()
        time.sleep(0.01)
    print("run %d: done=%s finish_calls=%d unreg_calls=%d obj=%d dict=%d" % (
        i, done["v"], calls["finish"], calls["unreg"],
        len(Appconfig.process_obj), len(Appconfig.proc_dict[PROJ])), flush=True)
    w.setParent(None)
    w.deleteLater()
    _app.processEvents()

print("final process_obj ids:", [id(p) for p in Appconfig.process_obj],
      flush=True)
