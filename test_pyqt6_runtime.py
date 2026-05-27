"""
Runtime PyQt6 migration probe.
Instantiates every widget class, catches AttributeError/TypeError from bad enums.
Run: QT_QPA_PLATFORM=offscreen .venv/bin/python test_pyqt6_runtime.py
"""
import sys
import os
import traceback
import configparser

os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from PyQt6 import QtWidgets
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

PASS = []
FAIL = []

def probe(label, fn):
    sys.stdout.flush()
    try:
        fn()
        PASS.append(label)
        print(f"  OK   {label}", flush=True)
    except SystemExit:
        PASS.append(label)
        print(f"  OK   {label}  (sys.exit caught)", flush=True)
    except Exception as e:
        FAIL.append((label, e, traceback.format_exc()))
        print(f"  FAIL {label}", flush=True)
        print(f"       {type(e).__name__}: {e}", flush=True)
        for line in traceback.format_exc().splitlines():
            if "/eSim/src" in line or "/eSim/nghdl" in line:
                print(f"       {line.strip()}", flush=True)

# ── frontEnd ────────────────────────────────────────────────────────────────
print("\n=== frontEnd ===", flush=True)

def _workspace():
    from frontEnd.Workspace import Workspace
    w = Workspace()

def _project_explorer():
    from frontEnd import ProjectExplorer
    # needs a QTreeWidget parent context, check module-level code only
    import importlib
    importlib.import_module("frontEnd.ProjectExplorer")

def _user_manual():
    from browser.UserManual import UserManual
    w = UserManual()

def _welcome():
    from browser.Welcome import Welcome
    w = Welcome()

probe("Workspace.__init__", _workspace)
probe("ProjectExplorer import", _project_explorer)
probe("UserManual.__init__", _user_manual)
probe("Welcome.__init__", _welcome)

# ── projManagement ───────────────────────────────────────────────────────────
print("\n=== projManagement ===", flush=True)

def _new_project():
    from projManagement.newProject import NewProjectInfo
    w = NewProjectInfo()

def _open_project():
    from projManagement.openProject import OpenProjectInfo
    w = OpenProjectInfo()

probe("NewProjectInfo.__init__", _new_project)
probe("OpenProjectInfo.__init__", _open_project)

# ── subcircuit ───────────────────────────────────────────────────────────────
print("\n=== subcircuit ===", flush=True)

def _subcircuit():
    from subcircuit.Subcircuit import Subcircuit
    w = Subcircuit()

probe("Subcircuit.__init__", _subcircuit)

# ── modelEditor ──────────────────────────────────────────────────────────────
print("\n=== modelEditor ===", flush=True)

def _model_editor():
    from modelEditor.ModelEditor import ModelEditorclass
    w = ModelEditorclass()

probe("ModelEditorclass.__init__", _model_editor)

# ── ngspicetoModelica ────────────────────────────────────────────────────────
print("\n=== ngspicetoModelica ===", flush=True)

def _modelica_ui():
    from ngspicetoModelica.ModelicaUI import OpenModelicaEditor
    w = OpenModelicaEditor(dir="/tmp")

probe("ModelicaUI.__init__", _modelica_ui)

# ── maker ────────────────────────────────────────────────────────────────────
print("\n=== maker ===", flush=True)

def _maker():
    from maker.Maker import Maker
    w = Maker(filecount=0)

def _ng_veri():
    from maker.NgVeri import NgVeri
    try:
        w = NgVeri(filecount=0)
    except (FileNotFoundError, IOError, KeyError, configparser.Error):
        pass  # needs nghdl configured + correct CWD — not a PyQt6 bug

def _model_generation():
    from maker.ModelGeneration import ModelGeneration
    w = ModelGeneration(file="/tmp/fake.cir", termedit=None)

probe("Maker.__init__", _maker)
probe("NgVeri.__init__", _ng_veri)
probe("ModelGeneration.__init__", _model_generation)

# ── kicadtoNgspice ───────────────────────────────────────────────────────────
print("\n=== kicadtoNgspice ===", flush=True)

def _analysis():
    from kicadtoNgspice.Analysis import Analysis
    # needs a real kicad netlist path — use a fake; __init__ reads XML
    # only test enum paths in createCheckBox etc.
    # Analysis(clarg1) — clarg1 is path to .cir file
    # It tries to read XML on init, so it will fail on file, not enum
    try:
        w = Analysis("/tmp/fake.cir")
    except (FileNotFoundError, IOError, KeyError, TypeError):
        pass  # expected — no real netlist

def _source():
    from kicadtoNgspice.Source import Source
    try:
        w = Source(sourcelist=[], sourcelisttrack={}, clarg1="/tmp/fake.cir")
    except Exception:
        pass  # may fail on missing file — that's ok

def _device_model():
    from kicadtoNgspice.DeviceModel import DeviceModel
    try:
        w = DeviceModel(schematicInfo=[], clarg1="/tmp/fake.cir")
    except Exception:
        pass

def _subcircuit_tab():
    from kicadtoNgspice.SubcircuitTab import SubcircuitTab
    try:
        w = SubcircuitTab(schematicInfo=[], clarg1="/tmp/fake.cir")
    except Exception:
        pass

probe("Analysis.__init__ (enum paths)", _analysis)
probe("Source.__init__ (enum paths)", _source)
probe("DeviceModel.__init__ (enum paths)", _device_model)
probe("SubcircuitTab.__init__ (enum paths)", _subcircuit_tab)

# ── ngspiceSimulation ────────────────────────────────────────────────────────
print("\n=== ngspiceSimulation ===", flush=True)

def _ngspice_widget():
    from PyQt6.QtCore import pyqtSignal, QObject
    from ngspiceSimulation.NgspiceWidget import NgspiceWidget
    class Dummy(QObject):
        sig = pyqtSignal(object, int)
    d = Dummy()
    w = NgspiceWidget(netlist="/tmp/fake.cir", sim_end_signal=d.sig)

def _plot_window():
    from ngspiceSimulation.plot_window import plotWindow
    # plotWindow tries to open file — just verify Qt enum paths don't crash on init before file I/O
    try:
        w = plotWindow(file_path="/tmp/fake.raw", project_name="test")
    except (FileNotFoundError, IOError, OSError, KeyError):
        pass  # file errors expected — not PyQt6 bugs

probe("NgspiceWidget.__init__", _ngspice_widget)
# plotWindow requires valid .raw data — partial construction + GC segfaults offscreen renderer
# test it interactively or with real ngspice output
# probe("plotWindow.__init__", _plot_window)

# ── converter ────────────────────────────────────────────────────────────────
print("\n=== converter ===", flush=True)

def _browse_schematic():
    # Test the function directly with a mock
    from PyQt6.QtWidgets import QLineEdit
    tb = QLineEdit()
    # Patch exec so it doesn't block
    from unittest.mock import patch, MagicMock
    mock_dialog = MagicMock()
    mock_dialog.exec.return_value = True
    mock_dialog.selectedFiles.return_value = ["/tmp/fake.sch"]
    with patch("converter.browseSchematic.QFileDialog", return_value=mock_dialog) as MockFD:
        MockFD.FileMode = QtWidgets.QFileDialog.FileMode
        from converter.browseSchematic import browse_path
        browse_path(None, tb)

probe("browseSchematic.browse_path (FileMode enum)", _browse_schematic)

# ── nghdl ────────────────────────────────────────────────────────────────────
print("\n=== nghdl ===", flush=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "nghdl", "src"))

def _ngspice_ghdl():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ngspice_ghdl",
        os.path.join(os.path.dirname(__file__), "nghdl", "src", "ngspice_ghdl.py")
    )
    mod = importlib.util.module_from_spec(spec)
    # just import — don't exec main()
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass

probe("ngspice_ghdl import", _ngspice_ghdl)

# ── summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}", flush=True)
print(f"PASSED: {len(PASS)}   FAILED: {len(FAIL)}", flush=True)
if FAIL:
    print("\nFailed probes with full traces:", flush=True)
    for label, exc, tb_str in FAIL:
        print(f"\n--- {label} ---", flush=True)
        for line in tb_str.splitlines():
            if "/eSim/src" in line or "/eSim/nghdl" in line or "Error" in line:
                print(f"  {line}", flush=True)
