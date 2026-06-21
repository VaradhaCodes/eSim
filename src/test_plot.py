import sys
import os
import tempfile
from PyQt6.QtWidgets import QApplication
from ngspiceSimulation.plot_window import plotWindow

app = QApplication(sys.argv)
tmpdir = tempfile.mkdtemp()
print("tmpdir:", tmpdir)
with open(os.path.join(tmpdir, "analysis"), "w") as f:
    f.write(".tran 1p 1n")
with open(os.path.join(tmpdir, "plot_data_i.txt"), "w") as f:
    f.write("")
with open(os.path.join(tmpdir, "plot_data_v.txt"), "w") as f:
    f.write("Transient Analysis\n")
    f.write("--------------------------------------------------------------------------------\n")
    f.write("Index\ttime\tsig1\n")
    f.write("--------------------------------------------------------------------------------\n")
    f.write("0\t0.0\t1\t\n")
    f.write("1\t1.0\t0\t\n")

print("Launching plotWindow...")
try:
    win = plotWindow(file_path=tmpdir, project_name="Test")
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
