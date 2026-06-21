import sys
import numpy as np
from PyQt6.QtWidgets import QApplication
sys.path.append('src')

# Mock to test plot_timing_diagram crash
class MockApp:
    pass

from ngspiceSimulation.plot_window import plotWindow
from maker.VerilogVerifier import VcdPlotWindow

app = QApplication(sys.argv)
timestamps = [0, 10, 20]
signals_data = {'a': [0, 1, 1], 'b': [1, 1, 0]}
signal_types = {'a': 'wire', 'b': 'wire'}

window = VcdPlotWindow(timestamps, signals_data, signal_types, "Verilog Sim", None)
window.show()

# Simulate ticking a signal
item = window.waveform_list.item(0)
window.on_waveform_toggle(item)

# Force the timer to fire synchronously for testing
window._refresh_timer.stop()
try:
    window.refresh_plot()
    print("refresh_plot completed successfully!")
except Exception as e:
    import traceback
    traceback.print_exc()
