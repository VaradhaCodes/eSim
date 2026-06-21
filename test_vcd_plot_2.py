import sys
import numpy as np
from PyQt6.QtWidgets import QApplication
sys.path.append('src')

from ngspiceSimulation.plot_window import plotWindow
from maker.VerilogVerifier import VcdPlotWindow

app = QApplication(sys.argv)
timestamps = [0, 10, 20]
signals_data = {'a': [0, 1, 1], 'b': [1, 1, 0]}
signal_types = {'a': 'wire', 'b': 'wire'}

window = VcdPlotWindow(timestamps, signals_data, signal_types, "Verilog Sim", None)
window.show()

item = window.waveform_list.item(0)
window.on_waveform_toggle(item)

window._refresh_timer.stop()
window.refresh_plot()

print("Number of lines in axes:", len(window.axes.lines))
for line in window.axes.lines:
    print("Line xdata:", line.get_xdata())
    print("Line ydata:", line.get_ydata())
