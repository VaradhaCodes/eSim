from PyQt6 import QtWidgets, QtCore
import sys

app = QtWidgets.QApplication(sys.argv)
win = QtWidgets.QMainWindow()
d1 = QtWidgets.QDockWidget("Dock 1")
d2 = QtWidgets.QDockWidget("Dock 2")
win.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea, d1)
win.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea, d2)
win.tabifyDockWidget(d1, d2)

for tb in win.findChildren(QtWidgets.QTabBar):
    print("TabBar parent:", type(tb.parent()))
    print("Is QTabWidget?", isinstance(tb.parent(), QtWidgets.QTabWidget))

app.quit()
