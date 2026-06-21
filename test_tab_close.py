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
    tb.setTabsClosable(True)
    def handle_close(index, tb=tb):
        print(f"Close requested on tab {index}, text: {tb.tabText(index)}")
        # How to find the dock widget?
        # QTabBar doesn't expose the widgets directly, but QMainWindow does.
    tb.tabCloseRequested.connect(handle_close)

win.show()
QtCore.QTimer.singleShot(1000, app.quit)
app.exec()
