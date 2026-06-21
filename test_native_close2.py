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
    def handle(index, tb=tb):
        text = tb.tabText(index)
        print("Closing text:", repr(text))
        for d in win.findChildren(QtWidgets.QDockWidget):
            print("Dock title:", repr(d.windowTitle()))
            if d.windowTitle() == text:
                print("Found match, closing...")
                d.setVisible(False)
    tb.tabCloseRequested.connect(handle)

def check_visibility():
    tb = win.findChildren(QtWidgets.QTabBar)[0]
    tb.tabCloseRequested.emit(1)
    print(f"d1 visible: {d1.isVisible()}, d2 visible: {d2.isVisible()}")
    app.quit()

QtCore.QTimer.singleShot(500, check_visibility)
win.show()
app.exec()
