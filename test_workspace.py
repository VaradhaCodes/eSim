import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

sys.path.insert(0, os.path.abspath('src/frontEnd'))
sys.path.insert(0, os.path.abspath('src'))
from Application import Application

app = QApplication(sys.argv)
appView = Application()
appView.splash = appView # Mock splash
appView.obj_workspace.returnWhetherClickedOrNot(appView)

def click_ok():
    print("Clicking OK...")
    appView.obj_workspace.okbtn.click()
    print("OK clicked.")
    QTimer.singleShot(1000, app.quit)

QTimer.singleShot(100, click_ok)
appView.obj_workspace.show()
sys.exit(app.exec())
