import sys
import os
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QApplication, QMainWindow, 
from PyQt6.QtCore import QTimer

sys.path.insert(0, os.path.abspath('src/frontEnd'))
sys.path.insert(0, os.path.abspath('src'))
from browser.Welcome import Welcome

class MockApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.kicad = ("KiCad", self)
        self.kicad.triggered.connect(lambda: print("KiCad Triggered!"))
        
        self.welcome = Welcome()
        self.setCentralWidget(self.welcome)
        
app = QApplication(sys.argv)
win = MockApp()

def test_click():
    print("Testing click on Welcome cards...")
    for i in range(win.welcome.grid.count()):
        widget = win.welcome.grid.itemAt(i).widget()
        if widget.attr_name == "kicad":
            # simulate click
            widget.trigger_callback("kicad")
            print("Callback manually fired")
            break
    QTimer.singleShot(1000, app.quit)

QTimer.singleShot(100, test_click)
win.show()
sys.exit(app.exec())
