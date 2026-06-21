import sys
from PyQt6.QtWidgets import QApplication, QTreeView
from PyQt6.QtGui import QPalette, QColor, QGuiApplication
from PyQt6.QtCore import Qt

app = QApplication(sys.argv)
app.setStyle("Fusion")

scheme = QGuiApplication.styleHints().colorScheme()
if scheme == Qt.ColorScheme.Dark:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#F9FAFB"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#F9FAFB"))
    palette.setColor(QPalette.ColorRole.Window, QColor("#111827"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#1F2937"))
    app.setPalette(palette)

tv = QTreeView()
tv.show()
sys.exit(0)
