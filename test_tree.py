import sys
from PyQt6.QtWidgets import QApplication, QTreeView
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QPalette, QColor, QGuiApplication
from PyQt6.QtCore import Qt

app = QApplication(sys.argv)
app.setStyle("Fusion")

palette = QPalette()
palette.setColor(QPalette.ColorRole.WindowText, QColor("#F9FAFB"))
palette.setColor(QPalette.ColorRole.Text, QColor("#F9FAFB"))
palette.setColor(QPalette.ColorRole.Window, QColor("#111827"))
palette.setColor(QPalette.ColorRole.Base, QColor("#1F2937"))
app.setPalette(palette)

# If we set stylesheet, native branches disappear
# app.setStyleSheet("QTreeView { background-color: #1F2937; }")

tv = QTreeView()
model = QStandardItemModel()
parentItem = model.invisibleRootItem()
item = QStandardItem("Parent")
item.appendRow(QStandardItem("Child"))
parentItem.appendRow(item)
tv.setModel(model)
tv.show()
sys.exit(0)
