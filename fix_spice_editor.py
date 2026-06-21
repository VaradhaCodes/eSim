import re

with open("src/frontEnd/SpiceEditor.py", "r") as f:
    c = f.read()

# 1. Remove old icon factories (lines 15 to 75 approx)
c = re.sub(r'# ---------------------------------------------------------------- glyph icons.*?class SpiceHighlighter', 'class SpiceHighlighter', c, flags=re.DOTALL)

# 2. Add zoomChanged signal to SpiceCodeEditor
c = re.sub(r'(class SpiceCodeEditor\(QtWidgets\.QPlainTextEdit\):\n\s+)', r'\1zoomChanged = QtCore.pyqtSignal(int)\n    ', c)

# 3. Emit zoomChanged
c = re.sub(r'(def _apply_font_size\(self\):.*?self\.setFont\(font\))', r'\1\n        self.zoomChanged.emit(self._font_size)', c, flags=re.DOTALL)

# 4. Replace FindReplaceDialog
new_dialog = '''class FindReplaceDialog(QtWidgets.QDialog):
    def __init__(self, editor, parent=None):
        super().__init__(parent)
        self.editor = editor
        self.setWindowTitle("Find and Replace")
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowType.Tool)
        self.setMinimumWidth(380)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)
        
        form_layout = QtWidgets.QFormLayout()
        
        self.find_input = QtWidgets.QLineEdit()
        self.find_input.setPlaceholderText("Search...")
        self.replace_input = QtWidgets.QLineEdit()
        self.replace_input.setPlaceholderText("Replace with...")
        
        form_layout.addRow("Find:", self.find_input)
        form_layout.addRow("Replace:", self.replace_input)
        layout.addLayout(form_layout)
        
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_find = QtWidgets.QPushButton("Find Next")
        self.btn_replace = QtWidgets.QPushButton("Replace")
        self.btn_replace_all = QtWidgets.QPushButton("Replace All")
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_find)
        btn_layout.addWidget(self.btn_replace)
        btn_layout.addWidget(self.btn_replace_all)
        
        layout.addLayout(btn_layout)
        
        self.btn_find.clicked.connect(self.find_next)
        self.btn_replace.clicked.connect(self.replace)
        self.btn_replace_all.clicked.connect(self.replace_all)'''
c = re.sub(r'class FindReplaceDialog\(QtWidgets\.QDialog\):.*?def find_next', new_dialog + '\n        \n    def find_next', c, flags=re.DOTALL)

# 5. Replace _setup_toolbar
new_toolbar = '''    def _setup_toolbar(self):
        self.toolbar = QtWidgets.QToolBar("Editor Actions")
        self.toolbar.setMovable(False)
        self.toolbar.setIconSize(QtCore.QSize(18, 18))
        self.addToolBar(self.toolbar)

        self.action_save = QtGui.QAction("Save", self)
        self.action_save.setShortcut("Ctrl+S")
        self.action_save.setToolTip("Save (Ctrl+S)")
        self.action_save.triggered.connect(self.save_file)
        self.toolbar.addAction(self.action_save)

        self.action_save_as = QtGui.QAction("Save As…", self)
        self.action_save_as.setShortcut("Ctrl+Shift+S")
        self.action_save_as.setToolTip("Save As… (Ctrl+Shift+S)")
        self.action_save_as.triggered.connect(self.save_file_as)
        self.toolbar.addAction(self.action_save_as)

        self.toolbar.addSeparator()

        self.action_find = QtGui.QAction("Find / Replace", self)
        self.action_find.setShortcut("Ctrl+F")
        self.action_find.setToolTip("Find / Replace (Ctrl+F)")
        self.action_find.triggered.connect(self.open_find)
        self.toolbar.addAction(self.action_find)

        self.action_wrap = QtGui.QAction("Word wrap", self)
        self.action_wrap.setCheckable(True)
        self.action_wrap.setToolTip("Toggle soft word-wrap inside the editor")
        self.action_wrap.toggled.connect(self._toggle_wrap)
        self.toolbar.addAction(self.action_wrap)

        self.toolbar.addSeparator()

        self.zoom_out_btn = QtWidgets.QToolButton()
        self.zoom_out_btn.setText(" - ")
        self.zoom_out_btn.setToolTip("Zoom Out (Ctrl+-)")
        self.zoom_out_btn.clicked.connect(lambda: self.editor.zoomOut(1))
        
        self.zoom_label = QtWidgets.QLabel(" 100% ")
        self.zoom_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.zoom_label.setMinimumWidth(45)
        
        self.zoom_in_btn = QtWidgets.QToolButton()
        self.zoom_in_btn.setText(" + ")
        self.zoom_in_btn.setToolTip("Zoom In (Ctrl++)")
        self.zoom_in_btn.clicked.connect(lambda: self.editor.zoomIn(1))

        self.toolbar.addWidget(self.zoom_out_btn)
        self.toolbar.addWidget(self.zoom_label)
        self.toolbar.addWidget(self.zoom_in_btn)

        self.editor.zoomChanged.connect(self._update_zoom_label)

        # Status bar
        self.status = QtWidgets.QStatusBar(self)
        self.setStatusBar(self.status)
        self._cursor_label = QtWidgets.QLabel("Ln 1, Col 1")
        self._enc_label = QtWidgets.QLabel("UTF-8")
        self.status.addPermanentWidget(self._cursor_label)
        self.status.addPermanentWidget(self._enc_label)
        self.editor.cursorPositionChanged.connect(self._refresh_cursor_label)
        self._refresh_cursor_label()

    def _update_zoom_label(self, font_size):
        pct = int((font_size / 11.0) * 100)
        self.zoom_label.setText(f" {pct}% ")'''
c = re.sub(r'    def _setup_toolbar\(self\):.*?    def _load_file\(self\):', new_toolbar + '\n\n    def _load_file(self):', c, flags=re.DOTALL)

# 6. Add SVG switching to update_theme_styles
svg_switch = '''        # Update SVGs based on theme
        icon_color = "dark" if is_dark else "light"
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.action_save.setIcon(QtGui.QIcon(os.path.join(base_dir, f"images/text_save_{icon_color}.svg")))
        self.action_save_as.setIcon(QtGui.QIcon(os.path.join(base_dir, f"images/text_save_as_{icon_color}.svg")))
        self.action_find.setIcon(QtGui.QIcon(os.path.join(base_dir, f"images/text_find_{icon_color}.svg")))
        self.action_wrap.setIcon(QtGui.QIcon(os.path.join(base_dir, f"images/text_wrap_{icon_color}.svg")))

        self.editor.setStyleSheet'''
c = c.replace('        self.editor.setStyleSheet', svg_switch)


with open("src/frontEnd/SpiceEditor.py", "w") as f:
    f.write(c)

print("Modifications applied successfully.")
