import os
import re
from PyQt6 import QtCore, QtGui, QtWidgets
from configuration.Appconfig import Appconfig

# Theme-aware icons for the editor toolbar. We reuse the same icon factories
# used by the dock chrome so Save / Find / Zoom stay legible in both themes.
try:
    from frontEnd.icon_paths import copy_icon, refresh_icon  # noqa: E402
except Exception:  # pragma: no cover — running outside frontEnd package
    copy_icon = None
    refresh_icon = None


class SpiceHighlighter(QtGui.QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlightingRules = []
        self._setup_rules()
        self.update_colors()

    def _setup_rules(self):
        # We store just the regex patterns here. Colors are assigned in update_colors()
        self.rules_def = {
            'component': [
                r'(?i)^\s*[RCLVIDQMJXKSWGEFHU][a-zA-Z0-9_]*\b', # eSim components start with these letters
            ],
            'control': [
                r'(?i)^\s*(run|plot|print|set|let|listing|help|destroy|display|quit|source|hardcopy|op|dc|ac|tran)\b'
            ],
            'keyword': [
                r'(?i)\B\.[a-zA-Z_]+\b', # SPICE directives like .model, .subckt, .tran
            ],
            'number': [
                r'(?i)\b[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?(meg|k|m|u|n|p|f|t|g)?\b'
            ],
            'comment': [
                r'^\s*\*.*',  # Standard SPICE comment
                r';.*'     # NGSpice inline comment
            ]
        }

    def update_colors(self):
        prefs = Appconfig().load_preferences()
        theme_mode = prefs.get("theme_mode", "System")
        
        if theme_mode == "Dark":
            is_dark = True
        elif theme_mode == "Light":
            is_dark = False
        else:
            is_dark = QtGui.QGuiApplication.styleHints().colorScheme() == QtCore.Qt.ColorScheme.Dark

        self.highlightingRules = []

        # Component Format
        compFormat = QtGui.QTextCharFormat()
        compFormat.setForeground(QtGui.QColor("#7DD3FC" if is_dark else "#006D9C"))
        compFormat.setFontWeight(QtGui.QFont.Weight.Bold)
        for pattern in self.rules_def['component']:
            self.highlightingRules.append((QtCore.QRegularExpression(pattern, QtCore.QRegularExpression.PatternOption.CaseInsensitiveOption), compFormat))

        # Control Commands Format
        controlFormat = QtGui.QTextCharFormat()
        controlFormat.setForeground(QtGui.QColor("#C4B5FD" if is_dark else "#6D28D9"))
        controlFormat.setFontWeight(QtGui.QFont.Weight.Bold)
        for pattern in self.rules_def['control']:
            self.highlightingRules.append((QtCore.QRegularExpression(pattern, QtCore.QRegularExpression.PatternOption.CaseInsensitiveOption), controlFormat))

        # Keyword Format (Directives)
        keywordFormat = QtGui.QTextCharFormat()
        keywordFormat.setForeground(QtGui.QColor("#7DD3FC" if is_dark else "#006D9C"))
        keywordFormat.setFontWeight(QtGui.QFont.Weight.Bold)
        for pattern in self.rules_def['keyword']:
            self.highlightingRules.append((QtCore.QRegularExpression(pattern, QtCore.QRegularExpression.PatternOption.CaseInsensitiveOption), keywordFormat))

        # Number Format
        numberFormat = QtGui.QTextCharFormat()
        numberFormat.setForeground(QtGui.QColor("#A7F3D0" if is_dark else "#047857"))
        for pattern in self.rules_def['number']:
            self.highlightingRules.append((QtCore.QRegularExpression(pattern, QtCore.QRegularExpression.PatternOption.CaseInsensitiveOption), numberFormat))

        # Comment Format
        commentFormat = QtGui.QTextCharFormat()
        commentFormat.setForeground(QtGui.QColor("#64748B" if is_dark else "#6B7280"))
        commentFormat.setFontItalic(True)
        for pattern in self.rules_def['comment']:
            self.highlightingRules.append((QtCore.QRegularExpression(pattern), commentFormat))

        self.rehighlight()

    def highlightBlock(self, text):
        for pattern, format in self.highlightingRules:
            iterator = pattern.globalMatch(text)
            while iterator.hasNext():
                match = iterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), format)


class LineNumberArea(QtWidgets.QWidget):
    def __init__(self, editor):
        super().__init__(editor)
        self.codeEditor = editor

    def sizeHint(self):
        return QtCore.QSize(self.codeEditor.lineNumberAreaWidth(), 0)

    def paintEvent(self, event):
        self.codeEditor.lineNumberAreaPaintEvent(event)


class SpiceCodeEditor(QtWidgets.QPlainTextEdit):
    zoomChanged = QtCore.pyqtSignal(int)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.lineNumberArea = LineNumberArea(self)
        self.blockCountChanged.connect(self.updateLineNumberAreaWidth)
        self.updateRequest.connect(self.updateLineNumberArea)
        self.updateLineNumberAreaWidth(0)

        # Standard font — portable monospace (Consolas is Windows-only).
        try:
            from frontEnd.widgets import mono_family
            _mono = mono_family()
        except Exception:
            _mono = "Consolas"
        font = QtGui.QFont(_mono, 11)
        font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        self.setFont(font)
        self._font_size = 11
        
        # Word wrap off by default for netlists
        self.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)

        self.highlighter = SpiceHighlighter(self.document())

    def lineNumberAreaWidth(self):
        digits = 1
        max_num = max(1, self.blockCount())
        while max_num >= 10:
            max_num //= 10
            digits += 1
        space = 10 + self.fontMetrics().horizontalAdvance('9') * digits
        return space

    def updateLineNumberAreaWidth(self, _):
        self.setViewportMargins(self.lineNumberAreaWidth(), 0, 0, 0)

    def updateLineNumberArea(self, rect, dy):
        if dy:
            self.lineNumberArea.scroll(0, dy)
        else:
            self.lineNumberArea.update(0, rect.y(), self.lineNumberArea.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self.updateLineNumberAreaWidth(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cr = self.contentsRect()
        self.lineNumberArea.setGeometry(QtCore.QRect(cr.left(), cr.top(), self.lineNumberAreaWidth(), cr.height()))

    def lineNumberAreaPaintEvent(self, event):
        prefs = Appconfig().load_preferences()
        theme_mode = prefs.get("theme_mode", "System")
        
        if theme_mode == "Dark":
            is_dark = True
        elif theme_mode == "Light":
            is_dark = False
        else:
            is_dark = QtGui.QGuiApplication.styleHints().colorScheme() == QtCore.Qt.ColorScheme.Dark

        bg_color = QtGui.QColor("#0D182B" if is_dark else "#EDF4FA")
        bg1 = QtGui.QColor("#09111F" if is_dark else "#F8FBFF")
        num = QtGui.QColor("#6B7F99" if is_dark else "#8A99AE")
        active = QtGui.QColor("#53D7FF" if is_dark else "#0077A8")

        painter = QtGui.QPainter(self.lineNumberArea)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(event.rect(), bg_color)

        current_block = self.textCursor().blockNumber()
        block = self.firstVisibleBlock()
        blockNumber = block.blockNumber()
        top = int(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + int(self.blockBoundingRect(block).height())

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(blockNumber + 1)
                if blockNumber == current_block:
                    # Pill highlight behind current line number
                    pill_rect = QtCore.QRect(4, top + 1, self.lineNumberArea.width() - 8, self.fontMetrics().height())
                    if is_dark:
                        painter.setBrush(QtGui.QColor(83, 215, 255, 32))
                    else:
                        painter.setBrush(QtGui.QColor(0, 119, 168, 22))
                    painter.setPen(QtCore.Qt.PenStyle.NoPen)
                    painter.drawRoundedRect(pill_rect, 6, 6)
                    painter.setPen(active)
                    f = QtGui.QFont(self.font())
                    f.setBold(True)
                    painter.setFont(f)
                else:
                    painter.setPen(num)
                    painter.setFont(self.font())
                painter.drawText(0, top, self.lineNumberArea.width() - 6, self.fontMetrics().height(),
                                 QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter, number)

            block = block.next()
            if not block.isValid():
                break
            top = bottom
            bottom = top + int(self.blockBoundingRect(block).height())
            blockNumber += 1

    def _highlight_current_line(self):
        selections = []
        if not self.isReadOnly():
            dark = self.palette().color(QtGui.QPalette.ColorRole.Window).lightness() < 128
            sel = QtWidgets.QTextEdit.ExtraSelection()
            sel.format.setBackground(QtGui.QColor(83, 215, 255, 14) if dark else QtGui.QColor(0, 119, 168, 14))
            sel.format.setProperty(QtGui.QTextFormat.Property.FullWidthSelection, True)
            sel.cursor = self.textCursor()
            sel.cursor.clearSelection()
            selections.append(sel)
        self.setExtraSelections(selections)

    def zoomIn(self, range=1):
        self._font_size += range
        self._apply_font_size()
        
    def zoomOut(self, range=1):
        self._font_size = max(6, self._font_size - range)
        self._apply_font_size()
        
    def _apply_font_size(self):
        font = self.font()
        font.setPointSize(self._font_size)
        self.setFont(font)
        self.zoomChanged.emit(self._font_size)
        # Notify the parent window to re-apply the stylesheet with the new font size
        parent = self.parent()
        while parent:
            if hasattr(parent, 'update_theme_styles'):
                parent.update_theme_styles()
                break
            parent = parent.parent()
        self.updateLineNumberAreaWidth(0)

    def wheelEvent(self, event):
        if event.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier:
            if event.angleDelta().y() > 0:
                self.zoomIn(1)
            else:
                self.zoomOut(1)
            return
        super().wheelEvent(event)





class SpiceEditorWindow(QtWidgets.QMainWindow):
    def __init__(self, filepath, parent=None):
        super().__init__(parent)
        self.filepath = os.path.abspath(filepath)
        self.is_modified = False
        
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QtWidgets.QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        self.central_widget.setObjectName("spiceEditorRoot")

        self.editor = SpiceCodeEditor()
        
        self.setWindowTitle(f"{os.path.basename(self.filepath)} - eSim Text Editor")
        self.resize(800, 600)
        
        self._setup_toolbar()
        self._setup_find_toolbar()
        
        self.main_layout.addWidget(self.find_widget)
        self.main_layout.addWidget(self.editor)
        
        self._load_file()
        
        self.editor.textChanged.connect(self._on_text_changed)
        
        # Initial Theme
        self.update_theme_styles()
        
        from frontEnd.motion import install_button_motion
        install_button_motion(self)
        
        # Current-line highlight
        self.editor.cursorPositionChanged.connect(self.editor._highlight_current_line)
        self.editor._highlight_current_line()

    def _setup_toolbar(self):
        self.toolbar = QtWidgets.QToolBar("Editor Actions")
        self.toolbar.setObjectName("spiceEditorToolbar")
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
        self.zoom_out_btn.setProperty("cssClass", "toolbarZoom")
        self.zoom_out_btn.setToolTip("Zoom Out (Ctrl+-)")
        self.zoom_out_btn.clicked.connect(lambda: self.editor.zoomOut(1))
        
        self.zoom_label = QtWidgets.QLabel(" 100% ")
        self.zoom_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.zoom_label.setMinimumWidth(45)
        self.zoom_label.setProperty("cssClass", "subtle")
        
        self.zoom_in_btn = QtWidgets.QToolButton()
        self.zoom_in_btn.setText(" + ")
        self.zoom_in_btn.setProperty("cssClass", "toolbarZoom")
        self.zoom_in_btn.setToolTip("Zoom In (Ctrl++)")
        self.zoom_in_btn.clicked.connect(lambda: self.editor.zoomIn(1))

        self.toolbar.addWidget(self.zoom_out_btn)
        self.toolbar.addWidget(self.zoom_label)
        self.toolbar.addWidget(self.zoom_in_btn)

        self.editor.zoomChanged.connect(self._update_zoom_label)

        # Status bar
        self.status = QtWidgets.QStatusBar(self)
        self.status.setObjectName("spiceEditorStatus")
        self.setStatusBar(self.status)
        self._cursor_label = QtWidgets.QLabel("Ln 1, Col 1")
        self._enc_label = QtWidgets.QLabel("UTF-8")
        self.status.addPermanentWidget(self._cursor_label)
        self.status.addPermanentWidget(self._enc_label)
        self.editor.cursorPositionChanged.connect(self._refresh_cursor_label)
        self._refresh_cursor_label()

    def _update_zoom_label(self, font_size):
        pct = int((font_size / 11.0) * 100)
        self.zoom_label.setText(f" {pct}% ")

    def _load_file(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    content = f.read()
                self.editor.setPlainText(content)
                self.editor.document().setModified(False)
                self.is_modified = False
                self._update_title()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to open file: {str(e)}")

    def _on_text_changed(self):
        mod = self.editor.document().isModified()
        if mod != self.is_modified:
            self.is_modified = mod
            self._update_title()

    def _update_title(self):
        title = f"{os.path.basename(self.filepath)}"
        if self.is_modified:
            title += " *"
        title += " - eSim Text Editor"
        self.setWindowTitle(title)

    def save_file(self):
        try:
            with open(self.filepath, 'w') as f:
                f.write(self.editor.toPlainText())
            self.is_modified = False
            self._update_title()
            self.statusBar().showMessage(f"Saved to {self.filepath}", 3000)
            return True
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save file: {str(e)}")
            return False

    def save_file_as(self):
        new_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save As", os.path.dirname(self.filepath), "All Files (*);;SPICE Files (*.cir *.net *.sub)"
        )
        if new_path:
            self.filepath = os.path.abspath(new_path)
            if self.save_file():
                self.statusBar().showMessage(f"Saved to {self.filepath}", 3000)
                
    def open_find(self):
        # Slide the find bar down (160ms OutCubic) instead of snapping in.
        already = self.find_widget.isVisible()
        self.find_widget.show()
        if not already:
            self.find_widget.setMaximumHeight(0)
            anim = QtCore.QPropertyAnimation(self.find_widget, b"maximumHeight", self)
            anim.setDuration(160)
            anim.setStartValue(0)
            anim.setEndValue(52)
            anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
            anim.finished.connect(lambda: self.find_widget.setMaximumHeight(16777215))
            self._find_anim = anim
            anim.start()
        self.find_input.setFocus()
        self.find_input.selectAll()

    def _setup_find_toolbar(self):
        self.find_widget = QtWidgets.QWidget()
        find_layout = QtWidgets.QHBoxLayout(self.find_widget)
        find_layout.setContentsMargins(6, 5, 6, 5)
        find_layout.setSpacing(6)
        
        self.find_input = QtWidgets.QLineEdit()
        self.find_input.setPlaceholderText("Find...")
        self.find_input.setClearButtonEnabled(True)
        self.find_input.returnPressed.connect(self.find_next)
        
        self.replace_input = QtWidgets.QLineEdit()
        self.replace_input.setPlaceholderText("Replace with...")
        self.replace_input.returnPressed.connect(self.replace_text)
        
        self.btn_find_next = QtWidgets.QPushButton("Find Next")
        self.btn_find_next.setProperty("cssClass", "secondary")
        self.btn_find_next.clicked.connect(self.find_next)
        
        self.btn_replace = QtWidgets.QPushButton("Replace")
        self.btn_replace.setProperty("cssClass", "secondary")
        self.btn_replace.clicked.connect(self.replace_text)
        
        self.btn_replace_all = QtWidgets.QPushButton("Replace All")
        self.btn_replace_all.setProperty("cssClass", "secondary")
        self.btn_replace_all.clicked.connect(self.replace_all)
        
        self.btn_close_find = QtWidgets.QPushButton("X")
        self.btn_close_find.setFixedWidth(30)
        self.btn_close_find.clicked.connect(self.find_widget.hide)
        self.btn_close_find.setProperty("cssClass", "icon")
        
        self.find_widget.setObjectName("editorFindBar")
        
        find_layout.addWidget(self.find_input)
        find_layout.addWidget(self.replace_input)
        find_layout.addWidget(self.btn_find_next)
        find_layout.addWidget(self.btn_replace)
        find_layout.addWidget(self.btn_replace_all)
        find_layout.addWidget(self.btn_close_find)
        
        self.find_widget.hide()
        
    def find_next(self):
        text = self.find_input.text()
        if not text: return
        if not self.editor.find(text):
            # Wrap around
            cursor = self.editor.textCursor()
            cursor.movePosition(QtGui.QTextCursor.MoveOperation.Start)
            self.editor.setTextCursor(cursor)
            if not self.editor.find(text):
                QtWidgets.QMessageBox.information(self, "Find", f'Cannot find "{text}"')
                
    def replace_text(self):
        text_to_find = self.find_input.text()
        replacement = self.replace_input.text()
        if not text_to_find: return
        
        cursor = self.editor.textCursor()
        if cursor.hasSelection() and cursor.selectedText().lower() == text_to_find.lower():
            cursor.insertText(replacement)
            self.find_next()
        else:
            self.find_next()

    def replace_all(self):
        text = self.find_input.text()
        if not text: return
        
        rep_text = self.replace_input.text()
        count = 0
        
        cursor = self.editor.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.Start)
        self.editor.setTextCursor(cursor)
        
        while self.editor.find(text):
            self.editor.textCursor().insertText(rep_text)
            count += 1
            
        QtWidgets.QMessageBox.information(self, "Replace All", f"Replaced {count} occurrences.")

    def closeEvent(self, event):
        if self.is_modified:
            reply = QtWidgets.QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Do you want to save before closing?",
                QtWidgets.QMessageBox.StandardButton.Save | QtWidgets.QMessageBox.StandardButton.Discard | QtWidgets.QMessageBox.StandardButton.Cancel,
                QtWidgets.QMessageBox.StandardButton.Save
            )
            
            if reply == QtWidgets.QMessageBox.StandardButton.Save:
                if not self.save_file():
                    event.ignore()
                    return
            elif reply == QtWidgets.QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
        event.accept()

    def update_theme_styles(self):
        prefs = Appconfig().load_preferences()
        theme_mode = prefs.get("theme_mode", "System")
        internal_bg_color = prefs.get("internal_bg_color", "system")
        editor_font_family = prefs.get("editor_font_family", "Consolas")
        editor_font_size = self.editor._font_size
        editor_tab_width = int(prefs.get("editor_tab_width", 4))

        if theme_mode == "Dark":
            is_dark = True
        elif theme_mode == "Light":
            is_dark = False
        else:
            is_dark = QtGui.QGuiApplication.styleHints().colorScheme() == QtCore.Qt.ColorScheme.Dark

        secondary_color = prefs.get("secondary_accent_color", "system")

        if internal_bg_color != "system":
            editor_bg = internal_bg_color
        else:
            editor_bg = "#0B1220" if is_dark else "#FBFCFE"

        editor_color = "#F8FAFC" if is_dark else "#162033"
        border_color = "#31415F" if is_dark else "#D8E1EC"
        caret_color = "#53D7FF" if is_dark else "#0077A8"

        if secondary_color != "system":
            window_bg = secondary_color
        else:
            window_bg = "#111827" if is_dark else "#f4f5f7"

        # Apply robust stylesheet to the editor and to the toolbar so the
        # toolbar buttons stay visible against the editor chrome in both
        # themes. Avoid QPushButton inline colors — they only fight the
        # active QSS.
        # Update SVGs based on theme
        icon_color = "dark" if is_dark else "light"
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.action_save.setIcon(QtGui.QIcon(os.path.join(base_dir, f"images/text_save_{icon_color}.svg")))
        self.action_save_as.setIcon(QtGui.QIcon(os.path.join(base_dir, f"images/text_save_as_{icon_color}.svg")))
        self.action_find.setIcon(QtGui.QIcon(os.path.join(base_dir, f"images/text_find_{icon_color}.svg")))
        self.action_wrap.setIcon(QtGui.QIcon(os.path.join(base_dir, f"images/text_wrap_{icon_color}.svg")))

        self.editor.setStyleSheet(f"""
            QPlainTextEdit {{
                background-color: {editor_bg};
                color: {editor_color};
                border: 1px solid {border_color};
                border-radius: 15px;
                padding: 12px 14px;
                font-family: "{editor_font_family}";
                font-size: {editor_font_size}pt;
                selection-background-color: {"#0E7490" if is_dark else "#0077A8"};
                selection-color: #FFFFFF;
            }}
            QPlainTextEdit:focus {{
                border: 1px solid {caret_color};
            }}
            QPlainTextEdit QScrollBar:vertical {{
                background: transparent;
                width: 12px;
                margin: 4px;
            }}
            QPlainTextEdit QScrollBar::handle:vertical {{
                background: {"#30415F" if is_dark else "#AFC0D3"};
                border-radius: 6px;
                min-height: 30px;
            }}
            QPlainTextEdit QScrollBar::handle:vertical:hover {{
                background: {"#53D7FF" if is_dark else "#0077A8"};
            }}
        """)
        # Font family + tab width settings on the QPlainTextEdit itself.
        f = QtGui.QFont(editor_font_family)
        f.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        f.setPointSize(editor_font_size)
        self.editor.setFont(f)
        self.editor._font_size = editor_font_size
        # Tab width is implemented via the font's explicit advance width.
        fm = QtGui.QFontMetrics(f)
        space_w = fm.horizontalAdvance(" ")
        if space_w > 0:
            self.editor.setTabStopDistance(space_w * editor_tab_width)

        # Toolbar / status bar styling inherits from the active QSS theme,
        # but in case the user has a fully custom theme we still make sure
        # the toolbar background matches the window to avoid stark contrast.
        # Re-trigger syntax highlighter colors
        self.editor.highlighter.update_colors()
        self.editor.viewport().update()
        self._refresh_cursor_label()

    def _refresh_cursor_label(self):
        if not hasattr(self, "_cursor_label"):
            return
        cursor = self.editor.textCursor()
        self._cursor_label.setText(f"Ln {cursor.blockNumber() + 1}, Col {cursor.columnNumber() + 1}")

    def _toggle_wrap(self, on: bool):
        mode = (
            QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth
            if on
            else QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap
        )
        self.editor.setLineWrapMode(mode)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.Type.PaletteChange or event.type() == QtCore.QEvent.Type.StyleChange:
            if getattr(self, '_updating_theme', False):
                return
            self._updating_theme = True
            try:
                self.update_theme_styles()
            finally:
                self._updating_theme = False
