"""Floating multi-tab editor window for eSim project text files.

One window hosts many files as tabs, sitting beside the schematic /
simulation rather than stacking behind them.  Carries the menu bar,
inline find bar and status bar; the per-file editor widgets stay small.

Falls back to a plain QPlainTextEdit-based editor when QScintilla is
not installed, so a missing optional dependency never bricks the
project explorer.
"""

import os

from PyQt6 import QtCore, QtGui, QtWidgets
from configuration import Dialogs

from codeEditor.FindBar import FindBar
from codeEditor.InfoBar import InfoBar
from codeEditor.PlainEditor import PlainEditor

try:
    from codeEditor.CodeEditor import CodeEditor
    HAS_QSCI = True
except ImportError:                       # QScintilla not installed
    CodeEditor = None
    HAS_QSCI = False


STYLE = """
QMainWindow, #editorCentral { background: #FFFFFF; }
QMenuBar { background: #F6F8FA; border-bottom: 1px solid #E1E4E8;
           padding: 2px 4px; }
QMenuBar::item { padding: 4px 10px; background: transparent;
                 border-radius: 6px; }
QMenuBar::item:selected { background: #E7ECF1; }
QMenu { background: #FFFFFF; border: 1px solid #D0D7DE;
        border-radius: 8px; padding: 4px; }
QMenu::item { padding: 5px 22px; border-radius: 5px; }
QMenu::item:selected { background: #E7ECF1; }
QTabWidget::pane { border: 0; border-top: 1px solid #E1E4E8; }
QTabBar { qproperty-drawBase: 0; }
QTabBar::tab { background: #EEF1F4; color: #41484F;
               padding: 6px 10px 6px 12px; margin-right: 2px;
               border-top-left-radius: 8px;
               border-top-right-radius: 8px; }
QTabBar::tab:selected { background: #FFFFFF; color: #1F2328;
                        border: 1px solid #E1E4E8; border-bottom: 0; }
QTabBar::tab:hover { background: #F6F8FA; }
QStatusBar { background: #F6F8FA; border-top: 1px solid #E1E4E8;
             color: #57606A; }
QStatusBar QLabel { color: #57606A; padding: 0 8px; }
#findBar { background: #F6F8FA; border: 1px solid #C7CDD4;
           border-radius: 8px; }
#findBar QLineEdit { border: 1px solid #D0D7DE; border-radius: 6px;
                     padding: 4px 8px; background: #FFFFFF;
                     selection-background-color: #CFE3FB; }
#findBar QLineEdit:focus { border: 1px solid #0366D6; }
#findBar QLineEdit[noMatch="true"] { border: 1px solid #E1604D;
                     background: #FDF1F0; }
#findCount { color: #57606A; }
QToolButton#findToggle, QToolButton#findTool {
    border: 1px solid transparent; border-radius: 6px;
    padding: 3px 7px; color: #41484F; font-weight: 600; }
QToolButton#findToggle:hover, QToolButton#findTool:hover {
    background: #E7ECF1; }
QToolButton#findToggle:checked {
    background: #DDEBFB; border: 1px solid #9CC4F0; color: #0366D6; }
QToolButton#findClose:hover { background: #FBD2D0; color: #B71C1C; }
QToolButton#findExpand { border: 1px solid #D0D7DE; border-radius: 6px;
    background: #FFFFFF; color: #57606A; font-size: 15px;
    font-weight: 700; padding: 0 6px; }
QToolButton#findExpand:hover { background: #E7ECF1; }
QToolButton#findExpand:checked { color: #0366D6; background: #DDEBFB;
    border: 1px solid #9CC4F0; }
#findBar QPushButton { border: 1px solid #D0D7DE; border-radius: 6px;
                       padding: 4px 12px; background: #FFFFFF; }
#findBar QPushButton:hover { background: #EEF1F4; }
QToolButton#tabClose { border: none; border-radius: 9px;
                       color: #8A9199; font-size: 14px;
                       padding: 0; }
QToolButton#tabClose:hover { background: #E1604D; color: #FFFFFF; }
#infoBar { background: #FCE5C0; border-bottom: 1px solid #E5C97A; }
QLabel#infoTitle { color: #5C4405; font-weight: 700; background: transparent; }
QLabel#infoMessage { color: #6B5410; background: transparent; }
QPushButton#infoAction { border: 1px solid #C9A227; border-radius: 6px;
    padding: 5px 14px; background: #F6D88A; color: #4D3A05;
    font-weight: 600; }
QPushButton#infoAction:hover { background: #F0CD6E; }
QToolButton#infoClose { border: none; border-radius: 6px; padding: 2px 6px;
    color: #6B5410; font-size: 14px; }
QToolButton#infoClose:hover { background: #E7C766; color: #4D3A05; }
"""


#: Every live EditorWindow, so eSim can flush unsaved buffers (e.g.
#: right before a simulation run reads the netlist off disk).
_OPEN_WINDOWS = set()


def flush_all_dirty():
    """Save every modified buffer in every open editor window.

    Returns the list of file names written.  Call this before any code
    that reads a project file off disk (simulation, conversion) so it
    never sees a stale on-disk copy while the editor holds newer edits.
    """
    saved = []
    for window in list(_OPEN_WINDOWS):
        saved.extend(window.flush_dirty())
    return saved


def create_editor(file_path, parent=None):
    """Return the best available editor widget for *file_path*."""
    if HAS_QSCI:
        return CodeEditor(file_path, parent)
    return PlainEditor(file_path, parent)


def _is_scintilla(editor):
    return hasattr(editor, "setCursorPosition")


def _cursor_line_col(editor):
    if _is_scintilla(editor):
        return editor.getCursorPosition()
    cursor = editor.textCursor()
    return cursor.blockNumber(), cursor.columnNumber()


class EditorWindow(QtWidgets.QMainWindow):
    """A standalone tabbed editor; reuse one instance per project."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("eSim Editor")
        self.resize(960, 680)
        self.setStyleSheet(STYLE)
        self._open_tabs = {}              # normalised path -> editor

        self._settings = QtCore.QSettings("eSim", "CodeEditor")
        self._zoom_level = int(self._settings.value("zoom", 0))
        self._wrap = self._settings.value("wrap", False, type=bool)
        self._ws = self._settings.value("whitespace", False, type=bool)
        _OPEN_WINDOWS.add(self)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setMovable(True)
        self.tabs.setDocumentMode(True)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        tab_bar = self.tabs.tabBar()
        tab_bar.setExpanding(False)
        tab_bar.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        tab_bar.customContextMenuRequested.connect(self._tab_menu)
        tab_bar.installEventFilter(self)      # middle-click to close

        central = QtWidgets.QWidget()
        central.setObjectName("editorCentral")
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.tabs, 1)
        self.setCentralWidget(central)

        # Find/replace is a floating overlay pinned to the editor's
        # top-right (VS Code style), not a row that steals layout space.
        self.find_bar = FindBar(central, host=self)
        shadow = QtWidgets.QGraphicsDropShadowEffect(self.find_bar)
        shadow.setBlurRadius(18)
        shadow.setOffset(0, 3)
        shadow.setColor(QtGui.QColor(0, 0, 0, 70))
        self.find_bar.setGraphicsEffect(shadow)

        self._build_menu()
        self._build_status_bar()

        geometry = self._settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        self._wrap_action.setChecked(self._wrap)
        self._ws_action.setChecked(self._ws)

    # ------------------------------------------------------------------
    # public API used by ProjectExplorer
    # ------------------------------------------------------------------
    def open(self, file_path):
        """Open *file_path* in a tab (focusing it if already open)."""
        # Re-arm the flush registry: a window hidden after its last tab
        # closed was discarded on closeEvent, but ProjectExplorer reuses
        # the same instance, so it must rejoin or its dirty buffers stop
        # being flushed before a simulation run.
        _OPEN_WINDOWS.add(self)
        key = os.path.normcase(os.path.abspath(file_path))
        if key in self._open_tabs:
            existing = self._open_tabs[key]
            # The file may have been regenerated on disk (e.g. the converter
            # rebuilds <proj>.cir) while this tab sat open. The watcher can
            # miss write+rename saves, so reload here unless the user has
            # unsaved edits of their own.
            if not existing.isModified():
                try:
                    existing.reload()
                except OSError:
                    pass
            self.tabs.setCurrentWidget(existing)
            self._raise()
            return

        try:
            editor = create_editor(file_path, self)
        except OSError as err:
            self.statusBar().showMessage("Could not open: %s" % err, 4000)
            return

        editor.modificationChanged.connect(
            lambda _m, e=editor: self._update_tab_title(e))
        editor.includeRequested.connect(self.open)
        editor.fileChangedOnDisk.connect(
            lambda e=editor: self._on_disk_change(e))
        if _is_scintilla(editor):
            editor.cursorPositionChanged.connect(
                lambda *_: self._update_status())
        else:
            editor.cursorPositionChanged.connect(self._update_status)

        index = self.tabs.addTab(editor, os.path.basename(file_path))
        self.tabs.setTabToolTip(index, file_path)
        self._add_close_button(index, editor)
        self._open_tabs[key] = editor
        self._apply_view_settings(editor)
        self.tabs.setCurrentWidget(editor)
        self._raise()

    def _apply_view_settings(self, editor):
        """Push persisted zoom / wrap / whitespace onto *editor*."""
        if not _is_scintilla(editor):
            return
        from PyQt6.Qsci import QsciScintilla
        editor.zoomTo(self._zoom_level)
        editor.setWrapMode(
            QsciScintilla.WrapMode.WrapWord if self._wrap
            else QsciScintilla.WrapMode.WrapNone)
        editor.setWhitespaceVisibility(
            QsciScintilla.WhitespaceVisibility.WsVisible if self._ws
            else QsciScintilla.WhitespaceVisibility.WsInvisible)

    # ------------------------------------------------------------------
    # menu
    # ------------------------------------------------------------------
    def _build_menu(self):
        bar = self.menuBar()

        file_menu = bar.addMenu("&File")
        self._action(file_menu, "&Save", "Ctrl+S", self._save)
        self._action(file_menu, "Save &All", "Ctrl+Shift+S",
                     self._save_all)
        file_menu.addSeparator()
        self._action(file_menu, "&Revert to Saved", None, self._revert)
        file_menu.addSeparator()
        self._action(file_menu, "&Close Tab", "Ctrl+W",
                     self._close_current)

        edit_menu = bar.addMenu("&Edit")
        self._action(edit_menu, "&Undo", "Ctrl+Z", self._edit("undo"))
        self._action(edit_menu, "&Redo", "Ctrl+Y", self._edit("redo"))
        edit_menu.addSeparator()
        self._action(edit_menu, "Cu&t", "Ctrl+X", self._edit("cut"))
        self._action(edit_menu, "&Copy", "Ctrl+C", self._edit("copy"))
        self._action(edit_menu, "&Paste", "Ctrl+V", self._edit("paste"))
        self._action(edit_menu, "Select &All", "Ctrl+A",
                     self._line_cmd("SCI_SELECTALL"))
        edit_menu.addSeparator()
        self._action(edit_menu, "Move Line &Up", "Alt+Up",
                     self._line_cmd("SCI_MOVESELECTEDLINESUP"))
        self._action(edit_menu, "Move Line Dow&n", "Alt+Down",
                     self._line_cmd("SCI_MOVESELECTEDLINESDOWN"))
        self._action(edit_menu, "De&lete Line", "Ctrl+Shift+K",
                     self._line_cmd("SCI_LINEDELETE"))
        edit_menu.addSeparator()
        self._action(edit_menu, "Toggle Co&mment", "Ctrl+/",
                     self._toggle_comment)

        search_menu = bar.addMenu("&Search")
        self._action(search_menu, "&Find", "Ctrl+F",
                     self.find_bar.open_find)
        self._action(search_menu, "&Replace", "Ctrl+H",
                     self.find_bar.open_replace)
        search_menu.addSeparator()
        self._action(search_menu, "Find &Next", "F3",
                     self.find_bar.find_next)
        self._action(search_menu, "Find &Previous", "Shift+F3",
                     self.find_bar.find_prev)
        self._action(search_menu, "&Go to Line", "Ctrl+G",
                     self._goto_dialog)

        view_menu = bar.addMenu("&View")
        self._action(view_menu, "Zoom &In", "Ctrl+=",
                     lambda: self._zoom(1))
        self._action(view_menu, "Zoom &Out", "Ctrl+-",
                     lambda: self._zoom(-1))
        self._action(view_menu, "&Reset Zoom", "Ctrl+0",
                     lambda: self._zoom(0))
        view_menu.addSeparator()
        self._wrap_action = self._action(
            view_menu, "&Word Wrap", "Alt+Z", self._toggle_wrap,
            checkable=True)
        self._ws_action = self._action(
            view_menu, "Show White&space", None, self._toggle_ws,
            checkable=True)

    def _action(self, menu, text, shortcut, slot, checkable=False):
        action = QtGui.QAction(text, self)
        if shortcut:
            action.setShortcut(QtGui.QKeySequence(shortcut))
        action.setCheckable(checkable)
        action.triggered.connect(slot)
        menu.addAction(action)
        return action

    def _edit(self, name):
        def run():
            editor = self._current()
            if editor is not None:
                getattr(editor, name)()
        return run

    def _line_cmd(self, const_name):
        def run():
            editor = self._current()
            if editor is None or not _is_scintilla(editor):
                return
            from PyQt6.Qsci import QsciScintilla
            editor.SendScintilla(getattr(QsciScintilla, const_name))
        return run

    # ------------------------------------------------------------------
    # status bar
    # ------------------------------------------------------------------
    def _build_status_bar(self):
        self._pos_label = QtWidgets.QLabel()
        self._lang_label = QtWidgets.QLabel()
        self._enc_label = QtWidgets.QLabel()
        self._eol_label = QtWidgets.QLabel()
        self._ro_label = QtWidgets.QLabel()
        for widget in (self._ro_label, self._enc_label, self._eol_label,
                       self._lang_label, self._pos_label):
            self.statusBar().addPermanentWidget(widget)

    def _update_status(self):
        editor = self._current()
        labels = (self._pos_label, self._lang_label, self._enc_label,
                  self._eol_label, self._ro_label)
        if editor is None:
            for lbl in labels:
                lbl.clear()
            return
        line, col = _cursor_line_col(editor)
        self._pos_label.setText("Ln %d, Col %d" % (line + 1, col + 1))
        self._lang_label.setText(editor.language())
        self._enc_label.setText(getattr(editor, "encoding", "utf-8"))
        self._eol_label.setText(editor.eol_label())
        self._ro_label.setText(
            "Read-only" if editor.isReadOnly() else "")

    # ------------------------------------------------------------------
    # commands on the active editor
    # ------------------------------------------------------------------
    def _current(self):
        return self.tabs.currentWidget()

    def _save(self):
        editor = self._current()
        if editor is not None:
            self._save_editor(editor)

    def _save_all(self):
        for index in range(self.tabs.count()):
            self._save_editor(self.tabs.widget(index))

    def flush_dirty(self):
        """Save every modified tab; return the saved file names."""
        saved = []
        for editor in self._all_editors():
            if editor.isModified() and self._save_editor(editor):
                saved.append(os.path.basename(editor.file_path))
        return saved

    def _save_editor(self, editor):
        if not editor.isModified():
            return True
        try:
            editor.save()
        except OSError as err:
            Dialogs.critical(
                self, "Save failed",
                "Could not save %s:\n%s"
                % (os.path.basename(editor.file_path), err))
            return False
        self.statusBar().showMessage(
            "Saved %s" % os.path.basename(editor.file_path), 2000)
        return True

    def _goto_dialog(self):
        editor = self._current()
        if editor is None:
            return
        total = (editor.lines() if _is_scintilla(editor)
                 else editor.document().blockCount())
        line, ok = QtWidgets.QInputDialog.getInt(
            self, "Go to Line", "Line (1-%d):" % total, 1, 1, total)
        if not ok:
            return
        if _is_scintilla(editor):
            editor.setCursorPosition(line - 1, 0)
            editor.ensureLineVisible(line - 1)
        else:
            block = editor.document().findBlockByLineNumber(line - 1)
            cursor = editor.textCursor()
            cursor.setPosition(block.position())
            editor.setTextCursor(cursor)
        editor.setFocus()

    def _toggle_comment(self):
        editor = self._current()
        if editor is not None:
            editor.toggle_comment()

    def _revert(self):
        """Discard buffer edits and re-read the active file from disk."""
        editor = self._current()
        if editor is None or not editor.isModified():
            return
        name = os.path.basename(editor.file_path)
        reply = Dialogs.question(
            self, "Revert", "Discard unsaved changes to %s?" % name,
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No)
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            editor.reload()
            self.statusBar().showMessage("Reverted %s" % name, 2000)

    def _zoom(self, delta):
        # Zoom is a window-wide preference: apply to every tab and
        # remember it across sessions.
        self._zoom_level = 0 if delta == 0 else self._zoom_level + delta
        for editor in self._all_editors():
            if _is_scintilla(editor):
                editor.zoomTo(self._zoom_level)
        self._settings.setValue("zoom", self._zoom_level)

    def _toggle_wrap(self, checked):
        from PyQt6.Qsci import QsciScintilla
        self._wrap = checked
        mode = (QsciScintilla.WrapMode.WrapWord if checked
                else QsciScintilla.WrapMode.WrapNone)
        for editor in self._all_editors():
            if _is_scintilla(editor):
                editor.setWrapMode(mode)
        self._settings.setValue("wrap", checked)

    def _toggle_ws(self, checked):
        from PyQt6.Qsci import QsciScintilla
        self._ws = checked
        vis = (QsciScintilla.WhitespaceVisibility.WsVisible if checked
               else QsciScintilla.WhitespaceVisibility.WsInvisible)
        for editor in self._all_editors():
            if _is_scintilla(editor):
                editor.setWhitespaceVisibility(vis)
        self._settings.setValue("whitespace", checked)

    # ------------------------------------------------------------------
    # tab / lifecycle handling
    # ------------------------------------------------------------------
    def _add_close_button(self, index, editor):
        btn = QtWidgets.QToolButton()
        btn.setObjectName("tabClose")
        btn.setText("✕")
        btn.setFixedSize(18, 18)
        btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        btn.setToolTip("Close (Ctrl+W)")
        btn.clicked.connect(lambda _c=False, e=editor: self._close_editor(e))
        self.tabs.tabBar().setTabButton(
            index, QtWidgets.QTabBar.ButtonPosition.RightSide, btn)

    def _on_tab_changed(self, _index):
        self._update_status()
        self._update_window_title()
        self.find_bar.set_editor(self._current())

    def _update_tab_title(self, editor):
        index = self.tabs.indexOf(editor)
        if index < 0:
            return
        name = os.path.basename(editor.file_path)
        if editor.isModified():
            name = "● " + name
        self.tabs.setTabText(index, name)
        if editor is self._current():
            self._update_window_title()

    def _update_window_title(self):
        editor = self._current()
        if editor is None:
            self.setWindowTitle("eSim Editor")
            return
        mark = "● " if editor.isModified() else ""
        self.setWindowTitle(
            "%s%s — eSim Editor"
            % (mark, os.path.basename(editor.file_path)))

    def eventFilter(self, obj, event):
        if (obj is self.tabs.tabBar()
                and event.type() == QtCore.QEvent.Type.MouseButtonRelease
                and event.button() == QtCore.Qt.MouseButton.MiddleButton):
            index = self.tabs.tabBar().tabAt(event.position().toPoint())
            if index >= 0:
                self._close_editor(self.tabs.widget(index))
                return True
        return super().eventFilter(obj, event)

    def _tab_menu(self, point):
        index = self.tabs.tabBar().tabAt(point)
        if index < 0:
            return
        menu = QtWidgets.QMenu(self)
        menu.addAction(
            "Close", lambda: self._close_editor(self.tabs.widget(index)))
        menu.addAction("Close Others", lambda: self._close_others(index))
        menu.addAction("Close All", self._close_all)
        menu.exec(self.tabs.tabBar().mapToGlobal(point))

    def _close_others(self, keep_index):
        keep = self.tabs.widget(keep_index)
        for editor in self._all_editors():
            if editor is not keep:
                self._close_editor(editor)

    def _close_all(self):
        for editor in self._all_editors():
            self._close_editor(editor)

    def _all_editors(self):
        return [self.tabs.widget(i) for i in range(self.tabs.count())]

    def _on_disk_change(self, editor):
        name = os.path.basename(editor.file_path)
        # No local edits at risk: silently pick up the new on-disk copy
        # (e.g. a fresh KiCad-to-Ngspice convert) -- nothing to confront
        # the user with.
        if not editor.isModified():
            editor.reload()
            self.statusBar().showMessage("Reloaded %s" % name, 2000)
            return
        # Unsaved edits would be lost on reload: show a GNOME-style inline
        # bar in the tab instead of a focus-stealing modal, so the user
        # can keep editing and decide when (or whether) to reload.
        InfoBar(
            editor,
            "File Has Changed on Disk",
            "The file has been changed by another program.",
            "Discard Changes and Reload",
            editor.reload)

    def _close_current(self):
        editor = self._current()
        if editor is not None:
            self._close_editor(editor)

    def _close_editor(self, editor):
        if not self._confirm_discard(editor):
            return
        index = self.tabs.indexOf(editor)
        key = os.path.normcase(os.path.abspath(editor.file_path))
        self._open_tabs.pop(key, None)
        if index >= 0:
            self.tabs.removeTab(index)
        editor.deleteLater()
        # An editor with no tabs is just empty chrome; close (hide) it.
        # ProjectExplorer caches the instance, so opening a file later
        # re-shows it via open()/_raise().
        if self.tabs.count() == 0:
            self.close()

    def _confirm_discard(self, editor):
        if not editor.isModified():
            return True
        name = os.path.basename(editor.file_path)
        reply = Dialogs.question(
            self, "Unsaved changes", "Save changes to %s?" % name,
            QtWidgets.QMessageBox.StandardButton.Save
            | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel)
        if reply == QtWidgets.QMessageBox.StandardButton.Cancel:
            return False
        if reply == QtWidgets.QMessageBox.StandardButton.Save:
            return self._save_editor(editor)
        return True

    def _position_find_bar(self):
        """Pin the find overlay to the editor's top-right corner."""
        # Bound late: FindBar can call this from its own constructor,
        # before ``self.find_bar`` is assigned.
        bar = getattr(self, "find_bar", None)
        if bar is None or not bar.isVisible():
            return
        bar.adjustSize()
        parent = bar.parentWidget()
        margin = 14
        x = parent.width() - bar.width() - margin
        y = self.tabs.tabBar().height() + 6
        bar.move(max(margin, x), y)
        bar.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_find_bar()

    def closeEvent(self, event):
        for index in range(self.tabs.count()):
            if not self._confirm_discard(self.tabs.widget(index)):
                event.ignore()
                return
        self._settings.setValue("geometry", self.saveGeometry())
        _OPEN_WINDOWS.discard(self)
        event.accept()

    def _raise(self):
        self.show()
        self.raise_()
        self.activateWindow()
