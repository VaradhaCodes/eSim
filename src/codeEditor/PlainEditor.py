"""Fallback editor used when QScintilla is not installed.

Mirrors the slice of the CodeEditor API the host window relies on so
eSim degrades gracefully instead of crashing on a missing optional
dependency.  Pure PyQt6 -- no Qsci import.
"""

import os
import tempfile

from PyQt6 import QtCore, QtWidgets

from codeEditor import lexers, theme


class PlainEditor(QtWidgets.QPlainTextEdit):
    """Minimal monospace editor with the CodeEditor public surface."""

    fileChangedOnDisk = QtCore.pyqtSignal()
    includeRequested = QtCore.pyqtSignal(str)

    def __init__(self, file_path, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.encoding = "utf-8"
        # Same resolver as the QScintilla editor this stands in for, so the
        # fallback is a different WIDGET, not a different typeface. "Monospace"
        # was an X11 alias that resolved per-platform (UI_AUDIT §2.9).
        #
        # The sheet is not belt-and-braces: QScintilla paints its own text and
        # so honours setFont, but this is a real QWidget, and eSim's app-wide
        # sheet styles `QWidget { font-family: ... }` — which beats setFont.
        # Without the local sheet the fallback editor rendered code in the UI's
        # proportional font, columns and all.
        font = theme.editor_font(11)
        self.setFont(font)
        self.setStyleSheet(theme.mono_font_css(font))
        with open(file_path, "r", encoding="utf-8",
                  errors="replace") as handle:
            self.setPlainText(handle.read())
        if lexers.is_generated(file_path):
            self.setReadOnly(True)
        self.document().setModified(False)

    def language(self):
        return lexers.language_for(self.file_path)

    def is_generated(self):
        return lexers.is_generated(self.file_path)

    def eol_label(self):
        return "LF"

    def isModified(self):
        return self.document().isModified()

    def setModified(self, value):
        self.document().setModified(value)

    def save(self):
        # Atomic write (temp + fsync + replace) so a crash mid-save
        # can't truncate the user's netlist.
        data = self.toPlainText()
        directory = os.path.dirname(self.file_path) or "."
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding=self.encoding) as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.file_path)
        except OSError:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        self.setModified(False)
        return True

    def reload(self):
        with open(self.file_path, "r", encoding="utf-8",
                  errors="replace") as handle:
            self.setPlainText(handle.read())
        self.setModified(False)

    def unlock(self):
        self.setReadOnly(False)

    def toggle_comment(self):
        pass

    def open_include_under_cursor(self):
        return False
