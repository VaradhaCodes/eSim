"""A single-file code editor buffer built on QsciScintilla.

The widget owns its whole file lifecycle: load, atomic save, dirty
tracking, encoding/EOL preservation, read-only-for-generated-files and
on-disk change detection.  The hosting window only drives it through
the public methods and signals below.
"""

import os
import tempfile

from PyQt6 import QtCore, QtGui
from PyQt6.Qsci import QsciScintilla

import re

from codeEditor import lexers, theme


#: Files larger than this skip syntax highlighting + folding and open
#: read-only, so a huge ``.cir.out``/``.log`` never freezes the UI.
MAX_SYNTAX_BYTES = 2_000_000


class CodeEditor(QsciScintilla):
    """Scintilla editor bound to one file on disk."""

    #: emitted when the file changes on disk while open
    fileChangedOnDisk = QtCore.pyqtSignal()
    #: emitted with a resolved path when the user asks to open an
    #: ``.include`` / ``.lib`` target under the cursor
    includeRequested = QtCore.pyqtSignal(str)

    _INCLUDE_RE = QtCore.QRegularExpression(
        r'^\s*\.(?:include|lib|inc)\s+"?([^"\s]+)"?',
        QtCore.QRegularExpression.PatternOption.CaseInsensitiveOption,
    )

    def __init__(self, file_path, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.encoding = "utf-8"
        self._eol = QsciScintilla.EolMode.EolUnix
        self._lexer = None
        self._reloading = False
        self._matches = []
        self._current_range = None        # last "current match" indicator
        self._margin_digits = 0
        self.oversize = False

        self._build_font()
        self._configure()
        self._load()
        self._apply_lexer()
        self._start_watch()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def language(self):
        return lexers.language_for(self.file_path)

    def is_generated(self):
        return lexers.is_generated(self.file_path)

    def eol_label(self):
        return {
            QsciScintilla.EolMode.EolWindows: "CRLF",
            QsciScintilla.EolMode.EolMac: "CR",
        }.get(self._eol, "LF")

    def save(self):
        """Atomically write the buffer back to disk.

        Returns True on success; raises OSError on write failure so the
        caller can surface it.
        """
        data = self.text()
        directory = os.path.dirname(self.file_path) or "."
        fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            # newline="" keeps the EOL already baked into the buffer.
            with os.fdopen(fd, "w", encoding=self.encoding,
                           newline="") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            self._reloading = True            # ignore our own change
            os.replace(tmp, self.file_path)
        except OSError:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        finally:
            self._refresh_watch()
        self.setModified(False)
        return True

    def reload(self):
        """Re-read the file from disk, discarding buffer edits.

        Cursor position and scroll are preserved across the reload so an
        external regeneration doesn't jump the user back to line 1.
        """
        line, col = self.getCursorPosition()
        top = self.firstVisibleLine()
        self._load()
        self.setModified(False)
        line = min(line, max(0, self.lines() - 1))
        self.setCursorPosition(line, col)
        self.setFirstVisibleLine(min(top, max(0, self.lines() - 1)))

    def unlock(self):
        """Drop the read-only guard on a generated file."""
        self.setReadOnly(False)

    def toggle_comment(self):
        """Comment / uncomment the selected lines (language aware)."""
        token = lexers.comment_token(self.file_path)
        if not token or self.isReadOnly():
            return
        start_line, _, end_line, end_col = self.getSelection()
        if start_line < 0:
            start_line = end_line = self.getCursorPosition()[0]
        elif end_col == 0 and end_line > start_line:
            end_line -= 1

        lines = [self.text(n) for n in range(start_line, end_line + 1)]
        commented = all(
            ln.lstrip().startswith(token)
            for ln in lines if ln.strip()
        )
        self.beginUndoAction()
        for line_no in range(start_line, end_line + 1):
            text = self.text(line_no)
            if not text.strip():
                continue
            if commented:
                self._uncomment_line(line_no, text, token)
            else:
                self.insertAt(token + " ", line_no,
                              len(text) - len(text.lstrip()))
        self.endUndoAction()

    def open_include_under_cursor(self):
        """If the current line is an .include/.lib, emit its path."""
        line_no = self.getCursorPosition()[0]
        match = self._INCLUDE_RE.match(self.text(line_no))
        if not match.hasMatch():
            return False
        target = match.captured(1)
        if not os.path.isabs(target):
            base = os.path.dirname(self.file_path)
            target = os.path.normpath(os.path.join(base, target))
        self.includeRequested.emit(target)
        return True

    # ------------------------------------------------------------------
    # setup helpers
    # ------------------------------------------------------------------
    def _build_font(self):
        self._mono = theme.editor_font()
        self.setFont(self._mono)

    def _configure(self):
        self.setUtf8(True)
        self.setMarginType(0, QsciScintilla.MarginType.NumberMargin)
        self.setMarginLineNumbers(0, True)
        self._update_margin_width()
        self.setCaretLineVisible(True)
        self.setBraceMatching(
            QsciScintilla.BraceMatch.SloppyBraceMatch)
        self.setAutoIndent(True)
        self.setIndentationsUseTabs(False)
        self.setTabWidth(4)
        self.setIndentationGuides(True)
        self.setFolding(QsciScintilla.FoldStyle.BoxedTreeFoldStyle)
        # On-demand completion only (Ctrl+Space), never auto-pops.
        self.setAutoCompletionSource(
            QsciScintilla.AutoCompletionSource.AcsDocument)
        self.setAutoCompletionThreshold(-1)
        self.setAutoCompletionCaseSensitivity(False)

        QtGui.QShortcut(
            QtGui.QKeySequence("Ctrl+Space"), self,
            activated=self.autoCompleteFromAll)

        self.textChanged.connect(self._update_margin_width)

    def _apply_lexer(self):
        if self.oversize:
            # No lexer + no folding for very large files; theme the
            # chrome only and lock editing.
            self.setLexer(None)
            self.setFolding(QsciScintilla.FoldStyle.NoFoldStyle)
            theme.apply(self, None, self._mono)
            self.setReadOnly(True)
            return
        self._lexer = lexers.make_lexer(self.file_path, self._mono, self)
        self.setLexer(self._lexer)
        theme.apply(self, self._lexer, self._mono)
        if self.is_generated():
            self.setReadOnly(True)

    # ------------------------------------------------------------------
    # find highlighting (driven by the FindBar)
    # ------------------------------------------------------------------
    def clear_search(self):
        """Remove all find-highlight indicators and stored matches."""
        last = max(0, self.lines() - 1)
        end = self.lineLength(last)
        self.clearIndicatorRange(0, 0, last, end, theme.SEARCH_INDICATOR)
        self.clearIndicatorRange(0, 0, last, end, theme.CURRENT_INDICATOR)
        self._matches = []
        self._current_range = None

    def search_all(self, query, regex=False, case=False, word=False):
        """Highlight every match of *query*; return the match count."""
        self.clear_search()
        if not query:
            return 0
        pattern = query if regex else re.escape(query)
        if word:
            pattern = r"\b(?:%s)\b" % pattern
        try:
            rx = re.compile(pattern, 0 if case else re.IGNORECASE)
        except re.error:
            return 0
        for match in rx.finditer(self.text()):
            if match.start() == match.end():
                continue
            self._matches.append((match.start(), match.end()))
            line1, col1 = self.lineIndexFromPosition(match.start())
            line2, col2 = self.lineIndexFromPosition(match.end())
            self.fillIndicatorRange(
                line1, col1, line2, col2, theme.SEARCH_INDICATOR)
        return len(self._matches)

    def match_count(self):
        return len(self._matches)

    def nearest_match_index(self):
        """Index of the first match at/after the caret (else 0)."""
        caret = self.SendScintilla(QsciScintilla.SCI_GETCURRENTPOS)
        for i, (start, _end) in enumerate(self._matches):
            if start >= caret:
                return i
        return 0

    def select_match(self, index):
        """Select match *index* and flag it as the current match."""
        if not self._matches:
            return
        index %= len(self._matches)
        # Clear only the previously highlighted current match, not the
        # whole document (matters on big files / rapid F3).
        if self._current_range is not None:
            pl1, pc1, pl2, pc2 = self._current_range
            self.clearIndicatorRange(
                pl1, pc1, pl2, pc2, theme.CURRENT_INDICATOR)
        start, end = self._matches[index]
        line1, col1 = self.lineIndexFromPosition(start)
        line2, col2 = self.lineIndexFromPosition(end)
        self.fillIndicatorRange(
            line1, col1, line2, col2, theme.CURRENT_INDICATOR)
        self._current_range = (line1, col1, line2, col2)
        self.setSelection(line1, col1, line2, col2)
        self.ensureLineVisible(line1)

    def replace_all_matches(self, replacement):
        """Replace every highlighted match with *replacement* (literal).

        Uses the same Python-regex matches as :meth:`search_all`, so the
        replace count matches the displayed find count exactly.  Returns
        the number replaced.
        """
        if not self._matches or self.isReadOnly():
            return 0
        count = len(self._matches)
        self.beginUndoAction()
        for start, end in reversed(self._matches):   # back-to-front
            line1, col1 = self.lineIndexFromPosition(start)
            line2, col2 = self.lineIndexFromPosition(end)
            self.setSelection(line1, col1, line2, col2)
            self.replaceSelectedText(replacement)
        self.endUndoAction()
        return count

    def _update_margin_width(self):
        digits = max(2, len(str(max(1, self.lines()))))
        if digits == self._margin_digits:
            return
        self._margin_digits = digits
        self.setMarginWidth(0, "0" * (digits + 1))

    # ------------------------------------------------------------------
    # disk I/O
    # ------------------------------------------------------------------
    def _load(self):
        try:
            self.oversize = os.path.getsize(self.file_path) > MAX_SYNTAX_BYTES
        except OSError:
            self.oversize = False
        raw = self._read_bytes()
        text, self.encoding = self._decode(raw)
        self._eol = self._detect_eol(text)
        self.setEolMode(self._eol)
        self.setText(text)
        self.convertEols(self._eol)

    def _read_bytes(self):
        with open(self.file_path, "rb") as handle:
            return handle.read()

    @staticmethod
    def _decode(raw):
        for enc in ("utf-8", "latin-1"):
            try:
                return raw.decode(enc), enc
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", "replace"), "utf-8"

    @staticmethod
    def _detect_eol(text):
        if "\r\n" in text:
            return QsciScintilla.EolMode.EolWindows
        if "\r" in text:
            return QsciScintilla.EolMode.EolMac
        return QsciScintilla.EolMode.EolUnix

    # ------------------------------------------------------------------
    # file-system watch (detect external regeneration)
    # ------------------------------------------------------------------
    def _start_watch(self):
        self._watcher = QtCore.QFileSystemWatcher(self)
        self._refresh_watch()
        self._watcher.fileChanged.connect(self._on_disk_change)

    def _refresh_watch(self):
        paths = self._watcher.files()
        if paths:
            self._watcher.removePaths(paths)
        if os.path.exists(self.file_path):
            self._watcher.addPath(self.file_path)

    def _on_disk_change(self, _path):
        # Editors that re-create the file (write+rename) drop the watch;
        # re-arm it and tell the host so it can prompt for a reload.
        QtCore.QTimer.singleShot(150, self._refresh_watch)
        if self._reloading:
            self._reloading = False
            return
        self.fileChangedOnDisk.emit()

    # ------------------------------------------------------------------
    # comment helper / context menu
    # ------------------------------------------------------------------
    def _uncomment_line(self, line_no, text, token):
        indent = len(text) - len(text.lstrip())
        rest = text[indent:]
        if rest.startswith(token + " "):
            cut = len(token) + 1
        else:
            cut = len(token)
        self.setSelection(line_no, indent, line_no, indent + cut)
        self.replaceSelectedText("")

    def contextMenuEvent(self, event):
        menu = self.createStandardContextMenu()
        line_no = self.getCursorPosition()[0]
        if self._INCLUDE_RE.match(self.text(line_no)).hasMatch():
            menu.addSeparator()
            action = menu.addAction("Open included file")
            action.triggered.connect(self.open_include_under_cursor)
        if self.isReadOnly() and self.is_generated():
            menu.addSeparator()
            unlock = menu.addAction("Edit anyway (unlock)")
            unlock.triggered.connect(self.unlock)
        menu.exec(event.globalPos())
