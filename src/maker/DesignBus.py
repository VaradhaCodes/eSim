# =========================================================================
#             FILE: DesignBus.py
#
#      DESCRIPTION: Single source of truth for one HDL design in the Flow
#                   Navigator (Author -> Verify -> Convert). The three stages
#                   are *views* on one design, not three editors that sync
#                   through disk. The design lives here, in memory; disk is
#                   only persistence.
#
#                   Why this exists: the stages used to hand the design around
#                   by writing a .v to disk and reading it back, so a watchdog
#                   was needed to notice the change -- and it could not tell our
#                   own write from an outside one, so it nagged on every tab
#                   switch. With one in-memory owner, in-app navigation never
#                   touches disk, and the watch fires only on genuine external
#                   edits (e.g. the file opened in another editor).
#
#                   Disk writes still happen, but only on purpose: an explicit
#                   Save, and a lazy materialize right before Convert (whose
#                   toolchain reads a real file path). They are echo-proof: the
#                   sha256 of the bytes we last wrote is remembered, so the
#                   watch ignores any disk event whose content matches it.
#
#  ORGANIZATION: eSim Team at FOSSEE, IIT Bombay
# =========================================================================
import hashlib
import os

try:
    import watchdog.events
    import watchdog.observers
    _HAS_WATCHDOG = True
except ImportError:
    # watchdog is an OPTIONAL dependency. Without it the whole in-app design
    # flow still works: Author -> Verify -> Convert navigation, explicit Save,
    # and the lazy materialize before Convert never need it -- only the passive
    # external-edit watch does. Importing maker.makerchip (-> DesignBus) must
    # never brick Model Creation just because watchdog is missing or fails to
    # load (M8), mirroring the graceful QScintilla fallback in
    # codeEditor.EditorWindow.
    watchdog = None
    _HAS_WATCHDOG = False
from PyQt6 import QtCore
from PyQt6.QtCore import pyqtSignal


def _hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


if _HAS_WATCHDOG:
    class _DiskWatchHandler(watchdog.events.PatternMatchingEventHandler):
        """Reports modifications of one watched file back to its DesignBus.

        Scoped to the file's *directory* (not the file) because many editors
        save atomically -- write a temp file, then rename it over the target --
        which a file-level watch misses. The pattern keeps only events for our
        file.
        """

        def __init__(self, bus, filepath):
            super().__init__(
                patterns=[filepath],
                ignore_directories=True,
                case_sensitive=True)
            self._bus = bus

        def on_modified(self, event):
            self._bus._on_disk_event()

        def on_created(self, event):
            self._bus._on_disk_event()

        def on_moved(self, event):
            # Atomic-save rename landing on our file.
            self._bus._on_disk_event()
else:
    # No base class to subclass when watchdog is absent; _start_watch bails
    # out before this is ever referenced.
    _DiskWatchHandler = None


class DesignBus(QtCore.QObject):
    """The one design shared by the Author / Verify / Convert stages.

    Hold the design text in ``content`` and let every stage render from / write
    to it. ``path`` is where it persists (empty until first materialize).
    ``_saved_hash`` is the sha256 of the bytes last exchanged with disk and is
    what makes the external-edit watch echo-proof.
    """

    #: content was replaced in memory (load / explicit set).
    contentChanged = pyqtSignal(str)
    #: the file on disk diverged from our content -- a real outside edit.
    externalChange = pyqtSignal(str)

    def __init__(self, filecount, parent=None):
        super().__init__(parent)
        self.filecount = filecount
        self._content = ""
        self._path = ""
        self._saved_hash = ""
        self._observer = None
        self._handler = None
        self._watch_file = None

    # ------------------------------------------------------------------ #
    #  In-memory model (no disk)
    # ------------------------------------------------------------------ #
    def get_content(self):
        return self._content

    def set_content(self, text):
        """Update the in-memory design. Idempotent: setting the same text is a
        no-op, so a view rendering exactly what it received cannot loop."""
        if text == self._content:
            return
        self._content = text
        self.contentChanged.emit(text)

    @property
    def path(self):
        return self._path

    def set_path(self, path):
        """Assign where the design will persist, WITHOUT writing it yet (a Verify
        -authored design gets a home so Convert can later materialize it). The
        bytes are written on the next save / materialize, not here."""
        if path and path != self._path:
            self._path = path
            self._mirror_slot()

    def is_dirty(self):
        if not self._path:
            return bool(self._content)
        return _hash_bytes(self._content.encode("utf-8")) != self._saved_hash

    # ------------------------------------------------------------------ #
    #  Disk I/O  (the ONLY place that reads/writes the design file)
    # ------------------------------------------------------------------ #
    def load_from_disk(self, path):
        """Read an existing .v as THE design (explicit Open / Reload)."""
        if not path or not os.path.isfile(path):
            return ""
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError:
            return ""
        self._content = data.decode("utf-8", errors="replace")
        self._path = path
        self._saved_hash = _hash_bytes(data)
        self._mirror_slot()
        self._start_watch()
        self.contentChanged.emit(self._content)
        return path

    def save_to_disk(self, path=None):
        """Persist the in-memory content. Sole writer of the design file."""
        target = path or self._path
        if not target:
            return ""
        data = self._content.encode("utf-8")
        # Record the hash BEFORE writing so the watch never echoes our own write,
        # even if its event fires before this call returns.
        self._saved_hash = _hash_bytes(data)
        try:
            os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
            with open(target, "wb") as fh:
                fh.write(data)
        except OSError:
            return ""
        self._path = target
        self._mirror_slot()
        self._start_watch()
        return target

    def materialize(self):
        """Ensure disk reflects the in-memory content, then return the path.
        Used right before Convert, which reads a real file. No-op when disk
        already matches what we hold."""
        if not self._path:
            return ""
        fresh = _hash_bytes(self._content.encode("utf-8")) == self._saved_hash
        if fresh and os.path.isfile(self._path):
            return self._path
        return self.save_to_disk()

    # ------------------------------------------------------------------ #
    #  Legacy compat: keep Maker.verilogFile[filecount] == path
    #
    #  NgVeri / ModelGeneration still read that slot as the design path. The
    #  slot is no longer shared *mutable* state edited from everywhere: this bus
    #  is its one writer, so the slot is just a derived mirror of ``path``.
    # ------------------------------------------------------------------ #
    def _mirror_slot(self):
        from . import Maker
        while len(Maker.verilogFile) <= self.filecount:
            Maker.verilogFile.append("")
        Maker.verilogFile[self.filecount] = self._path

    # ------------------------------------------------------------------ #
    #  External-edit watch (echo-proof, one observer per bus)
    # ------------------------------------------------------------------ #
    def _start_watch(self):
        if not _HAS_WATCHDOG:
            return          # watchdog absent: external-edit watch disabled
        if not self._path:
            return
        if self._observer is not None and self._watch_file == self._path:
            return                      # already watching the right file
        self.close()
        watch_dir = os.path.dirname(self._path) or "."
        if not os.path.isdir(watch_dir):
            return
        self._handler = _DiskWatchHandler(self, self._path)
        self._observer = watchdog.observers.Observer()
        self._observer.schedule(self._handler, path=watch_dir, recursive=False)
        self._observer.daemon = True
        self._watch_file = self._path
        self._observer.start()

    def _on_disk_event(self):
        """Runs in the watchdog thread. Compare the file to what WE last wrote:
        equal hash => our own write (echo) -> ignore; different => a real outside
        edit. ``externalChange`` is delivered queued to the GUI thread."""
        try:
            with open(self._path, "rb") as fh:
                data = fh.read()
        except OSError:
            return
        if _hash_bytes(data) == self._saved_hash:
            return
        self.externalChange.emit(self._path)

    def close(self):
        """Stop the watch. Call from the owner's teardown."""
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(timeout=2)
            except Exception:
                pass
        self._observer = None
        self._handler = None
        self._watch_file = None
