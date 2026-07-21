"""Batch 8 verification harness (thread-safe reporting + dialog reentrancy).

Proves, offscreen on this machine, the fixes for:
  M9   Appconfig.print_* marshal onto the GUI thread: a worker-thread print
       never touches the QTextEdit / QStatusBar from the worker; the slot runs
       on the GUI thread (queued), while a same-thread print stays direct and
       ordered. Pre-GUI (no reporter) still writes the plain-list sink, and a
       torn-down reporter falls back instead of raising.
  M12  the excepthook DEFERS its modal dialog via QTimer.singleShot(0, app, ...)
       instead of exec()ing inline -- so a raise from a paint/close/teardown
       handler (GUI thread) can't re-enter the event loop from inside that
       handler, and a raise from a worker thread is marshalled to the GUI
       thread. Proven by: after sys.excepthook returns the dialog has NOT run
       yet (deferred); it runs only on the next event-loop turn, on the GUI
       thread.

Each test is independent; a failing one can't sink the rest. Exit code is the
number of failed tests (0 = all pass).
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SRC = os.path.join(REPO, "src")

ISO = tempfile.mkdtemp(prefix="esim_b8_home_")
os.environ["HOME"] = ISO
os.environ["USERPROFILE"] = ISO
os.environ["APPDATA"] = os.path.join(ISO, "AppData", "Roaming")
os.environ["LOCALAPPDATA"] = os.path.join(ISO, "AppData", "Local")
os.makedirs(os.environ["APPDATA"], exist_ok=True)
os.makedirs(os.environ["LOCALAPPDATA"], exist_ok=True)
os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, SRC)
# frontEnd/Application.py does a bare `import pathmagic` (src/frontEnd/), so its
# package dir must be importable for the M12 excepthook tests.
sys.path.insert(0, os.path.join(SRC, "frontEnd"))

from PyQt6 import QtCore, QtWidgets  # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
_MAIN = QtCore.QThread.currentThread()

_results = []


def check(name, ok, detail=""):
    _results.append(ok)
    tag = "PASS" if ok else "FAIL"
    line = "[%s] %s" % (tag, name)
    if detail:
        line += "  --  " + detail
    print(line)


def _flush():
    """Drain the queued meta-calls + DeferredDelete a real loop would run."""
    _app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    _app.processEvents()
    _app.processEvents()


# --------------------------------------------------------------------------- #
class _EmitThread(QtCore.QThread):
    """Runs a callable off the GUI thread and records the thread it ran on."""

    def __init__(self, fn):
        super().__init__()
        self._fn = fn
        self.ran_on = None

    def run(self):
        self.ran_on = QtCore.QThread.currentThread()
        self._fn()


class _Probe(QtCore.QObject):
    """GUI-thread QObject whose slot records the delivery thread."""

    def __init__(self):
        super().__init__()
        self.seen_thread = None
        self.seen_line = None

    @QtCore.pyqtSlot(str)
    def on_note(self, line):
        self.seen_thread = QtCore.QThread.currentThread()
        self.seen_line = line


def _fresh_gui_sinks():
    """Install real QTextEdit + QStatusBar sinks and a fresh reporter."""
    from configuration.Appconfig import Appconfig
    Appconfig._reporter = None
    edit = QtWidgets.QTextEdit()
    edit.setReadOnly(True)
    Appconfig.noteArea['Note'] = edit
    Appconfig.statusbar = QtWidgets.QStatusBar()
    Appconfig.attach_gui_reporter()
    return Appconfig, edit


# --------------------------------------------------------------------------- #
def test_m9_worker_print_marshalled_to_gui_thread():
    """A print_* from a worker thread is QUEUED onto the GUI thread: the widget
    write happens on the GUI thread, and not before the event loop turns."""
    Appconfig, edit = _fresh_gui_sinks()
    probe = _Probe()                       # lives on the GUI thread
    Appconfig._reporter.note.connect(probe.on_note)

    marker = "WORKER-THREAD-LINE-42"
    th = _EmitThread(lambda: Appconfig().print_error(marker))
    th.start()
    th.wait()                              # worker done emitting; loop idle

    # Cross-thread emit must be QUEUED -> nothing delivered while the GUI thread
    # sat in wait() rather than in its event loop.
    before = marker in edit.toPlainText()
    check("M9 worker emit is deferred (not delivered inline on worker)",
          not before,
          "text present before processEvents" if before else "queued as expected")

    _flush()                               # now the GUI event loop runs the slot

    after = marker in edit.toPlainText()
    on_gui = probe.seen_thread is _MAIN
    off_worker = probe.seen_thread is not th
    check("M9 worker line reaches the QTextEdit after the loop turns", after)
    check("M9 slot ran on the GUI thread, not the worker thread",
          on_gui and off_worker,
          "delivered on %s (worker was %s)" % (
              "GUI" if on_gui else "?", "excluded" if off_worker else th))


def test_m9_same_thread_print_is_direct_and_ordered():
    """A print_* on the GUI thread delivers directly (AutoConnection), so the
    line is present WITHOUT spinning the event loop -- old inline behaviour."""
    Appconfig, edit = _fresh_gui_sinks()
    Appconfig().print_info("first")
    Appconfig().print_warning("second")
    txt = edit.toPlainText()
    direct = "first" in txt and "second" in txt      # no _flush() called
    ordered = txt.find("first") < txt.find("second")
    check("M9 same-thread print delivers directly (no loop needed)", direct)
    check("M9 same-thread print preserves order", ordered)
    bar = Appconfig.statusbar.currentMessage()
    check("M9 same-thread print mirrors to the status bar",
          "second" in bar, "status=%r" % bar)


def test_m9_pre_gui_falls_back_to_list_sink():
    """No reporter (headless / pre-GUI): print_* writes the plain-list sink and
    never needs Qt widgets."""
    from configuration.Appconfig import Appconfig
    Appconfig._reporter = None
    Appconfig.noteArea['Note'] = []
    Appconfig.statusbar = None
    Appconfig().print_error("headless-line")
    notes = Appconfig.noteArea['Note']
    ok = isinstance(notes, list) and any("headless-line" in n for n in notes)
    check("M9 pre-GUI print falls back to the plain-list sink", ok)


def test_m9_stale_reporter_falls_back_without_raising():
    """A reporter whose C++ object was destroyed (stale between QApplications)
    must not crash print_*: _dispatch catches RuntimeError and uses the direct
    sink, clearing the dead reporter."""
    from configuration.Appconfig import Appconfig
    Appconfig, edit = _fresh_gui_sinks()
    dead = Appconfig._reporter
    from PyQt6 import sip
    sip.delete(dead)                       # force RuntimeError on next emit
    Appconfig.noteArea['Note'] = []        # observe the fallback sink
    raised = False
    try:
        Appconfig().print_warning("after-teardown")
    except RuntimeError:
        raised = True
    notes = Appconfig.noteArea['Note']
    landed = isinstance(notes, list) and any("after-teardown" in n for n in notes)
    cleared = Appconfig._reporter is None
    check("M9 stale reporter does not raise into the caller", not raised)
    check("M9 stale reporter falls back to the direct sink", landed and cleared)


# --------------------------------------------------------------------------- #
_DIALOG_CALLS = []


def _install_hook_with_recording_dialog(with_reporter=True):
    """Install the real excepthook but swap Dialogs.critical for a recorder.
    Fresh install each call -> fresh seen_sites, so the per-site dedupe never
    suppresses a later test."""
    from frontEnd import Application
    from configuration import Dialogs
    from configuration.Appconfig import Appconfig

    def _record(parent, title, text, *a, **k):
        _DIALOG_CALLS.append(QtCore.QThread.currentThread())

    Dialogs.critical = _record             # hook resolves Dialogs.critical late
    Appconfig._reporter = None
    if with_reporter:
        Appconfig.attach_gui_reporter()    # GUI-thread reporter for post_to_gui
    Application._install_excepthook()
    return Application


def _raise_via_hook_on_gui():              # distinct raise site (dedupe by site)
    try:
        raise ValueError("gui-thread boom")
    except ValueError:
        sys.excepthook(*sys.exc_info())


def _raise_via_hook_on_worker():           # distinct raise site
    try:
        raise KeyError("worker-thread boom")
    except KeyError:
        sys.excepthook(*sys.exc_info())


def test_m12_gui_thread_raise_is_deferred_not_inline():
    """Even on the GUI thread the excepthook must POST the dialog, not exec() it
    inline -- otherwise a raise inside paint/close re-enters the event loop."""
    _install_hook_with_recording_dialog()
    _DIALOG_CALLS.clear()
    _raise_via_hook_on_gui()
    inline = len(_DIALOG_CALLS)            # must be 0: deferred
    check("M12 GUI-thread raise does not open the dialog inline (no reentrancy)",
          inline == 0, "dialog ran inline" if inline else "deferred")
    _flush()
    ran = len(_DIALOG_CALLS) == 1 and _DIALOG_CALLS[0] is _MAIN
    check("M12 deferred dialog then runs once, on the GUI thread", ran,
          "calls=%d" % len(_DIALOG_CALLS))


def test_m12_worker_thread_raise_marshalled_to_gui():
    """A raise on a worker thread (B1's trigger) never builds the dialog on the
    worker; it is marshalled to the GUI thread and still deferred. This is the
    path PyQt6's singleShot(0, app, show) could never actually deliver."""
    _install_hook_with_recording_dialog()
    _DIALOG_CALLS.clear()
    th = _EmitThread(_raise_via_hook_on_worker)
    th.start()
    th.wait()
    inline = len(_DIALOG_CALLS)            # worker returned; nothing built yet
    check("M12 worker-thread raise builds no dialog on the worker thread",
          inline == 0, "built on worker" if inline else "marshalled")
    _flush()
    ran = len(_DIALOG_CALLS) == 1 and _DIALOG_CALLS[0] is _MAIN
    check("M12 worker raise dialog then runs once, on the GUI thread", ran,
          "calls=%d" % len(_DIALOG_CALLS))


def _raise_via_hook_on_gui_early():        # distinct raise site
    try:
        raise RuntimeError("early-startup boom")
    except RuntimeError:
        sys.excepthook(*sys.exc_info())


def test_m12_no_reporter_gui_fallback_still_defers():
    """Before the reporter exists (early startup) a GUI-thread raise still gets
    a dialog via the singleShot(0, show) fallback -- deferred, not inline."""
    _install_hook_with_recording_dialog(with_reporter=False)
    _DIALOG_CALLS.clear()
    _raise_via_hook_on_gui_early()
    check("M12 no-reporter fallback does not open the dialog inline",
          len(_DIALOG_CALLS) == 0)
    _flush()
    ran = len(_DIALOG_CALLS) == 1 and _DIALOG_CALLS[0] is _MAIN
    check("M12 no-reporter fallback then shows the dialog on the GUI thread",
          ran, "calls=%d" % len(_DIALOG_CALLS))


# --------------------------------------------------------------------------- #
def main():
    print("=== batch 8 verification (M9 thread-safe reporting, M12 reentrancy) ===")
    for t in (
        test_m9_worker_print_marshalled_to_gui_thread,
        test_m9_same_thread_print_is_direct_and_ordered,
        test_m9_pre_gui_falls_back_to_list_sink,
        test_m9_stale_reporter_falls_back_without_raising,
        test_m12_gui_thread_raise_is_deferred_not_inline,
        test_m12_worker_thread_raise_marshalled_to_gui,
    ):
        try:
            t()
        except Exception as e:             # a harness/AssertionError is a fail
            import traceback
            _results.append(False)
            print("[FAIL] %s raised: %s" % (t.__name__, e))
            traceback.print_exc()
    failed = _results.count(False)
    print("=== %d/%d checks passed, %d failed ===" % (
        _results.count(True), len(_results), failed))
    return failed


if __name__ == "__main__":
    sys.exit(main())
