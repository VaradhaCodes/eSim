"""Tiny QThread wrapper for running a blocking backend call off the GUI thread.

This is the ONLY Qt-aware module in hdl/. It keeps the long iverilog/vvp work
(hdl.icarus) from freezing the UI: the widget starts a :class:`BackgroundJob`,
gets ``succeeded``/``failed`` on the GUI thread, and can hand the job an
``icarus.CancelToken`` to support a Cancel button.
"""
from PyQt6 import QtCore


class BackgroundJob(QtCore.QThread):
    """Run ``fn(*args, **kwargs)`` on a worker thread.

    Emits ``succeeded(result)`` with the return value, or ``failed(str)`` if the
    callable raised. The result is delivered via a queued signal, so the slot
    runs back on the GUI thread and may safely touch widgets."""

    succeeded = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, fn, *args, parent=None, **kwargs):
        super().__init__(parent)
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            result = self._fn(*self._args, **self._kwargs)
        except Exception as exc:  # surfaced to the UI as a failed signal
            self.failed.emit(str(exc))
            return
        self.succeeded.emit(result)
