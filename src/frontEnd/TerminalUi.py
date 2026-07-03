from PyQt6 import QtCore, QtGui, QtWidgets, uic
import os


class TerminalUi(QtWidgets.QMainWindow):
    """This is a class that represents the GUI required to provide
    details regarding the ngspice simulation. This GUI consists of
    a progress bar, a console window which displays the log of the
    simulation and button required for re-simulation and cancellation
    of the simulation"""
    def __init__(self, qProcess, args, ngspice_bin='ngspice'):
        """The constructor of the TerminalUi class
        param: qProcess: a PyQt QProcess that runs ngspice
        type: qProcess: :class:`QtCore.QProcess`
        param: args: arguments to be passed on to the ngspice call
        type: args: list
        param: ngspice_bin: ngspice executable to launch on re-run/redo (eSim's
               resolved ngspice; d_cosim netlists need the bundled build)
        type: ngspice_bin: str
        """
        super(TerminalUi, self).__init__()

        # Other variables
        # The console inherits its colours from the global QSS rule
        # `QTextEdit#simulationConsole` (see style_dark.qss / style_light.qss).
        # We deliberately do NOT keep an internal dark/light toggle here — that
        # was a legacy per-widget override that broke the user's selected
        # theme. The lightDarkModeButton now cycles the application's theme via
        # app.apply_theme() so the entire window stays in sync.
        self.qProcess = qProcess
        self.args = args
        self.ngspice_bin = ngspice_bin

        # Load the ui file
        uic.loadUi(os.path.join(os.path.dirname(__file__), "TerminalUi.ui"), self)

        # Define Our Widgets
        self.progressBar = self.findChild(
            QtWidgets.QProgressBar,
            "progressBar"
        )
        self.simulationConsole = self.findChild(
            QtWidgets.QTextEdit,
            "simulationConsole"
        )

        self.lightDarkModeButton = self.findChild(
            QtWidgets.QPushButton,
            "lightDarkModeButton"
        )
        self.cancelSimulationButton = self.findChild(
            QtWidgets.QPushButton,
            "cancelSimulationButton"
        )
        self.cancelSimulationButton.setEnabled(True)

        self.redoSimulationButton = self.findChild(
            QtWidgets.QPushButton,
            "redoSimulationButton"
        )
        self.redoSimulationButton.setEnabled(False)

        # Theme cycle — clicking this button asks the running application to
        # swap palette to the next theme (System → Light → Dark → System).
        # The QSS rule for `QTextEdit#simulationConsole` then re-applies
        # automatically via apply_theme.
        self.lightDarkModeButton.setToolTip(
            "Cycle application theme (Light → Dark → System)"
        )
        self.lightDarkModeButton.setText("◐")
        font = self.lightDarkModeButton.font()
        font.setPointSize(13)
        font.setBold(True)
        self.lightDarkModeButton.setFont(font)
        self.lightDarkModeButton.clicked.connect(self._cycle_theme)
        self.cancelSimulationButton.clicked.connect(self.cancelSimulation)
        self.redoSimulationButton.clicked.connect(self.redoSimulation)

        self.simulationCancelled = False
        self.show()

    @staticmethod
    def _cycle_theme():
        """Cycle the application's theme preference and re-apply."""
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        apply_fn = getattr(app, 'apply_theme', None)
        if apply_fn is None:
            return
        from frontEnd.theme_utils import get_preferences
        if os.name == 'nt':
            user_home = os.path.join('library', 'config')
        else:
            user_home = os.path.expanduser('~')
        prefs = get_preferences(user_home)
        order = ("System", "Light", "Dark")
        try:
            idx = order.index(prefs.get("theme_mode", "System"))
        except ValueError:
            idx = 0
        next_mode = order[(idx + 1) % len(order)]
        try:
            import json
            prefs["theme_mode"] = next_mode
            path = os.path.join(user_home, ".esim", "preferences.json")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as fh:
                json.dump(prefs, fh)
        except Exception:
            pass
        apply_fn()

    def cancelSimulation(self):
        """This function cancels the ongoing ngspice simulation.
        """
        self.cancelSimulationButton.setEnabled(False)
        self.redoSimulationButton.setEnabled(True)

        if (self.qProcess.state() == QtCore.QProcess.ProcessState.NotRunning):
            return

        self.simulationCancelled = True
        self.qProcess.kill()

        # To show progressBar completed
        self.progressBar.setMaximum(100)
        self.progressBar.setProperty("value", 100)

        cancelFormat = '<span style="color:#FF8624; font-size:26px;">{}</span>'
        self.simulationConsole.append(
            cancelFormat.format("Simulation Cancelled!"))
        self.simulationConsole.verticalScrollBar().setValue(
            self.simulationConsole.verticalScrollBar().maximum()
        )

    def redoSimulation(self):
        """This function reruns the ngspice simulation
        """
        self.Flag = False
        self.cancelSimulationButton.setEnabled(True)
        self.redoSimulationButton.setEnabled(False)

        if (self.qProcess.state() != QtCore.QProcess.ProcessState.NotRunning):
            return

        # To make the progressbar running
        self.progressBar.setMaximum(0)
        self.progressBar.setProperty("value", -1)

        self.simulationConsole.setText("")
        self.simulationCancelled = False

        self.Flag = self._resolveNgspicePlotChoice()

        self.qProcess.setProperty("redoPlotFlag", self.Flag)

        self.qProcess.start(self.ngspice_bin, self.args)

    # Persisted across runs in the shared QSettings store. When the user ticks
    # "Remember my choice" the popup is skipped on every later simulation and
    # the saved answer is reused. It can be re-enabled from
    # Preferences ▸ Simulation ("Ask before generating Ngspice plots").
    NGSPICE_REMEMBER_KEY = "ngspicePlots/remember"
    NGSPICE_FLAG_KEY = "ngspicePlots/flag"

    def _resolveNgspicePlotChoice(self):
        """Return whether to also generate ngspice plots for this run.

        If the user previously chose "Remember my choice", reuse the stored
        answer silently. Otherwise show the Yes/No popup (with a remember
        checkbox) and persist the decision when the box is ticked.
        """
        settings = QtCore.QSettings('eSim', 'eSim')
        if settings.value(self.NGSPICE_REMEMBER_KEY, False, type=bool):
            return settings.value(self.NGSPICE_FLAG_KEY, False, type=bool)

        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle("Ngspice Plots")
        msg_box.setText("Do you want Ngspice plots?")
        # Widen the message label so the window title ("Ngspice Plots") is not
        # clipped behind the close/minimise/maximise buttons on tight window
        # managers. QMessageBox ignores setMinimumWidth, so size via the label.
        msg_box.setStyleSheet("QLabel { min-width: 360px; }")

        remember_cb = QtWidgets.QCheckBox(
            "Remember my choice (re-enable in Preferences ▸ Simulation)")
        msg_box.setCheckBox(remember_cb)

        yes_button = msg_box.addButton(
            "Yes", QtWidgets.QMessageBox.ButtonRole.YesRole)
        msg_box.addButton("No", QtWidgets.QMessageBox.ButtonRole.NoRole)

        msg_box.exec()
        flag = msg_box.clickedButton() == yes_button

        if remember_cb.isChecked():
            settings.setValue(self.NGSPICE_REMEMBER_KEY, True)
            settings.setValue(self.NGSPICE_FLAG_KEY, flag)

        return flag

    # Note: the legacy `changeColor()` per-widget dark/light toggle has been
    # removed; use `_cycle_theme()` (the lightDarkModeButton target) instead so
    # the entire application stays in sync with the user's chosen theme.
