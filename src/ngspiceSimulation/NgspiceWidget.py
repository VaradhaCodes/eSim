import os
import shlex
import codecs
import logging
from typing import List, Optional
from PyQt6 import QtWidgets, QtCore
from configuration import Dialogs
from PyQt6.QtCore import pyqtSignal, pyqtSlot
from configuration.Appconfig import Appconfig
from frontEnd import TerminalUi
from maker import CosimConfig
from maker.CosimLogger import CosimLog
from configparser import ConfigParser

logger = logging.getLogger(__name__)


class NgspiceWidget(QtWidgets.QWidget):
    """Runs an NGSpice simulation and displays output in a terminal widget."""

    ERROR_FAILED_TO_START = 0
    ERROR_CRASHED = 1
    ERROR_TIMED_OUT = 2

    SUCCESS_FORMAT = ('<span style="color:#00ff00; font-size:26px;">'
                     '{}'
                     '</span>')
    FAILURE_FORMAT = ('<span style="color:#ff3333; font-size:26px;">'
                     '{}'
                     '</span>')

    def __init__(self, netlist: str, sim_end_signal: pyqtSignal, plotFlag: Optional[bool] = None) -> None:
        super().__init__()

        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                          QtWidgets.QSizePolicy.Policy.Expanding)
        self.setMinimumSize(300, 200)

        self.obj_appconfig = Appconfig()
        self.project_dir = self.obj_appconfig.current_project["ProjectName"]
        self.netlist_path = netlist
        self.sim_end_signal = sim_end_signal
        self.plotFlag = plotFlag
        self.command = netlist
        self.raw_file = self._raw_path(netlist)
        logger.info(f"Value of plotFlag: {self.plotFlag}")

        # Incremental UTF-8 decoders so a multibyte character split across two
        # readyRead chunks is reassembled instead of being mangled by a
        # per-chunk decode.
        self._stdout_decoder = codecs.getincrementaldecoder('utf-8')('replace')
        self._stderr_decoder = codecs.getincrementaldecoder('utf-8')('replace')
        self._main_pid: Optional[int] = None

        self.ngspice_args = self._prepare_ngspice_arguments(netlist)
        logger.info(f"NGSpice arguments: {self.ngspice_args}")

        # Use eSim's ngspice (the nghdl-simulator build that carries d_cosim +
        # ivlng) for ALL simulations; resolved without hardcoded paths.
        self.ngspice_bin = CosimConfig.ngspice_binary()
        self.uses_dcosim = self._netlist_uses_dcosim(netlist)

        self.process = QtCore.QProcess(self)
        self.terminal_ui = TerminalUi.TerminalUi(
            self.process, self.ngspice_args, self.ngspice_bin)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.terminal_ui)

        self._configure_process()
        self._start_process()

    @staticmethod
    def _raw_path(netlist: str) -> str:
        """Derive the rawfile path from the netlist path without clobbering it.

        eSim netlists are '<name>.cir.out' and the rawfile is '<name>.raw'. A
        bare str.replace('.cir.out', '.raw') is a no-op when the suffix is
        absent, which would make the rawfile path equal the netlist path and
        let ngspice overwrite the netlist via '-r'. Strip the known suffix when
        present, otherwise swap the final extension.
        """
        if netlist.endswith(".cir.out"):
            return netlist[:-len(".cir.out")] + ".raw"
        return os.path.splitext(netlist)[0] + ".raw"

    def _prepare_ngspice_arguments(self, netlist: str) -> List[str]:
        return ['-b', '-r', self.raw_file, netlist]

    def _configure_process(self) -> None:
        self.process.setWorkingDirectory(self.project_dir)
        self.process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.SeparateChannels)
        self.process.started.connect(self._on_process_started)
        self.process.readyReadStandardOutput.connect(self._handle_stdout)
        self.process.readyReadStandardError.connect(self._handle_stderr)
        self.process.finished.connect(
            lambda exit_code, exit_status: self.finish_simulation(
                exit_code, exit_status, self.sim_end_signal, False
            )
        )
        self.process.errorOccurred.connect(
            lambda: self.finish_simulation(None, None, self.sim_end_signal, True)
        )

    def _register_process(self, process: QtCore.QProcess) -> None:
        self.obj_appconfig.process_obj.append(process)
        current_project_name = self.obj_appconfig.current_project['ProjectName']
        if current_project_name in self.obj_appconfig.proc_dict:
            self.obj_appconfig.proc_dict[current_project_name].append(process.processId())

    def _unregister_process(self) -> None:
        """Drop the finished main-simulation process from the shared registries.

        Without this, every simulation leaves its QProcess in
        ``process_obj`` and its PID in ``proc_dict`` forever, so a long eSim
        session accumulates dead references. Only the main ngspice process is
        pruned here; the xterm/gaw helpers manage their own lifetimes.
        """
        try:
            if self.process in self.obj_appconfig.process_obj:
                self.obj_appconfig.process_obj.remove(self.process)
        except (ValueError, AttributeError):
            pass
        proj = self.obj_appconfig.current_project.get('ProjectName')
        pids = self.obj_appconfig.proc_dict.get(proj) if proj else None
        if pids and self._main_pid is not None:
            try:
                pids.remove(self._main_pid)
            except ValueError:
                pass

    @pyqtSlot()
    def _on_process_started(self) -> None:
        """Reset per-run decoder state at the start of every run, including redo.

        Redo reuses the same QProcess (via TerminalUi.redoSimulation), so the
        previous run's decoder state must be reset here. A redo also runs after
        the first run deregistered the process, so re-register it for cleanup.
        """
        self._stdout_decoder.reset()
        self._stderr_decoder.reset()
        self._main_pid = self.process.processId()
        if self.process not in self.obj_appconfig.process_obj:
            self._register_process(self.process)

    def _start_process(self) -> None:
        if self.uses_dcosim:
            if not CosimConfig.has_dcosim():
                Dialogs.warning(
                    self, "d_cosim unavailable",
                    CosimConfig.missing_reason() or
                    "This ngspice build cannot run d_cosim co-simulation.")
            # ngspice's ivlng adapter dlopens libvvp at runtime, so the iverilog
            # lib dir must be on the dynamic-loader search path.
            self._add_iverilog_libpath()
            # Tell both audiences this is a co-sim run and which libvvp ngspice
            # will dlopen -- ivlng load failures otherwise look like generic
            # ngspice noise. GUI sim console is plain-text, so banner it there;
            # full detail goes to the terminal + ~/.esim/dcosim.log.
            clog = CosimLog()
            clog.phase("d_cosim co-simulation run")
            clog.info("ngspice: " + str(self.ngspice_bin))
            clog.info("ivlng libvvp dir: " +
                      (CosimConfig.iverilog_libdir() or "<none>"))
            self.terminal_ui.simulationConsole.insertPlainText(
                "\n[eSim] d_cosim co-simulation: loading Verilog model(s) "
                "via ngspice ivlng/libvvp ...\n")
        logger.info(f"Launching ngspice -> {self.ngspice_bin}")
        self.process.start(self.ngspice_bin, self.ngspice_args)
        logger.debug(f"Process dictionary: {self.obj_appconfig.proc_dict}")
        self._register_process(self.process)
        # Capture the PID now (valid while running); processId() returns 0 once
        # the process has finished, so _unregister_process needs the cached value.
        self._main_pid = self.process.processId()

    def _add_iverilog_libpath(self) -> None:
        """Prepend the iverilog lib dir (libvvp) to the dynamic-loader search
        path so ngspice's ivlng adapter can load it. Cross-platform: PATH on
        Windows, LD_LIBRARY_PATH elsewhere."""
        lib = CosimConfig.iverilog_libdir()
        if not lib:
            return
        var = CosimConfig.loader_path_var()
        env = QtCore.QProcessEnvironment.systemEnvironment()
        existing = env.value(var, "")
        env.insert(var, lib + (os.pathsep + existing if existing else ""))
        self.process.setProcessEnvironment(env)

    @staticmethod
    def _netlist_uses_dcosim(netlist: str) -> bool:
        """True if the netlist instantiates a d_cosim code-model block."""
        try:
            with open(netlist, "r") as handle:
                return "d_cosim" in handle.read().lower()
        except OSError:
            return False

    def _is_linux(self) -> bool:
        return os.name != "nt"

    def _start_gaw_process(self) -> None:
        """Launch the GTK analog wave viewer on the rawfile.

        Only called after the batch run has finished and actually written the
        rawfile — starting it in __init__ (as before) opened gaw on a file that
        ngspice had not produced yet, flashing an empty viewer on every run.
        """
        try:
            self.gaw_process = QtCore.QProcess(self)
            self.gaw_command = f"gaw {shlex.quote(self.raw_file)}"
            self.gaw_process.start('sh', ['-c', self.gaw_command])
            logger.info(f"Started GAW with command: {self.gaw_command}")
        except Exception as e:
            logger.error(f"Failed to start GAW process: {e}")

    @pyqtSlot()
    def _handle_stdout(self) -> None:
        try:
            data = self.process.readAllStandardOutput().data()
            if data:
                text = self._stdout_decoder.decode(data)
                if text:
                    self.terminal_ui.simulationConsole.insertPlainText(text)
        except Exception as e:
            logger.error(f"Error reading stdout: {e}")

    @pyqtSlot()
    def _handle_stderr(self) -> None:
        """Read stderr, filter batch-mode noise, display the rest."""
        try:
            data = self.process.readAllStandardError().data()
            if not data:
                return
            text = self._stderr_decoder.decode(data)
            if not text:
                return
            filtered = '\n'.join(
                line for line in text.split('\n')
                if 'PrinterOnly' not in line and 'viewport for graphics' not in line
            )
            if filtered.strip():
                self.terminal_ui.simulationConsole.insertPlainText(filtered)
        except Exception as e:
            logger.error(f"Error reading stderr: {e}")

    def finish_simulation(self, exit_code: Optional[int],
                         exit_status: Optional[QtCore.QProcess.ExitStatus],
                         sim_end_signal: pyqtSignal,
                         has_error_occurred: bool) -> None:
        # Skip finished signal if cancellation triggered both finished and error signals
        if not has_error_occurred and self.terminal_ui.simulationCancelled:
            return

        # Resolve exit code and status before any UI work so finally block has them
        if exit_code is None:
            exit_code = self.process.exitCode()
        error_type = self.process.error()
        if error_type in (QtCore.QProcess.ProcessError.FailedToStart,
                          QtCore.QProcess.ProcessError.Crashed,
                          QtCore.QProcess.ProcessError.Timedout):
            exit_status = QtCore.QProcess.ExitStatus.CrashExit
        elif exit_status is None:
            exit_status = self.process.exitStatus()

        try:
            self._update_ui_after_simulation()

            if self.uses_dcosim:
                # Record the co-sim verdict where developers can grep it.
                dlog = CosimLog()
                if self._is_simulation_successful(
                        exit_status, exit_code, error_type):
                    dlog.ok("d_cosim run finished OK (ngspice rc=%s)."
                            % str(exit_code))
                else:
                    dlog.error("d_cosim run FAILED (ngspice rc=%s)."
                               % str(exit_code))
                    dlog.fix("Check the ngspice console above. If it mentions "
                             "ivlng/libvvp, the model failed to load -- rebuild "
                             "it in the NgVeri tab and confirm libvvp is found.")

            if self.terminal_ui.simulationCancelled:
                self._show_cancellation_message()
            elif self._is_simulation_successful(exit_status, exit_code, error_type):
                self._show_success_message()

                # On redo-simulation, TerminalUi sets "redoPlotFlag" on the process
                # to pass the user's plot choice back here
                redo_flag = self.process.property("redoPlotFlag")
                if redo_flag is not None:
                    self.plotFlag = redo_flag

                if self.plotFlag:
                    self.open_ngspice_plots()
            else:
                self._show_failure_message(error_type)

            self._scroll_terminal_to_bottom()
        except Exception as e:
            logger.error(f"finish_simulation UI error: {e}", exc_info=True)
        finally:
            # Drop the finished process from the shared registries so they don't
            # accumulate dead references across a long session.
            self._unregister_process()
            # Emit completion signal — must always run so plot window can open
            sim_end_signal.emit(exit_status, exit_code)

    def open_ngspice_plots(self) -> None:
        logger.info("Opening NGSpice native plots")

        if os.name == 'nt':
            try:
                parser_nghdl = ConfigParser()
                config_path = os.path.join('library', 'config', '.nghdl', 'config.ini')
                parser_nghdl.read(config_path)
                msys_home = parser_nghdl.get('COMPILER', 'MSYS_HOME')
                mintty_exe = os.path.join(msys_home, 'usr', 'bin', 'mintty.exe')

                self.mintty_process = QtCore.QProcess(self)
                self.mintty_process.setWorkingDirectory(self.project_dir)
                # Pass program + args directly — Qt handles quoting internally
                self.mintty_process.start(
                    mintty_exe, [self.ngspice_bin, '-p', self.command])
                logger.info(
                    f"Started mintty: {mintty_exe} "
                    f"{self.ngspice_bin} -p {self.command}")

            except Exception as e:
                logger.error(f"Failed to start Windows NGSpice plots: {e}")

        else:  # Linux/Unix
            try:
                # Re-run the netlist in an interactive ngspice so the user gets
                # a plot prompt with the vectors loaded. NOTE: passing the
                # rawfile as a positional arg does NOT work -- ngspice parses it
                # as a netlist and aborts ("UTF-8 syntax error ... fatal error").
                # Loading a raw needs the interactive `load` command, which the
                # batch '-r' output path doesn't give us, so we re-source the
                # netlist (wasteful but correct).
                # d_cosim runs need libvvp on the loader path for ngspice's
                # ivlng adapter; prepend it for this xterm only.
                ld_prefix = ""
                if self.uses_dcosim:
                    iv_lib = CosimConfig.iverilog_libdir()
                    if iv_lib:
                        var = CosimConfig.loader_path_var()
                        ld_prefix = (f"{var}={shlex.quote(iv_lib)}:"
                                     f"${{{var}}} ")
                # Quote all paths so spaces in project names don't break the shell command
                xterm_command = (
                    f"cd {shlex.quote(self.project_dir)} && "
                    f"{ld_prefix}{shlex.quote(self.ngspice_bin)} "
                    f"-r {shlex.quote(self.raw_file)} {shlex.quote(self.command)}"
                )
                self.xterm_process = QtCore.QProcess(self)
                self.xterm_process.start('xterm', ['-hold', '-e', 'sh', '-c', xterm_command])
                self._register_process(self.xterm_process)
                logger.info(f"Started xterm: {xterm_command}")

                # Launch the GTK analog wave viewer now that the rawfile exists.
                self._start_gaw_process()

            except Exception as e:
                logger.error(f"Failed to start Linux NGSpice plots: {e}")

    def _update_ui_after_simulation(self) -> None:
        self.terminal_ui.progressBar.setMaximum(100)
        self.terminal_ui.progressBar.setProperty("value", 100)
        self.terminal_ui.cancelSimulationButton.setEnabled(False)
        self.terminal_ui.redoSimulationButton.setEnabled(True)

    def _is_simulation_successful(self, exit_status: QtCore.QProcess.ExitStatus,
                                exit_code: int,
                                error_type: QtCore.QProcess.ProcessError) -> bool:
        return (exit_status == QtCore.QProcess.ExitStatus.NormalExit
                and exit_code == 0)

    def _show_cancellation_message(self) -> None:
        message_dialog = Dialogs.make_message_box(self)
        message_dialog.setModal(True)
        message_dialog.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        message_dialog.setWindowTitle("Warning Message")
        message_dialog.setText("Simulation was cancelled.")
        message_dialog.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        message_dialog.exec()

    def _show_success_message(self) -> None:
        success_message = self.SUCCESS_FORMAT.format("Simulation Completed Successfully!")
        self.terminal_ui.simulationConsole.append(success_message)

    def _show_failure_message(self, error_type: QtCore.QProcess.ProcessError) -> None:
        failure_message = self.FAILURE_FORMAT.format("Simulation Failed!")
        self.terminal_ui.simulationConsole.append(failure_message)

        error_message = self._get_error_message(error_type)
        error_dialog = Dialogs.make_error_message(self)
        error_dialog.setModal(True)
        error_dialog.setWindowTitle("Error Message")
        error_dialog.showMessage(error_message)
        error_dialog.exec()

    def _get_error_message(self, error_type: QtCore.QProcess.ProcessError) -> str:
        error_messages = {
            QtCore.QProcess.ProcessError.FailedToStart: (
                'Simulation failed to start. '
                'Ensure that eSim is installed correctly.'
            ),
            QtCore.QProcess.ProcessError.Crashed: (
                'Simulation crashed. Try again later.'
            ),
            QtCore.QProcess.ProcessError.Timedout: (
                'Simulation has timed out. Try to reduce the '
                'simulation time or the simulation step interval.'
            )
        }
        
        return error_messages.get(
            error_type, 
            'Simulation could not complete. Try again later.'
        )

    def _scroll_terminal_to_bottom(self) -> None:
        scrollbar = self.terminal_ui.simulationConsole.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(800, 600)
