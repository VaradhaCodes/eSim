# =========================================================================
#             FILE: NgVeri.py
#
#            USAGE: ---
#
#      DESCRIPTION: This define all components of the NgVeri Tab.
#
#          OPTIONS: ---
#     REQUIREMENTS: ---
#             BUGS: ---
#            NOTES: ---
#           AUTHOR: Sumanto Kar, sumantokar@iitb.ac.in, FOSSEE, IIT Bombay
# ACKNOWLEDGEMENTS: Rahul Paknikar, rahulp@iitb.ac.in, FOSSEE, IIT Bombay
#                Digvijay Singh, digvijay.singh@iitb.ac.in, FOSSEE, IIT Bombay
#                Prof. Maheswari R. and Team, VIT Chennai
#     GUIDED BY: Steve Hoover, Founder Redwood EDA
#                Kunal Ghosh, VLSI System Design Corp.Pvt.Ltd
#                Anagha Ghosh, VLSI System Design Corp.Pvt.Ltd
# OTHER CONTRIBUTERS:
#                Prof. Madhuri Kadam, Shree L. R. Tiwari College of Engineering
#                Rohinth Ram, Madras Institue of Technology
#                Charaan S., Madras Institue of Technology
#                Nalinkumar S., Madras Institue of Technology
#  ORGANIZATION: eSim Team at FOSSEE, IIT Bombay
#       CREATED: Monday 29, November 2021
#      REVISION: Tuesday 25, January 2022
# =========================================================================


# importing the files and libraries
from PyQt6 import QtCore, QtWidgets
from configuration import Dialogs
from configuration import paths
from frontEnd.theme_utils import zoom_px, on_zoom_changed
from . import Maker
from . import ModelGeneration
from . import createkicad
from . import createkicadCosim
from .model_teardown import (
    _safe_model_subdir, _resolve_backend, _ensure_modpath,
    _strip_modpath_line)
from . import CosimConfig
from .CosimLogger import CosimLog
from .RemoveItemsDialog import RemoveItemsDialog
from .hdl.jobs import BackgroundJob
import os
import shutil
from configuration.Appconfig import Appconfig


class NgVeri(QtWidgets.QWidget):
    '''
        This class create the NgVeri Tab
    '''
    def __init__(self, filecount):
        QtWidgets.QWidget.__init__(self)
        # Maker.addverilog(self)
        self.obj_Appconfig = Appconfig()

        # NGHDL may not be installed/configured yet. Read defensively so a
        # missing or partial ~/.nghdl/config.ini degrades this tab instead of
        # crashing the whole Makerchip dock -- NgVeri is built eagerly, unlike
        # the NGHDL tab which is already lazy+guarded.
        self.nghdl_home = CosimConfig.nghdl_cfg('NGHDL', 'NGHDL_HOME')
        self.release_dir = CosimConfig.nghdl_cfg('NGHDL', 'RELEASE')
        self.src_home = CosimConfig.nghdl_cfg('SRC', 'SRC_HOME')
        self.licensefile = CosimConfig.nghdl_cfg('SRC', 'LICENSE')
        self.config_available = bool(self.nghdl_home)
        digital = CosimConfig.digital_model_root()
        self.digital_home = digital + "/Ngveri"
        # NGHDL (GHDL) models live in a sibling tree under the same icm base.
        self.ghdl_home = digital + "/ghdl"
        # modelParamXML root (from the maker Appconfig, keyed off eSim_HOME).
        # Used to list/resolve d_cosim models, which live only under
        # NgVeriCosim/ and never appear in modpath.lst.
        try:
            self._xml_loc = createkicad.Appconfig.Appconfig.xml_loc
        except AttributeError:
            self._xml_loc = ""
        self.count = 0
        self.text = ""
        self.entry_var = {}
        self.createNgveriWidget()
        self.fname = ""
        self.filecount = filecount

    def createNgveriWidget(self):
        '''
            Creating the various components of the Widget(Ngveri Tab)
        '''
        self.grid = QtWidgets.QGridLayout()
        self.setLayout(self.grid)

        # Row 0 (options) keeps its natural height via the box's Fixed vertical
        # policy; row 1 (terminal) takes all remaining space. The old code
        # used AlignTop + a rowSpan-5/colSpan-0 span that starved row 0 and
        # crushed the options box's contents into each other.
        self.grid.addWidget(self.createoptionsBox(), 0, 0)
        self.grid.addWidget(self.creategroup(), 1, 0)
        self.grid.setRowStretch(0, 0)
        self.grid.setRowStretch(1, 1)


    def addverilog(self):
        '''
            Adding the verilog file in Maker tab to Ngveri Tab automatically
        '''
        # b=Maker.Maker(self)
        print(Maker.verilogFile)
        if self.filecount >= len(Maker.verilogFile) or \
                Maker.verilogFile[self.filecount] == "":
            Dialogs.critical(
                self,
                "Error Message",
                "<b>Error: No Verilog File Chosen. \
                Please choose a verilog file in Makerchip Tab</b>",
                QtWidgets.QMessageBox.StandardButton.Ok)
            self.obj_Appconfig.print_error(
                'No Verilog File Chosen. '
                'Please choose a verilog file in Makerchip Tab'
            )
            return

        self.fname = Maker.verilogFile[self.filecount]
        currentTermLogs = QtWidgets.QTextEdit()
        model = ModelGeneration.ModelGeneration(self.fname, currentTermLogs)
        if not model.require_legacy_toolchain():
            self.entry_var[0].append(currentTermLogs.toHtml())
            return
        file = os.path.splitext(os.path.basename(self.fname))[0]
        # If this name was previously built via d_cosim, ASK, then remove that
        # version first so the switch to the legacy NgVeri flow is clean. A
        # declined switch aborts the build.
        if not self._switch_backends_if_needed("ngveri", file):
            return
        # The remove-model dialog reads modpath.lst fresh each time it opens, so
        # a successful build (model.modpathlst()) is what makes this model show
        # up there -- no combo to pre-register into anymore.

        if not Maker.makerchipTOSAccepted(True):
            Dialogs.warning(
                self, "Warning Message",
                "Please accept the Makerchip Terms of Service "
                "to proceed further.",
                QtWidgets.QMessageBox.StandardButton.Ok
            )

            return

        # Fast file-generation stays on the GUI thread (it runs in
        # milliseconds and is also the ONLY place a dialog is raised --
        # verilogParse's name-mismatch box -- so no QWidget is ever touched
        # off-thread). If any of these fail, abort before spawning a worker.
        try:
            if model.verilogfile() == "Error":
                self._flush_build_logs(currentTermLogs)
                return
            if model.verilogParse() == "Error":
                self._flush_build_logs(currentTermLogs)
                return
            model.getPortInfo()
            model.cfuncmod()
            model.ifspecwrite()
            model.sim_main_header()
            model.sim_main()
            model.modpathlst()
        except Exception as err:
            currentTermLogs.append(
                "Error in Ngspice code model generation "
                "from Verilog: " + str(err))
            currentTermLogs.append(self._build_failure_html())
            self._flush_build_logs(currentTermLogs)
            return

        # The slow half -- verilator, make, the ngspice code-model rebuild and
        # `make install` -- can take several minutes. Run it on a worker thread
        # so the GUI stays responsive (the old QProcess+waitForFinished froze
        # eSim, "not responding" overlay and all, for the whole build). Disable
        # both convert buttons until it returns so a second build can't race
        # this one.
        self._set_convert_buttons_enabled(False)
        model.phase.connect(self._on_build_phase)
        self._show_build_progress(True)
        self._build_model = model            # keep refs alive for the build
        self._build_logs = currentTermLogs
        self._build_job = BackgroundJob(
            self._legacy_build_pipeline, model, parent=self)
        self._build_job.succeeded.connect(self._on_legacy_build_finished)
        self._build_job.failed.connect(self._on_legacy_build_error)
        self._build_job.finished.connect(self._build_job.deleteLater)
        self._build_job.start()

    def _legacy_build_pipeline(self, model):
        '''
            The slow half of the legacy NgVeri build, run on a BackgroundJob
            worker thread: verilator -> make -> copy artifacts -> ngspice
            `make` [-> `make install`]. Each step returns True only when its
            process exits cleanly (exit code 0), so we short-circuit at the
            first failure and base the verdict on real exit codes rather than a
            "is the word 'error' somewhere in the log" search (which both
            passed broken models and failed working ones). Touches only files +
            model.line (queued to the GUI terminal); never a widget directly.
        '''
        ok = (
            model.run_verilator()
            and model.make_verilator()
            and model.copy_verilator()
            and model.runMake()
        )
        if not ok:
            return False
        # make install on EVERY platform: the Windows nghdl tree is configured
        # with prefix=install_dir exactly like Ubuntu, so Ngveri.cm lands
        # where ngspice loads code models from (<install_dir>/lib/ngspice).
        # The old nt hand-copy targeted <NGHDL_HOME>/lib/ngspice, where
        # ngspice never looks.
        return model.runMakeInstall()

    def _on_legacy_build_finished(self, ok):
        '''GUI-thread epilogue for a completed legacy build (success path).'''
        logs = self._build_logs
        logs.append(self._diag_summary_html())
        if ok:
            logs.append('''
                <p style=\" font-size:16pt; font-weight:1000;
                color:#00FF00;\"> Model Created Successfully!
                </p>
            ''')
            placedName = os.path.splitext(
                os.path.basename(self.fname))[0].lower()
            logs.append(
                '<p style="color:#00AA00; font-weight:600;">'
                'Model "' + placedName + '" — place it from the '
                'eSim_Ngveri library in KiCad.</p>')
        else:
            logs.append(self._build_failure_html())
        self._flush_build_logs(logs)

    def _on_legacy_build_error(self, msg):
        '''GUI-thread epilogue when the build worker raised an exception.'''
        self._build_logs.append(
            "Error in Ngspice code model generation from Verilog: " + msg)
        self._build_logs.append(self._build_failure_html())
        self._flush_build_logs(self._build_logs)

    def _diag_summary_html(self):
        '''
            One-line tally of what the toolchain actually reported, printed
            just above the verdict.

            Several hundred lines of gcc/make output scroll past during a
            build, so without a closing count the user judges the run by how
            much coloured text went by -- and a perfectly good model whose
            build emitted a couple of routine compiler warnings reads as a
            failure. Say the numbers, and say plainly that warnings are not
            failures.
        '''
        model = getattr(self, '_build_model', None)
        errors = getattr(model, 'diag_errors', 0)
        warnings = getattr(model, 'diag_warnings', 0)

        def _count(n, word):
            return str(n) + ' ' + word + ('' if n == 1 else 's')

        text = ('Toolchain reported ' + _count(errors, 'error') + ' and ' +
                _count(warnings, 'warning') + '.')
        if warnings and not errors:
            text += ' Warnings do not stop a build.'
        return ('<p style="font-size:11pt; color:#666666;">' + text + '</p>')

    @staticmethod
    def _build_failure_html():
        return '''
            <p style=\" font-size:16pt; font-weight:1000;
            color:#FF0000;\">There was an error during model creation,
            <br/>Please rectify the error and try again!
            </p>
        '''

    def _set_convert_buttons_enabled(self, enabled):
        '''Enable/disable the two convert buttons around an async build.'''
        for name in ("addverilogbutton", "addcosimbutton"):
            btn = getattr(self, name, None)
            if btn is not None:
                btn.setEnabled(enabled)

    def _show_build_progress(self, on, message="Starting build…"):
        '''
            Show/hide the live build progress bar + phase label. Called on the
            GUI thread: on when a build is dispatched, off in the shared build
            epilogue (_flush_build_logs).
        '''
        if on:
            self.buildStatus.setText(message)
        self.buildStatus.setVisible(on)
        self.buildBar.setVisible(on)

    def _on_build_phase(self, phase):
        '''
            Slot for ModelGeneration.phase (queued from the build worker):
            name the step currently running under the spinning bar.
        '''
        self.buildStatus.setText(phase)

    def _flush_build_logs(self, logs):
        '''
            Append the build's captured HTML into the visible terminal, scroll
            to the bottom, re-enable the convert buttons and drop the build
            refs. Shared by the success, failure and pre-worker abort paths.
        '''
        self.entry_var[0].append(logs.toHtml())
        self.entry_var[0].verticalScrollBar().setValue(
            self.entry_var[0].verticalScrollBar().maximum()
        )
        self._show_build_progress(False)
        self._set_convert_buttons_enabled(True)
        self._build_model = None
        self._build_logs = None

    def addverilog_cosim(self):
        '''
            d_cosim (Icarus Verilog) flow. Compiles the chosen Verilog file to a
            vvp via iverilog and creates an "NgVeriCosim" KiCad symbol. Unlike
            "Convert Verilog to Ngspice" (legacy static Ngveri.cm), this needs no
            C/C++ compiler, never rebuilds ngspice, and runs fully locally (no
            Makerchip). Gated on the d_cosim toolchain being present.
        '''
        # Doctor gate: covers iverilog/vvp/libvvp AND the ngspice ivlng
        # adapter, with the exact probed paths + fix hints, so a broken
        # install fails here instead of at simulation time.
        from . import ToolchainCheck
        doctor_msg = ToolchainCheck.failure_message(ToolchainCheck.DCOSIM)
        if doctor_msg:
            Dialogs.warning(
                self, "d_cosim unavailable", doctor_msg,
                QtWidgets.QMessageBox.StandardButton.Ok)
            return

        if len(Maker.verilogFile) < (self.filecount + 1) or \
                Maker.verilogFile[self.filecount] == "":
            Dialogs.critical(
                self, "Error Message",
                "<b>Error: No Verilog File Chosen. Please choose a "
                "verilog file in Makerchip Tab</b>",
                QtWidgets.QMessageBox.StandardButton.Ok)
            return

        self.fname = Maker.verilogFile[self.filecount]
        currentTermLogs = QtWidgets.QTextEdit()
        model = ModelGeneration.ModelGeneration(self.fname, currentTermLogs)
        # Canonical model identity = lowercased basename. The symbol, param
        # XML, compiled vvp, picker entry and netlist all key off THIS one
        # name, so the build location matches the netlister's lookup (a
        # case-sensitive filesystem otherwise loses the vvp at sim time).
        modelname = os.path.splitext(
            os.path.basename(self.fname))[0].lower()

        # Fast GUI-thread half: the backend-switch prompt, source generation and
        # parse. These are the only dialog-raising steps, so they stay on the
        # GUI thread; abort before spawning a worker if any fails.
        try:
            # If this name was previously built via the legacy NgVeri flow,
            # ASK, then remove that version FIRST -- before verilogfile()
            # repopulates the shared <model>/ dir with the new source. Doing it
            # after would rmtree the freshly-copied .v out from under
            # build_cosim. A declined switch aborts the build.
            if not self._switch_backends_if_needed("cosim", modelname):
                return
            if model.verilogfile() == "Error":
                self._flush_build_logs(currentTermLogs)
                return
            if model.verilogParse(make_symbol=False) == "Error":
                self._flush_build_logs(currentTermLogs)
                return
        except Exception as err:
            currentTermLogs.append(
                "Error in d_cosim model creation: " + str(err))
            self._flush_build_logs(currentTermLogs)
            return

        # Slow half: the iverilog compile (build_cosim) can take a while on a
        # large model. Run it on a worker thread so the GUI stays responsive
        # (it used to block the event loop). KiCad symbol creation stays in the
        # epilogue because it may raise an overwrite-confirmation dialog, which
        # must run on the GUI thread. Disable both convert buttons until the
        # build returns so a second build can't race it.
        self._set_convert_buttons_enabled(False)
        model.phase.connect(self._on_build_phase)
        self._show_build_progress(True, "Building d_cosim model…")
        self._build_model = model            # keep refs alive for the build
        self._build_logs = currentTermLogs
        self._cosim_modelname = modelname
        self._build_job = BackgroundJob(model.build_cosim, parent=self)
        self._build_job.succeeded.connect(self._on_cosim_build_finished)
        self._build_job.failed.connect(self._on_cosim_build_error)
        self._build_job.finished.connect(self._build_job.deleteLater)
        self._build_job.start()

    def _on_cosim_build_finished(self, sim_lib):
        '''GUI-thread epilogue for a completed d_cosim build. build_cosim ran on
        the worker; the KiCad symbol is created here (it may prompt).'''
        model = self._build_model
        logs = self._build_logs
        modelname = self._cosim_modelname
        try:
            if sim_lib and sim_lib != "Error":
                model.clog.phase("Create KiCad symbol")
                schematicLib = createkicadCosim.CosimSchematic()
                schematicLib.init(modelname, model.modelpath, "icarus", sim_lib)
                if schematicLib.createKicadSymbol() != "Error":
                    # The model now exists on disk as NgVeriCosim/<name>.xml,
                    # which is what the remove-model dialog scans -- nothing to
                    # register in a combo anymore.
                    model.clog.ok(
                        'd_cosim model "' + modelname + '" created (Icarus). '
                        'Place it from the eSim_NgVeriCosim library.')
                else:
                    model.clog.error(
                        'KiCad symbol generation failed for "' + modelname +
                        '". The vvp built but the symbol was not created.')
            # sim_lib == "Error": build_cosim already logged the phased failure.
        except Exception as err:
            model.clog.error("Error in d_cosim model creation: " + str(err))
        self._flush_build_logs(logs)

    def _on_cosim_build_error(self, msg):
        '''GUI-thread epilogue when the d_cosim build worker raised.'''
        self._build_model.clog.error(
            "Error in d_cosim model creation: " + msg)
        self._flush_build_logs(self._build_logs)

    def addfile(self):
        '''
            This function is used to add additional files required
            by the verilog top module
        '''
        if len(Maker.verilogFile) < (self.filecount + 1):
            Dialogs.critical(
                self,
                "Error Message",
                "<b>Error: No Verilog File Chosen. \
                Please choose a verilog file in Makerchip Tab</b>",
                QtWidgets.QMessageBox.StandardButton.Ok)
            self.obj_Appconfig.print_error(
                'No Verilog File Chosen. Please choose \
                 a verilog file in Makerchip Tab')
            return

        self.fname = Maker.verilogFile[self.filecount]
        model = ModelGeneration.ModelGeneration(self.fname, self.entry_var[0])
        # model.verilogfile()
        model.addfile()

    def addfolder(self):
        '''
            This function is used to add additional folder required
            by the verilog top module.
        '''
        if len(Maker.verilogFile) < (self.filecount + 1):
            Dialogs.critical(
                self,
                "Error Message",
                "<b>Error: No Verilog File Chosen. \
                Please choose a verilog file in Makerchip Tab</b>",
                QtWidgets.QMessageBox.StandardButton.Ok)
            self.obj_Appconfig.print_error(
                'No Verilog File Chosen. Please choose \
                a verilog file in Makerchip Tab')
            return
        self.fname = Maker.verilogFile[self.filecount]
        model = ModelGeneration.ModelGeneration(self.fname, self.entry_var[0])
        # model.verilogfile()
        model.addfolder()

    def clearTerminal(self):
        '''
            This function is used to clear the terminal
        '''
        self.entry_var[0].setText("")

    def createoptionsBox(self):
        '''
            This function is used to create buttons/options
        '''
        self.optionsbox = QtWidgets.QGroupBox()
        self.optionsbox.setTitle("Select Options")

        # A single vertical stack gives a clean top-to-bottom read: one verb
        # heading, the two convert actions, then the low-stakes utilities.
        # (The old grid stretched every button to full width and left a hole
        # in the utility row.) The "which method / when" copy is NOT here --
        # it lives in the terminal side column so the top stays uncluttered.
        outer = QtWidgets.QVBoxLayout()
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(10)

        self.optionsgroupbtn = QtWidgets.QButtonGroup()

        # The verb is stated ONCE as a heading; the two buttons then only need
        # to name the method. Killing the repeated "Convert to Ngspice (...)"
        # prefix is what stops them reading as two identical buttons.
        convertHeading = QtWidgets.QLabel("Convert Verilog to Ngspice")
        convertHeading.setProperty("cssClass", "heading")
        outer.addWidget(convertHeading)

        expanding = QtWidgets.QSizePolicy.Policy.Expanding
        fixed = QtWidgets.QSizePolicy.Policy.Fixed

        # Two backends, same result -> equal weight (both primary, equal
        # width). Dual Co-sim sits FIRST (left = first in reading order) so a
        # user reaching for the simpler path lands on it without us having to
        # dim the other or slap on a "recommended" badge.
        convert_row = QtWidgets.QHBoxLayout()
        convert_row.setSpacing(12)

        self.addcosimbutton = QtWidgets.QPushButton("Dual Co-sim")
        self.addcosimbutton.setProperty("cssClass", "primary")
        self.addcosimbutton.setSizePolicy(expanding, fixed)
        self.addcosimbutton.setMinimumHeight(44)
        self.addcosimbutton.setToolTip(
            "Icarus Verilog co-simulation via ngspice d_cosim: "
            "no C/C++ compiler and no ngspice rebuild")
        self.optionsgroupbtn.addButton(self.addcosimbutton)
        self.addcosimbutton.clicked.connect(self.addverilog_cosim)
        convert_row.addWidget(self.addcosimbutton)

        self.addverilogbutton = QtWidgets.QPushButton("NgVeri")
        self.addverilogbutton.setProperty("cssClass", "primary")
        self.addverilogbutton.setSizePolicy(expanding, fixed)
        self.addverilogbutton.setMinimumHeight(44)
        self.addverilogbutton.setToolTip(
            "Compiles Verilog into a native ngspice code model "
            "(builds a C model and rebuilds ngspice)")
        self.addverilogbutton.setToolTipDuration(5000)
        self.optionsgroupbtn.addButton(self.addverilogbutton)
        self.addverilogbutton.clicked.connect(self.addverilog)
        convert_row.addWidget(self.addverilogbutton)

        outer.addLayout(convert_row)

        # Low-stakes utilities. Three equal-width buttons span the full row
        # (equal stretch, no trailing spacer) so the space is filled evenly
        # instead of leaving a gap between the dependency pair and Clear
        # Terminal.
        util_row = QtWidgets.QHBoxLayout()
        util_row.setSpacing(8)

        self.addfilebutton = QtWidgets.QPushButton("Add dependency files…")
        self.addfilebutton.setSizePolicy(expanding, fixed)
        self.optionsgroupbtn.addButton(self.addfilebutton)
        self.addfilebutton.clicked.connect(self.addfile)
        util_row.addWidget(self.addfilebutton, 1)

        self.addfolderbutton = QtWidgets.QPushButton("Add dependency folder…")
        self.addfolderbutton.setSizePolicy(expanding, fixed)
        self.optionsgroupbtn.addButton(self.addfolderbutton)
        self.addfolderbutton.clicked.connect(self.addfolder)
        util_row.addWidget(self.addfolderbutton, 1)

        self.clearTerminalBtn = QtWidgets.QPushButton("Clear Terminal")
        # Low-stakes utility — text-only tertiary so it recedes.
        self.clearTerminalBtn.setProperty("cssClass", "tertiary")
        self.clearTerminalBtn.setSizePolicy(expanding, fixed)
        self.optionsgroupbtn.addButton(self.clearTerminalBtn)
        self.clearTerminalBtn.clicked.connect(self.clearTerminal)
        util_row.addWidget(self.clearTerminalBtn, 1)

        outer.addLayout(util_row)

        self.optionsbox.setLayout(outer)

        # Fixed vertical policy: the box keeps its natural height instead of
        # being squeezed by the terminal group's row-span below it (which was
        # crushing the three rows into each other).
        self.optionsbox.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Fixed)

        return self.optionsbox

    def _list_models(self):
        '''
            Scan disk for every removable model and tag each with its backend.
            Disk is the single source of truth (survives restarts and backend
            switches): legacy NgVeri models are lines in modpath.lst; d_cosim
            models are NgVeriCosim/<name>.xml files (never in modpath.lst).

            Returns (names, badges) where badges maps each name to
            "NgVeri"/"d_cosim". modpath.lst is created on demand so a fresh
            install lists cleanly.
        '''
        badges = {}

        modpath_file = _ensure_modpath(self.digital_home + '/modpath.lst')
        with open(modpath_file) as fh:
            for line in fh:
                name = line.strip()
                if name:
                    badges[name] = "NgVeri"

        # NGHDL (GHDL) models are intentionally NOT listed here: they are
        # uninstalled from the NGHDL app's own "Remove Models" button (the
        # VHDL workflow), not this Verilog-side dialog.
        cosim_dir = os.path.join(self._xml_loc, 'NgVeriCosim')
        if os.path.isdir(cosim_dir):
            for fname in sorted(os.listdir(cosim_dir)):
                if fname.endswith('.xml'):
                    name = fname[:-4]
                    # d_cosim wins the badge: a name present in both flows is
                    # torn down by _model_backend below, which trusts the xml.
                    badges[name] = "d_cosim"

        return list(badges.keys()), badges

    def open_remove_models(self):
        '''
            Open the searchable, multi-select dialog and tear down whatever the
            user picks. Dispatches each name to the right backend teardown --
            legacy NgVeri (Verilator -> Ngveri.cm) vs d_cosim (Icarus ->
            eSim_NgVeriCosim) -- since the two share nothing on disk and the
            wrong teardown silently leaves a model behind.
        '''
        names, badges = self._list_models()
        if not names:
            Dialogs.information(
                self, "Remove Models",
                "There are no models to remove.",
                QtWidgets.QMessageBox.StandardButton.Ok)
            return

        dlg = RemoveItemsDialog(
            "Remove Models", names, badges=badges,
            item_noun="model", parent=self)
        if not dlg.exec():
            return

        log = CosimLog(self.entry_var[0])
        ngveri_needed = False
        for name in dlg.selected_items():
            # Belt-and-braces: a blank name must never reach the teardown
            # helpers (os.path.join(base, "") -> "base/" -> rmtree wipes all
            # models). The helpers guard too; this is defence in depth.
            if not name or not name.strip():
                continue
            # On-disk layout is the source of truth for which engine owns it.
            backend = self._model_backend(name)
            if backend == "cosim":
                self._remove_cosim_model(name)
            else:
                # Defer the (expensive) Ngveri.cm rebuild to one pass after
                # every model is gone, not once per model.
                self._remove_ngveri_model(name, rebuild=False)
                ngveri_needed = True

        if ngveri_needed:
            self._rebuild_ngveri_cm(log)

    def _model_backend(self, name):
        '''
            Resolve which backend created a model from the on-disk
            modelParamXML layout -- the single source of truth that survives
            restarts: NgVeriCosim/<name>.xml => cosim, Nghdl/<name>.xml =>
            nghdl, else legacy NgVeri (ngveri). Delegates to the tested
            free function _resolve_backend.
        '''
        return _resolve_backend(self._xml_loc, name)

    def _remove_cosim_model(self, text):
        '''
            Tear down a d_cosim (Icarus) model: drop its symbol from
            eSim_NgVeriCosim.kicad_sym + its NgVeriCosim/<name>.xml, remove the
            compiled vvp, and -- crucially -- strip the model from modpath.lst.

            The d_cosim vvp and a legacy NgVeri build share ONE directory
            (<DIGITAL_MODEL>/Ngveri/<model>/). If this model had ALSO been built
            via the legacy flow it has a modpath.lst line; rmtree'ing the shared
            dir here without dropping that line leaves a ghost entry that makes
            cmpp abort every later Ngveri.cm build. So we always strip the line.
            Still NO Ngveri.cm rebuild -- d_cosim stays light; the next legacy
            build (and prune_modpathlst) reconciles ngspice's side.
        '''
        log = CosimLog(self.entry_var[0])
        log.phase('REMOVE d_cosim model "' + str(text) + '"')

        if not text or not str(text).strip():
            log.warn("Refusing to remove a d_cosim model with a blank name.")
            return

        try:
            symbol = createkicadCosim.CosimSchematic()
            symbol.init(text, "")
            symbol.deleteKicadSymbol()
            log.info("Removed eSim_NgVeriCosim symbol + NgVeriCosim/" +
                     str(text) + ".xml")
        except Exception as err:
            log.warn("Could not remove d_cosim KiCad symbol for '" +
                     str(text) + "': " + str(err))

        # Drop the model from modpath.lst if present (guarded), so a model that
        # was built by BOTH flows doesn't leave a build-breaking ghost behind.
        self._strip_modpathlst(text, log)

        vvp = CosimConfig.cosim_vvp_path(text)
        if vvp:
            build_dir = os.path.dirname(vvp)
            # vvp = <DIGITAL_MODEL>/Ngveri/<text>/<text>, so the build dir's
            # basename must equal the model name. A blank name would collapse it
            # to the parent Ngveri/ dir -- bail rather than rmtree all models.
            if os.path.basename(os.path.normpath(build_dir)) != str(text).strip():
                log.warn("Refusing to remove unexpected d_cosim build dir: " +
                         build_dir)
                build_dir = None
        else:
            build_dir = None
        if build_dir:
            try:
                shutil.rmtree(build_dir)
                log.info("Removed build dir: " + build_dir)
            except FileNotFoundError:
                log.detail("Build dir already absent: " + build_dir)
            except OSError as err:
                log.warn("Could not remove d_cosim build dir '" +
                         build_dir + "': " + str(err))
        log.ok('d_cosim model "' + str(text) + '" removed.')

    def _strip_modpathlst(self, text, log=None):
        '''
            Remove every line equal to `text` from modpath.lst (idempotent).
            Returns True if a line was dropped. Logs via `log` when given.

            Delegates to the stdlib-only shared helper so the rewrite is atomic
            and identical to the NGHDL-side teardown: a crash part-way through
            truncates the list, and cmpp then aborts EVERY later code-model
            build with an error pointing nowhere near model removal.
        '''
        dropped = _strip_modpath_line(
            self.digital_home + '/modpath.lst', str(text))
        if dropped and log:
            log.info('Dropped "' + str(text) + '" from modpath.lst')
        return dropped

    def _remove_ngveri_model(self, text, rebuild=True):
        '''
            Tear down a legacy NgVeri (Verilator) model: drop it from
            modpath.lst, remove its eSim_Ngveri symbol + param XML, delete the
            per-model build dir, then rebuild/reinstall Ngveri.cm so ngspice
            truly unlinks it.

            Pass rebuild=False when removing several models in one pass; the
            caller then runs a single _rebuild_ngveri_cm() at the end rather
            than rebuilding the code model once per model.
        '''
        log = CosimLog(self.entry_var[0])
        log.phase('REMOVE NgVeri model "' + str(text) + '"')

        # Drop the model from modpath.lst (guarded: absent => no crash)
        self._strip_modpathlst(text, log)

        # Remove the KiCad symbol + orphan param XML too, so the model
        # actually disappears from eSim_Ngveri in KiCad (previously left
        # behind forever).
        try:
            symbol = createkicad.AutoSchematic()
            symbol.init(text, "")
            symbol.deleteKicadSymbol()
            log.info("Removed eSim_Ngveri symbol + param XML")
        except Exception as err:
            log.warn("Could not remove KiCad symbol for '" +
                     str(text) + "': " + str(err))

        # Drop BOTH per-model dirs:
        #   * source   <digital_home>/<model>  -- holds ifspec.ifs (what cmpp
        #     reads) + cfunc/sim_main + the shared d_cosim vvp, and
        #   * release  <release>/src/xspice/icm/Ngveri/<model>  -- holds the
        #     compiled .o/.a that otherwise get re-bundled, keeping the model
        #     answering in ngspice after its symbol/list line are gone.
        for label, base in (
                ("source", self.digital_home),
                ("release", os.path.join(
                    self.release_dir, "src/xspice/icm/Ngveri"))):
            model_dir = _safe_model_subdir(base, text)
            if model_dir is None:
                log.warn("Refusing to remove unsafe " + label +
                         " dir for model name: " + repr(text))
                continue
            try:
                shutil.rmtree(model_dir)
                log.info("Removed " + label + " dir: " + model_dir)
            except FileNotFoundError:
                log.detail(label + " dir already absent: " + model_dir)
            except OSError as err:
                log.warn("Could not remove " + label + " dir '" +
                         model_dir + "': " + str(err))

        log.ok('NgVeri model "' + str(text) + '" files removed.')
        if rebuild:
            self._rebuild_ngveri_cm(log)

    def _current_verilog_fname(self):
        '''
            The current tab's Verilog path, or "" when this tab has no slot in
            Maker.verilogFile yet (a remove-only session). The whole-icm
            rebuild ignores the value entirely, so an empty string is safe and
            this guard avoids an IndexError -- matching the length checks the
            add* siblings already do.
        '''
        vf = Maker.verilogFile
        return vf[self.filecount] if len(vf) > self.filecount else ""

    def _rebuild_ngveri_cm(self, log):
        '''
            Rebuild and reinstall the Ngveri.cm code model so ngspice unlinks
            every model already stripped from modpath.lst. Run once after a
            batch removal. prune_modpathlst() first sweeps any unrelated ghost
            entries so the rebuild can't fail on a dead line left by something
            else.
        '''
        self.fname = self._current_verilog_fname()
        model = ModelGeneration.ModelGeneration(
            self.fname, self.entry_var[0])
        if not model.require_legacy_toolchain():
            return
        model.prune_modpathlst()

        try:
            log.phase("Rebuild Ngveri.cm")
            # make + make install on every platform (Windows is configured
            # with prefix=install_dir like Ubuntu; see _legacy_build_pipeline).
            ok = model.runMake()
            ok = model.runMakeInstall() and ok
            if not ok:
                raise RuntimeError(
                    "the ngspice code-model rebuild returned a "
                    "non-zero exit status")
        except Exception as err:
            Dialogs.critical(
                self, "Error Message",
                "The ngspice code model could not be rebuilt after removal: " +
                str(err),
                QtWidgets.QMessageBox.StandardButton.Ok
            )
        else:
            log.ok("Ngveri.cm rebuilt.")

    # ------------------------------------------------------------------ #
    #  Backend switching (d_cosim <-> legacy NgVeri for the same model)
    # ------------------------------------------------------------------ #
    def _legacy_registered(self, name):
        '''True if `name` has a legacy NgVeri line in modpath.lst.'''
        try:
            with open(self.digital_home + '/modpath.lst') as f:
                return any(ln.strip() == str(name) for ln in f)
        except OSError:
            return False

    def _purge_legacy_registration(self, name, log):
        '''
            Light teardown of a legacy NgVeri model: drop its modpath.lst line,
            eSim_Ngveri symbol and both build dirs -- but NO Ngveri.cm rebuild,
            so a d_cosim create stays independent of the Verilator toolchain.
            prune_modpathlst() keeps later legacy builds safe.
        '''
        self._strip_modpathlst(name, log)
        try:
            symbol = createkicad.AutoSchematic()
            symbol.init(name, "")
            symbol.deleteKicadSymbol()
            log.info("Removed eSim_Ngveri symbol for " + str(name))
        except Exception as err:
            log.warn("Could not remove eSim_Ngveri symbol for '" +
                     str(name) + "': " + str(err))
        for label, base in (
                ("source", self.digital_home),
                ("release", os.path.join(
                    self.release_dir, "src/xspice/icm/Ngveri"))):
            d = _safe_model_subdir(base, name)
            if d is None:
                log.warn("Refusing to remove unsafe " + label +
                         " dir for model name: " + repr(name))
                continue
            try:
                shutil.rmtree(d)
                log.info("Removed " + label + " dir: " + d)
            except FileNotFoundError:
                pass
            except OSError as err:
                log.warn("Could not remove " + label + " dir '" +
                         d + "': " + str(err))

    def _confirm_switch(self, name, from_backend, to_backend):
        '''
            Yes/No dialog shown before replacing an existing model's backend.
            Returns True if the user agreed to switch.
        '''
        ret = Dialogs.question(
            self, "Switch model backend?",
            '<b>"' + str(name) + '"</b> already exists as a <b>' +
            from_backend + '</b> model.<br><br>Rebuild it as a <b>' +
            to_backend + '</b> model instead?<br>'
            'The existing ' + from_backend + ' version will be removed.',
            QtWidgets.QMessageBox.StandardButton.Yes |
            QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No)
        return ret == QtWidgets.QMessageBox.StandardButton.Yes

    def _switch_backends_if_needed(self, target, name):
        '''
            If `name` already exists under the OTHER backend, ASK the user, then
            remove that stale copy so building under `target` is a clean switch
            instead of a duplicate (two symbols, a co-mingled build dir, or a
            ghost modpath.lst entry). target is "ngveri" or "cosim".

            Returns True to proceed with the build, False if no switch is needed
            is also True; only a user-declined switch returns False (the caller
            must then abort the build).
        '''
        log = CosimLog(self.entry_var[0])
        low = str(name).lower()
        if target == "ngveri":
            cosim_xml = os.path.join(
                self._xml_loc, 'NgVeriCosim', low + '.xml')
            if not os.path.isfile(cosim_xml):
                return True
            if not self._confirm_switch(low, "d_cosim", "NgVeri (Verilator)"):
                log.warn('Switch cancelled -- "' + low + '" kept as a d_cosim '
                         'model. NgVeri build aborted.')
                return False
            log.phase('SWITCH backend: d_cosim -> NgVeri for "' + low + '"')
            log.info("Removing existing d_cosim version first.")
            self._remove_cosim_model(low)
            return True
        elif target == "cosim":
            if not (self._legacy_registered(name) or
                    self._legacy_registered(low)):
                return True
            reg = name if self._legacy_registered(name) else low
            if not self._confirm_switch(reg, "NgVeri (Verilator)", "d_cosim"):
                log.warn('Switch cancelled -- "' + str(reg) + '" kept as a '
                         'NgVeri model. d_cosim build aborted.')
                return False
            log.phase('SWITCH backend: NgVeri -> d_cosim for "' +
                      str(reg) + '"')
            log.info("Removing existing NgVeri version first "
                     "(no Ngveri.cm rebuild).")
            self._purge_legacy_registration(reg, log)
            return True
        return True

    def _lint_off_path(self):
        '''Path to library/tlv/lint_off.txt, anchored to the install root.'''
        return paths.library_path("tlv/lint_off.txt")

    def _list_lint_off(self):
        '''Current lint_off entries (one per non-blank line), in file order.'''
        try:
            with open(self._lint_off_path()) as fh:
                return [ln.strip() for ln in fh if ln.strip()]
        except OSError:
            return []

    def open_remove_lint_off(self):
        '''
            Open the searchable, multi-select dialog and drop the chosen
            lint_off entries from library/tlv/lint_off.txt in one pass.
        '''
        entries = self._list_lint_off()
        if not entries:
            Dialogs.information(
                self, "Remove lint_off",
                "There are no lint_off entries to remove.",
                QtWidgets.QMessageBox.StandardButton.Ok)
            return

        dlg = RemoveItemsDialog(
            "Remove lint_off", entries, item_noun="lint_off entry",
            parent=self)
        if not dlg.exec():
            return

        doomed = set(dlg.selected_items())
        try:
            kept = [e for e in self._list_lint_off() if e not in doomed]
            with open(self._lint_off_path(), 'w') as fh:
                fh.write("\n".join(kept) + ("\n" if kept else ""))
        except OSError as err:
            Dialogs.warning(
                self, "Warning",
                "Could not update lint_off.txt: " + str(err),
                QtWidgets.QMessageBox.StandardButton.Ok)

    def add_lint_off(self):
        '''
            This is to add lint_off comments needed by the verilator warnings.
            This function writes to the lint_off.txt in the library/tlv folder.
        '''
        text = self.entry_var[3].text().strip()
        if not text:
            return

        # Dedup against the file (the picker is now a dialog read fresh each
        # time, so there is no combo to query for an existing entry).
        if text not in self._list_lint_off():
            with open(self._lint_off_path(), 'a+') as fh:
                fh.write(text + "\n")
        self.entry_var[3].setText("")

    def creategroup(self):
        '''
            Creates various other groups like terminal, remove modlst,
            remove lint_off and add lint_off
        '''
        self.trbox = QtWidgets.QGroupBox()
        self.trbox.setTitle("Terminal")
        # self.trbox.setDisabled(True)
        # self.trbox.setVisible(False)
        self.trgrid = QtWidgets.QGridLayout()
        self.trbox.setLayout(self.trgrid)
        self.count = 0

        self.start = QtWidgets.QLabel("Terminal")
        # self.trgrid.addWidget(self.start, 2,0)
        self.trgrid.setContentsMargins(12, 12, 12, 12)
        self.trgrid.setHorizontalSpacing(12)

        # Left: the console fills the panel and grows with the window.
        self.entry_var[self.count] = QtWidgets.QTextEdit()
        self.entry_var[self.count].setReadOnly(1)
        self.entry_var[self.count].setMaximumWidth(1000)
        self.entry_var[self.count].setMaximumHeight(1000)
        self.trgrid.addWidget(self.entry_var[self.count], 0, 0)
        self.count += 1

        # Right: model-management controls stacked in a tidy column that hugs
        # the top of the console instead of floating in the grid's empty rows
        # (which used to spread the buttons out and leave a void beneath them).
        controls = QtWidgets.QVBoxLayout()
        controls.setSpacing(8)
        controls.setContentsMargins(0, 0, 0, 0)

        # A button opens a searchable, multi-select dialog instead of the old
        # giant QComboBox -- whose popup, crippled by the groupbox stylesheet,
        # listed every model with no scrollbar and deleted one at a time.
        self.entry_var[self.count] = QtWidgets.QPushButton(
            "Remove Verilog Models…")
        self.entry_var[self.count].clicked.connect(self.open_remove_models)
        controls.addWidget(self.entry_var[self.count])
        self.count += 1

        self.entry_var[self.count] = QtWidgets.QPushButton("Remove lint_off")
        self.entry_var[self.count].clicked.connect(self.open_remove_lint_off)
        controls.addWidget(self.entry_var[self.count])
        self.count += 1

        # lint_off entry + its Add button share one row so they read as a pair.
        add_row = QtWidgets.QHBoxLayout()
        add_row.setSpacing(6)
        self.entry_var[self.count] = QtWidgets.QLineEdit(self)
        self.entry_var[self.count].setPlaceholderText("lint_off entry")
        add_row.addWidget(self.entry_var[self.count], 1)
        self.count += 1
        self.entry_var[self.count] = QtWidgets.QPushButton("Add lint_off")
        self.entry_var[self.count].clicked.connect(self.add_lint_off)
        add_row.addWidget(self.entry_var[self.count])
        self.count += 1
        controls.addLayout(add_row)

        # The convert top bar stays uncluttered by parking the "which method,
        # when" explainer down here, in what used to be dead space beneath the
        # lint controls. Neutral copy: names what each backend does and states
        # they are equivalent, without claiming either is buggy or naming an
        # unverified cause ("doesn't complete" is true of any tool). "caps"
        # and "muted" QLabel classes are theme-safe (defined in both QSS).
        controls.addSpacing(14)

        convert_caption = QtWidgets.QLabel("CONVERT METHODS")
        convert_caption.setProperty("cssClass", "caps")
        controls.addWidget(convert_caption)

        self.convertHint = QtWidgets.QLabel(
            "Both convert your Verilog into an ngspice model — same result, "
            "different backend.\n\n"
            "Dual Co-sim runs Icarus through ngspice d_cosim: no compiler "
            "and no ngspice rebuild.\n\n"
            "NgVeri builds a native ngspice code model.\n\n"
            "If one doesn't complete, use the other.")
        self.convertHint.setWordWrap(True)
        self.convertHint.setProperty("cssClass", "muted")
        controls.addWidget(self.convertHint)

        # Stretch pins everything to the top; empty space collapses below,
        # flush with the console rather than wedged between the widgets.
        controls.addStretch(1)

        controls_box = QtWidgets.QWidget()
        controls_box.setLayout(controls)
        # The buttons in this column ("Add Project", "Remove Models", ...) take
        # their padding and font from the QSS, which scales with zoom -- so the
        # column that holds them has to as well, or their labels outgrow it.
        # Registered rather than set once: this panel outlives a zoom change.
        on_zoom_changed(
            controls_box,
            lambda z, w=controls_box: w.setFixedWidth(zoom_px(210, z)))
        self.trgrid.addWidget(
            controls_box, 0, 1, QtCore.Qt.AlignmentFlag.AlignTop)

        # Live build progress under the console: an indeterminate bar + a label
        # naming the current step, shown ONLY while a convert build runs on the
        # worker thread. The bar animates because the GUI event loop stays free
        # (the build is off-thread), and the label is driven by
        # ModelGeneration.phase -- so a long verilator/make reads as "working",
        # not "hung".
        self.buildStatus = QtWidgets.QLabel("")
        self.buildStatus.setProperty("cssClass", "muted")
        self.buildStatus.setVisible(False)
        self.buildBar = QtWidgets.QProgressBar()
        self.buildBar.setRange(0, 0)          # 0..0 == indeterminate "busy"
        self.buildBar.setTextVisible(False)
        self.buildBar.setVisible(False)
        progress_row = QtWidgets.QHBoxLayout()
        progress_row.setContentsMargins(0, 6, 0, 0)
        progress_row.setSpacing(10)
        progress_row.addWidget(self.buildStatus)
        progress_row.addWidget(self.buildBar, 1)
        progress_wrap = QtWidgets.QWidget()
        progress_wrap.setLayout(progress_row)
        self.trgrid.addWidget(progress_wrap, 1, 0, 1, 2)
        # Console takes all the vertical slack; the progress row stays compact.
        self.trgrid.setRowStretch(0, 1)
        self.trgrid.setRowStretch(1, 0)

        # Console soaks up horizontal space; the control column stays compact.
        self.trgrid.setColumnStretch(0, 1)
        self.trgrid.setColumnStretch(1, 0)

        # Border/title styling comes from the global Aurora QGroupBox QSS so
        # the group reads with the same gradient hairline as the rest of eSim.

        return self.trbox
