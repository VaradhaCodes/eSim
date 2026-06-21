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
from . import Maker
from . import ModelGeneration
from . import createkicad
from . import createkicadCosim
from . import CosimConfig
from .CosimLogger import CosimLog
import os
import shutil
from configuration.Appconfig import Appconfig
from configparser import ConfigParser


def _safe_model_subdir(base, name):
    """Resolve ``<base>/<name>`` for deletion, but ONLY when it is provably a
    single-component subdirectory strictly inside ``base``.

    Returns the absolute path, or ``None`` when ``name`` is empty/blank, holds a
    path separator, is ``.``/``..``, or resolves to ``base`` itself or outside
    it. Callers MUST treat ``None`` as "do not delete anything".

    This is the guard that stops a blank model name from collapsing
    ``os.path.join(base, "")`` to ``"base/"`` and ``shutil.rmtree`` wiping the
    whole models directory.
    """
    if not name or not str(name).strip():
        return None
    name = str(name).strip()
    if (os.sep in name or (os.altsep and os.altsep in name)
            or name in ('.', '..')):
        return None
    base_abs = os.path.abspath(base)
    target = os.path.abspath(os.path.join(base_abs, name))
    if target == base_abs:
        return None
    try:
        if os.path.commonpath([base_abs, target]) != base_abs:
            return None
    except ValueError:
        # Different drives (Windows) -> not a subpath.
        return None
    return target


class NgVeri(QtWidgets.QWidget):
    '''
        This class create the NgVeri Tab
    '''
    def __init__(self, filecount):
        QtWidgets.QWidget.__init__(self)
        # Maker.addverilog(self)
        self.obj_Appconfig = Appconfig()

        if os.name == 'nt':
            self.home = os.path.join('library', 'config')
        else:
            self.home = os.path.expanduser('~')

        self.parser = ConfigParser()
        self.parser.read(os.path.join(
            self.home, os.path.join('.nghdl', 'config.ini')))
        self.nghdl_home = self.parser.get('NGHDL', 'NGHDL_HOME')
        self.release_dir = self.parser.get('NGHDL', 'RELEASE')
        self.src_home = self.parser.get('SRC', 'SRC_HOME')
        self.licensefile = self.parser.get('SRC', 'LICENSE')
        self.digital_home = self.parser.get('NGHDL', 'DIGITAL_MODEL')
        self.digital_home = self.digital_home + "/Ngveri"
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

        self.grid.addWidget(self.createoptionsBox(), 0, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        self.grid.addWidget(self.creategroup(), 1, 0, 5, 0)

        self.show()

    def addverilog(self):
        '''
            Adding the verilog file in Maker tab to Ngveri Tab automatically
        '''
        # b=Maker.Maker(self)
        print(Maker.verilogFile)
        if Maker.verilogFile[self.filecount] == "":
            reply = QtWidgets.QMessageBox.critical(
                None,
                "Error Message",
                "<b>Error: No Verilog File Chosen. \
                Please choose a verilog file in Makerchip Tab</b>",
                QtWidgets.QMessageBox.StandardButton.Ok)
            if reply == QtWidgets.QMessageBox.StandardButton.Ok:
                self.obj_Appconfig.print_error(
                    'No Verilog File Chosen. '
                    'Please choose a verilog file in Makerchip Tab'
                )
                return

        self.fname = Maker.verilogFile[self.filecount]
        currentTermLogs = self.entry_var[0]
        model = ModelGeneration.ModelGeneration(self.fname, currentTermLogs)
        file = (os.path.basename(self.fname)).split('.')[0]
        # If this name was previously built via d_cosim, ASK, then remove that
        # version first so the switch to the legacy NgVeri flow is clean. A
        # declined switch aborts the build.
        if not self._switch_backends_if_needed("ngveri", file):
            return
        if self.entry_var[1].findText(file) == -1:
            self.entry_var[1].addItem(file)

        if not Maker.makerchipTOSAccepted(True):
            QtWidgets.QMessageBox.warning(
                None, "Warning Message",
                "Please accept the Makerchip Terms of Service "
                "to proceed further.",
                QtWidgets.QMessageBox.StandardButton.Ok
            )

            return

        try:
            model.verilogfile()
            error = model.verilogParse()
            if error != "Error":
                model.getPortInfo()
                model.cfuncmod()
                model.ifspecwrite()
                model.sim_main_header()
                model.sim_main()
                model.modpathlst()
                # Each build step now returns True only when its process
                # exits cleanly with code 0. Short-circuit so we stop at the
                # first failing step, and base the verdict on those real exit
                # codes instead of a fragile "is the word 'error' somewhere in
                # the terminal text" search (which both passed broken models
                # and failed working ones whose log merely mentioned "error").
                ok = (
                    model.run_verilator()
                    and model.make_verilator()
                    and model.copy_verilator()
                    and model.runMake()
                )

                if ok:
                    if os.name != 'nt':
                        ok = model.runMakeInstall()
                    else:
                        try:
                            shutil.copy(
                                self.release_dir +
                                "/src/xspice/icm/Ngveri/Ngveri.cm",
                                self.nghdl_home + "/lib/ngspice/"
                            )
                        except FileNotFoundError as err:
                            ok = False
                            currentTermLogs.append(
                                "Error in copying Ngveri code model: " +
                                str(err)
                            )

                if ok:
                    currentTermLogs.append('''
                        <p style=\" font-size:16pt; font-weight:1000;
                        color:#00FF00;\"> Model Created Successfully!
                        </p>
                    ''')
                    placedName = os.path.basename(
                        self.fname).split('.')[0].lower()
                    currentTermLogs.append(
                        '<p style="color:#00AA00; font-weight:600;">'
                        'Model "' + placedName + '" — place it from the '
                        'eSim_Ngveri library in KiCad.</p>')
                else:
                    currentTermLogs.append('''
                        <p style=\" font-size:16pt; font-weight:1000;
                        color:#FF0000;\">There was an error during model
                        creation,<br/>Please rectify the error and try again!
                        </p>
                    ''')

        except Exception as err:
            currentTermLogs.append(
                "Error in Ngspice code model generation " +
                "from Verilog: " + str(err)
            )
            currentTermLogs.append('''
                <p style=\" font-size:16pt; font-weight:1000;
                color:#FF0000;\">There was an error during model creation,
                <br/>Please rectify the error and try again!
                </p>
            ''')

        # Logs were appended live directly to self.entry_var[0]

        # Force scroll the terminal widget at bottom
        self.entry_var[0].verticalScrollBar().setValue(
            self.entry_var[0].verticalScrollBar().maximum()
        )

    def addverilog_cosim(self):
        '''
            d_cosim (Icarus Verilog) flow. Compiles the chosen Verilog file to a
            vvp via iverilog and creates an "NgVeriCosim" KiCad symbol. Unlike
            "Convert Verilog to Ngspice" (legacy static Ngveri.cm), this needs no
            C/C++ compiler, never rebuilds ngspice, and runs fully locally (no
            Makerchip). Gated on the d_cosim toolchain being present.
        '''
        if not CosimConfig.has_iverilog():
            QtWidgets.QMessageBox.warning(
                None, "d_cosim unavailable",
                "<b>" + (CosimConfig.missing_reason() or
                         "Icarus Verilog (with libvvp) not found.") + "</b>",
                QtWidgets.QMessageBox.StandardButton.Ok)
            return

        if len(Maker.verilogFile) < (self.filecount + 1) or \
                Maker.verilogFile[self.filecount] == "":
            QtWidgets.QMessageBox.critical(
                None, "Error Message",
                "<b>Error: No Verilog File Chosen. Please choose a "
                "verilog file in Makerchip Tab</b>",
                QtWidgets.QMessageBox.StandardButton.Ok)
            return

        self.fname = Maker.verilogFile[self.filecount]
        currentTermLogs = self.entry_var[0]
        model = ModelGeneration.ModelGeneration(self.fname, currentTermLogs)
        # Canonical model identity = lowercased basename. The symbol, param
        # XML, compiled vvp, picker entry and netlist all key off THIS one
        # name, so the build location matches the netlister's lookup (a
        # case-sensitive filesystem otherwise loses the vvp at sim time).
        modelname = (os.path.basename(self.fname)).split('.')[0].lower()

        try:
            # If this name was previously built via the legacy NgVeri flow,
            # ASK, then remove that version FIRST -- before verilogfile()
            # repopulates the shared <model>/ dir with the new source. Doing it
            # after would rmtree the freshly-copied .v out from under
            # build_cosim. A declined switch aborts the build.
            if not self._switch_backends_if_needed("cosim", modelname):
                return
            model.verilogfile()
            if model.verilogParse(make_symbol=False) == "Error":
                return
            sim_lib = model.build_cosim(engine="icarus")
            if sim_lib == "Error":
                # build_cosim already logged the phased failure + fix hint.
                pass
            else:
                model.clog.phase("Create KiCad symbol")
                schematicLib = createkicadCosim.CosimSchematic()
                schematicLib.init(modelname, model.modelpath, "icarus", sim_lib)
                if schematicLib.createKicadSymbol() != "Error":
                    # Register in the remove-picker ONLY after the model really
                    # exists, tagged so its deletion runs the d_cosim teardown.
                    combo = self.entry_var[1]
                    if combo.findText(modelname) == -1:
                        combo.addItem(modelname)
                        combo.setItemData(
                            combo.count() - 1, "cosim",
                            QtCore.Qt.ItemDataRole.UserRole)
                    model.clog.ok(
                        'd_cosim model "' + modelname + '" created (Icarus). '
                        'Place it from the eSim_NgVeriCosim library.')
                else:
                    model.clog.error(
                        'KiCad symbol generation failed for "' + modelname +
                        '". The vvp built but the symbol was not created.')
        except BaseException as err:
            model.clog.error("Error in d_cosim model creation: " + str(err))

        # Logs were appended live directly to self.entry_var[0]
        self.entry_var[0].verticalScrollBar().setValue(
            self.entry_var[0].verticalScrollBar().maximum()
        )

    def addfile(self):
        '''
            This function is used to add additional files required
            by the verilog top module
        '''
        if len(Maker.verilogFile) < (self.filecount + 1):
            reply = QtWidgets.QMessageBox.critical(
                None,
                "Error Message",
                "<b>Error: No Verilog File Chosen. \
                Please choose a verilog file in Makerchip Tab</b>",
                QtWidgets.QMessageBox.StandardButton.Ok)
            if reply == QtWidgets.QMessageBox.StandardButton.Ok:
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
            reply = QtWidgets.QMessageBox.critical(
                None,
                "Error Message",
                "<b>Error: No Verilog File Chosen. \
                Please choose a verilog file in Makerchip Tab</b>",
                QtWidgets.QMessageBox.StandardButton.Ok)
            if reply == QtWidgets.QMessageBox.StandardButton.Ok:
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
        self.optionsgrid = QtWidgets.QGridLayout()

        self.optionsgroupbtn = QtWidgets.QButtonGroup()

        self.addverilogbutton = QtWidgets.QPushButton(
            "Convert Verilog to Ngspice")
        self.addverilogbutton.setToolTip(
            "Requires internet connection for converting TL-Verilog models"
        )
        self.addverilogbutton.setToolTipDuration(5000)
        self.optionsgroupbtn.addButton(self.addverilogbutton)
        self.addverilogbutton.clicked.connect(self.addverilog)
        self.optionsgrid.addWidget(self.addverilogbutton, 0, 1)
        # self.optionsbox.setLayout(self.optionsgrid)
        # self.grid.addWidget(self.creategroup(), 1, 0, 5, 0)

        self.addfilebutton = QtWidgets.QPushButton("Add dependency files")
        self.optionsgroupbtn.addButton(self.addfilebutton)
        self.addfilebutton.clicked.connect(self.addfile)
        self.optionsgrid.addWidget(self.addfilebutton, 0, 2)
        # self.optionsbox.setLayout(self.optionsgrid)
        # self.grid.addWidget(self.creategroup(), 1, 0, 5, 0)

        self.addfolderbutton = QtWidgets.QPushButton("Add dependency folder")
        self.optionsgroupbtn.addButton(self.addfolderbutton)
        self.addfolderbutton.clicked.connect(self.addfolder)
        self.optionsgrid.addWidget(self.addfolderbutton, 0, 3)
        # self.optionsbox.setLayout(self.optionsgrid)
        # self.grid.addWidget(self.creategroup(), 1, 0, 5, 0)

        self.clearTerminalBtn = QtWidgets.QPushButton("Clear Terminal")
        self.optionsgroupbtn.addButton(self.clearTerminalBtn)
        self.clearTerminalBtn.clicked.connect(self.clearTerminal)
        self.optionsgrid.addWidget(self.clearTerminalBtn, 0, 4)

        self.addcosimbutton = QtWidgets.QPushButton(
            "Convert Verilog to Ngspice (d_cosim / Icarus)")
        self.addcosimbutton.setToolTip(
            "Icarus Verilog co-simulation via ngspice d_cosim: "
            "no C/C++ compiler and no ngspice rebuild")
        self.optionsgroupbtn.addButton(self.addcosimbutton)
        self.addcosimbutton.clicked.connect(self.addverilog_cosim)
        self.optionsgrid.addWidget(self.addcosimbutton, 1, 1, 1, 4)

        self.optionsbox.setLayout(self.optionsgrid)
        # self.grid.addWidget(self.creategroup(), 1, 0, 5, 0)

        return self.optionsbox

    def edit_modlst(self, text):
        '''
            Remove-model handler for the picker combo. Dispatches to the right
            backend teardown -- legacy NgVeri (Verilator -> Ngveri.cm) vs
            d_cosim (Icarus -> eSim_NgVeriCosim) -- since the two share nothing
            on disk and the wrong teardown silently leaves a model behind.
        '''
        if text == "Remove Verilog Models":
            return
        # A blank/whitespace selection must never reach the teardown helpers:
        # os.path.join(base, "") collapses to "base/" and rmtree would wipe the
        # entire models directory. (Defence in depth -- the helpers guard too.)
        if not text or not str(text).strip():
            return
        combo = self.entry_var[1]
        index = combo.findText(text)
        # Which backend owns this model: the on-disk modelParamXML layout is
        # the source of truth (it survives a backend switch, after which a
        # stale per-item UserRole tag would point at the old engine and
        # silently leave the model behind). The tag is only a fallback for a
        # name not yet on disk.
        backend = self._model_backend(text) \
            or combo.itemData(index, QtCore.Qt.ItemDataRole.UserRole)

        ret = QtWidgets.QMessageBox.warning(
            None, "Warning",
            '''<b>Do you want to remove the model: ''' + text,
            QtWidgets.QMessageBox.StandardButton.Ok,
            QtWidgets.QMessageBox.StandardButton.Cancel)

        # Reset the picker back to the sentinel with signals blocked. Mutating
        # the combo re-fires currentTextChanged synchronously, which would
        # re-enter this slot for the item the selection lands on -> a confirm
        # popup for EVERY model. Blocking signals means only the user's pick is
        # ever handled, and a cancelled remove leaves the list untouched.
        combo.blockSignals(True)
        if ret == QtWidgets.QMessageBox.StandardButton.Ok:
            combo.removeItem(index)
        combo.setCurrentIndex(0)
        combo.blockSignals(False)

        if ret != QtWidgets.QMessageBox.StandardButton.Ok:
            return

        if backend == "cosim":
            self._remove_cosim_model(text)
        else:
            self._remove_ngveri_model(text)

    def _model_backend(self, name):
        '''
            Resolve which backend created a model from the on-disk
            modelParamXML layout -- the single source of truth that survives
            restarts: NgVeriCosim/<name>.xml => d_cosim, else legacy NgVeri.
        '''
        cosim_xml = os.path.join(self._xml_loc, 'NgVeriCosim', name + '.xml')
        return "cosim" if os.path.isfile(cosim_xml) else "ngveri"

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
        '''
        path = self.digital_home + '/modpath.lst'
        try:
            with open(path) as f:
                lines = f.readlines()
        except OSError:
            return False
        kept = [ln for ln in lines if ln.strip() != str(text)]
        if len(kept) == len(lines):
            return False
        with open(path, 'w') as f:
            f.writelines(kept)
        if log:
            log.info('Dropped "' + str(text) + '" from modpath.lst')
        return True

    def _remove_ngveri_model(self, text):
        '''
            Tear down a legacy NgVeri (Verilator) model: drop it from
            modpath.lst, remove its eSim_Ngveri symbol + param XML, delete the
            per-model build dir, then rebuild/reinstall Ngveri.cm so ngspice
            truly unlinks it.
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

        self.fname = Maker.verilogFile[self.filecount]
        model = ModelGeneration.ModelGeneration(
            self.fname, self.entry_var[0])
        # Sweep any other ghosts before rebuilding, so this removal can't fail
        # on an unrelated dead entry.
        model.prune_modpathlst()

        try:
            log.phase("Rebuild Ngveri.cm")
            ok = model.runMake()
            if os.name != 'nt':
                ok = model.runMakeInstall() and ok
            else:
                shutil.copy(
                    self.release_dir + "/src/xspice/icm/Ngveri/Ngveri.cm",
                    self.nghdl_home + "/lib/ngspice/"
                )
            if not ok:
                raise RuntimeError(
                    "the ngspice code-model rebuild returned a "
                    "non-zero exit status")
        except Exception as err:
            QtWidgets.QMessageBox.critical(
                None, "Error Message",
                "The verilog model '" + str(text) +
                "' could not be removed: " + str(err),
                QtWidgets.QMessageBox.StandardButton.Ok
            )
        else:
            log.ok('NgVeri model "' + str(text) + '" removed.')

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

    def _drop_picker_item(self, name):
        '''Remove `name` from the remove-model combo, signal-safe.'''
        combo = self.entry_var[1]
        idx = combo.findText(str(name))
        if idx != -1:
            combo.blockSignals(True)
            combo.removeItem(idx)
            combo.blockSignals(False)

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
        ret = QtWidgets.QMessageBox.question(
            None, "Switch model backend?",
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
            self._drop_picker_item(low)
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
            self._drop_picker_item(reg)
            return True
        return True

    def lint_off_edit(self, text):
        '''
          This is to remove lint_off comments needed by the verilator warnings.
          This function writes to the lint_off.txt in the library/tlv folder.
        '''
        init_path = '../../'
        if os.name == 'nt':
            init_path = ''

        if text == "Remove lint_off":
            return
        # Same re-entrancy guard as edit_modlst: removeItem/setCurrentIndex
        # re-fire currentTextChanged synchronously, which without blockSignals
        # pops a confirm dialog for every remaining entry, not the picked one.
        combo = self.entry_var[2]
        combo.blockSignals(True)
        combo.removeItem(combo.findText(text))
        combo.setCurrentIndex(0)
        combo.blockSignals(False)
        ret = QtWidgets.QMessageBox.warning(
            None,
            "Warning",
            '''<b>Do you want to remove the lint off error: ''' +
            text,
            QtWidgets.QMessageBox.StandardButton.Ok,
            QtWidgets.QMessageBox.StandardButton.Cancel)

        if ret == QtWidgets.QMessageBox.StandardButton.Ok:
            try: 
                file_path = os.path.join(init_path, "library/tlv/lint_off.txt")
                with open(file_path, 'r') as file:
                    data = file.readlines()
                data = [line for line in data if line.strip() != text]
                with open(file_path, 'w') as file:
                    file.writelines(data)
                    
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    None,
                    "Warning",
                    f"Could not remove lint_off entry '{text}'",
                    QtWidgets.QMessageBox.StandardButton.Ok
                )

    def add_lint_off(self):
        '''
            This is to add lint_off comments needed by the verilator warnings.
            This function writes to the lint_off.txt in the library/tlv folder.
        '''
        init_path = '../../'
        if os.name == 'nt':
            init_path = ''

        text = self.entry_var[3].text()

        if self.entry_var[2].findText(text) == -1:
            self.entry_var[2].addItem(text)
            with open(init_path + "library/tlv/lint_off.txt", 'a+') as fh:
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
        self.entry_var[self.count] = QtWidgets.QTextEdit()
        self.entry_var[self.count].setObjectName("simulationConsole")
        self.entry_var[self.count].setReadOnly(1)
        self.trgrid.addWidget(self.entry_var[self.count], 1, 1, 5, 3)
        self.entry_var[self.count].setMaximumWidth(1000)
        self.entry_var[self.count].setMaximumHeight(1000)
        self.count += 1

        self.entry_var[self.count] = QtWidgets.QComboBox()
        self.entry_var[self.count].addItem("Remove Verilog Models")
        modpath_file = self.digital_home + '/modpath.lst'
        if not os.path.exists(modpath_file):
            os.makedirs(self.digital_home, exist_ok=True)
            open(modpath_file, 'w').close()
        with open(modpath_file, 'r') as fh:
            self.data = fh.readlines()
        combo = self.entry_var[self.count]
        for item in self.data:
            if item != "\n":
                combo.addItem(item.strip())
                combo.setItemData(combo.count() - 1, "ngveri",
                                  QtCore.Qt.ItemDataRole.UserRole)
        # Also list d_cosim (Icarus) models so they can be removed too. They
        # live only under modelParamXML/NgVeriCosim (never modpath.lst), so
        # loading from there is what makes them deletable across restarts.
        cosim_dir = os.path.join(self._xml_loc, 'NgVeriCosim')
        if os.path.isdir(cosim_dir):
            for fname in sorted(os.listdir(cosim_dir)):
                if fname.endswith('.xml'):
                    name = fname[:-4]
                    if combo.findText(name) == -1:
                        combo.addItem(name)
                        combo.setItemData(combo.count() - 1, "cosim",
                                          QtCore.Qt.ItemDataRole.UserRole)
        combo.currentTextChanged.connect(self.edit_modlst)
        self.trgrid.addWidget(combo, 1, 4, 1, 2)
        self.count += 1
        self.entry_var[self.count] = QtWidgets.QComboBox()
        self.entry_var[self.count].addItem("Remove lint_off")

        init_path = '../../'
        if os.name == 'nt':
            init_path = ''
        with open(init_path + "library/tlv/lint_off.txt", 'r') as fh:
            self.data = fh.readlines()
        for item in self.data:
            if item != "\n":
                self.entry_var[self.count].addItem(item.strip())
        self.entry_var[self.count].currentTextChanged.connect(self.lint_off_edit)
        self.trgrid.addWidget(self.entry_var[self.count], 2, 4, 1, 2)
        self.count += 1
        self.entry_var[self.count] = QtWidgets.QLineEdit(self)
        self.trgrid.addWidget(self.entry_var[self.count], 3, 4)
        self.entry_var[self.count].setMaximumWidth(100)
        self.count += 1
        self.entry_var[self.count] = QtWidgets.QPushButton("Add lint_off")
        self.entry_var[self.count].setMaximumWidth(100)
        self.trgrid.addWidget(self.entry_var[self.count], 3, 5)
        self.entry_var[self.count].clicked.connect(self.add_lint_off)

        self.count += 1

        # Routed through QSS via cssClass="themedGroupBox" rather than a
        # hardcoded `border: 1px solid gray` stylesheet, so the border picks
        # up the active theme automatically.
        self.trbox.setProperty('cssClass', 'themedGroupBox')

        return self.trbox
