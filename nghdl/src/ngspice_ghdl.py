#!/usr/bin/env python3

# This file create the GUI to install code model in the Ngspice.

import os
import sys
import shutil
import subprocess
from PyQt6 import QtGui, QtCore, QtWidgets
from configparser import ConfigParser
from Appconfig import Appconfig
from createKicadLibrary import AutoSchematic
from model_generation import ModelGeneration
# Vendored, byte-identical copies of eSim's helpers (drift-guarded by tests).
# NGHDL is packaged separately and must never import eSim, so the model
# removal here uses these local copies, exactly as the symbol writer does.
import kicad_symlib
import model_teardown
from RemoveItemsDialog import RemoveItemsDialog


class Mainwindow(QtWidgets.QWidget):

    def __init__(self, parent=None, embedded=False):
        # NOTE: this class is a QWidget. Initialise it as one (the previous
        # QtWidgets.QMainWindow.__init__ call worked only by accident).
        super().__init__(parent)
        # embedded=True  -> running as a tab inside eSim's Makerchip dock.
        # embedded=False -> running standalone via the `nghdl`/`nghdl -e` CLI.
        self.embedded = embedded
        # When embedded, behave like `nghdl -e` so the KiCad schematic symbol
        # is generated after a successful build.
        if embedded:
            Appconfig.esimFlag = 1
        # Remember the host (eSim) working directory. The upload flow changes
        # the CWD; this lets us always restore it and never strand eSim.
        self._home_cwd = os.getcwd()
        print("Initializing..........")

        # Per-user config lives at ~/.nghdl/config.ini on EVERY platform (the
        # Windows bootstrap writes it there too). The old nt branch read a
        # CWD-relative 'library/config/.nghdl/config.ini', which only worked
        # when eSim happened to be launched from its install root.
        self.home = os.path.expanduser('~')

        # Reading all variables from config.ini. A missing/empty config raises
        # here; when embedded the caller catches it and shows a placeholder
        # instead of letting NGHDL break the Makerchip dock.
        self.parser = ConfigParser()
        self.parser.read(
            os.path.join(self.home, os.path.join('.nghdl', 'config.ini'))
        )
        self.nghdl_home = self.parser.get('NGHDL', 'NGHDL_HOME')
        self.release_dir = self.parser.get('NGHDL', 'RELEASE')
        self.src_home = self.parser.get('SRC', 'SRC_HOME')
        self.licensefile = self.parser.get('SRC', 'LICENSE')
        # Printing LICENCE file on terminal (non-fatal if it is missing)
        try:
            with open(self.licensefile, 'r') as fileopen:
                print(fileopen.read())
        except OSError as e:
            print("Could not read NGHDL license file:", e)
        self.file_list = []       # to keep the supporting files
        self.filename = ''
        self.errorFlag = False    # to keep the check of "make install" errors
        self.initUI()

    def initUI(self):
        # The page is one linear task read top to bottom: (1) pick the VHDL
        # model, (2) optionally stage supporting .vhdl files, (3) Upload &
        # Build. The console shows the build output and owns the vertical
        # space. Model *uninstall* is maintenance, not part of that flow, so it
        # lives in a quiet footer with a distinct verb -- never mistaken for
        # the list's "Remove".

        # ---- Step 1: the top-level VHDL model file --------------------------
        self.ledit = QtWidgets.QLineEdit(self)
        self.ledit.setPlaceholderText("Select the .vhdl model file to build…")
        self.ledit.setClearButtonEnabled(True)
        self.browsebtn = QtWidgets.QPushButton('Browse…')
        self.browsebtn.setToolTip("Choose the top-level .vhdl file for your model.")
        self.browsebtn.clicked.connect(self.browseFile)

        modelbox = QtWidgets.QGroupBox("1 · VHDL model")
        modelrow = QtWidgets.QHBoxLayout()
        modelrow.setSpacing(8)
        modelrow.addWidget(self.ledit, 1)
        modelrow.addWidget(self.browsebtn)
        modelbox.setLayout(modelrow)

        # ---- Step 2: optional supporting (dependency) files ----------------
        # A compact list, not a full text editor -- these are just a handful of
        # file paths. Add/Remove are scoped right beside the list, so "Remove"
        # unambiguously means "drop from this list".
        self.filelist = QtWidgets.QListWidget(self)
        self.filelist.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.filelist.setMaximumHeight(130)
        self.filelist.setToolTip(
            "Dependency .vhdl files compiled alongside the model above.")
        self.filelist.itemSelectionChanged.connect(self._sync_file_buttons)

        self.addbtn = QtWidgets.QPushButton('Add…')
        self.addbtn.setToolTip(
            "Add dependency .vhdl file(s) to compile with the model.")
        self.addbtn.clicked.connect(self.addFiles)
        self.removebtn = QtWidgets.QPushButton('Remove')
        self.removebtn.setToolTip(
            "Remove the selected file(s) from this list. Does NOT uninstall a "
            "built model — use “Uninstall Models” above.")
        self.removebtn.setEnabled(False)
        self.removebtn.clicked.connect(self.removeFiles)

        fileshint = QtWidgets.QLabel(
            "Optional — extra .vhdl files your model depends on.")
        fileshint.setObjectName("hint")

        supportbox = QtWidgets.QGroupBox("2 · Supporting files")
        supportv = QtWidgets.QVBoxLayout()
        supportv.setSpacing(6)
        supportv.addWidget(fileshint)
        supportrow = QtWidgets.QHBoxLayout()
        supportrow.setSpacing(8)
        supportrow.addWidget(self.filelist, 1)
        filebtns = QtWidgets.QVBoxLayout()
        filebtns.setSpacing(6)
        filebtns.addWidget(self.addbtn)
        filebtns.addWidget(self.removebtn)
        filebtns.addStretch(1)
        supportrow.addLayout(filebtns)
        supportv.addLayout(supportrow)
        supportbox.setLayout(supportv)

        # ---- The one primary action ----------------------------------------
        self.uploadbtn = QtWidgets.QPushButton('Upload && Build')
        self.uploadbtn.setObjectName("primary")
        self.uploadbtn.setMinimumHeight(34)
        self.uploadbtn.setToolTip(
            "Compile the selected VHDL into an ngspice code model and KiCad "
            "symbol.")
        self.uploadbtn.clicked.connect(self.uploadModel)
        actionrow = QtWidgets.QHBoxLayout()
        actionrow.addStretch(1)
        actionrow.addWidget(self.uploadbtn)

        # ---- Console: build output, owns the vertical stretch --------------
        self.termedit = QtWidgets.QTextEdit(self)
        self.termedit.setReadOnly(True)
        mono = QtGui.QFontDatabase.systemFont(
            QtGui.QFontDatabase.SystemFont.FixedFont)
        mono.setPointSize(10)
        self.termedit.setFont(mono)
        pal = QtGui.QPalette()
        pal.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(17, 17, 17))
        self.termedit.setPalette(pal)
        self.termedit.setStyleSheet(
            "QTextEdit { color: #e6e6e6; background-color: #111111; "
            "border: none; }")
        self.process = QtCore.QProcess(self)

        consolebox = QtWidgets.QGroupBox("Console")
        consolev = QtWidgets.QVBoxLayout()
        consolev.addWidget(self.termedit)
        consolebox.setLayout(consolev)

        # ---- Header: page title + model maintenance (uninstall) ------------
        # Uninstall is maintenance, kept out of the numbered build flow so it is
        # never mistaken for a build step -- but it now sits top-right where it
        # is always visible and fully legible, instead of marooned in a footer
        # below the console where it read as an afterthought.
        pagetitle = QtWidgets.QLabel("Build an NGHDL model from VHDL")
        pagetitle.setObjectName("pagetitle")

        self.removemodelbtn = QtWidgets.QPushButton('Uninstall Models')
        self.removemodelbtn.setObjectName("uninstall")
        self.removemodelbtn.setMinimumHeight(30)
        self.removemodelbtn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.removemodelbtn.setToolTip(
            "Remove one or more NGHDL models already installed on this "
            "machine.")
        self.removemodelbtn.clicked.connect(self.openRemoveModels)

        header = QtWidgets.QHBoxLayout()
        header.addWidget(pagetitle)
        header.addStretch(1)
        header.addWidget(self.removemodelbtn)

        # Exit is created in every mode (the build callbacks toggle it), but a
        # tab has no business exiting all of eSim, so it is only *placed* in
        # the standalone window's footer.
        self.exitbtn = QtWidgets.QPushButton('Exit')
        self.exitbtn.clicked.connect(self.closeWindow)

        # ---- Assemble -------------------------------------------------------
        grid = QtWidgets.QVBoxLayout()
        grid.setSpacing(10)
        grid.addLayout(header)
        grid.addWidget(modelbox)
        grid.addWidget(supportbox)
        grid.addLayout(actionrow)
        grid.addWidget(consolebox, 1)   # console takes the remaining height
        # Standalone keeps a slim Exit footer; embedded there is nothing left
        # to show, so no footer row is added at all.
        if not self.embedded:
            footer = QtWidgets.QHBoxLayout()
            footer.addStretch(1)
            footer.addWidget(self.exitbtn)
            grid.addLayout(footer)
        self.setLayout(grid)

        # Rounded group-boxes matching the other Makerchip dock tabs, plus the
        # primary button's hover/pressed/disabled feedback states.
        self.setStyleSheet(
            "QGroupBox { border: 1px solid gray; border-radius: 9px; "
            "margin-top: 0.6em; } "
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; "
            "padding: 0 4px 0 4px; } "
            "QLabel#hint { font-style: italic; } "
            "QLabel#pagetitle { font-size: 15px; font-weight: 700; } "
            "QPushButton#uninstall { border: 1px solid #c0563a; "
            "color: #c0563a; border-radius: 6px; padding: 6px 16px; "
            "font-weight: 600; } "
            "QPushButton#uninstall:hover { "
            "background-color: rgba(192, 86, 58, 0.12); } "
            "QPushButton#uninstall:pressed { "
            "background-color: rgba(192, 86, 58, 0.22); } "
            "QPushButton#primary { background-color: #2e7d32; color: white; "
            "font-weight: bold; border-radius: 6px; padding: 7px 22px; } "
            "QPushButton#primary:hover { background-color: #33883a; } "
            "QPushButton#primary:pressed { background-color: #24632a; } "
            "QPushButton#primary:disabled { background-color: #b9c9ba; "
            "color: #eef3ee; }")

        # Standalone window chrome; skipped when embedded as a tab.
        if not self.embedded:
            self.setGeometry(300, 300, 640, 620)
            self.setWindowTitle("Ngspice Digital Model Creator (from VHDL)")
            # self.setWindowIcon(QtGui.QIcon('logo.png'))
            self.show()

    def closeWindow(self):
        try:
            self.process.close()
        except BaseException:
            pass
        print("Close button clicked")
        # Never exit the process when embedded - that would kill all of eSim.
        if not self.embedded:
            sys.exit()

    def closeEvent(self, event):
        # Kill any running build so closing the tab/eSim leaves no orphan.
        try:
            if self.process is not None:
                self.process.kill()
        except BaseException:
            pass
        super().closeEvent(event)

    def browseFile(self):
        print("Browse button clicked")
        self.filename = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open File', '.')[0]
        self.ledit.setText(self.filename)
        print("Vhdl file uploaded to process :", self.filename)

    def addFiles(self):
        print("Starts adding supporting files")
        for file in QtWidgets.QFileDialog.getOpenFileNames(
                self, "Add supporting .vhdl files")[0]:
            self.file_list.append(file)
        self._refresh_file_list()
        print("Supporting Files are :", self.file_list)

    def removeFiles(self):
        # Un-stage the selected supporting files from the pre-upload list.
        # (This only edits the list; it never uninstalls a built model.)
        selected_rows = {self.filelist.row(item)
                         for item in self.filelist.selectedItems()}
        if not selected_rows:
            return
        self.file_list = [f for i, f in enumerate(self.file_list)
                          if i not in selected_rows]
        self._refresh_file_list()

    def _refresh_file_list(self):
        """Rebuild the supporting-files list widget from file_list (the source
        of truth). Row order stays 1:1 with file_list so removeFiles can map a
        selected row straight back to an entry."""
        self.filelist.clear()
        for f in self.file_list:
            item = QtWidgets.QListWidgetItem(os.path.basename(str(f)))
            item.setToolTip(str(f))
            self.filelist.addItem(item)
        self._sync_file_buttons()

    def _sync_file_buttons(self):
        """Remove is only meaningful when something in the list is selected."""
        self.removebtn.setEnabled(bool(self.filelist.selectedItems()))

    # ------------------------------------------------------------------ #
    #  NGHDL model uninstall. Mirrors eSim's NgVeri._remove_nghdl_model
    #  against the ghdl/ tree, but uses NGHDL's OWN vendored helpers
    #  (model_teardown + kicad_symlib) so it works standalone, without
    #  importing eSim.
    # ------------------------------------------------------------------ #
    def _ghdl_home(self):
        '''<DIGITAL_MODEL>/ghdl -- where per-model source dirs and the NGHDL
        modpath.lst live (same value createModelDirectory computes).'''
        return os.path.join(
            self.parser.get('NGHDL', 'DIGITAL_MODEL'), "ghdl")

    def _list_nghdl_models(self):
        '''Installed NGHDL model names, from ghdl/modpath.lst -- the
        authoritative, always-present list of what ngspice has linked.'''
        modpath = os.path.join(self._ghdl_home(), "modpath.lst")
        try:
            with open(modpath) as f:
                return sorted({ln.strip() for ln in f if ln.strip()})
        except OSError:
            return []

    def openRemoveModels(self):
        '''Open the checkbox remover listing only NGHDL models and tear down
        whatever the user picks.'''
        proc = getattr(self, "process", None)
        if proc is not None and proc.state() != QtCore.QProcess.ProcessState.NotRunning:
            QtWidgets.QMessageBox.information(
                self, "Remove Models",
                "A build is in progress. Please wait for it to finish.")
            return
        names = self._list_nghdl_models()
        if not names:
            QtWidgets.QMessageBox.information(
                self, "Remove Models", "There are no NGHDL models to remove.")
            return
        # Same searchable, multi-select picker the eSim NgVeri tab uses, so the
        # remove-model UX is uniform across the app. Badge every row "Nghdl" for
        # the shared colour legend; the dialog only collects the choice, the
        # teardown still happens in _remove_nghdl_models.
        dlg = RemoveItemsDialog(
            "Remove NGHDL Models", names,
            badges={name: "Nghdl" for name in names},
            item_noun="model", parent=self)
        if dlg.exec():
            self._remove_nghdl_models(dlg.selected_items())

    def _remove_nghdl_models(self, names):
        '''Tear down each selected NGHDL model -- ghdl/modpath.lst line,
        eSim_Nghdl KiCad symbol, Nghdl/<name>.xml, and the source + release
        per-model dirs -- then rebuild ghdl.cm once so ngspice unlinks them.'''
        ghdl_home = self._ghdl_home()
        xml_loc = Appconfig.xml_loc
        removed_any = False
        for text in names:
            if not text or not str(text).strip():
                continue
            self.termedit.append('Removing NGHDL model "' + str(text) + '"')

            # Drop from ghdl/modpath.lst (absent file/line => no crash).
            if model_teardown._strip_modpath_line(
                    os.path.join(ghdl_home, "modpath.lst"), text):
                self.termedit.append('  dropped from ghdl/modpath.lst')

            # Strip the symbol + the orphan param XML (both idempotent).
            try:
                sym_path = model_teardown._nghdl_sym_path(self.src_home)
                parts = kicad_symlib._read_parts(sym_path)
                if parts.pop(str(text), None) is not None:
                    kicad_symlib._write_lib(sym_path, parts)
                    self.termedit.append('  removed eSim_Nghdl symbol')
                xml = os.path.join(xml_loc, 'Nghdl', str(text) + '.xml')
                try:
                    os.remove(xml)
                    self.termedit.append('  removed Nghdl/' + str(text) +
                                         '.xml')
                except FileNotFoundError:
                    pass
            except Exception as err:
                self.termedit.append("  WARN: symbol/XML removal failed: " +
                                     str(err))

            # Drop both per-model dirs (guarded against blank/unsafe names).
            for label, base in (
                    ("source", ghdl_home),
                    ("release", os.path.join(
                        self.release_dir, "src/xspice/icm/ghdl"))):
                model_dir = model_teardown._safe_model_subdir(base, text)
                if model_dir is None:
                    self.termedit.append("  WARN: refusing unsafe " + label +
                                         " dir for " + repr(text))
                    continue
                try:
                    shutil.rmtree(model_dir)
                    self.termedit.append("  removed " + label + " dir")
                except FileNotFoundError:
                    pass
                except OSError as err:
                    self.termedit.append("  WARN: could not remove " + label +
                                         " dir: " + str(err))
            removed_any = True

        if removed_any:
            self._rebuild_ghdl_cm()

    def _rebuild_ghdl_cm(self):
        '''Rebuild + reinstall ghdl.cm so ngspice unlinks every model already
        stripped from ghdl/modpath.lst. Prune ghost lines first (a single
        orphaned entry makes cmpp abort the whole build), then reuse the async
        make chain in rebuild-only mode (no schematic symbol is created).'''
        ghdl_home = self._ghdl_home()
        dropped = model_teardown._prune_modpath(
            os.path.join(ghdl_home, "modpath.lst"), ghdl_home)
        for name in dropped:
            self.termedit.append('Pruned stale model "' + name +
                                 '" from ghdl/modpath.lst (build dir missing).')
        # runMake/_createSchematicLib reference these; set them so the rebuild
        # path never depends on a prior uploadModel() having run.
        self.cur_dir = os.getcwd()
        self.errorFlag = False
        self.uploadbtn.setEnabled(False)
        self.termedit.append("Rebuilding ghdl.cm ...")
        self.runMake(rebuild_only=True)

    # Check extensions of all supporting files
    def checkSupportFiles(self):
        # Rebuild via comprehension: mutating the list while iterating it (the
        # old approach) skips the element after each removal, so back-to-back
        # non-.vhdl files could slip through.
        vhdl_only = [f for f in self.file_list
                     if os.path.splitext(str(f))[1] == ".vhdl"]
        if len(vhdl_only) != len(self.file_list):
            self.file_list = vhdl_only
            self._refresh_file_list()
            QtWidgets.QMessageBox.critical(
                self, 'Critical', '''<b>Important Message.</b>
                <br/><br/>Supporting files should be <b>.vhdl</b> file '''
            )

    def createModelDirectory(self):
        """Create the model directory. Returns False if the user cancels an
        overwrite (the upload is then aborted without touching eSim)."""
        print("Create Model Directory Called")
        self.digital_home = self.parser.get('NGHDL', 'DIGITAL_MODEL')
        self.digital_home = os.path.join(self.digital_home, "ghdl")
        self.modelname = os.path.basename(str(self.filename)).split('.')[0]
        print("Model to be created :", self.modelname)
        # An empty model name (no file chosen / odd filename) would make
        # model_path == the ghdl icm ROOT below -- and the overwrite branch
        # would rmtree every installed model plus modpath.lst. Refuse it.
        if not self.modelname:
            QtWidgets.QMessageBox.critical(
                self, "Error",
                "No VHDL file selected - cannot derive a model name.")
            return False
        # Work with an absolute path so we never have to chdir (chdir would
        # change eSim's process-global CWD).
        model_path = os.path.join(self.digital_home, self.modelname)
        # Looking if model directory is present or not
        if os.path.isdir(model_path):
            print("Model Already present")
            ret = QtWidgets.QMessageBox.warning(
                self, "Warning",
                "<b>This model already exist. Do you want to " +
                "overwrite it?</b><br/> If yes press ok, else cancel it and " +
                "change the name of your vhdl file.",
                QtWidgets.QMessageBox.StandardButton.Ok | QtWidgets.QMessageBox.StandardButton.Cancel
            )
            if ret == QtWidgets.QMessageBox.StandardButton.Ok:
                print("Overwriting existing model " + self.modelname)
                # A real rmtree (errors surfaced, not swallowed): a partial
                # failure left by ignore_errors=True keeps the dir alive and
                # the following mkdir then crashes with FileExistsError. Abort
                # cleanly instead, and makedirs(exist_ok=True) so a stray race
                # can't re-trigger that crash.
                try:
                    shutil.rmtree(model_path)
                except FileNotFoundError:
                    pass
                except OSError as e:
                    QtWidgets.QMessageBox.critical(
                        self, "Error",
                        "Could not overwrite the existing model: " + str(e))
                    return False
                os.makedirs(model_path, exist_ok=True)
            else:
                print("Model creation cancelled by user")
                return False
        else:
            print("Creating model " + self.modelname + " directory")
            os.mkdir(model_path)
        return True

    def addingModelInModpath(self):
        print("Adding Model " + self.modelname +
              " in Modpath file " + self.digital_home)
        # Adding name of model in the modpath file
        # Check if the string is already in the file
        with open(self.digital_home + "/modpath.lst", 'r+') as f:
            flag = 0
            for line in f:
                if line.strip() == self.modelname:
                    print("Found model "+self.modelname+" in the modpath.lst")
                    flag = 1
                    break

            if flag == 0:
                print("Adding model name "+self.modelname+" into modpath.lst")
                f.write(self.modelname + "\n")
            else:
                print("Model name is already into modpath.lst")

    def createModelFiles(self):
        print("Create Model Files Called")
        # This method must chdir into the model dir for the relative file ops
        # and compile script below. Wrap it so eSim's CWD is always restored,
        # even on error.
        try:
            os.chdir(self.cur_dir)
            print("Current Working directory changed to " + self.cur_dir)

            # Generate model corresponding to the uploaded VHDL file
            model = ModelGeneration(str(self.ledit.text()))
            model.readPortInfo()
            model.createCfuncModFile()
            model.createIfSpecFile()
            model.createTestbench()
            model.createServerScript()
            model.createSockScript()

            # Moving file to model directory
            path = os.path.join(self.digital_home, self.modelname)
            shutil.move("cfunc.mod", path)
            shutil.move("ifspec.ifs", path)

            # Creating directory inside model directoy
            print("Creating DUT directory at " + os.path.join(path, "DUTghdl"))
            os.mkdir(path + "/DUTghdl/")
            print("Copying required file to DUTghdl directory")
            shutil.move("connection_info.txt", path + "/DUTghdl/")
            shutil.move("start_server.sh", path + "/DUTghdl/")
            shutil.move("sock_pkg_create.sh", path + "/DUTghdl/")
            shutil.move(self.modelname + "_tb.vhdl", path + "/DUTghdl/")

            shutil.copy(str(self.filename), path + "/DUTghdl/")
            shutil.copy(os.path.join(self.home, self.src_home) +
                        "/src/ghdlserver/compile.sh", path + "/DUTghdl/")
            shutil.copy(os.path.join(self.home, self.src_home) +
                        "/src/ghdlserver/uthash.h", path + "/DUTghdl/")
            shutil.copy(os.path.join(self.home, self.src_home) +
                        "/src/ghdlserver/ghdlserver.c", path + "/DUTghdl/")
            shutil.copy(os.path.join(self.home, self.src_home) +
                        "/src/ghdlserver/ghdlserver.h", path + "/DUTghdl/")
            shutil.copy(os.path.join(self.home, self.src_home) +
                        "/src/ghdlserver/Utility_Package.vhdl",
                        path + "/DUTghdl/")
            shutil.copy(os.path.join(self.home, self.src_home) +
                        "/src/ghdlserver/Vhpi_Package.vhdl", path + "/DUTghdl/")

            # (Winsock is linked with -lws2_32 by the generated
            # start_server.sh; no bundled libws2_32.a copy is needed.)

            for file in self.file_list:
                shutil.copy(str(file), path + "/DUTghdl/")

            os.chdir(path + "/DUTghdl")
            if os.name == 'nt':
                # Compile ghdlserver.c with the MSYS2 mingw64 gcc. A bare
                # (non-login) bash has no /mingw64/bin on PATH, so export it
                # explicitly; run via cwd + relative script so spaced install
                # paths cannot split the command (this process chdir'd into
                # DUTghdl just above). Arg-list form: nothing to re-quote.
                self.msys_home = self.parser.get('COMPILER', 'MSYS_HOME')
                bash = os.path.join(self.msys_home, 'usr', 'bin', 'bash.exe')
                mingw_env = "export PATH=/mingw64/bin:/usr/bin:$PATH && "
                # CREATE_NO_WINDOW: this GUI process has no console, so each
                # bash.exe would otherwise flash its own blank console window.
                no_window = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
                subprocess.call([bash, '-c', mingw_env + 'sh compile.sh'],
                                creationflags=no_window)
                subprocess.call([bash, '-c', 'chmod a+x start_server.sh'],
                                creationflags=no_window)
                subprocess.call([bash, '-c', 'chmod a+x sock_pkg_create.sh'],
                                creationflags=no_window)
            else:
                subprocess.call("bash " + path + "/DUTghdl/compile.sh",
                                shell=True)
                subprocess.call("chmod a+x start_server.sh", shell=True)
                subprocess.call("chmod a+x sock_pkg_create.sh", shell=True)

            os.remove("compile.sh")
            # os.remove("ghdlserver.c")
        finally:
            os.chdir(self.cur_dir)

    # Slot to redirect stdout and stderr to window console
    @QtCore.pyqtSlot()
    def readAllStandard(self):
        proc = self.sender()
        if not isinstance(proc, QtCore.QProcess):
            return
        self.termedit.append(
            str(proc.readAllStandardOutput().data(), encoding='utf-8')
        )
        stderror = proc.readAllStandardError()
        if stderror.toUpper().contains(QtCore.QByteArray(b"ERROR")):
            self.errorFlag = True
        self.termedit.append(str(stderror.data(), encoding='utf-8'))

    def _apply_build_env(self, process):
        """On Windows, prepend the MSYS2 mingw64/usr dirs to the QProcess
        PATH: mingw32-make itself is started by full path, but its child
        gcc/g++/ar/sh processes resolve via PATH, which eSim's own process
        does not have. No-op on POSIX."""
        if os.name != 'nt':
            return
        msys_home = self.parser.get('COMPILER', 'MSYS_HOME')
        env = QtCore.QProcessEnvironment.systemEnvironment()
        extra = os.pathsep.join([
            os.path.join(msys_home, 'mingw64', 'bin'),
            os.path.join(msys_home, 'usr', 'bin'),
        ])
        existing = env.value('PATH', '')
        env.insert('PATH',
                   extra + (os.pathsep + existing if existing else ''))
        process.setProcessEnvironment(env)

    def runMake(self, rebuild_only=False):
        print("run Make Called")
        # rebuild_only=True is the post-removal path: build+install the icm
        # tree to refresh ghdl.cm, but skip the schematic-symbol creation in
        # _createSchematicLib (there is no new model to draw a symbol for).
        self._rebuild_only = rebuild_only
        self.release_home = self.parser.get('NGHDL', 'RELEASE')
        # Keep the icm path so make/make install run there via QProcess's
        # own working directory - we never chdir eSim's process into it.
        self.path_icm = os.path.join(self.release_home, "src/xspice/icm")

        try:
            if os.name == 'nt':
                self.msys_home = self.parser.get('COMPILER', 'MSYS_HOME')
                cmd = self.msys_home + "/mingw64/bin/mingw32-make.exe"
            else:
                cmd = "make"

            print("Running Make command in " + self.path_icm)
            self.process = QtCore.QProcess(self)
            self.process.setWorkingDirectory(self.path_icm)
            self._apply_build_env(self.process)
            self.process.readyReadStandardOutput.connect(self.readAllStandard)
            self.process.readyReadStandardError.connect(self.readAllStandard)
            # make install runs on every platform: the Windows nghdl tree is
            # configured with prefix=install_dir exactly like Ubuntu, so the
            # rebuilt ghdl.cm lands where ngspice loads code models from
            # (<install_dir>/lib/ngspice) instead of a hand-copied guess.
            self.process.finished.connect(self.runMakeInstall)
            self.process.start(cmd)
            print("make command process pid ---------- >", self.process.processId())

        except BaseException:
            print("There is error in 'make' ")
            if not self.embedded:
                sys.exit()
            self.uploadbtn.setEnabled(True)
            self.exitbtn.setEnabled(True)

    def runMakeInstall(self):
        print("run Make Install Called")
        try:
            if os.name == 'nt':
                self.msys_home = self.parser.get('COMPILER', 'MSYS_HOME')
                prog = self.msys_home + "/mingw64/bin/mingw32-make.exe"
                args = ["install"]
                # The configured tree bakes the BUILD machine's absolute
                # prefix into makedefs (pkglibdir/pkgdatadir), so a stock
                # `make install` on an end-user PC would write the rebuilt
                # ghdl.cm into the packager's path instead of this install's
                # install_dir. Override both on the command line (forward
                # slashes: these are make variables). Same fix as
                # ModelGeneration.runMakeInstall.
                nghdl_home = self.parser.get('NGHDL', 'NGHDL_HOME')
                if nghdl_home:
                    inst = os.path.join(nghdl_home, 'install_dir').replace(
                        '\\', '/')
                    args += ["pkglibdir=" + inst + "/lib/ngspice",
                             "pkgdatadir=" + inst + "/share/ngspice"]
            else:
                prog = "make"
                args = ["install"]
            print("Running Make Install")

            self.process = QtCore.QProcess(self)
            self.process.setWorkingDirectory(self.path_icm)
            self._apply_build_env(self.process)
            self.process.readyReadStandardOutput.connect(self.readAllStandard)
            self.process.readyReadStandardError.connect(self.readAllStandard)
            self.process.finished.connect(self.createSchematicLib)
            self.process.start(prog, args)

        except BaseException:
            print("There is error in 'make install' ")
            if not self.embedded:
                sys.exit()
            self.uploadbtn.setEnabled(True)
            self.exitbtn.setEnabled(True)

    def createSchematicLib(self):
        try:
            self._createSchematicLib()
        except Exception as e:
            print("createSchematicLib exception:", e)
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(
                self, 'Error', 'Library creation failed: ' + str(e)
            )
            self.uploadbtn.setEnabled(True)
            self.exitbtn.setEnabled(True)

    def _createSchematicLib(self):
        # (ghdl.cm is installed by `make install` into
        # <install_dir>/lib/ngspice on every platform now; the old nt-only
        # hand copy into the release tree pointed where ngspice never looks.)
        os.chdir(self.cur_dir)
        # Post-removal rebuild: ghdl.cm is refreshed (copied on nt / installed
        # on posix); there is no model to draw, so skip symbol creation.
        if getattr(self, "_rebuild_only", False):
            self.termedit.append("ghdl.cm rebuilt.")
            self._rebuild_only = False
            self.uploadbtn.setEnabled(True)
            self.exitbtn.setEnabled(True)
            self.removemodelbtn.setEnabled(True)
            return
        if Appconfig.esimFlag == 1:
            if not self.errorFlag:
                print('Creating library files................................')
                schematicLib = AutoSchematic(self, self.modelname)
                schematicLib.createKicadSymbol()
            else:
                QtWidgets.QMessageBox.critical(
                    self, 'Error', '''Cannot create Schematic Library of ''' +
                    '''your model. Resolve the <b>errors</b> shown on ''' +
                    '''console of NGHDL window. '''
                )
        else:
            QtWidgets.QMessageBox.information(
                self, 'Message', '''<b>Important Message</b><br/><br/>''' +
                '''To create Schematic Library of your model, ''' +
                '''use NGHDL through <b>eSim</b> '''
            )
        self.uploadbtn.setEnabled(True)
        self.exitbtn.setEnabled(True)

    def uploadModel(self):
        print("Upload button clicked")
        try:
            self.process.close()
        except BaseException:
            pass
        if not self.filename:
            QtWidgets.QMessageBox.warning(
                self, 'No File', 'Use Browse to select a .vhdl file first.')
            return
        try:
            self.file_extension = os.path.splitext(str(self.filename))[1]
            print("Uploaded File extension :" + self.file_extension)
            self.cur_dir = os.getcwd()
            print("Current Working Directory :" + self.cur_dir)
            self.checkSupportFiles()
            if self.file_extension == ".vhdl":
                self.errorFlag = False
                self.uploadbtn.setEnabled(False)
                self.exitbtn.setEnabled(False)
                self.termedit.append('<b style="color:yellow">Processing... do not close until Symbol Added dialog appears.</b>')
                if not self.createModelDirectory():
                    # User cancelled an overwrite - abort cleanly.
                    self.uploadbtn.setEnabled(True)
                    self.exitbtn.setEnabled(True)
                    return
                self.addingModelInModpath()
                self.createModelFiles()
                self.runMake()
            else:
                QtWidgets.QMessageBox.information(
                    self, 'Message', '''<b>Important Message.</b><br/>''' +
                    '''<br/>This accepts only <b>.vhdl</b> file '''
                )
        except Exception as e:
            # Restore eSim's CWD and re-enable controls so a failed upload
            # never leaves the host application or this tab in a bad state.
            try:
                os.chdir(self.cur_dir)
            except BaseException:
                pass
            self.uploadbtn.setEnabled(True)
            self.exitbtn.setEnabled(True)
            QtWidgets.QMessageBox.critical(self, 'Error', str(e))


def main():
    app = QtWidgets.QApplication(sys.argv)
    if len(sys.argv) > 1:
        if sys.argv[1] == '-e':
            Appconfig.esimFlag = 1

    # Mainwindow() object must be assigned to a variable.
    # Otherwise, it is destroyed as soon as it gets created.
    w = Mainwindow()    # noqa
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
