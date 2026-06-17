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

        if os.name == 'nt':
            self.home = os.path.join('library', 'config')
        else:
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
        self.uploadbtn = QtWidgets.QPushButton('Upload')
        self.uploadbtn.clicked.connect(self.uploadModel)
        self.uploadbtn.setStyleSheet(
            "background-color: #2e7d32; color: white; font-weight: bold;")
        self.exitbtn = QtWidgets.QPushButton('Exit')
        self.exitbtn.clicked.connect(self.closeWindow)
        self.browsebtn = QtWidgets.QPushButton('Browse')
        self.browsebtn.clicked.connect(self.browseFile)
        self.addbtn = QtWidgets.QPushButton('Add Files')
        self.addbtn.clicked.connect(self.addFiles)
        self.removebtn = QtWidgets.QPushButton('Remove Files')
        self.removebtn.clicked.connect(self.removeFiles)
        self.ledit = QtWidgets.QLineEdit(self)
        self.ledit.setPlaceholderText("Path to .vhdl file")
        self.sedit = QtWidgets.QTextEdit(self)
        self.process = QtCore.QProcess(self)
        self.termedit = QtWidgets.QTextEdit(self)
        self.termedit.setReadOnly(1)
        pal = QtGui.QPalette()
        bgc = QtGui.QColor(0, 0, 0)
        pal.setColor(QtGui.QPalette.ColorRole.Base, bgc)
        self.termedit.setPalette(pal)
        self.termedit.setStyleSheet("QTextEdit {color:white}")

        # Option buttons grouped like the Maker tab's "Select Options" box.
        optionsbox = QtWidgets.QGroupBox("Select Options")
        optionsgrid = QtWidgets.QGridLayout()
        optionsgrid.setSpacing(5)
        optionsgrid.addWidget(self.ledit, 0, 0, 1, 3)
        optionsgrid.addWidget(self.browsebtn, 0, 3)
        optionsgrid.addWidget(self.addbtn, 1, 0)
        optionsgrid.addWidget(self.removebtn, 1, 1)
        optionsgrid.addWidget(self.uploadbtn, 1, 2)
        # A tab has no business exiting the whole application, so the Exit
        # button is only shown in the standalone window.
        if not self.embedded:
            optionsgrid.addWidget(self.exitbtn, 1, 3)
        optionsbox.setLayout(optionsgrid)

        filesbox = QtWidgets.QGroupBox("Supporting files")
        fileslayout = QtWidgets.QVBoxLayout()
        fileslayout.addWidget(self.sedit)
        filesbox.setLayout(fileslayout)

        consolebox = QtWidgets.QGroupBox("Console")
        consolelayout = QtWidgets.QVBoxLayout()
        consolelayout.addWidget(self.termedit)
        consolebox.setLayout(consolelayout)

        grid = QtWidgets.QVBoxLayout()
        grid.setSpacing(5)
        grid.addWidget(optionsbox)
        grid.addWidget(filesbox)
        grid.addWidget(consolebox)
        self.setLayout(grid)

        # Rounded group-box borders matching the other Makerchip dock tabs.
        self.setStyleSheet(
            "QGroupBox { border: 1px solid gray; border-radius: 9px; "
            "margin-top: 0.5em; } "
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; "
            "padding: 0 3px 0 3px; }")

        # Standalone window chrome; skipped when embedded as a tab.
        if not self.embedded:
            self.setGeometry(300, 300, 600, 600)
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
        title = self.addbtn.text()
        for file in QtWidgets.QFileDialog.getOpenFileNames(self, title)[0]:
            self.sedit.append(str(file))
            self.file_list.append(file)
        print("Supporting Files are :", self.file_list)

    def removeFiles(self):
        self.fileRemover = FileRemover(self)

    # Check extensions of all supporting files
    def checkSupportFiles(self):
        nonvhdl_count = 0
        for file in self.file_list:
            extension = os.path.splitext(str(file))[1]
            if extension != ".vhdl":
                nonvhdl_count += 1
                self.file_list.remove(file)

        if nonvhdl_count > 0:
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
                shutil.rmtree(model_path, ignore_errors=True)
                os.mkdir(model_path)
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

            if os.name == 'nt':
                shutil.copy(os.path.join(self.home, self.src_home) +
                            "/src/ghdlserver/libws2_32.a", path + "/DUTghdl/")

            for file in self.file_list:
                shutil.copy(str(file), path + "/DUTghdl/")

            os.chdir(path + "/DUTghdl")
            if os.name == 'nt':
                # path to msys bin directory where bash is located
                self.msys_home = self.parser.get('COMPILER', 'MSYS_HOME')
                subprocess.call(self.msys_home + "/usr/bin/bash.exe " +
                                path + "/DUTghdl/compile.sh", shell=True)
                subprocess.call(self.msys_home + "/usr/bin/bash.exe -c " +
                                "'chmod a+x start_server.sh'", shell=True)
                subprocess.call(self.msys_home + "/usr/bin/bash.exe -c " +
                                "'chmod a+x sock_pkg_create.sh'", shell=True)
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

    def runMake(self):
        print("run Make Called")
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
            self.process.readyReadStandardOutput.connect(self.readAllStandard)
            self.process.readyReadStandardError.connect(self.readAllStandard)
            if os.name == "nt":
                self.process.finished.connect(self.createSchematicLib)
            else:
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
            else:
                prog = "make"
                args = ["install"]
            print("Running Make Install")

            self.process = QtCore.QProcess(self)
            self.process.setWorkingDirectory(self.path_icm)
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
        if os.name == "nt":
            # This copy uses paths relative to the icm build dir; run it there
            # without leaving eSim's CWD changed.
            _cwd = os.getcwd()
            try:
                os.chdir(self.path_icm)
                shutil.copy("ghdl/ghdl.cm", "../../../../lib/ngspice/")
            finally:
                os.chdir(_cwd)

        os.chdir(self.cur_dir)
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


class FileRemover(QtWidgets.QWidget):

    def __init__(self, main_obj):
        super(FileRemover, self).__init__()
        self.row = 0
        self.col = 0
        self.cb_dict = {}
        self.marked_list = []
        self.files = main_obj.file_list
        self.sedit = main_obj.sedit

        print(self.files)

        self.grid = QtWidgets.QGridLayout()
        removebtn = QtWidgets.QPushButton('Remove', self)
        removebtn.clicked.connect(self.removeFiles)

        self.grid.addWidget(self.createCheckBox(), 0, 0)
        self.grid.addWidget(removebtn, 1, 1)

        self.setLayout(self.grid)
        self.show()

    def createCheckBox(self):
        self.checkbox = QtWidgets.QGroupBox()
        self.checkbox.setTitle('Remove Files')
        self.checkgrid = QtWidgets.QGridLayout()

        self.checkgroupbtn = QtWidgets.QButtonGroup()

        for path in self.files:
            print(path)
            self.cb_dict[path] = QtWidgets.QCheckBox(path)
            self.checkgroupbtn.addButton(self.cb_dict[path])
            self.checkgrid.addWidget(self.cb_dict[path], self.row, self.col)
            self.row += 1

        self.checkgroupbtn.setExclusive(False)
        self.checkgroupbtn.buttonClicked.connect(self.mark_file)
        self.checkbox.setLayout(self.checkgrid)

        return self.checkbox

    def mark_file(self):
        for path in self.cb_dict:
            if self.cb_dict[path].isChecked():
                if path not in self.marked_list:
                    self.marked_list.append(path)
            else:
                if path in self.marked_list:
                    self.marked_list.remove(path)

    def removeFiles(self):
        for path in self.marked_list:
            print(path + " is removed")
            self.sedit.append(path + " removed")
            self.files.remove(path)

        self.sedit.clear()
        for path in self.files:
            self.sedit.append(path)

        self.marked_list[:] = []
        self.files[:] = []
        self.close()


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
