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
import os
import shutil
from configuration.Appconfig import Appconfig
from configparser import ConfigParser


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
        currentTermLogs = QtWidgets.QTextEdit()
        model = ModelGeneration.ModelGeneration(self.fname, currentTermLogs)
        file = (os.path.basename(self.fname)).split('.')[0]
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

        self.entry_var[0].append(currentTermLogs.toHtml())

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
        currentTermLogs = QtWidgets.QTextEdit()
        model = ModelGeneration.ModelGeneration(self.fname, currentTermLogs)
        file = (os.path.basename(self.fname)).split('.')[0]
        if self.entry_var[1].findText(file) == -1:
            self.entry_var[1].addItem(file)

        try:
            model.verilogfile()
            if model.verilogParse(make_symbol=False) == "Error":
                return
            sim_lib = model.build_cosim(engine="icarus")
            if sim_lib == "Error":
                currentTermLogs.append(
                    '<p style="color:#FF0000; font-weight:600;">'
                    'd_cosim model build failed.</p>')
            else:
                modelname = file.lower()
                schematicLib = createkicadCosim.CosimSchematic()
                schematicLib.init(modelname, model.modelpath, "icarus", sim_lib)
                if schematicLib.createKicadSymbol() != "Error":
                    currentTermLogs.append(
                        '<p style="color:#00AA00; font-weight:600;">'
                        'd_cosim model "' + modelname + '" created (Icarus). '
                        'Place it from the eSim_NgVeriCosim library.</p>')
        except BaseException as err:
            currentTermLogs.append(
                "Error in d_cosim model creation: " + str(err))

        self.entry_var[0].append(currentTermLogs.toHtml())
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
            This is used to remove models in modlst of Ngspice folder if
            the user wants to remove a model. Note: files do not get removed.
        '''
        if text == "Remove Verilog Models":
            return
        index = self.entry_var[1].findText(text)
        self.entry_var[1].removeItem(index)
        self.entry_var[1].setCurrentIndex(0)
        ret = QtWidgets.QMessageBox.warning(
            None, "Warning", '''<b>Do you want to remove the model: ''' +
            text,
            QtWidgets.QMessageBox.StandardButton.Ok, QtWidgets.QMessageBox.StandardButton.Cancel
        )
        if ret == QtWidgets.QMessageBox.StandardButton.Ok:
            mod = open(self.digital_home + '/modpath.lst', 'r')
            data = mod.readlines()
            mod.close()

            # Drop the model from modpath.lst (guarded: absent => no crash)
            if (text + "\n") in data:
                data.remove(text + "\n")
            mod = open(self.digital_home + '/modpath.lst', 'w')
            for item in data:
                mod.write(item)
            mod.close()

            # Remove the KiCad symbol + orphan param XML too, so the model
            # actually disappears from eSim_Ngveri in KiCad (previously left
            # behind forever).
            try:
                symbol = createkicad.AutoSchematic()
                symbol.init(text, "")
                symbol.deleteKicadSymbol()
            except Exception as err:
                print("Could not remove KiCad symbol for '" +
                      str(text) + "': " + str(err))

            # Drop the compiled per-model build dir under the release tree, so
            # the rebuild below truly unlinks the model. Without this its stale
            # .o/.a got re-bundled and the model kept answering in ngspice even
            # though the picker entry, symbol and modpath.lst line were gone.
            model_dir = os.path.join(
                self.release_dir, "src/xspice/icm/Ngveri", text)
            try:
                shutil.rmtree(model_dir)
            except FileNotFoundError:
                pass
            except OSError as err:
                print("Could not remove build dir '" +
                      model_dir + "': " + str(err))

            self.fname = Maker.verilogFile[self.filecount]
            model = ModelGeneration.ModelGeneration(
                self.fname, self.entry_var[0])

            try:
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
        index = self.entry_var[2].findText(text)
        self.entry_var[2].removeItem(index)
        self.entry_var[2].setCurrentIndex(0)
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
            file = open(init_path + "library/tlv/lint_off.txt", 'a+')
            file.write(text + "\n")
            file.close()
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
        self.modlst = open(modpath_file, 'r')
        self.data = self.modlst.readlines()
        self.modlst.close()
        for item in self.data:
            if item != "\n":
                self.entry_var[self.count].addItem(item.strip())
        self.entry_var[self.count].currentTextChanged.connect(self.edit_modlst)
        self.trgrid.addWidget(self.entry_var[self.count], 1, 4, 1, 2)
        self.count += 1
        self.entry_var[self.count] = QtWidgets.QComboBox()
        self.entry_var[self.count].addItem("Remove lint_off")

        init_path = '../../'
        if os.name == 'nt':
            init_path = ''
        self.lint_off = open(init_path + "library/tlv/lint_off.txt", 'r')

        self.data = self.lint_off.readlines()
        self.lint_off.close()
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

        # CSS
        self.trbox.setStyleSheet(" \
        QGroupBox { border: 1px solid gray; border-radius: \
        9px; margin-top: 0.5em; } \
        QGroupBox::title { subcontrol-origin: margin; left: \
         10px; padding: 0 3px 0 3px; } \
        ")

        return self.trbox
