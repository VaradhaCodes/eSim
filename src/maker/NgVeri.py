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
                model.run_verilator()
                model.make_verilator()
                model.copy_verilator()
                model.runMake()

                if os.name != 'nt':
                    model.runMakeInstall()
                else:
                    try:
                        shutil.copy(
                            self.release_dir +
                            "/src/xspice/icm/Ngveri/Ngveri.cm",
                            self.nghdl_home + "/lib/ngspice/"
                        )
                    except FileNotFoundError as err:
                        currentTermLogs.append(
                            "Error in copying Ngveri code model: " + str(err)
                        )

                if "error" not in currentTermLogs.toPlainText().lower():
                    currentTermLogs.append('''
                        <p style=\" font-size:16pt; font-weight:1000;
                        color:#00FF00;\"> Model Created Successfully!
                        </p>
                    ''')

        except BaseException as err:
            currentTermLogs.append(
                "Error in Ngspice code model generation " +
                "from Verilog: " + str(err)
            )

        if "error" in currentTermLogs.toPlainText().lower():
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
        self.optionsbox.setLayout(self.optionsgrid)
        # self.grid.addWidget(self.creategroup(), 1, 0, 5, 0)

        return self.optionsbox

    def edit_modlst(self, text):
        '''
            Remove a model added via NgVeri.  This now performs a full
            cleanup so the entry disappears from both the simulator and
            the KiCad symbol library:

              * `modpath.lst`  — the model is dropped from the build list.
              * Parameter XML  — `library/modelParamXML/Ngveri/<name>.xml`.
              * KiCad symbol   — every top-level `(symbol "<name>" ...)`
                                 block inside `eSim_Ngveri.kicad_sym`.
              * Source folder  — the generated Ngveri/<name>/ directory.
              * Rebuild + reinstall so ngspice forgets the model too.
        '''
        if text == "Remove Verilog Models":
            return
        index = self.entry_var[1].findText(text)
        self.entry_var[1].removeItem(index)
        self.entry_var[1].setCurrentIndex(0)
        ret = QtWidgets.QMessageBox.warning(
            None, "Warning",
            "<b>Remove the model '" + text + "'?</b><br/>"
            "This deletes its parameter XML, KiCad symbol and "
            "generated source files.",
            QtWidgets.QMessageBox.StandardButton.Ok,
            QtWidgets.QMessageBox.StandardButton.Cancel
        )
        if ret != QtWidgets.QMessageBox.StandardButton.Ok:
            return

        errors = []

        # 1. modpath.lst — drop the line if present.
        modpath = self.digital_home + '/modpath.lst'
        try:
            if os.path.exists(modpath):
                with open(modpath, 'r') as f:
                    data = f.readlines()
                data = [ln for ln in data if ln.strip() != text]
                with open(modpath, 'w') as f:
                    f.writelines(data)
        except OSError as err:
            errors.append("modpath.lst: " + str(err))

        # 2/3. Parameter XML and KiCad symbol — both routed through the
        # createkicad helper so paths stay consistent with model creation.
        schematicLib = createkicad.AutoSchematic()
        try:
            schematicLib.init(text, "")
            xml_path = os.path.join(
                schematicLib.xml_loc, 'Ngveri', text + '.xml')
            if os.path.exists(xml_path):
                os.remove(xml_path)
        except OSError as err:
            errors.append("XML: " + str(err))

        try:
            schematicLib.removeOldLibrary()
        except OSError as err:
            errors.append("kicad_sym: " + str(err))
        except PermissionError as err:
            errors.append(
                "kicad_sym (need write access to "
                + schematicLib.kicad_ngveri_sym + "): " + str(err))

        # 4. Generated model source folder.
        model_dir = os.path.join(self.digital_home, text)
        if os.path.isdir(model_dir):
            try:
                shutil.rmtree(model_dir)
            except OSError as err:
                errors.append("source dir: " + str(err))

        # 5. Rebuild & reinstall so the simulator forgets the model.
        try:
            self.fname = (
                Maker.verilogFile[self.filecount]
                if self.filecount < len(Maker.verilogFile) else ""
            )
            model = ModelGeneration.ModelGeneration(
                self.fname, self.entry_var[0])
            model.runMake()
            if os.name != 'nt':
                model.runMakeInstall()
            else:
                shutil.copy(
                    self.release_dir + "/src/xspice/icm/Ngveri/Ngveri.cm",
                    self.nghdl_home + "/lib/ngspice/"
                )
        except BaseException as err:
            errors.append("rebuild: " + str(err))

        if errors:
            QtWidgets.QMessageBox.critical(
                None, "Error Message",
                "Model '" + text + "' was only partially removed:<br/>"
                + "<br/>".join(errors),
                QtWidgets.QMessageBox.StandardButton.Ok
            )
        else:
            QtWidgets.QMessageBox.information(
                None, "Model Removed",
                "Model '" + text + "' was removed from ngspice and "
                "the KiCad symbol library.",
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
        self.modlst = open(self.digital_home + '/modpath.lst', 'r')
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
