# =========================================================================
#             FILE: Maker.py
#
#            USAGE: ---
#
#      DESCRIPTION: This define all components of the Makerchip Tab.
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
import hdlparse.verilog_parser as vlog
from PyQt6 import QtCore, QtWidgets
from configuration import Dialogs
from configuration.Appconfig import Appconfig
import os
import shutil
from os.path import expanduser
from .DesignBus import DesignBus
home = expanduser("~")

# declaring the global variables
# verilogFile[filecount] mirrors each design's path on disk. The DesignBus that
# owns the design is its one writer; NgVeri / ModelGeneration read this slot.
verilogFile = []


# This function is called to accept TOS of makerchip
def makerchipTOSAccepted(display=True):
    if not os.path.isfile(home + "/.makerchip_accepted"):
        if display:
            reply = Dialogs.warning(
                None, "Terms of Service", "Please review the Makerchip \
                       Terms of Service \
                       (<a href='https://www.makerchip.com/terms/'>\
                       https://www.makerchip.com/terms/</a>). \
                       Have you read and do you \
                       accept these Terms of Service?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
            )

            if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                f = open(home + "/.makerchip_accepted", "w")
                f.close()
                return True

        return False
    return True


# beginning class Maker. This class create the Maker Tab
class Maker(QtWidgets.QWidget):

    # initailising the varaibles
    def __init__(self, filecount, bus=None):
        QtWidgets.QWidget.__init__(self)
        self.count = 0
        self.text = ""
        self.verilogfile = ""
        self.filecount = filecount
        self.entry_var = {}
        self._applying = False
        self.bus = None
        self._owns_bus = False
        self.createMakerWidget()
        self.obj_Appconfig = Appconfig()
        verilogFile.append("")
        # The design lives in a DesignBus. The Flow Navigator injects a shared
        # one so Author/Verify/Convert are views on one design; standalone Maker
        # owns its own.
        self.set_design_bus(bus or DesignBus(self.filecount),
                            take_ownership=bus is None)

    # Creating the various components of the Widget(Maker Tab)
    def createMakerWidget(self):

        self.grid = QtWidgets.QGridLayout()
        self.setLayout(self.grid)

        self.grid.addWidget(self.createoptionsBox(), 0, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        self.grid.addWidget(self.creategroup(), 1, 0, 5, 0)
        # self.grid.addWidget(self.creategroup(), 1, 0, 5, 0)
        self.show()

    def set_design_bus(self, bus, take_ownership=False):
        """Bind this Author view to a shared design. Disconnects any previous
        bus, subscribes the editor, and renders the current design. The Flow
        Navigator calls this to make all stages share one design."""
        if self.bus is bus:
            return
        if self.bus is not None:
            try:
                self.bus.contentChanged.disconnect(self._render_from_bus)
            except (TypeError, RuntimeError):
                pass
            if self._owns_bus:
                self.bus.close()
        self.bus = bus
        self._owns_bus = take_ownership
        self.bus.contentChanged.connect(self._render_from_bus)
        self._render_from_bus(self.bus.get_content())

    def _on_editor_changed(self):
        """Editor edits flow straight into the shared design (in memory). Guarded
        so rendering the bus back into the editor cannot loop."""
        if self._applying or self.bus is None:
            return
        self.bus.set_content(self.entry_var[1].toPlainText())

    def _render_from_bus(self, text):
        """Show the shared design's current text + path. Guarded against the
        editor's own textChanged echoing back into the bus."""
        self.verilogfile = self.bus.path if self.bus is not None else ""
        self._applying = True
        try:
            if self.entry_var[1].toPlainText() != text:
                self.entry_var[1].setText(text)
            self.entry_var[0].setText(self.verilogfile)
        finally:
            self._applying = False

    def closeEvent(self, event):
        if self.bus is not None and self._owns_bus:
            self.bus.close()
        super().closeEvent(event)

    # This function is to Add new verilog file
    def addverilog(self):

        init_path = '../../'
        if os.name == 'nt':
            init_path = ''
        self.verilogfile = QtCore.QDir.toNativeSeparators(
            QtWidgets.QFileDialog.getOpenFileName(
                self, "Open Verilog Directory",
                init_path + "home", "*v"
            )[0]
        )
        if self.verilogfile == "":
            self.verilogfile = self.entry_var[0].text()

        if self.verilogfile == "":
            reply = Dialogs.critical(
                self,
                "Error Message",
                "<b>No Verilog File Chosen. \
                Please choose a verilog file.</b>",
                QtWidgets.QMessageBox.StandardButton.Ok | QtWidgets.QMessageBox.StandardButton.Cancel)

            if reply == QtWidgets.QMessageBox.StandardButton.Ok:
                self.addverilog()

                if self.verilogfile == "":
                    return

                self.obj_Appconfig.print_info('Add Verilog File Called')

            elif reply == QtWidgets.QMessageBox.StandardButton.Cancel:
                self.obj_Appconfig.print_info('No Verilog File Chosen')
                return

        # Loading routes through the shared design; the bus reads the file,
        # mirrors the slot, arms the external-edit watch, and renders the editor.
        self.bus.load_from_disk(self.verilogfile)

    def load_verilog(self, filepath):
        self.bus.load_from_disk(filepath)

    # This function is used to save the edited file in eSim
    def save(self):
        if self.bus is None:
            return
        self.bus.set_content(self.entry_var[1].toPlainText())
        if not self.bus.save_to_disk():
            self.msg = QtWidgets.QErrorMessage(self)
            self.msg.setModal(True)
            self.msg.setWindowTitle("Error Message")
            self.msg.showMessage(
                "Error in saving verilog file. Please check if it is chosen."
            )
            self.msg.exec()

    # This is used to run the makerchip-app
    def runmakerchip(self):
        init_path = '../../'
        if os.name == 'nt':
            init_path = ''
        try:
            if not makerchipTOSAccepted(True):
                return

            # Makerchip reads the design off disk; flush any in-editor edits
            # first so it sees the current text, not a stale file.
            if self.bus is not None:
                self.bus.materialize()

            print("Running Makerchip IDE...........................")
            # self.file = open(self.verilogfile,"w")
            # self.file.write(self.entry_var[1].toPlainText())
            # self.file.close()
            filename = self.verilogfile
            if self.verilogfile.split('.')[-1] != "tlv":
                reply = Dialogs.warning(
                    self,
                    "Do you want to automate the top module? ",
                    "<b>Click on YES button if you want the top module \
                    to be added automatically. A .tlv file will be created \
                    in the directory of current verilog file \
                    and the Makerchip IDE will be running on \
                    this file. Otherwise click on NO button. \
                    To not open Makerchip IDE, click on CANCEL button. </b>\
                    <br><br> NOTE: Makerchip IDE requires an active \
                    internet connection and a browser.",
                    QtWidgets.QMessageBox.StandardButton.Yes
                    | QtWidgets.QMessageBox.StandardButton.No
                    | QtWidgets.QMessageBox.StandardButton.Cancel)
                if reply == QtWidgets.QMessageBox.StandardButton.Cancel:
                    return
                if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                    with open(self.verilogfile) as fh:
                        code = fh.read()
                    text = code
                    filename = '.'.join(
                        self.verilogfile.split('.')[:-1]) + ".tlv"
                    file = os.path.basename('.'.join(
                        self.verilogfile.split('.')[:-1]))
                    code = code.replace(" wire ", " ")
                    code = code.replace(" reg ", " ")
                    vlog_ex = vlog.VerilogExtractor()
                    vlog_mods = vlog_ex.extract_objects_from_source(code)

                    # Find the module matching the file name up front. An empty
                    # or failed parse, or a name mismatch, is now reported
                    # clearly here instead of crashing later on a loop variable
                    # referenced after its loop.
                    module = None
                    for m in vlog_mods:
                        if m.name.lower() == file.lower():
                            module = m
                            break
                    if module is None:
                        Dialogs.critical(
                            self,
                            "Error Message",
                            "<b>Error: File name and module \
                            name are not same. Please \
                            ensure that they are same.</b>",
                            QtWidgets.QMessageBox.StandardButton.Ok)
                        self.obj_Appconfig.print_info(
                            'NgVeri stopped due to file \
name and module name not matching error')
                        return

                    with open(
                            init_path + "library/tlv/lint_off.txt") as fh:
                        lint_off = fh.readlines()
                    string = '''\\TLV_version 1d: tl-x.org\n\\SV\n'''
                    for item in lint_off:
                        string += "/* verilator lint_off " + \
                            item.strip("\n") + "*/  "
                    string += '''\n\n//Your Verilog/System \
Verilog Code Starts Here:\n''' + \
                        text + '''\n\n//Top Module Code \
Starts here:\n\tmodule top(input \
logic clk, input logic reset, input logic [31:0] cyc_cnt, \
output logic passed, output logic failed);\n'''
                    print(file)
                    for m in vlog_mods:
                        if m.name.lower() == file.lower():
                            for p in m.ports:
                                if str(
                                        p.name) != "clk" and str(
                                        p.name) != "reset" and str(
                                        p.name) != "cyc_cnt" and str(
                                        p.name) != "passed" and str(
                                        p.name) != "failed":
                                    string += '\t\tlogic ' + p.data_type\
                                     + " " + p.name + ";//" + p.mode + "\n"
                    string += "//The $random() can be replaced \
if user wants to assign values\n"
                    for m in vlog_mods:
                        if m.name.lower() == file.lower():
                            for p in m.ports:
                                if str(
                                        p.mode) == "input" or str(
                                        p.mode) == "inout":
                                    if str(
                                            p.name) != "clk" and str(
                                            p.name) != "reset" and str(
                                            p.name) != "cyc_cnt" and str(
                                            p.name) != "passed" and str(
                                            p.name) != "failed":
                                        string += '\t\tassign ' + p.name\
                                         + " = " + "$random();\n"

                    for m in vlog_mods:
                        if m.name.lower() == file.lower():
                            string += '\t\t' + m.name + " " + m.name + '('
                            i = 0
                            for p in m.ports:
                                i = i + 1
                                string += "."+p.name+"("+p.name+")"
                                if i == len(m.ports):
                                    string += ");\n\t\n\\TLV\n//\
Add \\TLV here if desired\
                                     \n\\SV\nendmodule\n\n"
                                else:
                                    string += ", "
                    # Write the .tlv only now that generation has fully
                    # succeeded, so a failure/return above never leaves a
                    # half-written, corrupt file on disk.
                    with open(filename, 'w') as f:
                        f.write(string)

            makerchip_bin = shutil.which('makerchip')
            if makerchip_bin is None:
                Dialogs.critical(
                    self, "Error Message",
                    "<b>Makerchip was not found on your system.</b><br/>"
                    "Install it with <i>pip install makerchip-app</i> and "
                    "make sure it is on your PATH, then try again.",
                    QtWidgets.QMessageBox.StandardButton.Ok)
                return
            self.process = QtCore.QProcess(self)
            self.process.errorOccurred.connect(self._makerchip_launch_error)
            print("File: " + filename)
            # Pass the file name as a separate argument (not one space-joined
            # string) so a path containing spaces is not split into pieces.
            self.process.start(makerchip_bin, [filename])
            print(
                "Makerchip IDE command process pid ---------->",
                self.process.processId())
        except Exception as e:
            print(e)
            self.msg = QtWidgets.QErrorMessage(self)
            self.msg.setModal(True)
            self.msg.setWindowTitle("Error Message")
            self.msg.showMessage(
                "Error in running Makerchip IDE. \
Please check if verilog file is chosen.")
            self.msg.exec()
            print("Error in running Makerchip IDE. \
Please check if verilog file is chosen.")
        #   initial = self.read_file()

        # while True:
        #     current = self.read_file()
        #     if initial != current:
        #         for line in current:
        #             if line not in initial:
        #                 print(line)
        #         initial = current
        # self.processfile = QtCore.QProcess(self)
        # self.processfile.start("python3 notify.py")
        # print(self.processfile.readChannel())

    def _makerchip_launch_error(self, _error):
        '''
            Called by QProcess.errorOccurred if the Makerchip IDE fails to
            start (e.g. not installed, or no working browser/network). Tells
            the user instead of failing silently.
        '''
        self.msg = QtWidgets.QErrorMessage(self)
        self.msg.setModal(True)
        self.msg.setWindowTitle("Error Message")
        self.msg.showMessage(
            "Could not launch the Makerchip IDE. It needs the makerchip-app "
            "installed, an active internet connection and a browser.")
        self.msg.exec()

    # This creates the buttons/options

    def createoptionsBox(self):

        self.optionsbox = QtWidgets.QGroupBox()
        self.optionsbox.setTitle("Select Options")
        self.optionsgrid = QtWidgets.QGridLayout()
        # self.optionsbox2 = QtWidgets.QGroupBox()
        # self.optionsbox2.setTitle("Note: Please save the file once edited")
        # self.optionsgrid2 = QtWidgets.QGridLayout()
        self.optionsgroupbtn = QtWidgets.QButtonGroup()

        self.verifier_btn = QtWidgets.QPushButton("Verilog Simulator IDE")
        self.verifier_btn.setStyleSheet(
            "background-color: #2e7d32; color: white; font-weight: bold;")
        self.optionsgroupbtn.addButton(self.verifier_btn)
        self.verifier_btn.clicked.connect(self.open_verifier)
        self.optionsgrid.addWidget(self.verifier_btn, 0, 0)

        self.addoptions = QtWidgets.QPushButton("Add Top Level Verilog Model")
        self.optionsgroupbtn.addButton(self.addoptions)
        self.addoptions.clicked.connect(self.addverilog)
        self.optionsgrid.addWidget(self.addoptions, 0, 1)
        # self.optionsbox.setLayout(self.optionsgrid)
        # self.grid.addWidget(self.creategroup(), 1, 0, 5, 0
        self.saveoption = QtWidgets.QPushButton("Save")
        self.optionsgroupbtn.addButton(self.saveoption)
        self.saveoption.clicked.connect(self.save)
        self.optionsgrid.addWidget(self.saveoption, 0, 3)
        # self.optionsbox.setLayout(self.optionsgrid)
        # self.grid.addWidget(self.creategroup(), 1, 0, 5, 0)
        self.runoptions = QtWidgets.QPushButton("Edit in Makerchip IDE")
        self.runoptions.setToolTip(
            "Requires internet connection and a browser"
        )
        self.runoptions.setToolTipDuration(5000)
        self.optionsgroupbtn.addButton(self.runoptions)
        self.runoptions.clicked.connect(self.runmakerchip)
        self.optionsgrid.addWidget(self.runoptions, 0, 4)
        # self.optionsbox.setLayout(self.optionsgrid)
        # self.grid.addWidget(self.creategroup(), 1, 0, 5, 0)
        if not makerchipTOSAccepted(False):
            self.acceptTOS = QtWidgets.QPushButton("Accept Makerchip TOS")
            self.optionsgroupbtn.addButton(self.acceptTOS)
            self.acceptTOS.clicked.connect(lambda: makerchipTOSAccepted(True))
            self.optionsgrid.addWidget(self.acceptTOS, 0, 5)
            # self.optionsbox.setLayout(self.optionsgrid)
            # self.grid.addWidget(self.creategroup(), 1, 0, 5, 0)
        self.optionsbox.setLayout(self.optionsgrid)
        return self.optionsbox

    def open_verifier(self):
        # When hosted in the Flow Navigator the Verilog Simulator IDE is a
        # docked stage, not a flying dialog: hand off to the navigator instead
        # of opening a top-level window. Standalone Maker keeps the dialog.
        hook = getattr(self, '_verify_hook', None)
        if hook is not None:
            hook()
            return
        if not hasattr(self, 'verifier_win'):
            from .VerilogVerifier import VerilogVerifier
            self.verifier_win = QtWidgets.QDialog(self.window())
            self.verifier_win.setWindowTitle("eSim-Verilog Simulator IDE")
            self.verifier_win.setWindowFlags(
                self.verifier_win.windowFlags()
                | QtCore.Qt.WindowType.WindowMaximizeButtonHint
                | QtCore.Qt.WindowType.WindowMinimizeButtonHint)
            layout = QtWidgets.QVBoxLayout(self.verifier_win)
            layout.setContentsMargins(0, 0, 0, 0)
            self.obj_VerilogVerifier = VerilogVerifier()

            # Share the one design: the dialog reads from / writes to the same
            # bus the Author editor uses, so there is no disk hand-off. Edits
            # made here are collected into the bus when the dialog closes, and
            # the Author editor -- which renders the bus live -- updates itself.
            self.obj_VerilogVerifier.set_design_bus(self.bus)
            self.verifier_win.finished.connect(
                lambda _=0: self.obj_VerilogVerifier.collect_into_bus())

            layout.addWidget(self.obj_VerilogVerifier)
            self.verifier_win.resize(1000, 700)

        self.obj_VerilogVerifier.render_from_bus()
        self.verifier_win.show()
        self.verifier_win.raise_()
        self.verifier_win.activateWindow()

    # This function adds the other parts of widget like text box
    def creategroup(self):
        self.trbox = QtWidgets.QGroupBox()
        self.trbox.setTitle(".tlv file")
        # self.trbox.setDisabled(True)
        # self.trbox.setVisible(False)
        self.trgrid = QtWidgets.QGridLayout()
        self.trbox.setLayout(self.trgrid)

        self.start = QtWidgets.QLabel("Path to .tlv file")
        self.trgrid.addWidget(self.start, 1, 0)
        self.count = 0
        self.entry_var[self.count] = QtWidgets.QLabel()
        self.trgrid.addWidget(self.entry_var[self.count], 1, 1)
        self.entry_var[self.count].setMaximumWidth(1000)
        self.count += 1

        # CSS
        self.trbox.setStyleSheet(" \
        QGroupBox { border: 1px solid gray; border-radius: \
        9px; margin-top: 0.5em; } \
        QGroupBox::title { subcontrol-origin: margin; left: \
         10px; padding: 0 3px 0 3px; } \
        ")

        self.start = QtWidgets.QLabel(".tlv code")
        # self.start2 = QtWidgets.QLabel("Note: \
        # Please save the file once edited")
        # self.start2.setStyleSheet("background-color: red")
        self.trgrid.addWidget(self.start, 2, 0)
        # self.trgrid.addWidget(self.start2, 3,0)
        self.entry_var[self.count] = QtWidgets.QTextEdit()
        self.trgrid.addWidget(self.entry_var[self.count], 2, 1)
        self.entry_var[self.count].setMaximumWidth(1000)
        self.entry_var[self.count].setMaximumHeight(1000)
        # Author edits stream into the shared design (see _on_editor_changed).
        self.entry_var[self.count].textChanged.connect(self._on_editor_changed)
        self.count += 1

        # CSS
        self.trbox.setStyleSheet(" \
        QGroupBox { border: 1px solid gray; border-radius: \
        9px; margin-top: 0.5em; } \
        QGroupBox::title { subcontrol-origin: margin; left: \
         10px; padding: 0 3px 0 3px; } \
        ")

        return self.trbox
