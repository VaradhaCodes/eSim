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
from PyQt6 import QtCore, QtGui, QtWidgets
from configuration import Dialogs
from configuration import paths
from configuration.Appconfig import Appconfig
from frontEnd.theme_utils import zoom_px, on_zoom_changed
import os
import re
from os.path import expanduser
from .DesignBus import DesignBus
from .MakerchipBridge import MakerchipBridge
from . import verilog_library
from .hdl.ports import top_module_name
from .VerilogVerifier import HdlEditor
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
        self._makerchip_bridge = None
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
        previous = self.verilogfile
        self.verilogfile = self.bus.path if self.bus is not None else ""
        self._applying = True
        try:
            if self.entry_var[1].toPlainText() != text:
                self.entry_var[1].setText(text)
            self.entry_var[0].setText(
                self.verilogfile or
                "(saved automatically once your design names a module)")
        finally:
            self._applying = False
        # The design moved to a different file -- either it was just named for
        # the first time, or its module was renamed and it followed. Re-read
        # the library so the list and its "current" marker keep up.
        if self.verilogfile != previous:
            self.refresh_library_list()

    def closeEvent(self, event):
        self.stop_makerchip_bridge()
        if self.bus is not None and self._owns_bus:
            self.bus.close()
        super().closeEvent(event)

    def stop_makerchip_bridge(self):
        """Release the loopback session when Author or its host closes."""
        if self._makerchip_bridge is not None:
            self._makerchip_bridge.stop()
            self._makerchip_bridge = None

    # This function is to Add new verilog file
    def addverilog(self):

        self.verilogfile = QtCore.QDir.toNativeSeparators(
            QtWidgets.QFileDialog.getOpenFileName(
                self, "Open Verilog Directory",
                os.path.join(paths.repo_root(), "home"), "*v"
            )[0]
        )
        if self.verilogfile == "":
            # Cancelled: fall back to the design already open. Read it from the
            # bus, NOT from the path label -- when no design has a file yet
            # that label holds placeholder prose, which used to be handed on as
            # if it were a path.
            self.verilogfile = self.bus.path if self.bus is not None else ""

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
        # imported=True: this is a file of the user's own, so eSim takes a
        # library copy and never writes back to the original unless asked.
        self.bus.load_from_disk(self.verilogfile, imported=True)

    def load_verilog(self, filepath):
        self.bus.load_from_disk(filepath, imported=True)

    NEW_MODULE_STUB = (
        "module {name} (\n"
        "    input  clk,\n"
        "    input  rst,\n"
        "    output reg out\n"
        ");\n\n"
        "  always @(posedge clk) begin\n"
        "    if (rst)\n"
        "      out <= 0;\n"
        "    else\n"
        "      out <= ~out;\n"
        "  end\n\n"
        "endmodule\n"
    )

    def new_module(self):
        '''
            Start a new design from a named stub.

            Before this there was no way to begin a design inside eSim at all:
            the editor opened empty, Save could not create a file (it had
            nothing to name one after and reported "please check if it is
            chosen"), and Convert had no path to build. The only working route
            was to write a correctly-named .v somewhere else and open it, which
            made the built-in editor decorative.
        '''
        if self.bus is None:
            return
        name, ok = QtWidgets.QInputDialog.getText(
            self, "New Verilog Module", "Module name:", text="my_module")
        if not ok:
            return
        name = name.strip()
        if not re.fullmatch(r'[A-Za-z_]\w*', name):
            Dialogs.warning(
                self, "New Verilog Module",
                "<b>'" + name + "' is not a usable module name.</b><br/><br/>"
                "Use letters, digits and underscore, and do not start with a "
                "digit -- the module name becomes the model name, a C "
                "function and a make target.",
                QtWidgets.QMessageBox.StandardButton.Ok)
            return
        # start_new flushes the design being replaced, so the one on screen is
        # safely on disk before it is swapped out.
        self.bus.start_new(self.NEW_MODULE_STUB.format(name=name))
        self.refresh_library_list()

    # This function is used to save the edited file in eSim
    def save(self):
        '''
            Persist the design now.

            Rarely needed -- the design autosaves into the Verilog library
            under its own module name -- so this exists for the two cases
            autosave deliberately does not cover: a design with no parseable
            module yet (ask where to put it), and a design imported from a file
            of the user's own (mirror the edit back to that file, which nothing
            else in eSim ever writes).
        '''
        if self.bus is None:
            return
        self.bus.set_content(self.entry_var[1].toPlainText())
        target = self.bus.flush_autosave() or self.bus.save_to_disk()
        if not target:
            target = self.save_as()
            if not target:
                return
        origin = self.bus.mirror_to_origin()
        if origin:
            self.obj_Appconfig.print_info(
                'Saved ' + target + ' (and back to ' + origin + ')')
        else:
            self.obj_Appconfig.print_info('Saved ' + target)
        self.refresh_library_list()

    def save_as(self):
        '''
            Ask where to put the design and write it there. Returns the path,
            or "" if the user cancelled. From then on the design lives where
            they chose -- autosave writes there and stops renaming it after the
            module, since that home was a decision, not a default.
        '''
        if self.bus is None:
            return ""
        module = verilog_library.top_module(self.bus.get_content())
        start = self.bus.path or os.path.join(
            verilog_library.library_root(), (module or "design") + ".v")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Verilog Design", start,
            "Verilog Files (*.v *.sv);;All Files (*)")
        if not path:
            return ""
        written = self.bus.save_to_disk(path)
        if not written:
            Dialogs.critical(
                self, "Error Message",
                "Could not write <b>" + path + "</b>. Check that the folder "
                "exists and is writable.")
        return written

    # Open the current design through Makerchip's supported browser plugin.
    def runmakerchip(self):
        try:
            if not makerchipTOSAccepted(True):
                return

            # Makerchip reads the design off disk; flush any in-editor edits
            # first so it sees the current text, not a stale file.
            if self.bus is not None:
                materialized = self.bus.materialize()
                if materialized:
                    self.verilogfile = materialized

            if not self.verilogfile or not os.path.isfile(self.verilogfile):
                Dialogs.critical(
                    self, "Error Message",
                    "<b>There is no saved Verilog design to open.</b><br/><br/>"
                    "Name a module in the Author editor or open a Verilog "
                    "file, then try again.",
                    QtWidgets.QMessageBox.StandardButton.Ok)
                return

            print("Running Makerchip IDE...........................")
            # self.file = open(self.verilogfile,"w")
            # self.file.write(self.entry_var[1].toPlainText())
            # self.file.close()
            filename = self.verilogfile
            if os.path.splitext(self.verilogfile)[1].lower() != ".tlv":
                reply = Dialogs.warning(
                    self,
                    "Do you want to automate the top module? ",
                    "<b>Choose Yes to generate a Makerchip-ready .tlv wrapper "
                    "with a top module, or No to open the Verilog file "
                    "unchanged.</b><br/><br/>Browser edits are saved "
                    "automatically to the file opened in Makerchip. The "
                    "generated .tlv file sits beside the current Verilog "
                    "file.<br/><br/>The opened source is processed by the "
                    "hosted Makerchip service. Makerchip requires an active "
                    "internet connection and a modern browser.",
                    QtWidgets.QMessageBox.StandardButton.Yes
                    | QtWidgets.QMessageBox.StandardButton.No
                    | QtWidgets.QMessageBox.StandardButton.Cancel)
                if reply == QtWidgets.QMessageBox.StandardButton.Cancel:
                    return
                if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                    with open(self.verilogfile) as fh:
                        code = fh.read()
                    text = code
                    filename = os.path.splitext(self.verilogfile)[0] + ".tlv"
                    # Word-boundary strip of the standalone wire/reg keywords;
                    # the old spaced-substring replace still mangled tokens at
                    # line-start or tab-adjacent (out_reg, wire_sel).
                    code = re.sub(r'\b(wire|reg)\b', ' ', code)
                    vlog_ex = vlog.VerilogExtractor()
                    vlog_mods = vlog_ex.extract_objects_from_source(code)

                    # Wrap the design's TOP module, whatever the file is
                    # called. This used to require the file name and the module
                    # name to match, which meant pasting a design in and
                    # pressing this button could only ever fail. An empty or
                    # failed parse is still reported clearly here rather than
                    # crashing later on a loop variable referenced after its
                    # loop.
                    top_name = top_module_name(text)
                    module = None
                    for m in vlog_mods:
                        if top_name and m.name == top_name:
                            module = m
                            break
                    if module is None and vlog_mods:
                        module = vlog_mods[-1]
                    if module is None:
                        Dialogs.critical(
                            self,
                            "Error Message",
                            "<b>Error: no Verilog module could be read from "
                            "this design.</b><br/><br/>Check it for a syntax "
                            "error, then try again.",
                            QtWidgets.QMessageBox.StandardButton.Ok)
                        self.obj_Appconfig.print_info(
                            'Makerchip stopped: no parseable module in the '
                            'current design')
                        return

                    with open(
                            paths.library_path("tlv/lint_off.txt")) as fh:
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
                    print(module.name)
                    # The three passes below all describe the SAME module, so
                    # they read it straight off `module` instead of re-scanning
                    # vlog_mods for a file-name match each time.
                    reserved = ("clk", "reset", "cyc_cnt", "passed", "failed")
                    for p in module.ports:
                        if str(p.name) not in reserved:
                            string += '\t\tlogic ' + p.data_type\
                                + " " + p.name + ";//" + p.mode + "\n"
                    string += "//The $random() can be replaced \
if user wants to assign values\n"
                    for p in module.ports:
                        if str(p.mode) in ("input", "inout") \
                                and str(p.name) not in reserved:
                            string += '\t\tassign ' + p.name\
                                + " = " + "$random();\n"

                    string += '\t\t' + module.name + " " + module.name + '('
                    i = 0
                    for p in module.ports:
                        i = i + 1
                        string += "."+p.name+"("+p.name+")"
                        if i == len(module.ports):
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

            print("File: " + filename)
            self.stop_makerchip_bridge()
            bridge = MakerchipBridge(filename)
            url = bridge.start()
            self._makerchip_bridge = bridge
            if not QtGui.QDesktopServices.openUrl(QtCore.QUrl(url)):
                bridge.stop()
                self._makerchip_bridge = None
                raise RuntimeError("the default browser could not be opened")
            self.obj_Appconfig.print_info(
                "Makerchip opened for " + filename)
        except Exception as e:
            print(e)
            Dialogs.critical(
                self, "Error Message",
                "Could not open Makerchip. Check that a default browser is "
                "configured and that the internet connection is available.")
            print("Could not open Makerchip IDE:", e)
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

    # This creates the buttons/options

    def createoptionsBox(self):

        self.optionsbox = QtWidgets.QGroupBox()
        self.optionsbox.setTitle("Select Options")
        self.optionsgrid = QtWidgets.QGridLayout()
        # Even gutters + equal column stretch so the action row reads as one
        # balanced toolbar instead of buttons drifting apart (the old layout
        # left an empty column between "Add Top Level" and "Save").
        self.optionsgrid.setHorizontalSpacing(12)
        self.optionsgrid.setContentsMargins(4, 4, 4, 4)
        self.optionsgroupbtn = QtWidgets.QButtonGroup()

        self.verifier_btn = QtWidgets.QPushButton("Verilog Simulator IDE")
        # Aurora styles this primary launcher via the verifierPrimary accent
        # instead of a hard-coded green that fought the cyan theme.
        self.verifier_btn.setProperty("cssClass", "verifierPrimary")
        self.optionsgroupbtn.addButton(self.verifier_btn)
        self.verifier_btn.clicked.connect(self.open_verifier)

        self.newoption = QtWidgets.QPushButton("New Verilog Module")
        self.newoption.setToolTip(
            "Start a new design. It is saved automatically under its own "
            "module name in your Verilog Library.")
        self.optionsgroupbtn.addButton(self.newoption)
        self.newoption.clicked.connect(self.new_module)

        self.addoptions = QtWidgets.QPushButton("Add Top Level Verilog Model")
        self.addoptions.setToolTip(
            "Open a .v file of your own. eSim works on a copy in your Verilog "
            "Library; your original file is only written when you press Save.")
        self.optionsgroupbtn.addButton(self.addoptions)
        self.addoptions.clicked.connect(self.addverilog)

        self.saveoption = QtWidgets.QPushButton("Save")
        # Save is the workhorse action here -- promote it to the cyan accent so
        # the row has a clear anchor instead of three near-invisible white cards
        # blending into the page.
        self.saveoption.setProperty("cssClass", "primary")
        self.optionsgroupbtn.addButton(self.saveoption)
        self.saveoption.clicked.connect(self.save)

        self.runoptions = QtWidgets.QPushButton("Edit in Makerchip IDE")
        self.runoptions.setToolTip(
            "Open this design in hosted Makerchip using your default browser. "
            "Edits autosave to the opened file; compilation runs in the "
            "Makerchip service."
        )
        self.runoptions.setToolTipDuration(5000)
        self.optionsgroupbtn.addButton(self.runoptions)
        self.runoptions.clicked.connect(self.runmakerchip)

        # Lay the buttons out in contiguous, equally-stretched columns. Taller
        # min-height gives the labels room to breathe (the old 32px buttons made
        # the centred text look cramped on wide cards).
        action_btns = [self.verifier_btn, self.newoption, self.addoptions,
                       self.saveoption, self.runoptions]
        if not makerchipTOSAccepted(False):
            self.acceptTOS = QtWidgets.QPushButton("Accept Makerchip TOS")
            self.optionsgroupbtn.addButton(self.acceptTOS)
            self.acceptTOS.clicked.connect(lambda: makerchipTOSAccepted(True))
            action_btns.append(self.acceptTOS)

        for col, btn in enumerate(action_btns):
            btn.setMinimumHeight(40)
            btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                              QtWidgets.QSizePolicy.Policy.Fixed)
            self.optionsgrid.addWidget(btn, 0, col)
            self.optionsgrid.setColumnStretch(col, 1)

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
        self.trbox.setTitle("Design")
        # Stack the fields vertically with section headers above each one.
        # The old two-column grid put the labels in column 0 next to a tall
        # editor in column 1, so ".tlv code" floated halfway down the left edge
        # and the empty path value sat with nothing beside it.
        self.trgrid = QtWidgets.QVBoxLayout()
        self.trgrid.setContentsMargins(12, 10, 12, 12)
        self.trgrid.setSpacing(6)
        self.trbox.setLayout(self.trgrid)

        path_header = QtWidgets.QLabel("Saved to")
        path_header.setProperty("cssClass", "caps")
        self.trgrid.addWidget(path_header)

        self.count = 0
        self.entry_var[self.count] = QtWidgets.QLabel()
        # Placeholder so the row isn't a blank gap before the file is saved.
        # It says what WILL happen rather than telling the user to press Save,
        # which used to be advice they could not act on -- Save could not
        # create a file for a design that had never had one.
        self.entry_var[self.count].setText(
            "(saved automatically once your design names a module)")
        self.entry_var[self.count].setProperty("cssClass", "muted")
        self.trgrid.addWidget(self.entry_var[self.count])
        self.count += 1

        # Border/title styling comes from the global Aurora QGroupBox QSS so
        # the group reads with the same gradient hairline as the rest of eSim.

        code_header = QtWidgets.QLabel("Verilog code")
        code_header.setProperty("cssClass", "caps")
        self.trgrid.addSpacing(4)
        self.trgrid.addWidget(code_header)

        # QsciScintilla-based Verilog editor (syntax highlight, line numbers,
        # mono font, theme-aware) replacing the bare QTextEdit white box. It
        # exposes toPlainText/setText aliases so existing call sites are
        # unchanged.
        self.entry_var[self.count] = HdlEditor("design.v")
        # Editor on the left, the library on the right: the designs already
        # saved are what a user reaches for most often after "what am I editing
        # right now", so they belong on the same screen rather than behind a
        # file dialog.
        editor_row = QtWidgets.QHBoxLayout()
        editor_row.setSpacing(12)
        editor_row.addWidget(self.entry_var[self.count], 1)
        editor_row.addWidget(self.createLibraryPanel())
        self.trgrid.addLayout(editor_row, 1)
        # Author edits stream into the shared design (see _on_editor_changed).
        self.entry_var[self.count].textChanged.connect(self._on_editor_changed)
        self.count += 1

        # Border/title styling comes from the global Aurora QGroupBox QSS so
        # the group reads with the same gradient hairline as the rest of eSim.

        return self.trbox

    # ------------------------------------------------------------------ #
    #  My Verilog Designs -- the library panel
    #
    #  Named to keep it distinct from Convert's "Remove Verilog Models…",
    #  which removes COMPILED models from ngspice. This lists source designs.
    # ------------------------------------------------------------------ #
    def createLibraryPanel(self):
        panel = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(panel)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)

        caption = QtWidgets.QLabel("MY VERILOG DESIGNS")
        caption.setProperty("cssClass", "caps")
        column.addWidget(caption)

        self.libraryList = QtWidgets.QListWidget()
        self.libraryList.setToolTip(
            "Every design you have written in eSim, newest first.\n"
            "Double-click to open one.")
        self.libraryList.itemDoubleClicked.connect(self.open_from_library)
        self.libraryList.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.libraryList.customContextMenuRequested.connect(
            self.show_library_context_menu)
        column.addWidget(self.libraryList, 1)

        self.libraryFolderBtn = QtWidgets.QPushButton("Open folder")
        self.libraryFolderBtn.setToolTip(
            "Show the library folder in your file manager")
        self.libraryFolderBtn.clicked.connect(self.open_library_folder)
        column.addWidget(self.libraryFolderBtn)

        # The buttons and rows here take their metrics from the QSS, which
        # scales with zoom, so the column that holds them has to as well --
        # same reason as the NgVeri control column.
        on_zoom_changed(panel, lambda z, w=panel: w.setFixedWidth(zoom_px(210, z)))
        self.refresh_library_list()
        return panel

    def refresh_library_list(self):
        '''Re-read the library from disk and show it, newest design first.'''
        widget = getattr(self, 'libraryList', None)
        if widget is None:
            return
        current = self.bus.path if self.bus is not None else ""
        widget.clear()
        for name, path, _mtime in verilog_library.list_designs():
            item = QtWidgets.QListWidgetItem(name)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, path)
            item.setToolTip(path)
            if current and os.path.normcase(path) == os.path.normcase(current):
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            widget.addItem(item)
        if widget.count() == 0:
            empty = QtWidgets.QListWidgetItem(
                "(nothing saved yet)")
            empty.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
            widget.addItem(empty)

    def open_from_library(self, item):
        path = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if not path or self.bus is None:
            return
        # NOT imported=True: this file already IS the library copy, so there is
        # no original elsewhere to mirror back to.
        self.bus.load_from_disk(path)
        self.refresh_library_list()

    def show_library_context_menu(self, pos):
        item = self.libraryList.itemAt(pos)
        if item is None:
            return
        path = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if not path:
            return
        menu = QtWidgets.QMenu(self)
        open_action = menu.addAction("Open")
        remove_action = menu.addAction("Remove from library…")
        action = menu.exec(self.libraryList.mapToGlobal(pos))
        if action == open_action:
            self.open_from_library(item)
        elif action == remove_action:
            self.remove_from_library(item.text())

    def remove_from_library(self, name):
        reply = Dialogs.question(
            self, "Remove design",
            "Delete <b>" + str(name) + "</b> and everything in its folder "
            "(its testbench and history too)?<br/><br/>"
            "This does not remove any model already built from it.",
            QtWidgets.QMessageBox.StandardButton.Yes |
            QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No)
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        if not verilog_library.remove_design(name):
            Dialogs.warning(
                self, "Remove design",
                "Could not remove <b>" + str(name) + "</b>.",
                QtWidgets.QMessageBox.StandardButton.Ok)
        self.refresh_library_list()

    def open_library_folder(self):
        root = verilog_library.library_root()
        try:
            os.makedirs(root, exist_ok=True)
        except OSError:
            pass
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(root))
