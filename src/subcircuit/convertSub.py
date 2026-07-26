from PyQt6 import QtWidgets
from configuration import Dialogs
from projManagement.Validation import Validation
from configuration.Appconfig import Appconfig
from subcircuit.subPaths import resolve_subcircuit, netlist_path
import os


# This class is called when user creates new Project
class convertSub(QtWidgets.QWidget):
    """
    Contains functions that checks project present for conversion and
    also function to convert Kicad Netlist to Ngspice Netlist.
    """

    def __init__(self, dockarea):
        super(convertSub, self).__init__()
        self.obj_validation = Validation()
        self.obj_appconfig = Appconfig()
        self.obj_dockarea = dockarea

    def createSub(self):
        """
        This function create command to call KiCad to Ngspice converter.
            If the netlist is not generated for selected project it will show
            error **The subcircuit does not contain any Kicad netlist file for
            conversion.**
            And if no project is selected for conversion, it again show error
            message to select a file or create a file.

        """
        print("Openinig Kicad-to-Ngspice converter from Subcircuit Module")
        self.projDir = self.obj_appconfig.current_subcircuit["SubcircuitName"]
        # Validating if current project is available or not
        if not self.obj_validation.validateKicad(self.projDir):
            self._error(
                'Please select the subcircuit first. You can either create '
                'new subcircuit or open existing subcircuit')
            return

        # The stem the user actually opened wins. Only when nothing was
        # recorded (a selection made by older code, or the folder changed
        # underneath us) do we re-derive it -- and then through the same
        # resolver Edit used, so the two can no longer disagree.
        stem = self.obj_appconfig.get_subcircuit_stem()
        if stem is None:
            stem, _status = resolve_subcircuit(self.projDir)
        if stem is None:
            self._error(
                'This folder holds several subcircuits and none of them is '
                'named after it, so eSim cannot tell which one to convert.\n\n'
                'Open the one you want with Edit first, then convert.')
            return

        if not self.obj_validation.validateCir(self.projDir, stem):
            self._error(
                'The subcircuit does not contain any Kicad netlist file'
                ' for conversion.')
            return

        self.projName = stem
        self.project = os.path.join(self.projDir, str(stem))
        self.obj_dockarea.kicadToNgspiceEditor(
            netlist_path(self.projDir, stem), "sub")

    def _error(self, message):
        """Show a modal converter error and mirror it into the eSim log."""
        self.msg = Dialogs.make_error_message(self)
        self.msg.setModal(True)
        self.msg.setWindowTitle("Error Message")
        self.msg.showMessage(message)
        self.obj_appconfig.print_error(message)
        self.msg.exec()
