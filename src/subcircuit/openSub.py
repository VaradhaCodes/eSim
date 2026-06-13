import os
import shlex
from PyQt6 import QtWidgets, QtCore
from configuration.Appconfig import Appconfig
from projManagement.Worker import WorkerThread
from projManagement.projectPaths import resolve_stem


# This class is called when User clicks on Edit Subcircuit Button.
class openSub(QtWidgets.QWidget):
    """
    It opens the existing subcircuit projects that are present in
    Subcircuit directory.
    """

    def __init__(self):
        super(openSub, self).__init__()
        self.obj_appconfig = Appconfig()

    def body(self):

        init_path = '../../'
        if os.name == 'nt':
            init_path = ''

        self.editfile = QtCore.QDir.toNativeSeparators(
            QtWidgets.QFileDialog.getExistingDirectory(
                None, "Open File", init_path + "library/SubcircuitLibrary"
            )
        )

        if self.editfile:
            self.obj_Appconfig = Appconfig()
            self.obj_Appconfig.current_subcircuit['SubcircuitName'] \
                = self.editfile

            # Resolve the subcircuit stem from its .sub anchor (not the folder
            # name), then open the matching schematic (kicad6 .kicad_sch or
            # kicad4 .sch). Path is quoted so directories with spaces work.
            stem, _status = resolve_stem(self.editfile, 'sub')
            self.schname = stem
            base = os.path.join(self.editfile, str(stem))
            if os.path.exists(base + ".kicad_sch"):
                schematic = base + ".kicad_sch"
            else:
                schematic = base + ".sch"
            self.cmd = "eeschema " + shlex.quote(schematic)
            self.obj_workThread = WorkerThread(self.cmd)
            self.obj_workThread.start()
