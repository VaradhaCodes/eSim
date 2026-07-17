import os
import shlex
from PyQt6 import QtWidgets, QtCore
from configuration.Appconfig import Appconfig
from configuration import paths
from configuration import Dialogs
from projManagement.Worker import WorkerThread
from projManagement.projectPaths import find_anchors, resolve_stem, \
    stem_from_file


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

        self.editfile = QtCore.QDir.toNativeSeparators(
            QtWidgets.QFileDialog.getExistingDirectory(
                self, "Open File", paths.library_path("SubcircuitLibrary")
            )
        )

        if self.editfile:
            self.obj_Appconfig = Appconfig()
            self.obj_Appconfig.current_subcircuit['SubcircuitName'] \
                = self.editfile

            # Resolve the subcircuit stem from its .sub anchor (not the folder
            # name), then open the matching schematic (kicad6 .kicad_sch or
            # kicad4 .sch). Path is quoted so directories with spaces work.
            sub_files = find_anchors(self.editfile, 'sub')

            # More than one .sub in the folder: resolve_stem cannot choose for
            # us (it yields None unless an anchor matches the folder name), and
            # str(None) used to send eeschema off to open "None.sch". Ask, the
            # way openProject does for a folder holding several projects.
            if len(sub_files) > 1:
                stem = self._chooseSubcircuit(sub_files)
                if stem is None:
                    self.obj_appconfig.print_info('No subcircuit opened')
                    return
            else:
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

    def _chooseSubcircuit(self, sub_files):
        """
        When a folder contains more than one .sub, ask the user which
        subcircuit to edit. Returns the chosen stem, or None if cancelled.
        """
        stems = [stem_from_file(p) for p in sub_files]
        choice, ok = QtWidgets.QInputDialog.getItem(
            Dialogs.resolve_parent(self), "Select Subcircuit",
            "This folder contains multiple eSim subcircuits.\n"
            "Choose one to edit:",
            stems, 0, False
        )
        if ok and choice:
            return str(choice)
        return None
