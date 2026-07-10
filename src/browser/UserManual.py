from PyQt6 import QtWidgets
import os
import subprocess
import webbrowser

from configuration import Dialogs
from configuration import paths


class UserManual(QtWidgets.QWidget):
    """
    This class opens the User Manual in the system PDF viewer / web browser
    when the help button is clicked.
    """

    # Candidates in preference order, resolved against the eSim install root
    # (never the process CWD — the app can be launched from anywhere).
    _CANDIDATES = (
        os.path.join("User-Manual", "eSim_Manual_2.5.pdf"),
        os.path.join("User-Manual", "eSim.html"),
    )

    def __init__(self):
        QtWidgets.QWidget.__init__(self)

        self.vlayout = QtWidgets.QVBoxLayout()

        manual = None
        for rel in self._CANDIDATES:
            candidate = os.path.join(paths.library_path("browser"), rel)
            if os.path.isfile(candidate):
                manual = candidate
                break

        if manual is None:
            Dialogs.warning(
                self, "User Manual not found",
                "The bundled user manual could not be found.\n\n"
                "You can read it online at "
                "https://esim.fossee.in/downloads (eSim documentation).")
            webbrowser.open("https://esim.fossee.in/resource")
        elif os.name == 'nt':
            os.startfile(os.path.realpath(manual))
        elif manual.endswith(".html"):
            webbrowser.open("file://" + os.path.realpath(manual))
        else:
            subprocess.Popen(
                ['xdg-open', os.path.realpath(manual)], shell=False
            )

        self.setLayout(self.vlayout)
