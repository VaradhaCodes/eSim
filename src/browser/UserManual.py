from PyQt6 import QtWidgets
import subprocess
import os


class UserManual(QtWidgets.QWidget):
    """
    This class opens User-Manual page in new tab of web browser
    when help button is clicked.
    """

    def __init__(self):
        QtWidgets.QWidget.__init__(self)

        self.vlayout = QtWidgets.QVBoxLayout()

        _BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        manual_path = os.path.join(_BASE_DIR, 'library', 'browser', 'User-Manual', 'eSim_Manual_2.5.pdf')

        if os.name == 'nt':
            os.startfile(manual_path)
        else:
            subprocess.Popen(
                ['xdg-open', manual_path], shell=False
            )

        self.setLayout(self.vlayout)
        self.show()
