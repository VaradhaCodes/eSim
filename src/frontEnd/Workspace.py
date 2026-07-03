# =========================================================================
#          FILE: Workspace.py
#
#   DESCRIPTION: Workspace selector — the small dialog that appears when
#                the user runs eSim for the first time (or resets their
#                workspace). Lets them pick a folder under which all
#                projects will live.
#
#       OPTIONS: ---
#  REQUIREMENTS: ---
#         NOTES: Layout uses QFormLayout + button row instead of a
#                hand-rolled QGridLayout, so widgets size cleanly on
#                HiDPI and the dialog can be resized without breaking.
#        AUTHOR: Fahim Khan, fahim.elex@gmail.com
#      MODIFIED: Rahul Paknikar, rahulp@iitb.ac.in + eSim UX refresh
#  ORGANIZATION: eSim Team at FOSSEE, IIT Bombay
#       CREATED: Wednesday 05 February 2015
#      REVISION: 2026 — layout refresh, removed blocking time.sleep,
#                 removed WindowStaysOnTopHint, replaced app-level
#                 globals with attribute dispatch.
# =========================================================================

from PyQt6 import QtCore, QtGui, QtWidgets
from configuration.Appconfig import Appconfig
from configuration import Dialogs
import os
import json


class Workspace(QtWidgets.QDialog):
    """
    Select-or-confirm your eSim workspace folder.

    The dialog is modal and blocks until the user chooses either:
      - the default workspace (~/eSim-Workspace), or
      - a custom folder, optionally marked as the default for next run.
    """

    def __init__(self, parent=None):
        super(Workspace, self).__init__(parent)
        self.obj_appconfig = Appconfig()
        self._app_view = None  # filled in by returnWhetherClickedOrNot()
        self.init_ui()

    # ------------------------------------------------------------------ UI
    def init_ui(self):
        self.setWindowTitle('Choose your workspace')
        self.setModal(True)
        self.setMinimumWidth(520)

        # Window icon
        init_path = '../../'
        if os.name == 'nt':
            init_path = ''
        if os.path.exists(init_path + 'images/logo.png'):
            self.setWindowIcon(QtGui.QIcon(init_path + 'images/logo.png'))

        # --- Header -----------------------------------------------------
        title = QtWidgets.QLabel('Workspace')
        title.setProperty('cssClass', 'title')

        sub = QtWidgets.QLabel(
            'eSim stores every project inside a single workspace folder. '
            'You can change this later from the toolbar.'
        )
        sub.setProperty('cssClass', 'muted')
        sub.setWordWrap(True)

        # --- Description card ------------------------------------------
        info_box = QtWidgets.QGroupBox()
        info_box.setTitle('What goes here?')
        info_layout = QtWidgets.QVBoxLayout(info_box)
        info_text = QtWidgets.QLabel(self.obj_appconfig.workspace_text)
        info_text.setWordWrap(True)
        info_text.setProperty('cssClass', 'subtle')
        info_layout.addWidget(info_text)

        # --- Form -------------------------------------------------------
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        form.setSpacing(10)

        path_row = QtWidgets.QHBoxLayout()
        self.workspace_loc = QtWidgets.QLineEdit(self.obj_appconfig.home)
        self.workspace_loc.setPlaceholderText('Path to your workspace folder')

        self.browse_btn = QtWidgets.QPushButton('Browse…')
        self.browse_btn.setProperty('cssClass', 'secondary')
        self.browse_btn.clicked.connect(self.browseLocation)

        path_row.addWidget(self.workspace_loc, 1)
        path_row.addWidget(self.browse_btn)

        self.chkbox = QtWidgets.QCheckBox('Use this workspace by default')
        self.chkbox.setCheckState(
            QtCore.Qt.CheckState(int(self.obj_appconfig.workspace_check))
        )

        form.addRow('Location', path_row)
        form.addRow('', self.chkbox)

        # --- Buttons ----------------------------------------------------
        button_box = QtWidgets.QDialogButtonBox()

        self.default_btn = QtWidgets.QPushButton('Use default')
        self.default_btn.clicked.connect(self.defaultWorkspace)

        self.ok_btn = QtWidgets.QPushButton('Use this folder')
        self.ok_btn.setDefault(True)
        self.ok_btn.clicked.connect(self.createWorkspace)

        button_box.addButton(
            self.default_btn,
            QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        button_box.addButton(
            self.ok_btn,
            QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)

        # Compose
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 20)
        outer.setSpacing(12)
        outer.addWidget(title)
        outer.addWidget(sub)
        outer.addSpacing(4)
        outer.addWidget(info_box)
        outer.addLayout(form)
        outer.addSpacing(4)
        outer.addWidget(button_box)

    # ------------------------------------------------------------------ helpers
    def returnWhetherClickedOrNot(self, appView):
        """Capture the main app view so we can refresh it after close."""
        self._app_view = appView

    def _refresh_project_explorer(self):
        """Reload the project explorer tree for the chosen workspace.

        Routes through ProjectExplorer.loadProjects() — our idempotent bulk
        loader that migrates entries to canonical keys, renders missing folders
        as stale and re-focuses the active project — instead of a hand-rolled
        clear()+addTreeNode loop. A failed refresh must never stop the main
        window from appearing.
        """
        view = self._app_view
        if view is None:
            return
        try:
            view.obj_Mainview.obj_projectExplorer.loadProjects()
        except Exception as e:
            print(f"[Workspace] project explorer refresh failed: {e}")

    def _finish_workspace_change(self):
        """Show the main window now that the workspace is chosen.

        Replaces the legacy ``time.sleep(1.5)`` on the GUI thread (which froze
        the window for 1.5 s after every workspace change). The refresh and the
        show() are split so the explorer rebuilds before the splash hides.
        """
        view = self._app_view
        if view is None:
            return
        try:
            if view.splash is not None:
                view.splash.close()
        except Exception:
            pass
        view.show()

    # ------------------------------------------------------------------ slots
    def defaultWorkspace(self):
        """User picked the default workspace — use it and close."""
        self.obj_appconfig.print_info(
            'Default workspace selected: '
            + self.obj_appconfig.default_workspace["workspace"]
        )
        self.accept()
        self._refresh_project_explorer()
        self._finish_workspace_change()

    def createWorkspace(self):
        """User picked a custom workspace — validate and save it."""
        self.obj_appconfig.workspace_check = self.chkbox.checkState().value

        path = self.workspace_loc.text().strip()
        if not path:
            Dialogs.warning(
                self, "Workspace",
                "Please choose a folder for your workspace."
            )
            return

        if os.name == 'nt':
            user_home = os.path.join('library', 'config')
        else:
            user_home = os.path.expanduser('~')

        try:
            with open(
                os.path.join(user_home, '.esim/workspace.txt'), 'w'
            ) as f:
                f.write(f"{self.obj_appconfig.workspace_check} {path}")
        except OSError as e:
            Dialogs.critical(
                self, "Workspace",
                f"Could not save workspace path:\n{e}"
            )
            return

        # Create folder if it doesn't exist
        try:
            if not os.path.isdir(path):
                os.makedirs(path)
        except OSError as e:
            Dialogs.critical(
                self, "Workspace",
                f"Could not create that folder:\n{e}"
            )
            return

        self.obj_appconfig.default_workspace["workspace"] = path
        self.obj_appconfig.print_info('Workspace: ' + path)
        self.accept()

        # Refresh project explorer cache
        self.obj_appconfig.dictPath["path"] = os.path.join(
            self.obj_appconfig.default_workspace["workspace"],
            '.projectExplorer.txt'
        )

        try:
            with open(self.obj_appconfig.dictPath["path"]) as f:
                self.obj_appconfig.project_explorer = json.load(f)
        except (OSError, ValueError):
            self.obj_appconfig.project_explorer = {}

        Appconfig.project_explorer = self.obj_appconfig.project_explorer

        self._refresh_project_explorer()
        self._finish_workspace_change()

    def browseLocation(self):
        self.workspace_directory = QtCore.QDir.toNativeSeparators(
            QtWidgets.QFileDialog.getExistingDirectory(
                self, 'Choose workspace folder', os.path.expanduser('~')
            )
        )
        if self.workspace_directory:
            self.workspace_loc.setText(self.workspace_directory)
