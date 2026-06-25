# =========================================================================
#          FILE: Application.py
#
#         USAGE: ---
#
#   DESCRIPTION: This main file use to start the Application
#
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: Fahim Khan, fahim.elex@gmail.com
#    MAINTAINED: Rahul Paknikar, rahulp@iitb.ac.in
#                Sumanto Kar, sumantokar@iitb.ac.in
#                Pranav P, pranavsdreams@gmail.com
#  ORGANIZATION: eSim Team at FOSSEE, IIT Bombay
#       CREATED: Tuesday 24 February 2015
#      REVISION: Wednesday 07 June 2023
# =========================================================================

import os
import sys
import traceback
import webbrowser

if os.name == 'nt':
    from frontEnd import pathmagic  # noqa:F401
    init_path = ''
else:
    import pathmagic    # noqa:F401
    init_path = '../../'

from PyQt6 import QtGui, QtCore, QtWidgets
from configuration import Dialogs
from PyQt6.QtCore import QSize
from configuration.Appconfig import Appconfig
from frontEnd import ProjectExplorer
from frontEnd import TimeExplorer
from frontEnd import Workspace
from frontEnd import DockArea
from frontEnd import theme_utils
from projManagement.openProject import OpenProjectInfo
from projManagement.newProject import NewProjectInfo
from projManagement.Kicad import Kicad
from projManagement.Validation import Validation
from projManagement import Worker

# Its our main window of application.


def create_rounded_icon(path, radius_ratio=0.08):
    """Load a PNG and clip it to softly rounded corners so the colourful
    toolbar icons sit on the Aurora chrome without hard black squares."""
    pixmap = QtGui.QPixmap(path)
    if pixmap.isNull():
        return QtGui.QIcon()

    radius = int(min(pixmap.width(), pixmap.height()) * radius_ratio)

    rounded = QtGui.QPixmap(pixmap.size())
    rounded.fill(QtCore.Qt.GlobalColor.transparent)

    painter = QtGui.QPainter(rounded)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    path_obj = QtGui.QPainterPath()
    path_obj.addRoundedRect(0, 0, pixmap.width(), pixmap.height(), radius, radius)
    painter.setClipPath(path_obj)
    painter.drawPixmap(0, 0, pixmap)
    painter.end()

    return QtGui.QIcon(rounded)


class Application(QtWidgets.QMainWindow):
    """This class initializes all objects used in this file."""
    global project_name
    simulationEndSignal = QtCore.pyqtSignal(QtCore.QProcess.ExitStatus, int)

    def __init__(self, *args):
        """Initialize main Application window."""

        # Calling __init__ of super class
        QtWidgets.QMainWindow.__init__(self, *args)

        # Set slot for simulation end signal to plot simulation data
        self.simulationEndSignal.connect(self.plotSimulationData)

        #the plotFlag
        self.plotFlag = False

        # Creating require Object
        self.obj_workspace = Workspace.Workspace()
        self.obj_Mainview = MainView()
        self.obj_kicad = Kicad(self.obj_Mainview.obj_dockarea)
        self.obj_appconfig = Appconfig()
        self.obj_validation = Validation()
        # Initialize all widget
        self.setCentralWidget(self.obj_Mainview)
        self.initToolBar()
        self.initMenuAndStatus()

        self.setGeometry(self.obj_appconfig._app_xpos,
                         self.obj_appconfig._app_ypos,
                         self.obj_appconfig._app_width,
                         self.obj_appconfig._app_heigth)
        self.setWindowTitle(
            self.obj_appconfig._APPLICATION + "-" + self.obj_appconfig._VERSION
        )
        self.showMaximized()
        self.setWindowIcon(QtGui.QIcon(init_path + 'images/logo.png'))

        # Aurora micro-interactions: install hover/press glow on the main
        # window's buttons. Gated inside (no-op unless the user enabled motion
        # in prefs); dock content added later re-installs this from DockArea.
        try:
            from frontEnd.motion import install_button_motion, apply_toolbar_depth
            install_button_motion(self)
            # Static floating-depth shadow on both toolbars (always on — only
            # the animated glow is perf-gated).
            apply_toolbar_depth(self)
        except Exception:
            pass

        self.systemTrayIcon = QtWidgets.QSystemTrayIcon(self)
        self.systemTrayIcon.setIcon(QtGui.QIcon(init_path + 'images/logo.png'))
        self.systemTrayIcon.setVisible(True)

    def initToolBar(self):
        """
        This function initializes Tool Bars.
        It setups the icons, short-cuts and defining functonality for:

            - Top-tool-bar (New project, Open project, Close project, \
                Mode switch, Help option)
            - Left-tool-bar (Open Schematic, Convert KiCad to Ngspice, \
                Simuation, Model Editor, Subcircuit, NGHDL, Modelica \
                Converter, OM Optimisation)
        """
        from frontEnd.icon_paths import (
            timeline_icon, workspace_icon,
            help_icon, dev_docs_icon, settings_icon
        )

        # Top Tool bar
        self.newproj = QtGui.QAction(
            create_rounded_icon(init_path + 'images/newProject.png'),
            'New Project', self
        )
        self.newproj.setShortcut('Ctrl+N')
        self.newproj.setToolTip('New Project (Ctrl+N) — Create a new eSim project')
        self.newproj.triggered.connect(self.new_project)

        self.openproj = QtGui.QAction(
            create_rounded_icon(init_path + 'images/openProject.png'),
            'Open Project', self
        )
        self.openproj.setShortcut('Ctrl+O')
        self.openproj.setToolTip('Open Project (Ctrl+O) — Open an existing project')
        self.openproj.triggered.connect(self.open_project)

        self.closeproj = QtGui.QAction(
            create_rounded_icon(init_path + 'images/closeProject.png'),
            'Close Project', self
        )
        self.closeproj.setShortcut('Ctrl+X')
        self.closeproj.setToolTip('Close Project (Ctrl+X) — Close the active project')
        self.closeproj.triggered.connect(self.close_project)

        self.wrkspce = QtGui.QAction(
            workspace_icon(),
            'Workspace', self
        )
        self.wrkspce.setShortcut('Ctrl+W')
        self.wrkspce.setToolTip('Change Workspace (Ctrl+W) — Choose another workspace')
        self.wrkspce.triggered.connect(self.change_workspace)

        # Project Snapshots / Timeline — view & restore project backups.
        self.timeline_action = QtGui.QAction(
            timeline_icon(),
            'Timeline', self
        )
        self.timeline_action.setToolTip('Timeline — View and restore project backups')
        self.timeline_action.triggered.connect(self.show_snapshots)
        # Back-compat alias (older code referenced act_snapshots).
        self.act_snapshots = self.timeline_action

        self.helpfile = QtGui.QAction(
            help_icon(),
            'User Manual', self
        )
        self.helpfile.setShortcut('F1')
        self.helpfile.setToolTip('User Manual (F1) — Open the eSim user manual')
        self.helpfile.triggered.connect(self.help_project)

        # added devDocs logo and called functions
        self.devdocs = QtGui.QAction(
            dev_docs_icon(),
            'Developer Docs', self
        )
        self.devdocs.setShortcut('Shift+F1')
        self.devdocs.setToolTip('Developer Docs (Shift+F1) — Open eSim developer docs')
        self.devdocs.triggered.connect(self.dev_docs)

        # Preferences: Aurora theme (Dark/Light/System) + accent picker.
        # Gear icon is theme-aware; theme_utils.apply_theme also refreshes it.
        self.preferences_action = QtGui.QAction(
            settings_icon(), 'Preferences', self
        )
        self.preferences_action.setShortcut('Ctrl+,')
        self.preferences_action.setToolTip(
            'Preferences (Ctrl+,) — Configure eSim'
        )
        self.preferences_action.triggered.connect(self.open_preferences)

        # Exit / About actions live in the menu bar (built later).
        self.exit_action = QtGui.QAction('Exit', self)
        self.exit_action.setShortcut('Ctrl+Q')
        self.exit_action.setToolTip('Quit eSim (Ctrl+Q)')
        self.exit_action.setMenuRole(QtGui.QAction.MenuRole.QuitRole)
        self.exit_action.triggered.connect(self.close)

        self.about_action = QtGui.QAction('About eSim', self)
        self.about_action.setMenuRole(QtGui.QAction.MenuRole.AboutRole)
        self.about_action.triggered.connect(self.show_about)

        # --- Top toolbar: icon-only tool actions; labels live in the menu bar.
        self.topToolbar = self.addToolBar('Top Tool Bar')
        self.topToolbar.setObjectName('topToolbar')
        self.topToolbar.setMovable(True)
        self.topToolbar.setFloatable(False)
        self.topToolbar.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.topToolbar.setIconSize(QSize(28, 28))
        self.topToolbar.addAction(self.newproj)
        self.topToolbar.addAction(self.openproj)
        self.topToolbar.addAction(self.closeproj)
        self.topToolbar.addSeparator()
        self.topToolbar.addAction(self.wrkspce)
        self.topToolbar.addSeparator()
        self.topToolbar.addAction(self.timeline_action)
        self.topToolbar.addSeparator()
        self.topToolbar.addAction(self.helpfile)
        self.topToolbar.addAction(self.devdocs)
        self.topToolbar.addAction(self.preferences_action)

        # ## This part is meant for SoC Generation which is currently  ##
        # ## under development and will be will be required in future. ##
        # self.soc = QtWidgets.QToolButton(self)
        # self.soc.setText('Generate SoC')
        # self.soc.setToolTip(
        #     '<b>SPICE to Verilog Conversion</b><br>' + \
        #     '<br>The feature is under development.' + \
        #     '<br>It will be released soon.' + \
        #     '<br><br>Thank you for your patience!!!'
        # )
        # self.soc.setStyleSheet(" \
        # QWidget { border-radius: 15px; border: 1px \
        #     solid gray; padding: 10px; margin-left: 20px; } \
        # ")
        # self.soc.clicked.connect(self.showSoCRelease)
        # self.topToolbar.addWidget(self.soc)

        # Expanding spacer pushes the view controls (zoom, theme toggle) and
        # the FOSSEE logo to the right edge — the 'view controls live right'
        # convention.
        self.topToolbar.addSeparator()
        self.spacer = QtWidgets.QWidget()
        self.spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred)
        self.topToolbar.addWidget(self.spacer)

        # Zoom box: [ - ] 100% [ + ]
        self.zoom_container = QtWidgets.QWidget()
        self.zoom_container.setMinimumWidth(120)
        zoom_layout = QtWidgets.QHBoxLayout(self.zoom_container)
        zoom_layout.setContentsMargins(0, 0, 0, 0)
        zoom_layout.setSpacing(0)

        self.zoom_out_btn = QtWidgets.QToolButton()
        self.zoom_out_btn.setText(" - ")
        self.zoom_out_btn.setToolTip("Decrease Zoom (-10%)")
        self.zoom_out_btn.setProperty("cssClass", "toolbarZoom")
        self.zoom_out_btn.clicked.connect(lambda: self.change_zoom(-10))
        zoom_layout.addWidget(self.zoom_out_btn)

        self.zoom_label = QtWidgets.QLabel(" 100% ")
        self.zoom_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.zoom_label.setMinimumWidth(44)
        self.zoom_label.setProperty("cssClass", "subtle")
        self.zoom_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Minimum,
            QtWidgets.QSizePolicy.Policy.Fixed)
        zoom_layout.addWidget(self.zoom_label)

        self.zoom_in_btn = QtWidgets.QToolButton()
        self.zoom_in_btn.setText(" + ")
        self.zoom_in_btn.setToolTip("Increase Zoom (+10%)")
        self.zoom_in_btn.setProperty("cssClass", "toolbarZoom")
        self.zoom_in_btn.clicked.connect(lambda: self.change_zoom(10))
        zoom_layout.addWidget(self.zoom_in_btn)

        self.topToolbar.addWidget(self.zoom_container)

        # Quick light/dark toggle on the right of the action bar.
        self.theme_toggle_btn = QtWidgets.QToolButton()
        self.theme_toggle_btn.setObjectName("themeToggleBtn")
        self.theme_toggle_btn.setText("◐")
        self.theme_toggle_btn.setProperty("cssClass", "toolbarZoom")
        self.theme_toggle_btn.setToolTip("Toggle light / dark theme")
        self.theme_toggle_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.theme_toggle_btn.clicked.connect(self._toggle_theme)
        self.topToolbar.addWidget(self.theme_toggle_btn)

        # Init zoom label from saved preference.
        from frontEnd.theme_utils import get_preferences
        u_home = os.path.join('library', 'config') if os.name == 'nt' \
            else os.path.expanduser('~')
        zp = get_preferences(u_home).get("zoom_level", 100)
        self.zoom_label.setText(f" {zp}% ")

        # FOSSEE logo kept top-right of the action bar (eSim brand).
        self.logo = QtWidgets.QLabel()
        self.logopic = QtGui.QPixmap(
            os.path.join(
                os.path.abspath(''), init_path + 'images', 'fosseeLogo.png'
            ))
        self.logopic = self.logopic.scaled(
            QSize(150, 150), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation)
        self.logo.setPixmap(self.logopic)
        self.logo.setStyleSheet("padding:0 15px 0 0;")
        self.topToolbar.addWidget(self.logo)

        # Left Tool bar Action Widget
        self.kicad = QtGui.QAction(
            create_rounded_icon(init_path + 'images/kicad.png'),
            'Open Schematic', self
        )
        self.kicad.setShortcut("Ctrl+K")
        self.kicad.setToolTip(
            "Open Schematic (Ctrl+K) - Design your circuit in KiCad")
        self.kicad.triggered.connect(self.obj_kicad.openSchematic)

        self.conversion = QtGui.QAction(
            create_rounded_icon(init_path + 'images/ki-ng.png'),
            'Convert to Ngspice', self
        )
        self.conversion.setShortcut("Ctrl+Alt+C")
        self.conversion.setToolTip(
            "Convert to Ngspice - Generate Ngspice netlist")
        self.conversion.triggered.connect(self.obj_kicad.openKicadToNgspice)

        self.ngspice = QtGui.QAction(
            create_rounded_icon(init_path + 'images/ngspice.png'),
            'Simulate', self
        )
        self.ngspice.setShortcut("Ctrl+G")
        self.ngspice.setToolTip("Simulate (Ctrl+G) - Run circuit simulation")
        self.ngspice.triggered.connect(self.plotFlagPopBox)

        self.model = QtGui.QAction(
            create_rounded_icon(init_path + 'images/model.png'),
            'Model Editor', self
        )
        self.model.setShortcut("Ctrl+M")
        self.model.setToolTip(
            "Model Editor (Ctrl+M) - Create or edit SPICE models")
        self.model.triggered.connect(self.open_modelEditor)

        self.subcircuit = QtGui.QAction(
            create_rounded_icon(init_path + 'images/subckt.png'),
            'Subcircuit', self
        )
        self.subcircuit.setShortcut("Ctrl+B")
        self.subcircuit.setToolTip(
            "Subcircuit (Ctrl+B) - Build reusable subcircuits")
        self.subcircuit.triggered.connect(self.open_subcircuit)

        # NGHDL is no longer a standalone toolbar button: it now lives as a
        # tab inside the Makerchip dock (Makerchip / NgVeri / NGHDL), so model
        # creation for both Verilog and VHDL is in one place.
        self.makerchip = QtGui.QAction(
            create_rounded_icon(init_path + 'images/makerchip.png'),
            'Model Creation (Verilog / VHDL)', self
        )
        self.makerchip.setToolTip(
            "Model Creation - Verilog / VHDL via Makerchip, NgVeri & NGHDL")
        self.makerchip.triggered.connect(self.open_makerchip)

        self.omedit = QtGui.QAction(
            create_rounded_icon(init_path + 'images/omedit.png'),
            'Modelica Converter', self
        )
        self.omedit.setToolTip(
            "Modelica Converter - Convert to Modelica format")
        self.omedit.triggered.connect(self.open_OMedit)

        self.omoptim = QtGui.QAction(
            create_rounded_icon(init_path + 'images/omoptim.png'),
            'OM Optimisation', self
        )
        self.omoptim.setToolTip("OM Optimisation - Run OpenModelica optimizer")
        self.omoptim.triggered.connect(self.open_OMoptim)

        self.conToeSim = QtGui.QAction(
            create_rounded_icon(init_path + 'images/icon.png'),
            'Schematic Converter', self
        )
        self.conToeSim.setToolTip(
            "Schematic Converter - Import PSpice/LTspice files")
        self.conToeSim.triggered.connect(self.open_conToeSim)

        # Adding Action Widget to tool bar — grouped into labelled clusters so
        # a new user can parse the rail at a glance instead of staring at ten
        # cryptic icons.
        self.lefttoolbar = QtWidgets.QToolBar('Left ToolBar')
        self.lefttoolbar.setObjectName('leftToolBar')
        self.lefttoolbar.setMovable(True)
        self.addToolBar(QtCore.Qt.ToolBarArea.LeftToolBarArea, self.lefttoolbar)

        def _rail_caption(text):
            lbl = QtWidgets.QLabel(text)
            lbl.setProperty("cssClass", "railCaption")
            lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            return lbl

        self.lefttoolbar.addWidget(_rail_caption("DESIGN"))
        self.lefttoolbar.addAction(self.kicad)
        self.lefttoolbar.addAction(self.conversion)
        self.lefttoolbar.addSeparator()
        self.lefttoolbar.addWidget(_rail_caption("SIMULATE"))
        self.lefttoolbar.addAction(self.ngspice)
        self.lefttoolbar.addAction(self.makerchip)
        self.lefttoolbar.addSeparator()
        self.lefttoolbar.addWidget(_rail_caption("MODEL"))
        self.lefttoolbar.addAction(self.model)
        self.lefttoolbar.addAction(self.subcircuit)
        self.lefttoolbar.addSeparator()
        self.lefttoolbar.addWidget(_rail_caption("CONVERT"))
        self.lefttoolbar.addAction(self.omedit)
        self.lefttoolbar.addAction(self.omoptim)
        self.lefttoolbar.addAction(self.conToeSim)
        self.lefttoolbar.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.lefttoolbar.setIconSize(QSize(40, 40))

        # Build the menu bar now that every toolbar action exists.
        self._build_menu_bar()

    def _build_menu_bar(self):
        """Build a proper menu bar with File/Edit/View/Tools/Help, wired to
        the same QAction objects the toolbar uses (our handlers)."""
        bar = self.menuBar()

        # ----- File -----
        file_menu = bar.addMenu('&File')
        file_menu.addAction(self.newproj)
        file_menu.addAction(self.openproj)
        file_menu.addAction(self.closeproj)
        file_menu.addSeparator()
        file_menu.addAction(self.wrkspce)
        file_menu.addSeparator()
        self.recent_projects_menu = file_menu.addMenu('Recent Projects')
        self._refresh_recent_projects_menu()
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        # ----- Edit -----
        edit_menu = bar.addMenu('&Edit')
        undo_action = QtGui.QAction('Undo', self)
        undo_action.setShortcut('Ctrl+Z')
        edit_menu.addAction(undo_action)
        redo_action = QtGui.QAction('Redo', self)
        redo_action.setShortcut('Ctrl+Shift+Z')
        edit_menu.addAction(redo_action)
        edit_menu.addSeparator()
        cut_action = QtGui.QAction('Cut', self)
        # No Ctrl+X here: it belongs to Close Project; a duplicate would make
        # the shortcut ambiguous and break both.
        edit_menu.addAction(cut_action)
        copy_action = QtGui.QAction('Copy', self)
        copy_action.setShortcut('Ctrl+C')
        edit_menu.addAction(copy_action)
        paste_action = QtGui.QAction('Paste', self)
        paste_action.setShortcut('Ctrl+V')
        edit_menu.addAction(paste_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.preferences_action)

        # ----- View -----
        view_menu = bar.addMenu('&View')
        fullscreen_action = QtGui.QAction('Toggle Fullscreen', self)
        fullscreen_action.setShortcut('F11')
        fullscreen_action.setCheckable(True)
        fullscreen_action.triggered.connect(self._toggle_fullscreen)
        view_menu.addAction(fullscreen_action)
        view_menu.addSeparator()
        project_explorer_action = QtGui.QAction('Project Explorer', self)
        project_explorer_action.setCheckable(True)
        project_explorer_action.setChecked(True)
        project_explorer_action.triggered.connect(
            lambda: self.obj_Mainview.obj_projectExplorer.setVisible(
                project_explorer_action.isChecked()
            )
        )
        view_menu.addAction(project_explorer_action)
        console_action = QtGui.QAction('Console', self)
        console_action.setCheckable(True)
        console_action.setChecked(False)
        console_action.triggered.connect(
            lambda: self.btn_log.setChecked(console_action.isChecked())
        )
        view_menu.addAction(console_action)

        # ----- Tools -----
        tools_menu = bar.addMenu('&Tools')
        tools_menu.addAction(self.kicad)
        tools_menu.addAction(self.conversion)
        tools_menu.addAction(self.ngspice)
        tools_menu.addSeparator()
        tools_menu.addAction(self.model)
        tools_menu.addAction(self.subcircuit)
        tools_menu.addSeparator()
        tools_menu.addAction(self.makerchip)
        tools_menu.addSeparator()
        tools_menu.addAction(self.omedit)
        tools_menu.addAction(self.omoptim)
        tools_menu.addSeparator()
        tools_menu.addAction(self.conToeSim)

        # ----- Help -----
        help_menu = bar.addMenu('&Help')
        help_menu.addAction(self.helpfile)
        help_menu.addAction(self.devdocs)
        help_menu.addSeparator()
        help_menu.addAction(self.about_action)

        # Round the corners of every menu-bar dropdown (no black squares).
        try:
            from frontEnd.motion import make_menu_rounded
            for _m in bar.findChildren(QtWidgets.QMenu):
                make_menu_rounded(_m)
        except Exception:
            pass

    def _refresh_recent_projects_menu(self):
        """Populate the Recent Projects submenu from project_explorer."""
        if not hasattr(self, 'recent_projects_menu'):
            return
        self.recent_projects_menu.clear()
        recent = list(self.obj_appconfig.project_explorer.keys())
        recent = [p for p in recent if os.path.isdir(p)][:8]
        if not recent:
            empty = QtGui.QAction('(none)', self)
            empty.setEnabled(False)
            self.recent_projects_menu.addAction(empty)
            return
        for path in recent:
            action = QtGui.QAction(os.path.basename(path) or path, self)
            action.setToolTip(path)
            action.triggered.connect(
                lambda checked=False, p=path: self._open_recent_project(p)
            )
            self.recent_projects_menu.addAction(action)

    def _open_recent_project(self, path):
        """Open a recent project by path."""
        try:
            open_proj = OpenProjectInfo()
            directory, filelist = open_proj.body(path)
            if directory and filelist:
                self.obj_Mainview.obj_projectExplorer.addTreeNode(
                    directory, filelist
                )
                self.obj_appconfig.set_current_project(directory)
                project_name = self.obj_appconfig.get_proj_stem()
                self.obj_Mainview.obj_timeExplorer.load_snapshots(project_name)
                self.obj_appconfig.save_current_project()
        except Exception as e:
            from frontEnd.dialogs import show_error
            show_error(self, "Open Project", f"Could not open {path}:\n{e}")
            self.obj_appconfig.print_warning(
                f"Recent project open failed: {e}")

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showMaximized()
        else:
            self.showFullScreen()

    def _toggle_theme(self):
        """Flip between forced Light and Dark and persist the choice."""
        import json
        from frontEnd.theme_utils import get_preferences
        user_home = os.path.join('library', 'config') if os.name == 'nt' \
            else os.path.expanduser('~')
        prefs = get_preferences(user_home)
        cur = prefs.get("theme_mode", "System")
        if cur == "Dark":
            new_mode = "Light"
        elif cur == "Light":
            new_mode = "Dark"
        else:
            scheme = QtGui.QGuiApplication.styleHints().colorScheme()
            new_mode = "Light" \
                if scheme == QtCore.Qt.ColorScheme.Dark else "Dark"
        prefs["theme_mode"] = new_mode
        path = os.path.join(user_home, ".esim", "preferences.json")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(prefs, f)
        except Exception:
            pass
        app = QtWidgets.QApplication.instance()
        fn = getattr(app, "apply_theme", None)
        if callable(fn):
            fn()

    def change_zoom(self, delta):
        """Adjust the global UI zoom (50–300%); persists + re-applies theme."""
        import json
        from frontEnd.theme_utils import get_preferences, apply_theme
        user_home = os.path.join('library', 'config') if os.name == 'nt' \
            else os.path.expanduser('~')
        prefs = get_preferences(user_home)
        current_zoom = prefs.get("zoom_level", 100)
        new_zoom = max(50, min(300, current_zoom + delta))
        if new_zoom != current_zoom:
            prefs["zoom_level"] = new_zoom
            path = os.path.join(user_home, ".esim", "preferences.json")
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    json.dump(prefs, f)
            except Exception:
                pass
            if hasattr(self, 'zoom_label'):
                self.zoom_label.setText(f" {new_zoom}% ")
            apply_theme(QtWidgets.QApplication.instance())
            if hasattr(self, 'topToolbar'):
                s = int(28 * (new_zoom / 100.0))
                self.topToolbar.setIconSize(QtCore.QSize(s, s))
            if hasattr(self, 'lefttoolbar'):
                s = int(40 * (new_zoom / 100.0))
                self.lefttoolbar.setIconSize(QtCore.QSize(s, s))

    def show_about(self):
        """Show the About eSim dialog with gradient-rich premium styling."""
        from frontEnd.dialogs import show_about_dialog
        show_about_dialog(self)

    def createPopupMenu(self):
        """Qt builds the toolbar/dock right-click menu internally and shows it
        immediately, so round its corners here before it is shown."""
        menu = super().createPopupMenu()
        if menu is not None:
            try:
                from frontEnd.motion import make_menu_rounded
                make_menu_rounded(menu)
            except Exception:
                pass
        return menu

    def _set_sim_status(self, state):
        """Tint the status-bar simulation dot: idle/running/ok/failed."""
        colors = {
            "idle": "#5F728D", "running": "#FACC15",
            "ok": "#42E6A4", "failed": "#FB7185",
        }
        c = colors.get(state, "#5F728D")
        if hasattr(self, 'sim_status_dot'):
            self.sim_status_dot.setStyleSheet(
                "color: %s; font-size: 13px; padding: 0 6px;" % c)

    def initMenuAndStatus(self):
        """No menu bar -- eSim is icon-driven. The snapshots panel is a
        top-toolbar icon (added in initToolBar), and the full console log
        toggles from a status-bar button while the status bar shows the latest
        log line."""
        # Status bar: mirrors the newest print_info/warning/error line; the
        # button toggles the full console log panel.
        bar = self.statusBar()
        self.obj_appconfig.__class__.statusbar = bar
        self.btn_log = QtWidgets.QToolButton()
        self.btn_log.setText('Console Log  ▴')
        self.btn_log.setCheckable(True)
        self.btn_log.setAutoRaise(True)
        self.btn_log.setToolTip('Show / hide the full console log')
        self.btn_log.toggled.connect(self._toggle_console_btn)
        bar.addPermanentWidget(self.btn_log)

        # Right-zone simulation-status dot: a tiny premium affordance —
        # grey idle, amber running, green ok, red failed.
        self.sim_status_dot = QtWidgets.QLabel("●")
        self.sim_status_dot.setObjectName("simStatusDot")
        self.sim_status_dot.setToolTip("Simulation status")
        self._set_sim_status("idle")
        bar.addPermanentWidget(self.sim_status_dot)
        bar.showMessage('eSim ready')

    def _toggle_console_btn(self, show):
        self.obj_Mainview.toggle_console(show)
        self.btn_log.setText('Console Log  ▾' if show else 'Console Log  ▴')

    def show_snapshots(self):
        """Open the Project Snapshots (timeline) panel on demand. It used to
        occupy the left face; now it is a non-modal dialog parented to the main
        window (so it can never hide behind it)."""
        te = self.obj_Mainview.obj_timeExplorer
        dlg = getattr(self, '_snap_dlg', None)
        if dlg is None:
            dlg = QtWidgets.QDialog(self)
            dlg.setWindowTitle('Project Snapshots')
            lay = QtWidgets.QVBoxLayout(dlg)
            lay.setContentsMargins(0, 0, 0, 0)
            dlg.resize(360, 480)
            self._snap_dlg = dlg
        # (Re)mount the kept TimeExplorer instance into the dialog.
        dlg.layout().addWidget(te)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def plotFlagPopBox(self):
        """This function displays a pop-up box with message- Do you want Ngspice plots? and oprions Yes and NO.
        
        If the user clicks on Yes, both the NgSpice and python plots are displayed and if No is clicked then only the python plots."""

        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle("Ngspice Plots")
        msg_box.setText("Do you want Ngspice plots?")
        
        yes_button = msg_box.addButton("Yes", QtWidgets.QMessageBox.ButtonRole.YesRole)
        no_button = msg_box.addButton("No", QtWidgets.QMessageBox.ButtonRole.NoRole)

        msg_box.exec()

        if msg_box.clickedButton() == yes_button:
            self.plotFlag = True  
        else:
            self.plotFlag = False  

        self.open_ngspice()

    def closeEvent(self, event):
        '''
        This function closes the ongoing program (process).
        When exit button is pressed a Message box pops out with \
        exit message and buttons 'Yes', 'No'.

            1. If 'Yes' is pressed:
                - check that program (process) in procThread_list \
                  (a list made in Appconfig.py):

                    - if available it terminates that program.
                    - if the program (process) is not available, \
                      then check it in process_obj (a list made in \
                      Appconfig.py) and if found, it closes the program.

            2. If 'No' is pressed:
                - the program just continues as it was doing earlier.
        '''
        exit_msg = "Are you sure you want to exit the program?"
        exit_msg += " All unsaved data will be lost."
        reply = Dialogs.question(
            self, 'Message', exit_msg, QtWidgets.QMessageBox.StandardButton.Yes,
            QtWidgets.QMessageBox.StandardButton.No
        )

        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            for proc in self.obj_appconfig.procThread_list:
                try:
                    proc.terminate()
                except BaseException:
                    pass
            try:
                for process_object in self.obj_appconfig.process_obj:
                    try:
                        process_object.close()
                    except BaseException:
                        pass
            except BaseException:
                pass

            # Check if "Open project" and "New project" window is open.
            # If yes, just close it when application is closed.
            try:
                self.project.close()
            except BaseException:
                pass
            event.accept()
            self.systemTrayIcon.showMessage('Exit', 'eSim is Closed.')

        elif reply == QtWidgets.QMessageBox.StandardButton.No:
            event.ignore()

    def new_project(self):
        """This function call New Project Info class."""
        text, ok = QtWidgets.QInputDialog.getText(
            self, 'New Project Info', 'Enter Project Name:'
        )
        updated = False

        if ok:
            self.projname = (str(text))
            self.project = NewProjectInfo()
            directory, filelist = self.project.createProject(self.projname)

            if directory and filelist:
                self.obj_Mainview.obj_projectExplorer.addTreeNode(
                    directory, filelist
                )
                self.obj_appconfig.set_current_project(directory)
                project_name = self.obj_appconfig.get_proj_stem()
                self.obj_Mainview.obj_timeExplorer.load_snapshots(project_name)
                self.obj_appconfig.save_current_project()
                updated = True

        if not updated:
            print("No new project created")
            self.obj_appconfig.print_info('No new project created')
            try:
                self.obj_appconfig.print_info(
                    'Current project is : ' +
                    self.obj_appconfig.current_project["ProjectName"]
                )
            except BaseException:
                pass

    def open_project(self):
        """This project call Open Project Info class."""
        print("Function : Open Project")
        self.project = OpenProjectInfo()
        try:
            directory, filelist = self.project.body()
            if not directory:
                return
            self.obj_appconfig.set_current_project(directory)
            self.obj_Mainview.obj_projectExplorer.addTreeNode(
                directory, filelist)
            project_name = self.obj_appconfig.get_proj_stem()
            self.obj_Mainview.obj_timeExplorer.load_snapshots(project_name)
            self.obj_appconfig.save_current_project()
        except BaseException:
            pass

    def close_project(self):
        """
        This function closes the saved project.
        It first checks whether project (file) is present in list.

            - If present:
                - it first kills that process-id.
                - closes that file.
                - Shows message "Current project <path_to_file> is closed"

            - If not present: pass
        """
        print("Function : Close Project")
        current_project = self.obj_appconfig.current_project['ProjectName']
        if current_project is None:
            pass
        else:
            temp = self.obj_appconfig.current_project['ProjectName']
            for pid in self.obj_appconfig.proc_dict.get(temp, []):
                try:
                    os.kill(pid, 9)
                except BaseException:
                    pass
            self.obj_Mainview.obj_dockarea.closeDock()
            closed_stem = self.obj_appconfig.get_proj_stem() \
                or os.path.basename(current_project)
            self.obj_appconfig.set_current_project(None)
            self.obj_appconfig.save_current_project()
            self.systemTrayIcon.showMessage(
                'Close', 'Current project ' +
                closed_stem + ' is Closed.'
            )

    def change_workspace(self):
        """
        This function call changes Workspace
        """
        print("Function : Change Workspace")
        self.obj_workspace.returnWhetherClickedOrNot(self)
        self.hide()
        self.obj_workspace.show()

    def help_project(self):
        """
        This function opens usermanual in dockarea.
            - It prints the message ""Function : Help""
            - Uses print_info() method of class Appconfig
              from Configuration/Appconfig.py file.
            - Call method usermanual() from ./DockArea.py.
        """
        print("Function : Help")
        self.obj_appconfig.print_info('Help is called')
        print("Current Project is : ", self.obj_appconfig.current_project)
        self.obj_Mainview.obj_dockarea.usermanual()

    def dev_docs(self):
        """
        This function guides the user to readthedocs website for the developer docs
        """
        print("Function : DevDocs")
        self.obj_appconfig.print_info('DevDocs is called')
        print("Current Project is : ", self.obj_appconfig.current_project)
        webbrowser.open("https://esim.readthedocs.io/en/latest/index.html")

    def open_preferences(self):
        """Open the Aurora Preferences dialog (theme + accent picker). The
        dialog live-applies via app.apply_theme(), wired in main()."""
        from frontEnd.PreferencesDialog import PreferencesDialog
        dlg = PreferencesDialog(self)
        dlg.exec()

    @QtCore.pyqtSlot(QtCore.QProcess.ExitStatus, int)
    def plotSimulationData(self, exitStatus, exitCode):
        """Enables interaction for new simulation and
           displays the plotter dock where graphs can be plotted.
        """
        self.ngspice.setEnabled(True)
        self.conversion.setEnabled(True)
        self.closeproj.setEnabled(True)
        self.wrkspce.setEnabled(True)

        if exitStatus == QtCore.QProcess.ExitStatus.NormalExit and exitCode == 0:
            self._set_sim_status("ok")
            try:
                self.obj_Mainview.obj_dockarea.plottingEditor()
            except Exception as e:
                self._set_sim_status("failed")
                self.msg = Dialogs.make_error_message(self)
                self.msg.setModal(True)
                self.msg.setWindowTitle("Error Message")
                self.msg.showMessage(
                    'Data could not be plotted. Please try again.'
                )
                self.msg.exec()
                print("Exception Message:", str(e), traceback.format_exc())
                self.obj_appconfig.print_error('Exception Message : '
                                               + str(e))
        else:
            self._set_sim_status("failed")

    def open_ngspice(self):
        """This Function execute ngspice on current project."""
        # Flush any unsaved edits in the code editor first, so the
        # simulation reads the netlist the user is actually looking at.
        try:
            from codeEditor import EditorWindow
            EditorWindow.flush_all_dirty()
        except Exception:
            pass

        projDir = self.obj_appconfig.current_project["ProjectName"]

        if projDir is not None:
            projName = self.obj_appconfig.get_proj_stem()
            ngspiceNetlist = os.path.join(projDir, projName + ".cir.out")

            if not os.path.isfile(ngspiceNetlist):
                print(
                    "Netlist file (*.cir.out) not found."
                )
                self.msg = Dialogs.make_error_message(self)
                self.msg.setModal(True)
                self.msg.setWindowTitle("Error Message")
                self.msg.showMessage(
                    'Netlist (*.cir.out) not found.'
                )
                self.msg.exec()
                return

            self.obj_Mainview.obj_dockarea.ngspiceEditor(
                projName, ngspiceNetlist, self.simulationEndSignal, self.plotFlag)

            self._set_sim_status("running")
            self.ngspice.setEnabled(False)
            self.conversion.setEnabled(False)
            self.closeproj.setEnabled(False)
            self.wrkspce.setEnabled(False)

        else:
            self.msg = Dialogs.make_error_message(self)
            self.msg.setModal(True)
            self.msg.setWindowTitle("Error Message")
            self.msg.showMessage(
                'Please select the project first.'
                ' You can either create new project or open existing project'
            )
            self.msg.exec()

    def open_subcircuit(self):
        """
        This function opens 'subcircuit' option in left-tool-bar.
        When 'subcircuit' icon is clicked wich is present in
        left-tool-bar of main page:

            - Meassge shown on screen "Subcircuit editor is called".
            - 'subcircuiteditor()' function is called using object
              'obj_dockarea' of class 'Mainview'.
        """
        print("Function : Subcircuit editor")
        self.obj_appconfig.print_info('Subcircuit editor is called')
        self.obj_Mainview.obj_dockarea.subcircuiteditor()

    def open_makerchip(self):
        """
        This function opens 'subcircuit' option in left-tool-bar.
        When 'subcircuit' icon is clicked wich is present in
        left-tool-bar of main page:

            - Meassge shown on screen "Subcircuit editor is called".
            - 'subcircuiteditor()' function is called using object
              'obj_dockarea' of class 'Mainview'.
        """
        print("Function : Makerchip and Verilator to Ngspice Converter")
        self.obj_appconfig.print_info('Makerchip is called')
        self.obj_Mainview.obj_dockarea.makerchip()

    def open_modelEditor(self):
        """
        This function opens model editor option in left-tool-bar.
        When model editor icon is clicked which is present in
        left-tool-bar of main page:

            - Meassge shown on screen "Model editor is called".
            - 'modeleditor()' function is called using object
              'obj_dockarea' of class 'Mainview'.
        """
        print("Function : Model editor")
        self.obj_appconfig.print_info('Model editor is called')
        self.obj_Mainview.obj_dockarea.modelEditor()

    def open_OMedit(self):
        """
        This function calls ngspice to OMEdit converter and then launch OMEdit.
        """
        self.obj_appconfig.print_info('OMEdit is called')
        self.projDir = self.obj_appconfig.current_project["ProjectName"]

        if self.projDir is not None:
            if self.obj_validation.validateCirOut(self.projDir):
                self.projName = self.obj_appconfig.get_proj_stem()
                self.ngspiceNetlist = os.path.join(
                    self.projDir, self.projName + ".cir.out"
                )
                self.modelicaNetlist = os.path.join(
                    self.projDir, self.projName + ".mo"
                )

                """
                try:
                    # Creating a command for Ngspice to Modelica converter
                    self.cmd1 = "
                        python3 ../ngspicetoModelica/NgspicetoModelica.py "\
                            + self.ngspiceNetlist
                    self.obj_workThread1 = Worker.WorkerThread(self.cmd1)
                    self.obj_workThread1.start()
                    if self.obj_validation.validateTool("OMEdit"):
                        # Creating command to run OMEdit
                        self.cmd2 = "OMEdit "+self.modelicaNetlist
                        self.obj_workThread2 = Worker.WorkerThread(self.cmd2)
                        self.obj_workThread2.start()
                    else:
                        self.msg = Dialogs.make_message_box(self)
                        self.msgContent = "There was an error while
                            opening OMEdit.<br/>\
                        Please make sure OpenModelica is installed in your\
                            system. <br/>\
                        To install it on Linux : Go to\
                            <a href=https://www.openmodelica.org/download/\
                                download-linux>OpenModelica Linux</a> and  \
                                    install nigthly build release.<br/>\
                        To install it on Windows : Go to\
                         <a href=https://www.openmodelica.org/download/\
                        download-windows>OpenModelica Windows</a>\
                         and install latest version.<br/>"
                        self.msg.setTextFormat(QtCore.Qt.TextFormat.RichText)
                        self.msg.setText(self.msgContent)
                        self.msg.setWindowTitle("Missing OpenModelica")
                        self.obj_appconfig.print_info(self.msgContent)
                        self.msg.exec()

                except Exception as e:
                    self.msg = Dialogs.make_error_message(self)
                    self.msg.setModal(True)
                    self.msg.setWindowTitle(
                        "Ngspice to Modelica conversion error")
                    self.msg.showMessage(
                        'Unable to convert NgSpice netlist to\
                            Modelica netlist :'+str(e))
                    self.msg.exec()
                    self.obj_appconfig.print_error(str(e))
                """

                self.obj_Mainview.obj_dockarea.modelicaEditor(self.projDir)

            else:
                self.msg = Dialogs.make_error_message(self)
                self.msg.setModal(True)
                self.msg.setWindowTitle("Missing Ngspice Netlist")
                self.msg.showMessage(
                    'Current project does not contain any Ngspice file. ' +
                    'Please create Ngspice file with extension .cir.out'
                )
                self.msg.exec()
        else:
            self.msg = Dialogs.make_error_message(self)
            self.msg.setModal(True)
            self.msg.setWindowTitle("Error Message")
            self.msg.showMessage(
                'Please select the project first. You can either ' +
                'create a new project or open an existing project'
            )
            self.msg.exec()

    def open_OMoptim(self):
        """
        This function uses validateTool() method from Validation.py:

            - If 'OMOptim' is present in executables list then
              it passes command 'OMOptim' to WorkerThread class of Worker.py
            - If 'OMOptim' is not present, then it shows error message with
              link to download it on Linux and Windows.
        """
        print("Function : OMOptim")
        self.obj_appconfig.print_info('OMOptim is called')
        # Check if OMOptim is installed
        if self.obj_validation.validateTool("OMOptim"):
            # Creating a command to run
            self.cmd = "OMOptim"
            self.obj_workThread = Worker.WorkerThread(self.cmd)
            self.obj_workThread.start()
        else:
            self.msg = Dialogs.make_message_box(self)
            self.msgContent = (
                "There was an error while opening OMOptim.<br/>"
                "Please make sure OpenModelica is installed in your"
                " system.<br/>"
                "To install it on Linux : Go to <a href="
                "https://www.openmodelica.org/download/download-linux"
                ">OpenModelica Linux</a> and install nightly build"
                " release.<br/>"
                "To install it on Windows : Go to <a href="
                "https://www.openmodelica.org/download/download-windows"
                ">OpenModelica Windows</a> and install latest version.<br/>"
            )
            self.msg.setTextFormat(QtCore.Qt.TextFormat.RichText)
            self.msg.setText(self.msgContent)
            self.msg.setWindowTitle("Error Message")
            self.obj_appconfig.print_info(self.msgContent)
            self.msg.exec()

    def open_conToeSim(self):
        print("Function : Schematic converter")
        self.obj_appconfig.print_info('Schematic converter is called')
        self.obj_Mainview.obj_dockarea.eSimConverter()

# This class initialize the Main View of Application
class MainView(QtWidgets.QWidget):
    """
    This class defines whole view and style of main page:

        - Position of tool bars:
            - Top tool bar.
            - Left tool bar.
        - Project explorer Area.
        - Dock area.
        - Console area.
    """

    def __init__(self, *args):
        # call init method of superclass
        QtWidgets.QWidget.__init__(self, *args)

        self.obj_appconfig = Appconfig()

        self.leftSplit = QtWidgets.QSplitter()
        self.middleSplit = QtWidgets.QSplitter()

        self.mainLayout = QtWidgets.QVBoxLayout()
        # Intermediate Widget
        self.middleContainer = QtWidgets.QWidget()
        self.middleContainerLayout = QtWidgets.QVBoxLayout()

        # Area to be included in MainView
        self.noteArea = QtWidgets.QTextEdit()
        self.noteArea.setReadOnly(True)

        # Set explicit scrollbar policy
        self.noteArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.noteArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.obj_appconfig.noteArea['Note'] = self.noteArea
        self.obj_appconfig.noteArea['Note'].append(
            '        eSim Started......')
        self.obj_appconfig.noteArea['Note'].append('Project Selected : None')
        self.obj_appconfig.noteArea['Note'].append('\n')

        # Enhanced CSS with proper scrollbar styling
        self.noteArea.setObjectName("mainNoteConsole")

        self.obj_dockarea = DockArea.DockArea()
        self.obj_projectExplorer = ProjectExplorer.ProjectExplorer()
        self.obj_timeExplorer = TimeExplorer.TimeExplorer()
        self.obj_projectExplorer.set_time_explorer(self.obj_timeExplorer)

        # Adding content to vertical middle Split.
        self.middleSplit.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.middleSplit.addWidget(self.obj_dockarea)
        self.middleSplit.addWidget(self.noteArea)

        # Adding middle split to Middle Container Widget
        self.middleContainerLayout.addWidget(self.middleSplit)
        self.middleContainer.setLayout(self.middleContainerLayout)

        # Adding content of left split. The TimeExplorer (snapshots) used to
        # sit here under the project tree; it now lives in the View menu
        # (Application.show_snapshots), so the project tree gets the full
        # column. The instance is still created above and kept loaded.
        self.leftPanel = QtWidgets.QVBoxLayout()
        self.leftPanelWidget = QtWidgets.QWidget()
        self.leftPanel.addWidget(self.obj_projectExplorer)
        self.leftPanelWidget.setLayout(self.leftPanel)
        self.leftSplit.addWidget(self.leftPanelWidget)
        self.leftSplit.addWidget(self.middleContainer)

        # Adding to main Layout
        self.mainLayout.addWidget(self.leftSplit)
        self.leftSplit.setSizes([int(self.width() / 4.5), self.height()])
        # Console starts collapsed: the dock area owns the full work height and
        # the status bar carries the latest message. Expand on demand via the
        # View menu / status-bar log button (toggle_console).
        self.collapse_console_area()
        self.setLayout(self.mainLayout)

    def collapse_console_area(self):
        """Collapse the console panel; the dock area takes the full height."""
        total = sum(self.middleSplit.sizes()) or self.height()
        self.middleSplit.setSizes([total, 0])

    def restore_console_area(self):
        """Expand the console panel to ~28% of the work-area height."""
        total = sum(self.middleSplit.sizes()) or self.height()
        console = int(total * 0.28)
        self.middleSplit.setSizes([total - console, console])

    def is_console_visible(self):
        sizes = self.middleSplit.sizes()
        return len(sizes) > 1 and sizes[1] > 4

    def toggle_console(self, show):
        """Show/hide the full console log panel (View menu / status bar)."""
        if show:
            self.restore_console_area()
        else:
            self.collapse_console_area()


# It is main function of the module and starts the application
def main(args):
    """
    The splash screen opened at the starting of screen is performed
    by this function.
    """
    print("Starting eSim......")
    # Set non-native dialogs globally
    # NOTE: AA_DontUseNativeDialogs removed in Qt6.
    # Native dialog behavior is now controlled per-dialog via QFileDialog.Option.
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
        QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QtWidgets.QApplication(args)
    app.setApplicationName("eSim")

    # Aurora design system: attach a bound apply_theme to the app instance so
    # the Preferences dialog can live-apply, theme once before widgets build,
    # and follow OS light/dark changes. Guarded so theming cannot block startup.
    try:
        def _apply_theme(*_args):
            theme_utils.apply_theme(app)
        app.apply_theme = _apply_theme
        _apply_theme()
        QtGui.QGuiApplication.styleHints().colorSchemeChanged.connect(_apply_theme)
        # Bundled Inter font for the Aurora type scale (QSS has fallbacks).
        font_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'images', 'fonts',
            'Inter-VariableFont_slnt,wght.ttf'
        )
        if os.path.exists(font_path):
            QtGui.QFontDatabase.addApplicationFont(font_path)
    except Exception as e:
        print("Theme load failed, continuing unthemed:", str(e))

    # App-wide Aurora polish: translucent menus (so the QSS rounded corners are
    # genuine, not square) + a Show-time effect refresh so drop-shadows never
    # render stale after a tab/page switch or maximize. Static; guarded so they
    # cannot block startup.
    try:
        from frontEnd.motion import install_popup_motion, install_effect_refresh
        install_popup_motion(app)
        install_effect_refresh(app)
    except Exception:
        pass

    appView = Application()
    last_project_path = appView.obj_appconfig.load_last_project()
    if last_project_path:
        try:
            open_proj = OpenProjectInfo()
            directory, filelist = open_proj.body(last_project_path)
            if directory:
                appView.obj_Mainview.obj_projectExplorer.addTreeNode(
                    directory, filelist)
        except Exception as e:
            print("Could not restore last project:", str(e))
    appView.obj_Mainview.obj_timeExplorer.load_last_snapshots()
    appView.hide()

    splash_pix = QtGui.QPixmap(init_path + 'images/splash_screen_esim.png')
    splash_pix = splash_pix.scaledToWidth(
        int(splash_pix.width() * 0.8),
        QtCore.Qt.TransformationMode.SmoothTransformation)

    # Proportional rounded mask cuts the heavy black splash corners.
    radius = int(min(splash_pix.width(), splash_pix.height()) * 0.10)
    rounded_splash = QtGui.QPixmap(splash_pix.size())
    rounded_splash.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(rounded_splash)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    path_obj = QtGui.QPainterPath()
    path_obj.addRoundedRect(
        0, 0, splash_pix.width(), splash_pix.height(), radius, radius)
    painter.setClipPath(path_obj)
    painter.drawPixmap(0, 0, splash_pix)
    painter.end()

    class FadingSplash(QtWidgets.QSplashScreen):
        def __init__(self, pixmap):
            transparent_base = QtGui.QPixmap(pixmap.size())
            transparent_base.fill(QtCore.Qt.GlobalColor.transparent)
            super().__init__(
                transparent_base, QtCore.Qt.WindowType.WindowStaysOnTopHint)
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
            self.base_pixmap = pixmap
            self.opacity = 0.0

        def setOpacity(self, opacity):
            self.opacity = opacity
            self.repaint()

        def paintEvent(self, event):
            painter = QtGui.QPainter(self)
            painter.setOpacity(self.opacity)
            painter.drawPixmap(0, 0, self.base_pixmap)
            painter.end()

    splash = FadingSplash(rounded_splash)
    splash.setDisabled(True)
    splash.show()

    # Prolonged fade-in (1500ms) that manually fades the pixmap.
    splash._fade_anim = QtCore.QVariantAnimation(splash)
    splash._fade_anim.setDuration(1500)
    splash._fade_anim.setStartValue(0.0)
    splash._fade_anim.setEndValue(1.0)
    splash._fade_anim.setEasingCurve(QtCore.QEasingCurve.Type.InOutQuad)
    splash._fade_anim.valueChanged.connect(splash.setOpacity)

    loop = QtCore.QEventLoop()
    splash._fade_anim.finished.connect(loop.quit)
    splash._fade_anim.start()
    loop.exec()

    appView.splash = splash
    appView.obj_workspace.returnWhetherClickedOrNot(appView)

    try:
        if os.name == 'nt':
            user_home = os.path.join('library', 'config')
        else:
            user_home = os.path.expanduser('~')

        with open(os.path.join(user_home, ".esim/workspace.txt"), 'r') as file:
            work = int(file.read(1))
    # ValueError: an empty/truncated workspace.txt makes int('') fail; treat it
    # the same as a missing file and fall back to the workspace picker, rather
    # than letting the exception abort startup.
    except (IOError, ValueError):
        work = 0

    if work != 0:
        appView.obj_workspace.defaultWorkspace()
    else:
        appView.obj_workspace.show()

    sys.exit(app.exec())


# Call main function
if __name__ == '__main__':
    # Create and display the splash screen
    try:
        main(sys.argv)
    except Exception as err:
        print("Error: ", err)
