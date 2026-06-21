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
    init_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) + os.sep

from PyQt6 import QtGui, QtCore, QtWidgets
from PyQt6.QtCore import QSize
from configuration.Appconfig import Appconfig
from frontEnd import ProjectExplorer
from frontEnd import TimeExplorer
from frontEnd import Workspace
from frontEnd import DockArea
from projManagement.openProject import OpenProjectInfo
from projManagement.newProject import NewProjectInfo
from projManagement.Kicad import Kicad
from projManagement.Validation import Validation
from projManagement import Worker

# Its our main window of application.


def create_rounded_icon(path, radius_ratio=0.08):
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

        self.setGeometry(self.obj_appconfig._app_xpos,
                         self.obj_appconfig._app_ypos,
                         self.obj_appconfig._app_width,
                         self.obj_appconfig._app_heigth)
        self.setWindowTitle(
            self.obj_appconfig._APPLICATION + "-" + self.obj_appconfig._VERSION
        )
        self.showMaximized()
        self.setWindowIcon(QtGui.QIcon(init_path + 'images/logo.png'))

        self.systemTrayIcon = QtWidgets.QSystemTrayIcon(self)
        self.systemTrayIcon.setIcon(QtGui.QIcon(init_path + 'images/logo.png'))
        self.systemTrayIcon.setVisible(True)

        self.setStatusBar(QtWidgets.QStatusBar(self))
        self.statusBar().showMessage("Ready")

        # Status-bar right zone: simulation-status dot + FOSSEE logo (moved
        # here from the toolbar). The dot is a tiny premium affordance:
        # grey idle, amber running, green ok, red failed.
        self.sim_status_dot = QtWidgets.QLabel("●")
        self.sim_status_dot.setObjectName("simStatusDot")
        self.sim_status_dot.setToolTip("Simulation status")
        self._set_sim_status("idle")
        self.statusBar().addPermanentWidget(self.sim_status_dot)
        if hasattr(self, 'logo'):
            self.statusBar().addPermanentWidget(self.logo)

        from frontEnd.motion import install_button_motion, apply_toolbar_depth
        install_button_motion(self)
        apply_toolbar_depth(self)

        # Left toolbar added AFTER top toolbar → higher z-order.
        # Its opaque fill (#0D1728 / #FFFFFF) paints over the top bar's
        # downward shadow at the joint, hiding the seam. Left rail shadow
        # offset pushed down (y=6 in apply_toolbar_depth) so upward blur
        # bleed into the corner is minimal. No raise_() needed.

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
            # From System: flip to the opposite of what's currently shown.
            scheme = QtGui.QGuiApplication.styleHints().colorScheme()
            new_mode = "Light" if scheme == QtCore.Qt.ColorScheme.Dark else "Dark"
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

    def createPopupMenu(self):
        """Qt builds this menu (toolbar/dock right-click) internally and shows
        it immediately, so it never passed through our creation-site rounding
        and showed solid black corners. Round it here BEFORE Qt shows it (the
        translucent attribute must be set before the native window exists)."""
        menu = super().createPopupMenu()
        if menu is not None:
            try:
                from frontEnd.motion import make_menu_rounded
                make_menu_rounded(menu)
            except Exception:
                pass
        return menu

    def show_about(self):
        """Show the About eSim dialog with gradient-rich premium styling."""
        from frontEnd.dialogs import show_about_dialog
        show_about_dialog(self)

    def _build_menu_bar(self):
        """Build a proper menu bar with File/Edit/View/Tools/Help."""
        bar = self.menuBar()

        # ----- File -----
        file_menu = bar.addMenu('&File')
        file_menu.addAction(self.newproj)
        file_menu.addAction(self.openproj)
        file_menu.addAction(self.closeproj)
        file_menu.addSeparator()
        file_menu.addAction(self.wrkspce)
        file_menu.addSeparator()
        # Recent-projects submenu (populated when projects are opened)
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
        cut_action.setShortcut('Ctrl+X')
        # Note: Ctrl+X is already taken by Close Project; allow it for text
        # editing inside the dock area as a no-op global shortcut handle.
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

        # Time explorer action has been moved to the top toolbar

        console_action = QtGui.QAction('Console', self)
        console_action.setCheckable(True)
        console_action.setChecked(True)
        console_action.triggered.connect(
            lambda: (
                self.obj_Mainview.restore_console_area()
                if console_action.isChecked()
                else self.obj_Mainview.collapse_console_area()
            )
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
        tools_menu.addAction(self.nghdl)
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

        # Round the corners of every menu-bar dropdown (transparent corners,
        # no black squares). Set before first show so the native window is
        # created translucent.
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
            from projManagement.openProject import OpenProjectInfo
            open_proj = OpenProjectInfo()
            directory, filelist = open_proj.body(path)
            if directory and filelist:
                self.obj_Mainview.obj_projectExplorer.addTreeNode(
                    directory, filelist
                )
                self.obj_appconfig.current_project["ProjectName"] = directory
                project_name = os.path.basename(directory)
                self.obj_appconfig.save_current_project()
        except Exception as e:
            from frontEnd.dialogs import show_error
            show_error(self, "Open Project", f"Could not open {path}:\n{e}")
            self.obj_appconfig.print_warning(f"Recent project open failed: {e}")

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showMaximized()
        else:
            self.showFullScreen()

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

        from frontEnd.icon_paths import (
            timeline_icon, workspace_icon,
            close_proj_icon, help_icon, dev_docs_icon, settings_icon
        )

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

        self.timeline_action = QtGui.QAction(
            timeline_icon(),
            'Timeline', self
        )
        self.timeline_action.setToolTip('Timeline — View and restore project backups')
        self.timeline_action.triggered.connect(self.show_timeline)

        self.helpfile = QtGui.QAction(
            help_icon(),
            'User Manual', self
        )
        self.helpfile.setShortcut('F1')
        self.helpfile.setToolTip('User Manual (F1) — Open the eSim user manual')
        self.helpfile.triggered.connect(self.help_project)

        self.devdocs = QtGui.QAction(
            dev_docs_icon(),
            'Developer Docs', self
        )
        self.devdocs.setShortcut('Shift+F1')
        self.devdocs.setToolTip('Developer Docs (Shift+F1) — Open eSim developer docs')
        self.devdocs.triggered.connect(self.dev_docs)

        self.preferences_action = QtGui.QAction(
            settings_icon(),
            'Preferences', self
        )
        self.preferences_action.setShortcut('Ctrl+,')
        self.preferences_action.setToolTip('Preferences (Ctrl+,) — Configure eSim')
        self.preferences_action.triggered.connect(self.open_preferences)

        self.exit_action = QtGui.QAction('Exit', self)
        self.exit_action.setShortcut('Ctrl+Q')
        self.exit_action.setToolTip('Quit eSim (Ctrl+Q)')
        self.exit_action.setMenuRole(QtGui.QAction.MenuRole.QuitRole)
        self.exit_action.triggered.connect(self.close)

        self.about_action = QtGui.QAction('About eSim', self)
        self.about_action.setMenuRole(QtGui.QAction.MenuRole.AboutRole)
        self.about_action.triggered.connect(self.show_about)

        # --- Top toolbar: keep tool icons only, labels live in the menu bar ---
        self.topToolbar = self.addToolBar('Main')
        self.topToolbar.setObjectName('topToolbar')
        # Movable again -> Qt draws the native drag handle (gripper) on the
        # toolbar's leading edge, so the top bar can be re-arranged like before.
        self.topToolbar.setMovable(True)
        self.topToolbar.setFloatable(False)
        self.topToolbar.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
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
        
        # Expanding spacer pushes the view controls (zoom, theme toggle) to
        # the right edge — a well-established 'view controls live right'
        # convention. The FOSSEE logo moves out of the action bar entirely
        # (into the status bar) to free this horizontal space.
        self.topToolbar.addSeparator()
        self._tb_spacer = QtWidgets.QWidget()
        self._tb_spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred)
        self.topToolbar.addWidget(self._tb_spacer)

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
        self.zoom_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
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

        # Init zoom label
        from frontEnd.theme_utils import get_preferences
        u_home = os.path.join('library', 'config') if os.name == 'nt' else os.path.expanduser('~')
        zp = get_preferences(u_home).get("zoom_level", 100)
        self.zoom_label.setText(f" {zp}% ")

        # Menu-bar build must run AFTER all toolbar actions are wired,
        # because File/Edit/View/Tools menus reference the same QAction
        # objects. We construct left-toolbar actions further down; defer
        # this assembly to just before the function returns.
        self._deferred_menu_bar = self._build_menu_bar

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

        # FOSSEE logo is built here but mounted in the status bar (see
        # __init__) so it no longer eats prime action-bar real estate.
        self.logo = QtWidgets.QLabel()
        self.logopic = QtGui.QPixmap(
            os.path.join(
                os.path.abspath(''), init_path + 'images', 'fosseeLogo.png'
            ))
        self.logopic = self.logopic.scaled(
            QSize(96, 96), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation)

        self.logo.setPixmap(self.logopic)

        # Left Tool bar Action Widget
        self.kicad = QtGui.QAction(
            create_rounded_icon(init_path + 'images/kicad.png'),
            'Open Schematic', self
        )
        self.kicad.setShortcut("Ctrl+K")
        self.kicad.setToolTip("Open Schematic (Ctrl+K) - Design your circuit in KiCad")
        self.kicad.triggered.connect(self.obj_kicad.openSchematic)

        self.conversion = QtGui.QAction(
            create_rounded_icon(init_path + 'images/ki-ng.png'),
            'Convert to Ngspice', self
        )
        self.conversion.setShortcut("Ctrl+C")
        self.conversion.setToolTip("Convert to Ngspice (Ctrl+C) - Generate Ngspice netlist")
        self.conversion.triggered.connect(self.obj_kicad.openKicadToNgspice)

        self.ngspice = QtGui.QAction(
            create_rounded_icon(init_path + 'images/ngspice.png'),
            'Simulate', self
        )
        self.ngspice.setShortcut("Ctrl+S")
        self.ngspice.setToolTip("Simulate (Ctrl+S) - Run circuit simulation")
        self.ngspice.triggered.connect(self.plotFlagPopBox)

        self.model = QtGui.QAction(
            create_rounded_icon(init_path + 'images/model.png'),
            'Model Editor', self
        )
        self.model.setShortcut("Ctrl+M")
        self.model.setToolTip("Model Editor (Ctrl+M) - Create or edit SPICE models")
        self.model.triggered.connect(self.open_modelEditor)

        self.subcircuit = QtGui.QAction(
            create_rounded_icon(init_path + 'images/subckt.png'),
            'Subcircuit', self
        )
        self.subcircuit.setShortcut("Ctrl+B")
        self.subcircuit.setToolTip("Subcircuit (Ctrl+B) - Build reusable subcircuits")
        self.subcircuit.triggered.connect(self.open_subcircuit)

        self.nghdl = QtGui.QAction(
            create_rounded_icon(init_path + 'images/nghdl.png'), 'NGHDL', self
        )
        self.nghdl.setShortcut("Ctrl+H")
        self.nghdl.setToolTip("NGHDL (Ctrl+H) - Add VHDL digital models")
        self.nghdl.triggered.connect(self.open_nghdl)

        self.makerchip = QtGui.QAction(
            create_rounded_icon(init_path + 'images/makerchip.png'),
            'Makerchip', self
        )
        self.makerchip.setToolTip("Makerchip - Verilog design via Makerchip")
        self.makerchip.triggered.connect(self.open_makerchip)

        self.omedit = QtGui.QAction(
            create_rounded_icon(init_path + 'images/omedit.png'),
            'Modelica Converter', self
        )
        self.omedit.setToolTip("Modelica Converter - Convert to Modelica format")
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
        self.conToeSim.setToolTip("Schematic Converter - Import PSpice/LTspice files")
        self.conToeSim.triggered.connect(self.open_conToeSim)

        # Adding Action Widget to tool bar
        self.lefttoolbar = QtWidgets.QToolBar('Left ToolBar')
        self.lefttoolbar.setObjectName('leftToolBar')
        # Native handle fully enabled — user can drag to float, dock any edge,
        # or tabify. Same freedom as the top toolbar.
        self.lefttoolbar.setMovable(True)
        self.addToolBar(QtCore.Qt.ToolBarArea.LeftToolBarArea, self.lefttoolbar)

        def _rail_caption(text):
            lbl = QtWidgets.QLabel(text)
            lbl.setProperty("cssClass", "railCaption")
            lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            return lbl

        # Group the tools into labelled clusters so a new user can parse the
        # rail at a glance instead of staring at ten cryptic icons.
        self.lefttoolbar.addWidget(_rail_caption("DESIGN"))
        self.lefttoolbar.addAction(self.kicad)
        self.lefttoolbar.addAction(self.conversion)
        self.lefttoolbar.addSeparator()
        self.lefttoolbar.addWidget(_rail_caption("SIMULATE"))
        self.lefttoolbar.addAction(self.ngspice)
        self.lefttoolbar.addAction(self.nghdl)
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

        # Native handle at toolbar top provides full docking (movable=True).
        # No custom grip needed — native Qt handle does vertical + float + all areas.

        self.lefttoolbar.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.lefttoolbar.setIconSize(QSize(40, 40))

        # Now that every toolbar action exists, we can wire the menu bar
        # which references them. Deferred from earlier in this method
        # because left-toolbar actions are defined further down.
        self._deferred_menu_bar()

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
        reply = QtWidgets.QMessageBox.question(
            self, 'Message', exit_msg, QtWidgets.QMessageBox.StandardButton.Yes,
            QtWidgets.QMessageBox.StandardButton.No
        )

        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            for proc in self.obj_appconfig.procThread_list:
                try:
                    proc.terminate()
                except Exception:
                    pass
            try:
                for process_object in self.obj_appconfig.process_obj:
                    try:
                        process_object.close()
                    except Exception:
                        pass
            except Exception:
                pass

            # Check if "Open project" and "New project" window is open.
            # If yes, just close it when application is closed.
            try:
                self.project.close()
            except Exception:
                pass
            event.accept()
            self.systemTrayIcon.showMessage('Exit', 'eSim is Closed.')

        elif reply == QtWidgets.QMessageBox.StandardButton.No:
            event.ignore()

    def change_zoom(self, delta):
        from frontEnd.theme_utils import get_preferences, apply_theme
        import json
        user_home = os.path.join('library', 'config') if os.name == 'nt' else os.path.expanduser('~')
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
            except: pass
            if hasattr(self, 'zoom_label'):
                self.zoom_label.setText(f" {new_zoom}% ")
            apply_theme(QtWidgets.QApplication.instance())
            if hasattr(self, 'topToolbar'):
                scaled_size_top = int(28 * (new_zoom / 100.0))
                self.topToolbar.setIconSize(QtCore.QSize(scaled_size_top, scaled_size_top))
            if hasattr(self, 'lefttoolbar'):
                scaled_size_left = int(40 * (new_zoom / 100.0))
                self.lefttoolbar.setIconSize(QtCore.QSize(scaled_size_left, scaled_size_left))

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
            except Exception:
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
        except Exception:
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

    def show_timeline(self):
        project_dir = self.obj_appconfig.current_project["ProjectName"]
        if not project_dir:
            from frontEnd.dialogs import show_error
            show_error(self, "No Active Project", "Please open a project to view its timeline/backups.")
            return
            
        if not hasattr(self, 'time_explorer_dialog'):
            from frontEnd import TimeExplorer
            self.time_explorer_dialog = TimeExplorer.TimeExplorer(self)
            self.time_explorer_dialog.current_project = self.obj_appconfig.current_project
            self.time_explorer_dialog.current_project_path = self.obj_appconfig.current_project
            
        project_name = os.path.basename(project_dir)
        self.time_explorer_dialog.load_snapshots(project_name)
        self.time_explorer_dialog.show()
        self.time_explorer_dialog.raise_()
        self.time_explorer_dialog.activateWindow()

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
        """Opens the Preferences Dialog"""
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
                from frontEnd.dialogs import show_error
                show_error(self, "Plotting Error", 'Data could not be plotted. Please try again.')
                print("Exception Message:", str(e), traceback.format_exc())
                self.obj_appconfig.print_error('Exception Message : '
                                               + str(e))
        else:
            self._set_sim_status("failed")

    def open_ngspice(self):
        """This Function execute ngspice on current project."""
        projDir = self.obj_appconfig.current_project["ProjectName"]

        if projDir is not None:
            projName = self.obj_appconfig.get_proj_stem()
            ngspiceNetlist = os.path.join(projDir, projName + ".cir.out")

            if not os.path.isfile(ngspiceNetlist):
                print(
                    "Netlist file (*.cir.out) not found."
                )
                from frontEnd.dialogs import show_error
                show_error(self, "File Not Found", 'Netlist (*.cir.out) not found.')
                return

            self.obj_Mainview.obj_dockarea.ngspiceEditor(
                projName, ngspiceNetlist, self.simulationEndSignal, self.plotFlag)

            self._set_sim_status("running")
            self.ngspice.setEnabled(False)
            self.conversion.setEnabled(False)
            self.closeproj.setEnabled(False)
            self.wrkspce.setEnabled(False)

        else:
            from frontEnd.dialogs import show_error
            show_error(self, "No Project Selected", 'Please select or create a project first.')

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

    def open_nghdl(self):
        """
        This function calls NGHDL option in left-tool-bar.
        It uses validateTool() method from Validation.py:

            - If 'nghdl' is present in executables list then
              it passes command 'nghdl -e' to WorkerThread class of
              Worker.py.
            - If 'nghdl' is not present, then it shows error message.
        """
        print("Function : NGHDL")
        self.obj_appconfig.print_info('NGHDL is called')

        if self.obj_validation.validateTool('nghdl'):
            self.cmd = 'nghdl -e'
            self.obj_workThread = Worker.WorkerThread(self.cmd)
            self.obj_workThread.start()
        else:
            from frontEnd.dialogs import show_error
            show_error(self, 'NGHDL Error', 'Error while opening NGHDL. Please make sure it is installed.')

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
                        self.msg = QtWidgets.QMessageBox()
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
                    self.msg = QtWidgets.QErrorMessage()
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

                from frontEnd.dialogs import show_error
                show_error(self, "Missing Ngspice Netlist", 'Current project does not contain any Ngspice file. Please create Ngspice file with extension .cir.out')
        else:
            from frontEnd.dialogs import show_error
            show_error(self, "No Project Selected", 'Please select or create a project first.')

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
            from frontEnd.dialogs import show_error
            msgContent = (
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
            self.obj_appconfig.print_info(msgContent)
            # Cannot use show_error for rich text easily, so keep QMessageBox but use correct API
            msg = QtWidgets.QMessageBox(self)
            msg.setIcon(QtWidgets.QMessageBox.Icon.Critical)
            msg.setWindowTitle("Error Message")
            msg.setTextFormat(QtCore.Qt.TextFormat.RichText)
            msg.setText(msgContent)
            msg.exec()

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
        self.noteArea.setObjectName('mainNoteConsole')
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
        self.noteArea.append('\n')

        self.obj_dockarea = DockArea.DockArea()
        self.obj_projectExplorer = ProjectExplorer.ProjectExplorer()

        # Adding content to vertical middle Split.
        self.middleSplit.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.middleSplit.addWidget(self.obj_dockarea)
        # noteArea moved to left panel

        # Adding middle split to Middle Container Widget
        self.middleContainerLayout.addWidget(self.middleSplit)
        self.middleContainer.setLayout(self.middleContainerLayout)

        # Adding content of left split
        self.leftVerticalSplitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.leftVerticalSplitter.addWidget(self.obj_projectExplorer)
        self.leftVerticalSplitter.addWidget(self.noteArea)
        
        self.leftSplit.addWidget(self.leftVerticalSplitter)
        self.leftSplit.addWidget(self.middleContainer)

        # Adding to main Layout
        self.mainLayout.addWidget(self.leftSplit)
        self.leftSplit.setSizes([int(self.width() / 4.5), self.height()])
        self.leftVerticalSplitter.setSizes([int(self.height() * 0.7), int(self.height() * 0.3)])
        self.middleSplit.setSizes([self.width(), self.height()])
        self.setLayout(self.mainLayout)

    def collapse_console_area(self):
        """Collapse the console area to minimal height."""
        current_sizes = self.leftVerticalSplitter.sizes()
        total_height = sum(current_sizes)
        minimal_console_height = 0
        explorer_height = total_height - minimal_console_height
        self.leftVerticalSplitter.setSizes([explorer_height, minimal_console_height])

    def restore_console_area(self):
        """Restore the console area to normal height."""
        total_height = sum(self.leftVerticalSplitter.sizes())
        explorer_height = int(total_height * 0.7)  # 70% for explorer
        console_height = total_height - explorer_height  # 30% for console
        self.leftVerticalSplitter.setSizes([explorer_height, console_height])


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
    import os
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QtWidgets.QApplication(args)
    app.setApplicationName("eSim")

    # Install global popup motion (fade menus)
    try:
        from frontEnd.motion import install_popup_motion
        install_popup_motion(app)
    except Exception as e:
        print(f"Could not install popup motion: {e}")

    # Install global drop-shadow refresh: re-validates a widget's shadow cache
    # on Show so buttons never paint blank after a hide->show (stack/tab switch
    # in maximized/fullscreen state).
    try:
        from frontEnd.motion import install_effect_refresh
        install_effect_refresh(app)
    except Exception as e:
        print(f"Could not install effect refresh: {e}")

    # Load global stylesheet dynamically based on OS theme and preferences
    from PyQt6 import QtGui
    from frontEnd.theme_utils import apply_theme as _apply_theme

    def apply_theme(*args):
        _apply_theme(app)
            
    # Attach to app instance so PreferencesDialog can call it
    app.apply_theme = apply_theme

    # Apply theme initially and listen for OS changes
    apply_theme()
    QtGui.QGuiApplication.styleHints().colorSchemeChanged.connect(apply_theme)

            
    # Load custom font
    font_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'images', 'fonts')
    font_path = os.path.join(font_dir, 'Inter-VariableFont_slnt,wght.ttf')
    if os.path.exists(font_path):
        QtGui.QFontDatabase.addApplicationFont(font_path)

    # Auto-repair corrupted KiCad symbol libraries before anything else
    repair_messages = []
    try:
        from maker.KicadSymbolFixer import repair_all_sym_files
        repair_messages = repair_all_sym_files()
    except Exception as e:
        print(f"[KicadSymbolFixer] Warning: auto-repair skipped: {e}")

    appView = Application()
    
    # Log any symbol fixes to the GUI console
    for msg in repair_messages:
        appView.obj_appconfig.print_warning(msg)
        
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
    # Timeline is now opened on demand via toolbar
    appView.hide()

    splash_pix = QtGui.QPixmap(init_path + 'images/splash_screen_esim.png')
    splash_pix = splash_pix.scaledToWidth(int(splash_pix.width() * 0.8), QtCore.Qt.TransformationMode.SmoothTransformation)
    
    # Apply a proportional rounded mask to the splash screen to cut out the heavy black corners
    radius = int(min(splash_pix.width(), splash_pix.height()) * 0.10)
    rounded_splash = QtGui.QPixmap(splash_pix.size())
    rounded_splash.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(rounded_splash)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    path_obj = QtGui.QPainterPath()
    path_obj.addRoundedRect(0, 0, splash_pix.width(), splash_pix.height(), radius, radius)
    painter.setClipPath(path_obj)
    painter.drawPixmap(0, 0, splash_pix)
    painter.end()
    
    class FadingSplash(QtWidgets.QSplashScreen):
        def __init__(self, pixmap):
            # Create a completely transparent pixmap of the same size for the base class
            # so it sizes the window correctly but doesn't draw the opaque image itself!
            transparent_base = QtGui.QPixmap(pixmap.size())
            transparent_base.fill(QtCore.Qt.GlobalColor.transparent)
            
            super().__init__(transparent_base, QtCore.Qt.WindowType.WindowStaysOnTopHint)
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
    
    # Prolonged fade-in animation (1500ms) that manually fades the pixmap
    splash._fade_anim = QtCore.QVariantAnimation(splash)
    splash._fade_anim.setDuration(1500)
    splash._fade_anim.setStartValue(0.0)
    splash._fade_anim.setEndValue(1.0)
    splash._fade_anim.setEasingCurve(QtCore.QEasingCurve.Type.InOutQuad)
    splash._fade_anim.valueChanged.connect(splash.setOpacity)
    
    # Force the event loop to pump so the animation plays before app loads
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
