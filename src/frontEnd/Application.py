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
        # Top Tool bar
        self.newproj = QtGui.QAction(
            QtGui.QIcon(init_path + 'images/newProject.png'),
            '<b>New Project</b>', self
        )
        self.newproj.setShortcut('Ctrl+N')
        self.newproj.triggered.connect(self.new_project)

        self.openproj = QtGui.QAction(
            QtGui.QIcon(init_path + 'images/openProject.png'),
            '<b>Open Project</b>', self
        )
        self.openproj.setShortcut('Ctrl+O')
        self.openproj.triggered.connect(self.open_project)

        self.closeproj = QtGui.QAction(
            QtGui.QIcon(init_path + 'images/closeProject.png'),
            '<b>Close Project</b>', self
        )
        self.closeproj.setShortcut('Ctrl+X')
        self.closeproj.triggered.connect(self.close_project)

        self.wrkspce = QtGui.QAction(
            QtGui.QIcon(init_path + 'images/workspace.ico'),
            '<b>Change Workspace</b>', self
        )
        self.wrkspce.setShortcut('Ctrl+W')
        self.wrkspce.triggered.connect(self.change_workspace)

        self.helpfile = QtGui.QAction(
            QtGui.QIcon(init_path + 'images/helpProject.png'),
            '<b>Help</b>', self
        )
        self.helpfile.setShortcut('Ctrl+H')
        self.helpfile.triggered.connect(self.help_project)

        # added devDocs logo and called functions
        self.devdocs = QtGui.QAction(
            QtGui.QIcon(init_path + 'images/dev_docs.png'),
            '<b>Dev Docs</b>', self
        )
        self.devdocs.setShortcut('Ctrl+D')
        self.devdocs.triggered.connect(self.dev_docs)

        self.topToolbar = self.addToolBar('Top Tool Bar')
        self.topToolbar.setObjectName('topToolbar')
        self.topToolbar.addAction(self.newproj)
        self.topToolbar.addAction(self.openproj)
        self.topToolbar.addAction(self.closeproj)
        self.topToolbar.addAction(self.wrkspce)
        self.topToolbar.addAction(self.helpfile)
        self.topToolbar.addAction(self.devdocs)

        # Project Snapshots (was the left-face timeline). Inline with the other
        # top-toolbar icons, before the spacer/logo, same size/look.
        self.act_snapshots = QtGui.QAction(
            QtGui.QIcon(init_path + 'images/history.png'),
            '<b>Project Snapshots</b>', self)
        self.act_snapshots.triggered.connect(self.show_snapshots)
        self.topToolbar.addAction(self.act_snapshots)

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

        # This part is setting fossee logo to the right
        # corner in the application window.
        self.spacer = QtWidgets.QWidget()
        self.spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding)
        self.topToolbar.addWidget(self.spacer)
        self.logo = QtWidgets.QLabel()
        self.logopic = QtGui.QPixmap(
            os.path.join(
                os.path.abspath(''), init_path + 'images', 'fosseeLogo.png'
            ))
        self.logopic = self.logopic.scaled(
            QSize(150, 150), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.logo.setPixmap(self.logopic)
        self.logo.setStyleSheet("padding:0 15px 0 0;")
        self.topToolbar.addWidget(self.logo)

        # Left Tool bar Action Widget
        self.kicad = QtGui.QAction(
            QtGui.QIcon(init_path + 'images/kicad.png'),
            '<b>Open Schematic</b>', self
        )
        self.kicad.triggered.connect(self.obj_kicad.openSchematic)

        self.conversion = QtGui.QAction(
            QtGui.QIcon(init_path + 'images/ki-ng.png'),
            '<b>Convert KiCad to Ngspice</b>', self
        )
        self.conversion.triggered.connect(self.obj_kicad.openKicadToNgspice)

        self.ngspice = QtGui.QAction(
            QtGui.QIcon(init_path + 'images/ngspice.png'),
            '<b>Simulate</b>', self
        )
        self.ngspice.triggered.connect(self.plotFlagPopBox)

        self.model = QtGui.QAction(
            QtGui.QIcon(init_path + 'images/model.png'),
            '<b>Model Editor</b>', self
        )
        self.model.triggered.connect(self.open_modelEditor)

        self.subcircuit = QtGui.QAction(
            QtGui.QIcon(init_path + 'images/subckt.png'),
            '<b>Subcircuit</b>', self
        )
        self.subcircuit.triggered.connect(self.open_subcircuit)

        # NGHDL is no longer a standalone toolbar button: it now lives as a
        # tab inside the Makerchip dock (Makerchip / NgVeri / NGHDL), so model
        # creation for both Verilog and VHDL is in one place.
        self.makerchip = QtGui.QAction(
            QtGui.QIcon(init_path + 'images/makerchip.png'),
            '<b>Model Creation (Verilog / VHDL)</b>', self
        )
        self.makerchip.triggered.connect(self.open_makerchip)

        self.omedit = QtGui.QAction(
            QtGui.QIcon(init_path + 'images/omedit.png'),
            '<b>Modelica Converter</b>', self
        )
        self.omedit.triggered.connect(self.open_OMedit)

        self.omoptim = QtGui.QAction(
            QtGui.QIcon(init_path + 'images/omoptim.png'),
            '<b>OM Optimisation</b>', self
        )
        self.omoptim.triggered.connect(self.open_OMoptim)

        self.conToeSim = QtGui.QAction(
            QtGui.QIcon(init_path + 'images/icon.png'),
            '<b>Schematic converter</b>', self
        )
        self.conToeSim.triggered.connect(self.open_conToeSim)

        # Adding Action Widget to tool bar
        self.lefttoolbar = QtWidgets.QToolBar('Left ToolBar')
        self.lefttoolbar.setObjectName('leftToolBar')
        self.addToolBar(QtCore.Qt.ToolBarArea.LeftToolBarArea, self.lefttoolbar)
        self.lefttoolbar.addAction(self.kicad)
        self.lefttoolbar.addAction(self.conversion)
        self.lefttoolbar.addAction(self.ngspice)
        self.lefttoolbar.addAction(self.model)
        self.lefttoolbar.addAction(self.subcircuit)
        self.lefttoolbar.addAction(self.makerchip)
        self.lefttoolbar.addAction(self.omedit)
        self.lefttoolbar.addAction(self.omoptim)
        self.lefttoolbar.addAction(self.conToeSim)
        self.lefttoolbar.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.lefttoolbar.setIconSize(QSize(40, 40))

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
            try:
                self.obj_Mainview.obj_dockarea.plottingEditor()
            except Exception as e:
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
    app = QtWidgets.QApplication(args)
    app.setApplicationName("eSim")

    # Aurora design system: apply the global theme once, before any widgets
    # are built. Guarded so a theming failure can never block app startup.
    try:
        theme_utils.apply_theme(app)
    except Exception as e:
        print("Theme load failed, continuing unthemed:", str(e))

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
    splash = QtWidgets.QSplashScreen(
        splash_pix, QtCore.Qt.WindowType.WindowStaysOnTopHint
    )
    splash.setMask(splash_pix.mask())
    splash.setDisabled(True)
    splash.show()

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
