from PyQt6 import QtCore, QtWidgets
from configuration import Dialogs
from ngspiceSimulation import plotWindow
from ngspiceSimulation.NgspiceWidget import NgspiceWidget
from configuration.Appconfig import Appconfig
from modelEditor.ModelEditor import ModelEditorclass
from subcircuit.Subcircuit import Subcircuit
from maker.makerchip import makerchip
from kicadtoNgspice.KicadtoNgspice import MainWindow
from browser.Welcome import Welcome
from browser.UserManual import UserManual
from ngspicetoModelica.ModelicaUI import OpenModelicaEditor
from PyQt6.QtWidgets import QLineEdit, QLabel, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt6.QtCore import Qt
import os
from converter.pspiceToKicad import PspiceConverter
from converter.ltspiceToKicad import LTspiceConverter
from converter.LtspiceLibConverter import LTspiceLibConverter
from converter.libConverter import PspiceLibConverter
from converter.browseSchematic import browse_path
dockList = ['Welcome']
count = 1
dock = {}


class WaveformDock(QtWidgets.QDockWidget):
    """The Verilog waveform's own eSim tab.

    Deletes itself on close (the plot widget is heavy, so it must not linger
    across repeated simulate/close cycles) and emits ``closed`` first so the
    host can drop its bookkeeping and return focus to the Verify stage.
    """

    closed = QtCore.pyqtSignal()

    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.setObjectName(title)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)


class DockArea(QtWidgets.QMainWindow):
    """
    This class contains function for designing UI of all the editors
    in dock area part:

        - Test Editor.
        - Model Editor.
        - Python Plotting.
        - Ngspice Editor.
        - Kicad to Ngspice Editor.
        - Subcircuit Editor.
        - Modelica editor.
    """

    def __init__(self):
        """This act as constructor for class DockArea."""
        QtWidgets.QMainWindow.__init__(self)
        self.obj_appconfig = Appconfig()
        # Track plotting docks
        self.active_plotting_docks = set()
        # Verilog waveform tab per Model Creation dock (source dock -> wave
        # dock), so a re-simulate reuses one viewer instead of stacking tabs.
        self._wave_docks = {}

        for dockName in dockList:
            dock[dockName] = QtWidgets.QDockWidget(dockName)
            self.welcomeWidget = QtWidgets.QWidget()
            self.welcomeLayout = QtWidgets.QVBoxLayout()
            self.welcomeLayout.addWidget(Welcome())  # Call browser

            # Adding to main Layout
            self.welcomeWidget.setLayout(self.welcomeLayout)
            dock[dockName].setWidget(self.welcomeWidget)
            # No title strip here either -- the bottom tab already labels it, so
            # the home tab matches the title-less tool docks (see
            # apply_fullscreen_feature).
            dock[dockName].setTitleBarWidget(QtWidgets.QWidget())
            # Welcome is the permanent "home" tab: keep it movable/floatable but
            # drop the Closable feature so its title bar has no close button.
            # The tab-strip X is stripped separately in enable_tab_close_buttons.
            dock[dockName].setFeatures(
                QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable |
                QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable)
            self.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea, dock[dockName])

        # self.tabifyDockWidget(dock['Notes'],dock['Blank'])
        self.show()

    def tabifyDockWidget(self, first, second):
        """Tabify two docks, then (re)arm the close-X on the bottom tab bar so
        every tabified tool can be closed straight from its tab."""
        super().tabifyDockWidget(first, second)
        self.enable_tab_close_buttons()

    def enable_tab_close_buttons(self):
        """Turn on the close button for the QMainWindow's own dock tab bars.

        Only the dock-area tab bars are targeted (parent is the QMainWindow, not
        a QTabWidget), so tab strips *inside* a tool (NgVeri, Flow Navigator …)
        are left untouched."""
        for tb in self.findChildren(QtWidgets.QTabBar):
            if isinstance(tb.parent(), QtWidgets.QTabWidget):
                continue
            tb.setTabsClosable(True)
            try:
                tb.tabCloseRequested.disconnect()
            except Exception:
                pass
            tb.tabCloseRequested.connect(
                lambda index, tab_bar=tb:
                self.handle_tab_close(index, tab_bar))
            # Welcome is the permanent home tab -- strip its close button so it
            # can never be closed from the tab strip.
            for i in range(tb.count()):
                if tb.tabText(i).replace('&', '').strip() == 'Welcome':
                    tb.setTabButton(
                        i,
                        QtWidgets.QTabBar.ButtonPosition.RightSide,
                        None)

    def _forget_dock(self, child):
        """Drop every reference eSim keeps to a dock so closing a tab cannot
        leave a zombie member behind (which would corrupt the saved layout)."""
        keys_to_delete = [k for k, v in dock.items() if v is child]
        for k in keys_to_delete:
            del dock[k]
        try:
            self.active_plotting_docks.discard(child)
        except Exception:
            pass
        # Drop any waveform tab bound to this source dock (and vice-versa).
        try:
            for src in [s for s, w in self._wave_docks.items()
                        if w is child or s is child]:
                del self._wave_docks[src]
        except Exception:
            pass
        for docks in self.obj_appconfig.dock_dict.values():
            if child in docks:
                docks.remove(child)

    def _destroy_dock(self, child):
        """Close, unparent and schedule deletion of a dock + forget it."""
        child.close()
        try:
            self.removeDockWidget(child)
        except Exception:
            pass
        self._forget_dock(child)
        child.deleteLater()

    def handle_tab_close(self, index, tab_bar):
        """Close the dock behind the clicked tab (matched by title), tearing it
        fully down rather than just hiding it."""
        tab_text = tab_bar.tabText(index).replace('&', '').strip()
        if tab_text.endswith('...'):
            tab_text = tab_text[:-3].strip()

        # Welcome is the permanent home tab -- never destroy it.
        if tab_text == 'Welcome':
            return

        for child in self.findChildren(QtWidgets.QDockWidget):
            if not child.isVisible():
                continue
            title = child.windowTitle().replace('&', '').strip()
            if title == tab_text or (tab_text and title.startswith(tab_text)):
                self._destroy_dock(child)
                return

        # Fallback: close by visible-index when the title match fails.
        visible = [d for d in self.findChildren(QtWidgets.QDockWidget)
                   if d.isVisible()]
        if index < len(visible):
            self._destroy_dock(visible[index])

    def get_main_view_reference(self):
        """Get reference to the MainView widget."""
        parent = self.parent()
        while parent:
            if hasattr(parent, 'collapse_console_area'):
                return parent
            parent = parent.parent()
        return None

    def on_dock_activated(self, dock_widget):
        """Handle when any dock becomes active. The console is on-demand now
        (the status bar carries the latest line), so we only auto-collapse for
        plotting docks and never force it back open."""
        main_view = self.get_main_view_reference()
        if not main_view:
            return

        if dock_widget in self.active_plotting_docks:
            main_view.collapse_console_area()

    def apply_fullscreen_feature(self, dock_widget, original_widget):
        """Mount a dock's content inside a rounded Aurora card.

        Fullscreen is no longer dock chrome at all: it is a small per-panel
        control living in each working panel's own header (see
        frontEnd.FullScreen.FullScreenToggle). Here we wrap the tool content in
        a ``#dockCard`` frame (themed surface + 16px radius) sitting inside a
        thin holder whose margins reveal the darker dock base around it, so a
        docked tool reads as a raised floating card.

        The holder is the dock's *direct* child, so FullScreenToggle (which
        reparents the dock's direct child) carries the whole card in and out of
        fullscreen and the look survives the round-trip. The card is a plain
        QFrame with no graphics effect, so QWebEngineView tools (Makerchip /
        User Manual) keep rendering."""
        if not dock_widget.objectName():
            dock_widget.setObjectName(dock_widget.windowTitle() or "dock")

        card = QtWidgets.QFrame()
        card.setObjectName("dockCard")
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.setSpacing(0)
        card_layout.addWidget(original_widget)

        holder = QtWidgets.QWidget()
        holder_layout = QtWidgets.QVBoxLayout(holder)
        holder_layout.setContentsMargins(10, 8, 10, 10)
        holder_layout.setSpacing(0)
        holder_layout.addWidget(card)

        dock_widget.setWidget(holder)

        # Kill the dock's title bar ("Netlist-IC1-3", "Makerchip-1", ...). The
        # docks are tabified, so the bottom tab already names each panel and
        # carries its own close button -- the title strip just duplicated that
        # name and ate vertical space. An empty title-bar widget removes the
        # strip while keeping windowTitle intact (the tab still reads it).
        dock_widget.setTitleBarWidget(QtWidgets.QWidget())

        # Freshly-mounted tool content carries its own buttons; re-install the
        # Aurora hover/press glow so they animate too (gated inside motion — a
        # no-op when the user has motion off).
        try:
            from frontEnd.motion import install_button_motion
            install_button_motion(self)
        except Exception:
            pass

    def createTestEditor(self):
        """This function create widget for Library Editor"""
        global count

        self.testWidget = QtWidgets.QWidget()
        self.testArea = QtWidgets.QTextEdit()
        self.testLayout = QtWidgets.QVBoxLayout()
        self.testLayout.addWidget(self.testArea)

        # Adding to main Layout
        self.testWidget.setLayout(self.testLayout)
        dock['Tips-' + str(count)] = \
            QtWidgets.QDockWidget('Tips-' + str(count))
        self.apply_fullscreen_feature(
            dock['Tips-' + str(count)], self.testWidget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea,
                           dock['Tips-' + str(count)])
        self.tabifyDockWidget(
            dock['Welcome'], dock['Tips-' + str(count)])

        dock['Tips-' + str(count)].setVisible(True)
        dock['Tips-' + str(count)].setFocus()

        dock['Tips-' + str(count)].raise_()

        temp = self.obj_appconfig.current_project['ProjectName']
        if temp:
            self.obj_appconfig.dock_dict[temp].append(
                dock['Tips-' + str(count)]
            )
        count = count + 1

    def plottingEditor(self):
        """This function create widget for interactive PythonPlotting."""
        self.projDir = self.obj_appconfig.current_project["ProjectName"]
        self.projName = self.obj_appconfig.get_proj_stem() \
            or os.path.basename(self.projDir)
        dockName = f'Plotting-{self.projName}-'
        # self.project = os.path.join(self.projDir, self.projName)

        global count
        self.plottingWidget = QtWidgets.QWidget()

        self.plottingLayout = QtWidgets.QVBoxLayout()
        self.plottingLayout.addWidget(plotWindow(self.projDir, self.projName))

        # Adding to main Layout
        self.plottingWidget.setLayout(self.plottingLayout)
        dock[dockName + str(count)
             ] = QtWidgets.QDockWidget(dockName
                                       + str(count))
        self.apply_fullscreen_feature(
            dock[dockName + str(count)], self.plottingWidget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea,
                           dock[dockName + str(count)])
        self.tabifyDockWidget(dock['Welcome'],
                              dock[dockName + str(count)])
        
        # Track this as a plotting dock
        self.active_plotting_docks.add(dock[dockName + str(count)])
        
        # Connect to tab change signal
        try:
            self.tabifiedDockWidgetActivated.connect(self.on_dock_activated)
        except (RuntimeError, TypeError):
            pass  # In case signal is already connected

        dock[dockName + str(count)].setVisible(True)
        dock[dockName + str(count)].setFocus()
        dock[dockName + str(count)].raise_()

        # Collapse console immediately
        main_view = self.get_main_view_reference()
        if main_view:
            QtCore.QTimer.singleShot(100, main_view.collapse_console_area)

        temp = self.obj_appconfig.current_project['ProjectName']
        if temp:
            self.obj_appconfig.dock_dict[temp].append(
                dock[dockName + str(count)]
            )
        count = count + 1

    def ngspiceEditor(self, projName, netlist, simEndSignal, plotFlag):
        """ This function creates widget for Ngspice window."""
        global count
        self.ngspiceWidget = QtWidgets.QWidget()

        self.ngspiceLayout = QtWidgets.QVBoxLayout()
        self.ngspiceLayout.addWidget(
            NgspiceWidget(netlist, simEndSignal, plotFlag)
        )

        # Adding to main Layout
        self.ngspiceWidget.setLayout(self.ngspiceLayout)
        dockName = f'Simulation-{projName}-'
        dock[dockName + str(count)
             ] = QtWidgets.QDockWidget(dockName
                                       + str(count))
        self.apply_fullscreen_feature(
            dock[dockName + str(count)], self.ngspiceWidget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea,
                           dock[dockName + str(count)])
        self.tabifyDockWidget(dock['Welcome'],
                              dock[dockName
                                   + str(count)])


        dock[dockName + str(count)].setVisible(True)
        dock[dockName + str(count)].setFocus()
        dock[dockName + str(count)].raise_()

        temp = self.obj_appconfig.current_project['ProjectName']
        if temp:
            self.obj_appconfig.dock_dict[temp].append(
                dock[dockName + str(count)]
            )
        count = count + 1

    def eSimConverter(self):
        """This function creates a widget for eSimConverter."""
        global count

        dockName = 'Schematic Converter-'

        self.eConWidget = QtWidgets.QWidget()
        self.eConLayout = QVBoxLayout()  # QVBoxLayout for the main layout

        file_path_layout = QHBoxLayout()  # QHBoxLayout for file path line
        lib_path_layout = QHBoxLayout()

        file_path_text_box = QLineEdit()
        file_path_text_box.setFixedHeight(30)
        file_path_text_box.setFixedWidth(800)
        file_path_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        file_path_layout.addWidget(file_path_text_box)

        browse_button = QPushButton("Browse")
        browse_button.setFixedSize(100, 30)
        browse_button.clicked.connect(lambda: browse_path(self,file_path_text_box))
        file_path_layout.addWidget(browse_button)

        self.eConLayout.addLayout(file_path_layout)  # Add file path layout to main layout

        button_layout = QHBoxLayout()  # QHBoxLayout for the buttons

        self.pspice_converter = PspiceConverter(self)
        self.ltspice_converter = LTspiceConverter(self)
        self.pspiceLib_converter = PspiceLibConverter(self)
        self.ltspiceLib_converter = LTspiceLibConverter(self)

        upload_button2 = QPushButton("Convert PSpice library")
        upload_button2.setFixedSize(180, 30)
        upload_button2.clicked.connect(lambda: self.pspiceLib_converter.upload_file_Pspice(file_path_text_box.text()))
        button_layout.addWidget(upload_button2)

        upload_button1 = QPushButton("Convert Pspice schematic")
        upload_button1.setFixedSize(180, 30)
        upload_button1.clicked.connect(lambda: self.pspice_converter.upload_file_Pspice(file_path_text_box.text()))
        button_layout.addWidget(upload_button1)

        upload_button3 = QPushButton("Convert LTspice library")
        upload_button3.setFixedSize(184, 30)
        upload_button3.clicked.connect(lambda: self.ltspiceLib_converter.upload_file_LTspice(file_path_text_box.text()))
        button_layout.addWidget(upload_button3)

        upload_button = QPushButton("Convert LTspice schematic")
        upload_button.setFixedSize(184, 30)
        upload_button.clicked.connect(lambda: self.ltspice_converter.upload_file_LTspice(file_path_text_box.text()))
        button_layout.addWidget(upload_button)

        self.eConLayout.addLayout(button_layout)

        self.eConWidget.setLayout(self.eConLayout)

        # lib_path_text_box = QLineEdit()
        # lib_path_text_box.setFixedHeight(30)
        # lib_path_text_box.setFixedWidth(800)
        # lib_path_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # lib_path_layout.addWidget(lib_path_text_box)

        # browse_button1 = QPushButton("Browse lib")
        # browse_button1.setFixedSize(110, 30)
        # browse_button1.clicked.connect(lambda: browse_path(self,lib_path_text_box))
        # lib_path_layout.addWidget(browse_button1)

        # self.eConLayout.addLayout(lib_path_layout)

        # Add the description HTML content
        description_html = """
            <html>
                <head>
                    <style>
                        body {
                            font-family: sans-serif;
                            margin: 0px;
                            padding: 0px;
                            background-color: white;
                            border: 4px solid  black;
                            font-size: 10pt; /* Adjust the font size as needed */
                        }

                        h1{
                            font-weight: bold;
                            font-size: 9pt;
                            color: #eeeeee;
                            padding: 10px;
                            background-color: #165982;
                            border: 4px outset  #0E324B;
                        }
                    </style>
                </head>

                <body>
                    <h1>About eSim Converter</h1>
                    <p>
                        <b>Pspice to eSim </b> will convert the PSpice Schematic and Library files to KiCad Schematic and
                        Library files respectively with proper mapping of the components and the wiring. By this way one 
                        will be able to simulate their schematic in PSpice and get the PCB layout in KiCad.</b> 
                        <br/><br/>
                        <b>LTspice to eSim </b> will convert symbols and schematic from LTspice to Kicad.The goal is to design and
                        simulate under LTspice and to automatically transfer the circuit under Kicad to draw the PCB.</b>
                    </p>
                </body>
            </html>
        """

        self.description_label = QLabel()
        self.description_label.setFixedHeight(160)
        self.description_label.setFixedWidth(950)
        self.description_label.setAlignment(Qt.AlignmentFlag.AlignBottom)
        self.description_label.setWordWrap(True)
        self.description_label.setText(description_html)
        self.eConLayout.addWidget(self.description_label)  # Add the description label to the layout

        self.eConWidget.setLayout(self.eConLayout)

        dock[dockName + str(count)] = QtWidgets.QDockWidget(dockName + str(count))
        self.apply_fullscreen_feature(
            dock[dockName + str(count)], self.eConWidget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea, dock[dockName + str(count)])
        self.tabifyDockWidget(dock['Welcome'], dock[dockName + str(count)])


        dock[dockName + str(count)].setVisible(True)
        dock[dockName + str(count)].setFocus()
        dock[dockName + str(count)].raise_()

        count = count + 1

    def modelEditor(self):
        """This function defines UI for model editor."""
        print("in model editor")
        global count

        projDir = self.obj_appconfig.current_project["ProjectName"]
        if projDir is None:
            """ when projDir is None that is clicking on subcircuit icon
                without any project selection """
            self.msg = Dialogs.make_error_message(self)
            self.msg.setModal(True)
            self.msg.setWindowTitle("Error Message")
            self.msg.showMessage(
                'Please select the project first.'
                ' You can either create new project or open existing project'
            )
            self.msg.exec()
            return
        projName = os.path.basename(projDir)
        dockName = f'Model Editor-{projName}-'

        self.modelwidget = QtWidgets.QWidget()

        self.modellayout = QtWidgets.QVBoxLayout()
        # No wrapper margins — the editor manages its own padding and should
        # fill the dock edge to edge rather than sit inside a dead border.
        self.modellayout.setContentsMargins(0, 0, 0, 0)
        self.modellayout.addWidget(ModelEditorclass())

        # Adding to main Layout
        self.modelwidget.setLayout(self.modellayout)

        dock[dockName +
             str(count)] = QtWidgets.QDockWidget(dockName
                                                 + str(count))
        self.apply_fullscreen_feature(
            dock[dockName + str(count)], self.modelwidget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea,
                           dock[dockName + str(count)])
        self.tabifyDockWidget(dock['Welcome'],
                              dock[dockName + str(count)])


        dock[dockName + str(count)].setVisible(True)
        dock[dockName + str(count)].setFocus()
        dock[dockName + str(count)].raise_()

        count = count + 1

    def _closeExistingConverters(self):
        """Tear down any open KiCad-to-Ngspice converter dock.

        The converter uses module-level globals and class-level TrackWidget
        state as its data bus, so two live converter docks would clobber each
        other -- silently writing one project's circuit into another's
        .cir.out. Enforce the single-consumer invariant the converter was
        written against by destroying any existing converter dock before
        opening a new one. Also stops the module `dock` dict from leaking a
        fresh entry on every open.
        """
        for key in [k for k in dock if k.startswith('Netlist-')]:
            d = dock.pop(key, None)
            if d is None:
                continue
            # Drop it from per-project bookkeeping so closing the project
            # later cannot double-free an already-deleted dock.
            for docks in self.obj_appconfig.dock_dict.values():
                if d in docks:
                    docks.remove(d)
            try:
                self.removeDockWidget(d)
                d.setParent(None)
                d.deleteLater()
            except RuntimeError:
                # Already deleted on the Qt side; nothing to do.
                pass

    def kicadToNgspiceEditor(self, clarg1, clarg2=None):
        """
        This function is creating Editor UI for Kicad to Ngspice conversion.
        """
        global count

        # Keep at most one converter live; see _closeExistingConverters.
        self._closeExistingConverters()

        projDir = self.obj_appconfig.current_project["ProjectName"]
        projName = os.path.basename(projDir)
        dockName = f'Netlist-{projName}-'

        self.kicadToNgspiceWidget = QtWidgets.QWidget()
        self.kicadToNgspiceLayout = QtWidgets.QVBoxLayout()
        self.kicadToNgspiceLayout.addWidget(MainWindow(clarg1, clarg2))

        self.kicadToNgspiceWidget.setLayout(self.kicadToNgspiceLayout)
        dock[dockName + str(count)] = \
            QtWidgets.QDockWidget(dockName + str(count))
        self.apply_fullscreen_feature(
            dock[dockName + str(count)], self.kicadToNgspiceWidget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea,
                           dock[dockName + str(count)])
        self.tabifyDockWidget(dock['Welcome'],
                              dock[dockName + str(count)])


        dock[dockName + str(count)].setVisible(True)
        dock[dockName + str(count)].setFocus()
        dock[dockName + str(count)].raise_()
        dock[dockName + str(count)].activateWindow()

        temp = self.obj_appconfig.current_project['ProjectName']
        if temp:
            self.obj_appconfig.dock_dict[temp].append(
                dock[dockName + str(count)]
            )
        count = count + 1

    def subcircuiteditor(self):
        """This function creates a widget for different subcircuit options."""
        global count

        projDir = self.obj_appconfig.current_project["ProjectName"]

        """ Checks projDir variable has valid value 
        & is not None before calling os.path.basename """

        if projDir is not None:
            projName = os.path.basename(projDir)
            dockName = f'Subcircuit-{projName}-'

            self.subcktWidget = QtWidgets.QWidget()
            self.subcktLayout = QtWidgets.QVBoxLayout()
            self.subcktLayout.addWidget(Subcircuit(self))

            self.subcktWidget.setLayout(self.subcktLayout)
            dock[dockName +
                str(count)] = QtWidgets.QDockWidget(dockName
                                                    + str(count))
            self.apply_fullscreen_feature(
                dock[dockName + str(count)], self.subcktWidget)
            self.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea,
                            dock[dockName + str(count)])
            self.tabifyDockWidget(dock['Welcome'],
                                dock[dockName + str(count)])


            dock[dockName + str(count)].setVisible(True)
            dock[dockName + str(count)].setFocus()
            dock[dockName + str(count)].raise_()

            count = count + 1

        else:
            """ when projDir is None that is clicking on subcircuit icon
                without any project selection """
            self.msg = Dialogs.make_error_message(self)
            self.msg.setModal(True)
            self.msg.setWindowTitle("Error Message")
            self.msg.showMessage(
                'Please select the project first.'
                ' You can either create new project or open existing project'
            )
            self.msg.exec()

    def show_welcome(self):
        """Bring the permanent Welcome (home) tab to the front.

        Backs the top-toolbar Home button: from anywhere in the app the user
        lands back on the Welcome dashboard to navigate out again.
        """
        welcome = dock.get('Welcome')
        if welcome is not None:
            welcome.setVisible(True)
            welcome.raise_()
            welcome.setFocus()

    def makerchip(self, select_vhdl=False):
        """This function creates a widget for different subcircuit options.

        ``select_vhdl`` opens the Flow Navigator straight on the VHDL / NGHDL
        path (backs the dedicated NGHDL launcher) instead of the default
        Verilog Author stage.
        """
        global count

        projDir = self.obj_appconfig.current_project["ProjectName"]
        if projDir is None:
            """ when projDir is None that is clicking on subcircuit icon
                without any project selection """
            self.msg = Dialogs.make_error_message(self)
            self.msg.setModal(True)
            self.msg.setWindowTitle("Error Message")
            self.msg.showMessage(
                'Please select the project first.'
                ' You can either create new project or open existing project'
            )
            self.msg.exec()
            return
        projName = os.path.basename(projDir)
        # Tab/dock label, matching the launcher action "Model Creation
        # (Verilog / VHDL)" and the sibling docks' "<Tool>-<proj>-<n>" form
        # (Simulation-RLC-2, Plotting-RLC-3).
        dockName = f'Model Creation-{projName}-'

        self.makerWidget = QtWidgets.QWidget()
        self.makerLayout = QtWidgets.QVBoxLayout()
        maker = makerchip(self)
        # NGHDL launcher: jump the Flow Navigator to the VHDL / NGHDL path so the
        # user lands on the digital-model stage directly.
        if select_vhdl:
            try:
                maker.flow._select_mode("vhdl")
            except Exception:
                pass
        self.makerLayout.addWidget(maker)

        self.makerWidget.setLayout(self.makerLayout)
        dock[dockName +
             str(count)] = QtWidgets.QDockWidget(dockName
                                                 + str(count))
        self.apply_fullscreen_feature(
            dock[dockName + str(count)], self.makerWidget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea,
                           dock[dockName + str(count)])
        self.tabifyDockWidget(dock['Welcome'],
                              dock[dockName + str(count)])

        # Verify-stage waveforms open as their own full-width eSim tab next to
        # this Model Creation dock. Bind the source dock + its navigator so
        # "Back to Verify" (and tab-close) return to the exact instance that
        # produced the plot, even with several Model Creation docks open.
        source_dock = dock[dockName + str(count)]
        maker.flow.waveformRequested.connect(
            lambda plot, src=source_dock, flow=maker.flow:
            self._show_waveform_dock(plot, src, flow))

        # No generic '.QWidget' box here: the legacy rounded-grey border boxed
        # every plain QWidget inside the panel (notably FlowNavigator's stage
        # tab strip, strangling the Author/Verify/Convert selector). The
        # FlowNavigator supplies its own header styling, so the panel sits
        # edge-to-edge with zero margins.
        self.makerLayout.setContentsMargins(0, 0, 0, 0)
        self.makerLayout.setSpacing(0)

        dock[dockName + str(count)].setVisible(True)
        dock[dockName + str(count)].setFocus()
        dock[dockName + str(count)].raise_()

        count = count + 1

    def _show_waveform_dock(self, plot, source_dock, flow):
        """Host a Verilog simulation waveform as its own full-width eSim tab.

        A re-simulate reuses the source dock's existing waveform tab (swapping
        in the new plot) instead of stacking tabs -- matching how Vivado /
        ModelSim keep a single waveform viewer that updates. The tab carries a
        "Back to Verify" control; that button and closing the tab both return to
        the Verify stage that produced the plot.
        """
        existing = self._wave_docks.get(source_dock)
        if existing is not None:
            # Swap the plot inside the live tab and bring it forward.
            old = existing._wave_plot
            if old is not None:
                existing._wave_layout.removeWidget(old)
                old.deleteLater()
            existing._wave_layout.addWidget(plot, 1)
            existing._wave_plot = plot
            existing.setVisible(True)
            existing.raise_()
            return

        wave_dock = WaveformDock("Waveform-" + source_dock.windowTitle(), self)

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # A slim header with the single round-trip control back to the editor.
        # objectName + selector so the band reliably paints (a bare-property
        # sheet on a plain QWidget often does not) and matches the Flow
        # Navigator's own header palette.
        header = QtWidgets.QWidget()
        header.setObjectName("waveHeader")
        # Theme-aware: #waveHeader is styled in style_{dark,light}.qss so it
        # tracks the active theme instead of the old hard-coded light palette.
        hrow = QtWidgets.QHBoxLayout(header)
        hrow.setContentsMargins(10, 6, 10, 6)
        back_btn = QtWidgets.QPushButton("← Back to Verify")
        back_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        back_btn.setProperty("cssClass", "secondary")
        back_btn.clicked.connect(
            lambda: self._return_to_verify(source_dock, flow))
        hrow.addWidget(back_btn)
        hrow.addStretch(1)
        layout.addWidget(header)
        layout.addWidget(plot, 1)

        # Remember the swappable plot + its layout for reuse on re-simulate.
        wave_dock._wave_plot = plot
        wave_dock._wave_layout = layout

        self.apply_fullscreen_feature(wave_dock, container)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea,
                           wave_dock)
        self.tabifyDockWidget(dock['Welcome'], wave_dock)

        self._wave_docks[source_dock] = wave_dock

        # Closing the tab drops the mapping and returns to Verify; the dock
        # deletes itself (WA_DeleteOnClose), so nothing leaks across cycles.
        def _on_closed():
            if self._wave_docks.get(source_dock) is wave_dock:
                del self._wave_docks[source_dock]
            self._return_to_verify(source_dock, flow)
        wave_dock.closed.connect(_on_closed)

        wave_dock.setVisible(True)
        wave_dock.setFocus()
        wave_dock.raise_()

    def _return_to_verify(self, source_dock, flow):
        """Bring the Model Creation dock that owns this plot forward and select
        its Verify stage."""
        try:
            flow.goto_verify()
        except Exception:
            pass
        source_dock.setVisible(True)
        source_dock.raise_()

    def usermanual(self):
        """This function creates a widget for user manual."""
        global count
        self.usermanualWidget = QtWidgets.QWidget()
        self.usermanualLayout = QtWidgets.QVBoxLayout()
        self.usermanualLayout.addWidget(UserManual())

        self.usermanualWidget.setLayout(self.usermanualLayout)
        dock['User Manual-' +
             str(count)] = QtWidgets.QDockWidget('User Manual-' + str(count))
        self.apply_fullscreen_feature(
            dock['User Manual-' + str(count)], self.usermanualWidget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea,
                           dock['User Manual-' + str(count)])
        self.tabifyDockWidget(dock['Welcome'],
                              dock['User Manual-' + str(count)])


        dock['User Manual-' + str(count)].setVisible(True)
        dock['User Manual-' + str(count)].setFocus()
        dock['User Manual-' + str(count)].raise_()

        count = count + 1

    def modelicaEditor(self, projDir):
        """This function sets up the UI for ngspice to modelica conversion."""
        global count

        projName = os.path.basename(projDir)
        dockName = f'Modelica-{projName}-'

        self.modelicaWidget = QtWidgets.QWidget()
        self.modelicaLayout = QtWidgets.QVBoxLayout()
        self.modelicaLayout.addWidget(OpenModelicaEditor(projDir))

        self.modelicaWidget.setLayout(self.modelicaLayout)
        dock[dockName + str(count)
             ] = QtWidgets.QDockWidget(dockName + str(count))
        self.apply_fullscreen_feature(
            dock[dockName + str(count)], self.modelicaWidget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea,
                           dock[dockName
                                + str(count)])
        self.tabifyDockWidget(dock['Welcome'], dock[dockName
                                                    + str(count)])

        dock[dockName + str(count)].setVisible(True)
        dock[dockName + str(count)].setFocus()
        dock[dockName + str(count)].raise_()

        temp = self.obj_appconfig.current_project['ProjectName']
        if temp:
            self.obj_appconfig.dock_dict[temp].append(
                dock[dockName + str(count)]
            )

        count = count + 1

    def closeDock(self):
        """
        This function checks for the project in **dock_dict**
        and closes it.
        """
        self.temp = self.obj_appconfig.current_project['ProjectName']
        for dockwidget in self.obj_appconfig.dock_dict[self.temp]:
            dockwidget.close()
