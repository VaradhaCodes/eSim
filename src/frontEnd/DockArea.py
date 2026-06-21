from PyQt6 import QtCore, QtWidgets
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

        # Drag-and-drop docking: a tool is undocked/redocked by dragging its
        # title bar (see widgets.DockTitleBar). Accept the drops here and keep a
        # lazily-built drop placeholder overlay.
        self.setAcceptDrops(True)
        self._dock_drop_overlay = None
        self._drag_target_dock = None

        for dockName in dockList:
            dock[dockName] = QtWidgets.QDockWidget(dockName)
            self.welcomeWidget = QtWidgets.QWidget()
            self.welcomeLayout = QtWidgets.QVBoxLayout()
            self.welcomeLayout.addWidget(Welcome())  # Call browser

            # Adding to main Layout
            self.welcomeWidget.setLayout(self.welcomeLayout)
            
            # Note: Explicitly not using apply_fullscreen_feature for the Welcome screen 
            # as it looks too big on the main home window.
            dock[dockName].setWidget(self.welcomeWidget)
            

            self.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea, dock[dockName])

        self.tabifiedDockWidgetActivated.connect(self.on_dock_activated)

        # Install tactile button motion ONCE here. Previously every call to
        # apply_fullscreen_feature() re-ran install_button_motion(self) on the
        # whole dock area, stacking a fresh event-filter on every existing
        # button each time a tool opened (O(n^2) filters / leak).
        from frontEnd.motion import install_button_motion
        install_button_motion(self)

        # Enable close buttons on bottom tabs

        # self.tabifyDockWidget(dock['Notes'],dock['Blank'])
        self.show()


    def tabifyDockWidget(self, first, second):
        super().tabifyDockWidget(first, second)
        self.enable_tab_close_buttons()

    def enable_tab_close_buttons(self):
        """Finds all QTabBars in the main window and enables their close buttons."""
        for tb in self.findChildren(QtWidgets.QTabBar):
            if not isinstance(tb.parent(), QtWidgets.QTabWidget):
                tb.setTabsClosable(True)
                try:
                    tb.tabCloseRequested.disconnect()
                except Exception:
                    pass
                tb.tabCloseRequested.connect(
                    lambda index, tab_bar=tb: self.handle_tab_close(index, tab_bar))

    # ---- drag-and-drop docking -------------------------------------------
    def _ensure_drop_overlay(self):
        if self._dock_drop_overlay is None:
            from frontEnd.widgets import DockDropOverlay
            self._dock_drop_overlay = DockDropOverlay(self)
        return self._dock_drop_overlay

    def begin_dock_drag(self, dock):
        """Called by a DockTitleBar as a drag starts: remember the dragged dock
        and raise the drop placeholder over the whole dock area for the drag."""
        self._drag_target_dock = dock
        ov = self._ensure_drop_overlay()
        ov.setGeometry(self.rect())
        ov.show_active()

    def end_dock_drag(self):
        """Called when the drag's QDrag.exec returns: hide the placeholder."""
        self._drag_target_dock = None
        if self._dock_drop_overlay is not None:
            self._dock_drop_overlay.hide()

    def dock_area_global_rect(self):
        """The dock area's rect in global screen coords.

        Used by DockTitleBar's non-Wayland redock watch to tell whether a
        free-floating tool window has been dragged back over the dock area.
        ``mapToGlobal`` is unreliable on Wayland, so this is only consulted off
        Wayland (the watch never runs there)."""
        try:
            tl = self.mapToGlobal(self.rect().topLeft())
            return QtCore.QRect(tl, self.size())
        except Exception:
            return QtCore.QRect()

    def point_over_dock_area(self, global_point):
        """True if a global-screen point falls inside the dock area."""
        try:
            return self.dock_area_global_rect().contains(global_point)
        except Exception:
            return False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._dock_drop_overlay is not None and self._dock_drop_overlay.isVisible():
            self._dock_drop_overlay.setGeometry(self.rect())

    # Fallback drop handlers (the raised overlay normally catches the drop, but
    # if a drop lands on the bare dock area these accept it so QDrag.exec still
    # returns MoveAction and DockTitleBar re-docks).
    def dragEnterEvent(self, event):
        from frontEnd.widgets import DockTitleBar
        if event.mimeData().hasFormat(DockTitleBar.MIME):
            event.setDropAction(QtCore.Qt.DropAction.MoveAction)
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        from frontEnd.widgets import DockTitleBar
        if event.mimeData().hasFormat(DockTitleBar.MIME):
            event.setDropAction(QtCore.Qt.DropAction.MoveAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        from frontEnd.widgets import DockTitleBar
        if event.mimeData().hasFormat(DockTitleBar.MIME):
            event.setDropAction(QtCore.Qt.DropAction.MoveAction)
            event.accept()
        else:
            event.ignore()

    def handle_tab_close(self, index, tab_bar):
        tab_text = tab_bar.tabText(index).replace('&', '').strip()
        # Strip Qt truncation ellipsis from end
        if tab_text.endswith('...'):
            tab_text = tab_text[:-3].strip()

        # Find the matching visible dock widget by title prefix
        for child in self.findChildren(QtWidgets.QDockWidget):
            if not child.isVisible():
                continue
            title = child.windowTitle().replace('&', '').strip()
            if title == tab_text or title.startswith(tab_text):
                child.close()
                try:
                    self.removeDockWidget(child)
                except Exception:
                    pass
                child.deleteLater()

                # Clean up global dock dict
                keys_to_delete = [k for k, v in dock.items() if v is child]
                for k in keys_to_delete:
                    del dock[k]

                main_view = self.get_main_view_reference()
                if main_view:
                    main_view.restore_console_area()
                return

        # Fallback: close the dock at this tab position by index in
        # the visible dock list (last resort when title match fails)
        visible = [d for d in self.findChildren(QtWidgets.QDockWidget) if d.isVisible()]
        if index < len(visible):
            child = visible[index]
            child.close()
            try:
                self.removeDockWidget(child)
            except Exception:
                pass
            child.deleteLater()
            keys_to_delete = [k for k, v in dock.items() if v is child]
            for k in keys_to_delete:
                del dock[k]
            main_view = self.get_main_view_reference()
            if main_view:
                main_view.restore_console_area()

    def apply_fullscreen_feature(self, dock_widget, original_widget):
        """Wraps a dock's inner widget with a Fullscreen pop-out button.

        The button stays in its original position (a thin strip pinned to the
        top-right of every tool window, just under the QDockWidget title bar).
        We only swap the unicode-glyph label for a proper SVG icon and tighten
        its chrome so it reads as a small toolbar control rather than a wide
        bilingual button.
        """
        import re
        from frontEnd.icon_paths import fullscreen_icon, dock_back_icon
        from frontEnd.widgets import FloatingDockHost
        title = dock_widget.windowTitle()
        disp_title = re.sub(r'-\d+$', '', title).rstrip('-').strip() or title

        # The tool content lives inside a rounded "card" (dockCard) that floats
        # inside a transparent host. The host's margins are the gap that reveals
        # the workspace behind it, and the host paints the faint card shadow.
        # The dock itself keeps its NATIVE title bar (see below) so Qt's own
        # drag-to-float / drag-to-re-dock / move works exactly like the Welcome
        # tab — the rounded card is just the dock's internal content.
        card = QtWidgets.QFrame()
        card.setObjectName("dockCard")
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # NOTE: we deliberately keep the dock's NATIVE title bar (no
        # setTitleBarWidget below). Qt's native title bar IS the dock's drag
        # handle, and native QDockWidget docking — drag out to undock, drag the
        # ghost preview anywhere, hover back over a dock area to re-dock — is
        # implemented by Qt with an internal rubber-band preview + reparent on
        # drop, NOT a real top-level window move. That is the one cross-window
        # gesture Wayland routes reliably (it never asks the compositor for a
        # window position or a global cursor), so it works on Wayland, X11,
        # Windows and macOS alike. The earlier custom title bar (QDrag /
        # startSystemMove) DISABLED this native flow and could not replace it on
        # Wayland — that was the "undock then freeze" bug. The native title bar
        # is styled to look like the card's quiet header strip (QDockWidget::title
        # in the QSS), and the Fullscreen / Close actions live on the card's own
        # top-right chrome overlay (built below), so nothing is lost.
        fs_icon = fullscreen_icon(14)
        close_icon_svg = None
        try:
            from frontEnd.icon_paths import close_icon as _ci
            close_icon_svg = _ci(14)
        except Exception:
            pass

        fs_btn = QtWidgets.QPushButton(" Fullscreen")
        fs_btn.setProperty("cssClass", "secondary")
        fs_btn.setProperty("dockPopButton", "true")
        fs_btn.setProperty("isPoppedOut", "false")
        fs_btn.setToolTip("Pop this tool out into its own window")
        fs_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        if fs_icon:
            fs_btn.setIcon(fs_icon)
        fs_btn.setStyleSheet(
            "QPushButton[dockPopButton=\"true\"] {"
            "  padding-left: 10px; padding-right: 10px;"
            "}"
        )

        close_btn = QtWidgets.QPushButton("  Close")
        close_btn.setProperty("cssClass", "danger")
        close_btn.setProperty("dockPopButton", "true")
        close_btn.setProperty("isCloseBtn", "true")
        close_btn.setToolTip("Close this tool window")
        close_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        if close_icon_svg:
            close_btn.setIcon(close_icon_svg)

        popout_state = {"win": None}
        def custom_close():
            if popout_state["win"]:
                popout_state["win"].close()
            if dock_widget.widget() != wrapper:
                dock_widget.setWidget(wrapper)
            dock_widget.setVisible(False)
            dock_widget.close()

            # Fully tear the dock down (mirror handle_tab_close). It used to be
            # only hidden, so every close left a hidden zombie tab member in the
            # dock area; those stale members corrupt QMainWindow's saved dock
            # layout, so the next tool of the same kind re-opens as a stuck
            # floating window. Remove it from the layout, drop the global ref,
            # and schedule deletion.
            try:
                self.removeDockWidget(dock_widget)
            except Exception:
                pass
            for _k in [k for k, v in dock.items() if v is dock_widget]:
                del dock[_k]
            dock_widget.deleteLater()

            # If the user closes a tool, we fallback to Welcome, so restore console
            main_view = self.get_main_view_reference()
            if main_view:
                main_view.restore_console_area()
            
        close_btn.clicked.connect(custom_close)

        layout.addWidget(original_widget)

        # Chrome buttons no longer sit on a strip ABOVE the tool — they overlay
        # the top-right CORNER of the tool card itself (on the same line as the
        # tool's own top tab bar), reclaiming the full title-strip height. The
        # overlay is a child of `card`, so it travels with the card into the
        # pop-out dialog automatically (no separate header strip to carry).
        chrome = QtWidgets.QWidget(card)
        chrome.setObjectName("dockToolChrome")
        chrome.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        ch = QtWidgets.QHBoxLayout(chrome)
        ch.setContentsMargins(6, 3, 6, 3)
        ch.setSpacing(6)
        ch.addWidget(fs_btn)
        ch.addWidget(close_btn)

        def _place_chrome():
            try:
                chrome.adjustSize()
                x = card.width() - chrome.width() - 10
                chrome.move(max(0, x), 6)
                chrome.raise_()
            except RuntimeError:
                pass

        # Keep the overlay pinned to the top-right whenever the card resizes
        # (dock resize, splitter drag, pop-out to a bigger window).
        _orig_card_resize = card.resizeEvent
        def _card_resize(e):
            _orig_card_resize(e)
            _place_chrome()
        card.resizeEvent = _card_resize
        QtCore.QTimer.singleShot(0, _place_chrome)

        # Float the card: transparent host (margins = the gap) paints the
        # faint shadow behind the rounded card. `wrapper` stays the moved-around
        # widget so the pop-out / re-dock flow below is unchanged.
        wrapper = FloatingDockHost(card)

        dock_widget.setWidget(wrapper)
        # No setTitleBarWidget(): keep the native title bar so Qt's own
        # drag-to-undock / drag-anywhere / hover-to-redock works (see the long
        # note above). Make the drag/float affordances explicit and ensure the
        # dock can be moved and floated.
        dock_widget.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable)

        def toggle_popout():
            if not popout_state["win"]:
                win = QtWidgets.QDialog(self.window())
                win.setWindowTitle(title)
                win.setWindowFlags(win.windowFlags() | QtCore.Qt.WindowType.WindowMaximizeButtonHint | QtCore.Qt.WindowType.WindowMinimizeButtonHint)

                win_layout = QtWidgets.QVBoxLayout(win)
                win_layout.setContentsMargins(0, 0, 0, 0)

                # The Fullscreen/Close buttons overlay the card's top-right
                # corner (children of `card`), so they travel with the card into
                # this dialog automatically — no separate header strip needed.
                win_layout.addWidget(wrapper)
                QtCore.QTimer.singleShot(0, _place_chrome)

                # Apply depth shadow to popped-out window
                try:
                    from frontEnd.motion import apply_popup_depth
                    apply_popup_depth(win)
                except Exception:
                    pass

                fs_btn.setText("  Dock to eSim")
                dock_back = dock_back_icon(14)
                if dock_back:
                    fs_btn.setIcon(dock_back)
                fs_btn.setProperty("isPoppedOut", "true")
                fs_btn.style().unpolish(fs_btn)
                fs_btn.style().polish(fs_btn)
                fs_btn.setToolTip("Put this tool back into the main window")

                def on_close(event):
                    # The dock may already be torn down (tool/app closed while
                    # popped out); putting the card back into a deleted dock
                    # raises RuntimeError and would crash eSim from this
                    # closeEvent. Guard it and just let the window close.
                    try:
                        dock_widget.setWidget(wrapper)
                    except RuntimeError:
                        popout_state["win"] = None
                        event.accept()
                        return
                    try:
                        fs_btn.setText("  Fullscreen")
                        fs_btn.setIcon(fs_icon)
                        fs_btn.setProperty("isPoppedOut", "false")
                        fs_btn.style().unpolish(fs_btn)
                        fs_btn.style().polish(fs_btn)
                        fs_btn.setToolTip("Pop this tool out into its own window")
                    except RuntimeError:
                        pass
                    popout_state["win"] = None
                    QtCore.QTimer.singleShot(0, _place_chrome)
                    event.accept()

                win.closeEvent = on_close
                popout_state["win"] = win
                win.resize(1000, 700)
                win.showMaximized()
            else:
                popout_state["win"].close()

        fs_btn.clicked.connect(toggle_popout)

        # Attach the already-installed tactile filter to just the two new
        # chrome buttons rather than re-installing across the whole dock area
        # (see DockArea.__init__ for the one-time install).
        filt = getattr(self, '_esim_press_motion_filter', None)
        if filt is not None:
            for b in (fs_btn, close_btn):
                b.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
                b.installEventFilter(filt)
        else:
            from frontEnd.motion import install_button_motion
            install_button_motion(self)

    def get_main_view_reference(self):
        """Get reference to the MainView widget."""
        parent = self.parent()
        while parent:
            if hasattr(parent, 'collapse_console_area'):
                return parent
            parent = parent.parent()
        return None

    def on_dock_activated(self, dock_widget):
        """Handle when any dock becomes active."""
        main_view = self.get_main_view_reference()
        if not main_view:
            return
            
        # If the welcome tab is activated, restore console, otherwise collapse it
        if dock_widget.windowTitle() == 'Welcome':
            main_view.restore_console_area()
        else:
            main_view.collapse_console_area()

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
        self.apply_fullscreen_feature(dock['Tips-' + str(count)], self.testWidget)
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
        self.apply_fullscreen_feature(dock[dockName + str(count)], self.plottingWidget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea,
                           dock[dockName + str(count)])
        self.tabifyDockWidget(dock['Welcome'],
                              dock[dockName + str(count)])
        
        # Track this as a plotting dock
        self.active_plotting_docks.add(dock[dockName + str(count)])

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
        self.apply_fullscreen_feature(dock[dockName + str(count)], self.ngspiceWidget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea,
                           dock[dockName + str(count)])
        self.tabifyDockWidget(dock['Welcome'],
                              dock[dockName
                                   + str(count)])

        # CSS
        

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

        # Description panel — previous version was an inline
        # `<html><style>...background-color: white; #165982</style</html>`
        # block that ignored the active theme. Now built as nested
        # native widgets: eyebrow + heading + body inherit colour from
        # style_*.qss via cssClass tokens, so the panel tracks theme
        # switches and accent swaps automatically.
        from PyQt6.QtWidgets import QFrame
        self.description_panel = QtWidgets.QFrame()
        self.description_panel.setObjectName('converterDescription')
        description_layout = QtWidgets.QVBoxLayout(self.description_panel)
        description_layout.setContentsMargins(18, 14, 18, 14)
        description_layout.setSpacing(6)

        eyebrow = QtWidgets.QLabel('ABOUT eSim CONVERTER')
        eyebrow.setProperty('cssClass', 'caps')
        description_layout.addWidget(eyebrow)

        heading = QtWidgets.QLabel('Schematic Format Converters')
        heading.setProperty('cssClass', 'heading')
        description_layout.addWidget(heading)

        body = QtWidgets.QLabel(
            "<b>Pspice to eSim</b> will convert the PSpice Schematic and "
            "Library files to KiCad Schematic and Library files "
            "respectively with proper mapping of the components and the "
            "wiring. This lets you simulate a schematic in PSpice and then "
            "lay it out as a PCB in KiCad."
            "<br/><br/>"
            "<b>LTspice to eSim</b> will convert symbols and schematics "
            "from LTspice to KiCad. The goal is to design and simulate "
            "under LTspice and to automatically transfer the circuit into "
            "KiCad for PCB drawing."
        )
        body.setWordWrap(True)
        body.setProperty('cssClass', 'subtle')
        body.setTextFormat(QtCore.Qt.TextFormat.RichText)
        description_layout.addWidget(body)

        # Back-compat: keep `description_label` as an alias for any
        # historic downstream callers; route it to the panel we
        # actually added to the layout.
        self.description_label = self.description_panel
        self.eConLayout.addWidget(self.description_panel)  # themed summary panel

        self.eConWidget.setLayout(self.eConLayout)

        dock[dockName + str(count)] = QtWidgets.QDockWidget(dockName + str(count))
        self.apply_fullscreen_feature(dock[dockName + str(count)], self.eConWidget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea, dock[dockName + str(count)])
        self.tabifyDockWidget(dock['Welcome'], dock[dockName + str(count)])

        # CSS
        

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
            self.msg = QtWidgets.QErrorMessage()
            self.msg.setModal(True)
            self.msg.setWindowTitle("Error Message")
            self.msg.showMessage(
                'Please select the project first.'
                ' You can either create new project or open existing project'
            )
            self.msg.exec()
            return
        projName = self.obj_appconfig.get_proj_stem() \
            or os.path.basename(projDir)
        dockName = f'Model Editor-{projName}-'

        self.modelwidget = QtWidgets.QWidget()

        self.modellayout = QtWidgets.QVBoxLayout()
        self.modellayout.addWidget(ModelEditorclass())

        # Adding to main Layout
        self.modelwidget.setLayout(self.modellayout)

        dock[dockName +
             str(count)] = QtWidgets.QDockWidget(dockName
                                                 + str(count))
        self.apply_fullscreen_feature(dock[dockName + str(count)], self.modelwidget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea,
                           dock[dockName + str(count)])
        self.tabifyDockWidget(dock['Welcome'],
                              dock[dockName + str(count)])

        # CSS
        

        dock[dockName + str(count)].setVisible(True)
        dock[dockName + str(count)].setFocus()
        dock[dockName + str(count)].raise_()

        count = count + 1

    def kicadToNgspiceEditor(self, clarg1, clarg2=None):
        """
        This function is creating Editor UI for Kicad to Ngspice conversion.
        """
        global count

        projDir = self.obj_appconfig.current_project["ProjectName"]
        projName = self.obj_appconfig.get_proj_stem() \
            or os.path.basename(projDir)
        dockName = f'Netlist-{projName}-'

        self.kicadToNgspiceWidget = QtWidgets.QWidget()
        self.kicadToNgspiceLayout = QtWidgets.QVBoxLayout()
        self.kicadToNgspiceLayout.addWidget(MainWindow(clarg1, clarg2))

        self.kicadToNgspiceWidget.setLayout(self.kicadToNgspiceLayout)
        dock[dockName + str(count)] = \
            QtWidgets.QDockWidget(dockName + str(count))
        self.apply_fullscreen_feature(dock[dockName + str(count)], self.kicadToNgspiceWidget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea,
                           dock[dockName + str(count)])
        self.tabifyDockWidget(dock['Welcome'],
                              dock[dockName + str(count)])

        # CSS
        

        dock[dockName + str(count)].setVisible(True)
        dock[dockName + str(count)].setFocus()
        dock[dockName + str(count)].raise_()
        dock['Ngspice-' + str(count)].raise_()


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
            projName = self.obj_appconfig.get_proj_stem() \
                or os.path.basename(projDir)
            dockName = f'Subcircuit-{projName}-'

            self.subcktWidget = QtWidgets.QWidget()
            self.subcktLayout = QtWidgets.QVBoxLayout()
            self.subcktLayout.addWidget(Subcircuit(self))

            self.subcktWidget.setLayout(self.subcktLayout)
            dock[dockName +
                str(count)] = QtWidgets.QDockWidget(dockName
                                                    + str(count))
            self.apply_fullscreen_feature(dock[dockName + str(count)], self.subcktWidget)
            self.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea,
                            dock[dockName + str(count)])
            self.tabifyDockWidget(dock['Welcome'],
                                dock[dockName + str(count)])

            # CSS
            

            dock[dockName + str(count)].setVisible(True)
            dock[dockName + str(count)].setFocus()
            dock[dockName + str(count)].raise_()

    
            count = count + 1

        else:
            """ when projDir is None that is clicking on subcircuit icon
                without any project selection """
            self.msg = QtWidgets.QErrorMessage()
            self.msg.setModal(True)
            self.msg.setWindowTitle("Error Message")
            self.msg.showMessage(
                'Please select the project first.'
                ' You can either create new project or open existing project'
            )
            self.msg.exec()

    def makerchip(self):
        """This function creates a widget for different subcircuit options."""
        global count

        projDir = self.obj_appconfig.current_project["ProjectName"]
        if projDir is None:
            """ when projDir is None that is clicking on subcircuit icon
                without any project selection """
            self.msg = QtWidgets.QErrorMessage()
            self.msg.setModal(True)
            self.msg.setWindowTitle("Error Message")
            self.msg.showMessage(
                'Please select the project first.'
                ' You can either create new project or open existing project'
            )
            self.msg.exec()
            return
        projName = self.obj_appconfig.get_proj_stem() \
            or os.path.basename(projDir)
        dockName = f'Makerchip-{projName}-'

        self.makerWidget = QtWidgets.QWidget()
        self.makerLayout = QtWidgets.QVBoxLayout()
        self.makerLayout.addWidget(makerchip(self))

        self.makerWidget.setLayout(self.makerLayout)
        dock[dockName +
             str(count)] = QtWidgets.QDockWidget(dockName
                                                 + str(count))
        self.apply_fullscreen_feature(dock[dockName + str(count)], self.makerWidget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea,
                           dock[dockName + str(count)])
        self.tabifyDockWidget(dock['Welcome'],
                              dock[dockName + str(count)])

        # Track this as a plotting dock so console is collapsed
        self.active_plotting_docks.add(dock[dockName + str(count)])
        
        dock[dockName + str(count)].setVisible(True)
        dock[dockName + str(count)].setFocus()
        dock[dockName + str(count)].raise_()

        # Collapse console immediately
        main_view = self.get_main_view_reference()
        if main_view:
            QtCore.QTimer.singleShot(100, main_view.collapse_console_area)

        count = count + 1

    def usermanual(self):
        """This function creates a widget for user manual."""
        global count
        self.usermanualWidget = QtWidgets.QWidget()
        self.usermanualLayout = QtWidgets.QVBoxLayout()
        self.usermanualLayout.addWidget(UserManual())

        self.usermanualWidget.setLayout(self.usermanualLayout)
        dock['User Manual-' +
             str(count)] = QtWidgets.QDockWidget('User Manual-' + str(count))
        self.apply_fullscreen_feature(dock['User Manual-' + str(count)], self.usermanualWidget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea,
                           dock['User Manual-' + str(count)])
        self.tabifyDockWidget(dock['Welcome'],
                              dock['User Manual-' + str(count)])

        # CSS
        

        dock['User Manual-' + str(count)].setVisible(True)
        dock['User Manual-' + str(count)].setFocus()
        dock['User Manual-' + str(count)].raise_()


        count = count + 1

    def modelicaEditor(self, projDir):
        """This function sets up the UI for ngspice to modelica conversion."""
        global count

        projName = self.obj_appconfig.get_proj_stem() \
            or os.path.basename(projDir)
        dockName = f'Modelica-{projName}-'

        self.modelicaWidget = QtWidgets.QWidget()
        self.modelicaLayout = QtWidgets.QVBoxLayout()
        self.modelicaLayout.addWidget(OpenModelicaEditor(projDir))

        self.modelicaWidget.setLayout(self.modelicaLayout)
        dock[dockName + str(count)
             ] = QtWidgets.QDockWidget(dockName + str(count))
        self.apply_fullscreen_feature(dock[dockName + str(count)], self.modelicaWidget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea,
                           dock[dockName
                                + str(count)])
        self.tabifyDockWidget(dock['Welcome'], dock[dockName
                                                    + str(count)])

        dock[dockName + str(count)].setVisible(True)
        dock[dockName + str(count)].setFocus()
        dock[dockName + str(count)].raise_()


        # CSS
        
        temp = self.obj_appconfig.current_project['ProjectName']
        if temp:
            self.obj_appconfig.dock_dict[temp].append(
                dock[dockName + str(count)]
            )

        count = count + 1

    def closeDock(self):
        """
        This function checks for the project in **dock_dict**
        and closes it while cleaning up global references to prevent memory leaks.
        """
        self.temp = self.obj_appconfig.current_project['ProjectName']
        if self.temp in self.obj_appconfig.dock_dict:
            for dockwidget in self.obj_appconfig.dock_dict[self.temp]:
                dockwidget.close()
                dockwidget.deleteLater()
                # Clean up from global dock dictionary
                keys_to_delete = [k for k, v in dock.items() if v == dockwidget]
                for k in keys_to_delete:
                    del dock[k]
            self.obj_appconfig.dock_dict[self.temp] = []
