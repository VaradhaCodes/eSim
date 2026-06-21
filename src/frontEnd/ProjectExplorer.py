from PyQt6 import QtCore, QtGui, QtWidgets
import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from configuration.Appconfig import Appconfig
from projManagement.Validation import Validation
from projManagement.projectPaths import resolve_stem

class ProjectTreeWidget(QtWidgets.QTreeWidget):
    orderChanged = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(False) # Turn off default drop indicator
        self.setAnimated(True)
        
        self.dummy = None
        self.dragged_item = None
        self.anim = None
        self._ghost = None
        self._ghost_hot = QtCore.QPoint(15, 15)

    def _move_ghost(self, local_pos):
        """Move the drag ghost to a viewport-local cursor position. The ghost
        is a CHILD of the viewport (not a top-level window): a top-level
        overlay froze centre-screen because Wayland blocks absolute window
        moves, while a child widget positions reliably on any compositor."""
        if self._ghost is not None:
            self._ghost.move(local_pos - self._ghost_hot)
            self._ghost.raise_()

    def _kill_ghost(self):
        """Tear down the drag ghost overlay."""
        if self._ghost is not None:
            self._ghost.hide()
            self._ghost.deleteLater()
            self._ghost = None

    def startDrag(self, supportedActions):
        items = self.selectedItems()
        if not items:
            return
        self.dragged_item = items[0]
        
        if self.dragged_item.parent() is not None:
            self.dragged_item = None
            return
            
        text = self.dragged_item.text(0)
        font = self.font()
        font.setBold(True)
        fm = QtGui.QFontMetrics(font)
        text_rect = fm.boundingRect(text)
        
        width = text_rect.width() + 34
        height = max(34, text_rect.height() + 16)
        # Do NOT force a device-pixel-ratio on a QDrag pixmap. On several Linux
        # compositors (X11 + Wayland) a DPR-scaled drag pixmap renders
        # invisibly — the ghost never appears under the cursor. A plain
        # logical-size pixmap always shows. Crispness loss on 4K is negligible
        # next to "the drag ghost is invisible".
        pixmap = QtGui.QPixmap(width + 18, height + 18)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)

        dark = self.palette().color(QtGui.QPalette.ColorRole.Window).lightness() < 128
        try:
            from frontEnd import tokens as _tok
            _t = _tok.theme(dark)
            accent_a = QtGui.QColor(_t["accent"])
            accent_b = QtGui.QColor(_t["accent_2"])
        except Exception:
            accent_a = QtGui.QColor("#53D7FF" if dark else "#0077A8")
            accent_b = QtGui.QColor("#9B7CFF" if dark else "#6D5DF6")
        surface = QtGui.QColor(18, 30, 51, 235) if dark else QtGui.QColor(255, 255, 255, 242)
        text_color = QtGui.QColor("#F8FBFF" if dark else "#142033")

        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        # Bake the translucency into the pixmap (the ghost is a child widget,
        # which ignores windowOpacity) → reads as a "transparent version of
        # itself" gliding through the list.
        painter.setOpacity(0.82)

        for radius, alpha, dy in ((24, 42, 7), (16, 28, 4), (8, 18, 2)):
            painter.setBrush(QtGui.QColor(0, 0, 0, alpha))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.drawRoundedRect(8, 8 + dy, width, height, 14, 14)

        grad = QtGui.QLinearGradient(8, 8, width + 8, height + 8)
        grad.setColorAt(0.0, surface)
        grad.setColorAt(1.0, QtGui.QColor(surface).lighter(112 if dark else 101))
        painter.setBrush(grad)
        painter.setPen(QtGui.QPen(accent_a, 1.2))
        painter.drawRoundedRect(8, 8, width, height, 14, 14)

        accent_line = QtGui.QLinearGradient(10, 10, width - 10, 10)
        accent_line.setColorAt(0.0, accent_a)
        accent_line.setColorAt(1.0, accent_b)
        painter.setPen(QtGui.QPen(QtGui.QBrush(accent_line), 2))
        painter.drawLine(22, 12, width - 10, 12)

        painter.setFont(font)
        painter.setPen(text_color)
        painter.drawText(24, 8, width - 32, height, QtCore.Qt.AlignmentFlag.AlignVCenter, text)
        painter.end()
        
        drag = QtGui.QDrag(self)
        # Do NOT rely on QDrag.setPixmap for the ghost: on this WM the native
        # drag pixmap renders invisibly. Give the native drag an empty pixmap
        # and show our OWN ghost (a child of the viewport) that tracks the
        # cursor in dragMoveEvent.
        _blank = QtGui.QPixmap(1, 1)
        _blank.fill(QtCore.Qt.GlobalColor.transparent)
        drag.setPixmap(_blank)
        drag.setHotSpot(QtCore.QPoint(0, 0))
        drag.setMimeData(self.model().mimeData([self.indexFromItem(self.dragged_item)]))

        self._ghost_hot = QtCore.QPoint(15, 15)
        # Child of the viewport — NOT a top-level window. A top-level overlay
        # froze centre-screen (Wayland ignores absolute window moves); a child
        # widget moves in viewport-local coords and works on any compositor.
        self._ghost = QtWidgets.QLabel(self.viewport())
        self._ghost.setObjectName("projectDragGhost")
        self._ghost.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._ghost.setPixmap(pixmap)
        self._ghost.resize(pixmap.size())
        self._move_ghost(self.viewport().mapFromGlobal(QtGui.QCursor.pos()))
        self._ghost.show()
        
        idx = self.indexOfTopLevelItem(self.dragged_item)
        # Gap height = the dragged row's real height, so the placeholder is the
        # exact size of the item that left → no growing/shrinking gap.
        self._dummy_h = max(26, self.visualItemRect(self.dragged_item).height() or 28)
        self.dragged_item.setHidden(True)

        self.dummy = QtWidgets.QTreeWidgetItem()
        self.dummy.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
        self.dummy.setBackground(0, QtCore.Qt.GlobalColor.transparent)
        self.dummy.setSizeHint(0, QtCore.QSize(100, self._dummy_h))
        self.insertTopLevelItem(idx, self.dummy)
        
        action = drag.exec(supportedActions, QtCore.Qt.DropAction.MoveAction)

        self._kill_ghost()
        if self.dragged_item:
            self.dragged_item.setHidden(False)
            self.dragged_item = None
        if self.dummy:
            if self.anim:
                self.anim.stop()
            dummy_idx = self.indexOfTopLevelItem(self.dummy)
            if dummy_idx != -1:
                self.takeTopLevelItem(dummy_idx)
            self.dummy = None

    def dragMoveEvent(self, event):
        event.accept()
        # keep the translucent ghost under the cursor (viewport-local coords)
        self._move_ghost(event.position().toPoint())
        target = self.itemAt(event.position().toPoint())
        
        if not target or target.parent() is not None:
            return
            
        target_idx = self.indexOfTopLevelItem(target)
        if self.dummy:
            dummy_idx = self.indexOfTopLevelItem(self.dummy)
            if dummy_idx != -1 and dummy_idx != target_idx:
                
                target_rect = self.visualItemRect(target)
                mouse_y = event.position().toPoint().y()
                
                # Anti-jitter logic: Only swap if mouse cleanly crosses the center of the target item
                if target_idx > dummy_idx and mouse_y < target_rect.center().y():
                    return
                if target_idx < dummy_idx and mouse_y > target_rect.center().y():
                    return
                
                self.takeTopLevelItem(dummy_idx)
                
                insert_idx = target_idx
                if target_idx > dummy_idx:
                    insert_idx -= 1
                    
                self.insertTopLevelItem(insert_idx, self.dummy)
                # Keep the gap a fixed height that simply relocates. The old
                # per-swap QVariantAnimation animated the gap 0->34 on every
                # swap, so it collapsed and re-grew each time the placeholder
                # moved — that was the jitter. A constant-size gap glides.
                self.dummy.setSizeHint(0, QtCore.QSize(100, self._dummy_h))

    def dropEvent(self, event):
        self._kill_ghost()
        if self.dummy and self.dragged_item:
            idx = self.indexOfTopLevelItem(self.dummy)
            self.takeTopLevelItem(idx)
            self.dummy = None
            
            old_idx = self.indexOfTopLevelItem(self.dragged_item)
            self.takeTopLevelItem(old_idx)
            self.insertTopLevelItem(idx, self.dragged_item)
            self.dragged_item.setHidden(False)
            self.setCurrentItem(self.dragged_item)
            self.dragged_item = None
            self.orderChanged.emit()
            
        event.accept()


# This is main class for Project Explorer Area.
class ProjectExplorer(QtWidgets.QWidget):
    """
    This class contains function:

        - One work as a constructor(__init__).
        - For saving data.
        - for renaming project.
        - for refreshing project.
        - for removing project.
    """

    def __init__(self):
        """
        This method is doing following tasks:
            - Working as a constructor for class ProjectExplorer.
            - view of project explorer area.
        """
        QtWidgets.QWidget.__init__(self)
        self.obj_appconfig = Appconfig()
        self.obj_validation = Validation()
        self.treewidget = ProjectTreeWidget()
        self.treewidget.setAnimated(True) # Smooth drop-down animation for branch expansion
        self.treewidget.orderChanged.connect(self.saveProjectOrder)
        self.main_layout = QtWidgets.QVBoxLayout()
        self.fs_watcher = QtCore.QFileSystemWatcher()
        header = QtWidgets.QTreeWidgetItem(["Projects", "path"])
        self.treewidget.setHeaderItem(header)
        self.treewidget.setColumnHidden(1, True)

        # CSS
        _BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        init_path = _BASE_DIR + os.sep
        # We assemble the stylesheet as a string but defer setStyleSheet until
        # first show via _apply_tree_stylesheet(), so a freshly-applied theme
        # (which arrives as a QPalette + QSS on the application) takes effect on
        # the tree at construction. The viewport also gets its own sheet so
        # dark-mode surfaces don't leave a white sub-rectangle behind.
        self._base_dir = init_path
        self._dark_mode = bool(QtGui.QGuiApplication.styleHints().colorScheme() ==
                               QtCore.Qt.ColorScheme.Dark)

        self.treewidget.setObjectName('projectTree')
        self._apply_tree_stylesheet()

        for parents, children in list(
                self.obj_appconfig.project_explorer.items()):
            os.path.join(parents)
            if os.path.exists(parents):
                pathlist = parents.split(os.sep)
                parentnode = QtWidgets.QTreeWidgetItem(
                    self.treewidget, [pathlist[-1], parents]
                )
                parentnode.setIcon(0, QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DirIcon))
                parentnode.setFlags(parentnode.flags() | QtCore.Qt.ItemFlag.ItemIsDragEnabled | QtCore.Qt.ItemFlag.ItemIsDropEnabled)
            for files in children:
                childnode = QtWidgets.QTreeWidgetItem(
                    parentnode, [files, os.path.join(parents, files)]
                )
                childnode.setIcon(0, QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileIcon))
                self.fs_watcher.addPath(parents)
        self.main_layout.addWidget(self.treewidget)
        
        try:
            from frontEnd.elevation import elevate
            elevate(self.treewidget, "e2")
        except Exception:
            eff = QtWidgets.QGraphicsDropShadowEffect(self.treewidget)
            eff.setBlurRadius(22)
            eff.setOffset(0, 6)
            eff.setColor(QtGui.QColor(0, 0, 0, 62))
            self.treewidget.setGraphicsEffect(eff)
        
        # We must NOT refresh instantly on expand because destroying and recreating
        # the tree children mid-expansion completely breaks the native QTreeWidget
        # drop-down animation, making it look broken/jittery.
        # The fs_watcher handles directory updates perfectly anyway.
        # self.treewidget.expanded.connect(self.refreshInstant)
        self.treewidget.doubleClicked.connect(self.openProject)
        self.treewidget.itemClicked.connect(self._toggle_expansion_on_click)
        self.treewidget.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.treewidget.customContextMenuRequested.connect(self.openMenu)
        self.setLayout(self.main_layout)
        self.show()

    def _toggle_expansion_on_click(self, item, column):
        # Only toggle if it's a top-level project folder (they have children)
        if item.childCount() > 0:
            item.setExpanded(not item.isExpanded())
            
        # Ensure that simply clicking ANY item in the tree (file or folder)
        # sets its parent project as the "active" project for the rest of eSim.
        top_level_item = item if item.childCount() > 0 else item.parent()
        if top_level_item:
            projectPath = str(top_level_item.text(1))
            self.obj_appconfig.set_current_project(projectPath)

    def _apply_tree_stylesheet(self) -> None:
        sheet = (
            "#projectTree { "
            "  background-color: palette(base); "
            "  alternate-background-color: palette(alternate-base); "
            "  border: 1px solid palette(mid); "
            "  border-radius: 14px; "
            "  padding: 6px; "
            "  outline: 0; "
            "} "
            "#projectTree::item { "
            "  min-height: 26px; "
            "  padding: 6px 9px; "
            "  border-radius: 7px; "
            "  margin: 1px 2px; "
            "  border: 1px solid transparent; "
            "} "
            "#projectTree::item:hover { "
            "  background-color: palette(alternate-base); "
            "  border: 1px solid palette(midlight); "
            "} "
            "#projectTree::item:selected, #projectTree::item:selected:active { "
            "  background-color: palette(alternate-base); "
            "  color: palette(text); "
            "  border: 2px solid palette(highlight); "
            "} "
            # The branch cell (indent column left of a name) shows a stray
            # highlighted square on select/hover: the view paints the full-row
            # highlight FIRST, then the branch on top, so a transparent branch
            # just reveals it. Paint LEAF branches (sub-files, :!has-children)
            # with the tree's own base colour so their empty box disappears.
            # Parent projects (:has-children) are left highlighted on purpose —
            # their branch holds the dynamic expand/collapse arrow, so it is not
            # an empty box.
            "#projectTree::branch:!has-children:selected, "
            "#projectTree::branch:!has-children:selected:active, "
            "#projectTree::branch:!has-children:hover { background-color: palette(base); } "
            "#projectTree::branch:has-children:!has-siblings:closed, "
            "#projectTree::branch:closed:has-children:has-siblings { "
            "  border-image: none; "
            "} "
            "#projectTree::branch:open:has-children:!has-siblings, "
            "#projectTree::branch:open:has-children:has-siblings { "
            "  border-image: none; "
            "} "
            "#projectTree::drop-indicator { "
            "  height: 3px; "
            "  background-color: palette(highlight); "
            "  border-radius: 2px; "
            "  margin: 0 4px; "
            "}"
        )
        self.treewidget.setStyleSheet(sheet)
        vp_color = "palette(base)"
        self.treewidget.viewport().setStyleSheet(
            f"QWidget {{ background-color: {vp_color}; }}"
        )

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        # Application-level palette changes (theme or accent) propagate via
        # QEvent::PaletteChange. Reinstalling the stylesheet is enough to
        # refresh ``palette(...)`` lookups.
        if event.type() in (
            QtCore.QEvent.Type.PaletteChange,
            QtCore.QEvent.Type.StyleChange,
        ):
            try:
                self._apply_tree_stylesheet()
            except Exception:
                pass
    
    def saveProjectOrder(self):
        new_dict = {}
        for i in range(self.treewidget.topLevelItemCount()):
            item = self.treewidget.topLevelItem(i)
            path = item.text(1)
            if path in self.obj_appconfig.project_explorer:
                new_dict[path] = self.obj_appconfig.project_explorer[path]
                
        # Append any items that might have been skipped to prevent data loss
        for k, v in self.obj_appconfig.project_explorer.items():
            if k not in new_dict:
                new_dict[k] = v
                
        self.obj_appconfig.project_explorer = new_dict
        with open(self.obj_appconfig.dictPath["path"], 'w') as f:
            json.dump(self.obj_appconfig.project_explorer, f)
            

    def handleDirectoryChanged(self, path):
        for i in range(self.treewidget.topLevelItemCount()):
            item = self.treewidget.topLevelItem(i)
            if item.text(1) == path and item.isExpanded():
                index = self.treewidget.indexFromItem(item)
                self.refreshProject(indexItem=index)

    def refreshInstant(self):
        for i in range(self.treewidget.topLevelItemCount()):
            if self.treewidget.topLevelItem(i).isExpanded():
                index = self.treewidget.indexFromItem(
                    self.treewidget.topLevelItem(i))
                self.refreshProject(indexItem=index)

    def addTreeNode(self, parents, children):
        os.path.join(parents)
        pathlist = parents.split(os.sep)
        parentnode = QtWidgets.QTreeWidgetItem(
            self.treewidget, [pathlist[-1], parents]
        )
        parentnode.setIcon(0, QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DirIcon))
        parentnode.setFlags(parentnode.flags() | QtCore.Qt.ItemFlag.ItemIsDragEnabled | QtCore.Qt.ItemFlag.ItemIsDropEnabled)
        for files in children:
            childnode = QtWidgets.QTreeWidgetItem(
                parentnode, [files, os.path.join(parents, files)]
            )
            childnode.setIcon(0, QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileIcon))

        # Initialise running-state slots under this project. Without a
        # guard, the very first project added on a fresh install would
        # raise KeyError because proc_dict/dock_dict are plain dicts.
        project_name = self.obj_appconfig.current_project.get('ProjectName')
        if not project_name:
            return
        self.obj_appconfig.proc_dict.setdefault(project_name, [])
        self.obj_appconfig.dock_dict.setdefault(project_name, [])

    def openMenu(self, position):
        indexes = self.treewidget.selectedIndexes()
        if not indexes:
            return

        level = 0
        index = indexes[0]
        while index.parent().isValid():
            index = index.parent()
            level += 1

        menu = QtWidgets.QMenu(self.treewidget)
        try:
            from frontEnd.motion import make_menu_rounded
            make_menu_rounded(menu)
        except Exception:
            pass
        if level == 0:
            renameProject = menu.addAction(self.tr("Rename Project"))
            renameProject.triggered.connect(self.renameProject)
            menu.addSeparator()
            deleteproject = menu.addAction(self.tr("Remove Project"))
            deleteproject.setIcon(QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TrashIcon))
            deleteproject.triggered.connect(self.removeProject)
            menu.addSeparator()
            refreshproject = menu.addAction(self.tr("Refresh"))
            refreshproject.setIcon(QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_BrowserReload))
            refreshproject.triggered.connect(self.refreshProject)
        elif level == 1:
            openfile = menu.addAction(self.tr("Open"))
            openfile.triggered.connect(self.openProject)
            snapshot = menu.addAction(self.tr("Snapshot"))
            snapshot.triggered.connect(self.takeSnapshot)

        menu.exec(self.treewidget.viewport().mapToGlobal(position))

    def openProject(self):
        item = self.treewidget.currentItem()
        if not item: return
        filename = str(item.text(0))
        self.filePath = str(item.text(1))

        if (os.path.isfile(str(self.filePath))):
            from frontEnd.SpiceEditor import SpiceEditorWindow
            self.textwindow = SpiceEditorWindow(self.filePath)
            
            # Center on parent
            parent_geo = self.window().geometry()
            x = parent_geo.x() + (parent_geo.width() - 800) // 2
            y = parent_geo.y() + (parent_geo.height() - 600) // 2
            self.textwindow.move(x, y)
            
            self.textwindow.show()
        else:
            self.refreshProject(self.filePath)

            self.obj_appconfig.print_info(
                'The current project is: ' + self.filePath
            )

            self.obj_appconfig.set_current_project(str(self.filePath))
            # setdefault, not assignment: addTreeNode runs again on every refresh,
            # and clobbering these would drop the PIDs/docks already tracked for an
            # open project (orphaning its KiCad/ngspice windows on close).
            projName = self.obj_appconfig.current_project['ProjectName']
            self.obj_appconfig.proc_dict.setdefault(projName, [])
            self.obj_appconfig.dock_dict.setdefault(projName, [])

    def removeProject(self):
        """
        This function removes the project in explorer area by right \
        clicking on project and selecting remove option.
        """
        item = self.treewidget.currentItem()
        if not item: return
        filePath = str(item.text(1))
        
        from frontEnd.dialogs import ask_question
        if not ask_question(self.treewidget, "Remove Project", f"Are you sure you want to remove the project from the explorer?\n(Files on disk will NOT be deleted)"):
            return

        self.int = self.treewidget.indexOfTopLevelItem(item)
        if self.int >= 0:
            self.treewidget.takeTopLevelItem(self.int)

        if self.obj_appconfig.current_project["ProjectName"] == filePath:
            self.obj_appconfig.set_current_project(None)

        del self.obj_appconfig.project_explorer[filePath]
        with open(self.obj_appconfig.dictPath["path"], 'w') as f:
            json.dump(self.obj_appconfig.project_explorer, f)

    def refreshProject(self, filePath=None, indexItem=None):
        """
        This function refresh the project in explorer area by right \
        clicking on project and selecting refresh option.
        """

        if not filePath or filePath is None:
            if indexItem is None:
                item = self.treewidget.currentItem()
            else:
                item = self.treewidget.itemFromIndex(indexItem)
                
            if not item: return
            filePath = str(item.text(1))

        if os.path.exists(filePath):
            if not os.path.isdir(filePath):
                return False
                
            filelistnew = os.listdir(filePath)
            if indexItem is None:
                parentnode = self.treewidget.currentItem()
            else:
                parentnode = self.treewidget.itemFromIndex(indexItem)
                
            if not parentnode: return False
                
            count = parentnode.childCount()
            for i in range(count):
                parentnode.removeChild(parentnode.child(0))
            for files in filelistnew:
                childnode = QtWidgets.QTreeWidgetItem(
                    parentnode, [files, os.path.join(filePath, files)]
                )
                childnode.setIcon(0, QtWidgets.QApplication.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileIcon))

            self.obj_appconfig.project_explorer[filePath] = filelistnew
            try:
                with open(self.obj_appconfig.dictPath["path"], 'w') as f:
                    json.dump(self.obj_appconfig.project_explorer, f)
            except Exception:
                pass
            return True

        else:
            print(f"Selected project not found on disk: {filePath}")
            return False

    def renameProject(self):
        """
        This function renames the project present in project explorer area.
        It validates first:

            - If project names is not empty.
            - Project name does not contain spaces between them.
            - Project name is different between what it was earlier.
            - Project name should not exist.

        After project name is changed, it recreates the project explorer tree.
        """
        item = self.treewidget.currentItem()
        if not item: return
        self.baseFileName = str(item.text(0))
        filePath = str(item.text(1))

        newBaseFileName, ok = QtWidgets.QInputDialog.getText(
            self, 'Rename Project', 'Project Name:',
            QtWidgets.QLineEdit.EchoMode.Normal, self.baseFileName
        )

        if ok and newBaseFileName:
            newBaseFileName = str(newBaseFileName)

            if not newBaseFileName.strip():
                print("Project name cannot be empty")
                print("==================")
                msg = QtWidgets.QErrorMessage(self)
                msg.setModal(True)
                msg.setWindowTitle("Error Message")
                msg.showMessage('The project name cannot be empty')
                msg.exec()

            elif self.baseFileName == newBaseFileName:
                print("Project name has to be different")
                print("==================")
                msg = QtWidgets.QErrorMessage(self)
                msg.setModal(True)
                msg.setWindowTitle("Error Message")
                msg.showMessage('The project name has to be different')
                msg.exec()

            elif self.refreshProject(filePath):

                projectPath = None
                projectFiles = None

                for parents, children in list(
                        self.obj_appconfig.project_explorer.items()):
                    if filePath == parents:
                        if os.path.exists(parents):
                            projectPath, projectFiles = parents, children
                        break

                self.workspace = \
                    self.obj_appconfig.default_workspace['workspace']
                newBaseFileName = str(newBaseFileName).rstrip().lstrip()
                projDir = os.path.join(self.workspace, str(newBaseFileName))

                reply = self.obj_validation.validateNewproj(str(projDir))

                if not (projectPath and projectFiles):
                    print("Selected project not found")
                    print("Project Path :", projectPath)
                    print("Project Files :", projectFiles)
                    print("==================")
                    msg = QtWidgets.QErrorMessage(self)
                    msg.setModal(True)
                    msg.setWindowTitle("Error Message")
                    msg.showMessage('Selected project does not exist.')
                    msg.exec()

                elif reply == "VALID":
                    # rename project folder
                    updatedProjectFiles = []

                    # Inner files are named after the project *stem* (resolved
                    # from the .proj anchor), which may differ from the folder
                    # name. Match/replace on the stem so files are renamed even
                    # when the folder was named differently; renaming to the new
                    # name re-aligns folder and stem.
                    oldStem = resolve_stem(projectPath, 'proj')[0] \
                        or self.baseFileName

                    updatedProjectPath = newBaseFileName.join(
                        projectPath.rsplit(self.baseFileName, 1))
                    print("Renaming " + projectPath + " to " +
                          updatedProjectPath)
                    os.rename(projectPath, updatedProjectPath)

                    projectFiles = os.listdir(updatedProjectPath)
                    # rename files matching project name
                    try:
                        for projectFile in projectFiles:
                            if oldStem in projectFile:
                                oldFilePath = os.path.join(updatedProjectPath,
                                                           projectFile)
                                projectFile = projectFile.replace(
                                    oldStem, newBaseFileName, 1)
                                newFilePath = os.path.join(
                                    updatedProjectPath, projectFile)
                                print("Renaming " + oldFilePath + " to " +
                                      newFilePath)
                                os.rename(oldFilePath, newFilePath)
                                updatedProjectFiles.append(projectFile)

                    except Exception as e:
                        print("==================")
                        print("Error! Revert renaming project")

                        # Revert updatedProjectFiles
                        for projectFile in updatedProjectFiles:
                            print("Reverting the rename of " +
                                  newFilePath + " to " + oldFilePath)
                            newFilePath = os.path.join(
                                            updatedProjectPath, projectFile)
                            projectFile = projectFile.replace(
                                    newBaseFileName, oldStem, 1)
                            oldFilePath = os.path.join(
                                    updatedProjectPath, projectFile)
                            os.rename(newFilePath, oldFilePath)

                        # Revert project folder name
                        os.rename(updatedProjectPath, projectPath)
                        print("==================")
                        msg = QtWidgets.QErrorMessage(self)
                        msg.setModal(True)
                        msg.setWindowTitle("Error Message")
                        msg.showMessage(str(e))
                        msg.exec()
                        return

                    # update project_explorer dictionary
                    del self.obj_appconfig.project_explorer[projectPath]
                    self.obj_appconfig.project_explorer[updatedProjectPath] = \
                        updatedProjectFiles

                    # save project_explorer dictionary on disk
                    with open(self.obj_appconfig.dictPath["path"], 'w') as f:
                        json.dump(self.obj_appconfig.project_explorer, f)

                    # recreate project explorer tree
                    self.treewidget.clear()
                    for parent, children in \
                            self.obj_appconfig.project_explorer.items():
                        if os.path.exists(parent):
                            self.addTreeNode(parent, children)

                elif reply == "CHECKEXIST":
                    print("Project name already exists.")
                    print("==========================")
                    msg = QtWidgets.QErrorMessage(self)
                    msg.setModal(True)
                    msg.setWindowTitle("Error Message")
                    msg.showMessage(
                        'The project "' + newBaseFileName +
                        '" already exist. Please select a different name or' +
                        ' delete existing project'
                    )
                    msg.exec()

                elif reply == "CHECKNAME":
                    print("Name can not contain space between them")
                    print("===========================")
                    msg = QtWidgets.QErrorMessage(self)
                    msg.setModal(True)
                    msg.setWindowTitle("Error Message")
                    msg.showMessage(
                        'The project name should not ' +
                        'contain space between them'
                    )
                    msg.exec()

    def set_time_explorer(self, time_explorer_widget):
        self.time_explorer = time_explorer_widget

    def takeSnapshot(self):
        index = self.treewidget.currentIndex()
        file_path = str(index.sibling(index.row(), 1).data()) 
        file_name = os.path.basename(file_path)

        if not os.path.isfile(file_path):
            QtWidgets.QMessageBox.warning(self, "Snapshot Failed", "Selected item is not a file.")
            return

        project_path = self.obj_appconfig.current_project["ProjectName"]
        project_name = os.path.basename(project_path)

        snapshot_dir = os.path.join(Path.home(), ".esim", "history", project_name)
        os.makedirs(snapshot_dir, exist_ok=True)

        formatted_time = datetime.now().strftime("%I.%M %p %d-%m-%Y")
        snapshot_name = f"{file_name}({formatted_time})"
        snapshot_path = os.path.join(snapshot_dir, snapshot_name)

        shutil.copy2(file_path, snapshot_path)

        if hasattr(self, 'time_explorer'):
            self.time_explorer.add_snapshot(file_name, formatted_time)
        else:
            print(f"Snapshot taken: {snapshot_path}")
