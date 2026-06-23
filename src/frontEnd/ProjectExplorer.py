from PyQt6 import QtCore, QtGui, QtWidgets
from configuration import Dialogs
import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from configuration.Appconfig import Appconfig
from projManagement.Validation import Validation
from projManagement.projectPaths import resolve_stem, canonical_path, \
    same_project
from codeEditor import EditorWindow


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

    # Data roles stored on each top-level (project) item so identity and
    # display stay decoupled: STEM_ROLE keeps the un-disambiguated base label
    # (the project stem) so collisions can be re-resolved on every change;
    # STALE_ROLE flags a project whose folder no longer exists on disk.
    STEM_ROLE = QtCore.Qt.ItemDataRole.UserRole
    STALE_ROLE = QtCore.Qt.ItemDataRole.UserRole + 1

    def __init__(self):
        """
        This method is doing following tasks:
            - Working as a constructor for class ProjectExplorer.
            - view of project explorer area.
        """
        QtWidgets.QWidget.__init__(self)
        self.obj_appconfig = Appconfig()
        self.obj_validation = Validation()
        # One reusable editor window per project (keyed by project name).
        self.editor_windows = {}
        self.treewidget = QtWidgets.QTreeWidget()
        self.window = QtWidgets.QVBoxLayout()
        self.fs_watcher = QtCore.QFileSystemWatcher()
        header = QtWidgets.QTreeWidgetItem(["Projects", "path"])
        self.treewidget.setHeaderItem(header)
        self.treewidget.setColumnHidden(1, True)


        self.loadProjects()
        self.window.addWidget(self.treewidget)
        self.fs_watcher.directoryChanged.connect(self.handleDirectoryChanged)
        self.treewidget.expanded.connect(self.refreshInstant)
        self.treewidget.doubleClicked.connect(self.openProject)
        self.treewidget.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.treewidget.customContextMenuRequested.connect(self.openMenu)
        self.setLayout(self.window)
        self.show()

    def loadProjects(self):
        """
        Render the saved project list into the tree, collapsed.

        Single entry point for *bulk* loading (first construction, workspace
        open/switch). Idempotent: clears the tree, migrates persisted entries
        to canonical identity keys -- collapsing any that point at the same
        folder (legacy raw strings, symlink/'..'/case variants) so a project
        appears exactly once -- builds each node collapsed, then refreshes
        labels and persists once. Missing folders are kept but shown as stale
        rather than silently dropped, so the user can see and remove them.

        Distinct from addTreeNode, which is for the user *opening one*
        project: that focuses and expands the single node it touches. Bulk
        loading must never expand every project, so it does not go through
        addTreeNode -- the active project (e.g. a restored last project) is
        re-focused at the end instead.
        """
        # Rebuild project_explorer in place: it is a shared class-level dict
        # that other Appconfig instances (openProject, newProject, Workspace)
        # read and mutate. Replacing it with a new object would shadow it on
        # this instance only and desync everyone else, so clear()+update()
        # keeps the one shared dict's identity.
        migrated = {}
        for parents, children in list(
                self.obj_appconfig.project_explorer.items()):
            key = canonical_path(parents)
            if key:
                migrated[key] = children
        self.obj_appconfig.project_explorer.clear()
        self.obj_appconfig.project_explorer.update(migrated)

        self.treewidget.clear()
        for parents, children in migrated.items():
            self._buildNode(parents, children)
        self._refreshLabels()
        self._persist()
        self._focusCurrentProject()

    def _focusCurrentProject(self):
        """
        After a bulk load, expand + select the active project so a restored
        'last project' stays focused. No-op when nothing is open or the
        active project is not in the current list (e.g. a different
        workspace was opened).
        """
        proj = self.obj_appconfig.current_project.get('ProjectName')
        if not proj:
            return
        node = self._findNode(proj)
        if node is not None:
            self.treewidget.setCurrentItem(node)
            node.setExpanded(True)

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
        """
        Register a project in the explorer tree, keyed by its canonical path.

        Idempotent: opening a project already present does NOT create a second
        node -- it refreshes that project's file list and selects it. This is
        what stops the same project being added over and over (open it 100
        times, get one node), and what de-duplicates the same folder reached
        via a symlink, a '..' path, a trailing slash or a different-case
        spelling, since all collapse to the same canonical key.
        """
        key = canonical_path(parents)
        if not key:
            return

        existing = self._findNode(key)
        if existing is not None:
            # Already open: refresh children to the current on-disk list and
            # focus it instead of duplicating the node.
            self._fillChildren(existing, key, children)
        else:
            existing = self._buildNode(key, children)
        self.treewidget.setCurrentItem(existing)
        existing.setExpanded(True)

        self.obj_appconfig.project_explorer[key] = children
        self._persist()
        self._refreshLabels()

        # setdefault, not assignment: addTreeNode runs again on every refresh,
        # and clobbering these would drop the PIDs/docks already tracked for an
        # open project (orphaning its KiCad/ngspice windows on close).
        projName = self.obj_appconfig.current_project['ProjectName']
        self.obj_appconfig.proc_dict.setdefault(projName, [])
        self.obj_appconfig.dock_dict.setdefault(projName, [])

    # ---- project-node helpers (identity, display, staleness) ---------------

    def _projectLabel(self, path):
        """Base display label for a project: its stem, else the folder name."""
        stem, _status = resolve_stem(path, 'proj')
        return stem or os.path.basename(os.path.normpath(path)) or path

    def _buildNode(self, path, children):
        """
        Create a top-level project node for ``path`` (assumed canonical).
        Fills children from the on-disk file list; if the folder is missing,
        the node is created in the 'stale' state instead.
        """
        base = self._projectLabel(path)
        node = QtWidgets.QTreeWidgetItem(self.treewidget, [base, path])
        node.setData(0, self.STEM_ROLE, base)
        self._fillChildren(node, path, children)
        return node

    def _fillChildren(self, node, path, children):
        """Repopulate a project node's file rows, or mark it stale."""
        node.takeChildren()
        if os.path.exists(path):
            node.setData(0, self.STALE_ROLE, False)
            self._clearStale(node, path)
            for files in children:
                QtWidgets.QTreeWidgetItem(
                    node, [files, os.path.join(path, files)]
                )
            if path not in self.fs_watcher.directories():
                self.fs_watcher.addPath(path)
        else:
            self._markStale(node, path)

    def _markStale(self, node, path):
        """Style a project whose folder no longer exists on disk."""
        node.setData(0, self.STALE_ROLE, True)
        node.setForeground(0, QtGui.QBrush(QtGui.QColor('gray')))
        font = node.font(0)
        font.setItalic(True)
        node.setFont(0, font)
        node.setToolTip(
            0, path + '  —  missing on disk '
            '(right-click ▸ Remove Project)')
        if path in self.fs_watcher.directories():
            self.fs_watcher.removePath(path)

    def _clearStale(self, node, path):
        """Undo stale styling once a project's folder is back/refreshed."""
        node.setForeground(0, QtGui.QBrush())
        font = node.font(0)
        font.setItalic(False)
        node.setFont(0, font)
        node.setToolTip(0, path)

    def _findNode(self, path):
        """Top-level item whose folder is the same project as ``path``."""
        key = canonical_path(path)
        for i in range(self.treewidget.topLevelItemCount()):
            item = self.treewidget.topLevelItem(i)
            if canonical_path(item.text(1)) == key:
                return item
        return None

    def _locHint(self, path):
        """Short parent-folder hint to disambiguate same-named projects."""
        parent = os.path.dirname(path)
        home = os.path.expanduser('~')
        if parent == home or parent.startswith(home + os.sep):
            parent = '~' + parent[len(home):]
        return parent or os.sep

    def _refreshLabels(self):
        """
        Resolve display labels for all projects. Projects with a unique stem
        show just the stem; projects whose stems collide are disambiguated by
        their parent folder so they are never indistinguishable. Stale projects
        are suffixed '(missing)'. Full path is always in the tooltip.
        """
        groups = {}
        for i in range(self.treewidget.topLevelItemCount()):
            item = self.treewidget.topLevelItem(i)
            base = item.data(0, self.STEM_ROLE) or item.text(0)
            groups.setdefault(base, []).append(item)

        for base, items in groups.items():
            collide = len(items) > 1
            for item in items:
                label = base
                if collide:
                    label += '  (' + self._locHint(item.text(1)) + ')'
                if item.data(0, self.STALE_ROLE):
                    label += '  (missing)'
                item.setText(0, label)

    def _persist(self):
        """Write the project list to disk; tolerate an unwritable workspace."""
        try:
            with open(self.obj_appconfig.dictPath["path"], 'w') as fh:
                json.dump(self.obj_appconfig.project_explorer, fh)
        except OSError as err:
            print("Could not save project list:", err)

    def openMenu(self, position):
        indexes = self.treewidget.selectedIndexes()
        if not indexes:
            return

        level = 0
        index = indexes[0]
        while index.parent().isValid():
            index = index.parent()
            level += 1

        menu = QtWidgets.QMenu()
        if level == 0:
            renameProject = menu.addAction(self.tr("Rename Project"))
            renameProject.triggered.connect(self.renameProject)
            deleteproject = menu.addAction(self.tr("Remove Project"))
            deleteproject.triggered.connect(self.removeProject)
            refreshproject = menu.addAction(self.tr("Refresh"))
            refreshproject.triggered.connect(self.refreshProject)
        elif level == 1:
            openfile = menu.addAction(self.tr("Open"))
            openfile.triggered.connect(self.openProject)
            snapshot = menu.addAction(self.tr("Snapshot"))
            snapshot.triggered.connect(self.takeSnapshot)

        menu.exec(self.treewidget.viewport().mapToGlobal(position))

    def openProject(self):
        self.indexItem = self.treewidget.currentIndex()
        self.filePath = str(
            self.indexItem.sibling(self.indexItem.row(), 1).data()
        )

        if (os.path.isfile(str(self.filePath))):
            self.openInEditor(str(self.filePath))
        else:
            self.refreshProject(self.filePath)

            self.obj_appconfig.print_info(
                'The current project is: ' + self.filePath
            )

            self.obj_appconfig.set_current_project(str(self.filePath))
            (
                self.obj_appconfig.
                proc_dict[self.obj_appconfig.current_project['ProjectName']]
            ) = []
            if (
                self.obj_appconfig.current_project['ProjectName'] not in
                self.obj_appconfig.dock_dict
            ):
                (
                    self.obj_appconfig.
                    dock_dict[
                        self.obj_appconfig.current_project['ProjectName']]
                ) = []

    def openInEditor(self, filePath):
        """Open a project text file in the eSim code editor.

        Reuses one tabbed editor window per project, so a file already
        open just gets focused instead of spawning another window.
        """
        self._editorWindow().open(filePath)

    def _editorWindow(self):
        """Return the editor window for the current project."""
        project = self.obj_appconfig.current_project.get('ProjectName')
        key = project or '__noproject__'
        window = self.editor_windows.get(key)
        if window is None:
            window = EditorWindow.EditorWindow()
            self.editor_windows[key] = window
        # Tie its lifecycle to the project so closeDock() reaps it
        # (re-register in case the project was closed and reopened).
        if project and project in self.obj_appconfig.dock_dict:
            if window not in self.obj_appconfig.dock_dict[project]:
                self.obj_appconfig.dock_dict[project].append(window)
        return window

    def removeProject(self):
        """
        This function removes the project in explorer area by right \
        clicking on project and selecting remove option.
        """
        self.indexItem = self.treewidget.currentIndex()
        filePath = str(
            self.indexItem.sibling(self.indexItem.row(), 1).data()
        )
        self.int = self.indexItem.row()
        self.treewidget.takeTopLevelItem(self.int)

        key = canonical_path(filePath)
        if same_project(
                self.obj_appconfig.current_project["ProjectName"], key):
            self.obj_appconfig.set_current_project(None)

        if key in self.fs_watcher.directories():
            self.fs_watcher.removePath(key)

        # Drop the canonical key and any legacy alias resolving to the same
        # project, so a removed project never lingers under a stale spelling.
        for stored in [
            k for k in self.obj_appconfig.project_explorer
            if canonical_path(k) == key
        ]:
            self.obj_appconfig.project_explorer.pop(stored, None)

        self._persist()
        self._refreshLabels()

    def refreshProject(self, filePath=None, indexItem=None):
        """
        This function refresh the project in explorer area by right \
        clicking on project and selecting refresh option.
        """

        if not filePath or filePath is None:
            if indexItem is None:
                self.indexItem = self.treewidget.currentIndex()
            else:
                self.indexItem = indexItem

            filePath = str(
                self.indexItem.sibling(self.indexItem.row(), 1).data()
            )

        if os.path.exists(filePath):
            filelistnew = os.listdir(os.path.join(filePath))
            if indexItem is None:
                parentnode = self.treewidget.currentItem()
            else:
                parentnode = self.treewidget.itemFromIndex(self.indexItem)
            count = parentnode.childCount()
            for i in range(count):
                parentnode.removeChild(parentnode.child(0))
            for files in filelistnew:
                QtWidgets.QTreeWidgetItem(
                    parentnode, [files, os.path.join(filePath, files)]
                )

            # Key by canonical identity and clear any prior stale state -- a
            # refresh that succeeds means the folder is back/valid.
            key = canonical_path(filePath)
            self.obj_appconfig.project_explorer[key] = filelistnew
            parentnode.setData(0, self.STALE_ROLE, False)
            self._clearStale(parentnode, filePath)
            self._refreshLabels()
            self._persist()
            return True

        else:
            # Folder vanished (moved/deleted/unmounted): keep the node but show
            # it as stale so the user can locate or remove it, rather than
            # silently losing the project.
            node = self._findNode(filePath)
            if node is not None:
                self._markStale(node, canonical_path(filePath))
                self._refreshLabels()
            print("Selected project not found")
            print("==================")
            msg = QtWidgets.QErrorMessage(self)
            msg.setModal(True)
            msg.setWindowTitle("Error Message")
            msg.showMessage('Selected project does not exist.')
            msg.exec()
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
        self.indexItem = self.treewidget.currentIndex()
        self.baseFileName = str(self.indexItem.data())
        filePath = str(
                    self.indexItem.sibling(self.indexItem.row(), 1).data()
                )

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

                    # rename project folder
                    try:
                        os.rename(projectPath, updatedProjectPath)
                    except BaseException as e:
                        msg = QtWidgets.QErrorMessage(self)
                        msg.setModal(True)
                        msg.setWindowTitle("Error Message")
                        msg.showMessage(str(e))
                        msg.exec()
                        return

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

                    except BaseException as e:
                        print("==================")
                        print("Error! Revert renaming project")

                        # Revert updatedProjectFiles
                        for projectFile in updatedProjectFiles:
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

                    # update project_explorer dictionary (canonical key)
                    updatedProjectPath = canonical_path(updatedProjectPath)
                    del self.obj_appconfig.project_explorer[projectPath]
                    self.obj_appconfig.project_explorer[updatedProjectPath] = \
                        updatedProjectFiles

                    # Keep current_project pointing at the renamed folder if it
                    # was the active project, so identity comparisons elsewhere
                    # don't go stale against the old path.
                    if same_project(
                            self.obj_appconfig.current_project["ProjectName"],
                            projectPath):
                        self.obj_appconfig.set_current_project(
                            updatedProjectPath)

                    # remove the old folder from the watcher
                    if projectPath in self.fs_watcher.directories():
                        self.fs_watcher.removePath(projectPath)

                    # save project_explorer dictionary on disk
                    self._persist()

                    # recreate project explorer tree (addTreeNode is idempotent
                    # and renders missing folders as stale, not dropped)
                    self.treewidget.clear()
                    # Snapshot: addTreeNode writes back into project_explorer.
                    for parent, children in list(
                            self.obj_appconfig.project_explorer.items()):
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
                    msg.exec_()

    def set_time_explorer(self, time_explorer_widget):
        self.time_explorer = time_explorer_widget

    def takeSnapshot(self):
        index = self.treewidget.currentIndex()
        file_path = str(index.sibling(index.row(), 1).data()) 
        file_name = os.path.basename(file_path)

        if not os.path.isfile(file_path):
            Dialogs.warning(self, "Snapshot Failed", "Selected item is not a file.")
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
