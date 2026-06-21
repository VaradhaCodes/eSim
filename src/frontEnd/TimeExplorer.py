import os
import re
import shutil
import json
import datetime
from PyQt6 import QtWidgets, QtCore


class TimeExplorer(QtWidgets.QDialog):
    """Time Explorer — visual timeline of full project backups.

    Backups are stored under ~/.esim/backups/<project_name>/ as .zip archives.

    The dialog exposes three actions:
      - Refresh    — re-read the backup folder from disk
      - Restore    — extract the selected backup zip over the current project
      - Delete     — remove the selected (or all) backups
    """

    if os.name == 'nt':
        user_home = os.path.join('library', 'config')
    else:
        user_home = os.path.expanduser('~')

    current_project = {"ProjectName": None}
    current_project_path = {"ProjectPath": None}

    def __init__(self, parent=None):
        super(TimeExplorer, self).__init__(parent)
        self.setWindowTitle('Timeline & Backups')
        self.resize(500, 350)
        self.setModal(False) # Allows the user to keep it open while working

        # ---- Header bar (title + action icons) ----
        self.header_layout = QtWidgets.QHBoxLayout()
        self.header_layout.setContentsMargins(8, 8, 8, 0)
        self.header_layout.setSpacing(6)

        self.title_label = QtWidgets.QLabel('Project Backups')
        self.title_label.setProperty('cssClass', 'title')
        self.header_layout.addWidget(self.title_label)

        self.header_layout.addStretch(1)

        # Icon-only refresh button
        from frontEnd.icon_paths import refresh_icon
        self.refresh_btn = QtWidgets.QPushButton()
        self.refresh_btn.setIcon(refresh_icon())
        self.refresh_btn.setProperty('cssClass', 'icon')
        self.refresh_btn.setToolTip('Refresh backup list')
        self.refresh_btn.clicked.connect(self._refresh)
        self.header_layout.addWidget(self.refresh_btn)

        # ---- Snapshot list ----
        self.treewidget = QtWidgets.QTreeWidget()
        self.treewidget.setHeaderLabels(['Backup Archive', 'Date Created'])
        self.treewidget.setColumnWidth(0, 250)
        self.treewidget.setAlternatingRowColors(True)
        self.treewidget.setRootIsDecorated(False)
        self.treewidget.setUniformRowHeights(True)

        # ---- Action bar ----
        self.actions_layout = QtWidgets.QHBoxLayout()
        self.actions_layout.setContentsMargins(8, 8, 8, 8)
        self.actions_layout.setSpacing(6)

        self.backup_btn = QtWidgets.QPushButton('Quick Backup')
        self.backup_btn.setProperty('cssClass', 'primary')
        self.backup_btn.setToolTip('Save a full snapshot of the project')
        self.backup_btn.clicked.connect(self.quick_backup)

        self.restore_btn = QtWidgets.QPushButton('Restore')
        self.restore_btn.setProperty('cssClass', 'secondary')
        self.restore_btn.setToolTip('Extract the selected backup over the current project')
        self.restore_btn.clicked.connect(self.restore_snapshots)

        self.delete_btn = QtWidgets.QPushButton('Delete')
        self.delete_btn.setProperty('cssClass', 'danger')
        self.delete_btn.setToolTip('Delete the selected backup (or all if none selected)')
        self.delete_btn.clicked.connect(self.clear_snapshots)

        self.actions_layout.addWidget(self.backup_btn)
        self.actions_layout.addStretch(1)
        self.actions_layout.addWidget(self.restore_btn)
        self.actions_layout.addWidget(self.delete_btn)

        # ---- Compose final layout ----
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        main_layout.addLayout(self.header_layout)
        main_layout.addWidget(self.treewidget)
        main_layout.addLayout(self.actions_layout)

        # Selection wiring
        self.treewidget.itemDoubleClicked.connect(self.restore_snapshots)

    # ------------------------------------------------------------------ helpers
    def _snapshot_dir(self):
        project_name = self.current_project['ProjectName']
        if not project_name:
            return None
        return os.path.join(self.user_home, '.esim', 'backups', project_name)

    def _project_path(self):
        return self.current_project.get('ProjectName')

    # ------------------------------------------------------------------ public api
    def add_snapshot(self, file_name, timestamp):
        item = QtWidgets.QTreeWidgetItem([file_name, timestamp])
        self.treewidget.addTopLevelItem(item)

    def load_snapshots(self, project_name):
        self.treewidget.clear()
        self.current_project['ProjectName'] = project_name
        snapshot_dir = self._snapshot_dir()
        if not snapshot_dir or not os.path.exists(snapshot_dir):
            return
        
        # Load all zip files in the backup directory
        for filename in sorted(os.listdir(snapshot_dir), reverse=True):
            if filename.endswith(".zip"):
                path = os.path.join(snapshot_dir, filename)
                mtime = os.path.getmtime(path)
                timestamp = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                self.add_snapshot(filename, timestamp)

    def load_last_snapshots(self):
        try:
            path = os.path.join(self.user_home, '.esim', 'last_project.json')
            with open(path, 'r') as f:
                data = json.load(f)
            project_path = data.get('ProjectName')
            self.current_project_path['ProjectPath'] = project_path
            if project_path and os.path.exists(project_path):
                project_name = os.path.basename(project_path)
                self.current_project['ProjectName'] = project_name
                self.load_snapshots(project_name)
        except Exception as e:
            print(f'[TimeExplorer] Could not load last snapshots: {e}')

    def _refresh(self):
        name = self.current_project.get('ProjectName')
        if name:
            self.load_snapshots(name)

    def quick_backup(self):
        project_dir = self._project_path()
        if not project_dir:
            QtWidgets.QMessageBox.warning(self, "No Active Project", "Please open a project first to back it up.")
            return

        project_name = os.path.basename(project_dir)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self._snapshot_dir()
        if backup_dir:
            os.makedirs(backup_dir, exist_ok=True)
            default_path = os.path.join(backup_dir, f"{project_name}_{timestamp}.zip")
        else:
            default_path = f"{project_name}_{timestamp}.zip"
            
        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Project Backup",
            default_path,
            "Zip Archives (*.zip);;All Files (*)"
        )
        if not save_path:
            return
            
        # shutil.make_archive expects the base name without .zip
        if save_path.endswith('.zip'):
            save_path_base = save_path[:-4]
        else:
            save_path_base = save_path
            
        try:
            # Create a zip archive of the project directory
            archive_path = shutil.make_archive(save_path_base, 'zip', project_dir)
            QtWidgets.QMessageBox.information(
                self, "Backup Successful", 
                f"Project backed up to:\n{archive_path}"
            )
            self._refresh()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Backup Failed", f"Could not create backup:\n{e}")

    # ------------------------------------------------------------------ commands
    def clear_snapshots(self):
        project_name = self.current_project['ProjectName']
        if not project_name:
            return
        snapshot_dir = self._snapshot_dir()

        selected = self.treewidget.selectedItems()
        if selected:
            item = selected[0]
            file_name = item.text(0)
            timestamp = item.text(1)
            snapshot_path = os.path.join(snapshot_dir, file_name)

            confirm = QtWidgets.QMessageBox.question(
                self, 'Delete Backup',
                f"Delete this backup archive?\n\n{file_name}\nCreated: {timestamp}",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if confirm == QtWidgets.QMessageBox.StandardButton.Yes:
                try:
                    os.remove(snapshot_path)
                    idx = self.treewidget.indexOfTopLevelItem(item)
                    self.treewidget.takeTopLevelItem(idx)
                except OSError as e:
                    QtWidgets.QMessageBox.warning(
                        self, 'Delete failed',
                        f"Could not delete backup:\n{e}"
                    )
        else:
            count = self.treewidget.topLevelItemCount()
            if count == 0:
                QtWidgets.QMessageBox.information(
                    self, 'Nothing to delete',
                    'There are no backups for this project.'
                )
                return
            confirm = QtWidgets.QMessageBox.question(
                self, 'Delete all backups',
                f"Delete all {count} backup(s) for '{project_name}'?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )
            if confirm == QtWidgets.QMessageBox.StandardButton.Yes:
                deleted = 0
                last_error = None
                for filename in os.listdir(snapshot_dir):
                    if filename.endswith(".zip"):
                        path = os.path.join(snapshot_dir, filename)
                        try:
                            os.remove(path)
                            deleted += 1
                        except OSError as e:
                            last_error = e
                self.treewidget.clear()
                if last_error:
                    QtWidgets.QMessageBox.warning(
                        self, 'Some files skipped',
                        f"Deleted {deleted} backup(s). One or more "
                        f"files could not be removed:\n{last_error}"
                    )
                else:
                    QtWidgets.QMessageBox.information(
                        self, 'Backups deleted',
                        f"{deleted} backup(s) deleted."
                    )

    def restore_snapshots(self):
        project_dir = self._project_path()
        if not project_dir:
            QtWidgets.QMessageBox.warning(
                self, 'No active project',
                'Please open a project first to restore a backup into it.'
            )
            return

        snapshot_dir = self._snapshot_dir()
        selected_items = self.treewidget.selectedItems()
        
        default_path = snapshot_dir if snapshot_dir else self.user_home
        if selected_items and snapshot_dir:
            default_path = os.path.join(snapshot_dir, selected_items[0].text(0))

        open_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Backup Archive to Restore",
            default_path,
            "Zip Archives (*.zip);;All Files (*)"
        )
        if not open_path:
            return

        file_name = os.path.basename(open_path)
        confirm = QtWidgets.QMessageBox.warning(
            self, 'Restore Project Backup',
            f"Restore '{file_name}' over the current project?\n\n"
            f"WARNING: This will overwrite files in the current project directory with the versions from the backup archive.",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if confirm == QtWidgets.QMessageBox.StandardButton.Yes:
            try:
                # Unpack the zip archive directly into the project directory
                shutil.unpack_archive(open_path, project_dir, 'zip')
                QtWidgets.QMessageBox.information(
                    self, 'Restore complete',
                    'Backup successfully restored.'
                )
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self, 'Restore failed',
                    f"Could not restore backup:\n{e}"
                )
