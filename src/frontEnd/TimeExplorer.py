import os
import re
import shutil
import json
from datetime import datetime
from PyQt6 import QtWidgets
from configuration import Dialogs
from configuration.Appconfig import Appconfig


class TimeExplorer(QtWidgets.QWidget):
    """Timeline of per-file project snapshots.

    Snapshots are individual file copies stored under
    ``~/.esim/history/<project_name>/`` as ``<file>(<timestamp>)``. They are
    created from the project tree's *Snapshot* action (one file) or the
    *Quick Backup* button here (every file in the project). The panel mounts
    inside the on-demand *Project Snapshots* dialog (Application.show_snapshots).

    The look is the Aurora design's: a header row (title + icon-only refresh),
    an alternating-row list, and a primary/secondary/danger action bar.
    """

    if os.name == 'nt':
        user_home = os.path.join('library', 'config')
    else:
        user_home = os.path.expanduser('~')

    current_project = {"ProjectName": None}
    current_project_path = {"ProjectPath": None}

    # <file>(I.MM AM/PM dd-mm-yyyy)
    SNAP_RE = re.compile(
        r"(.+)\((\d{1,2}\.\d{2} [APM]{2} \d{2}-\d{2}-\d{4})\)$")

    def __init__(self, parent=None):
        super(TimeExplorer, self).__init__(parent)

        # ---- Header bar (title + icon-only refresh) ----
        self.header_layout = QtWidgets.QHBoxLayout()
        self.header_layout.setContentsMargins(8, 8, 8, 0)
        self.header_layout.setSpacing(6)

        self.title_label = QtWidgets.QLabel('Project Snapshots')
        self.title_label.setProperty('cssClass', 'title')
        self.header_layout.addWidget(self.title_label)
        self.header_layout.addStretch(1)

        self.refresh_btn = QtWidgets.QPushButton()
        try:
            from frontEnd.icon_paths import refresh_icon
            self.refresh_btn.setIcon(refresh_icon())
        except Exception:
            self.refresh_btn.setText('Refresh')
        self.refresh_btn.setProperty('cssClass', 'icon')
        self.refresh_btn.setToolTip('Refresh snapshot list')
        self.refresh_btn.clicked.connect(self._refresh)
        self.header_layout.addWidget(self.refresh_btn)

        # ---- Snapshot list ----
        self.treewidget = QtWidgets.QTreeWidget()
        self.treewidget.setHeaderLabels(['Snapshot', 'Date Created'])
        self.treewidget.setColumnWidth(0, 200)
        self.treewidget.setAlternatingRowColors(True)
        self.treewidget.setRootIsDecorated(False)
        self.treewidget.setUniformRowHeights(True)
        self.treewidget.itemDoubleClicked.connect(self.restore_snapshots)

        # ---- Action bar ----
        self.button_layout = QtWidgets.QHBoxLayout()
        self.button_layout.setContentsMargins(8, 8, 8, 8)
        self.button_layout.setSpacing(6)

        self.backup_btn = QtWidgets.QPushButton('Quick Backup')
        # Aurora filled-primary class (green CTA). The byte-identical design QSS
        # ships no generic `.primary`; `verifierPrimary` is the only filled
        # primary style, and pairs with restore=secondary / clear=danger.
        self.backup_btn.setProperty('cssClass', 'verifierPrimary')
        self.backup_btn.setToolTip('Snapshot every file in the current project')
        self.backup_btn.clicked.connect(self.quick_backup)

        # Kept under the historical attribute names so external references
        # (if any) keep working.
        self.restore = QtWidgets.QPushButton('Restore')
        self.restore.setProperty('cssClass', 'secondary')
        self.restore.setToolTip(
            'Restore the selected snapshot into the project')
        self.restore.clicked.connect(self.restore_snapshots)

        self.clear = QtWidgets.QPushButton('Delete')
        self.clear.setProperty('cssClass', 'danger')
        self.clear.setToolTip(
            'Delete the selected snapshot (or all if none selected)')
        self.clear.clicked.connect(self.clear_snapshots)

        self.button_layout.addWidget(self.backup_btn)
        self.button_layout.addStretch(1)
        self.button_layout.addWidget(self.restore)
        self.button_layout.addWidget(self.clear)

        # ---- Compose ----
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(8, 8, 8, 8)
        self.layout.setSpacing(8)
        self.layout.addLayout(self.header_layout)
        self.layout.addWidget(self.treewidget)
        self.layout.addLayout(self.button_layout)

    # ------------------------------------------------------------ helpers
    def _snapshot_dir(self, project_name=None):
        name = project_name or self.current_project.get("ProjectName")
        if not name:
            return None
        return os.path.join(self.user_home, ".esim", "history", name)

    @staticmethod
    def _live_project_dir():
        """The active project's folder, from the single source of truth."""
        try:
            return Appconfig().current_project.get("ProjectName")
        except Exception:
            return None

    def _destination_dir(self):
        return self.current_project_path.get("ProjectPath") \
            or self._live_project_dir()

    def _refresh(self):
        name = self.current_project.get("ProjectName")
        if name:
            self.load_snapshots(name)

    # ------------------------------------------------------------ public api
    def add_snapshot(self, file_name, timestamp):
        item = QtWidgets.QTreeWidgetItem([file_name, timestamp])
        self.treewidget.addTopLevelItem(item)

    def load_snapshots(self, project_name):
        self.treewidget.clear()
        self.current_project["ProjectName"] = project_name
        # Keep the restore destination aligned with the live project folder.
        live = self._live_project_dir()
        if live:
            self.current_project_path["ProjectPath"] = live
        snapshot_dir = self._snapshot_dir(project_name)
        if not snapshot_dir or not os.path.exists(snapshot_dir):
            return
        for filename in sorted(os.listdir(snapshot_dir), reverse=True):
            match = self.SNAP_RE.match(filename)
            if match:
                self.add_snapshot(match.group(1), match.group(2))
            else:
                print(f"Skipping unmatched snapshot file: {filename}")

    def load_last_snapshots(self):
        try:
            path = os.path.join(
                self.user_home, ".esim", "last_project.json")
            with open(path, "r") as f:
                data = json.load(f)
            project_path = data.get("ProjectName", None)
            self.current_project_path["ProjectPath"] = project_path
            if project_path and os.path.exists(project_path):
                project_name = os.path.basename(project_path)
                self.current_project["ProjectName"] = project_name
                self.load_snapshots(project_name)
        except Exception as e:
            print(f"Error loading last snapshots: {e}")

    def quick_backup(self):
        """Snapshot every file in the current project into its history dir."""
        project_dir = self._live_project_dir()
        if not project_dir or not os.path.isdir(project_dir):
            Dialogs.warning(
                self, "No Active Project",
                "Please open a project first to back it up.")
            return
        project_name = self.current_project.get("ProjectName") \
            or os.path.basename(os.path.normpath(project_dir))
        snapshot_dir = self._snapshot_dir(project_name)
        os.makedirs(snapshot_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%I.%M %p %d-%m-%Y")
        backed = 0
        for entry in sorted(os.listdir(project_dir)):
            src = os.path.join(project_dir, entry)
            if not os.path.isfile(src):
                continue
            dst = os.path.join(snapshot_dir, f"{entry}({timestamp})")
            try:
                shutil.copy2(src, dst)
                backed += 1
            except Exception as e:
                print(f"Quick Backup: could not copy {entry}: {e}")
        self.load_snapshots(project_name)
        Dialogs.information(
            self, "Quick Backup", f"{backed} file(s) snapshotted.")

    # ------------------------------------------------------------ commands
    def clear_snapshots(self):
        project_name = self.current_project["ProjectName"]
        if not project_name:
            return
        snapshot_dir = self._snapshot_dir(project_name)

        selected = self.treewidget.selectedItems()
        if selected:
            item = selected[0]
            file_name = item.text(0)
            timestamp = item.text(1)
            snapshot_path = os.path.join(
                snapshot_dir, f"{file_name}({timestamp})")

            confirm = Dialogs.question(
                self, "Confirm Deletion",
                "Are you sure you want to delete this snapshot?\n\n"
                f"{file_name}",
                Dialogs.Button.Yes | Dialogs.Button.No
            )
            if confirm == Dialogs.Button.Yes:
                try:
                    os.remove(snapshot_path)
                    self.treewidget.takeTopLevelItem(
                        self.treewidget.indexOfTopLevelItem(item))
                except Exception as e:
                    Dialogs.warning(
                        self, "Error", f"Could not delete snapshot:\n{e}")
        else:
            if self.treewidget.topLevelItemCount() == 0:
                Dialogs.information(
                    self, "Nothing to delete",
                    "There are no snapshots for this project.")
                return
            confirm = Dialogs.question(
                self, "Clear All Snapshots",
                "No file selected.\nDo you want to delete ALL snapshots for "
                f"'{project_name}'?",
                Dialogs.Button.Yes | Dialogs.Button.No
            )
            if confirm == Dialogs.Button.Yes:
                deleted = 0
                for filename in os.listdir(snapshot_dir):
                    path = os.path.join(snapshot_dir, filename)
                    try:
                        os.remove(path)
                        deleted += 1
                    except Exception as e:
                        print(f"Error deleting {filename}: {e}")
                self.treewidget.clear()
                Dialogs.information(
                    self, "Deleted", f"{deleted} snapshots deleted.")

    def restore_snapshots(self):
        project_name = self.current_project["ProjectName"]
        if not project_name:
            return
        snapshot_dir = self._snapshot_dir(project_name)
        if not snapshot_dir or not os.path.exists(snapshot_dir):
            Dialogs.warning(
                self, "No Snapshots", "No snapshots found for this project.")
            return

        dest_dir = self._destination_dir()
        if not dest_dir:
            Dialogs.warning(
                self, "No Active Project",
                "No project folder to restore into.")
            return

        selected_items = self.treewidget.selectedItems()
        if selected_items:
            item = selected_items[0]
            file_name = item.text(0)
            timestamp = item.text(1)
            snapshot_path = os.path.join(
                snapshot_dir, f"{file_name}({timestamp})")
            destination_path = os.path.join(dest_dir, file_name)

            confirm = Dialogs.question(
                self, "Confirm Restore",
                "Do you want to restore this snapshot?\n\n"
                f"{file_name}",
                Dialogs.Button.Yes | Dialogs.Button.No
            )
            if confirm == Dialogs.Button.Yes:
                try:
                    if os.path.exists(destination_path):
                        os.remove(destination_path)
                    shutil.copy2(snapshot_path, destination_path)
                    if os.path.exists(snapshot_path):
                        os.remove(snapshot_path)
                    self.treewidget.takeTopLevelItem(
                        self.treewidget.indexOfTopLevelItem(item))
                    Dialogs.information(
                        self, "Restored",
                        f"{file_name} has been restored.")
                except Exception as e:
                    Dialogs.warning(
                        self, "Error", f"Could not restore:\n{e}")
        else:
            confirm = Dialogs.question(
                self, "Restore All Snapshots",
                "No file selected.\nDo you want to restore ALL snapshot files?",
                Dialogs.Button.Yes | Dialogs.Button.No
            )
            if confirm == Dialogs.Button.Yes:
                restored = 0
                for filename in os.listdir(snapshot_dir):
                    match = self.SNAP_RE.match(filename)
                    if match:
                        file_base = match.group(1)
                        snapshot_path = os.path.join(snapshot_dir, filename)
                        destination_path = os.path.join(dest_dir, file_base)
                        try:
                            if os.path.exists(destination_path):
                                os.remove(destination_path)
                            shutil.copy2(snapshot_path, destination_path)
                            if os.path.exists(snapshot_path):
                                os.remove(snapshot_path)
                            restored += 1
                        except Exception as e:
                            print(f"Could not restore {file_base}: {e}")
                self.treewidget.clear()
                Dialogs.information(
                    self, "Restored", f"{restored} snapshot(s) restored.")
