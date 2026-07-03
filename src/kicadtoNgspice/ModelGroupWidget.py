# ==============================================================================
#  ModelGroupWidget.py -- a collapsible "assign once, override per instance"
#  group box for the Device Modeling and Subcircuits tabs.
#
#  One ModelGroupWidget represents all instances that share a model (e.g. the
#  five eSim_NPN transistors). It shows:
#
#      v eSim_NPN (transistor)  q1, q2, q5    [ /lib/bc547.lib   ] [Browse]
#          q1   (follows group)
#          q2   (follows group)
#          q5   [ /lib/bc547_alt.lib ] [Browse]  * overridden     [Reset]
#
#  The group path field fans its value out to every instance that is still
#  "inheriting"; editing one instance's field (or its Browse) detaches just that
#  instance ("override") so later group changes leave it alone, and Reset
#  re-attaches it.
#
#  Design / why it is shaped this way:
#    * The per-instance QLineEdit handed in (InstanceRow.path_edit) is the SAME
#      widget the converter already registers in TrackWidget (entry_var) and
#      that three consumers read per instance (Convert -> netlist, callConvert ->
#      Previous_Values.xml, tab reload). This widget never replaces that storage;
#      it only arranges those edits and drives them. So grouping stays a pure
#      view/controller layer and the downstream netlist is byte-for-byte
#      unchanged.
#    * Override is detected with QLineEdit.textEdited (user keystrokes only),
#      never textChanged, so the programmatic setText() used for fan-out does NOT
#      look like a manual override.
#    * resolve_fn(ref, path) is the single hook back to the tab; the tab uses it
#      to validate and update its deviceModelTrack / subcircuitTrack entry, i.e.
#      exactly what trackLibraryWithoutButton already does.
# ==============================================================================
from PyQt6 import QtCore, QtWidgets


class InstanceRow:
    """One instance inside a model group.

    ref       : reference designator (q1, x1, ...)
    path_edit : the canonical per-instance QLineEdit (a TrackWidget entry_var).
    browse_fn : optional zero-arg callable that opens a file dialog and returns
                the chosen path (or '' / None if cancelled).
    extras    : optional list of (label_text, QWidget) shown after the path on
                this instance's row -- used for MOSFET W/L/M, which are always
                per-instance and never inherited.
    """

    def __init__(self, ref, path_edit, browse_fn=None, extras=None):
        self.ref = ref
        self.path_edit = path_edit
        self.browse_fn = browse_fn
        self.extras = extras or []
        self.overridden = False
        # Populated by the widget when it builds this row:
        self._marker = None
        self._reset_btn = None


class ModelGroupWidget(QtWidgets.QGroupBox):
    """Collapsible group of instances that share one model/library."""

    def __init__(self, title, instance_rows, resolve_fn=None,
                 group_browse_fn=None, parent=None):
        super().__init__(parent)
        self._rows = list(instance_rows)
        self._resolve_fn = resolve_fn or (lambda ref, path: None)
        self._group_browse_fn = group_browse_fn

        refs = ", ".join(r.ref for r in self._rows)
        self._title_text = title
        self.setTitle("")           # custom header row instead of the box title

        outer = QtWidgets.QVBoxLayout()
        outer.setContentsMargins(8, 6, 8, 6)
        self.setLayout(outer)

        # -- header: expand toggle + summary + group path + group Browse -------
        header = QtWidgets.QGridLayout()

        self._toggle = QtWidgets.QToolButton()
        self._toggle.setCheckable(True)
        self._toggle.setChecked(False)
        self._toggle.setAutoRaise(True)
        self._toggle.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(_arrow(False))
        self._toggle.setText(" %s  [%s]" % (title, refs))
        self._toggle.clicked.connect(self._on_toggle)
        header.addWidget(self._toggle, 0, 0)

        self._group_edit = QtWidgets.QLineEdit()
        self._group_edit.setPlaceholderText(
            "Assign one file for all %d instances" % len(self._rows))
        self._group_edit.editingFinished.connect(
            lambda: self.set_group_path(self._group_edit.text()))
        header.addWidget(self._group_edit, 0, 1)

        group_browse = QtWidgets.QPushButton("Browse")
        group_browse.clicked.connect(self._on_group_browse)
        header.addWidget(group_browse, 0, 2)
        header.setColumnStretch(1, 1)
        outer.addLayout(header)

        # -- collapsible body: one row per instance ----------------------------
        self._body = QtWidgets.QWidget()
        body_grid = QtWidgets.QGridLayout()
        body_grid.setContentsMargins(24, 2, 2, 2)
        self._body.setLayout(body_grid)

        for i, row in enumerate(self._rows):
            body_grid.addWidget(QtWidgets.QLabel(row.ref), i, 0)
            body_grid.addWidget(row.path_edit, i, 1)

            row_browse = QtWidgets.QPushButton("Browse")
            row_browse.clicked.connect(self._make_row_browse(row))
            body_grid.addWidget(row_browse, i, 2)

            row._marker = QtWidgets.QLabel("")
            body_grid.addWidget(row._marker, i, 3)

            row._reset_btn = QtWidgets.QPushButton("Reset")
            row._reset_btn.setVisible(False)
            row._reset_btn.clicked.connect(self._make_row_reset(row))
            body_grid.addWidget(row._reset_btn, i, 4)

            # MOSFET-style per-instance extras (W/L/M); never inherited.
            col = 5
            for label_text, widget in row.extras:
                body_grid.addWidget(QtWidgets.QLabel(label_text), i, col)
                body_grid.addWidget(widget, i, col + 1)
                col += 2

            # User keystrokes (not programmatic setText) => override this row.
            row.path_edit.textEdited.connect(self._make_row_edited(row))

        body_grid.setColumnStretch(1, 1)
        outer.addWidget(self._body)
        self._body.setVisible(False)

        self._derive_initial()

    # -- public API (also the unit-test surface) -------------------------------

    def set_group_path(self, path):
        """Set the group default and push it to every inheriting instance."""
        if self._group_edit.text() != path:
            self._group_edit.setText(path)
        for row in self._rows:
            if not row.overridden:
                self._apply(row, path)

    def override(self, ref, path):
        """Detach one instance and give it its own path."""
        row = self._row(ref)
        row.path_edit.setText(path)
        self._set_overridden(row, True)
        self._resolve_fn(row.ref, path)

    def reset_row_by_ref(self, ref):
        """Re-attach one instance to the group default."""
        self._reset(self._row(ref))

    def resolved(self):
        """Return {ref: current path text} -- what each instance would convert
        with right now. Mirrors the per-instance entries the tab tracks."""
        return {r.ref: r.path_edit.text() for r in self._rows}

    def group_path(self):
        return self._group_edit.text()

    def is_overridden(self, ref):
        return self._row(ref).overridden

    def is_expanded(self):
        # Tracked via the toggle's checked state, not body.isVisible(): a widget
        # only reports visible once it and all ancestors are shown, which is
        # false in headless tests and before the tab is displayed.
        return self._toggle.isChecked()

    def set_expanded(self, expanded):
        self._toggle.setChecked(expanded)
        self._on_toggle()

    # -- internals -------------------------------------------------------------

    def _row(self, ref):
        for r in self._rows:
            if r.ref == ref:
                return r
        raise KeyError(ref)

    def _apply(self, row, path):
        if row.path_edit.text() != path:
            row.path_edit.setText(path)
        self._resolve_fn(row.ref, path)

    def _reset(self, row):
        self._set_overridden(row, False)
        path = self._group_edit.text()
        if row.path_edit.text() != path:
            row.path_edit.setText(path)
        self._resolve_fn(row.ref, path)

    def _set_overridden(self, row, value):
        row.overridden = value
        if row._marker is not None:
            row._marker.setText("* overridden" if value else "")
        if row._reset_btn is not None:
            row._reset_btn.setVisible(value)

    def _derive_initial(self):
        """Set the starting group/override state from whatever the instance
        edits already hold (e.g. values restored from Previous_Values.xml).

        Rule: if every instance carries the same non-empty value, that becomes
        the group default and all instances inherit. Otherwise the group is left
        blank and any instance that has a value is shown as an override (and the
        group is expanded so the user sees them). This never calls resolve_fn --
        restored values are already tracked.
        """
        vals = [r.path_edit.text() for r in self._rows]
        nonempty = [v for v in vals if v]
        uniform = bool(nonempty) and all(v == nonempty[0] for v in vals)
        if uniform:
            self._group_edit.setText(nonempty[0])
            for row in self._rows:
                self._set_overridden(row, False)
        else:
            self._group_edit.setText("")
            any_override = False
            for row in self._rows:
                has_val = bool(row.path_edit.text())
                self._set_overridden(row, has_val)
                any_override = any_override or has_val
            if any_override:
                self.set_expanded(True)

    # -- Qt signal adapters ----------------------------------------------------

    def _on_toggle(self):
        expanded = self._toggle.isChecked()
        self._toggle.setArrowType(_arrow(expanded))
        self._animate_body(expanded)

    def _animate_body(self, expanded):
        """Slide the instance list open/closed instead of snapping it.

        Falls back to a plain show/hide when motion is off or the widget is not
        on screen (headless tests, pre-display construction) -- the animation's
        ``finished`` callback never fires there, so the body must reach its final
        visibility synchronously."""
        body = self._body
        if not _motion_enabled() or not self.isVisible():
            body.setMaximumHeight(_UNCAPPED)
            body.setVisible(expanded)
            return

        target = body.sizeHint().height()

        # Scale the duration to the distance travelled. A fixed time over a small
        # one-row span leaves the decelerating tail of OutCubic moving <1px per
        # frame; the integer maximumHeight then holds for several frames and jumps
        # 1px at a time -> visible stutter. Tying duration to height keeps every
        # frame moving >=1px so a 1-row slide is as smooth as a 5-row one. A tall
        # body (~150px) still lands at ~180ms, so multi-row feel is unchanged.
        distance = target if expanded else body.height()
        anim = QtCore.QPropertyAnimation(body, b"maximumHeight", self)
        anim.setDuration(max(100, min(200, int(distance * 1.2))))
        anim.setEasingCurve(QtCore.QEasingCurve.Type.OutCubic)
        if expanded:
            body.setMaximumHeight(0)
            body.setVisible(True)
            anim.setStartValue(0)
            anim.setEndValue(target)
            # Drop the height cap once open so the row grid can grow naturally.
            anim.finished.connect(lambda: body.setMaximumHeight(_UNCAPPED))
        else:
            anim.setStartValue(body.height())
            anim.setEndValue(0)

            def _done():
                body.setVisible(False)
                body.setMaximumHeight(_UNCAPPED)
            anim.finished.connect(_done)
        self._body_anim = anim          # keep a ref so it is not GC'd mid-run
        anim.start()

    def _on_group_browse(self):
        if not self._group_browse_fn:
            return
        path = self._group_browse_fn()
        if path:
            self.set_group_path(path)

    def _make_row_browse(self, row):
        def handler():
            if not row.browse_fn:
                return
            path = row.browse_fn()
            if path:
                self.override(row.ref, path)
        return handler

    def _make_row_reset(self, row):
        return lambda: self._reset(row)

    def _make_row_edited(self, row):
        def handler(text):
            self._set_overridden(row, True)
            self._resolve_fn(row.ref, text)
        return handler


def _arrow(expanded):
    return (QtCore.Qt.ArrowType.DownArrow if expanded
            else QtCore.Qt.ArrowType.RightArrow)


# QWIDGETSIZE_MAX -- the "no cap" value to restore on QWidget.maximumHeight once
# the open animation finishes.
_UNCAPPED = 16777215


def _motion_enabled():
    """True only when the app's motion preference is on. Defaults to False (no
    animation, plain show/hide) if frontEnd isn't importable -- keeps headless
    tests on the synchronous path."""
    try:
        from frontEnd.motion import motion_enabled
        return motion_enabled()
    except Exception:
        return False
