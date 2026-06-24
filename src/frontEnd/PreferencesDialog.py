import os
import webbrowser
from PyQt6 import QtWidgets, QtGui, QtCore

from configuration.Appconfig import Appconfig


NAMED_SWATCHES = [
    ("Default",   "default"),
    ("Indigo",    "#4F46E5"),
    ("Sky",       "#0EA5E9"),
    ("Teal",      "#0D9488"),
    ("Emerald",   "#059669"),
    ("Amber",     "#D97706"),
    ("Crimson",   "#DC2626"),
    ("Magenta",   "#C026D3"),
    ("Slate",     "#475569"),
    ("Charcoal",  "#1F2937"),
]


def _border_for_theme() -> str:
    pal = QtWidgets.QApplication.palette()
    return pal.color(QtGui.QPalette.ColorRole.Mid).name()


class ThemePreview(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("themePreview")
        self.setMinimumHeight(86)
        self.setFixedHeight(86)
        self.colors = {}

    def setColors(self, colors):
        self.colors = dict(colors)
        self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        try:
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            r = self.rect().adjusted(1, 1, -1, -1)
            bg = QtGui.QColor(self.colors.get("bg", "#FFFFFF"))
            panel = QtGui.QColor(self.colors.get("panel", "#F8FBFF"))
            accent = QtGui.QColor(self.colors.get("accent", "#0077A8"))
            text = QtGui.QColor(self.colors.get("text", "#142033"))
            p.setPen(QtGui.QPen(QtGui.QColor(self.colors.get("border", "#D6E1EE")), 1))
            p.setBrush(bg)
            p.drawRoundedRect(r, 14, 14)
            p.setPen(QtCore.Qt.PenStyle.NoPen)

            # Panel inset
            p.setBrush(panel)
            p.drawRoundedRect(r.adjusted(10, 12, -10, -10), 10, 10)

            # Accent bar with gradient. NB: QLinearGradient needs QPointF —
            # passing the QPoint from QRect.topLeft() raises TypeError mid-paint,
            # which used to leave the QPainter active and corrupt the backing
            # store (blank sibling buttons until hovered). Use a QRectF.
            bar = QtCore.QRectF(r.adjusted(18, 22, -self.width() // 2, -28))
            grad = QtGui.QLinearGradient(bar.topLeft(), bar.topRight())
            grad.setColorAt(0, accent)
            grad.setColorAt(1, accent.lighter(125))
            p.setBrush(grad)
            p.drawRoundedRect(r.adjusted(18, 22, -self.width() // 2, -24), 6, 6)

            # Button preview
            p.setBrush(accent)
            btn_w = int((r.width() - 40) * 0.32)
            p.drawRoundedRect(
                r.left() + 18, r.top() + 52, btn_w, 12, 5, 5
            )

            p.setPen(text)
            f = QtGui.QFont()
            f.setPointSize(9)
            p.setFont(f)
            p.drawText(r.adjusted(20, 42, -20, -14), QtCore.Qt.AlignmentFlag.AlignLeft, "Preview")
        finally:
            p.end()


class PreferencesDialog(QtWidgets.QDialog):
    """User-facing appearance preferences with premium visual polish.

    Tabs:
        * **Appearance** — theme mode, accent swatches, color pickers, live preview.
        * **Editor**     — monospace font family + size for in-app code editors.

    Live updates: Apply or Save persist settings and re-theme the UI.
    """

    def __init__(self, parent=None):
        super().__init__(None)
        self.appconfig = Appconfig()
        self.prefs = self.appconfig.load_preferences()
        # Snapshot for Cancel revert (live-apply changes the running theme).
        self._orig_prefs = dict(self.prefs)

        self.setWindowTitle("Preferences")
        self.setWindowFlags(
            QtCore.Qt.WindowType.Dialog
            | QtCore.Qt.WindowType.WindowTitleHint
            | QtCore.Qt.WindowType.WindowCloseButtonHint
        )
        self.setMinimumWidth(580)

        icon_dir = os.path.dirname(__file__)
        for _rel in ("..", "..", "images", "preferences.png"):
            icon_dir = os.path.dirname(icon_dir)
        icon_dir = os.path.join(icon_dir, "images", "preferences.png")
        if os.path.exists(icon_dir):
            self.setWindowIcon(QtGui.QIcon(icon_dir))

        self.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        self.setObjectName('preferencesDialog')

        self._build_ui()
        self._load_into_widgets()

        from frontEnd.motion import install_button_motion
        install_button_motion(self)

    # ------------------------------------------------------------------ build
    def _build_ui(self):
        self.setMinimumSize(720, 520)
        root = QtWidgets.QVBoxLayout(self)
        root.setSpacing(14)
        root.setContentsMargins(20, 20, 20, 20)

        # Header (gradient title)
        header_layout = QtWidgets.QHBoxLayout()
        header_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        icon_label = QtWidgets.QLabel()
        # Use the same themed SVG gear as the Preferences action in the top
        # toolbar (icon_paths.settings_icon) — not an emoji / loose PNG — so the
        # header icon matches the toolbar and tracks the theme colour.
        try:
            from frontEnd.icon_paths import settings_icon
            icon_label.setPixmap(settings_icon(28).pixmap(28, 28))
        except Exception:
            icon_path = self._find_icon()
            if icon_path and os.path.exists(icon_path):
                pixmap = QtGui.QPixmap(icon_path).scaled(
                    32, 32,
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
                icon_label.setPixmap(pixmap)
        # Formal solid-colour heading (no gradient text); colour from the
        # QLabel[cssClass="title"] QSS rule so it tracks the active theme.
        header = QtWidgets.QLabel("Preferences")
        header.setProperty("cssClass", "title")
        header_layout.addWidget(icon_label)
        header_layout.addWidget(header)
        header_layout.addStretch()
        root.addLayout(header_layout)

        # Body: left nav rail + stacked pages (settings-app pattern)
        body = QtWidgets.QHBoxLayout()
        body.setSpacing(14)
        self.nav = QtWidgets.QListWidget()
        self.nav.setObjectName('prefsNav')
        self.nav.setFixedWidth(168)
        self.nav.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        for label in ("Appearance", "Editor", "About"):
            QtWidgets.QListWidgetItem(label, self.nav)

        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self._build_appearance_tab())
        self.stack.addWidget(self._build_editor_tab())
        self.stack.addWidget(self._build_about_tab())
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        # Re-validate drop-shadow effects when a page is re-shown, otherwise
        # buttons on a previously-hidden page can paint blank (until hovered)
        # after a stack switch while the dialog is maximized/fullscreen.
        self.stack.currentChanged.connect(self._on_pref_page_changed)
        self.nav.setCurrentRow(0)

        body.addWidget(self.nav)
        body.addWidget(self.stack, 1)
        root.addLayout(body, 1)

        # Footer — live-apply means a single primary 'Done'. Reset / Open-json
        # are power-user affordances on the left.
        footer = QtWidgets.QHBoxLayout()
        self.btn_reset = QtWidgets.QPushButton("Reset to Defaults")
        self.btn_reset.setProperty("cssClass", "tertiary")
        self.btn_reset.setToolTip("Restore every preference to its factory default.")
        self.btn_reset.clicked.connect(self._reset_to_defaults_live)

        self.btn_open_json = QtWidgets.QPushButton("Open preferences.json")
        self.btn_open_json.setProperty("cssClass", "tertiary")
        self.btn_open_json.setToolTip(
            "Open the file eSim reads/writes for these preferences — handy for "
            "backups, syncing across machines, or manual edits."
        )
        self.btn_open_json.clicked.connect(self._open_preferences_file)

        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setProperty("cssClass", "secondary")
        self.btn_cancel.setToolTip("Discard changes made in this session.")
        self.btn_cancel.clicked.connect(self.reject)

        self.btn_done = QtWidgets.QPushButton("Done")
        self.btn_done.setDefault(True)
        self.btn_done.setToolTip("Keep changes and close.")
        self.btn_done.clicked.connect(self._save_and_close)

        footer.addWidget(self.btn_reset)
        footer.addWidget(self.btn_open_json)
        footer.addStretch()
        footer.addWidget(self.btn_cancel)
        footer.addWidget(self.btn_done)
        root.addLayout(footer)

    def _on_pref_page_changed(self, idx):
        """Refresh graphics effects on the newly shown page (and once more on
        the next tick, after layout/expose settles) so buttons never render
        blank after a stack switch in maximized/fullscreen state."""
        try:
            from frontEnd.elevation import refresh_effects
            page = self.stack.widget(idx)
            refresh_effects(page)
            QtCore.QTimer.singleShot(
                0, lambda p=page: refresh_effects(p))
        except Exception:
            pass

    def _build_about_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.setContentsMargins(8, 16, 8, 8)
        v.setSpacing(12)
        try:
            from frontEnd.widgets import AuroraHeroFrame
            hero = AuroraHeroFrame()
            hero.setMinimumHeight(120)
            hv = QtWidgets.QVBoxLayout(hero)
            hv.setContentsMargins(22, 18, 22, 18)
            t = QtWidgets.QLabel("eSim")
            t.setStyleSheet("color:#FFFFFF; font-size:26px; font-weight:850; background:transparent;")
            s = QtWidgets.QLabel("Open Source EDA for Circuit Design")
            s.setStyleSheet("color:rgba(255,255,255,0.9); font-size:12px; background:transparent;")
            hv.addWidget(t)
            hv.addWidget(s)
            v.addWidget(hero)
        except Exception:
            pass
        info = QtWidgets.QLabel(
            "Circuit design, simulation, analysis and PCB layout in one "
            "integrated environment. Built on KiCad and ngspice.\n\n"
            "FOSSEE, IIT Bombay — esim.fossee.in"
        )
        info.setWordWrap(True)
        info.setProperty("cssClass", "muted")
        v.addWidget(info)
        v.addStretch(1)
        return page

    def _build_appearance_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidget(page)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")

        form = QtWidgets.QFormLayout(page)
        form.setSpacing(16)
        form.setContentsMargins(8, 16, 8, 8)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)

        # ── Section: Theme ───────────────────────────────────────────
        theme_group = QtWidgets.QGroupBox("Display Mode")
        theme_group.setProperty("cssClass", "themedGroupBox")
        theme_gl = QtWidgets.QVBoxLayout(theme_group)
        theme_gl.setSpacing(6)

        # Hidden combo retained as the data model so the rest of the dialog's
        # logic (findData/currentData) is unchanged; the visible UI is a
        # segmented control.
        self.theme_combo = QtWidgets.QComboBox()
        self.theme_combo.setVisible(False)
        for label, key in [
            ("Follow operating system", "System"),
            ("Always light",           "Light"),
            ("Always dark",            "Dark"),
        ]:
            self.theme_combo.addItem(label, key)

        seg = QtWidgets.QWidget()
        segh = QtWidgets.QHBoxLayout(seg)
        segh.setContentsMargins(0, 0, 0, 0)
        segh.setSpacing(0)
        self._theme_btns = {}
        for key, lbl in [("System", "Auto"), ("Light", "Light"), ("Dark", "Dark")]:
            b = QtWidgets.QPushButton(lbl)
            b.setCheckable(True)
            b.setProperty("cssClass", "segmentBtn")
            b.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            b.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed,
                            QtWidgets.QSizePolicy.Policy.Fixed)
            b.setMinimumWidth(96)
            b.clicked.connect(lambda _c=False, k=key: self._set_theme_mode(k))
            self._theme_btns[key] = b
            segh.addWidget(b)
        # Trailing stretch so the segment stays a compact group on the left
        # instead of each button expanding to fill (and overflowing) the row.
        segh.addStretch(1)

        theme_row_help = QtWidgets.QLabel(
            "eSim will automatically match your OS theme, or you can force light or dark."
        )
        theme_row_help.setProperty("cssClass", "subtle")
        theme_row_help.setWordWrap(True)
        theme_gl.addWidget(seg)
        theme_gl.addWidget(self.theme_combo)
        theme_gl.addWidget(theme_row_help)
        form.addRow(theme_group)

        # ── Section: Accent ──────────────────────────────────────────
        accent_group = QtWidgets.QGroupBox("Accent Color")
        accent_group.setProperty("cssClass", "themedGroupBox")
        accent_gl = QtWidgets.QVBoxLayout(accent_group)
        accent_gl.setSpacing(8)

        # Swatch row
        swatch_row = QtWidgets.QWidget()
        swatch_layout = QtWidgets.QHBoxLayout(swatch_row)
        swatch_layout.setContentsMargins(0, 0, 0, 0)
        swatch_layout.setSpacing(4)
        self._swatch_buttons = []
        for display_name, hexcode in NAMED_SWATCHES:
            btn = QtWidgets.QPushButton("")
            btn.setFixedSize(32, 32)
            btn.setToolTip(display_name)
            btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            btn.setProperty("cssClass", "swatch")
            btn.setProperty("swatch_value", hexcode)
            bg = "#165982" if hexcode == "default" else hexcode
            btn.setStyleSheet(
                f"background-color: {bg}; "
                "border-radius: 16px;"
            )
            btn.clicked.connect(lambda _checked, val=hexcode: self._on_swatch_clicked(val))
            swatch_layout.addWidget(btn)
            self._swatch_buttons.append(btn)
        swatch_layout.addStretch()
        accent_gl.addWidget(swatch_row)

        # Custom picker
        picker_row = QtWidgets.QHBoxLayout()
        self.primary_color_btn = QtWidgets.QPushButton()
        self.primary_color_btn.setFixedSize(140, 28)
        self.primary_color_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.primary_color_btn.setProperty("cssClass", "colorPick")
        self.primary_color_btn.setToolTip("Pick a custom accent color from the color dialog.")
        self.primary_color_btn.clicked.connect(self._choose_primary_color)
        picker_label = QtWidgets.QLabel("Custom:")
        picker_label.setProperty("cssClass", "subtle")
        picker_row.addWidget(picker_label)
        picker_row.addWidget(self.primary_color_btn)
        picker_row.addStretch()
        accent_gl.addLayout(picker_row)

        accent_help = QtWidgets.QLabel(
            "Used for buttons, selections, active borders, and interactive highlights."
        )
        accent_help.setProperty("cssClass", "subtle")
        accent_help.setWordWrap(True)
        accent_gl.addWidget(accent_help)
        form.addRow(accent_group)

        # ── Section: Surfaces ────────────────────────────────────────
        surface_group = QtWidgets.QGroupBox("Window & Panel Colors")
        surface_group.setProperty("cssClass", "themedGroupBox")
        surface_gl = QtWidgets.QVBoxLayout(surface_group)
        surface_gl.setSpacing(8)

        # Window tint
        win_row = QtWidgets.QHBoxLayout()
        win_label = QtWidgets.QLabel("Window:")
        win_label.setFixedWidth(70)
        self.secondary_color_btn = QtWidgets.QPushButton()
        self.secondary_color_btn.setFixedSize(120, 28)
        self.secondary_color_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.secondary_color_btn.setProperty("cssClass", "colorPick")
        self.secondary_color_btn.setToolTip("Background tint of the main window.")
        self.secondary_color_btn.clicked.connect(self._choose_secondary_color)
        win_row.addWidget(win_label)
        win_row.addWidget(self.secondary_color_btn)
        win_row.addStretch()

        # Panel tint
        panel_row = QtWidgets.QHBoxLayout()
        panel_label = QtWidgets.QLabel("Panels:")
        panel_label.setFixedWidth(70)
        self.internal_bg_color_btn = QtWidgets.QPushButton()
        self.internal_bg_color_btn.setFixedSize(120, 28)
        self.internal_bg_color_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.internal_bg_color_btn.setProperty("cssClass", "colorPick")
        self.internal_bg_color_btn.setToolTip("Surface color for lists, trees, and code editors.")
        self.internal_bg_color_btn.clicked.connect(self._choose_internal_bg_color)
        panel_row.addWidget(panel_label)
        panel_row.addWidget(self.internal_bg_color_btn)
        panel_row.addStretch()

        surface_gl.addLayout(win_row)
        surface_gl.addLayout(panel_row)
        surface_help = QtWidgets.QLabel("Set custom tints for the main window background and inner panels.")
        surface_help.setProperty("cssClass", "subtle")
        surface_help.setWordWrap(True)
        surface_gl.addWidget(surface_help)
        form.addRow(surface_group)

        # ── Live Preview ─────────────────────────────────────────────
        preview_group = QtWidgets.QGroupBox("Live Preview")
        preview_group.setProperty("cssClass", "themedGroupBox")
        preview_gl = QtWidgets.QVBoxLayout(preview_group)
        self.preview_strip = ThemePreview()
        preview_gl.addWidget(self.preview_strip)
        form.addRow(preview_group)

        wrapper = QtWidgets.QVBoxLayout()
        wrapper.setContentsMargins(0, 0, 0, 0)
        wrapper.addWidget(scroll)
        container = QtWidgets.QWidget()
        container.setLayout(wrapper)
        return container

    def _build_editor_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)
        form.setSpacing(14)
        form.setContentsMargins(8, 16, 8, 8)

        # Font group
        font_group = QtWidgets.QGroupBox("Editor Typography")
        font_group.setProperty("cssClass", "themedGroupBox")
        font_gl = QtWidgets.QFormLayout(font_group)
        font_gl.setSpacing(10)

        self.font_family_combo = QtWidgets.QFontComboBox()
        self.font_family_combo.setEditable(False)
        self.font_family_combo.setToolTip("Monospace fonts are recommended for source editors.")
        font_gl.addRow("Font Family:", self.font_family_combo)

        self.font_size_spin = QtWidgets.QSpinBox()
        self.font_size_spin.setRange(8, 24)
        self.font_size_spin.setSuffix(" pt")
        self.font_size_spin.setToolTip("Point size used inside code editors and the console output.")
        font_gl.addRow("Font Size:", self.font_size_spin)

        self.tab_width_spin = QtWidgets.QSpinBox()
        self.tab_width_spin.setRange(2, 8)
        self.tab_width_spin.setSuffix(" spaces")
        self.tab_width_spin.setToolTip("Visual equivalent of a tab character inside the code editors.")
        font_gl.addRow("Tab Width:", self.tab_width_spin)

        form.addRow(font_group)

        hint = QtWidgets.QLabel(
            "These settings affect Verilog, SPICE (.cir) and Makerchip editors "
            "as well as the simulation console. They apply the next time an "
            "editor is opened."
        )
        hint.setProperty("cssClass", "subtle")
        hint.setWordWrap(True)
        form.addRow(hint)

        # Interface animation group: the Aurora button glows are opt-in because
        # they drive continuous repaints. Default off (see Appconfig defaults).
        motion_group = QtWidgets.QGroupBox("Interface Animation")
        motion_group.setProperty("cssClass", "themedGroupBox")
        motion_gl = QtWidgets.QVBoxLayout(motion_group)
        motion_gl.setSpacing(10)
        self.motion_checkbox = QtWidgets.QCheckBox(
            "Enable button glow animations")
        self.motion_checkbox.setToolTip(
            "Animated hover/press glows on buttons and dialogs. Off by default "
            "for performance; takes effect the next time a dialog opens.")
        motion_gl.addWidget(self.motion_checkbox)
        form.addRow(motion_group)

        return page

    # ------------------------------------------------------------------ helpers
    def _find_icon(self):
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(here, "..", "..", "images", "preferences.png"),
            os.path.abspath("images/preferences.png"),
        ]
        for p in candidates:
            p = os.path.normpath(p)
            if os.path.exists(p):
                return p
        return None

    def _refresh_color_buttons(self):
        self._update_color_btn(self.primary_color_btn, self.current_primary_color, "#165982")
        self._update_color_btn(self.secondary_color_btn, self.current_secondary_color, "#FAFBFC")
        self._update_color_btn(self.internal_bg_color_btn, self.current_internal_bg_color, "#FFFFFF")
        # Live preview
        bg = (
            self.current_internal_bg_color
            if self.current_internal_bg_color != "system"
            else ("#1F2937" if self._is_dark() else "#FFFFFF")
        )
        accent = (
            self.current_primary_color
            if self.current_primary_color != "default"
            else ("#53D7FF" if self._is_dark() else "#0077A8")
        )
        panel = (
            self.current_secondary_color
            if self.current_secondary_color != "system"
            else ("#050812" if self._is_dark() else "#F3F7FC")
        )
        text_color = "#F9FAFB" if self._is_dark() else "#142033"
        border_color = "#1D2B45" if self._is_dark() else "#D6E1EE"
        self.preview_strip.setColors({
            "bg": panel,
            "panel": bg,
            "accent": accent,
            "text": text_color,
            "border": border_color,
        })

    def _update_color_btn(self, btn, color_value, default_hex):
        if color_value in ("default", "system", None):
            display_color = default_hex
        else:
            display_color = color_value
        btn.setStyleSheet(
            f"background-color: {display_color}; "
            f"border: 1px solid {_border_for_theme()}; "
            "border-radius: 8px;"
        )

    def _swatch_match(self, value):
        for btn in self._swatch_buttons:
            if btn.property("swatch_value") == value:
                return btn
        return None

    def _on_swatch_clicked(self, value):
        self.current_primary_color = value
        for btn in self._swatch_buttons:
            selected = btn.property("swatch_value") == value
            btn.setProperty("selected", selected)
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        self._refresh_color_buttons()
        self._live_apply()

    def _is_dark(self) -> bool:
        data = self.theme_combo.currentData()
        if data == "Dark":
            return True
        if data == "Light":
            return False
        return (
            QtGui.QGuiApplication.styleHints().colorScheme()
            == QtCore.Qt.ColorScheme.Dark
        )

    # ------------------------------------------------------------------ theme segment
    def _set_theme_mode(self, key):
        idx = self.theme_combo.findData(key)
        if idx >= 0:
            self.theme_combo.setCurrentIndex(idx)
        self._sync_theme_segment()
        self._refresh_color_buttons()
        self._live_apply()

    def _sync_theme_segment(self):
        if not hasattr(self, "_theme_btns"):
            return
        cur = self.theme_combo.currentData()
        for k, b in self._theme_btns.items():
            b.setChecked(k == cur)

    # ------------------------------------------------------------------ live apply
    def _live_apply(self):
        """Debounced re-theme so the running app updates as the user tweaks."""
        if not hasattr(self, "_live_timer"):
            self._live_timer = QtCore.QTimer(self)
            self._live_timer.setSingleShot(True)
            self._live_timer.setInterval(160)
            self._live_timer.timeout.connect(self._apply_preferences)
        self._live_timer.start()

    def _reset_to_defaults_live(self):
        self._reset_to_defaults()
        self._sync_theme_segment()
        self._live_apply()

    def reject(self):
        """Cancel reverts to the snapshot taken when the dialog opened.

        If nothing actually changed during the session (the common "open then
        immediately close/X" case), DON'T re-apply the theme: a needless
        re-polish nudges the toolbars slightly larger. Only revert + re-theme
        when the live preview actually diverged from the opening snapshot.
        """
        try:
            path = os.path.join(self.appconfig.user_home, ".esim", "preferences.json")
            try:
                with open(path, "r") as f:
                    current = json_load(f)
            except Exception:
                current = {}
            keys = ("theme_mode", "accent_color", "secondary_accent_color",
                    "internal_bg_color", "editor_font_family",
                    "editor_font_size", "editor_tab_width")
            # Compare only keys actually present on disk: _orig_prefs is the
            # normalized (defaults-filled) snapshot, while the file holds only
            # explicitly-saved keys, so a plain compare would see phantom diffs.
            changed = any(k in current and current.get(k) != self._orig_prefs.get(k)
                          for k in keys)
            if changed:
                self.appconfig.save_preferences(
                    self._orig_prefs.get("theme_mode", "System"),
                    self._orig_prefs.get("accent_color", "default"),
                    self._orig_prefs.get("secondary_accent_color", "system"),
                    self._orig_prefs.get("internal_bg_color", "system"),
                )
                with open(path, "r") as f:
                    existing = json_load(f)
                existing.update(self._orig_prefs)
                with open(path, "w") as f:
                    json_dump(existing, f)
                app = QtWidgets.QApplication.instance()
                fn = getattr(app, "apply_theme", None)
                if callable(fn):
                    fn()
        except Exception as exc:
            print("Preferences cancel-revert failed:", exc)
        super().reject()

    # ------------------------------------------------------------------ loaders
    def _load_into_widgets(self):
        mode = self.prefs.get("theme_mode", "System")
        idx = self.theme_combo.findData(mode)
        if idx < 0:
            idx = 0
        self.theme_combo.setCurrentIndex(idx)
        self.theme_combo.currentIndexChanged.connect(self._refresh_color_buttons)
        self._sync_theme_segment()

        self.current_primary_color = self.prefs.get("accent_color", "default")
        self.current_secondary_color = self.prefs.get("secondary_accent_color", "system")
        self.current_internal_bg_color = self.prefs.get("internal_bg_color", "system")
        self._refresh_color_buttons()

        try:
            from frontEnd.widgets import mono_family
            _default_mono = mono_family()
        except Exception:
            _default_mono = "Consolas"
        font_family = self.prefs.get("editor_font_family", _default_mono)
        idx = self.font_family_combo.findText(font_family)
        if idx < 0:
            self.font_family_combo.setCurrentText(_default_mono)
        else:
            self.font_family_combo.setCurrentIndex(idx)
        self.font_size_spin.setValue(int(self.prefs.get("editor_font_size", 11)))
        self.tab_width_spin.setValue(int(self.prefs.get("editor_tab_width", 4)))
        self.motion_checkbox.setChecked(
            bool(self.prefs.get("enable_motion", False)))

        # Live-apply editor changes too.
        self.font_family_combo.currentFontChanged.connect(lambda *_: self._live_apply())
        self.font_size_spin.valueChanged.connect(lambda *_: self._live_apply())
        self.tab_width_spin.valueChanged.connect(lambda *_: self._live_apply())
        # Persist the motion toggle; install happens next dialog open.
        self.motion_checkbox.toggled.connect(lambda *_: self._live_apply())

    # ------------------------------------------------------------------ actions
    def _choose_primary_color(self):
        initial = QtGui.QColor(
            "#165982" if self.current_primary_color == "default" else self.current_primary_color
        )
        color = QtWidgets.QColorDialog.getColor(initial, self, "Select Accent Color")
        if color.isValid():
            self.current_primary_color = color.name()
            self._refresh_color_buttons()
            self._live_apply()

    def _choose_secondary_color(self):
        initial = QtGui.QColor(
            "#FAFBFC" if self.current_secondary_color == "system" else self.current_secondary_color
        )
        color = QtWidgets.QColorDialog.getColor(initial, self, "Select Window Color")
        if color.isValid():
            self.current_secondary_color = color.name()
            self._refresh_color_buttons()
            self._live_apply()

    def _choose_internal_bg_color(self):
        initial = QtGui.QColor(
            "#FFFFFF" if self.current_internal_bg_color == "system" else self.current_internal_bg_color
        )
        color = QtWidgets.QColorDialog.getColor(initial, self, "Select Panel Color")
        if color.isValid():
            self.current_internal_bg_color = color.name()
            self._refresh_color_buttons()
            self._live_apply()

    def _reset_to_defaults(self):
        self.theme_combo.setCurrentIndex(self.theme_combo.findData("System"))
        self.current_primary_color = "default"
        self.current_secondary_color = "system"
        self.current_internal_bg_color = "system"
        self.font_family_combo.setCurrentText("Consolas")
        self.font_size_spin.setValue(11)
        self.tab_width_spin.setValue(4)
        self._refresh_color_buttons()

    def _open_preferences_file(self):
        prefs_path = os.path.join(self.appconfig.user_home, ".esim", "preferences.json")
        if not os.path.exists(prefs_path):
            self._apply_preferences()
        try:
            url = QtCore.QUrl.fromLocalFile(prefs_path)
            if QtGui.QDesktopServices.openUrl(url):
                return
        except Exception:
            pass
        try:
            if os.name == "nt":
                os.startfile(prefs_path)
            else:
                opener = "open" if sys_platform_darwin() else "xdg-open"
                import subprocess
                subprocess.Popen([opener, prefs_path])
        except Exception as e:
            QtWidgets.QMessageBox.information(
                self,
                "Preferences file",
                f"Preferences are stored at:\n{prefs_path}\n\n(Automatic open failed: {e})",
            )

    def _collect_prefs(self) -> dict:
        return {
            "theme_mode":                  self.theme_combo.currentData(),
            "accent_color":                self.current_primary_color,
            "secondary_accent_color":      self.current_secondary_color,
            "internal_bg_color":           self.current_internal_bg_color,
            "editor_font_family":          self.font_family_combo.currentText(),
            "editor_font_size":            int(self.font_size_spin.value()),
            "editor_tab_width":            int(self.tab_width_spin.value()),
            "enable_motion":               self.motion_checkbox.isChecked(),
        }

    def _apply_preferences(self):
        prefs = self._collect_prefs()
        self.appconfig.save_preferences(
            prefs["theme_mode"],
            prefs["accent_color"],
            prefs["secondary_accent_color"],
            prefs["internal_bg_color"],
        )
        try:
            path = os.path.join(self.appconfig.user_home, ".esim", "preferences.json")
            with open(path, "r") as f:
                existing = json_load(f)
            existing.update(prefs)
            with open(path, "w") as f:
                json_dump(existing, f)
        except Exception:
            pass

        app = QtWidgets.QApplication.instance()
        fn = getattr(app, "apply_theme", None)
        if callable(fn):
            try:
                fn()
            except Exception as exc:
                print("apply_theme failed:", exc)

        for w in QtWidgets.QApplication.topLevelWidgets():
            if hasattr(w, "update_theme_styles"):
                try:
                    w.update_theme_styles()
                except Exception:
                    pass

    def _save_and_close(self):
        self._apply_preferences()
        self.accept()


def sys_platform_darwin() -> bool:
    import sys
    return sys.platform == "darwin"


import json as _json


def json_load(fp):
    return _json.load(fp)


def json_dump(data, fp):
    return _json.dump(data, fp, indent=2)
