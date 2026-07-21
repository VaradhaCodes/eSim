# =========================================================================
#             FILE: FlowNavigator.py
#
#      DESCRIPTION: Workflow-shaped shell for eSim's HDL -> ngspice-block
#                   subsystem. Two languages, two distinct paths, picked by a
#                   single segmented toggle so they never collide:
#
#                       Verilog:  Author  ->  Verify  ->  Convert
#                       VHDL:     NGHDL  (its own self-contained tool)
#
#                   The Verilog path is a plain pill tab strip (free
#                   navigation, no wizard gating, no completion ticks). The
#                   VHDL path hands the whole panel to the embedded NGHDL
#                   window. There is no separate "Place" stage -- Convert
#                   already reports the KiCad library the generated symbol
#                   lands in.
#
#                   Each stage reuses the existing widget unchanged. Heavy
#                   stages (the Verilog Simulator IDE and the embedded NGHDL
#                   window) are built lazily and guarded so a missing/broken
#                   tool can never take the dock down.
#
#  ORGANIZATION: eSim Team at FOSSEE, IIT Bombay
# =========================================================================
from PyQt6 import QtCore, QtGui, QtWidgets

try:
    from frontEnd import tokens
except Exception:  # pragma: no cover - flat sys.path (script / harness run)
    import tokens

from . import Maker
from . import NgVeri
from .DesignBus import DesignBus


def _make_bus_closer(bus):
    """destroyed-slot teardown for the design watch.

    ``closeEvent`` stops the DesignBus watchdog on an explicit close, but the
    Model Creation dock is usually torn down by parent destruction (Close
    Project / tab close -> deleteLater), which Qt never delivers a closeEvent
    for -- so the observer thread leaked (H4 / R3-2). Wiring this to
    ``destroyed`` closes the class. It captures only the bus (a plain object),
    never the dying widget, and is idempotent with closeEvent's own close()."""
    def _close(*_a):
        try:
            bus.close()
        except Exception:
            pass
    return _close


# Verilog-path stage ids (also the content-stack order). NGHDL is the VHDL
# path and lives outside this sequence.
AUTHOR, VERIFY, CONVERT, NGHDL = range(4)

VERILOG_STAGES = (AUTHOR, VERIFY, CONVERT)

# Language modes for the top segmented toggle.
VERILOG, VHDL = "verilog", "vhdl"


class FlowNavigator(QtWidgets.QWidget):
    """Two-path container for the HDL -> block workflow.

    A language segment chooses Verilog or VHDL; the Verilog side exposes the
    Author / Verify / Convert stages as plain tabs, the VHDL side embeds NGHDL.
    """

    #: re-emitted from the Verify stage's VerilogVerifier.waveformReady so the
    #: host (DockArea) can give the plot its own full-width eSim tab. The signal
    #: exists on the navigator from construction even though the verifier is
    #: built lazily, so the host can connect once when the dock is created.
    waveformRequested = QtCore.pyqtSignal(object)

    def __init__(self, filecount, parent=None):
        super().__init__(parent)
        self.filecount = filecount
        self._built = {}        # stage id -> True once its widget is realised
        self._mode = VERILOG    # active language path
        self._current = AUTHOR  # active Verilog stage (kept across modes)
        self.obj_Maker = None
        self.obj_NgVeri = None
        self.obj_Verifier = None
        # The one design every Verilog stage is a view on. Author and Verify
        # render from / collect into it in memory; Convert materializes it to
        # disk. The bus also watches the file for genuine external edits.
        self.bus = DesignBus(filecount, self)
        self.bus.externalChange.connect(self._on_external_change)
        # Stop the watch even when dock destruction skips closeEvent.
        self.destroyed.connect(_make_bus_closer(self.bus))
        self._build_ui()

    # ------------------------------------------------------------------ #
    #  Layout
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # --- Top bar -------------------------------------------------- #
        # Left  : a segmented [ Verilog | VHDL ] language toggle (filled pill).
        # Middle: the Verilog stage tabs (plain pill tabs, shown only in
        #         Verilog mode -- in VHDL mode NGHDL owns the whole panel).
        # Right : the contextual fullscreen toggle.
        # Stages are freely clickable in any order: no chevrons, no Next
        # button, no gating, no completion ticks. The strip is a guide, not a
        # wizard. Styling tracks the Aurora light/dark palette (neutral greys +
        # the cyan accent) via _apply_pill_theme(), matching the rest of eSim.
        self.tabbar = QtWidgets.QWidget()
        self.tabbar.setObjectName("flowTabBar")
        # Colors come from _apply_pill_theme() (Aurora light/dark tokens),
        # applied at the end of _build_ui and again on every PaletteChange.
        tb = QtWidgets.QHBoxLayout(self.tabbar)
        # Symmetric vertical margins: the toggle and the stage tabs float
        # centred in the bar instead of sitting flush on its bottom border.
        tb.setContentsMargins(12, 8, 12, 8)
        tb.setSpacing(0)

        self._build_mode_toggle(tb)

        # A thin vertical divider sets the language segment apart from the
        # stage tabs so the two controls never read as one strip.
        divider = QtWidgets.QFrame()
        divider.setFrameShape(QtWidgets.QFrame.Shape.VLine)
        # Theme-neutral hairline: a low-alpha grey reads on both the Aurora
        # light and dark palettes (the old #d4d8dc glared on the dark theme).
        divider.setStyleSheet("color: rgba(124,141,168,0.40);")
        tb.addSpacing(12)
        tb.addWidget(divider)
        tb.addSpacing(12)

        self._build_stage_tabs(tb)

        tb.addStretch(1)

        # Contextual fullscreen toggle, in this panel's own header (not a
        # global toolbar): fullscreen the Makerchip panel and dock it back.
        from frontEnd.FullScreen import FullScreenToggle
        tb.addWidget(FullScreenToggle())

        outer.addWidget(self.tabbar)

        # A non-modal bar shown ONLY when the design file is changed outside eSim
        # (e.g. opened in another editor). It replaces the old modal "click
        # Refresh" popup that fired on eSim's own writes during navigation.
        outer.addWidget(self._build_reload_bar())

        # --- Content -------------------------------------------------- #
        # One holder per panel (Author/Verify/Convert/NGHDL). Heavy ones start
        # empty and are swapped in on first visit (see _ensure_stage).
        self.stack = QtWidgets.QStackedWidget()
        outer.addWidget(self.stack, 1)

        self._panels = {}
        for stage in (AUTHOR, VERIFY, CONVERT, NGHDL):
            holder = QtWidgets.QStackedWidget()  # one-slot refillable holder
            self._panels[stage] = holder
            self.stack.addWidget(holder)

        # Start on the Verilog path, Author stage.
        self._mode_btns[VERILOG].setChecked(True)
        self._tabs[AUTHOR].setChecked(True)
        self._select_mode(VERILOG)

        # Paint the segmented toggle / stage tabs / reload bar from the active
        # Aurora palette (re-runs on PaletteChange via changeEvent).
        self._apply_pill_theme()

    def _build_mode_toggle(self, tb):
        """The [ Verilog | VHDL ] segmented control."""
        self._mode_group = QtWidgets.QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_btns = {}

        # Styling (Aurora light/dark tokens + per-end pill radii) is applied
        # in _apply_pill_theme(); here we only build the widgets.
        for mode, label, tip in (
                (VERILOG, "Verilog", "Verilog flow: author, verify, convert"),
                (VHDL, "VHDL", "VHDL flow: the NGHDL model creator")):
            btn = QtWidgets.QPushButton(label)
            btn.setCheckable(True)
            btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            btn.setToolTip(tip)
            self._mode_group.addButton(btn)
            self._mode_btns[mode] = btn
            tb.addWidget(btn)
        self._mode_btns[VERILOG].clicked.connect(
            lambda: self._select_mode(VERILOG))
        self._mode_btns[VHDL].clicked.connect(
            lambda: self._select_mode(VHDL))

    def _build_stage_tabs(self, tb):
        """The Verilog Author / Verify / Convert stage tabs."""
        self._stage_tabs_w = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(self._stage_tabs_w)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        labels = {AUTHOR: "Author", VERIFY: "Verify", CONVERT: "Convert"}
        hints = {
            AUTHOR:  "Write / load your Verilog module(s)",
            VERIFY:  "Compile and simulate (Icarus); view the waveform",
            CONVERT: "Build an ngspice code-model + KiCad symbol"}

        # These are the panel's primary navigation, so they read big and bold.
        # font-weight is kept CONSTANT (600) across states on purpose: bolding
        # only on :checked made the selected label wider than the size hint Qt
        # computed for the normal weight, so the text was clipped at the
        # corners. The active stage is signalled by colour + a filled pill
        # tint (no underline: Qt curls a border-bottom's ends when the top
        # corners carry a radius, which read as a wobbly squiggle on Windows).
        # Colours come from _apply_pill_theme() (Aurora light/dark tokens).
        self._stage_group = QtWidgets.QButtonGroup(self)
        self._stage_group.setExclusive(True)
        self._tabs = {}
        for stage in VERILOG_STAGES:
            btn = QtWidgets.QPushButton(labels[stage])
            btn.setCheckable(True)
            btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            btn.setToolTip(hints[stage])
            self._stage_group.addButton(btn, stage)
            self._tabs[stage] = btn
            row.addWidget(btn)
        self._stage_group.idClicked.connect(self._goto_stage)
        tb.addWidget(self._stage_tabs_w)

    # ------------------------------------------------------------------ #
    #  Theming — Aurora light/dark tokens (segmented toggle + stage tabs)
    # ------------------------------------------------------------------ #
    def _is_dark(self):
        """True when the active Aurora palette is the dark variant."""
        app = QtWidgets.QApplication.instance()
        if app is not None:
            try:
                c = app.palette().color(QtGui.QPalette.ColorRole.Window)
                if c.isValid():
                    return c.lightness() < 128
            except Exception:
                pass
        return False

    def _pill_tokens(self):
        """Color set for the flow chrome, keyed to the active theme.

        Every value is READ from Aurora's tokens (frontEnd/tokens.py) — the
        previous version claimed the same thing while hand-copying the hexes,
        so a palette retune drifted this bar silently (UI_AUDIT §2.3). Four
        mappings are deliberately asymmetric and each is called out below;
        everything else is the same token in both themes.
        """
        dark = self._is_dark()
        t = tokens.theme(dark)
        acc = tokens.hex_to_rgb(t["accent"])
        warn = tokens.hex_to_rgb(t["warning"])

        def rgba(rgb, a):
            return "rgba(%d,%d,%d,%s)" % (rgb[0], rgb[1], rgb[2], a)

        return {
            # The strip itself. Dark lifts it off the window (bg_raise); light
            # sits it FLAT on the window (bg) — light chrome strips read as
            # part of the page, and lifting it here shifts the whole header.
            "bar_bg": t["bg_raise"] if dark else t["bg"],
            "bar_border": t["stroke"],
            "seg_bg": t["surface"], "seg_border": t["stroke"],
            "seg_fg": t["text_muted"], "seg_hover_fg": t["text"],
            "accent": t["accent"], "accent_fg": t["text_invert"],
            "stage_fg": t["text_muted"], "stage_hover_fg": t["text"],
            # Hover moves one step AWAY from the strip in the direction of
            # contrast: dark lifts to surface_2, light sinks to bg_sunken —
            # which is exactly what the light sheet's own hover rules paint.
            "stage_hover_bg": t["surface_2"] if dark else t["bg_sunken"],
            "stage_checked_fg": t["accent"],
            # The checked tint needs more alpha on dark to read at all.
            "stage_checked_bg": rgba(acc, "0.18" if dark else "0.13"),
            "reload_bg": rgba(warn, "0.10"),
            "reload_border": t["warning"],
            # Dark can put warning-yellow text on a dark wash; light cannot —
            # #D97706 on its own 10% tint is 3.2:1. Amber-800 is 5.9:1, the
            # same call STYLE_LIGHT's InfoBar makes (S2), and the one value
            # here that is not a token.
            "reload_fg": t["warning"] if dark else "#92400E",
        }

    def _apply_pill_theme(self):
        """(Re)paint the tab bar, segmented toggle, stage tabs and reload bar
        from the active Aurora palette. Safe to call repeatedly; runs once at
        build time and again on every PaletteChange."""
        t = self._pill_tokens()
        if hasattr(self, "tabbar"):
            self.tabbar.setStyleSheet(
                f"QWidget#flowTabBar {{ background:{t['bar_bg']};"
                f" border-bottom:1px solid {t['bar_border']}; }}")
        # Segmented [ Verilog | VHDL ] toggle — per-end radii read as one pill.
        base = (f"QPushButton {{ border:1px solid {t['seg_border']};"
                f" background:{t['seg_bg']}; color:{t['seg_fg']};"
                " font-size:13px; padding:8px 18px; }"
                f" QPushButton:hover {{ color:{t['seg_hover_fg']}; }}"
                f" QPushButton:checked {{ background:{t['accent']};"
                f" color:{t['accent_fg']}; border-color:{t['accent']}; }}")
        ends = {
            VERILOG: " QPushButton { border-top-left-radius:5px;"
                     " border-bottom-left-radius:5px; }",
            VHDL:    " QPushButton { border-top-right-radius:5px;"
                     " border-bottom-right-radius:5px; border-left:none; }",
        }
        for mode, btn in getattr(self, "_mode_btns", {}).items():
            btn.setStyleSheet(base + ends[mode])
        # Author / Verify / Convert stage tabs. The active stage is a filled
        # rounded pill (tint + accent text) — deliberately NOT an underline:
        # Qt renders a border-bottom with top-corner radii as a curled
        # "smile" under the tab, which looked broken on Windows.
        pill_qss = (
            f"QPushButton {{ border:none; background:transparent;"
            f" color:{t['stage_fg']}; font-size:14px; font-weight:600;"
            " padding:8px 22px; border-radius:8px; }"
            f"QPushButton:hover {{ color:{t['stage_hover_fg']};"
            f" background:{t['stage_hover_bg']}; }}"
            f"QPushButton:checked {{ color:{t['stage_checked_fg']};"
            f" background:{t['stage_checked_bg']}; }}")
        for btn in getattr(self, "_tabs", {}).values():
            btn.setStyleSheet(pill_qss)
        # External-edit reload banner — Aurora warning tones.
        if hasattr(self, "_reload_bar"):
            self._reload_bar.setStyleSheet(
                f"QFrame#reloadBar {{ background:{t['reload_bg']};"
                f" border-bottom:1px solid {t['reload_border']}; }}"
                f" QLabel {{ color:{t['reload_fg']}; }}")

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.Type.PaletteChange:
            self._apply_pill_theme()

    # ------------------------------------------------------------------ #
    #  Mode + stage switching (lazy build + guard)
    # ------------------------------------------------------------------ #
    def _select_mode(self, mode):
        """Switch language path. Verilog shows the stage tabs and the last
        active stage; VHDL hides them and hands the panel to NGHDL."""
        if mode == VHDL:
            self._sync_out(self._current)   # flush the Verilog design first
            self._mode = VHDL
            # Keep the segmented toggle in sync when invoked programmatically
            # (NGHDL launcher / Welcome tile): otherwise the pill still reads
            # "Verilog" while the NGHDL panel is shown.
            self._mode_btns[VHDL].setChecked(True)
            self._stage_tabs_w.setVisible(False)
            self._ensure_stage(NGHDL)
            self.stack.setCurrentIndex(NGHDL)
        else:
            self._mode = VERILOG
            self._mode_btns[VERILOG].setChecked(True)
            self._stage_tabs_w.setVisible(True)
            self._goto_stage(self._current)

    def _goto_stage(self, stage):
        """Switch to a Verilog stage. Free navigation -- any stage, any order,
        no gating. The one current design is carried between stages so they
        always work on the same thing (see _sync_out / _sync_in). Safe from a
        tab click or programmatically."""
        if stage not in VERILOG_STAGES:
            return
        prev = self._current
        self._current = stage
        # Selecting a stage implies the Verilog path.
        if self._mode != VERILOG:
            self._mode = VERILOG
            self._mode_btns[VERILOG].setChecked(True)
            self._stage_tabs_w.setVisible(True)
        btn = self._tabs.get(stage)
        if btn is not None and not btn.isChecked():
            btn.setChecked(True)
        self._sync_out(prev)            # capture the stage we are leaving
        self._ensure_stage(stage)
        self._sync_in(stage)            # feed the shared design to the new one
        self.stack.setCurrentIndex(stage)

    def goto_verify(self):
        """Bring the Verify stage forward. Used by the waveform tab's 'Back to
        Verify' control so closing/leaving the plot lands on the editor that
        produced it (not whatever stage happens to be current)."""
        self._goto_stage(VERIFY)

    # ------------------------------------------------------------------ #
    #  Shared current-design model
    #
    #  There is ONE design, held in memory by ``self.bus`` (a DesignBus). Every
    #  Verilog stage is a *view* on it: Author and Verify render from / collect
    #  into it, Convert materializes it to a file at action time. Switching tabs
    #  carries the design in memory -- no disk round-trip -- so navigating in any
    #  order (or skipping Verify) always lands on the same current design, with
    #  no "send"/hand-off step. Convert still reads Maker.verilogFile[filecount],
    #  which the bus mirrors.
    # ------------------------------------------------------------------ #
    def _sync_out(self, stage):
        """Collect a stage's edits into the shared design before leaving it (in
        memory). Author streams its edits live, so only Verify needs collecting;
        Convert never edits the source."""
        if stage == VERIFY and self.obj_Verifier is not None:
            try:
                self.obj_Verifier.collect_into_bus()
            except Exception:
                pass

    def _sync_in(self, stage):
        """Show the shared design in the stage becoming visible. Author renders
        live via the bus signal, so it needs nothing here; Verify re-renders;
        Convert materializes the design to disk so its toolchain reads a current
        file."""
        try:
            if stage == VERIFY and self.obj_Verifier is not None:
                self.obj_Verifier.render_from_bus()
            elif stage == CONVERT:
                self.bus.materialize()
        except Exception:
            pass

    def _set_panel(self, stage, widget):
        holder = self._panels[stage]
        while holder.count():
            holder.removeWidget(holder.widget(0))
        holder.addWidget(widget)

    def _ensure_stage(self, stage):
        if self._built.get(stage):
            return
        self._built[stage] = True
        try:
            if stage == AUTHOR:
                self._set_panel(AUTHOR, self._scroll(self._make_author()))
            elif stage == VERIFY:
                self._set_panel(VERIFY, self._make_verify())
            elif stage == CONVERT:
                self._set_panel(CONVERT, self._scroll(self._make_convert()))
            elif stage == NGHDL:
                self._set_panel(NGHDL, self._make_nghdl())
        except Exception as e:       # never let one stage take down the dock
            self._built[stage] = False
            self._set_panel(stage, self._placeholder(
                "This stage could not be loaded.", str(e)))

    @staticmethod
    def _scroll(widget):
        area = QtWidgets.QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(widget)
        return area

    # ------------------------------------------------------------------ #
    #  External-edit reload bar (non-modal)
    # ------------------------------------------------------------------ #
    def _build_reload_bar(self):
        self._reload_bar = QtWidgets.QFrame()
        self._reload_bar.setObjectName("reloadBar")
        # Colors set by _apply_pill_theme() (Aurora warning tones, theme-aware).
        row = QtWidgets.QHBoxLayout(self._reload_bar)
        row.setContentsMargins(12, 6, 12, 6)
        self._reload_label = QtWidgets.QLabel()
        row.addWidget(self._reload_label)
        row.addStretch(1)
        reload_btn = QtWidgets.QPushButton("Reload")
        reload_btn.clicked.connect(self._do_reload)
        keep_btn = QtWidgets.QPushButton("Keep mine")
        keep_btn.clicked.connect(self._dismiss_reload)
        row.addWidget(reload_btn)
        row.addWidget(keep_btn)
        self._reload_bar.setVisible(False)
        return self._reload_bar

    def _on_external_change(self, path):
        """The design file changed on disk outside eSim. Offer a reload instead
        of nagging modally -- and never clobber unsaved edits automatically."""
        import os
        self._reload_label.setText(
            # U+FE0E after the sign forces the monochrome text glyph; without
            # it Windows substitutes the Segoe UI Emoji colour warning sign.
            f"⚠︎  {os.path.basename(path)} was changed on disk "
            f"outside eSim.")
        self._reload_bar.setVisible(True)

    def _do_reload(self):
        self._reload_bar.setVisible(False)
        if self.bus.path:
            self.bus.load_from_disk(self.bus.path)
            self._sync_in(self._current)   # re-render the visible stage

    def _dismiss_reload(self):
        self._reload_bar.setVisible(False)

    def closeEvent(self, event):
        if self.bus is not None:
            self.bus.close()
        super().closeEvent(event)

    # ------------------------------------------------------------------ #
    #  Stage factories
    # ------------------------------------------------------------------ #
    def _make_author(self):
        """Author stage = the existing Maker widget (load .v / edit / TLV
        + Makerchip web IDE). Its 'Verilog Simulator IDE' button no longer
        opens a flying dialog -- it navigates to the Verify stage instead."""
        self.obj_Maker = Maker.Maker(self.filecount, bus=self.bus)
        self.obj_Maker._verify_hook = lambda: self._goto_stage(VERIFY)
        return self.obj_Maker

    def _make_verify(self):
        """Verify stage = the Verilog Simulator IDE, embedded (no flying
        window). It edits the shared current design: the Flow Navigator feeds
        it the design on entry and collects its edits back on exit (see
        _sync_in / _sync_out), so there is no manual 'send' step."""
        from .VerilogVerifier import VerilogVerifier
        self.obj_Verifier = VerilogVerifier()
        self.obj_Verifier.set_design_bus(self.bus)
        # Bubble the waveform up to the host so it docks as its own eSim tab.
        self.obj_Verifier.waveformReady.connect(self.waveformRequested)
        return self.obj_Verifier

    def _make_convert(self):
        """Convert stage = the existing NgVeri widget. Both convert backends --
        legacy Verilator (static Ngveri.cm) and d_cosim Icarus -- are offered
        side by side, equal billing. On success it reports the KiCad library
        the symbol lands in, so there is no separate 'Place' stage."""
        self.obj_NgVeri = NgVeri.NgVeri(self.filecount)
        return self.obj_NgVeri

    def _make_nghdl(self):
        """VHDL path = the embedded NGHDL window (GHDL toolchain), built lazily
        and guarded exactly as before. Doctor-gated: a missing GHDL/make/
        ghdl.cm shows an actionable placeholder instead of letting the NGHDL
        window fail mid-build later."""
        from . import ToolchainCheck
        doctor_msg = ToolchainCheck.failure_message(ToolchainCheck.NGHDL)
        if doctor_msg:
            doctor_msg += ('\nAfter fixing, close and reopen this Makerchip '
                           'tab.')
            return self._placeholder(
                "<b>The NGHDL / GHDL VHDL toolchain is incomplete.</b>",
                doctor_msg.replace('\n', '<br/>'))
        from ngspice_ghdl import Mainwindow
        return Mainwindow(embedded=True)

    # ------------------------------------------------------------------ #
    def _placeholder(self, message, detail):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(24, 24, 24, 24)
        body = message
        if detail:
            # Inline HTML cannot inherit a QSS colour, so the tone has to be
            # named here -- from the palette, not from a grey that belongs to
            # neither theme (it was #8A939B, 3.0:1 on the light page).
            muted = tokens.theme(self._is_dark())["text_muted"]
            body += ("<br/><br/><small style='color:%s'>Details: " % muted +
                     detail + "</small>")
        label = QtWidgets.QLabel(body)
        label.setWordWrap(True)
        label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        lay.addWidget(label)
        lay.addStretch(1)
        return w
