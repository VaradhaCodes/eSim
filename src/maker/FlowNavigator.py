# =========================================================================
#             FILE: FlowNavigator.py
#
#      DESCRIPTION: Workflow-shaped shell for eSim's HDL -> ngspice-block
#                   subsystem. Replaces the flat Makerchip / NgVeri / NGHDL
#                   tab strip (and the flying "Verilog Simulator IDE" dialog)
#                   with a Vivado/Quartus-style stage rail:
#
#                       Author  ->  Verify  ->  Convert  ->  Place
#                                                 (VHDL: NGHDL)
#
#                   Each stage reuses the existing widget unchanged; the rail
#                   just makes the edit -> verify -> build-block -> place flow
#                   visible and exposes one stage at a time (progressive
#                   disclosure). Heavy stages (the Verilog Simulator IDE and the
#                   embedded NGHDL window) are built lazily and guarded so a
#                   missing/broken tool can never take the dock down.
#
#  ORGANIZATION: eSim Team at FOSSEE, IIT Bombay
# =========================================================================
import os

from PyQt6 import QtCore, QtWidgets

from . import Maker
from . import NgVeri


# Stage identifiers (also the QStackedWidget order).
AUTHOR, VERIFY, CONVERT, NGHDL, PLACE = range(5)


class FlowNavigator(QtWidgets.QWidget):
    """Stage-rail container for the HDL -> block workflow."""

    def __init__(self, filecount, parent=None):
        super().__init__(parent)
        self.filecount = filecount
        self._built = {}        # stage id -> True once its widget is realised
        self._complete = set()  # stages the user has finished (rail check-mark)
        self.obj_Maker = None
        self.obj_NgVeri = None
        self.obj_Verifier = None
        self._build_ui()

    # ------------------------------------------------------------------ #
    #  Layout
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Left: the flow rail.
        self.rail = QtWidgets.QListWidget()
        self.rail.setFixedWidth(190)
        self.rail.setSpacing(2)
        self.rail.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.rail.setStyleSheet("""
            QListWidget { background:#f4f6f8; border-right:1px solid #d0d5da;
                          padding-top:8px; font-size:13px; }
            QListWidget::item { padding:10px 12px; margin:1px 6px;
                                border-radius:6px; color:#3a444e; }
            QListWidget::item:selected { background:#1565c0; color:white;
                                         font-weight:bold; }
            QListWidget::item:hover:!selected { background:#e3e8ee; }
        """)
        self._stage_labels = [
            "1  ·  Author", "2  ·  Verify", "3  ·  Convert",
            "VHDL  ·  NGHDL", "4  ·  Place"]
        hints = [
            "Write / load your HDL module(s)",
            "Compile and simulate (Icarus); view waveform",
            "Build an ngspice code-model + KiCad symbol",
            "VHDL path: author → simulate → model (GHDL)",
            "Drop the generated block in your schematic"]
        for label, hint in zip(self._stage_labels, hints):
            item = QtWidgets.QListWidgetItem(label)
            item.setToolTip(hint)
            self.rail.addItem(item)

        # Right: a header + the stacked stage panels.
        right = QtWidgets.QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(0)

        header_bar = QtWidgets.QWidget()
        header_bar.setStyleSheet(
            "QWidget { background:#ffffff; border-bottom:1px solid #e0e4e8; }")
        hb = QtWidgets.QHBoxLayout(header_bar)
        hb.setContentsMargins(14, 6, 10, 6)
        self.header = QtWidgets.QLabel()
        self.header.setStyleSheet("color:#5a6570; font-size:13px;")
        self.header.setTextFormat(QtCore.Qt.TextFormat.RichText)
        hb.addWidget(self.header)
        hb.addStretch(1)
        # One primary action per stage: advance the flow.
        self.btn_next = QtWidgets.QPushButton("Next  ▸")
        self.btn_next.setStyleSheet(
            "QPushButton { background:#1565c0; color:white; border:none;"
            " border-radius:5px; padding:5px 14px; font-weight:bold; }"
            " QPushButton:disabled { background:#c4ccd4; }")
        self.btn_next.clicked.connect(self._go_next)
        hb.addWidget(self.btn_next)
        right.addWidget(header_bar)

        self.stack = QtWidgets.QStackedWidget()
        right.addWidget(self.stack, 1)

        # Stage panels in AUTHOR..PLACE order. Heavy ones start as a
        # placeholder and are swapped in on first visit (see _ensure_stage).
        self._panels = {}
        for stage in (AUTHOR, VERIFY, CONVERT, NGHDL, PLACE):
            holder = QtWidgets.QStackedWidget()  # one-slot holder we can refill
            self._panels[stage] = holder
            self.stack.addWidget(holder)

        outer.addWidget(self.rail)
        outer.addLayout(right, 1)

        self.rail.currentRowChanged.connect(self._on_stage_changed)
        self.rail.setCurrentRow(AUTHOR)

    # ------------------------------------------------------------------ #
    #  Stage activation (lazy build + guard)
    # ------------------------------------------------------------------ #
    def _on_stage_changed(self, stage):
        if stage < 0:
            return
        self._ensure_stage(stage)
        self.stack.setCurrentIndex(stage)
        self._update_header(stage)
        # Returning to Author re-arms Maker's file-watch toggle, exactly as the
        # old tab-change signal did (the toggle thread is torn down elsewhere).
        if stage == AUTHOR and self.obj_Maker is not None:
            self.obj_Maker.refresh_change()

    #: Linear common path (the VHDL/NGHDL branch sits outside it).
    _LINEAR = (AUTHOR, VERIFY, CONVERT, PLACE)

    def _next_stage(self, stage):
        """The stage the Next button should advance to, or None at the end."""
        if stage in self._LINEAR:
            i = self._LINEAR.index(stage)
            if i + 1 < len(self._LINEAR):
                return self._LINEAR[i + 1]
            return None
        return PLACE   # from the NGHDL branch, rejoin at Place

    def _go_next(self):
        nxt = self._next_stage(self.rail.currentRow())
        if nxt is not None:
            self.rail.setCurrentRow(nxt)

    def _update_header(self, stage):
        crumbs = ["Author", "Verify", "Convert", "NGHDL (VHDL)", "Place"]
        parts = []
        for i, name in enumerate(crumbs):
            if i == stage:
                parts.append(f"<b>{name}</b>")
            elif i in self._complete:
                parts.append(f"<span style='color:#2e7d32'>✓ {name}</span>")
            else:
                parts.append(f"<span style='color:#9aa4ad'>{name}</span>")
        sep = " &nbsp;&rsaquo;&nbsp; "
        self.header.setText("Model Creation &nbsp;&mdash;&nbsp; " +
                            sep.join(parts))

        # One primary action per stage: label/disable the Next button.
        nxt = self._next_stage(stage)
        if nxt is None:
            self.btn_next.setText("Done")
            self.btn_next.setEnabled(False)
        else:
            self.btn_next.setText("Next: %s  ▸" % crumbs[nxt])
            self.btn_next.setEnabled(True)

    def _mark_complete(self, stage):
        """Flag a stage finished: a green ✓ in the rail + breadcrumb. Called
        when Verify reports a clean simulation, so the user is steered to
        Convert next (progressive disclosure)."""
        if stage in self._complete:
            self._update_header(self.rail.currentRow())
            return
        self._complete.add(stage)
        item = self.rail.item(stage)
        if item is not None and not item.text().startswith("✓"):
            item.setText("✓  " + self._stage_labels[stage])
        self._update_header(self.rail.currentRow())

    def _set_panel(self, stage, widget):
        holder = self._panels[stage]
        # Wrap big editors in a scroll area, mirroring the old tabbed layout.
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
            elif stage == PLACE:
                self._set_panel(PLACE, self._make_place())
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
    #  Stage factories
    # ------------------------------------------------------------------ #
    def _make_author(self):
        """Author stage = the existing Maker widget (load .v / edit / TL-Verilog
        + Makerchip web IDE, now just one tool among the author actions). Its
        green 'Verilog Simulator IDE' button no longer opens a flying dialog --
        it navigates to the Verify stage instead."""
        self.obj_Maker = Maker.Maker(self.filecount)
        # Hook: Maker.open_verifier defers to this instead of a QDialog.
        self.obj_Maker._verify_hook = lambda: self.rail.setCurrentRow(VERIFY)
        return self.obj_Maker

    def _make_verify(self):
        """Verify stage = the Verilog Simulator IDE, embedded (no flying
        window). Its 'Send to Makerchip' feeds the Author file model and then
        advances the flow to Convert."""
        from .VerilogVerifier import VerilogVerifier
        self.obj_Verifier = VerilogVerifier()
        self.obj_Verifier.sendToNgVeri.connect(self._on_verified)
        # A clean simulation marks Verify done and steers toward Convert.
        self.obj_Verifier.simulationSucceeded.connect(
            lambda: self._mark_complete(VERIFY))
        return self.obj_Verifier

    def _make_convert(self):
        """Convert stage = the existing NgVeri widget. Both convert backends --
        legacy Verilator (static Ngveri.cm) and d_cosim Icarus -- are offered
        side by side, equal billing."""
        self.obj_NgVeri = NgVeri.NgVeri(self.filecount)
        return self.obj_NgVeri

    def _make_nghdl(self):
        """VHDL path = the embedded NGHDL window (GHDL toolchain), built lazily
        and guarded exactly as before."""
        from ngspice_ghdl import Mainwindow
        return Mainwindow(embedded=True)

    def _make_place(self):
        text = (
            "<h3>Place your generated block</h3>"
            "<p>Once <b>Convert</b> reports success, your block is a symbol in "
            "the eSim KiCad libraries:</p>"
            "<ul>"
            "<li><b>eSim_Ngveri</b> — legacy Verilator models</li>"
            "<li><b>eSim_NgVeriCosim</b> — d_cosim (Icarus) models</li>"
            "<li><b>eSim_Nghdl</b> — VHDL (NGHDL) models</li>"
            "</ul>"
            "<p>In KiCad's schematic editor, press <b>A</b> to add a symbol and "
            "pick it from that library, wire it up, then run <b>Simulate</b> "
            "back in eSim.</p>")
        return self._placeholder(text, "")

    # ------------------------------------------------------------------ #
    #  Cross-stage wiring
    # ------------------------------------------------------------------ #
    def _on_verified(self, filepath):
        """A design was verified and sent on: load it into the Author model
        (so the Convert stage picks it up via the existing file flow) and jump
        to Convert."""
        if self.obj_Maker is not None:
            try:
                self.obj_Maker.load_verilog(filepath)
            except Exception:
                pass
        self._mark_complete(VERIFY)   # sending implies it verified
        self._ensure_stage(CONVERT)
        self.rail.setCurrentRow(CONVERT)

    # ------------------------------------------------------------------ #
    def _placeholder(self, message, detail):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(24, 24, 24, 24)
        body = message
        if detail:
            body += ("<br/><br/><small style='color:#8a939b'>Details: " +
                     detail + "</small>")
        label = QtWidgets.QLabel(body)
        label.setWordWrap(True)
        label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        lay.addWidget(label)
        lay.addStretch(1)
        return w
