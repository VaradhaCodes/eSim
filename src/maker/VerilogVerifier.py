import os
import re
import shutil
import tempfile
from PyQt6 import QtCore, QtGui, QtWidgets

# eSim is PyQt6-only. These aliases keep the call sites below terse.
TEXT_SELECTABLE = QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
WIN_MAX = QtCore.Qt.WindowType.WindowMaximizeButtonHint
WIN_MIN = QtCore.Qt.WindowType.WindowMinimizeButtonHint
ORIENT_HORIZ = QtCore.Qt.Orientation.Horizontal
FONT_BOLD = QtGui.QFont.Weight.Bold
TAB_RIGHT = QtWidgets.QTabBar.ButtonPosition.RightSide

import numpy as np
from ngspiceSimulation.plot_window import plotWindow

# Shared d_cosim toolchain resolver: it already knows where iverilog lives
# (env ESIM_IVERILOG -> ~/.nghdl/config.ini [COSIM] -> PATH), recorded by the
# installer. The verifier reuses it so both features find the same iverilog
# instead of maintaining a second, PATH-only locator that misses prefix builds.
try:
    from . import CosimConfig
    from .hdl import icarus, jobs
    from .hdl.vcd import parse_vcd_for_plot
    from .hdl.ports import extract_ports, generate_stub_testbench
except ImportError:  # launched with src/ on sys.path (absolute-import style)
    from maker import CosimConfig
    from maker.hdl import icarus, jobs
    from maker.hdl.vcd import parse_vcd_for_plot
    from maker.hdl.ports import extract_ports, generate_stub_testbench

class VcdPlotWindow(plotWindow):
    def __init__(self, timestamps, signals_data, signal_types, project_name="Verilog Simulation", parent=None):
        self.timestamps = timestamps
        self.signals_data = signals_data
        self.signal_types = signal_types
        # Pass a dummy path to bypass NGSpice reading
        super().__init__("dummy_path", project_name, parent)

    def _parse_tran_start_time(self):
        return 0.0

    def load_simulation_data(self):
        self._cursor_interp_cache.clear()
        self._drawn_signature = None
        self._tran_start_time = self._parse_tran_start_time()
        
        class MockDataExt:
            def __init__(self):
                self.x = np.array([])
                self.y = []
                self.NBList = []
                self.NBIList = []
                self.volts_length = 0
                self.analysisType = 1 # TRANSIENT
                self.dec = 0
            def computeAxes(self): pass
            def numVals(self): return [len(self.y), self.volts_length]
            def openFile(self, path): return [1, 0]

        self.obj_dataext = MockDataExt()
        
        if self.timestamps and self.signals_data:
            self.obj_dataext.x = np.array(self.timestamps, dtype=np.float64)
            for name, y_vals in self.signals_data.items():
                self.obj_dataext.NBList.append(name)
                self.obj_dataext.y.append(np.array(y_vals, dtype=np.float64))
            self.obj_dataext.volts_length = len(self.obj_dataext.NBList)
            
        self.plot_type = [1, 0]
        self._rebuild_nb_sorted()
        self.data_info = self.obj_dataext.numVals()
        self.volts_length = self.data_info[1]
        
        self.analysis_label.setText("Verilog Transient Analysis")
        self.populate_waveform_list()
        self._apply_persisted_layout()
        
        self.radio_timing.setEnabled(True)
        self.radio_timing.setChecked(True)

# Default Verilog Design Example
DEFAULT_DESIGN = """module counter (
    input clk,
    input rst,
    output reg [3:0] count
);

  always @(posedge clk or posedge rst) begin
    if (rst) 
      count <= 0;
    else 
      count <= count + 1;
  end

endmodule
"""

# Default Testbench Example
DEFAULT_TB = """`timescale 1ns/1ps

module tb_counter;
  reg clk;
  reg rst;
  wire [3:0] count;

  counter uut (
    .clk(clk),
    .rst(rst),
    .count(count)
  );

  always #5 clk = ~clk;

  initial begin
    clk = 0;
    rst = 1;
    #20;
    rst = 0;
    #200;
    $finish;
  end

  initial begin
    $dumpfile("sim_out.vcd");
    $dumpvars(0, tb_counter);
  end

endmodule
"""



class LineNumberArea(QtWidgets.QWidget):
    def __init__(self, editor):
        super().__init__(editor)
        self.codeEditor = editor

    def sizeHint(self):
        return QtCore.QSize(self.codeEditor.lineNumberAreaWidth(), 0)

    def paintEvent(self, event):
        self.codeEditor.lineNumberAreaPaintEvent(event)

class CodeEditor(QtWidgets.QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.lineNumberArea = LineNumberArea(self)
        self.blockCountChanged.connect(self.updateLineNumberAreaWidth)
        self.updateRequest.connect(self.updateLineNumberArea)
        self.updateLineNumberAreaWidth(0)

    def lineNumberAreaWidth(self):
        digits = 1
        max_num = max(1, self.blockCount())
        while max_num >= 10:
            max_num //= 10
            digits += 1
        space = 5 + self.fontMetrics().horizontalAdvance('9') * digits
        return space

    def updateLineNumberAreaWidth(self, _):
        self.setViewportMargins(self.lineNumberAreaWidth(), 0, 0, 0)

    def updateLineNumberArea(self, rect, dy):
        if dy:
            self.lineNumberArea.scroll(0, dy)
        else:
            self.lineNumberArea.update(0, rect.y(), self.lineNumberArea.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self.updateLineNumberAreaWidth(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cr = self.contentsRect()
        self.lineNumberArea.setGeometry(QtCore.QRect(cr.left(), cr.top(), self.lineNumberAreaWidth(), cr.height()))

    def lineNumberAreaPaintEvent(self, event):
        painter = QtGui.QPainter(self.lineNumberArea)
        painter.fillRect(event.rect(), QtGui.QColor("#f8f9fa")) # Light grey gutter
        
        # Subtle right border for the gutter
        painter.setPen(QtGui.QColor("#dee2e6"))
        painter.drawLine(self.lineNumberArea.width() - 1, 0, self.lineNumberArea.width() - 1, event.rect().height())
        
        block = self.firstVisibleBlock()
        blockNumber = block.blockNumber()
        top = self.blockBoundingGeometry(block).translated(self.contentOffset()).top()
        bottom = top + self.blockBoundingRect(block).height()
        
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(blockNumber + 1)
                painter.setPen(QtGui.QColor("#adb5bd")) # Subtle text color
                painter.drawText(0, int(top), self.lineNumberArea.width() - 4, self.fontMetrics().height(),
                                 QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter, number)
            
            block = block.next()
            top = bottom
            bottom = top + self.blockBoundingRect(block).height()
            blockNumber += 1

class ConsoleEdit(QtWidgets.QTextEdit):
    error_clicked = QtCore.pyqtSignal(int, str)
    
    def mouseDoubleClickEvent(self, e):
        cursor = self.cursorForPosition(e.pos())
        cursor.select(QtGui.QTextCursor.SelectionType.LineUnderCursor)
        line_text = cursor.selectedText()

        # Match any .v or .sv filename in iverilog error format: filename.v:15: message
        match = re.search(r'([\w.-]+\.(?:v|sv)):(\d+):', line_text)
        if match:
            filename = match.group(1)
            line_num = int(match.group(2))
            self.error_clicked.emit(line_num, filename)

        super().mouseDoubleClickEvent(e)


class VerilogHighlighter(QtGui.QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlightingRules = []

        keywordFormat = QtGui.QTextCharFormat()
        keywordFormat.setForeground(QtGui.QColor("#0000ff")) # Blue for keywords
        keywordFormat.setFontWeight(QtGui.QFont.Weight.Bold)
        keywords = [
            "module", "endmodule", "input", "output", "inout", "wire", "reg", "logic",
            "assign", "always", "initial", "begin", "end", "if", "else", "for", "while",
            "case", "endcase", "posedge", "negedge", "parameter", "localparam", "integer"
        ]
        for word in keywords:
            pattern = QtCore.QRegularExpression(rf"\b{word}\b")
            self.highlightingRules.append((pattern, keywordFormat))

        # Numbers
        numberFormat = QtGui.QTextCharFormat()
        numberFormat.setForeground(QtGui.QColor("#098658")) # Greenish-teal for numbers
        self.highlightingRules.append((QtCore.QRegularExpression(r"\b\d+'[bBoOdDhH][0-9a-fA-F_xzXZ]+\b"), numberFormat))
        self.highlightingRules.append((QtCore.QRegularExpression(r"\b\d+\b"), numberFormat))

        # Comments
        commentFormat = QtGui.QTextCharFormat()
        commentFormat.setForeground(QtGui.QColor("#008000")) # Dark green for comments
        self.highlightingRules.append((QtCore.QRegularExpression(r"//[^\n]*"), commentFormat))
        self.highlightingRules.append((QtCore.QRegularExpression(r"/\*[\s\S]*?\*/"), commentFormat))

        # System Tasks
        sysTaskFormat = QtGui.QTextCharFormat()
        sysTaskFormat.setForeground(QtGui.QColor("#795e26")) # Brown for $tasks
        self.highlightingRules.append((QtCore.QRegularExpression(r"\$\w+\b"), sysTaskFormat))

    def highlightBlock(self, text):
        for pattern, format in self.highlightingRules:
            matchIterator = pattern.globalMatch(text)
            while matchIterator.hasNext():
                match = matchIterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), format)


class VerilogVerifier(QtWidgets.QWidget):
    sendToNgVeri = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.popup_dialog = None
        self.current_timestamps = None
        self.current_signals_data = None
        self.current_signal_types = None

        # Async run state: the in-flight compile/sim worker thread and its
        # cancel token (None when idle). See _start_job / cancel_running_job.
        self._active_job = None
        self._active_cancel = None
        self._busy_buttons = []

        self.init_ui()

    def prompt_install_or_path(self, tool_name):
        win_link = "https://bleyer.org/icarus/"
        lin_link = "sudo apt update && sudo apt install -y iverilog"
        
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle(f"{tool_name} Not Found")
        win_hint = r"C:\iverilog\bin"
        lin_hint = "/usr/bin or /usr/local/bin"
        
        full_text = (
            f"The executable '{tool_name}' was not found on your system PATH.\n\n"
            f"You can install it from/using:\n- Windows: {win_link}\n- Linux: {lin_link}\n\n"
            f"(Note: eSim does not endorse or vouch for the security of external links. "
            f"Please verify and download at your own discretion.)\n\n"
            f"If it is already installed but auto-detect failed, you can manually locate the executable '{tool_name}'.\n\n"
            f"Hint: On Windows, it is usually found in {win_hint}. On Linux, check {lin_hint}."
        )
        msg.setText(full_text)
        msg.setTextInteractionFlags(TEXT_SELECTABLE)
        
        btn_select = msg.addButton("Manually Locate Executable", QtWidgets.QMessageBox.ButtonRole.ActionRole)
        msg.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        
        msg.exec()
        
        if msg.clickedButton() == btn_select:
            filter_str = f"{tool_name}.exe (*.exe);;All Files (*)" if os.name == 'nt' else f"{tool_name} ({tool_name});;All Files (*)"
            filepath, _ = QtWidgets.QFileDialog.getOpenFileName(self, f"Locate {tool_name} Executable", "", filter_str)
            if filepath:
                return filepath
        return None

    def silent_find_iverilog(self):
        # Single source of truth: CosimConfig resolves env -> ~/.nghdl/config.ini
        # [COSIM] -> bundled library/bin -> PATH -> platform default. Shared with
        # the d_cosim build so both features find the same iverilog, including
        # prefix builds that are not on PATH.
        return CosimConfig.iverilog_binary()

    def silent_find_vvp(self):
        return CosimConfig.vvp_binary()

    def find_iverilog(self):
        path = self.silent_find_iverilog()
        if path:
            return path

        path = self.prompt_install_or_path("iverilog")
        if path:
            # Persist into config.ini (not private QSettings) so the d_cosim
            # build sees the manually-located compiler too.
            CosimConfig.set_manual_paths(iverilog_path=path)
            self.check_iverilog_lock()
            return path
        return None

    def find_vvp(self):
        path = self.silent_find_vvp()
        if path:
            return path

        path = self.prompt_install_or_path("vvp")
        if path:
            CosimConfig.set_manual_paths(vvp_path=path)
            return path
        return None

    def attempt_manual_unlock(self):
        # Try to automatically detect first in case it was just installed
        path = self.silent_find_iverilog()

        if not path:
            path = self.prompt_install_or_path("iverilog")

        if path:
            # Auto-pair vvp from the same dir and persist both in config.ini so
            # d_cosim shares the manually-located toolchain.
            vvp_path = os.path.join(
                os.path.dirname(path), "vvp.exe" if os.name == 'nt' else "vvp")
            CosimConfig.set_manual_paths(
                iverilog_path=path,
                vvp_path=vvp_path if os.path.exists(vvp_path) else None)
            self.check_iverilog_lock()

    def check_iverilog_lock(self):
        if not self.silent_find_iverilog():
            self.lock_ui()
        else:
            self.unlock_ui()

    def lock_ui(self):
        self.editor_tabs.setEnabled(False)
        self.btn_load_source.setEnabled(False)
        self.btn_load_tb.setEnabled(False)
        self.btn_save_file.setEnabled(False)
        self.btn_save_as.setEnabled(False)
        self.btn_syntax.setEnabled(False)
        self.btn_stub.setEnabled(False)
        self.btn_simulate.setEnabled(False)
        self.btn_send.setEnabled(False)
        self.btn_add_module.setEnabled(False)
        self.btn_auto_detect.setEnabled(False)
        self.hierarchy_list.setEnabled(False)
        
        self.btn_unlock.setVisible(True)
        
        msg = (
            "Icarus Verilog Dependency Missing\n"
            "---------------------------------\n"
            "The Verilog Simulator requires Icarus Verilog to compile and simulate code.\n\n"
            "Installation Options:\n"
            "- Windows: https://bleyer.org/icarus/\n"
            "  (IMPORTANT: You MUST check the 'Install MinGW Dependencies (DLL Files)' box during installation!)\n"
            "- Linux: sudo apt update && sudo apt install -y iverilog\n\n"
            "(Note: eSim does not endorse or vouch for the security of external links. \n"
            "Please verify and download at your own discretion.)\n\n"
            "Once installed, use the 'Locate Icarus Verilog' button below to enable the tool."
        )
        self.console.setPlainText(msg)
        self.console.setStyleSheet("QTextEdit { background-color: #f8d7da; color: #721c24; padding: 10px; border: 1px solid #f5c6cb; font-family: Consolas, monospace; font-size: 11pt; }")

    def unlock_ui(self):
        self.editor_tabs.setEnabled(True)
        self.btn_load_source.setEnabled(True)
        self.btn_load_tb.setEnabled(True)
        self.btn_save_file.setEnabled(True)
        self.btn_save_as.setEnabled(True)
        self.btn_syntax.setEnabled(True)
        self.btn_stub.setEnabled(True)
        self.btn_simulate.setEnabled(True)
        self.btn_send.setEnabled(True)
        self.btn_add_module.setEnabled(True)
        self.btn_auto_detect.setEnabled(True)
        self.hierarchy_list.setEnabled(True)
        
        self.btn_unlock.setVisible(False)
        
        self.console.clear()
        self.console.setStyleSheet(
            "QTextEdit { background-color: #fcfcfc; color: #495057; border: 1px solid #dee2e6; border-radius: 4px; padding: 10px; font-family: Consolas, monospace; font-size: 11pt; }"
        )
        self.log("System Unlocked. Icarus Verilog detected.")

    def init_ui(self):
        self.setStyleSheet("""
            QWidget {
                font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
            }
            QPushButton {
                background-color: #f8f9fa;
                border: 1px solid #ced4da;
                border-radius: 6px;
                padding: 6px 16px;
                color: #495057;
                font-weight: 600;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
                border-color: #adb5bd;
                color: #212529;
            }
            QPushButton:pressed {
                background-color: #dee2e6;
                border-color: #adb5bd;
            }
            QTabWidget::pane {
                border: 1px solid #dee2e6;
                background-color: #ffffff;
                border-radius: 4px;
                border-top-left-radius: 0px;
            }
            QTabBar::tab {
                background: #f8f9fa;
                border: 1px solid #dee2e6;
                border-bottom: none;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                color: #6c757d;
                font-weight: 500;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                color: #212529;
                font-weight: bold;
                border-top: 3px solid #007bff;
            }
            QTabBar::tab:hover:!selected {
                background: #e9ecef;
                color: #495057;
            }
            QListWidget {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                background-color: #ffffff;
                alternate-background-color: #f8f9fa;
                font-size: 13px;
            }
            QListWidget::item {
                border-radius: 4px;
                margin: 2px;
            }
            QListWidget::item:selected {
                background-color: #e7f1ff;
                color: #0c63e4;
                font-weight: bold;
            }
            QSplitter::handle {
                background-color: #e9ecef;
                margin: 2px;
                border-radius: 2px;
            }
            QSplitter::handle:hover {
                background-color: #ced4da;
            }
        """)
        
        main_layout = QtWidgets.QVBoxLayout(self)
        
        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        main_layout.addWidget(main_splitter)
        
        # 1. Top widget containing the tabbed editors
        top_container = QtWidgets.QWidget()
        top_layout = QtWidgets.QVBoxLayout(top_container)
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        def setup_popout(widget_to_pop, parent_container, insert_index=0, title="", extra_widgets=None):
            popout_btn = QtWidgets.QPushButton("🗗 Fullscreen")
            popout_btn.setStyleSheet("font-weight: bold; color: #444444; padding: 2px 6px; border: 1px solid transparent;")
            popout_btn.setFlat(True)
            if isinstance(widget_to_pop, QtWidgets.QTabWidget):
                if extra_widgets:
                    corner_widget = QtWidgets.QWidget()
                    layout = QtWidgets.QHBoxLayout(corner_widget)
                    layout.setContentsMargins(0, 0, 0, 0)
                    for w in extra_widgets:
                        layout.addWidget(w)
                    layout.addWidget(popout_btn)
                    widget_to_pop.setCornerWidget(corner_widget)
                else:
                    widget_to_pop.setCornerWidget(popout_btn)
                
            popout_state = {"win": None}
            
            def toggle_popout():
                if not popout_state["win"]:
                    win = QtWidgets.QDialog(self.window())
                    win.setWindowTitle(title)
                    win.setWindowFlags(win.windowFlags() | WIN_MAX | WIN_MIN)
                    layout = QtWidgets.QVBoxLayout(win)
                    layout.setContentsMargins(0, 0, 0, 0)
                    layout.addWidget(widget_to_pop)
                    popout_btn.setText("🡮 Dock to IDE")
                    
                    def on_close(event):
                        parent_container.insertWidget(insert_index, widget_to_pop)
                        popout_btn.setText("🗗 Fullscreen")
                        popout_state["win"] = None
                        event.accept()
                        
                    win.closeEvent = on_close
                    popout_state["win"] = win
                    win.resize(1000, 700)
                    win.showMaximized()
                else:
                    popout_state["win"].close()
                    
            popout_btn.clicked.connect(toggle_popout)

        top_h_splitter = QtWidgets.QSplitter(ORIENT_HORIZ)
        top_layout.addWidget(top_h_splitter)
        
        # Sidebar for module hierarchy
        sidebar_widget = QtWidgets.QWidget()
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        
        lbl_sidebar = QtWidgets.QLabel("Module Hierarchy")
        lbl_sidebar.setFont(QtGui.QFont("Segoe UI", 10, FONT_BOLD))
        sidebar_layout.addWidget(lbl_sidebar)
        
        self.btn_auto_detect = QtWidgets.QPushButton("Auto-Detect")
        self.btn_auto_detect.clicked.connect(self.auto_detect_hierarchy)
        sidebar_layout.addWidget(self.btn_auto_detect)
        
        self.hierarchy_list = QtWidgets.QListWidget()
        self.hierarchy_list.itemDoubleClicked.connect(self.hierarchy_double_clicked)
        self.hierarchy_list.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.hierarchy_list.customContextMenuRequested.connect(self.show_hierarchy_context_menu)
        sidebar_layout.addWidget(self.hierarchy_list)
        
        top_h_splitter.addWidget(sidebar_widget)

        self.editor_tabs = QtWidgets.QTabWidget()
        self.editor_tabs.setTabsClosable(True)
        self.editor_tabs.tabCloseRequested.connect(self.close_tab)
        self.editor_tabs.tabBarDoubleClicked.connect(self.rename_tab)
        # Track the last-active design tab so auto_generate_tb knows which module to target
        self._last_active_design_idx = None
        def _on_tab_changed(idx):
            w = self.editor_tabs.widget(idx)
            if w in getattr(self, 'design_views', []):
                self._last_active_design_idx = idx
        self.editor_tabs.currentChanged.connect(_on_tab_changed)
        
        # Add right-click context menu to the tab bar
        self.editor_tabs.tabBar().setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.editor_tabs.tabBar().customContextMenuRequested.connect(self.show_tab_context_menu)
        
        corner_widget = QtWidgets.QWidget()
        corner_layout = QtWidgets.QHBoxLayout(corner_widget)
        corner_layout.setContentsMargins(0, 0, 0, 0)
        
        self.btn_add_module = QtWidgets.QPushButton("➕ Add Module")
        self.btn_add_module.setFlat(True)
        self.btn_add_module.clicked.connect(self.add_module_tab)
        corner_layout.addWidget(self.btn_add_module)
        
        self.popout_btn = QtWidgets.QPushButton("🗗 Fullscreen")
        self.popout_btn.setStyleSheet("""
            QPushButton {
                font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
                font-weight: 600;
                color: #495057;
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 4px;
                padding: 4px 10px;
                margin: 2px 4px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
                border-color: #ced4da;
                color: #212529;
            }
        """)
        corner_layout.addWidget(self.popout_btn)
        
        self.editor_tabs.setCornerWidget(corner_widget)
        
        # Override setup_popout to use our existing button
        popout_state = {"win": None}
        def toggle_popout():
            if not popout_state["win"]:
                win = QtWidgets.QDialog(self.window())
                win.setWindowTitle("Verilog Code Editor")
                win.setWindowFlags(win.windowFlags() | WIN_MAX | WIN_MIN)
                layout = QtWidgets.QVBoxLayout(win)
                layout.setContentsMargins(0, 0, 0, 0)
                layout.addWidget(self.editor_tabs)
                self.popout_btn.setText("🡮 Dock to IDE")
                
                def on_close(event):
                    top_h_splitter.insertWidget(1, self.editor_tabs)
                    self.popout_btn.setText("🗗 Fullscreen")
                    popout_state["win"] = None
                    event.accept()
                    
                win.closeEvent = on_close
                popout_state["win"] = win
                win.resize(1000, 700)
                win.showMaximized()
            else:
                popout_state["win"].close()
        self.popout_btn.clicked.connect(toggle_popout)

        top_h_splitter.addWidget(self.editor_tabs)
        top_h_splitter.setSizes([200, 800])
        
        self.font = QtGui.QFont("Consolas", 11)
        
        # Design editor (Module 1)
        self.design_views = []
        self.add_module_tab("design.v", DEFAULT_DESIGN)
        
        # Testbench editor (always pinned to the end)
        self.tb_view = CodeEditor()
        self.tb_view.filepath = None
        self.tb_view.setFont(self.font)
        self.tb_view.setPlainText(DEFAULT_TB)
        self.tb_highlighter = VerilogHighlighter(self.tb_view.document())
        self.editor_tabs.addTab(self.tb_view, "Testbench (tb_design.v)")
        # Disable close button for tb_view
        self.editor_tabs.tabBar().setTabButton(self.editor_tabs.count()-1, TAB_RIGHT, None)
        
        # Dynamically update the testbench tab name
        self.tb_view.textChanged.connect(self.update_tb_tab_name)
        self.update_tb_tab_name()

        
        # Controls panel under editors
        controls_layout = QtWidgets.QHBoxLayout()
        top_layout.addLayout(controls_layout)
        
        self.btn_load_source = QtWidgets.QPushButton("Load Source .v")
        self.btn_load_source.clicked.connect(self.load_source_files)
        controls_layout.addWidget(self.btn_load_source)
        
        self.btn_load_tb = QtWidgets.QPushButton("Load TB .v")
        self.btn_load_tb.clicked.connect(self.load_tb_file)
        controls_layout.addWidget(self.btn_load_tb)
        
        self.btn_save_file = QtWidgets.QPushButton("Save")
        self.btn_save_file.clicked.connect(self.save_file)
        controls_layout.addWidget(self.btn_save_file)
        
        self.btn_save_as = QtWidgets.QPushButton("Save As")
        self.btn_save_as.clicked.connect(self.save_as_file)
        controls_layout.addWidget(self.btn_save_as)
        
        self.btn_syntax = QtWidgets.QPushButton("Check Syntax")
        self.btn_syntax.clicked.connect(self.check_syntax)
        controls_layout.addWidget(self.btn_syntax)
        
        self.btn_stub = QtWidgets.QPushButton("Auto-Generate Testbench")
        self.btn_stub.clicked.connect(self.auto_generate_tb)
        controls_layout.addWidget(self.btn_stub)
        
        self.btn_simulate = QtWidgets.QPushButton("Simulate")
        self.btn_simulate.clicked.connect(self.simulate_and_wave)
        controls_layout.addWidget(self.btn_simulate)

        # Shown only while a compile/sim runs on the worker thread; clicking it
        # kills the underlying iverilog/vvp process via the active CancelToken.
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.cancel_running_job)
        self.btn_cancel.setVisible(False)
        controls_layout.addWidget(self.btn_cancel)

        self.btn_export_csv = QtWidgets.QPushButton("Export CSV")
        self.btn_export_csv.clicked.connect(self.export_csv)
        self.btn_export_csv.setEnabled(False)
        controls_layout.addWidget(self.btn_export_csv)
        
        self.btn_send = QtWidgets.QPushButton("Send to Makerchip")
        self.btn_send.clicked.connect(self.send_to_makerchip)
        controls_layout.addWidget(self.btn_send)
        
        self.btn_unlock = QtWidgets.QPushButton("Locate Icarus Verilog...")
        self.btn_unlock.clicked.connect(self.attempt_manual_unlock)
        self.btn_unlock.setVisible(False)
        controls_layout.addWidget(self.btn_unlock)
        
        main_splitter.addWidget(top_container)
        
        # Find/Replace Toolbar
        self.find_widget = QtWidgets.QWidget()
        find_layout = QtWidgets.QHBoxLayout(self.find_widget)
        find_layout.setContentsMargins(0, 0, 0, 0)
        
        self.find_input = QtWidgets.QLineEdit()
        self.find_input.setPlaceholderText("Find...")
        self.replace_input = QtWidgets.QLineEdit()
        self.replace_input.setPlaceholderText("Replace with...")
        
        self.btn_find_next = QtWidgets.QPushButton("Find Next")
        self.btn_find_next.clicked.connect(self.find_next)
        
        self.btn_replace = QtWidgets.QPushButton("Replace")
        self.btn_replace.clicked.connect(self.replace_text)
        
        self.btn_close_find = QtWidgets.QPushButton("X")
        self.btn_close_find.setFixedWidth(30)
        self.btn_close_find.clicked.connect(self.find_widget.hide)
        
        find_layout.addWidget(self.find_input)
        find_layout.addWidget(self.replace_input)
        find_layout.addWidget(self.btn_find_next)
        find_layout.addWidget(self.btn_replace)
        find_layout.addWidget(self.btn_close_find)
        
        self.find_widget.hide()
        top_layout.addWidget(self.find_widget)
        
        shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+F"), self)
        shortcut.activated.connect(self.show_find_toolbar)
        
        # 2. Bottom widget containing console log and Waveform viewer
        bottom_container = QtWidgets.QWidget()
        bottom_layout = QtWidgets.QHBoxLayout(bottom_container)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        
        bottom_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        bottom_layout.addWidget(bottom_splitter)
        
        # Console output styled like Vivado TCL Console
        self.console_tabs = QtWidgets.QTabWidget()
        self.console = ConsoleEdit()
        self.console.setReadOnly(True)
        self.console.setPlaceholderText("Console logs will appear here...\nDouble-click a syntax error (e.g. design.v:5: error) to jump directly to the line.")
        self.console.setFont(QtGui.QFont("Consolas", 11))
        self.console.error_clicked.connect(self.jump_to_error)
        self.console.setStyleSheet("""
            QTextEdit {
                background-color: #0c101f;
                color: #39ff14;
                border: 1px solid #2d3548;
                padding: 8px;
            }
        """)
        self.console_tabs.addTab(self.console, "Console Output")
        bottom_splitter.addWidget(self.console_tabs)
        
        self.btn_copy_console = QtWidgets.QPushButton("📋 Copy")
        self.btn_copy_console.setFlat(True)
        self.btn_copy_console.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(self.console.toPlainText()))
        
        setup_popout(self.console_tabs, bottom_splitter, 0, "Verilog Console Output", extra_widgets=[self.btn_copy_console])
        
        # Inline Waveform viewer (native eSim plotWindow)
        self.wave_tabs = QtWidgets.QTabWidget()
        self.load_empty_waveform()
        bottom_splitter.addWidget(self.wave_tabs)
        setup_popout(self.wave_tabs, bottom_splitter, 1, "Waveform Viewer")
        
        main_splitter.addWidget(bottom_container)
        
        main_splitter.setSizes([450, 250])
        bottom_splitter.setSizes([400, 400])
        
        # Check for Icarus Verilog on boot and lock if missing
        self.check_iverilog_lock()

    def add_module_tab(self, name=None, content="", filepath=None):
        if not name:
            name = f"module_{len(self.design_views) + 1}.v"

        editor = CodeEditor()
        editor.filepath = filepath
        editor.setFont(self.font)
        editor.setPlainText(content)
        highlighter = VerilogHighlighter(editor.document())

        # Keep track of highlighters to prevent garbage collection
        if not hasattr(self, 'highlighters'):
            self.highlighters = []
        self.highlighters.append(highlighter)

        # Append BEFORE insertTab so _on_tab_changed sees it immediately
        self.design_views.append(editor)

        # Insert before testbench tab
        tb_index = self.editor_tabs.count() - 1 if self.editor_tabs.count() > 0 else 0
        self.editor_tabs.insertTab(tb_index, editor, name)
        self.editor_tabs.setCurrentWidget(editor)

        self.update_hierarchy_list()

    def close_tab(self, index):
        # Prevent closing Testbench
        if index == self.editor_tabs.count() - 1:
            return

        widget = self.editor_tabs.widget(index)
        if widget in self.design_views:
            self.design_views.remove(widget)
        self.editor_tabs.removeTab(index)
        # Reset last-active design index if the closed tab was the remembered one
        if getattr(self, '_last_active_design_idx', None) is not None:
            if self._last_active_design_idx == index:
                self._last_active_design_idx = None
            elif self._last_active_design_idx > index:
                self._last_active_design_idx -= 1
        self.update_hierarchy_list()
        
    def rename_tab(self, index):
        if index == self.editor_tabs.count() - 1:
            return # Don't rename testbench
        current_name = self.editor_tabs.tabText(index)
        new_name, ok = QtWidgets.QInputDialog.getText(self, "Rename Module", "Enter new module name:", text=current_name)
        if ok and new_name.strip():
            self.editor_tabs.setTabText(index, new_name.strip())
            self.update_hierarchy_list()

    def show_tab_context_menu(self, pos):
        index = self.editor_tabs.tabBar().tabAt(pos)
        if index >= 0 and index < self.editor_tabs.count() - 1: # Prevent operations on testbench
            menu = QtWidgets.QMenu(self)
            rename_action = menu.addAction("Rename Module")
            delete_action = menu.addAction("Delete Module")
            action = menu.exec(self.editor_tabs.tabBar().mapToGlobal(pos))
            if action == rename_action:
                self.rename_tab(index)
            elif action == delete_action:
                self.close_tab(index)

    def hierarchy_double_clicked(self, item):
        name = item.data(QtCore.Qt.ItemDataRole.UserRole)
        # Find index in editor_tabs
        for i in range(self.editor_tabs.count()):
            if self.editor_tabs.tabText(i) == name:
                self.rename_tab(i)
                break

    def show_hierarchy_context_menu(self, pos):
        item = self.hierarchy_list.itemAt(pos)
        if item:
            name = item.data(QtCore.Qt.ItemDataRole.UserRole)
            menu = QtWidgets.QMenu(self)
            rename_action = menu.addAction("Rename Module")
            delete_action = menu.addAction("Delete Module")
            action = menu.exec(self.hierarchy_list.mapToGlobal(pos))
            if action:
                # Find index in editor_tabs
                index = -1
                for i in range(self.editor_tabs.count()):
                    if self.editor_tabs.tabText(i) == name:
                        index = i
                        break
                if index != -1:
                    if action == rename_action:
                        self.rename_tab(index)
                    elif action == delete_action:
                        self.close_tab(index)

    def move_hierarchy_item(self, item, direction):
        row = self.hierarchy_list.row(item)
        names = [self.hierarchy_list.item(i).data(QtCore.Qt.ItemDataRole.UserRole) for i in range(self.hierarchy_list.count())]
        
        if direction == "up" and row > 0:
            names[row], names[row-1] = names[row-1], names[row]
            self.update_hierarchy_list(names)
            self.hierarchy_list.setCurrentRow(row - 1)
        elif direction == "down" and row < len(names) - 1:
            names[row], names[row+1] = names[row+1], names[row]
            self.update_hierarchy_list(names)
            self.hierarchy_list.setCurrentRow(row + 1)

    def update_hierarchy_list(self, sorted_names=None):
        self.hierarchy_list.clear()
        
        # If sorted_names provided, use that order. Otherwise, use tab order.
        if sorted_names:
            names = sorted_names
        else:
            names = []
            for i in range(self.editor_tabs.count()):
                # Only include tabs that are in design_views (excludes testbench)
                if hasattr(self, 'design_views') and self.editor_tabs.widget(i) in self.design_views:
                    names.append(self.editor_tabs.tabText(i))
                
        for name in names:
            item = QtWidgets.QListWidgetItem()
            item.setData(QtCore.Qt.ItemDataRole.UserRole, name)
            widget = QtWidgets.QWidget()
            layout = QtWidgets.QHBoxLayout(widget)
            layout.setContentsMargins(5, 2, 5, 2)
            
            lbl = QtWidgets.QLabel(name)
            
            # Override global QPushButton padding for these tiny buttons
            btn_style = "QPushButton { padding: 0px; font-weight: bold; font-size: 14px; color: #212529; }"
            
            btn_up = QtWidgets.QPushButton("▲")
            btn_up.setFixedSize(24, 24)
            btn_up.setStyleSheet(btn_style)
            btn_up.clicked.connect(lambda checked, i=item: self.move_hierarchy_item(i, "up"))
            
            btn_down = QtWidgets.QPushButton("▼")
            btn_down.setFixedSize(24, 24)
            btn_down.setStyleSheet(btn_style)
            btn_down.clicked.connect(lambda checked, i=item: self.move_hierarchy_item(i, "down"))
            
            layout.addWidget(lbl)
            layout.addStretch()
            layout.addWidget(btn_up)
            layout.addWidget(btn_down)
            
            # Force height to properly contain the 24px buttons
            widget.setMinimumHeight(32)
            item.setSizeHint(QtCore.QSize(0, 32))
            
            self.hierarchy_list.addItem(item)
            self.hierarchy_list.setItemWidget(item, widget)

    def auto_detect_hierarchy(self):
        import re
        
        modules = {}
        for i in range(self.editor_tabs.count()):
            if self.editor_tabs.widget(i) not in getattr(self, 'design_views', []):
                continue
            name = self.editor_tabs.tabText(i)
            editor = self.editor_tabs.widget(i)
            code = editor.toPlainText()
            
            # Find module definition name
            match = re.search(r'module\s+(\w+)\s*[\(#]', code)
            mod_name = match.group(1) if match else name
            
            modules[name] = {
                'mod_name': mod_name,
                'code': code,
                'dependencies': set()
            }
            
        # Find instantiations
        for name, data in modules.items():
            for other_name, other_data in modules.items():
                if name == other_name: continue
                # Very simple heuristic: if other module's name appears as a word in this code
                # and isn't the module definition itself
                if re.search(rf'\b{other_data["mod_name"]}\b\s+\w+', data['code']):
                    data['dependencies'].add(other_name)
                    
        # Topological sort
        sorted_names = []
        visited = set()
        
        def visit(n):
            if n in visited: return
            visited.add(n)
            for dep in modules[n]['dependencies']:
                visit(dep)
            sorted_names.insert(0, n)
            
        for name in modules.keys():
            visit(name)
            
        self.update_hierarchy_list(sorted_names)

    def log(self, text):
        self.console.append(text)
        self.console.moveCursor(QtGui.QTextCursor.MoveOperation.End)

    def load_empty_waveform(self):
        if hasattr(self, 'btn_export_csv'):
            self.btn_export_csv.setEnabled(False)
        self.wave_tabs.clear()
        placeholder = QtWidgets.QLabel("No Waveform Data Available")
        placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.wave_tabs.addTab(placeholder, "Waveform Viewer")

    def get_current_editor(self):
        return self.editor_tabs.currentWidget()
        
    def show_find_toolbar(self):
        self.find_widget.show()
        self.find_input.setFocus()
        self.find_input.selectAll()
        
    def find_next(self):
        text = self.find_input.text()
        if not text: return
        editor = self.get_current_editor()
        if not editor.find(text):
            # Wrap around
            cursor = editor.textCursor()
            cursor.movePosition(QtGui.QTextCursor.MoveOperation.Start)
            editor.setTextCursor(cursor)
            editor.find(text)
            
    def replace_text(self):
        text_to_find = self.find_input.text()
        replacement = self.replace_input.text()
        if not text_to_find: return
        
        editor = self.get_current_editor()
        cursor = editor.textCursor()
        if cursor.hasSelection() and cursor.selectedText() == text_to_find:
            cursor.insertText(replacement)
            self.find_next()
        else:
            self.find_next()

    def jump_to_error(self, line_num, filename):
        """Jump to the error line, searching all tabs by their exact label (basename match).

        Since simulate_and_wave now writes each module as its own named file,
        iverilog error output will contain the actual tab label (e.g. alu.v:3:).
        We do a clean two-step lookup: exact label match first, then canonical
        testbench alias, then give up without falling back to design_views[0].
        """
        editor = None
        tab_idx = -1
        basename = os.path.basename(filename)

        # Step 1: exact tab label match (handles any module filename)
        for i in range(self.editor_tabs.count()):
            label = self.editor_tabs.tabText(i)
            if label == basename:
                editor = self.editor_tabs.widget(i)
                tab_idx = i
                break

        # Step 2: canonical testbench aliases that iverilog may use
        if editor is None and basename in ('tb.v', 'tb_design.v'):
            editor = self.tb_view
            tab_idx = self.editor_tabs.indexOf(self.tb_view)

        # Do not fall back to design_views[0] — a wrong jump is worse than no jump
        if editor is None:
            return

        self.editor_tabs.setCurrentIndex(tab_idx)
            
        doc = editor.document()
        block = doc.findBlockByNumber(line_num - 1)
        if block.isValid():
            cursor = editor.textCursor()
            cursor.setPosition(block.position())
            editor.setTextCursor(cursor)
            editor.setFocus()
            
            # Apply persistent red background highlight
            selection = QtWidgets.QTextEdit.ExtraSelection()
            line_color = QtGui.QColor(255, 100, 100, 100) # Semi-transparent red
            selection.format.setBackground(line_color)
            selection.format.setProperty(QtGui.QTextFormat.Property.FullWidthSelection, True)
            selection.cursor = cursor
            selection.cursor.clearSelection()
            
            # Append to existing selections so we don't clear others
            current_selections = editor.extraSelections()
            current_selections.append(selection)
            editor.setExtraSelections(current_selections)

    def highlight_errors_from_log(self, log_text):
        """Highlight error lines in all tabs, matching by filename from the log."""
        # Clear all existing error highlights first
        for v in getattr(self, 'design_views', []):
            v.setExtraSelections([])
        self.tb_view.setExtraSelections([])

        # Build a map of tab_label -> editor for all design tabs + testbench
        tab_editors = {}
        for i in range(self.editor_tabs.count()):
            w = self.editor_tabs.widget(i)
            label = self.editor_tabs.tabText(i)
            tab_editors[label] = w
        # Also register canonical compile names used by iverilog for testbench.
        # Note: 'design.v' is intentionally NOT registered here — since simulate_and_wave
        # and check_syntax both write per-tab named files, no error should ever reference
        # the generic 'design.v'. If it appears, it is from old code paths and should not
        # silently redirect to design_views[0].
        tab_editors['tb.v'] = self.tb_view
        tab_editors['tb_design.v'] = self.tb_view

        # Accumulate per-editor selections
        editor_selections = {}

        matches = re.finditer(r'([\w./-]+\.(?:v|sv)):(\d+):', log_text)
        for match in matches:
            filename = match.group(1)
            line_num = int(match.group(2))
            # Try to find the editor; look by basename too
            editor = tab_editors.get(filename) or tab_editors.get(os.path.basename(filename))
            if editor is None:
                continue

            doc = editor.document()
            block = doc.findBlockByNumber(line_num - 1)
            if not block.isValid():
                continue

            cursor = editor.textCursor()
            cursor.setPosition(block.position())

            selection = QtWidgets.QTextEdit.ExtraSelection()
            fmt = QtGui.QTextCharFormat()
            fmt.setBackground(QtGui.QColor(255, 100, 100, 60))
            fmt.setUnderlineStyle(QtGui.QTextCharFormat.UnderlineStyle.SpellCheckUnderline)
            fmt.setUnderlineColor(QtGui.QColor("red"))
            fmt.setProperty(QtGui.QTextFormat.Property.FullWidthSelection, True)
            selection.format = fmt
            selection.cursor = cursor
            selection.cursor.clearSelection()
            editor_selections.setdefault(id(editor), (editor, []))[1].append(selection)

        for _, (editor, selections) in editor_selections.items():
            editor.setExtraSelections(selections)
        
        self.analyze_syntax_error(log_text)

    def analyze_syntax_error(self, log_text):
        hints = []
        log_lower = log_text.lower()
        if "syntax error" in log_lower:
            hints.append("💡 Hint (Syntax Error): You might be missing a semicolon (;), misspelled a keyword, or forgot an 'endmodule'.")
        if "unknown module type" in log_lower:
            hints.append("💡 Hint (Unknown Module): You instantiated a module that hasn't been defined. Check for typos or ensure all required modules are present.")
        if "is not a valid l-value" in log_lower:
            hints.append("💡 Hint (Invalid L-Value): You cannot assign to a 'wire' inside an 'always' block (use a 'reg' instead). Or you tried to use a 'reg' in an 'assign' statement.")
        if "undeclared identifier" in log_lower or "not declared" in log_lower:
            hints.append("💡 Hint (Undeclared Identifier): You forgot to declare a 'wire' or 'reg' for a signal, or there is a typo in its name.")
        if "unconnected" in log_lower and "port" in log_lower:
            hints.append("💡 Hint (Unconnected Port): A port on a module instance is left unconnected. This might be intentional, but double check your port mappings.")
        if "expecting" in log_lower and "endmodule" in log_lower:
            hints.append("💡 Hint (Expecting Endmodule): Your module structure is incomplete. Make sure every 'module' has a matching 'endmodule'.")
        if "multiple drivers" in log_lower:
            hints.append("💡 Hint (Multiple Drivers): You are trying to assign a value to the same signal from more than one place (like two 'always' blocks, or an 'always' block and an 'assign').")
            
        if hints:
            self.log("\n--- AI Syntax Analysis & Hints ---")
            for hint in hints:
                self.log(hint)


    def load_source_files(self):
        filepaths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Open Verilog Source Files", "", "Verilog Files (*.v *.sv);;All Files (*)")
        if filepaths:
            # If the only design tab is the default counter, automatically close it
            if len(self.design_views) == 1:
                first_editor = self.design_views[0]
                if first_editor.toPlainText().strip() == DEFAULT_DESIGN.strip():
                    self.close_tab(0)

            for filepath in filepaths:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        code = f.read()
                    name = os.path.basename(filepath)
                    self.add_module_tab(name, code, filepath=filepath)
                    self.log(f"Loaded source file: {name}")
                except Exception as e:
                    self.log(f"Failed to load {filepath}: {e}")

    def load_tb_file(self):
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Verilog Testbench File", "", "Verilog Files (*.v *.sv);;All Files (*)")
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    code = f.read()
                self.tb_view.setPlainText(code)
                self.tb_view.filepath = filepath
                self.editor_tabs.setCurrentIndex(self.editor_tabs.count() - 1)
                self.log(f"Loaded testbench: {os.path.basename(filepath)}")
            except Exception as e:
                self.log(f"Failed to load testbench: {e}")

    def update_tb_tab_name(self):
        import re
        text = self.tb_view.toPlainText()
        match = re.search(r'\bmodule\s+(\w+)', text)
        idx = self.editor_tabs.indexOf(self.tb_view)
        if match:
            mod_name = match.group(1)
            self.editor_tabs.setTabText(idx, f"Testbench ({mod_name}.v)")
        else:
            self.editor_tabs.setTabText(idx, "Testbench (tb_design.v)")

    def save_file(self):
        editor = self.get_current_editor()
        if hasattr(editor, 'filepath') and editor.filepath:
            try:
                with open(editor.filepath, 'w', encoding='utf-8') as f:
                    f.write(editor.toPlainText())
                self.log(f"Saved {os.path.basename(editor.filepath)}")
            except Exception as e:
                self.log(f"Failed to save file: {e}")
        else:
            self.save_as_file()

    def save_as_file(self):
        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Verilog File", "", "Verilog Files (*.v *.sv);;All Files (*)")
        if filepath:
            try:
                editor = self.get_current_editor()
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(editor.toPlainText())
                editor.filepath = filepath
                
                # Optionally rename the tab to match the new file name
                name = os.path.basename(filepath)
                idx = self.editor_tabs.indexOf(editor)
                if idx != -1 and idx != self.editor_tabs.count() - 1:
                    self.editor_tabs.setTabText(idx, name)
                    self.update_hierarchy_list()
                    
                self.log(f"Saved active tab to {name}.")
            except Exception as e:
                self.log(f"Failed to save file: {e}")

    def render_waveform(self, timestamps, signals_data, signal_types, popup=False):
        self.current_timestamps = timestamps
        self.current_signals_data = signals_data
        self.current_signal_types = signal_types
        
        if not timestamps or not signals_data:
            self.load_empty_waveform()
            return
            
        # Instantiate the native eSim plotWindow adapted for VCD
        self.plot_window = VcdPlotWindow(timestamps, signals_data, signal_types, "Verilog Simulation", self)
        
        self.wave_tabs.clear()
        self.wave_tabs.addTab(self.plot_window, "Waveform Viewer")
        
        self.btn_export_csv.setEnabled(True)
            
    def export_csv(self):
        if not hasattr(self, 'current_timestamps') or not self.current_timestamps:
            self.log("Error: No simulation data to export.")
            return
            
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Simulation Data as CSV",
            "simulation_results.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not filename:
            return
            
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                raw_signals_dict = getattr(self, 'current_raw_signals_data', self.current_signals_data)
                timescale = getattr(self, 'current_timescale', "Time")
                
                # Order signals: clk and reset first, then alphabetical
                all_signals = list(raw_signals_dict.keys())
                priority = []
                for s in ['clk', 'clock', 'reset', 'rst']:
                    if s in all_signals:
                        priority.append(s)
                
                other_signals = sorted([s for s in all_signals if s not in priority])
                signals = priority + other_signals
                
                # Normalize timescale string (strip extra spaces, e.g. '1 ns / 1 ps' -> '1ns/1ps')
                timescale_norm = re.sub(r'\s+', '', timescale)
                header = [f"Time ({timescale_norm})"] + signals
                f.write(','.join(header) + '\n')

                last_row_vals = None
                total_rows = len(self.current_timestamps)

                for i, t in enumerate(self.current_timestamps):
                    row_vals = []
                    for sig in signals:
                        row_vals.append(str(raw_signals_dict[sig][i]))

                    # Skip duplicate unchanged rows, but ALWAYS write the last row
                    is_last = (i == total_rows - 1)
                    if last_row_vals is not None and row_vals == last_row_vals and not is_last:
                        continue
                        
                    last_row_vals = row_vals
                    
                    row = [str(t)] + row_vals
                    f.write(','.join(row) + '\n')
                    
            self.log(f"Successfully exported CSV to: {filename}")
        except Exception as e:
            self.log(f"Error exporting CSV: {e}")

    def get_design_code(self):
        code_blocks = []
        
        # Fallback if hierarchy is empty (e.g. startup glitches)
        if self.hierarchy_list.count() == 0 and hasattr(self, 'design_views'):
            for editor in self.design_views:
                code_blocks.append(editor.toPlainText())
            return "\n".join(code_blocks)

        # Iterate through hierarchy list from top to bottom
        for i in range(self.hierarchy_list.count()):
            name = self.hierarchy_list.item(i).data(QtCore.Qt.ItemDataRole.UserRole)
            # Find the corresponding tab
            for j in range(self.editor_tabs.count()): 
                if self.editor_tabs.tabText(j) == name and self.editor_tabs.widget(j) in getattr(self, 'design_views', []):
                    editor = self.editor_tabs.widget(j)
                    code_blocks.append(f"// --- {name} ---\n{editor.toPlainText()}\n")
                    break
        return "\n".join(code_blocks)

    def get_tb_code(self):
        return self.tb_view.toPlainText()

    # Button Event Handlers
    def _design_sources(self):
        """Ordered (filename, code) for each design tab, named by its tab label
        so iverilog diagnostics reference the tab the user actually sees. The
        name always ends in .v/.sv so highlight_errors_from_log can match it."""
        sources = []
        for i in range(self.editor_tabs.count()):
            widget = self.editor_tabs.widget(i)
            if widget not in getattr(self, 'design_views', []):
                continue
            tab_name = self.editor_tabs.tabText(i)
            safe_name = tab_name if tab_name.endswith(('.v', '.sv')) \
                else tab_name + '.v'
            sources.append((safe_name, widget.toPlainText()))
        return sources

    # ------------------------------------------------------------------ #
    #  Async job plumbing (compile/sim run off the GUI thread)
    # ------------------------------------------------------------------ #
    def _job_running(self):
        return self._active_job is not None and self._active_job.isRunning()

    def _start_job(self, work, on_done, on_fail, cancel, busy):
        """Run ``work()`` on a worker thread; deliver its result to ``on_done``
        (or its error to ``on_fail``) back on the GUI thread. ``busy`` buttons
        are disabled and a Cancel button shown for the duration; ``cancel`` is
        the token ``work`` threads into the backend so Cancel can kill it."""
        self._active_cancel = cancel
        self._busy_buttons = busy
        for b in busy:
            b.setEnabled(False)
        self.btn_cancel.setVisible(True)
        self.btn_cancel.setEnabled(True)

        job = jobs.BackgroundJob(work, parent=self)
        self._active_job = job

        def finish():
            self._active_job = None
            self._active_cancel = None
            for b in busy:
                b.setEnabled(True)
            self.btn_cancel.setVisible(False)

        def ok(result):
            finish()
            on_done(result)

        def err(msg):
            finish()
            on_fail(msg)

        job.succeeded.connect(ok)
        job.failed.connect(err)
        job.finished.connect(job.deleteLater)
        job.start()

    def cancel_running_job(self):
        if self._active_cancel is not None:
            self.log("Cancelling...")
            self.btn_cancel.setEnabled(False)
            self._active_cancel.cancel()

    def closeEvent(self, event):
        # Don't let a worker thread outlive the widget.
        if self._job_running():
            if self._active_cancel is not None:
                self._active_cancel.cancel()
            self._active_job.wait(3000)
        super().closeEvent(event)

    def check_syntax(self):
        if self._job_running():
            return
        self.log("\n--- Checking Syntax ---")

        # Clear previous error highlights
        for v in self.design_views:
            v.setExtraSelections([])
        self.tb_view.setExtraSelections([])

        if not self.design_views:
            self.log("Error: No design modules found.")
            return

        iverilog = self.find_iverilog()
        if not iverilog:
            self.log("Error: 'iverilog' was not found. Please install it or specify the path.")
            return

        self.log(f"--- Using compiler at: {iverilog} ---")

        sources = self._design_sources()
        tb_code = self.get_tb_code()
        if tb_code and tb_code.strip():
            sources.append(("tb_design.v", tb_code))

        tmpdir = tempfile.mkdtemp()
        cancel = icarus.CancelToken()

        def work():
            return icarus.compile_design(
                iverilog, sources, tmpdir, std="-g2012", warnings=True,
                cancel=cancel)

        def done(res):
            try:
                if cancel.cancelled:
                    self.log("Syntax check cancelled.")
                elif res.ok:
                    self.log("Syntax OK (compiled successfully with iverilog)")
                else:
                    self.log("Syntax errors found:")
                    self.log(res.stderr)
                    self.highlight_errors_from_log(res.stderr)
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        def fail(msg):
            self.log(f"Syntax check error: {msg}")
            shutil.rmtree(tmpdir, ignore_errors=True)

        self._start_job(work, done, fail, cancel,
                        busy=[self.btn_syntax, self.btn_simulate])

    def auto_generate_tb(self):
        """Generate a testbench stub for the last-active design tab.
        
        If the user was viewing the testbench tab when they clicked the button,
        we remember which design tab was previously active (stored in
        _last_active_design_idx) so we target the correct module.
        """
        self.btn_stub.setEnabled(False)

        active_editor = self.get_current_editor()
        if active_editor == self.tb_view:
            # Use the remembered last-active design tab if available
            if hasattr(self, '_last_active_design_idx') and self._last_active_design_idx is not None:
                idx = self._last_active_design_idx
                widget = self.editor_tabs.widget(idx)
                if widget in getattr(self, 'design_views', []):
                    active_editor = widget
                else:
                    active_editor = self.design_views[0] if self.design_views else None
            elif hasattr(self, 'design_views') and self.design_views:
                active_editor = self.design_views[0]
            else:
                self.log("Error: No design modules found.")
                self.btn_stub.setEnabled(True)
                return

        if active_editor is None:
            self.log("Error: No design modules found.")
            self.btn_stub.setEnabled(True)
            return

        code = active_editor.toPlainText()
        if not code or not code.strip():
            self.log("Error: Current module editor is empty. Write your design first.")
            self.btn_stub.setEnabled(True)
            return

        module_name, ports = extract_ports(code)
        if not module_name:
            self.log("Error: Could not find any Verilog module declaration.")
            self.btn_stub.setEnabled(True)
            return

        tb_code = generate_stub_testbench(module_name, ports)
        self.tb_view.setPlainText(tb_code)
        # Jump to the testbench tab (always last)
        self.editor_tabs.setCurrentIndex(self.editor_tabs.count() - 1)
        self.log(f"Generated testbench stub for module '{module_name}'.")
        self.btn_stub.setEnabled(True)

    def simulate_and_wave(self):
        if self._job_running():
            return
        self.log("\n--- Starting Simulation ---")

        design_code = self.get_design_code()
        if not design_code or not design_code.strip():
            self.log("Error: Design editor is empty.")
            return

        tb_code = self.get_tb_code()
        if not tb_code or not tb_code.strip():
            self.log("Error: Testbench editor is empty. Use Auto-Generate first.")
            return

        iverilog = self.find_iverilog()
        vvp = self.find_vvp()
        if not iverilog or not vvp:
            self.log("Error: 'iverilog' or 'vvp' binaries were not found on the system.")
            return

        # Each design tab is written under its own tab-label filename (via
        # _design_sources) so iverilog error output references the tab the user
        # sees and highlight_errors_from_log can find it.
        sources = self._design_sources() + [("tb_design.v", tb_code)]
        self.log(f"Compiling {len(sources) - 1} module(s) + testbench...")

        tmpdir = tempfile.mkdtemp()
        cancel = icarus.CancelToken()
        libdir = CosimConfig.iverilog_libdir()

        def work():
            return icarus.build_and_simulate(
                iverilog, vvp, sources, tmpdir, libdir=libdir, cancel=cancel)

        def done(run):
            try:
                self._render_sim_result(run, cancelled=cancel.cancelled)
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        def fail(msg):
            self.log(f"Simulation error: {msg}")
            shutil.rmtree(tmpdir, ignore_errors=True)

        self._start_job(work, done, fail, cancel,
                        busy=[self.btn_simulate, self.btn_syntax])

    def _render_sim_result(self, run, cancelled=False):
        """Log/highlight/plot the outcome of build_and_simulate, on the GUI
        thread. Mirrors the old inline simulate_and_wave reporting exactly."""
        if cancelled:
            self.log("Simulation cancelled.")
            return

        if not run.compile.ok:
            self.log("Compilation Failed:")
            self.log(run.compile.stderr)
            self.highlight_errors_from_log(run.compile.stderr)
            return

        sim = run.sim
        self.log("Simulation console output:")
        if sim.stdout:
            self.log(sim.stdout)
        if sim.stderr:
            self.log(sim.stderr)

        if not sim.ok:
            self.log(f"Error: vvp crashed or failed with exit code {sim.returncode}")
            if sim.returncode in [3221225781, -1073741515]:
                self.log("IT LOOKS LIKE A MISSING DLL ERROR (0xC0000135).")
                self.log("Please try to reinstall Icarus Verilog and ensure the "
                         "'Install MinGW Dependencies (DLL Files)' option is CHECKED.")

        if run.vcd_content:
            self.log("Parsing VCD file...")
            try:
                timestamps, signals_data, signal_types, raw_signals, timescale = parse_vcd_for_plot(run.vcd_content)
                self.current_raw_signals_data = raw_signals
                self.current_timescale = timescale
                self.render_waveform(timestamps, signals_data, signal_types, popup=True)
                self.log("Simulation completed and waveform rendered successfully.")
            except Exception as ex:
                self.log(f"Error parsing VCD file: {ex}")
        else:
            self.log("Error: No VCD file produced. Make sure your testbench contains "
                     '$dumpfile("sim_out.vcd") and $dumpvars(0, ...).')

    def send_to_makerchip(self):
        self.btn_send.setEnabled(False)
        
        self.log("Auto-detecting hierarchy before sending to Makerchip...")
        self.auto_detect_hierarchy()
        
        code = self.get_design_code()
        
        if not code or not code.strip():
            self.log("Error: Design editor is empty.")
            self.btn_send.setEnabled(True)
            return
            
        module_name, _ = extract_ports(code)
        if not module_name:
            self.log("Error: Could not extract module name from design.")
            self.btn_send.setEnabled(True)
            return
            
        dest_dir = os.path.join(os.path.expanduser("~"), "eSim-Workspace", "verified_verilog")
        os.makedirs(dest_dir, exist_ok=True)
        
        filepath = os.path.join(dest_dir, f"{module_name}.v")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code)
            
        self.log(f"Saved verified file to: {filepath}")
        self.sendToNgVeri.emit(filepath)
        self.btn_send.setEnabled(True)
