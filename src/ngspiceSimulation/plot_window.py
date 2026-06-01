# ngspiceSimulation/plot_window.py
"""
Plot Window Module

This module provides the main plotting window for NGSpice simulation results
with support for AC, DC, and Transient analysis visualization.
"""

from __future__ import division
import os
import re
import sys
import json
import traceback
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from PyQt6 import QtGui, QtCore, QtWidgets
from PyQt6.QtCore import Qt, QSettings, pyqtSignal
from PyQt6.QtWidgets import (QWidget, QVBoxLayout,
                             QHBoxLayout, QListWidget, QListWidgetItem, QPushButton,
                             QCheckBox, QGroupBox, QRadioButton, QButtonGroup,
                             QLabel, QLineEdit, QSlider, QDoubleSpinBox, QMenu,
                             QFileDialog, QColorDialog, QInputDialog,
                             QMessageBox, QStatusBar,
                             QSplitter, QToolButton, QWidgetAction, QGridLayout,
                             QSizePolicy, QScrollArea)
from PyQt6.QtGui import (QColor, QBrush, QPalette, QKeySequence, QShortcut,
                         QPainter, QPixmap, QFont, QAction, QIcon, QPen)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backend_bases import NavigationToolbar2
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, ScalarFormatter

from configuration.Appconfig import Appconfig
from .plotting_widgets import CollapsibleBox
from .data_extraction import DataExtraction

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_WIDTH = 1400
DEFAULT_WINDOW_HEIGHT = 800
DEFAULT_DPI = 100
DEFAULT_FIGURE_SIZE = (10, 8)
DEFAULT_LINE_THICKNESS = 1.5
DEFAULT_VERTICAL_SPACING = 1.2
DEFAULT_ZOOM_FACTOR = 0.9
CURSOR_ALPHA = 0.7
THRESHOLD_ALPHA = 0.5
LEGEND_FONT_SIZE = 9
DEFAULT_EXPORT_DPI = 300

# pane scrolls vertically once MIN_STACKED_PANE_HEIGHT_PX * N exceeds viewport
MIN_STACKED_PANE_HEIGHT_PX = 120
DIVIDER_HIT_TOLERANCE_PX = 6

# stacked view uses a wider debounce to coalesce a rapid-toggle burst into one rebuild
REFRESH_DEBOUNCE_MS = 80
STACKED_REFRESH_DEBOUNCE_MS = 160

VIBRANT_COLOR_PALETTE = [
    '#E53935',
    '#1E88E5',
    '#43A047',
    '#FB8C00',
    '#8E24AA',
    '#00ACC1',
    '#D81B60',
    '#6D4C41',
    '#FDD835',
    '#039BE5',
    '#C0CA33',
    '#37474F'
]

TIME_UNIT_THRESHOLD_PS = 1e-9
TIME_UNIT_THRESHOLD_NS = 1e-6
TIME_UNIT_THRESHOLD_US = 1e-3
TIME_UNIT_THRESHOLD_MS = 1

FREQ_UNIT_THRESHOLD_KHZ = 1e3
FREQ_UNIT_THRESHOLD_MHZ = 1e6
FREQ_UNIT_THRESHOLD_GHZ = 1e9

LINE_STYLES = [
    ('-', "Solid"),
    ('--', "Dashed"),
    (':', "Dotted"),
    ('steps-post', "Step (Post)")
]

THICKNESS_OPTIONS = [
    (1.0, "1 px"),
    (1.5, "1.5 px"),
    (2.0, "2 px"),
    (3.0, "3 px")
]


class Trace:
    """Single class to manage all trace properties."""

    def __init__(self, index: int, name: str, color: str = None,
                 thickness: float = DEFAULT_LINE_THICKNESS, style: str = '-',
                 visible: bool = False) -> None:
        self.index = index
        self.name = name
        self.color = color or VIBRANT_COLOR_PALETTE[0]
        self.thickness = thickness
        self.style = style
        self.visible = visible
        self.line_object: Optional[Line2D] = None

    def update_line(self, **kwargs) -> None:
        if self.line_object:
            if 'color' in kwargs:
                self.color = kwargs['color']
                self.line_object.set_color(self.color)
            if 'thickness' in kwargs:
                self.thickness = kwargs['thickness']
                self.line_object.set_linewidth(self.thickness)
            if 'style' in kwargs:
                self.style = kwargs['style']
                if self.style != 'steps-post':
                    self.line_object.set_linestyle(self.style)


class CustomListWidget(QListWidget):
    """Plain multi-select list. Drag-source removed in v3.1 along with the
    multi-signal-pane model — stacked view now keeps one trace per pane,
    so there's nothing to drag onto. Pane reorder is done via Move Up/Down
    in the right-click menu or by Alt-dragging a pane vertically.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setSelectionMode(QListWidget.SelectionMode.MultiSelection)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)


def _safe_eval(expr: str, data_map: dict) -> "np.ndarray":
    """Evaluate a sanitized expression over pre-resolved trace arrays.

    Caller must run _resolve_expr() first — trace names are replaced with
    _trace_N_ identifiers before this function sees the expression, so the
    AST parser never encounters NGSpice names like v(net1) or i(r1).

    Allowed: trace identifiers (keys of data_map), numeric literals,
    +  -  *  /  ** (unary - and +), abs sqrt log log10 exp sin cos tan.
    Arrays of mismatched length are trimmed to the shorter one before ops.
    Raises ValueError for unknown names, disallowed constructs, or syntax errors.
    """
    import ast
    import operator as op

    _NUMPY_FNS: dict = {
        'abs': np.abs, 'sqrt': np.sqrt,
        'log': np.log, 'log10': np.log10, 'exp': np.exp,
        'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
    }
    _BINOPS: dict = {
        ast.Add: op.add, ast.Sub: op.sub,
        ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Pow: op.pow,
    }
    _UNOPS: dict = {
        ast.USub: op.neg,
        ast.UAdd: lambda x: x,
    }

    def _align(a, b):
        if hasattr(a, '__len__') and hasattr(b, '__len__') and len(a) != len(b):
            n = min(len(a), len(b))
            return a[:n], b[:n]
        return a, b

    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return np.float64(node.value)
        if isinstance(node, ast.Name):
            if node.id in data_map:
                return data_map[node.id]
            raise ValueError(f"Unknown identifier '{node.id}' — "
                             "trace names are substituted before evaluation, "
                             "so this indicates a typo or unsupported construct.")
        if isinstance(node, ast.BinOp):
            fn = _BINOPS.get(type(node.op))
            if fn is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            l, r = _align(_eval(node.left), _eval(node.right))
            return fn(l, r)
        if isinstance(node, ast.UnaryOp):
            fn = _UNOPS.get(type(node.op))
            if fn is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
            return fn(_eval(node.operand))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError(
                    "Attribute calls (e.g. np.sin) are not allowed. "
                    "Use the bare function name instead: sin(…)")
            fn = _NUMPY_FNS.get(node.func.id)
            if fn is None:
                raise ValueError(
                    f"Unknown function '{node.func.id}'. "
                    f"Allowed: {', '.join(sorted(_NUMPY_FNS))}")
            if node.keywords:
                raise ValueError("Keyword arguments are not allowed in function calls.")
            args = [_eval(a) for a in node.args]
            return fn(*args)
        raise ValueError(
            f"Unsupported expression node '{type(node).__name__}'. "
            "Only arithmetic operators and whitelisted functions are allowed.")

    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError as exc:
        raise ValueError(f"Syntax error: {exc.msg}") from exc
    return np.asarray(_eval(tree.body), dtype=float)


def _canonical_expr(sanitized: str) -> str:
    """Return canonical form of a sanitized expression for duplicate detection.

    Commutative operators (+, *) flatten their operand chains and sort them,
    so 'a+b+c' and 'c+a+b' produce the same string.  Non-commutative ops
    (-, /, **) keep original order, so 'a-b' and 'b-a' remain distinct.
    Falls back to whitespace-stripped input on parse error.
    """
    import ast

    def _collect(node, op_type: type, results: list) -> None:
        if isinstance(node, ast.BinOp) and isinstance(node.op, op_type):
            _collect(node.left, op_type, results)
            _collect(node.right, op_type, results)
        else:
            results.append(_norm(node))

    def _norm(node) -> str:
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, (ast.Add, ast.Mult)):
                sym = '+' if isinstance(node.op, ast.Add) else '*'
                parts: list = []
                _collect(node, type(node.op), parts)
                parts.sort()
                result = parts[0]
                for p in parts[1:]:
                    result = f'({result}{sym}{p})'
                return result
            left, right = _norm(node.left), _norm(node.right)
            sym = {ast.Sub: '-', ast.Div: '/', ast.Pow: '**'}.get(type(node.op), '?')
            return f'({left}{sym}{right})'
        if isinstance(node, ast.UnaryOp):
            sym = '-' if isinstance(node.op, ast.USub) else '+'
            return f'({sym}{_norm(node.operand)})'
        if isinstance(node, ast.Call):
            func = node.func.id if isinstance(node.func, ast.Name) else repr(node.func)
            return f'{func}({",".join(_norm(a) for a in node.args)})'
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant):
            return str(node.value)
        return repr(node)

    try:
        return _norm(ast.parse(sanitized, mode='eval').body)
    except Exception:
        return sanitized.replace(' ', '')


def _format_measurement(value: float, unit: str) -> str:
    """Format a voltage or current with SI prefix. Covers pA/nV to handle
    very small signals without silently returning '0'."""
    abs_val = abs(value)
    if unit == "A":
        if abs_val >= 1:       return f"{value:.3g} A"
        if abs_val >= 1e-3:    return f"{value * 1e3:.3g} mA"
        if abs_val >= 1e-6:    return f"{value * 1e6:.3g} µA"
        if abs_val >= 1e-9:    return f"{value * 1e9:.3g} nA"
        if abs_val >= 1e-12:   return f"{value * 1e12:.3g} pA"
        return f"{value:.3g} A"
    else:
        if abs_val >= 1:       return f"{value:.3g} V"
        if abs_val >= 1e-3:    return f"{value * 1e3:.3g} mV"
        if abs_val >= 1e-6:    return f"{value * 1e6:.3g} µV"
        if abs_val >= 1e-9:    return f"{value * 1e9:.3g} nV"
        if abs_val >= 1e-12:   return f"{value * 1e12:.3g} pV"
        return f"{value:.3g} V"


def _format_frequency(freq_hz: float) -> str:
    """Format a frequency in Hz with an appropriate SI prefix."""
    if freq_hz >= 1e9:   return f"{freq_hz / 1e9:.3g} GHz"
    if freq_hz >= 1e6:   return f"{freq_hz / 1e6:.3g} MHz"
    if freq_hz >= 1e3:   return f"{freq_hz / 1e3:.3g} kHz"
    return                      f"{freq_hz:.3g} Hz"


# numpy 2.0 renamed trapz → trapezoid
_trapz = getattr(np, 'trapezoid', None) or np.trapz


def _detect_frequency(time_data: "np.ndarray",
                      logic_normalized: "np.ndarray") -> "Optional[float]":
    """Return signal frequency in Hz if periodic, else None.

    Uses rising-edge timing with linear interpolation for sub-sample accuracy.
    Requires ≥2 complete cycles and CV < 10% to reject non-periodic signals.
    """
    transitions = np.diff(logic_normalized.astype(np.int8))
    rising_idx = np.where(transitions == 1)[0]
    if len(rising_idx) < 3:
        return None
    # Interpolate crossing time: edge is between sample i and i+1, midpoint
    # gives sub-sample accuracy for non-uniform (adaptive-step) time grids.
    edge_times = (time_data[rising_idx] + time_data[rising_idx + 1]) / 2.0
    periods = np.diff(edge_times)
    if len(periods) == 0:
        return None
    mean_p = float(np.mean(periods))
    if mean_p <= 0:
        return None
    if len(periods) > 1 and float(np.std(periods)) / mean_p > 0.10:
        return None
    return 1.0 / mean_p


class plotWindow(QWidget):
    """Main plotting widget for NGSpice simulation results."""

    def __init__(self, file_path: str, project_name: str, parent=None) -> None:
        super().__init__(parent)

        self.file_path = file_path
        self.project_name = project_name
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.setMinimumSize(400, 300)
        self.obj_appconfig = Appconfig()
        logger.info(f"Complete Project Path: {self.file_path}")
        logger.info(f"Project Name: {self.project_name}")
        self.obj_appconfig.print_info(f'NGSpice simulation called: {self.file_path}')
        self.obj_appconfig.print_info(f'PythonPlotting called: {self.file_path}')

        self._initialize_data_structures()
        self._initialize_configuration()
        self.create_main_frame()
        self.load_simulation_data()
        self.apply_theme()
        self._setup_matplotlib_style()

    def _initialize_data_structures(self) -> None:
        self._em_cache: Optional[int] = None  # invalidated by changeEvent on FontChange
        self._resize_timer: QtCore.QTimer = QtCore.QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(120)
        self._resize_timer.timeout.connect(self._do_deferred_resize)
        self._controls_timer: QtCore.QTimer = QtCore.QTimer(self)
        self._controls_timer.setSingleShot(True)
        self._controls_timer.setInterval(150)
        self._controls_timer.timeout.connect(self.refresh_plot)
        # debounce: coalesces rapid toggles so stacked rebuild runs once after burst settles
        self._refresh_timer: QtCore.QTimer = QtCore.QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(REFRESH_DEBOUNCE_MS)
        self._refresh_timer.timeout.connect(self.refresh_plot)
        self.traces: Dict[int, Trace] = {}
        # cursor_lines[i]: axvlines per pane; empty inner list = cursor not yet rendered
        self.cursor_lines: List[List[Optional[Line2D]]] = []
        self.cursor_positions: List[Optional[float]] = []
        self._current_analysis_type: str = ''
        self.timing_annotations: Dict[int, Any] = {}
        self.color_palette = VIBRANT_COLOR_PALETTE.copy()
        self.logic_thresholds: Dict[int, float] = {}
        self.vertical_spacing = DEFAULT_VERTICAL_SPACING
        self._func_line: Optional[Line2D] = None
        self._drag_cursor_idx: Optional[int] = None
        self._current_view_mode: str = 'normal'  # 'normal' | 'timing' | 'stacked'
        self.panes: List[Any] = []
        # incremental-refresh: skip full rebuild when composition unchanged; _force_full_refresh overrides
        self._drawn_signature: Optional[tuple] = None
        self._force_full_refresh: bool = False
        # layout freeze: stacked rebuild sets _pending_freeze; draw callback snapshots geometry and drops solver
        self._pending_freeze: bool = False
        # display-only scale: line data stays in raw SI; ticks formatted as raw * _x_scale
        self._x_scale: float = 1.0
        self._x_unit: str = 's'
        # view state: one-shot ylim snapshots + persistent locks, both keyed by anchor trace name
        self._saved_xlim: Optional[Tuple[float, float]] = None
        self._saved_pane_ylims: Dict[str, Tuple[float, float]] = {}   # one-shot
        self._locked_ylims: Dict[str, Tuple[float, float]] = {}        # persistent
        # pane groups: outer = pane order, inner = trace indices; empty = one trace per pane
        self._pane_groups: List[List[int]] = []
        # pane_lock_y keyed by anchor trace name; actual ylim in _locked_ylims
        self._pane_lock_y: Dict[str, bool] = {}
        self._global_stats_visible: bool = True
        # func traces: (label, x, y, color, thickness, style)
        self._func_traces: List[Tuple[str, "np.ndarray", "np.ndarray", str, float, str]] = []
        # canonical expr per func trace for O(1) dup-check
        self._func_canonical: List[str] = []
        # visibility parallel to _func_traces
        self._func_visible: List[bool] = []
        # sorted longest-first so longer names match before their substrings
        self._nb_sorted: List[Tuple[int, str]] = []
        # pending layout from config, resolved after populate_waveform_list sets up NBList
        self._pending_layout: Optional[Dict[str, Any]] = None
        # pane height ratios; empty = equal heights
        self._pane_heights: List[float] = []
        # transient drag state for divider resize and pane reorder
        self._divider_drag: Optional[Dict[str, Any]] = None
        self._pane_drag: Optional[Dict[str, Any]] = None
        # Mouse-move dedup: skip setText/anchor lookup when state unchanged.
        self._last_hover_axes: Any = None
        self._last_hover_anchor: Optional[str] = None
        self._last_coord_text: str = ''
        self._last_cursor_shape_was_resize: bool = False
        self._blit_background: Optional[Any] = None
        # interp cache per cursor; invalidated by new data or x_pos/visible change
        self._cursor_interp_cache: List[Optional[Dict]] = []
        # .tran start offset parsed once; 0.0 if not a tran sim
        self._tran_start_time: float = 0.0

    def _initialize_configuration(self) -> None:
        self.config_dir = Path.home() / '.pythonPlotting'
        self.config_file = self.config_dir / 'config.json'
        self.config: Dict[str, Any] = self.load_config()
        self.settings = QSettings('eSim', 'PythonPlotting')

    def load_config(self) -> Dict[str, Any]:
        try:
            self.config_dir.mkdir(exist_ok=True)
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as config_file:
                    config = json.load(config_file)
                    if 'theme' in config:
                        del config['theme']
                    return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
        return {'trace_colours': {}, 'trace_thickness': {}, 'trace_style': {}, 'experimental_acdc': False}

    def save_config(self) -> None:
        try:
            self.config_dir.mkdir(exist_ok=True)
            self.config['trace_colours'] = {t.name: t.color for t in self.traces.values()}
            self.config['trace_thickness'] = {t.name: t.thickness for t in self.traces.values()}
            self.config['trace_style'] = {t.name: t.style for t in self.traces.values()}
            # Stacked-view layout, keyed by trace NAME so it survives schematic
            # changes that renumber NBList. Lists of names per pane preserve
            # both pane order and intra-pane signal order.
            self.config['stacked_pane_groups'] = [
                [self.traces[i].name for i in g
                 if i in self.traces]
                for g in self._pane_groups
            ]
            self.config['stacked_lock_y'] = dict(self._pane_lock_y)
            self.config['stacked_locked_ylims'] = {
                name: list(lims) for name, lims in self._locked_ylims.items()
            }
            self.config['stacked_stats_visible'] = self.stats_check.isChecked()
            # Persist per-pane height ratios alongside their anchor name so a
            # schematic edit that drops a signal also drops its custom height.
            self.config['stacked_pane_heights'] = {
                self.traces[g[0]].name: float(self._pane_heights[i])
                for i, g in enumerate(self._pane_groups)
                if g and g[0] in self.traces and i < len(self._pane_heights)
            }
            temp_file = self.config_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as config_file:
                json.dump(self.config, config_file, indent=2)
            temp_file.replace(self.config_file)
        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.save_config()
        # Cancel deferred timers so no queued tick fires refresh_plot after the
        # figure/canvas below are torn down (would touch a closed figure).
        self._refresh_timer.stop()
        self._controls_timer.stop()
        self._resize_timer.stop()
        if hasattr(self, 'canvas'):
            self.canvas.close()
        if hasattr(self, 'fig'):
            plt.close(self.fig)
        super().closeEvent(event)

    def apply_theme(self) -> None:
        em      = self._em
        sb_w    = max(6,  em // 2)
        ind     = max(12, em - 2)
        sldr    = max(10, em - 4)
        sldr_m  = -(sldr // 2)
        item_h  = max(28, em + 12)
        item_pv = max(4,  em // 4)
        item_ph = max(6,  em // 2)
        btn_pv  = max(3,  em // 4)
        btn_ph  = max(6,  em // 2)
        btn_h   = max(24, em + 8)
        le_p    = max(4,  em // 3)

        theme_stylesheet = f"""
        QMenuBar {{ border-radius: 8px; background-color: #FFFFFF; border: 1px solid #E0E0E0; padding: 2px; }}
        QStatusBar {{ border-radius: 8px; background-color: #FFFFFF; border: 1px solid #E0E0E0; padding: 2px; }}
        QWidget {{ background-color: #FFFFFF; color: #212121; }}
        QListWidget {{ background-color: #FFFFFF; border: 1px solid #E0E0E0; padding: 2px; outline: none; selection-background-color: transparent; selection-color: inherit; }}
        QListWidget::item {{ min-height: {item_h}px; padding: {item_pv}px {item_ph}px; margin: 1px 2px; background-color: transparent; border: none; }}
        QListWidget::item:selected {{ background-color: transparent; border: none; }}
        QListWidget::item:hover {{ background-color: rgba(0, 0, 0, 0.04); }}
        QListWidget::item:focus {{ outline: none; }}
        QGroupBox {{ border: 1px solid #E0E0E0; margin-top: 0.5em; padding-top: 0.5em; }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; }}
        QPushButton {{ background-color: #FFFFFF; border: 1px solid #E0E0E0; padding: {btn_pv}px {btn_ph}px; min-height: {btn_h}px; font-weight: 500; }}
        QPushButton:hover {{ background-color: #F2F2F2; border-color: #1976D2; }}
        QPushButton:pressed {{ background-color: #E0E0E0; }}
        QCheckBox::indicator {{ width: {ind}px; height: {ind}px; }}
        QMenu {{ background-color: #FFFFFF; border: 1px solid #E0E0E0; }}
        QMenu::item:selected {{ background-color: #E3F2FD; }}
        QLineEdit {{ border: 1px solid #E0E0E0; padding: {le_p}px {btn_ph}px; background-color: #FAFAFA; }}
        QLineEdit:focus {{ border-color: #1976D2; background-color: #FFFFFF; }}
        QSlider::groove:horizontal {{ border: 1px solid #E0E0E0; height: 4px; background: #E0E0E0; }}
        QSlider::handle:horizontal {{ background: #1976D2; border: 1px solid #1976D2; width: {sldr}px; height: {sldr}px; margin: {sldr_m}px 0; }}
        QScrollBar:vertical {{ background-color: #F5F5F5; width: {sb_w}px; border: none; border-radius: {sb_w // 2}px; }}
        QScrollBar::handle:vertical {{ background-color: #BDBDBD; border-radius: {sb_w // 2}px; min-height: 20px; margin: 2px; }}
        QScrollBar::handle:vertical:hover {{ background-color: #9E9E9E; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
        QSplitter::handle:horizontal {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0.49 transparent, stop:0.5 #D0D0D0, stop:0.51 transparent); }}
        QSplitter::handle:horizontal:hover {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0.45 transparent, stop:0.5 #1976D2, stop:0.55 transparent); }}
        """
        self.setStyleSheet(theme_stylesheet)

    def create_main_frame(self) -> None:
        main_widget_layout = QVBoxLayout(self)
        main_widget_layout.setContentsMargins(5, 5, 5, 5)
        self.menu_bar = QtWidgets.QMenuBar(self)
        main_widget_layout.addWidget(self.menu_bar)
        content_widget = QWidget()
        content_widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        main_layout = QHBoxLayout(content_widget)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.splitter.setHandleWidth(5)
        em = self._em
        self.left_panel = self.create_waveform_list()
        self.left_panel.setMinimumWidth(em * 10)
        self.splitter.addWidget(self.left_panel)
        self.center_widget = self.create_plot_area()
        self.center_widget.setMinimumWidth(em * 18)
        self.splitter.addWidget(self.center_widget)
        right_widget = self.create_control_panel()
        self.right_panel = QScrollArea()
        self.right_panel.setWidget(right_widget)
        self.right_panel.setWidgetResizable(True)
        self.right_panel.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.right_panel.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.right_panel.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.right_panel.setMinimumWidth(em * 9)
        self.splitter.addWidget(self.right_panel)
        self.splitter.setStretchFactor(0, 20)
        self.splitter.setStretchFactor(1, 63)
        self.splitter.setStretchFactor(2, 17)
        main_layout.addWidget(self.splitter)
        main_widget_layout.addWidget(content_widget)
        self.status_bar = QStatusBar()
        self.coord_label = QLabel("X: --, Y: --")
        self.status_bar.addWidget(self.coord_label)
        self.measure_label = QLabel("")
        self.status_bar.addPermanentWidget(self.measure_label)
        main_widget_layout.addWidget(self.status_bar)
        self.create_menu_bar()
        self.setWindowTitle(f'Python Plotting - {self.project_name}')

    def create_waveform_list(self) -> QWidget:
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        em = self._em
        self.analysis_label = QLabel()
        self.analysis_label.setStyleSheet(
            f"font-weight: bold; font-size: {max(11, em - 4)}px; padding: {max(3, em // 5)}px;"
        )
        left_layout.addWidget(self.analysis_label)
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search waveforms...")
        self.search_box.textChanged.connect(self.filter_waveforms)
        left_layout.addWidget(self.search_box)
        self.waveform_list = CustomListWidget()
        self.waveform_list.itemClicked.connect(self.on_waveform_toggle)
        self.waveform_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.waveform_list.customContextMenuRequested.connect(self.show_list_context_menu)
        self.waveform_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.waveform_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_layout.addWidget(self.waveform_list)
        QShortcut(QKeySequence.StandardKey.SelectAll, self.waveform_list,
                  activated=self.select_all_waveforms)
        return left_widget

    def create_plot_area(self) -> QWidget:
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        # constrained_layout handles multi-pane spacing automatically (no manual
        # tight_layout calls needed). Required for the stacked-view feature
        # where N subplots share an X axis and hspace must stay consistent.
        self.fig = Figure(figsize=DEFAULT_FIGURE_SIZE, dpi=DEFAULT_DPI,
                          constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.nav_toolbar = NavigationToolbar(self.canvas, self)
        for _a in self.nav_toolbar.actions():
            if _a.text() in ('Subplots', 'Customize'):
                self.nav_toolbar.removeAction(_a)
        _icon_sz = self.nav_toolbar.iconSize()
        _tb_h    = self.nav_toolbar.sizeHint().height()
        _btn_style = (
            "QToolButton { border: none; background: transparent; border-radius: 3px; }"
            "QToolButton:hover { background: rgba(0,0,0,0.06); }"
            "QToolButton:checked { background: rgba(25,118,210,0.12); }"
        )
        _fig_btn = QToolButton()
        _fig_btn.setIcon(self.nav_toolbar._icon('qt4_editor_options'))
        _fig_btn.setIconSize(_icon_sz)
        _fig_btn.setFixedSize(_tb_h, _tb_h)
        _fig_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        _fig_btn.setToolTip('Figure Options (P)')
        _fig_btn.setStyleSheet(_btn_style)
        _fig_btn.clicked.connect(self.open_figure_options)
        self._focus_btn = QToolButton()
        self._focus_btn.setIcon(self._make_focus_icon(_icon_sz.width()))
        self._focus_btn.setIconSize(_icon_sz)
        self._focus_btn.setFixedSize(_tb_h, _tb_h)
        self._focus_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self._focus_btn.setCheckable(True)
        self._focus_btn.setToolTip('Focus plot — hide panels (F)')
        self._focus_btn.setStyleSheet(_btn_style)
        self._focus_btn.toggled.connect(self._toggle_focus_mode)
        QShortcut(QKeySequence('F'), self, activated=self._focus_btn.toggle)
        toolbar_row = QHBoxLayout()
        toolbar_row.setContentsMargins(0, 0, 0, 0)
        toolbar_row.setSpacing(0)
        toolbar_row.addWidget(self.nav_toolbar)
        toolbar_row.addWidget(_fig_btn)
        toolbar_row.addWidget(self._focus_btn)
        center_layout.addLayout(toolbar_row)
        # Wrap canvas in QScrollArea so stacked-view with many panes scrolls
        # vertically instead of squashing every signal to ~30 pixels. Canvas
        # min-height is bumped per refresh from _set_canvas_height_for_panes.
        self.canvas_scroll = QScrollArea()
        self.canvas_scroll.setWidget(self.canvas)
        self.canvas_scroll.setWidgetResizable(True)
        self.canvas_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.canvas_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.canvas_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        center_layout.addWidget(self.canvas_scroll)
        self.canvas.mpl_connect('resize_event', self._on_canvas_resize)
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        self.canvas.mpl_connect('button_release_event', self.on_canvas_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        # Freeze the constrained_layout solver right after a stacked rebuild's
        # draw has solved it — see _pending_freeze / _on_draw_event.
        self.canvas.mpl_connect('draw_event', self._on_draw_event)
        self.canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.canvas.installEventFilter(self)
        return center_widget

    def create_control_panel(self) -> QWidget:
        em = self._em
        iv = max(2, em // 6)
        ih = max(4, em // 4)
        sp = max(1, em // 8)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(0)

        # View Mode
        mode_box = CollapsibleBox("View Mode")
        mode_group = QWidget()
        mode_layout = QVBoxLayout(mode_group)
        mode_layout.setContentsMargins(ih, iv, ih, iv)
        mode_layout.setSpacing(sp)
        self._view_mode_group = QButtonGroup(self)
        self.radio_standard = QRadioButton("Standard")
        self.radio_standard.setChecked(True)
        self.radio_stacked = QRadioButton("Stacked")
        self.radio_stacked.setToolTip(
            "Each signal in its own pane with shared X axis. "
            "Preserves real amplitude and per-signal Y autoscale.")
        self.radio_timing = QRadioButton("Digital Timing (Simplified)")
        self.radio_timing.setToolTip(
            "Square-wave view for digital/logic signals. "
            "Only available for transient analysis.")
        for btn in (self.radio_standard, self.radio_stacked, self.radio_timing):
            self._view_mode_group.addButton(btn)
            mode_layout.addWidget(btn)
        self._view_mode_group.buttonToggled.connect(self.on_view_mode_changed)
        mode_box.addWidget(mode_group)
        right_layout.addWidget(mode_box)

        # Display Options
        display_box = CollapsibleBox("Display Options")
        display_group = QWidget()
        display_layout = QVBoxLayout(display_group)
        display_layout.setContentsMargins(ih, iv, ih, iv)
        display_layout.setSpacing(sp)
        self.grid_check = QCheckBox("Show Grid")
        self.grid_check.setChecked(True)
        self.grid_check.stateChanged.connect(self.toggle_grid)
        display_layout.addWidget(self.grid_check)
        self.legend_check = QCheckBox("Show Legend")
        self.legend_check.setChecked(False)
        self.legend_check.stateChanged.connect(self.toggle_legend)
        display_layout.addWidget(self.legend_check)
        self.autoscale_check = QCheckBox("Autoscale")
        self.autoscale_check.setChecked(True)
        self.autoscale_check.stateChanged.connect(self.refresh_plot)
        display_layout.addWidget(self.autoscale_check)
        self.stats_check = QCheckBox("Show Stats")
        self.stats_check.setToolTip("Show min/max/RMS/frequency stats on each stacked pane")
        self.stats_check.setChecked(True)
        self.stats_check.setVisible(False)
        self.stats_check.stateChanged.connect(self.refresh_plot)
        display_layout.addWidget(self.stats_check)
        display_box.addWidget(display_group)
        right_layout.addWidget(display_box)

        # Digital Timing Controls
        self.timing_box = CollapsibleBox("Digital Timing Controls")
        timing_group = QWidget()
        timing_layout = QVBoxLayout(timing_group)
        timing_layout.setContentsMargins(ih, iv, ih, iv)
        timing_layout.setSpacing(sp)
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Threshold:"))
        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setRange(-100, 100)
        self.threshold_spinbox.setDecimals(3)
        self.threshold_spinbox.setSingleStep(0.1)
        self.threshold_spinbox.setSuffix("")
        self.threshold_spinbox.setSpecialValueText("Auto")
        self.threshold_spinbox.setValue(self.threshold_spinbox.minimum())
        self.threshold_spinbox.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.threshold_spinbox.valueChanged.connect(self.on_threshold_changed)
        threshold_layout.addWidget(self.threshold_spinbox)
        timing_layout.addLayout(threshold_layout)
        spacing_layout = QHBoxLayout()
        spacing_layout.addWidget(QLabel("Spacing:"))
        self.spacing_slider = QSlider(Qt.Orientation.Horizontal)
        self.spacing_slider.setRange(100, 200)
        self.spacing_slider.setValue(120)
        self.spacing_slider.valueChanged.connect(self.on_spacing_changed)
        self.spacing_label = QLabel("1.2x")
        spacing_layout.addWidget(self.spacing_slider)
        spacing_layout.addWidget(self.spacing_label)
        timing_layout.addLayout(spacing_layout)
        self.timing_box.addWidget(timing_group)
        self.timing_box.content_area.setEnabled(False)
        self.timing_box.setVisible(False)
        right_layout.addWidget(self.timing_box)

        # Cursor Measurements
        cursor_box = CollapsibleBox("Cursor Measurements")
        cursor_group = QWidget()
        cursor_layout = QVBoxLayout(cursor_group)
        cursor_layout.setContentsMargins(ih, iv, ih, iv)
        cursor_layout.setSpacing(sp)

        self.cursor1_label = QLabel('<b style="color:#e53935">C1</b>  <span style="color:#aaa">not set</span>')
        self.cursor1_label.setWordWrap(True)
        self.cursor1_label.setStyleSheet("font-size: 13px; padding: 3px 0;")
        self.cursor2_label = QLabel('<b style="color:#1976d2">C2</b>  <span style="color:#aaa">not set</span>')
        self.cursor2_label.setWordWrap(True)
        self.cursor2_label.setStyleSheet("font-size: 13px; padding: 3px 0;")
        self.delta_label = QLabel('<b style="color:#e65100">ΔX</b>  <span style="color:#aaa">—</span>')
        self.delta_label.setStyleSheet("font-size: 13px; padding: 3px 0;")

        def _cursor_sep() -> QLabel:
            s = QLabel()
            s.setFixedHeight(1)
            s.setStyleSheet("background-color: #d0d0d0; margin: 2px 0;")
            return s

        cursor_layout.setSpacing(8)
        cursor_layout.addWidget(self.cursor1_label)
        cursor_layout.addWidget(_cursor_sep())
        cursor_layout.addWidget(self.cursor2_label)
        cursor_layout.addWidget(_cursor_sep())
        cursor_layout.addWidget(self.delta_label)
        cursor_help = QLabel(
            "L-click = Cursor 1   ·   Middle / R-click = Cursor 2\n"
            "R-click in stacked view = pane menu")
        cursor_help.setStyleSheet("color: #757575; font-size: 11px;")
        cursor_help.setWordWrap(True)
        cursor_layout.addWidget(cursor_help)
        self.clear_cursors_btn = QPushButton("Clear Cursors")
        self.clear_cursors_btn.clicked.connect(self.clear_cursors)
        cursor_layout.addWidget(self.clear_cursors_btn)
        cursor_box.addWidget(cursor_group)
        right_layout.addWidget(cursor_box)

        # Export Tools
        export_box = CollapsibleBox("Export Tools")
        export_group = QWidget()
        export_layout = QVBoxLayout(export_group)
        export_layout.setContentsMargins(ih, iv, ih, iv)
        export_layout.setSpacing(sp)
        self.export_btn = QPushButton("Export Image")
        self.export_btn.clicked.connect(self.export_image)
        export_layout.addWidget(self.export_btn)
        self.func_input = QLineEdit()
        self.func_input.setPlaceholderText("e.g., v(net1) + v(net2)  or  abs(v(net1))")
        self.func_input.returnPressed.connect(self.plot_function)
        export_layout.addWidget(self.func_input)
        self.plot_func_btn = QPushButton("Plot Function")
        self.plot_func_btn.clicked.connect(self.plot_function)
        export_layout.addWidget(self.plot_func_btn)
        export_box.addWidget(export_group)
        right_layout.addWidget(export_box)

        right_layout.addStretch()
        return right_widget

    def create_menu_bar(self) -> None:
        file_menu = self.menu_bar.addMenu('File')
        export_action = QAction('Export Image...', self)
        export_action.triggered.connect(self.export_image)
        file_menu.addAction(export_action)
        view_menu = self.menu_bar.addMenu('View')
        zoom_in_action = QAction('Zoom In', self)
        zoom_in_action.setShortcut('Ctrl++')
        zoom_in_action.triggered.connect(self.zoom_in)
        view_menu.addAction(zoom_in_action)
        zoom_out_action = QAction('Zoom Out', self)
        zoom_out_action.setShortcut('Ctrl+-')
        zoom_out_action.triggered.connect(self.zoom_out)
        view_menu.addAction(zoom_out_action)
        reset_view_action = QAction('Reset View', self)
        reset_view_action.setShortcut('Ctrl+0')
        reset_view_action.triggered.connect(self.reset_view)
        view_menu.addAction(reset_view_action)

    def _rebuild_nb_sorted(self) -> None:
        """Cache NBList sorted longest-first for use in _resolve_expr."""
        self._nb_sorted = sorted(
            enumerate(self.obj_dataext.NBList),
            key=lambda t: len(t[1]), reverse=True
        )

    def _parse_tran_start_time(self) -> float:
        try:
            with open(os.path.join(self.file_path, "analysis"), 'r') as f:
                parts = f.read().strip().split()
            if len(parts) >= 4 and parts[0] == '.tran':
                return float(parts[3])
        except Exception:
            pass
        return 0.0

    def load_simulation_data(self) -> None:
        self._cursor_interp_cache.clear()
        # new data → force full rebuild on next refresh
        self._drawn_signature = None
        self._tran_start_time = self._parse_tran_start_time()
        self.obj_dataext = DataExtraction()
        self.plot_type = self.obj_dataext.openFile(self.file_path)
        self.obj_dataext.computeAxes()
        self._rebuild_nb_sorted()
        self.data_info = self.obj_dataext.numVals()
        self.volts_length = self.data_info[1]
        if self.plot_type[0] == DataExtraction.AC_ANALYSIS:
            self.analysis_label.setText("AC Analysis")
        elif self.plot_type[0] == DataExtraction.TRANSIENT_ANALYSIS:
            self.analysis_label.setText("Transient Analysis")
        else:
            self.analysis_label.setText("DC Analysis")
        self.populate_waveform_list()
        # NBList ready; resolve persisted stacked-view layout
        self._apply_persisted_layout()
        is_transient = self.plot_type[0] == DataExtraction.TRANSIENT_ANALYSIS
        self.radio_timing.setEnabled(is_transient)
        if not is_transient:
            if self.radio_timing.isChecked():
                self.radio_standard.setChecked(True)
            self.radio_timing.setToolTip("Only available for transient analysis")
        else:
            self.radio_timing.setToolTip(
                "Square-wave view for digital/logic signals. "
                "Only available for transient analysis.")

    def create_colored_icon(self, color: QColor, is_selected: bool) -> QtGui.QIcon:
        pixmap = QPixmap(18, 18)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if is_selected:
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(1, 1, 16, 16)
        else:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            pen = QtGui.QPen(QColor("#9E9E9E"))
            pen.setWidth(1)
            painter.setPen(pen)
            painter.drawEllipse(2, 2, 14, 14)
        painter.end()
        return QtGui.QIcon(pixmap)

    def populate_waveform_list(self) -> None:
        self.waveform_list.clear()
        self.traces.clear()
        saved_colors = self.config.get('trace_colours', {})
        saved_thickness = self.config.get('trace_thickness', {})
        saved_style = self.config.get('trace_style', {})
        for i, node_name in enumerate(self.obj_dataext.NBList):
            color = saved_colors.get(node_name, self.color_palette[i % len(self.color_palette)])
            thickness = saved_thickness.get(node_name, DEFAULT_LINE_THICKNESS)
            style = saved_style.get(node_name, '-')
            self.traces[i] = Trace(index=i, name=node_name, color=color, thickness=thickness, style=style)
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, i)
            item.setToolTip("Voltage signal" if i < self.obj_dataext.volts_length else "Current signal")
            self.waveform_list.addItem(item)
            self.update_list_item_appearance(item, i)

    def filter_waveforms(self, text: str) -> None:
        needle = text.lower()
        for i in range(self.waveform_list.count()):
            item = self.waveform_list.item(i)
            if item:
                item.setHidden(needle not in item.text().lower())

    def on_waveform_toggle(self, item: QListWidgetItem) -> None:
        index = item.data(Qt.ItemDataRole.UserRole)
        # Negative UserRole = function trace item; positive = simulation trace.
        if isinstance(index, int) and index < 0:
            f_idx = -index - 1
            if f_idx < len(self._func_visible):
                self._func_visible[f_idx] = not self._func_visible[f_idx]
                label, _fx, _fy, color, *_ = self._func_traces[f_idx]
                self._update_func_item_appearance(
                    item, label, color, self._func_visible[f_idx])
            self._schedule_refresh()
            return
        # item.isSelected() is unreliable when setItemWidget is used — clicks land
        # on the child widget and never update Qt's selection model. Toggle instead.
        self.traces[index].visible = not self.traces[index].visible
        self.update_list_item_appearance(item, index)
        self._schedule_refresh()

    def update_list_item_appearance(self, item: QListWidgetItem, index: int) -> None:
        t = self.traces[index]
        widget = QWidget()
        # CRITICAL: the row's custom QWidget AND its children must NOT eat
        # mouse events. WA_TransparentForMouseEvents on the parent only
        # affects that exact widget — Qt does NOT auto-propagate to
        # children. Set it on every interactive child so the QListWidget
        # gets press/move events and can initiate drags.
        transparent = Qt.WidgetAttribute.WA_TransparentForMouseEvents
        widget.setAttribute(transparent, True)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(10)
        icon_label = QLabel()
        icon_label.setAttribute(transparent, True)
        color = QColor(t.color) if t.visible else QColor("#9E9E9E")
        icon = self.create_colored_icon(color, t.visible)
        icon_label.setPixmap(icon.pixmap(18, 18))
        text_label = QLabel(t.name)
        text_label.setAttribute(transparent, True)
        text_label.setStyleSheet("color: #212121; font-weight: 500;" if t.visible else "color: #757575; font-weight: normal;")
        layout.addWidget(icon_label)
        layout.addWidget(text_label)
        layout.addStretch()
        self.waveform_list.setItemWidget(item, widget)
        item.setText(t.name)

    def _update_func_item_appearance(self, item: QListWidgetItem,
                                      label: str, color: str,
                                      visible: bool) -> None:
        """Set visual appearance of a function-trace waveform list item.

        Mirrors update_list_item_appearance but for func traces.  The 'ƒ'
        prefix and italic style distinguish them from simulation signals.
        """
        transparent = Qt.WidgetAttribute.WA_TransparentForMouseEvents
        widget = QWidget()
        widget.setAttribute(transparent, True)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(10)
        icon_label = QLabel()
        icon_label.setAttribute(transparent, True)
        icon_color = QColor(color) if visible else QColor("#9E9E9E")
        icon = self.create_colored_icon(icon_color, visible)
        icon_label.setPixmap(icon.pixmap(18, 18))
        display = f"ƒ  {label}"
        text_label = QLabel(display)
        text_label.setAttribute(transparent, True)
        text_label.setStyleSheet(
            f"color: {color}; font-weight: 500; font-style: italic;" if visible
            else "color: #757575; font-weight: normal; font-style: italic;")
        layout.addWidget(icon_label)
        layout.addWidget(text_label)
        layout.addStretch()
        self.waveform_list.setItemWidget(item, widget)
        item.setText(display)

    def _sync_func_trace_list(self) -> None:
        """Remove all func-trace list items and re-add from current _func_traces.

        Called after any mutation of _func_traces so the waveform list stays
        in sync (correct count, correct negative UserRole indices, correct
        labels/colours after a middle-of-list removal re-numbers everything).
        """
        for i in range(self.waveform_list.count() - 1, -1, -1):
            it = self.waveform_list.item(i)
            if it is not None:
                val = it.data(Qt.ItemDataRole.UserRole)
                if isinstance(val, int) and val < 0:
                    self.waveform_list.takeItem(i)
        for f_idx, (label, _fx, _fy, color, *_) in enumerate(self._func_traces):
            visible = (self._func_visible[f_idx]
                       if f_idx < len(self._func_visible) else True)
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, -(f_idx + 1))
            item.setToolTip(f"Function trace: {label}")
            self.waveform_list.addItem(item)
            self._update_func_item_appearance(item, label, color, visible)

    def select_all_waveforms(self) -> None:
        self.waveform_list.setUpdatesEnabled(False)
        try:
            for i in range(self.waveform_list.count()):
                item = self.waveform_list.item(i)
                if item and not item.isHidden():
                    index = item.data(Qt.ItemDataRole.UserRole)
                    if isinstance(index, int) and index < 0:
                        f_idx = -index - 1
                        if f_idx < len(self._func_visible):
                            self._func_visible[f_idx] = True
                            label, _fx, _fy, color, *_ = self._func_traces[f_idx]
                            self._update_func_item_appearance(item, label, color, True)
                    else:
                        self.traces[index].visible = True
                        self.update_list_item_appearance(item, index)
        finally:
            self.waveform_list.setUpdatesEnabled(True)
        self._schedule_refresh()

    def deselect_all_waveforms(self) -> None:
        for t in self.traces.values():
            t.visible = False
        for f_idx in range(len(self._func_visible)):
            self._func_visible[f_idx] = False
        self.waveform_list.setUpdatesEnabled(False)
        try:
            for i in range(self.waveform_list.count()):
                item = self.waveform_list.item(i)
                if item:
                    index = item.data(Qt.ItemDataRole.UserRole)
                    if isinstance(index, int) and index < 0:
                        f_idx = -index - 1
                        if f_idx < len(self._func_traces):
                            label, _fx, _fy, color, *_ = self._func_traces[f_idx]
                            self._update_func_item_appearance(item, label, color, False)
                    else:
                        self.update_list_item_appearance(item, index)
        finally:
            self.waveform_list.setUpdatesEnabled(True)
        self._schedule_refresh()

    def show_list_context_menu(self, position: QtCore.QPoint) -> None:
        item = self.waveform_list.itemAt(position)
        menu = QMenu()

        select_all_action = menu.addAction("Select All")
        select_all_action.setShortcut(QKeySequence.StandardKey.SelectAll)
        select_all_action.triggered.connect(self.select_all_waveforms)
        deselect_action = menu.addAction("Deselect All")
        deselect_action.triggered.connect(self.deselect_all_waveforms)

        if item:
            index = item.data(Qt.ItemDataRole.UserRole)
            menu.addSeparator()

            if isinstance(index, int) and index < 0:
                f_idx = -index - 1
                color_menu = menu.addMenu("Change colour ▶")
                self._populate_func_color_menu(color_menu, f_idx)
                thickness_menu = menu.addMenu("Thickness ▶")
                for _thickness, _tlabel in THICKNESS_OPTIONS:
                    _act = thickness_menu.addAction(_tlabel)
                    _act.triggered.connect(
                        lambda checked=False, t=_thickness, fi=f_idx: self._change_func_thickness(fi, t))
                style_menu = menu.addMenu("Style ▶")
                for _style, _slabel in LINE_STYLES:
                    _act = style_menu.addAction(_slabel)
                    _act.triggered.connect(
                        lambda checked=False, s=_style, fi=f_idx: self._change_func_style(fi, s))
                menu.addSeparator()
                is_vis = (f_idx < len(self._func_visible) and self._func_visible[f_idx])
                hide_show_action = menu.addAction("Hide" if is_vis else "Show")
                hide_show_action.triggered.connect(
                    lambda checked=False, fi=f_idx: self._toggle_func_trace_visibility(fi))
                remove_action = menu.addAction("Remove")
                remove_action.triggered.connect(
                    lambda checked=False, fi=f_idx: self._remove_function_pane(fi))
            else:
                color_menu = menu.addMenu("Change colour ▶")
                self.populate_color_menu(color_menu, [item])

                thickness_menu = menu.addMenu("Thickness ▶")
                for thickness, label in THICKNESS_OPTIONS:
                    action = thickness_menu.addAction(label)
                    action.triggered.connect(lambda checked, t=thickness: self.change_thickness([item], t))

                style_menu = menu.addMenu("Style ▶")
                for style, label in LINE_STYLES:
                    action = style_menu.addAction(label)
                    action.triggered.connect(lambda checked, s=style: self.change_style([item], s))

                menu.addSeparator()

                rename_action = menu.addAction("Rename...")
                rename_action.triggered.connect(lambda: self.rename_trace(item))

                t = self.traces[index]
                hide_show_action = menu.addAction("Hide" if t.visible else "Show")
                hide_show_action.triggered.connect(lambda: self.toggle_trace_visibility([item]))

                menu.addSeparator()

                properties_action = menu.addAction("Figure Options...")
                properties_action.triggered.connect(self.open_figure_options)

        menu.exec(self.waveform_list.mapToGlobal(position))

    def populate_color_menu(self, menu: QMenu, selected_items: List[QListWidgetItem]) -> None:
        color_widget = QWidget()
        color_widget.setStyleSheet("background-color: #FFFFFF;")
        grid_layout = QGridLayout(color_widget)
        grid_layout.setSpacing(2)
        for i, color in enumerate(self.color_palette):
            btn = QPushButton()
            btn.setFixedSize(24, 24)
            btn.setStyleSheet(f"QPushButton{{background-color:{color};border:1px solid #E0E0E0;border-radius:2px;}}QPushButton:hover{{border:2px solid #212121;}}")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda checked, c=color: self.change_color_and_close(selected_items, c, menu))
            grid_layout.addWidget(btn, i // 4, i % 4)
        widget_action = QWidgetAction(menu)
        widget_action.setDefaultWidget(color_widget)
        menu.addAction(widget_action)
        menu.addSeparator()
        more_action = menu.addAction("More...")
        more_action.triggered.connect(lambda: self.change_color_dialog(selected_items))

    def change_color_and_close(self, items: List[QListWidgetItem], color: str, menu: QMenu) -> None:
        self.change_color(items, color)
        parent = menu.parent()
        while isinstance(parent, QMenu):
            parent.close()
            parent = parent.parent()

    def change_color(self, items: List[QListWidgetItem], color: str) -> None:
        for item in items:
            index = item.data(Qt.ItemDataRole.UserRole)
            self.traces[index].update_line(color=color)
            self.update_list_item_appearance(item, index)
            if self._current_view_mode == 'timing' and self.panes:
                self.update_timing_tick_colors()
                for ann_text in self.timing_annotations.get(index, []):
                    ann_text.set_color(color)
            elif self._current_view_mode == 'stacked' and self.panes:
                for pane_idx, group in enumerate(self._pane_groups):
                    if group and group[0] == index and pane_idx < len(self.panes):
                        self.panes[pane_idx].set_title(
                            self.traces[index].name, loc='left', color=color,
                            fontsize=LEGEND_FONT_SIZE, fontweight='bold', pad=3)
                        break
        self.save_config()
        self.canvas.draw()

    def update_timing_tick_colors(self) -> None:
        # No-op outside timing view: ytick labels in normal/stacked views are
        # numeric voltage/current ticks, not trace names, so colouring them
        # by trace would corrupt the axis legend.
        if self._current_view_mode != 'timing' or not self.panes:
            return
        visible = list(reversed(self.visible_traces))
        ytick_labels = self.axes.get_yticklabels()
        for i, label in enumerate(ytick_labels):
            if i < len(visible):
                label.set_color(visible[i].color)

    def change_color_dialog(self, items: List[QListWidgetItem]) -> None:
        color = QColorDialog.getColor()
        if color.isValid():
            self.change_color(items, color.name())

    def change_thickness(self, items: List[QListWidgetItem], thickness: float) -> None:
        for item in items:
            self.traces[item.data(Qt.ItemDataRole.UserRole)].update_line(thickness=thickness)
        self.save_config()
        self.canvas.draw()

    def change_style(self, items: List[QListWidgetItem], style: str) -> None:
        needs_replot = style == 'steps-post'
        for item in items:
            index = item.data(Qt.ItemDataRole.UserRole)
            if needs_replot:
                self.traces[index].style = style
            else:
                self.traces[index].update_line(style=style)
        self.save_config()
        if needs_replot:
            self.refresh_plot()
        else:
            self.canvas.draw()

    def rename_trace(self, item: QListWidgetItem) -> None:
        index = item.data(Qt.ItemDataRole.UserRole)
        t = self.traces[index]
        new_name, ok = QInputDialog.getText(self, "Rename Trace", "New name:", text=t.name)
        if ok and new_name and new_name != t.name:
            t.name = new_name
            self.obj_dataext.NBList[index] = new_name
            self._rebuild_nb_sorted()
            self.update_list_item_appearance(item, index)
            if self.legend_check.isChecked():
                # Name change must re-label the legend; signature can't see it.
                self._force_full_refresh = True
                self.refresh_plot()

    def toggle_trace_visibility(self, items: List[QListWidgetItem]) -> None:
        # Use t.visible as single source of truth — same path as left-click toggle.
        # Going directly to line_object.set_visible() bypasses refresh_plot and
        # gets stomped the next time anything triggers a redraw.
        any_visible = any(self.traces[item.data(Qt.ItemDataRole.UserRole)].visible for item in items)
        for item in items:
            index = item.data(Qt.ItemDataRole.UserRole)
            self.traces[index].visible = not any_visible
            self.update_list_item_appearance(item, index)
        self._schedule_refresh()

    def open_figure_options(self) -> None:
        try:
            if hasattr(self.fig.canvas, 'toolbar') and hasattr(self.fig.canvas.toolbar, 'edit_parameters'):
                # matplotlib's built-in editor already handles multi-axes —
                # it shows a per-axes selector so each pane can be edited.
                self.fig.canvas.toolbar.edit_parameters()
                return
            from matplotlib.backends.qt_compat import QtWidgets
            from matplotlib.backends.qt_editor import _formlayout
            if hasattr(_formlayout, 'FormDialog'):
                current_title = self.fig._suptitle.get_text() if self.fig._suptitle is not None else ''
                options: List[Tuple[str, Any]] = [('Title', current_title)]
                # Multi-pane: only X (shared via sharex) + suptitle are global.
                # Per-pane Y limits and labels are skipped to avoid a 7-field
                # dialog that can only touch one pane meaningfully.
                multi = len(self.panes) > 1
                if self.panes:
                    options.append(('X Label', self.panes[-1].get_xlabel()))
                    options.append(('X Min', self.axes.get_xlim()[0]))
                    options.append(('X Max', self.axes.get_xlim()[1]))
                    if not multi:
                        options.append(('Y Label', self.axes.get_ylabel()))
                        options.append(('Y Min', self.axes.get_ylim()[0]))
                        options.append(('Y Max', self.axes.get_ylim()[1]))
                dialog = _formlayout.FormDialog(options, parent=self, title='Figure Options')
                if dialog.exec():
                    results = dialog.get_results()
                    if not results:
                        return
                    self.fig.suptitle(results[0])
                    if self.panes and len(results) > 1:
                        self.panes[-1].set_xlabel(results[1])
                        self.axes.set_xlim(results[2], results[3])
                        if not multi and len(results) > 4:
                            self.axes.set_ylabel(results[4])
                            self.axes.set_ylim(results[5], results[6])
                    self.canvas.draw()
            else:
                QMessageBox.information(self, "Figure Options", "Figure options are limited in this environment.\nYou can use the zoom and pan tools in the toolbar.")
        except Exception as e:
            logger.error(f"Error opening figure options: {e}")
            QMessageBox.information(self, "Figure Options", "Basic figure editing is available through the toolbar.")

    def _update_mode_controls(self) -> None:
        stacked = self.radio_stacked.isChecked()
        normal  = self.radio_standard.isChecked()
        self.autoscale_check.setVisible(normal)
        self.stats_check.setVisible(stacked)
        self.legend_check.setVisible(not stacked)

    def on_view_mode_changed(self, button, checked: bool) -> None:
        if not checked:
            return
        timing = self.radio_timing.isChecked()
        self.timing_box.setVisible(timing)
        self.timing_box.content_area.setEnabled(timing)
        self._update_mode_controls()
        self.refresh_plot()

    @staticmethod
    def _make_focus_icon(size: int) -> QIcon:
        px = QPixmap(size, size)
        px.fill(Qt.GlobalColor.transparent)
        p = QPainter(px)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor('#444444'), max(1, size // 12), Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        m = max(2, size // 6)
        a = max(3, size // 4)
        for cx, cy in ((m, m), (size-m, m), (m, size-m), (size-m, size-m)):
            dx = a if cx == m else -a
            dy = a if cy == m else -a
            p.drawLine(cx, cy + dy, cx, cy)
            p.drawLine(cx, cy, cx + dx, cy)
        p.end()
        return QIcon(px)

    def _toggle_focus_mode(self, focused: bool) -> None:
        self.left_panel.setVisible(not focused)
        self.right_panel.setVisible(not focused)
        self._focus_btn.setToolTip('Restore panels (F)' if focused else 'Focus plot — hide panels (F)')

    def _build_panes(self, n: int, sharex: bool = True,
                     hspace: float = 0.08) -> List[Any]:
        """Create N stacked subplots, store in self.panes, return them.

        n==1 — single Axes (equivalent to add_subplot(111)).
        n>=2 — N vertically stacked Axes with shared X-axis by default.

        In stacked view, self._pane_heights (when present and matching N)
        becomes the gridspec height_ratios so individual panes can be
        resized via the divider-drag handlers.

        Caller must fig.clear() before invoking; this helper does not clear.
        self.axes is aliased to self.panes[0] so legacy single-axes call
        sites keep working unchanged.
        """
        if n <= 1:
            self.panes = [self.fig.add_subplot(111)]
        else:
            gridspec_kw: Dict[str, Any] = {'hspace': hspace}
            if (self._current_view_mode == 'stacked'
                    and self._pane_heights
                    and all(h > 0 for h in self._pane_heights)):
                # _pane_heights tracks group panes only; pad with 1.0 for any
                # trailing function panes so the gridspec still matches N.
                heights = list(self._pane_heights)
                while len(heights) < n:
                    heights.append(1.0)
                gridspec_kw['height_ratios'] = heights[:n]
            axes = self.fig.subplots(
                n, 1, sharex=sharex, gridspec_kw=gridspec_kw,
            )
            # subplots returns ndarray when n>1
            self.panes = list(axes) if hasattr(axes, '__iter__') else [axes]
        self.axes = self.panes[0]
        self._set_canvas_height_for_panes(n)
        return self.panes

    def _set_canvas_height_for_panes(self, n: int) -> None:
        """Force canvas tall enough for N stacked panes; let it shrink elsewhere.

        Without this, QScrollArea.widgetResizable=True squashes the canvas
        to the viewport even when N=20. We set a min-height proportional
        to pane count so the scroll bar appears as soon as panes would
        otherwise become unreadable.
        """
        if not hasattr(self, 'canvas_scroll') or not hasattr(self, 'canvas'):
            return
        viewport_h = self.canvas_scroll.viewport().height()
        if self._current_view_mode == 'stacked' and n > 1:
            wanted = max(viewport_h, int(n * MIN_STACKED_PANE_HEIGHT_PX))
            self.canvas.setMinimumHeight(wanted)
        else:
            # Non-stacked modes fit the viewport — drop the floor.
            self.canvas.setMinimumHeight(0)

    def _sync_pane_groups_to_visible(self) -> None:
        """Reconcile self._pane_groups + heights with current visibility.

        Invariant: each group has exactly ONE trace index. Stacked view is
        always "one signal per pane" in v3.1+.

        - Invisible traces are dropped (their pane disappears).
        - Existing pane order is preserved.
        - Newly-visible traces (not already in any pane) join at the bottom
          with default height 1.0.
        """
        visible_set = {t.index for t in self.visible_traces}
        new_groups: List[List[int]] = []
        kept_heights: List[float] = []
        for orig_idx, group in enumerate(self._pane_groups):
            # Keep only the first surviving trace — enforces 1-per-pane
            # for any legacy config that had multi-trace groups.
            survivor = next((i for i in group if i in visible_set), None)
            if survivor is None:
                continue
            new_groups.append([survivor])
            visible_set.discard(survivor)
            if orig_idx < len(self._pane_heights):
                kept_heights.append(self._pane_heights[orig_idx])
            else:
                kept_heights.append(1.0)
        for idx in sorted(visible_set):
            new_groups.append([idx])
            kept_heights.append(1.0)
        self._pane_groups = new_groups
        self._pane_heights = kept_heights

    def _pane_anchor_name(self, ax) -> Optional[str]:
        """Return the name of the first visible trace plotted on ax, or None.

        Used as the lookup key for per-pane ylim preservation across
        refresh_plot. Walks self.traces in insertion order so anchors stay
        stable even if Y autoscaling slightly changes line ordering.
        """
        for t in self.traces.values():
            if t.visible and t.line_object is not None and t.line_object.axes is ax:
                return t.name
        return None

    def _build_pane_anchor_map(self) -> Dict[int, str]:
        """Return {id(ax): anchor_name} covering all current panes, in O(traces).

        Callers that loop over all panes should build this once and do O(1)
        dict lookups per pane, rather than calling _pane_anchor_name() per pane
        which is O(traces) each and makes capture/restore O(N²) in stacked view.
        """
        result: Dict[int, str] = {}
        for t in self.traces.values():
            lo = t.line_object
            if t.visible and lo is not None:
                ax_id = id(lo.axes)
                if ax_id not in result:
                    result[ax_id] = t.name
        return result

    def _capture_view_state(self) -> None:
        """Snapshot xlim and per-pane ylim before a refresh that rebuilds axes."""
        if not self.panes:
            return
        self._saved_xlim = self.axes.get_xlim()
        self._saved_pane_ylims = {}
        anchor_map = self._build_pane_anchor_map()
        for ax in self.panes:
            anchor = anchor_map.get(id(ax))
            if anchor is not None:
                self._saved_pane_ylims[anchor] = ax.get_ylim()

    def _restore_view_state(self) -> None:
        """Re-apply preserved xlim + per-pane ylim after refresh rebuilds axes.

        Two restore paths, evaluated in order so locks always win:

        1. **Locked panes** — any pane whose anchor is in self._pane_lock_y
           and has a stored ylim in self._locked_ylims gets that ylim
           re-applied. Survives every refresh until the lock is cleared.

        2. **One-shot preserve-zoom** — entries in self._saved_pane_ylims
           (set by _capture_view_state when autoscale is off in a mode-
           preserving refresh) are applied to whatever pane currently
           anchors that trace, then the dict is cleared.

        Calling this with both buckets empty is a no-op, so refresh_plot
        can call it unconditionally.
        """
        if not self.panes:
            return
        anchor_map = self._build_pane_anchor_map()
        # Apply persistent locks first
        for ax in self.panes:
            anchor = anchor_map.get(id(ax))
            if anchor is None:
                continue
            if self._pane_lock_y.get(anchor) and anchor in self._locked_ylims:
                ax.set_ylim(self._locked_ylims[anchor])
        # Then one-shot preserve-zoom — skip panes already pinned by a lock
        if self._saved_xlim is not None:
            self.axes.set_xlim(self._saved_xlim)
        for ax in self.panes:
            anchor = anchor_map.get(id(ax))
            if anchor is None or self._pane_lock_y.get(anchor):
                continue
            if anchor in self._saved_pane_ylims:
                ax.set_ylim(self._saved_pane_ylims[anchor])
        self._saved_xlim = None
        self._saved_pane_ylims = {}

    def _snapshot_pane_for_lock(self, anchor: str) -> None:
        """Capture the current ylim of the pane that anchors `anchor`.

        Used by the (forthcoming) menu's Lock-Y toggle to seed
        self._locked_ylims; without this seed the lock would have nothing
        to restore on the first refresh after locking.
        """
        if not anchor or not self.panes:
            return
        anchor_map = self._build_pane_anchor_map()
        for ax in self.panes:
            if anchor_map.get(id(ax)) == anchor:
                self._locked_ylims[anchor] = ax.get_ylim()
                return

    def _clear_pane_lock(self, anchor: str) -> None:
        """Remove a pane lock so the next refresh autoscales freely."""
        self._pane_lock_y.pop(anchor, None)
        self._locked_ylims.pop(anchor, None)

    # ── Pane divider resize (mouse drag between panes) ───────────────────

    def _divider_under_mouse(self, event) -> Optional[int]:
        """Return upper-pane index when the mouse is near a divider gap.

        Only meaningful in stacked mode with N>=2 panes. Returns the index
        of the pane ABOVE the divider — i.e. the pane whose height changes
        in tandem with the one below during a drag.
        """
        if (self._current_view_mode != 'stacked'
                or len(self.panes) < 2
                or event.y is None):
            return None
        # Only consider the dividers between group panes; func panes are
        # not part of _pane_heights, so don't expose their boundaries.
        upper_count = min(len(self._pane_groups), len(self.panes)) - 1
        for i in range(upper_count):
            bottom = self.panes[i].bbox.y0     # bottom edge of upper pane
            top = self.panes[i + 1].bbox.y1    # top edge of lower pane
            mid = (bottom + top) / 2.0
            if abs(event.y - mid) <= DIVIDER_HIT_TOLERANCE_PX:
                return i
        return None

    def _start_divider_drag(self, upper_idx: int, event) -> None:
        """Capture initial heights + cached pixel geometry for the drag."""
        if upper_idx + 1 >= len(self._pane_heights):
            return
        upper_bb = self.panes[upper_idx].bbox.height
        lower_bb = self.panes[upper_idx + 1].bbox.height
        self._divider_drag = {
            'upper_idx': upper_idx,
            'start_y': event.y,
            'start_height_upper': self._pane_heights[upper_idx],
            'start_height_lower': self._pane_heights[upper_idx + 1],
            'combined_px': max(1.0, upper_bb + lower_bb),
            'moved': False,
        }
        # Capture axis positions in figure coords BEFORE disabling the
        # constrained_layout solver. During drag we reposition axes via
        # ax.set_position() so the solver never runs per mouse-move.
        self._drag_ax_positions = [ax.get_position() for ax in self.panes]
        self.fig.set_layout_engine('none')
        # Decimate line data so each draw_idle renders far fewer segments.
        # Full data is restored in _finish_divider_drag (or by refresh_plot).
        _MAX_DRAG_PTS = 800
        self._saved_line_data: Dict[int, Any] = {}
        for t in self.traces.values():
            if t.line_object is not None:
                xd = t.line_object.get_xdata()
                yd = t.line_object.get_ydata()
                n = len(xd)
                if n > _MAX_DRAG_PTS:
                    step = max(1, n // _MAX_DRAG_PTS)
                    self._saved_line_data[t.index] = (xd, yd)
                    t.line_object.set_data(xd[::step], yd[::step])
        self.canvas.setCursor(Qt.CursorShape.SizeVerCursor)

    def _update_divider_drag(self, event) -> None:
        """Live-redistribute height ratios via in-place gridspec mutation.

        Critically, this does NOT call refresh_plot — a full rebuild on
        every mouse-motion event tanks performance (~50ms × 30Hz = jank).
        Instead it mutates the GridSpec height_ratios on the live axes
        and asks for a deferred redraw. On release, _finish_divider_drag
        does one full refresh to let constrained_layout fully settle.
        """
        d = self._divider_drag
        if d is None or event.y is None:
            return
        i = d['upper_idx']
        # See _start_divider_drag for sign convention. Positive delta_px
        # = cursor moved down = upper pane grows.
        delta_px = d['start_y'] - event.y
        sum_h = d['start_height_upper'] + d['start_height_lower']
        frac = delta_px / d['combined_px']
        new_upper = max(0.1, min(sum_h - 0.1,
                                 d['start_height_upper'] + frac * sum_h))
        new_lower = sum_h - new_upper
        self._pane_heights[i] = new_upper
        self._pane_heights[i + 1] = new_lower
        d['moved'] = True
        self._apply_height_ratios_live()

    def _apply_height_ratios_live(self) -> None:
        """Reposition panes during drag without running the constraint solver."""
        if not self.panes or len(self.panes) < 2:
            return
        self._reposition_panes_from_heights()
        self.canvas.draw_idle()

    def _reposition_panes_from_heights(self) -> None:
        """Distribute pane Bboxes in figure coords from _pane_heights ratios.

        Uses positions captured at drag-start as the outer envelope so the
        figure margins stay exactly as constrained_layout left them. Only
        vertical redistribution changes; horizontal extents are preserved
        per-pane (important when Y-axis labels have different widths).
        """
        positions = getattr(self, '_drag_ax_positions', None)
        if not positions or len(positions) != len(self.panes):
            return
        n = len(self.panes)
        n_heights = min(n, len(self._pane_heights))

        top    = positions[0].y1
        bottom = positions[-1].y0
        # Average gap between adjacent panes (hspace in figure coords)
        if n > 1:
            total_gap = sum(
                max(0.0, positions[i].y0 - positions[i + 1].y1)
                for i in range(n - 1)
            )
            avg_gap = total_gap / (n - 1)
        else:
            avg_gap = 0.0

        usable = max(0.01, (top - bottom) - avg_gap * (n - 1))
        ratios  = [self._pane_heights[i] if i < n_heights else 1.0 for i in range(n)]
        ratio_sum = max(0.01, sum(ratios))

        current_top = top
        for i, ax in enumerate(self.panes):
            h = max(0.005, usable * (ratios[i] / ratio_sum))
            ax.set_position([
                positions[i].x0,
                current_top - h,
                positions[i].x1 - positions[i].x0,
                h,
            ])
            current_top -= h + avg_gap

    def _finish_divider_drag(self) -> None:
        """End drag: restore CL + data, settle layout with one full refresh."""
        if self._divider_drag is None:
            return
        moved = self._divider_drag.get('moved', False)
        self._divider_drag = None
        self._drag_ax_positions = []
        # Restore full line data.  If moved=True, refresh_plot rebuilds
        # everything via fig.clear() anyway; still restore so the
        # no-move path (moved=False) shows correct data after cleanup.
        saved = getattr(self, '_saved_line_data', {})
        for idx, (xd, yd) in saved.items():
            if idx in self.traces and self.traces[idx].line_object is not None:
                self.traces[idx].line_object.set_data(xd, yd)
        self._saved_line_data = {}
        self.fig.set_layout_engine('constrained')
        self.canvas.unsetCursor()
        if moved:
            # Heights changed; pane geometry must be rebuilt + CL re-settled.
            self._force_full_refresh = True
            self.refresh_plot()
        else:
            self.canvas.draw_idle()

    # ── Pane reorder (Alt + left-drag) ───────────────────────────────────

    def _start_pane_drag(self, pane_idx: int) -> None:
        self._pane_drag = {'from_idx': pane_idx}
        self.canvas.setCursor(Qt.CursorShape.ClosedHandCursor)

    def _finish_pane_drag(self, event) -> None:
        d = self._pane_drag
        if d is None:
            return
        self._pane_drag = None
        self.canvas.unsetCursor()
        from_idx = d['from_idx']
        # Determine target pane from release position
        to_idx = None
        if event.inaxes is not None:
            to_idx = self._pane_index_of(event.inaxes)
        if to_idx is None or to_idx == from_idx:
            return
        if not (0 <= from_idx < len(self._pane_groups)
                and 0 <= to_idx < len(self._pane_groups)):
            return
        # Move the group; also move its height entry alongside so the user's
        # custom sizing follows the pane.
        grp = self._pane_groups.pop(from_idx)
        self._pane_groups.insert(to_idx, grp)
        if from_idx < len(self._pane_heights):
            h = self._pane_heights.pop(from_idx)
            if to_idx < len(self._pane_heights):
                self._pane_heights.insert(to_idx, h)
            else:
                self._pane_heights.append(h)
        # Reorder keeps the visible set (signature) the same — force the rebuild.
        self._force_full_refresh = True
        self.refresh_plot()

    def _apply_persisted_layout(self) -> None:
        """Hydrate name-keyed config entries into index-keyed live state.

        Runs once after populate_waveform_list has built self.traces.
        Stale names (signals not present in this simulation) are silently
        dropped — keeps the config forward-compatible across schematic
        edits. Function-trace panes are NOT persisted because their
        underlying expressions may reference signals that no longer exist.
        """
        name_to_idx = {t.name: t.index for t in self.traces.values()}

        groups_named = self.config.get('stacked_pane_groups')
        if isinstance(groups_named, list):
            resolved: List[List[int]] = []
            for group in groups_named:
                if not isinstance(group, list):
                    continue
                indices = [name_to_idx[n] for n in group if n in name_to_idx]
                if indices:
                    resolved.append(indices)
            if resolved:
                self._pane_groups = resolved

        lock = self.config.get('stacked_lock_y')
        if isinstance(lock, dict):
            self._pane_lock_y = {n: bool(v) for n, v in lock.items()
                                 if n in name_to_idx}

        ylims = self.config.get('stacked_locked_ylims')
        if isinstance(ylims, dict):
            for n, lims in ylims.items():
                if (n in name_to_idx and isinstance(lims, (list, tuple))
                        and len(lims) == 2):
                    self._locked_ylims[n] = (float(lims[0]), float(lims[1]))

        stats = self.config.get('stacked_stats_visible')
        if isinstance(stats, bool):
            self.stats_check.setChecked(stats)

        # Pane heights — name-keyed in config, projected back onto the
        # restored _pane_groups order. Missing anchors fall back to 1.0
        # which is the default equal-height ratio.
        heights_named = self.config.get('stacked_pane_heights')
        if isinstance(heights_named, dict) and self._pane_groups:
            resolved_heights: List[float] = []
            for g in self._pane_groups:
                if not g or g[0] not in self.traces:
                    resolved_heights.append(1.0)
                    continue
                anchor = self.traces[g[0]].name
                try:
                    resolved_heights.append(
                        max(0.1, float(heights_named.get(anchor, 1.0))))
                except (TypeError, ValueError):
                    resolved_heights.append(1.0)
            self._pane_heights = resolved_heights

    # ── Stacked-view pane context menu ────────────────────────────────────

    def _pane_index_of(self, ax) -> Optional[int]:
        """Return the index into self._pane_groups for the given Axes."""
        for i, pane in enumerate(self.panes):
            if pane is ax:
                # _pane_groups may include synthetic function panes appended
                # to the bottom; only the first len(_pane_groups) panes are
                # real-trace panes.
                if i < len(self._pane_groups):
                    return i
        return None

    def _func_pane_index_of(self, ax) -> Optional[int]:
        """Return the index into self._func_traces for a function pane."""
        n_groups = len(self._pane_groups)
        for i, pane in enumerate(self.panes):
            if pane is ax and n_groups <= i < n_groups + len(self._func_traces):
                return i - n_groups
        return None

    def _show_pane_context_menu(self, event) -> None:
        """Right-click menu for a stacked-view pane. Each pane holds exactly
        one signal — multi-signal grouping was removed in v3.1 because the
        "graph in a graph" feel was more confusing than useful. Pane order
        and sizing are still user-controlled via Move Up/Down / Alt-drag /
        divider drag; lock-Y, stats, and the function-pane workflow stay.
        """
        # Function pane? Show the slim function-pane menu and exit.
        f_idx = self._func_pane_index_of(event.inaxes)
        if f_idx is not None:
            self._show_function_pane_menu(event, f_idx)
            return

        pane_idx = self._pane_index_of(event.inaxes)
        if pane_idx is None:
            return
        group = self._pane_groups[pane_idx]
        anchor = (self.traces[group[0]].name
                  if group and group[0] in self.traces else None)
        menu = QMenu(self)

        # Cursor placement — top of menu so it's always reachable. lambda
        # accepts the leading 'checked' bool QAction.triggered always emits
        # (forgetting it sinks click_x and lands the cursor at x=0).
        click_x = event.xdata
        if click_x is not None:
            c1 = menu.addAction("Set Cursor 1 here")
            c1.triggered.connect(
                lambda checked=False, x=click_x: self.set_cursor(0, x))
            c2 = menu.addAction("Set Cursor 2 here")
            c2.triggered.connect(
                lambda checked=False, x=click_x: self.set_cursor(1, x))
            menu.addSeparator()

        up_act = menu.addAction("Move pane up")
        up_act.setEnabled(pane_idx > 0)
        up_act.triggered.connect(
            lambda checked=False, p=pane_idx: self._move_pane(p, -1))
        down_act = menu.addAction("Move pane down")
        down_act.setEnabled(pane_idx < len(self._pane_groups) - 1)
        down_act.triggered.connect(
            lambda checked=False, p=pane_idx: self._move_pane(p, +1))

        menu.addSeparator()

        if anchor:
            lock_act = menu.addAction("Lock Y")
            lock_act.setCheckable(True)
            lock_act.setChecked(self._pane_lock_y.get(anchor, False))
            lock_act.triggered.connect(
                lambda checked=False, a=anchor: self._toggle_pane_lock(a))
            reset_act = menu.addAction("Reset Y autoscale")
            reset_act.triggered.connect(
                lambda checked=False, a=anchor: self._reset_pane_y(a))

        menu.addSeparator()

        hide_act = menu.addAction("Hide this signal")
        hide_act.triggered.connect(
            lambda checked=False, p=pane_idx: self._hide_pane_signal(p))

        menu.addSeparator()

        func_act = menu.addAction("Plot function in new pane...")
        func_act.triggered.connect(self._dialog_plot_function)
        clear_func = menu.addAction("Clear function panes")
        clear_func.setEnabled(bool(self._func_traces))
        clear_func.triggered.connect(self._clear_function_panes)

        # Position menu where the click happened. event.guiEvent is a
        # QMouseEvent on Qt backends; fall back to canvas centre if missing.
        if hasattr(event, 'guiEvent') and event.guiEvent is not None:
            menu.exec(event.guiEvent.globalPosition().toPoint())
        else:
            menu.exec(self.canvas.mapToGlobal(QtCore.QPoint(0, 0)))

    def _hide_pane_signal(self, pane_idx: int) -> None:
        """Hide the trace owning this pane (≡ unticking it in the waveform list)."""
        if not (0 <= pane_idx < len(self._pane_groups)):
            return
        grp = self._pane_groups[pane_idx]
        if not grp:
            return
        trace_idx = grp[0]
        if trace_idx in self.traces:
            self.traces[trace_idx].visible = False
            for i in range(self.waveform_list.count()):
                item = self.waveform_list.item(i)
                if item and item.data(Qt.ItemDataRole.UserRole) == trace_idx:
                    self.update_list_item_appearance(item, trace_idx)
                    break
            self._schedule_refresh()

    # ── Pane menu action handlers ────────────────────────────────────────

    def _ensure_heights_match_groups(self) -> None:
        """Pad/truncate self._pane_heights so it parallels self._pane_groups.

        Sync-helpers and the move/reorder paths call this whenever the pane
        count changes. Missing entries default to 1.0 (equal); surplus are
        dropped from the tail.
        """
        n = len(self._pane_groups)
        if len(self._pane_heights) < n:
            self._pane_heights.extend([1.0] * (n - len(self._pane_heights)))
        elif len(self._pane_heights) > n:
            self._pane_heights = self._pane_heights[:n]

    def _move_pane(self, pane_idx: int, direction: int) -> None:
        new_idx = pane_idx + direction
        if 0 <= new_idx < len(self._pane_groups):
            self._pane_groups[pane_idx], self._pane_groups[new_idx] = (
                self._pane_groups[new_idx], self._pane_groups[pane_idx])
            # Heights move with the pane so the user's custom sizing follows.
            if (pane_idx < len(self._pane_heights)
                    and new_idx < len(self._pane_heights)):
                self._pane_heights[pane_idx], self._pane_heights[new_idx] = (
                    self._pane_heights[new_idx], self._pane_heights[pane_idx])
            self._ensure_heights_match_groups()
            # Pane reorder leaves the visible set (and thus the signature)
            # unchanged, so force the rebuild that re-lays-out the panes.
            self._force_full_refresh = True
            self.refresh_plot()

    def _toggle_pane_lock(self, anchor: str) -> None:
        if self._pane_lock_y.get(anchor):
            self._clear_pane_lock(anchor)
        else:
            self._snapshot_pane_for_lock(anchor)
            self._pane_lock_y[anchor] = True
        # Lock/unlock changes ylim handling, which the fast path skips —
        # force the full rebuild so _restore_view_state runs.
        self._force_full_refresh = True
        self.refresh_plot()

    def _reset_pane_y(self, anchor: str) -> None:
        self._clear_pane_lock(anchor)
        # Drop one-shot snapshot too so the upcoming refresh gets a fresh fit
        self._saved_pane_ylims.pop(anchor, None)
        # Need a real re-autoscale of this pane, which only the full path does.
        self._force_full_refresh = True
        self.refresh_plot()

    def _dialog_plot_function(self) -> None:
        text, ok = QInputDialog.getText(
            self, "Plot function in new pane",
            "Expression (e.g. v(in) - v(out)):")
        if ok and text:
            prev = self.func_input.text()
            try:
                self.func_input.setText(text)
                self.plot_function()
            finally:
                # Restore the right-panel input so the user's persistent
                # expression isn't clobbered by the menu-driven dialog.
                self.func_input.setText(prev)

    def _toggle_func_trace_visibility(self, f_idx: int) -> None:
        if not (0 <= f_idx < len(self._func_visible)):
            return
        self._func_visible[f_idx] = not self._func_visible[f_idx]
        label, _fx, _fy, color, *_ = self._func_traces[f_idx]
        for i in range(self.waveform_list.count()):
            it = self.waveform_list.item(i)
            if it and it.data(Qt.ItemDataRole.UserRole) == -(f_idx + 1):
                self._update_func_item_appearance(it, label, color, self._func_visible[f_idx])
                break
        self._schedule_refresh()

    def _populate_func_color_menu(self, menu: QMenu, f_idx: int) -> None:
        color_widget = QWidget()
        color_widget.setStyleSheet("background-color: #FFFFFF;")
        grid_layout = QGridLayout(color_widget)
        grid_layout.setSpacing(2)
        for i, c in enumerate(self.color_palette):
            btn = QPushButton()
            btn.setFixedSize(24, 24)
            btn.setStyleSheet(
                f"QPushButton{{background-color:{c};border:1px solid #E0E0E0;border-radius:2px;}}"
                f"QPushButton:hover{{border:2px solid #212121;}}")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(
                lambda checked=False, col=c, fi=f_idx, m=menu: (
                    self._change_func_color(fi, col), m.close()))
            grid_layout.addWidget(btn, i // 4, i % 4)
        wa = QWidgetAction(menu)
        wa.setDefaultWidget(color_widget)
        menu.addAction(wa)
        menu.addSeparator()
        more = menu.addAction("More...")
        more.triggered.connect(lambda: self._change_func_color_dialog(f_idx))

    def _change_func_color(self, f_idx: int, color: str) -> None:
        if not (0 <= f_idx < len(self._func_traces)):
            return
        label, fx, fy, _, thickness, style = self._func_traces[f_idx]
        self._func_traces[f_idx] = (label, fx, fy, color, thickness, style)
        for i in range(self.waveform_list.count()):
            it = self.waveform_list.item(i)
            if it and it.data(Qt.ItemDataRole.UserRole) == -(f_idx + 1):
                vis = f_idx < len(self._func_visible) and self._func_visible[f_idx]
                self._update_func_item_appearance(it, label, color, vis)
                break
        self.refresh_plot()

    def _change_func_color_dialog(self, f_idx: int) -> None:
        color = QColorDialog.getColor()
        if color.isValid():
            self._change_func_color(f_idx, color.name())

    def _change_func_thickness(self, f_idx: int, thickness: float) -> None:
        if not (0 <= f_idx < len(self._func_traces)):
            return
        label, fx, fy, color, _, style = self._func_traces[f_idx]
        self._func_traces[f_idx] = (label, fx, fy, color, thickness, style)
        self.refresh_plot()

    def _change_func_style(self, f_idx: int, style: str) -> None:
        if not (0 <= f_idx < len(self._func_traces)):
            return
        label, fx, fy, color, thickness, _ = self._func_traces[f_idx]
        self._func_traces[f_idx] = (label, fx, fy, color, thickness, style)
        self.refresh_plot()

    def _clear_function_panes(self) -> None:
        if self._func_traces:
            self._func_traces.clear()
            self._func_canonical.clear()
            self._func_visible.clear()
            self._sync_func_trace_list()
            self.refresh_plot()

    def _remove_function_pane(self, f_idx: int) -> None:
        if 0 <= f_idx < len(self._func_traces):
            del self._func_traces[f_idx]
            if f_idx < len(self._func_canonical):
                del self._func_canonical[f_idx]
            if f_idx < len(self._func_visible):
                del self._func_visible[f_idx]
            self._sync_func_trace_list()
            self.refresh_plot()

    def _show_function_pane_menu(self, event, f_idx: int) -> None:
        """Slim context menu for a function pane: cursor placement + remove."""
        menu = QMenu(self)
        click_x = event.xdata
        if click_x is not None:
            c1 = menu.addAction("Set Cursor 1 here")
            c1.triggered.connect(lambda checked=False, x=click_x: self.set_cursor(0, x))
            c2 = menu.addAction("Set Cursor 2 here")
            c2.triggered.connect(lambda checked=False, x=click_x: self.set_cursor(1, x))
            menu.addSeparator()
        label = self._func_traces[f_idx][0] if f_idx < len(self._func_traces) else ''
        rem = menu.addAction(f'Remove "{label}"')
        rem.triggered.connect(lambda i=f_idx: self._remove_function_pane(i))
        clear = menu.addAction("Clear all function panes")
        clear.setEnabled(bool(self._func_traces))
        clear.triggered.connect(self._clear_function_panes)
        if hasattr(event, 'guiEvent') and event.guiEvent is not None:
            menu.exec(event.guiEvent.globalPosition().toPoint())
        else:
            menu.exec(self.canvas.mapToGlobal(QtCore.QPoint(0, 0)))

    def _schedule_refresh(self) -> None:
        """Coalesce rapid visibility toggles into a single deferred refresh.

        Restarting the single-shot timer on each call means a burst of clicks
        collapses to one refresh_plot once the user stops. Used by every
        waveform/func-trace visibility toggle; direct refresh_plot calls
        (view-mode change, autoscale, etc.) cancel any pending tick via the
        stop() at the top of refresh_plot so they never double-rebuild.

        The window is mode-aware: a stacked toggle is a full pane rebuild
        (tens of ms, growing with pane count), so a wider window is needed to
        actually collapse a human-paced click burst — otherwise each click
        (>80ms apart) fires its own rebuild and the view stutters. Normal view
        toggles take the cheap incremental path, so they stay snappy at 80ms.
        The list item icon/text update synchronously either way, so clicks
        always feel instant; only the plot redraw is deferred.
        """
        self._refresh_timer.setInterval(
            STACKED_REFRESH_DEBOUNCE_MS if self.radio_stacked.isChecked()
            else REFRESH_DEBOUNCE_MS)
        self._refresh_timer.start()

    def _composition_signature(self, mode: str) -> tuple:
        """Fingerprint of everything that determines the pane/artist structure.

        When this is unchanged between two refreshes, the existing axes and
        Line2D objects are already correct (trace data is static after load),
        so refresh_plot can skip the full fig.clear() teardown.

        Per mode it captures only what is *structural* for that mode:

        - normal: one shared Axes regardless of how many traces are visible,
          so the visible set is deliberately EXCLUDED — visibility toggles are
          handled incrementally via set_visible. What IS structural: the
          analysis path (plot vs semilogx vs step), which traces use steps
          (changes the artist type), the visible function-overlay set, and
          whether the legend is shown.
        - stacked: one pane per visible trace, so the visible set + per-trace
          steps flag + visible func panes + stats overlay are all structural.
        - timing: rows are laid out by the visible set; threshold and spacing
          change every row's geometry.

        Changes the signature cannot see (pane reorder, divider resize, lock
        toggle, rename) set self._force_full_refresh instead.
        """
        vis_func = tuple(i for i in range(len(self._func_traces))
                         if i < len(self._func_visible) and self._func_visible[i])
        if mode == 'normal':
            steps = tuple(sorted(i for i, t in self.traces.items()
                                 if t.style == 'steps-post'))
            return ('normal', self._current_analysis_type, steps, vis_func,
                    self.legend_check.isChecked())
        if mode == 'stacked':
            vis = self.visible_traces
            return ('stacked',
                    tuple(t.index for t in vis),
                    tuple(t.style == 'steps-post' for t in vis),
                    vis_func,
                    self.stats_check.isChecked())
        # timing
        return ('timing',
                tuple(t.index for t in self.visible_traces),
                vis_func,
                self.threshold_spinbox.value(),
                self.vertical_spacing)

    def refresh_plot(self) -> None:
        # Cancel any pending debounced refresh — this call supersedes it, so a
        # queued timer tick must not fire a second redundant rebuild afterwards.
        self._refresh_timer.stop()
        force_full = self._force_full_refresh
        self._force_full_refresh = False

        next_mode = ('timing' if self.radio_timing.isChecked()
                     else 'stacked' if self.radio_stacked.isChecked()
                     else 'normal')
        new_sig = self._composition_signature(next_mode)

        # ── Incremental fast path ────────────────────────────────────────
        # Taken only when the pane composition is provably unchanged from what
        # is currently drawn: same mode, matching signature, live panes, and no
        # caller-forced full rebuild. Then the axes + lines already exist and
        # are correct, so we avoid fig.clear() entirely.
        if (not force_full and self.panes
                and self._drawn_signature is not None
                and new_sig == self._drawn_signature
                and self._current_view_mode == next_mode):
            if next_mode == 'normal':
                # Normal view keeps ALL prior limits/cursors; only line
                # visibility may differ. 0-visible needs the placeholder text,
                # so fall through to the full rebuild for that case.
                if self.visible_traces:
                    self._incremental_refresh_normal()
                    return
            else:
                # Stacked/timing: identical composition + static data means the
                # rendered figure is already correct. Just redraw.
                for ax in self.panes:
                    ax.grid(self.grid_check.isChecked())
                self.canvas.draw_idle()
                return

        # ── Full rebuild ─────────────────────────────────────────────────
        # Preserve zoom when autoscale is off.
        # Capture only when staying in the SAME ylim-meaningful mode: timing
        # uses [0..N] normalized space, stacked uses per-trace SI units —
        # restoring one across modes would clip signals or scramble panes.
        capture_state = (not self.autoscale_check.isChecked()
                         and self._current_view_mode == next_mode
                         and next_mode in ('normal', 'stacked')
                         and bool(self.panes))
        if capture_state:
            self._capture_view_state()

        # Re-enable constrained_layout before rebuilding: a previous stacked
        # refresh may have frozen it (engine off + pinned positions). The new
        # panes must be solved once by CL; _freeze_layout re-freezes at the end
        # for multi-pane stacked. Cheap single-pane modes stay CL-managed.
        self.fig.set_layout_engine('constrained')

        self._func_line = None  # fig.clear() below wipes all artists
        self.timing_annotations.clear()
        # Any in-progress cursor drag and the blit snapshot become invalid
        # once fig.clear() tears down the figure.  Reset them here so the
        # restore path that follows starts from a clean state.
        self._drag_cursor_idx = None
        self._blit_background = None
        self.fig.clear()
        # Hover-cache held references to the old Axes; invalidate before
        # _build_panes hands out fresh ones.
        self._last_hover_axes = None
        self._last_hover_anchor = None
        for t in self.traces.values():
            t.line_object = None
        # Set view mode BEFORE plot path runs so callees (update_timing_tick_colors,
        # legend handling, etc.) can branch on the new mode instead of the prior one.
        self._current_view_mode = next_mode
        if next_mode == 'timing':
            self._build_panes(1)
            self.plot_timing_diagram()
        elif next_mode == 'stacked':
            self.plot_stacked_diagram()
        else:
            if self.plot_type[0] == DataExtraction.AC_ANALYSIS:
                if self.plot_type[1] == 1:
                    self.on_push_decade()
                else:
                    self.on_push_ac()
            elif self.plot_type[0] == DataExtraction.TRANSIENT_ANALYSIS:
                self.on_push_trans()
            else:
                self.on_push_dc()
        if self.panes:
            for ax in self.panes:
                ax.grid(self.grid_check.isChecked())
            # Restore unconditionally: capture_state fills saved_pane_ylims
            # for the preserve-zoom path, AND lock-Y entries persist there
            # independently. _restore_view_state is a no-op when both are
            # empty, so calling it is always safe.
            self._restore_view_state()
            if self.legend_check.isChecked():
                self.position_legend()
        self._restore_cursors()
        # Record what we just drew so the next refresh can skip the rebuild if
        # nothing structural changed.
        self._drawn_signature = new_sig
        # Arm the free post-draw freeze for multi-pane stacked: the draw below
        # solves CL once, then _on_draw_event pins the result and drops the
        # engine so later draws skip the solver. Doing this inline (an extra
        # synchronous layout pass) is what made rapid toggling lag, so we let
        # the draw we already need do the work. Single-pane modes keep CL on —
        # cheap there, and it keeps tick-label margins adaptive.
        self._pending_freeze = (self._current_view_mode == 'stacked'
                                and len(self.panes) > 1)
        self.canvas.draw_idle()

    def _incremental_refresh_normal(self) -> None:
        """Update the shared-axes normal view in place — no fig.clear().

        Used when the composition signature is unchanged but trace visibility
        may have toggled. Reconciles each trace's Line2D (lazily creating one
        for a newly-visible trace, hiding rather than destroying one that was
        switched off), then re-fits/legends/cursors exactly as a full rebuild
        would, all without tearing the figure down.
        """
        for idx, t in self.traces.items():
            if t.visible:
                if t.line_object is None:
                    self._draw_normal_trace_line(t)
                else:
                    t.line_object.set_visible(True)
            elif t.line_object is not None:
                # Keep the artist for cheap re-show; just hide it.
                t.line_object.set_visible(False)

        # Re-fit only when autoscale is on; otherwise leave the user's zoom.
        # visible_only=True excludes the hidden (kept) lines from the bounds.
        if self.autoscale_check.isChecked():
            self.axes.relim(visible_only=True)
            self.axes.autoscale_view()

        first_visible = next((i for i in sorted(self.traces)
                              if self.traces[i].visible), None)
        if first_visible is not None:
            self.axes.set_ylabel('Voltage (V)' if first_visible < self.volts_length
                                 else 'Current (A)')

        if self.legend_check.isChecked():
            # legend() replaces any existing legend; ≥1 visible is guaranteed
            # by the caller, so position_legend always has a handle to draw.
            self.position_legend()

        # Cursor axvlines persist on the live axes (no fig.clear), so they need
        # no re-creation — but the sidebar readouts depend on the visible set,
        # which just changed, so refresh those.
        if any(p is not None for p in self.cursor_positions):
            self._refresh_cursor_readouts()

        self.canvas.draw_idle()

    def _on_draw_event(self, event) -> None:
        """Freeze the layout for FREE, right after a stacked rebuild's draw.

        The CL solver (~60% of a stacked draw's Python time) re-runs on every
        draw — even when pane geometry is unchanged. We can't avoid the one
        solve that the rebuild's own draw performs, but we CAN stop it repeating
        on subsequent zoom/pan/cursor draws: this fires after that draw has
        already solved CL, so we snapshot the EXACT solved positions (margins
        match CL by construction — rotated y-labels, stats titles included) and
        drop the engine. No extra layout pass, so rapid toggling stays cheap.
        """
        if not (self._pending_freeze and self.panes
                and self._current_view_mode == 'stacked'
                and len(self.panes) > 1):
            return
        self._pending_freeze = False
        positions = [ax.get_position().frozen() for ax in self.panes]
        self.fig.set_layout_engine('none')       # stop the solver
        for ax, pos in zip(self.panes, positions):
            ax.set_position(pos)                 # pin the CL-solved geometry

    def position_legend(self) -> None:
        if not (self.panes and self.legend_check.isChecked()):
            return
        # Stacked view: each pane already has a single-trace caption (set by
        # the stacked plot path), so a combined legend on the top pane would
        # be redundant noise.
        if self._current_view_mode == 'stacked':
            return
        handles, labels = [], []
        for idx in sorted(self.traces.keys()):
            t = self.traces[idx]
            if t.visible and t.line_object:
                handles.append(t.line_object)
                labels.append(t.name)
        if not handles:
            return
        ncol = max(1, min(4, len(handles)))
        legend = self.axes.legend(
            handles, labels,
            loc='best',
            ncol=ncol,
            frameon=True,
            fancybox=False,
            shadow=False,
            framealpha=0.95,
            columnspacing=1.2,
            handlelength=1.5,
        )
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('#E0E0E0')
        legend.get_frame().set_linewidth(1)

    def _get_transient_start_idx(self, time_data: "np.ndarray") -> int:
        """Return the index into time_data where the .tran start time begins, or 0."""
        if self._tran_start_time > 0:
            return int(np.searchsorted(time_data, self._tran_start_time))
        return 0

    def plot_timing_diagram(self) -> None:
        """Plot digital timing diagram with normalized trace heights."""
        self.timing_annotations.clear()

        if self.plot_type[0] != DataExtraction.TRANSIENT_ANALYSIS:
            self.axes.text(0.5, 0.5, 'Digital timing view is only\navailable for transient analysis.',
                           ha='center', va='center', transform=self.axes.transAxes,
                           color='#757575')
            self.axes.set_yticks([])
            self.axes.set_yticklabels([])
            return

        visible_indices = [t.index for t in self.visible_traces]
        if not visible_indices:
            self.axes.text(0.5, 0.5, 'Select a waveform to display',
                           ha='center', va='center', transform=self.axes.transAxes)
            self.axes.set_yticks([])
            self.axes.set_yticklabels([])
            return

        manual_threshold = (None if self.threshold_spinbox.value() == self.threshold_spinbox.minimum()
                            else self.threshold_spinbox.value())
        if manual_threshold is None:
            self.threshold_spinbox.setSpecialValueText("Auto (midpoint)")
        self.logic_thresholds = {}

        # Build local float arrays for all traces — never touch obj_dataext
        time_data = np.asarray(self.obj_dataext.x, dtype=float)
        y_data = {i: np.asarray(self.obj_dataext.y[i], dtype=float)
                  for i in range(len(self.obj_dataext.y))}

        if self.plot_type[0] == DataExtraction.TRANSIENT_ANALYSIS:
            start_idx = self._get_transient_start_idx(time_data)
            if 0 < start_idx < len(time_data):
                time_data = time_data[start_idx:]
                y_data = {i: arr[start_idx:] for i, arr in y_data.items()}

        # Each trace occupies exactly 1.0 normalized unit of y-space.
        # spacing = vertical_spacing (e.g. 1.2 → 20% gap between traces).
        # This guarantees uniform height for all signals regardless of voltage domain.
        spacing = self.vertical_spacing
        yticks, ylabels = [], []

        for rank, idx in enumerate(visible_indices[::-1]):
            raw_data = y_data[idx]

            # Safety clamp — guards against malformed simulation output where a
            # y array is shorter or longer than the time axis. Use a local
            # trace_time so time_data is never mutated across iterations.
            n = min(len(raw_data), len(time_data))
            raw_data = raw_data[:n]
            trace_time = time_data[:n]

            trace_vmin, trace_vmax = np.min(raw_data), np.max(raw_data)
            trace_unit = "V" if idx < self.obj_dataext.volts_length else "A"

            if trace_vmax - trace_vmin < 1e-10:
                # Constant (DC) signal — state indeterminate, park at 0.5.
                # No threshold line drawn (nothing to threshold against).
                logic_normalized = np.full(n, 0.5)
            else:
                # Per-trace threshold: midpoint of its own swing (CMOS VDD/2 convention).
                # Manual override applies the user's voltage, normalized into [0,1] for
                # this trace so the axhline always sits within the trace bounds.
                threshold = (manual_threshold if manual_threshold is not None
                             else (trace_vmin + trace_vmax) / 2.0)
                logic_normalized = np.where(raw_data > threshold, 1.0, 0.0)
                threshold_norm = float(np.clip(
                    (threshold - trace_vmin) / (trace_vmax - trace_vmin), 0.0, 1.0
                ))
                self.logic_thresholds[idx] = threshold_norm

            logic_offset = logic_normalized + rank * spacing

            t = self.traces[idx]
            line, = self.axes.step(trace_time, logic_offset, where="post",
                                   linewidth=t.thickness, color=t.color, label=t.name)
            t.line_object = line

            # y_center is always rank * spacing + 0.5 in normalized space.
            y_center = rank * spacing + 0.5
            yticks.append(y_center)
            ylabels.append(t.name)

            ann = []
            xform = self.axes.get_yaxis_transform()
            if trace_vmax - trace_vmin < 1e-10:
                ann.append(self.axes.text(
                    1.01, y_center,
                    f"DC: {_format_measurement(float(trace_vmax), trace_unit)}",
                    transform=xform, va='center', ha='left',
                    color=t.color, clip_on=False))
            else:
                ann.append(self.axes.text(
                    1.01, rank * spacing + 0.82,
                    f"H: {_format_measurement(float(trace_vmax), trace_unit)}",
                    transform=xform, va='center', ha='left',
                    color=t.color, clip_on=False))
                ann.append(self.axes.text(
                    1.01, rank * spacing + 0.18,
                    f"L: {_format_measurement(float(trace_vmin), trace_unit)}",
                    transform=xform, va='center', ha='left',
                    color=t.color, clip_on=False))
                freq = _detect_frequency(trace_time, logic_normalized)
                if freq is not None:
                    ann.append(self.axes.text(
                        1.01, y_center, _format_frequency(freq),
                        transform=xform, va='center', ha='left',
                        color=t.color, alpha=0.75, clip_on=False))
            self.timing_annotations[idx] = ann

        # Func traces as additional timing channels — normalized same as sim signals.
        n_sim = len(visible_indices)
        vis_func = [
            (f_idx, self._func_traces[f_idx])
            for f_idx in range(len(self._func_traces))
            if f_idx < len(self._func_visible) and self._func_visible[f_idx]
        ]
        xform = self.axes.get_yaxis_transform()
        for func_slot, (f_idx, (flabel, fx, fy, fcolor, fthickness, _fs)) in enumerate(vis_func):
            rank = n_sim + func_slot
            n_pts = min(len(fx), len(fy))
            if n_pts < 2:
                continue
            fy_arr = np.asarray(fy[:n_pts], dtype=float)
            fx_arr = np.asarray(fx[:n_pts], dtype=float)
            fmin, fmax = float(np.min(fy_arr)), float(np.max(fy_arr))
            y_center = rank * spacing + 0.5
            if fmax - fmin < 1e-10:
                logic = np.full(n_pts, 0.5)
                self.axes.text(1.01, y_center, f"DC: {fmax:.4g}",
                               transform=xform, va='center', ha='left',
                               color=fcolor, clip_on=False,
                               fontsize=max(7, LEGEND_FONT_SIZE - 1))
            else:
                logic = np.where(fy_arr > (fmin + fmax) / 2.0, 1.0, 0.0)
                self.axes.text(1.01, rank * spacing + 0.82, f"H: {fmax:.4g}",
                               transform=xform, va='center', ha='left',
                               color=fcolor, clip_on=False,
                               fontsize=max(7, LEGEND_FONT_SIZE - 1))
                self.axes.text(1.01, rank * spacing + 0.18, f"L: {fmin:.4g}",
                               transform=xform, va='center', ha='left',
                               color=fcolor, clip_on=False,
                               fontsize=max(7, LEGEND_FONT_SIZE - 1))
                freq = _detect_frequency(fx_arr, logic)
                if freq is not None:
                    self.axes.text(1.01, y_center, _format_frequency(freq),
                                   transform=xform, va='center', ha='left',
                                   color=fcolor, alpha=0.75, clip_on=False,
                                   fontsize=max(7, LEGEND_FONT_SIZE - 1))
            self.axes.step(fx_arr, logic + rank * spacing, where='post',
                           color=fcolor, linewidth=fthickness, label=flabel)
            yticks.append(y_center)
            ylabels.append(f'ƒ {flabel}')

        # Y-axis bounds: total count includes func trace slots.
        total_count = n_sim + len(vis_func)
        total_height = max(total_count - 1, 0) * spacing + 1.0
        margin = 0.15 * spacing
        self.axes.set_ylim(-margin, total_height + margin)
        self.axes.set_yticks(yticks)
        self.axes.set_yticklabels(ylabels)

        self.update_timing_tick_colors()
        self.set_time_axis_label(time_data)

        # Threshold lines for sim signals only.
        for rank, idx in enumerate(visible_indices[::-1]):
            if idx in self.logic_thresholds:
                self.axes.axhline(y=self.logic_thresholds[idx] + rank * spacing,
                                  color='red', linestyle=':', alpha=THRESHOLD_ALPHA, linewidth=0.8)

    def _render_pane_stats(self, ax, group: List[int],
                           x_arr: "np.ndarray") -> None:
        """Draw a min/max/p-p/RMS (+ freq for periodic transient) overlay.

        One text row per trace in the group, anchored top-right via axes
        fraction so it survives pane resize / zoom. Skipped silently when
        the group has no plottable traces.
        """
        if not group:
            return
        rows: List[str] = []
        for trace_idx in group:
            t = self.traces.get(trace_idx)
            if t is None:
                continue
            y_arr = np.asarray(self.obj_dataext.y[trace_idx], dtype=float)
            n_pts = min(len(y_arr), len(x_arr))
            if n_pts < 2:
                continue
            y = y_arr[:n_pts]
            x = x_arr[:n_pts]
            unit = 'V' if trace_idx < self.obj_dataext.volts_length else 'A'
            ymin = float(np.min(y))
            ymax = float(np.max(y))
            pp = ymax - ymin
            # Trapezoid integration is correct for adaptive-timestep ngspice
            # output where sample spacing is non-uniform (up to 200x ratio).
            # Simple mean/mean² gives wrong DC and RMS on such data.
            T = float(x[-1] - x[0])
            dc = float(_trapz(y, x) / T)
            rms_total_sq = float(_trapz(y * y, x) / T)
            # AC RMS = sqrt(RMS² - DC²) — signal amplitude without DC offset.
            rms_ac = float(np.sqrt(max(0.0, rms_total_sq - dc * dc)))
            # Drop min/max (already visible from Y-axis ticks) and name
            # (already the left title). Keep only the high-value stats.
            parts = [f"p-p={_format_measurement(pp, unit)}",
                     f"DC={_format_measurement(dc, unit)}",
                     f"RMS={_format_measurement(rms_ac, unit)}"]
            if self._current_analysis_type == 'transient' and pp > 1e-12:
                mid = (ymin + ymax) / 2.0
                logic = np.where(y > mid, 1.0, 0.0)
                freq = _detect_frequency(x, logic)
                if freq is not None:
                    parts.append(f"f={_format_frequency(freq)}")
            rows.append("  ".join(parts))
        if not rows:
            return
        # No bbox — stats are in the title margin above the spine, no waveform
        # behind them, so a white background box is unnecessary and its padding
        # would straddle the spine into the axes area.
        ax.set_title("\n".join(rows), loc='right',
                     fontsize=max(7, LEGEND_FONT_SIZE - 1),
                     color='#444444', pad=4)

    def _render_func_pane_stats(self, ax, fx: "np.ndarray", fy: "np.ndarray") -> None:
        x = np.asarray(fx, dtype=float)
        y = np.asarray(fy, dtype=float)
        n = min(len(x), len(y))
        if n < 2:
            return
        x, y = x[:n], y[:n]
        ymin, ymax = float(np.min(y)), float(np.max(y))
        pp = ymax - ymin
        T = float(x[-1] - x[0])
        if T <= 0:
            return
        dc = float(_trapz(y, x) / T)
        rms_ac = float(np.sqrt(max(0.0, float(_trapz(y * y, x) / T) - dc * dc)))

        def _fmt(v: float) -> str:
            a = abs(v)
            if a >= 1:      return f"{v:.3g}"
            if a >= 1e-3:   return f"{v * 1e3:.3g}m"
            if a >= 1e-6:   return f"{v * 1e6:.3g}µ"
            if a >= 1e-9:   return f"{v * 1e9:.3g}n"
            return f"{v:.3g}"

        parts = [f"p-p={_fmt(pp)}", f"DC={_fmt(dc)}", f"RMS={_fmt(rms_ac)}"]
        if self._current_analysis_type == 'transient' and pp > 1e-12:
            freq = _detect_frequency(x, np.where(y > (ymin + ymax) / 2.0, 1.0, 0.0))
            if freq is not None:
                parts.append(f"f={_format_frequency(freq)}")
        ax.set_title("  ".join(parts), loc='right',
                     fontsize=max(7, LEGEND_FONT_SIZE - 1),
                     color='#444444', pad=4)

    def plot_stacked_diagram(self) -> None:
        """Stacked-pane view: one pane per visible trace + one per func trace.

        Each entry in self._pane_groups is a single-element list containing
        the trace.index of that pane's signal. Function panes follow at the
        bottom. Heights, lock-Y, stats, and pane-name anchor live on the
        first (and only) trace in the group.

        Function traces (set by plot_function while stacked is active) tail
        at the bottom as one extra pane each.
        """
        # Bring _pane_groups in line with the current visibility set
        self._sync_pane_groups_to_visible()

        if not self._pane_groups and not self._func_traces:
            self._build_panes(1)
            self.axes.text(0.5, 0.5, 'Select a waveform to display',
                           ha='center', va='center',
                           transform=self.axes.transAxes)
            self.axes.set_yticks([])
            self.axes.set_yticklabels([])
            return

        is_transient = self.plot_type[0] == DataExtraction.TRANSIENT_ANALYSIS
        is_ac        = self.plot_type[0] == DataExtraction.AC_ANALYSIS
        is_log       = is_ac and self.plot_type[1] == 1
        is_dc        = self.plot_type[0] == DataExtraction.DC_ANALYSIS

        x_data = np.asarray(self.obj_dataext.x, dtype=float)
        if is_transient:
            start_idx = self._get_transient_start_idx(x_data)
            if 0 < start_idx < len(x_data):
                x_data = x_data[start_idx:]

        n_groups = len(self._pane_groups)
        # Only visible func traces get their own pane.
        _vis_func = [i for i in range(len(self._func_traces))
                     if i < len(self._func_visible) and self._func_visible[i]]
        n_funcs = len(_vis_func)
        n = n_groups + n_funcs
        self._build_panes(n)

        for pane_idx, group in enumerate(self._pane_groups):
            ax = self.panes[pane_idx]
            if not group or group[0] not in self.traces:
                ax.set_ylim(-1, 1)
                if pane_idx < n - 1:
                    ax.tick_params(labelbottom=False)
                continue
            t = self.traces[group[0]]
            raw_y = np.asarray(self.obj_dataext.y[t.index], dtype=float)
            n_pts = min(len(raw_y), len(x_data))
            if n_pts == 0:
                ax.set_ylim(-1, 1)
                if pane_idx < n - 1:
                    ax.tick_params(labelbottom=False)
                continue
            y = raw_y[:n_pts]
            trace_x = x_data[:n_pts]

            plot_style = '-' if t.style == 'steps-post' else t.style
            if is_log:
                line, = ax.semilogx(trace_x, y, color=t.color,
                                    linewidth=t.thickness,
                                    linestyle=plot_style)
            elif t.style == 'steps-post' and (is_transient or is_dc):
                line, = ax.step(trace_x, y, where='post', color=t.color,
                                linewidth=t.thickness)
            else:
                line, = ax.plot(trace_x, y, color=t.color,
                                linewidth=t.thickness, linestyle=plot_style)
            t.line_object = line

            ax.set_title(t.name, loc='left', color=t.color,
                         fontsize=LEGEND_FONT_SIZE, fontweight='bold', pad=3)

            is_voltage = t.index < self.obj_dataext.volts_length
            unit = 'V' if is_voltage else 'A'
            ax.set_ylabel(unit, rotation=0, labelpad=8, va='center')
            ax.yaxis.set_major_formatter(FuncFormatter(
                lambda v, _pos, _u=unit: _format_measurement(float(v), _u)))

            ymin = float(np.min(y))
            ymax = float(np.max(y))
            if abs(ymax - ymin) < 1e-12:
                center = (ymin + ymax) / 2.0
                ax.set_ylim(center - 1.0, center + 1.0)
            else:
                margin = 0.1 * (ymax - ymin)
                ax.set_ylim(ymin - margin, ymax + margin)

            if pane_idx < n - 1:
                ax.tick_params(labelbottom=False)
                # Visible separator hint: gray bottom spine reads as a row
                # divider in the strip chart.
                ax.spines['bottom'].set_color('#BDBDBD')
                ax.spines['bottom'].set_linewidth(1.0)

            if self.stats_check.isChecked():
                self._render_pane_stats(ax, group, x_data)

        # Trailing function-trace panes. Only visible func traces get a pane.
        # _vis_func holds the original indices into _func_traces so labels
        # and colours stay correct after partial hide/show.
        for pane_slot, f_idx in enumerate(_vis_func):
            pane_offset = n_groups + pane_slot
            if pane_offset >= len(self.panes):
                break
            label, fx, fy, color, thickness, style = self._func_traces[f_idx]
            ax = self.panes[pane_offset]
            plot_style = '-' if style == 'steps-post' else style
            if style == 'steps-post':
                ax.step(fx, fy, where='post', color=color, linewidth=thickness)
            else:
                ax.plot(fx, fy, color=color, linewidth=thickness, linestyle=plot_style)
            ax.set_title(label, loc='left', color=color,
                         fontsize=LEGEND_FONT_SIZE, fontweight='bold', pad=3)
            if len(fy):
                ymin = float(np.min(fy))
                ymax = float(np.max(fy))
                if abs(ymax - ymin) < 1e-12:
                    center = (ymin + ymax) / 2.0
                    ax.set_ylim(center - 1.0, center + 1.0)
                else:
                    margin = 0.1 * (ymax - ymin)
                    ax.set_ylim(ymin - margin, ymax + margin)
            if pane_offset < n - 1:
                ax.tick_params(labelbottom=False)
                ax.spines['bottom'].set_color('#BDBDBD')
                ax.spines['bottom'].set_linewidth(1.0)

            if self.stats_check.isChecked():
                self._render_func_pane_stats(ax, fx, fy)

        # Bottom-pane X label / formatter. Existing helpers already target
        # self.panes[-1], so the multi-pane case is free.
        if is_ac:
            self.set_freq_axis_label()
        elif is_transient:
            self.set_time_axis_label(x_data)
        else:  # DC sweep
            self._reset_x_axis_scaling()
            self.panes[-1].set_xlabel('Voltage Sweep (V)')


    def _reset_x_axis_scaling(self) -> None:
        """Drop any SI-unit formatter on the X axis (identity tick labels).

        Used when the X axis no longer represents time/frequency — e.g. the
        Lissajous case in plot_function where X becomes a voltage trace.
        """
        self._x_scale = 1.0
        self._x_unit = ''
        for ax in self.panes:
            ax.xaxis.set_major_formatter(ScalarFormatter())

    def _apply_x_axis_scaling(self, scale: float, unit: str,
                              label_prefix: str) -> None:
        """Display-only X-axis scaling via FuncFormatter.

        Line data and xlim stay in raw SI units; tick labels show raw * scale.
        This keeps event.xdata, cursor positions, and stored data coherent and
        eliminates the previous mutate-on-every-refresh xdata bug. The label
        is only attached to the bottom-most pane so stacked panes share one
        unified X axis caption.
        """
        self._x_scale = scale
        self._x_unit = unit
        fmt = FuncFormatter(lambda v, _pos, _s=scale: f"{v * _s:g}")
        for ax in self.panes:
            ax.xaxis.set_major_formatter(fmt)
            ax.set_xlabel('')
        if self.panes:
            self.panes[-1].set_xlabel(f'{label_prefix} ({unit})')

    def set_time_axis_label(self, time_data: Optional["np.ndarray"] = None) -> None:
        if not self.panes or not hasattr(self.obj_dataext, 'x'):
            return
        if time_data is None:
            time_data = np.asarray(self.obj_dataext.x, dtype=float)
        if len(time_data) < 2:
            self._x_scale, self._x_unit = 1.0, 's'
            self.panes[-1].set_xlabel('Time (s)')
            return
        scale, unit = self._get_time_scale_and_unit(time_data)
        self._apply_x_axis_scaling(scale, unit, 'Time')
        self.axes.set_xlim(float(time_data[0]), float(time_data[-1]))

    def on_threshold_changed(self, value: float) -> None:
        if self.radio_timing.isChecked():
            self._controls_timer.start()

    def on_spacing_changed(self, value: int) -> None:
        self.vertical_spacing = value / 100.0
        self.spacing_label.setText(f"{self.vertical_spacing:.1f}x")
        if self.radio_timing.isChecked():
            self._controls_timer.start()

    def _find_nearest_cursor(self, event) -> Optional[int]:
        """Return cursor index if the click is within 8px of an existing cursor.

        Reads from cursor_positions (raw units) rather than per-pane line
        xdata — keeps multi-pane hit-testing simple and pane-agnostic, since
        all panes share the same X under sharex.
        """
        if (not self.cursor_positions
                or not self.panes
                or event.xdata is None):
            return None
        xlim = self.axes.get_xlim()
        width_px = self.axes.get_window_extent().width
        if width_px == 0:
            return None
        threshold = 8 * (xlim[1] - xlim[0]) / width_px
        for i, x_pos in enumerate(self.cursor_positions):
            if x_pos is None:
                continue
            if abs(event.xdata - x_pos) < threshold:
                return i
        return None

    def _update_cursor_position(self, cursor_num: int, x_pos: float) -> None:
        """Move an existing cursor line without recreating it (fast drag path).

        Skips the per-trace interpolated readout during active drag — that
        was O(N_traces × log(samples)) per mouse-move event and produced
        visible lag with 5+ traces. The full readout is rebuilt on release
        via on_canvas_release → set_cursor-equivalent retouch.
        """
        if (cursor_num >= len(self.cursor_lines)
                or not self.cursor_lines[cursor_num]):
            self.set_cursor(cursor_num, x_pos)
            return
        for line in self.cursor_lines[cursor_num]:
            if line is not None:
                line.set_xdata([x_pos, x_pos])
        scale = self._current_axis_scale()
        self.cursor_positions[cursor_num] = x_pos
        # Lightweight X-only update during drag — full per-signal Y readout
        # is deferred to on_canvas_release to keep drag smooth.
        unit_str = (self._x_unit or '').strip()
        label = self.cursor1_label if cursor_num == 0 else self.cursor2_label
        _c_color = '#e53935' if cursor_num == 0 else '#1976d2'
        label.setText(
            f'<span style="font-weight:700;color:{_c_color}">C{cursor_num + 1}</span>'
            f'<span style="color:#999"> @ </span>'
            f'<span style="font-weight:600;color:#333">{x_pos * scale:.4g} {unit_str}</span>'
        )
        two_cursors = (len(self.cursor_positions) >= 2
                       and all(p is not None for p in self.cursor_positions[:2]))
        if two_cursors:
            delta_raw = abs(self.cursor_positions[1] - self.cursor_positions[0])
            self.delta_label.setText(
                f'<span style="font-weight:700;color:#e65100">ΔX</span>'
                f' <span style="font-family:monospace;font-weight:600;color:#333">{delta_raw * scale:.4g} {unit_str}</span>'
            )
            self._update_measure_label(delta_raw, scale)
        # Blit path: restore static background, draw only cursor lines, blit.
        # Falls back to draw_idle() if snapshot is stale/missing.
        if self._blit_background is not None:
            self.canvas.restore_region(self._blit_background)
            for pane_lines in self.cursor_lines:
                for pane_idx, line in enumerate(pane_lines):
                    if line is not None and pane_idx < len(self.panes):
                        self.panes[pane_idx].draw_artist(line)
            self.canvas.blit(self.fig.bbox)
        else:
            self.canvas.draw_idle()

    def _get_time_scale_and_unit(self, time_data: Optional["np.ndarray"] = None) -> Tuple[float, str]:
        """Single source of truth for time-axis unit selection.

        All callers (set_time_axis_label, _current_time_scale, set_cursor)
        derive their scale factor from here — ensures they can never diverge.
        time_data defaults to obj_dataext.x; pass a trimmed slice when a
        subset of the axis is being displayed (e.g. transient start offset).
        """
        if time_data is None:
            time_data = np.asarray(self.obj_dataext.x, dtype=float)
        time_span = abs(time_data[-1] - time_data[0]) if len(time_data) > 1 else 0.0
        if time_span == 0:                         return 1.0,  's'
        if time_span < TIME_UNIT_THRESHOLD_PS:     return 1e12, 'ps'
        if time_span < TIME_UNIT_THRESHOLD_NS:     return 1e9,  'ns'
        if time_span < TIME_UNIT_THRESHOLD_US:     return 1e6,  'µs'
        if time_span < TIME_UNIT_THRESHOLD_MS:     return 1e3,  'ms'
        return 1.0, 's'

    def _current_time_scale(self) -> float:
        return self._get_time_scale_and_unit()[0]

    def _current_axis_scale(self) -> float:
        if self._current_analysis_type in ('ac_log', 'ac_linear'):
            return self._get_freq_scale_and_unit()[0]
        return self._get_time_scale_and_unit()[0]

    def _update_measure_label(self, delta_original: float, scale: float) -> None:
        if self._current_analysis_type in ('ac_log', 'ac_linear'):
            _, unit = self._get_freq_scale_and_unit()
            self.measure_label.setText(f"ΔF: {delta_original * scale:.6g} {unit}")
        else:
            if delta_original > 0:
                self.measure_label.setText(f"Freq: {1.0 / delta_original:.6g} Hz")

    def _cursor_visible_key(self) -> tuple:
        sim_key = tuple(t.index for t in self.visible_traces)
        func_key = tuple(i for i, v in enumerate(self._func_visible) if v)
        return (sim_key, func_key)

    def _get_cursor_interp(self, x_pos: float, cursor_num: int) -> Optional[Dict]:
        if cursor_num >= len(self._cursor_interp_cache):
            return None
        entry = self._cursor_interp_cache[cursor_num]
        if entry is None:
            return None
        if entry['x_pos'] == x_pos and entry['visible_key'] == self._cursor_visible_key():
            return entry
        return None

    def _set_cursor_interp(self, x_pos: float, cursor_num: int,
                           sim_y: Dict, func_y: Dict) -> None:
        while len(self._cursor_interp_cache) <= cursor_num:
            self._cursor_interp_cache.append(None)
        self._cursor_interp_cache[cursor_num] = {
            'x_pos': x_pos,
            'visible_key': self._cursor_visible_key(),
            'sim_y': sim_y,
            'func_y': func_y,
        }

    def _update_cursor_panel(self, cursor_num: int, x_pos: Optional[float]) -> None:
        """Rebuild the sidebar cursor label with X position + per-signal Y values.

        Shows one line per visible trace so users can read the exact value at
        the cursor for every pane simultaneously — especially useful in stacked
        view where each pane has its own Y scale.
        """
        label_widget = self.cursor1_label if cursor_num == 0 else self.cursor2_label
        color = '#e53935' if cursor_num == 0 else '#1976d2'
        c_label = f'C{cursor_num + 1}'

        if x_pos is None or not hasattr(self.obj_dataext, 'x'):
            label_widget.setText(
                f'<b style="color:{color}">{c_label}</b>'
                f'  <span style="color:#aaa">not set</span>'
            )
            return

        x_full = np.asarray(self.obj_dataext.x, dtype=float)
        scale = self._x_scale or 1.0
        unit_label = (self._x_unit or '').strip()
        x_display = f"{x_pos * scale:.4g}"
        if unit_label:
            x_display += f" {unit_label}"

        html = (f'<span style="font-weight:700;color:{color}">{c_label}</span>'
                f'<span style="color:#999"> @ </span>'
                f'<span style="font-weight:600;color:#333">{x_display}</span>')

        visible = self.visible_traces
        rows = ''

        cached = self._get_cursor_interp(x_pos, cursor_num)
        if cached:
            sim_y = cached['sim_y']
            func_y = cached['func_y']
        else:
            sim_y: Dict[int, float] = {}
            if visible and len(x_full) >= 2:
                for t in visible:
                    try:
                        y_arr = np.asarray(self.obj_dataext.y[t.index], dtype=float)
                    except (IndexError, TypeError):
                        continue
                    n_pts = min(len(y_arr), len(x_full))
                    if n_pts < 2:
                        continue
                    sim_y[t.index] = float(np.interp(x_pos, x_full[:n_pts], y_arr[:n_pts]))
            func_y: Dict[int, float] = {}
            for f_idx, (_, fx, fy, _fcolor, *_) in enumerate(self._func_traces):
                if not (f_idx < len(self._func_visible) and self._func_visible[f_idx]):
                    continue
                n_pts = min(len(fx), len(fy))
                if n_pts < 2:
                    continue
                try:
                    func_y[f_idx] = float(np.interp(x_pos, fx[:n_pts], fy[:n_pts]))
                except Exception:
                    continue
            self._set_cursor_interp(x_pos, cursor_num, sim_y, func_y)

        for t in visible:
            y_val = sim_y.get(t.index)
            if y_val is None:
                continue
            unit = 'V' if t.index < self.obj_dataext.volts_length else 'A'
            value = _format_measurement(y_val, unit)
            rows += (f'<tr>'
                     f'<td style="color:#555;padding-right:8px">{t.name}</td>'
                     f'<td style="font-family:monospace;font-weight:600;color:#333">{value}</td>'
                     f'</tr>')

        for f_idx, (flabel, _fx, _fy, fcolor, *_) in enumerate(self._func_traces):
            if not (f_idx < len(self._func_visible) and self._func_visible[f_idx]):
                continue
            y_val = func_y.get(f_idx)
            if y_val is None:
                continue
            short = flabel if len(flabel) <= 20 else flabel[:18] + '…'
            rows += (f'<tr>'
                     f'<td style="color:{fcolor};padding-right:8px">'
                     f'ƒ {short}</td>'
                     f'<td style="font-family:monospace;font-weight:600;color:#333">{y_val:.4g}</td>'
                     f'</tr>')

        if rows:
            html += f'<table style="margin-top:4px;margin-bottom:4px">{rows}</table>'
        label_widget.setText(html)
        label_widget.updateGeometry()

    def _format_cursor_readout(self, x_pos: float) -> str:
        """Single-line "X | sig1=Y1 | sig2=Y2 …" readout at the cursor X.

        Uses np.interp against the raw simulation x/y arrays so the values
        are accurate even between sample points. SI prefix per signal via
        _format_measurement. Truncated to keep the status bar one line.
        """
        visible = self.visible_traces
        if not visible or not hasattr(self.obj_dataext, 'x'):
            return ''
        x_full = np.asarray(self.obj_dataext.x, dtype=float)
        if len(x_full) < 2:
            return ''
        scale = self._x_scale or 1.0
        unit_label = self._x_unit or ''
        parts = [f"X={x_pos * scale:.4g} {unit_label}".rstrip()]

        # Cursor readout is always for cursor 0 (single-cursor mode).
        cached = self._get_cursor_interp(x_pos, 0)
        if cached:
            sim_y = cached['sim_y']
            func_y = cached['func_y']
        else:
            sim_y = {}
            for t in visible:
                try:
                    y_arr = np.asarray(self.obj_dataext.y[t.index], dtype=float)
                except (IndexError, TypeError):
                    continue
                n_pts = min(len(y_arr), len(x_full))
                if n_pts < 2:
                    continue
                try:
                    sim_y[t.index] = float(np.interp(x_pos, x_full[:n_pts], y_arr[:n_pts]))
                except Exception:
                    continue
            func_y = {}
            for f_idx, (_, fx, fy, _color, *_) in enumerate(self._func_traces):
                if not (f_idx < len(self._func_visible) and self._func_visible[f_idx]):
                    continue
                n_pts = min(len(fx), len(fy))
                if n_pts < 2:
                    continue
                try:
                    func_y[f_idx] = float(np.interp(x_pos, fx[:n_pts], fy[:n_pts]))
                except Exception:
                    continue
            self._set_cursor_interp(x_pos, 0, sim_y, func_y)

        for t in visible:
            y_val = sim_y.get(t.index)
            if y_val is None:
                continue
            is_voltage = t.index < self.obj_dataext.volts_length
            unit = 'V' if is_voltage else 'A'
            parts.append(f"{t.name}={_format_measurement(y_val, unit)}")

        for f_idx, (flabel, _fx, _fy, _color, *_) in enumerate(self._func_traces):
            if not (f_idx < len(self._func_visible) and self._func_visible[f_idx]):
                continue
            y_val = func_y.get(f_idx)
            if y_val is None:
                continue
            short = flabel if len(flabel) <= 14 else flabel[:12] + '…'
            parts.append(f"ƒ({short})={y_val:.4g}")

        # Limit total length so it never wraps the status bar
        readout = " | ".join(parts)
        return readout if len(readout) < 220 else readout[:217] + '…'

    def _get_freq_scale_and_unit(self, freq_data: Optional["np.ndarray"] = None) -> Tuple[float, str]:
        if freq_data is None:
            freq_data = np.asarray(self.obj_dataext.x, dtype=float)
        freq_max = np.max(np.abs(freq_data)) if len(freq_data) > 0 else 0.0
        if freq_max == 0:                                return 1.0,   'Hz'
        if freq_max >= FREQ_UNIT_THRESHOLD_GHZ:          return 1e-9,  'GHz'
        if freq_max >= FREQ_UNIT_THRESHOLD_MHZ:          return 1e-6,  'MHz'
        if freq_max >= FREQ_UNIT_THRESHOLD_KHZ:          return 1e-3,  'kHz'
        return 1.0, 'Hz'

    def set_freq_axis_label(self, freq_data: Optional["np.ndarray"] = None) -> None:
        if not self.panes or not hasattr(self.obj_dataext, 'x'):
            return
        if freq_data is None:
            freq_data = np.asarray(self.obj_dataext.x, dtype=float)
        if len(freq_data) < 2:
            self._x_scale, self._x_unit = 1.0, 'Hz'
            self.panes[-1].set_xlabel('Frequency (Hz)')
            return
        scale, unit = self._get_freq_scale_and_unit(freq_data)
        self._apply_x_axis_scaling(scale, unit, 'Frequency')
        self.axes.set_xlim(float(freq_data[0]), float(freq_data[-1]))

    def _begin_cursor_blit(self) -> None:
        """Save static background for blit-based cursor drag.

        Marks all cursor lines animated=True so canvas.draw() excludes them,
        then snapshots the result. Each move then does restore+draw_artist+blit
        — O(cursor lines) instead of O(full figure). Called once at drag start.
        """
        for pane_lines in self.cursor_lines:
            for line in pane_lines:
                if line is not None:
                    line.set_animated(True)
        self.canvas.draw()
        self._blit_background = self.canvas.copy_from_bbox(self.fig.bbox)

    def _end_cursor_blit(self) -> None:
        """Restore normal render path after cursor drag ends."""
        if self._blit_background is None:
            return
        self._blit_background = None
        for pane_lines in self.cursor_lines:
            for line in pane_lines:
                if line is not None:
                    line.set_animated(False)
        self.canvas.draw_idle()

    def on_canvas_click(self, event) -> None:
        if not self.panes:
            return
        if self.nav_toolbar.mode:
            return

        # Divider drag — pressed in the gap between two panes (event.inaxes
        # is None there, so handle it before the inaxes guard below).
        if (self._current_view_mode == 'stacked'
                and event.button == 1
                and event.inaxes is None):
            div = self._divider_under_mouse(event)
            if div is not None:
                self._start_divider_drag(div, event)
                return

        if event.inaxes not in self.panes:
            return

        modifier = (event.key or '').lower()

        # Alt + left-click in stacked: start a pane reorder drag. Finish on
        # release over the destination pane.
        if (self._current_view_mode == 'stacked'
                and event.button == 1
                and 'alt' in modifier):
            idx = self._pane_index_of(event.inaxes)
            if idx is not None:
                self._start_pane_drag(idx)
                return

        # Right-click in stacked mode opens the per-pane context menu. The
        # menu's "Set Cursor N here" items keep cursor placement available
        # to laptop users who don't have a middle mouse button.
        if event.button == 3 and self._current_view_mode == 'stacked':
            self._show_pane_context_menu(event)
            return

        x_target = event.xdata

        near = self._find_nearest_cursor(event)
        if event.button == 1:
            if near is not None:
                self._drag_cursor_idx = near
                self._begin_cursor_blit()
            else:
                self._drag_cursor_idx = None
                self.set_cursor(0, x_target)
        elif event.button == 2:  # middle-click: cursor 2
            self._drag_cursor_idx = None
            self.set_cursor(1, x_target)
        elif event.button == 3:  # right-click (non-stacked): cursor 2
            self._drag_cursor_idx = None
            self.set_cursor(1, x_target)

    def on_canvas_release(self, event) -> None:
        # If a cursor was being dragged, recompute the full per-signal
        # readout now (skipped during motion for performance).
        had_cursor_drag = self._drag_cursor_idx is not None
        last_cursor_idx = self._drag_cursor_idx
        self._drag_cursor_idx = None
        if had_cursor_drag:
            self._end_cursor_blit()
        if had_cursor_drag and last_cursor_idx is not None:
            if last_cursor_idx < len(self.cursor_positions):
                x_pos = self.cursor_positions[last_cursor_idx]
                if x_pos is not None:
                    # Full per-signal Y readout now that drag is done.
                    self._update_cursor_panel(last_cursor_idx, x_pos)
                    two = (len(self.cursor_positions) >= 2
                           and all(p is not None for p in self.cursor_positions[:2]))
                    if not two:
                        self.measure_label.setText(
                            self._format_cursor_readout(x_pos))
        if self._divider_drag is not None:
            self._finish_divider_drag()
        if self._pane_drag is not None:
            self._finish_pane_drag(event)

    def set_cursor(self, cursor_num: int, x_pos: float) -> None:
        # x_pos is in raw SI units (matches line data + xlim post-formatter).
        # Stored positions are raw; displayed labels apply _x_scale at format time.
        # In multi-pane mode the cursor draws one axvline per pane so the
        # vertical line spans the full stack.
        if not self.panes:
            return
        if x_pos is None:
            return
        self._end_cursor_blit()  # no-op if not blitting; clears stale snapshot
        scale = self._current_axis_scale()

        # Pad lists so cursor_num is a valid index. Without this, calling
        # set_cursor(1, x) on empty cursor_lines would silently append at
        # slot 0 → "cursor 2" lands in cursor 1's slot.
        while len(self.cursor_lines) <= cursor_num:
            self.cursor_lines.append([])
            self.cursor_positions.append(None)

        # Tear down old lines for this cursor (every pane) before re-drawing
        for old in self.cursor_lines[cursor_num]:
            if old is None:
                continue
            try:
                old.remove()
            except ValueError:
                pass  # already cleared by fig.clear()

        color = 'red' if cursor_num == 0 else 'blue'
        new_lines: List[Optional[Line2D]] = [
            ax.axvline(x=x_pos, color=color, linestyle='--', alpha=CURSOR_ALPHA)
            for ax in self.panes
        ]
        self.cursor_lines[cursor_num] = new_lines
        self.cursor_positions[cursor_num] = x_pos

        self._update_cursor_panel(cursor_num, x_pos)

        two_cursors = (len(self.cursor_positions) >= 2
                       and all(p is not None for p in self.cursor_positions[:2]))
        if two_cursors:
            delta_raw = abs(self.cursor_positions[1] - self.cursor_positions[0])
            self.delta_label.setText(
                f'<span style="font-weight:700;color:#e65100">ΔX</span>'
                f' <span style="font-family:monospace;font-weight:600;color:#333">{delta_raw * scale:.4g} {(self._x_unit or "").strip()}</span>'
            )
            self._update_measure_label(delta_raw, scale)
        else:
            self.measure_label.setText(self._format_cursor_readout(x_pos))
        self.canvas.draw()

    def clear_cursors(self) -> None:
        for pane_lines in self.cursor_lines:
            for line in pane_lines:
                if line is None:
                    continue
                try:
                    line.remove()
                except ValueError:
                    pass  # already removed by fig.clear()
        self.cursor_lines.clear()
        self.cursor_positions.clear()
        self.cursor1_label.setText('<b style="color:#e53935">C1</b>  <span style="color:#aaa">not set</span>')
        self.cursor2_label.setText('<b style="color:#1976d2">C2</b>  <span style="color:#aaa">not set</span>')
        self.delta_label.setText('<b style="color:#e65100">ΔX</b>  <span style="color:#aaa">—</span>')
        self.measure_label.setText("")
        self.canvas.draw()

    def _restore_cursors(self) -> None:
        """Re-create cursor axvlines after fig.clear(), using stored positions.

        Positions are raw SI units and match the current xlim directly — no
        scale factor applied at draw time (formatter handles tick display).
        Each cursor draws one axvline per pane so the line spans the full stack.
        """
        if not self.panes or not self.cursor_positions:
            return
        colors = ['red', 'blue']
        rebuilt: List[List[Optional[Line2D]]] = []
        for i, x_pos in enumerate(self.cursor_positions):
            if x_pos is None:
                rebuilt.append([])
                continue
            color = colors[i] if i < len(colors) else 'green'
            pane_lines: List[Optional[Line2D]] = [
                ax.axvline(x=x_pos, color=color,
                           linestyle='--', alpha=CURSOR_ALPHA)
                for ax in self.panes
            ]
            rebuilt.append(pane_lines)
        self.cursor_lines = rebuilt
        if rebuilt:
            logger.debug("Restored %d cursor(s) after plot refresh", len(rebuilt))
        self._refresh_cursor_readouts()

    def _refresh_cursor_readouts(self) -> None:
        """Recompute the sidebar cursor labels (C1/C2/ΔX/measure) in place.

        Does interpolated Y lookups against the current visible-trace set, so
        it must run after panes and _x_scale/_x_unit are set up. Split out of
        _restore_cursors so the incremental refresh — where the axvlines
        already exist and must NOT be re-created — can refresh just the
        readouts when the visible set changes.
        """
        scale = self._current_axis_scale()
        for i, x_pos in enumerate(self.cursor_positions):
            if x_pos is not None:
                self._update_cursor_panel(i, x_pos)
        two_cursors = (len(self.cursor_positions) >= 2
                       and all(p is not None for p in self.cursor_positions[:2]))
        if two_cursors:
            delta_raw = abs(self.cursor_positions[1] - self.cursor_positions[0])
            self.delta_label.setText(
                f'<span style="font-weight:700;color:#e65100">ΔX</span>'
                f' <span style="font-family:monospace;font-weight:600;color:#333">'
                f'{delta_raw * scale:.4g} {(self._x_unit or "").strip()}</span>'
            )
            self._update_measure_label(delta_raw, scale)
        elif self.cursor_positions and self.cursor_positions[0] is not None:
            self.measure_label.setText(
                self._format_cursor_readout(self.cursor_positions[0]))

    def on_mouse_move(self, event) -> None:
        # Active drags get priority — fast path, no allocations.
        if self._divider_drag is not None:
            self._update_divider_drag(event)
            return
        if self._drag_cursor_idx is not None and event.xdata is not None:
            # Cursor drag — only needs the X update; skip coord-label work.
            self._update_cursor_position(self._drag_cursor_idx, event.xdata)
            return

        if event.inaxes:
            base = f"X: {event.xdata:.6g}, Y: {event.ydata:.6g}"
            if self._current_view_mode == 'stacked':
                # Anchor lookup is O(N traces); only walk when the hovered
                # pane actually changes between move events.
                if event.inaxes is not self._last_hover_axes:
                    self._last_hover_axes = event.inaxes
                    self._last_hover_anchor = self._pane_anchor_name(event.inaxes)
                if self._last_hover_anchor:
                    base = f"{base}  |  Pane: {self._last_hover_anchor}"
            if base != self._last_coord_text:
                self.coord_label.setText(base)
                self._last_coord_text = base
            # Reset resize-cursor state when we re-enter an axes
            if self._last_cursor_shape_was_resize:
                self.canvas.unsetCursor()
                self._last_cursor_shape_was_resize = False
        else:
            # Show resize cursor when hovering a divider gap in stacked mode
            want_resize = (self._current_view_mode == 'stacked'
                           and self._divider_under_mouse(event) is not None)
            if want_resize and not self._last_cursor_shape_was_resize:
                self.canvas.setCursor(Qt.CursorShape.SizeVerCursor)
                self._last_cursor_shape_was_resize = True
            elif not want_resize and self._last_cursor_shape_was_resize:
                self.canvas.unsetCursor()
                self._last_cursor_shape_was_resize = False
            if self._last_coord_text != "X: --, Y: --":
                self.coord_label.setText("X: --, Y: --")
                self._last_coord_text = "X: --, Y: --"
            self._last_hover_axes = None
            self._last_hover_anchor = None

    def on_key_press(self, event) -> None:
        if event.key == 'g': self.grid_check.toggle()
        elif event.key == 'l': self.legend_check.toggle()
        elif event.key == 'f': self._focus_btn.toggle()
        elif event.key == 'p': self.open_figure_options()
        elif event.key == 'escape':
            mode = str(self.nav_toolbar.mode).lower()
            if 'zoom' in mode:
                self.nav_toolbar.zoom()
            elif 'pan' in mode:
                self.nav_toolbar.pan()
            else:
                self.clear_cursors()

    def on_scroll(self, event) -> None:
        if not event.inaxes: return
        xlim, ylim = event.inaxes.get_xlim(), event.inaxes.get_ylim()
        zoom_factor = DEFAULT_ZOOM_FACTOR if event.button == 'up' else 1 / DEFAULT_ZOOM_FACTOR
        if event.key == 'control':
            x_center, y_center = event.xdata, event.ydata
            x_range, y_range = (xlim[1] - xlim[0]) * zoom_factor, (ylim[1] - ylim[0]) * zoom_factor
            x_ratio, y_ratio = (x_center - xlim[0]) / (xlim[1] - xlim[0]), (y_center - ylim[0]) / (ylim[1] - ylim[0])
            event.inaxes.set_xlim(x_center - x_range * x_ratio, x_center + x_range * (1 - x_ratio))
            event.inaxes.set_ylim(y_center - y_range * y_ratio, y_center + y_range * (1 - y_ratio))
        elif event.key == 'shift':
            pan_distance = (xlim[1] - xlim[0]) * 0.1 * (-1 if event.button == 'up' else 1)
            event.inaxes.set_xlim(xlim[0] + pan_distance, xlim[1] + pan_distance)
        self.canvas.draw()

    def eventFilter(self, obj, event) -> bool:
        if (obj is self.canvas and
                event.type() == QtCore.QEvent.Type.Wheel and
                self._current_view_mode == 'stacked'):
            mods = event.modifiers()
            ctrl = Qt.KeyboardModifier.ControlModifier
            shift = Qt.KeyboardModifier.ShiftModifier
            if not (mods & ctrl) and not (mods & shift):
                QtWidgets.QApplication.sendEvent(
                    self.canvas_scroll.verticalScrollBar(), event)
                return True
        return super().eventFilter(obj, event)

    def export_image(self) -> None:
        file_name, file_filter = QFileDialog.getSaveFileName(self, "Export Image", "", "PNG Files (*.png);;SVG Files (*.svg);;All Files (*)")
        if file_name:
            try:
                fmt = 'svg' if "svg" in file_filter else 'png'
                if '.' not in os.path.basename(file_name): file_name += f'.{fmt}'
                self.fig.savefig(file_name, format=fmt, dpi=DEFAULT_EXPORT_DPI, bbox_inches='tight')
                self.status_bar.showMessage(f"Image exported to {file_name}", 3000)
            except Exception as e:
                logger.error(f"Error exporting image: {e}")
                QMessageBox.warning(self, "Export Error", f"Failed to export image: {str(e)}")

    def clear_plot(self) -> None:
        self.timing_annotations.clear()
        self.deselect_all_waveforms()

    def _zoom_panes(self, factor: float) -> None:
        """Apply a symmetric zoom around the centre of each pane.

        factor < 1 zooms in (narrower range); factor > 1 zooms out.
        X is set on self.axes only — sharex propagates to all panes when
        stacked. Y is set per-pane so each retains its own scale.
        """
        if not self.panes:
            return
        xlim = self.axes.get_xlim()
        x_center = (xlim[0] + xlim[1]) / 2
        x_half = (xlim[1] - xlim[0]) * factor / 2
        self.axes.set_xlim(x_center - x_half, x_center + x_half)
        for ax in self.panes:
            ylim = ax.get_ylim()
            y_center = (ylim[0] + ylim[1]) / 2
            y_half = (ylim[1] - ylim[0]) * factor / 2
            ax.set_ylim(y_center - y_half, y_center + y_half)
        self.canvas.draw()

    def zoom_in(self) -> None:
        self._zoom_panes(DEFAULT_ZOOM_FACTOR)

    def zoom_out(self) -> None:
        self._zoom_panes(1 / DEFAULT_ZOOM_FACTOR)

    def reset_view(self) -> None:
        if self.panes:
            self.nav_toolbar.home()

    def toggle_grid(self) -> None:
        if self.panes:
            for ax in self.panes:
                ax.grid(self.grid_check.isChecked())
            self.canvas.draw()

    def toggle_legend(self) -> None:
        self.refresh_plot()

    def _resolve_expr(self, expr: str) -> "Tuple[str, Dict[str, np.ndarray]]":
        """Substitute NGSpice trace names in expr with safe Python identifiers.

        NGSpice names like v(net1), i(r1), v-minus contain characters that
        are invalid Python identifiers and would cause SyntaxError or wrong
        AST parses.  This method finds every known trace name in `expr`
        (longest first to prevent partial-match corruption, e.g. v(n1) inside
        v(n10)), replaces each with _trace_N_, and builds the corresponding
        data_map for _safe_eval.

        Pure-identifier names (no special characters) use regex word boundaries
        so that trace 'v' does not silently replace 'v' inside 'vout'.

        Returns (sanitized_expr, {identifier: array}).
        """
        indexed = self._nb_sorted
        x_len = len(self.obj_dataext.x)
        data_map: Dict[str, np.ndarray] = {}
        sanitized = expr
        for idx, name in indexed:
            if not name:
                continue
            placeholder = f'_trace_{idx}_'
            escaped = re.escape(name)
            has_special = bool(re.search(r'[^A-Za-z0-9_]', name))
            pattern = escaped if has_special else (
                r'(?<![A-Za-z0-9_])' + escaped + r'(?![A-Za-z0-9_])')
            if not re.search(pattern, sanitized):
                continue
            sanitized = re.sub(pattern, placeholder, sanitized)
            raw = np.asarray(self.obj_dataext.y[idx], dtype=float)
            data_map[placeholder] = raw[:x_len]
        return sanitized, data_map

    def plot_function(self) -> None:
        function_text = self.func_input.text().strip()
        if not function_text:
            QMessageBox.warning(self, "Input Error", "Function expression cannot be empty.")
            return
        if not hasattr(self.obj_dataext, 'NBList') or not self.obj_dataext.NBList:
            QMessageBox.warning(self, "No Data", "No simulation data loaded.")
            return

        if ' vs ' in function_text:
            self._plot_lissajous(function_text)
            return

        # Resolve NGSpice trace names → safe identifiers, then evaluate.
        try:
            sanitized, data_map = self._resolve_expr(function_text)
            canonical = _canonical_expr(sanitized)
            for i, (existing_text, *_) in enumerate(self._func_traces):
                ex_canonical = (self._func_canonical[i]
                                if i < len(self._func_canonical) else None)
                if ex_canonical is None:
                    try:
                        ex_san, _ = self._resolve_expr(existing_text)
                        ex_canonical = _canonical_expr(ex_san)
                    except Exception:
                        ex_canonical = existing_text.replace(' ', '')
                match = ex_canonical == canonical
                if match:
                    QMessageBox.information(
                        self, "Already Plotted",
                        f"'{existing_text}' is already on the plot.\n"
                        "(Note: a+b and b+a are treated as equal.)")
                    return
            y_data = _safe_eval(sanitized, data_map)
            x_data = np.asarray(self.obj_dataext.x, dtype=float)
            n = min(len(x_data), len(y_data))
            x_data, y_data = x_data[:n], y_data[:n]
        except ValueError as exc:
            names = self.obj_dataext.NBList
            preview = ', '.join(names[:8])
            if len(names) > 8:
                preview += f'  … ({len(names)} total)'
            QMessageBox.warning(
                self, "Evaluation Error",
                f"{exc}\n\nAvailable traces:\n{preview}\n\n"
                "Allowed functions: abs sqrt log log10 exp sin cos tan\n"
                "Allowed operators: + - * / **")
            return
        except Exception as exc:
            QMessageBox.warning(self, "Evaluation Error", f"Unexpected error: {exc}")
            return

        func_palette = ['#9C27B0', '#FF6D00', '#00897B', '#5E35B1', '#F4511E']
        color = func_palette[len(self._func_traces) % len(func_palette)]
        self._func_traces.append((function_text, x_data, y_data, color, DEFAULT_LINE_THICKNESS, '-'))
        self._func_canonical.append(canonical)
        self._func_visible.append(True)
        self._sync_func_trace_list()
        self.refresh_plot()
        self.func_input.clear()

    def _plot_lissajous(self, function_text: str) -> None:
        """Plot 'signal_y vs signal_x' as an X-Y (Lissajous) curve.

        Lissajous plots repurpose the X axis to a signal rather than time/freq,
        so they can't coexist with stacked view (shared-X constraint).  The
        result is drawn directly on self.axes and is NOT stored in _func_traces
        — it does not survive refresh_plot.  Users who need persistence should
        disable Stacked View before plotting.
        """
        if self._current_view_mode == 'stacked':
            QMessageBox.information(
                self, "Lissajous Plot",
                "X-Y (Lissajous) plotting requires a single time/frequency axis.\n"
                "Disable Stacked View first.")
            return
        parts = function_text.split(' vs ', 1)
        y_name, x_name = parts[0].strip(), parts[1].strip()
        if not y_name or not x_name:
            QMessageBox.warning(self, "Syntax Error",
                                "Lissajous format: 'signal_y vs signal_x'")
            return
        names = self.obj_dataext.NBList
        missing = [n for n in (y_name, x_name) if n not in names]
        if missing:
            preview = ', '.join(names[:8])
            if len(names) > 8:
                preview += f'  … ({len(names)} total)'
            QMessageBox.warning(
                self, "Trace Not Found",
                f"Not found: {', '.join(missing)}\n\nAvailable traces:\n{preview}")
            return
        x_idx = names.index(x_name)
        y_idx = names.index(y_name)
        x_data = np.asarray(self.obj_dataext.y[x_idx], dtype=float)
        y_data = np.asarray(self.obj_dataext.y[y_idx], dtype=float)
        n = min(len(x_data), len(y_data))
        if n == 0:
            QMessageBox.warning(self, "No Data", "Selected traces contain no data.")
            return
        # Remove any previous lissajous line before drawing the new one.
        if self._func_line is not None:
            try:
                self._func_line.remove()
            except ValueError:
                pass
            self._func_line = None
        self._reset_x_axis_scaling()
        is_voltage_x = x_idx < self.volts_length
        is_voltage_y = y_idx < self.volts_length
        line, = self.axes.plot(x_data[:n], y_data[:n], label=function_text)
        self._func_line = line
        self.axes.set_xlabel(f"{x_name} ({'V' if is_voltage_x else 'A'})")
        self.axes.set_ylabel(f"{y_name} ({'V' if is_voltage_y else 'A'})")
        if self.legend_check.isChecked():
            self.position_legend()
        self.canvas.draw()


    def _draw_normal_trace_line(self, t: "Trace",
                                x_data: "Optional[np.ndarray]" = None) -> "Line2D":
        """Plot one trace on the shared normal-view axes and store its line.

        Shared by the full rebuild (_plot_analysis_data) and the incremental
        refresh (_incremental_refresh_normal) so the artist type — step vs
        semilogx vs plot — is chosen identically on both paths. Branches on
        self._current_analysis_type, which the full rebuild sets first.
        """
        if x_data is None:
            x_data = np.asarray(self.obj_dataext.x, dtype=float)
        y_data = np.asarray(self.obj_dataext.y[t.index], dtype=float)
        n_pts = min(len(x_data), len(y_data))
        x_plot, y_plot = x_data[:n_pts], y_data[:n_pts]
        analysis_type = self._current_analysis_type
        plot_style = '-' if t.style == 'steps-post' else t.style
        plot_kwargs: dict = {}
        if t.style == 'steps-post' and analysis_type in ['transient', 'dc']:
            plot_func = self.axes.step
            plot_kwargs['where'] = 'post'
        elif analysis_type == 'ac_log':
            plot_func = self.axes.semilogx
        else:
            plot_func = self.axes.plot
        line, = plot_func(x_plot, y_plot, color=t.color, label=t.name,
                          linewidth=t.thickness, linestyle=plot_style, **plot_kwargs)
        t.line_object = line
        return line

    def _plot_analysis_data(self, analysis_type: str) -> None:
        self._current_analysis_type = analysis_type
        self._build_panes(1)
        traces_plotted = 0
        first_visible = None
        x_data = np.asarray(self.obj_dataext.x, dtype=float)
        for idx, t in self.traces.items():
            if not t.visible:
                continue
            traces_plotted += 1
            if first_visible is None:
                first_visible = idx
            self._draw_normal_trace_line(t, x_data)

        if analysis_type in ['ac_linear', 'ac_log']:
            self.set_freq_axis_label()
        elif analysis_type == 'dc':
            self.axes.set_xlabel('Voltage Sweep (V)')

        if first_visible is not None:
            self.axes.set_ylabel('Voltage (V)' if first_visible < self.volts_length else 'Current (A)')

        if traces_plotted == 0:
            self.axes.text(0.5, 0.5, 'Please select a waveform to plot', ha='center', va='center', transform=self.axes.transAxes)

        if analysis_type == 'transient':
            self.set_time_axis_label()

        # Overlay visible function traces on the shared axes.
        # Stacked mode renders these as separate panes in plot_stacked_diagram,
        # so this block is normal-mode-only (single axes).
        for _f_idx, (_label, _fx, _fy, _color, _thickness, _style) in enumerate(self._func_traces):
            if not (_f_idx < len(self._func_visible) and self._func_visible[_f_idx]):
                continue
            _n = min(len(_fx), len(_fy))
            if _n > 0:
                _plot_style = '-' if _style == 'steps-post' else _style
                if _style == 'steps-post':
                    self.axes.step(_fx[:_n], _fy[:_n], where='post',
                                   color=_color, label=_label, linewidth=_thickness)
                else:
                    self.axes.plot(_fx[:_n], _fy[:_n], color=_color,
                                   label=_label, linewidth=_thickness, linestyle=_plot_style)


    def on_push_decade(self) -> None:
        self._plot_analysis_data('ac_log')

    def on_push_ac(self) -> None:
        self._plot_analysis_data('ac_linear')

    def on_push_trans(self) -> None:
        self._plot_analysis_data('transient')

    def on_push_dc(self) -> None:
        self._plot_analysis_data('dc')

    def _setup_matplotlib_style(self) -> None:
        dpi = max(72, self.logicalDpiY())
        base_pt = max(6.5, round(8.0 * 96.0 / dpi, 1))
        plt.rcParams.update({
            'font.size':         base_pt,
            'axes.labelsize':    base_pt + 1,
            'axes.titlesize':    base_pt + 1,
            'xtick.labelsize':   base_pt,
            'ytick.labelsize':   base_pt,
            'legend.fontsize':   base_pt,
            'keymap.fullscreen': [],
        })

    def _on_canvas_resize(self, event) -> None:
        self._resize_timer.start()  # restart on every event; fires 120ms after last one

    def _do_deferred_resize(self) -> None:
        if hasattr(self, 'canvas'):
            self.canvas.draw_idle()

    @property
    def _em(self) -> int:
        """Font height in pixels — base unit for all adaptive sizing."""
        if self._em_cache is None:
            self._em_cache = max(12, QtGui.QFontMetrics(self.font()).height())
        return self._em_cache

    @property
    def visible_traces(self) -> List[Trace]:
        """Ordered list of visible traces (waveform-list insertion order).

        Single source of truth for "what gets plotted". All plot paths
        (normal/timing/stacked) and multi-pane logic key off this ordering
        so panes, legend, cursor readouts, and exports stay consistent.
        """
        return [self.traces[i] for i in sorted(self.traces.keys())
                if self.traces[i].visible]

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        if not getattr(self, '_splitter_initialized', False):
            QtCore.QTimer.singleShot(0, self._init_splitter_sizes)

    def _init_splitter_sizes(self) -> None:
        if getattr(self, '_splitter_initialized', False):
            return
        total = self.splitter.width()
        if total > 100:
            left_w  = int(total * 0.20)
            right_w = max(
                self.right_panel.minimumWidth(),
                self.right_panel.widget().sizeHint().width() + 8,
            )
            center_w = max(self.center_widget.minimumWidth(), total - left_w - right_w)
            self.splitter.setSizes([left_w, center_w, right_w])
            self._splitter_initialized = True
        else:
            QtCore.QTimer.singleShot(50, self._init_splitter_sizes)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self.parent():
            self.parent().updateGeometry()

    def changeEvent(self, event: QtCore.QEvent) -> None:
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.Type.FontChange:
            self._em_cache = None

    def sizeHint(self) -> QtCore.QSize:
        em = self._em
        return QtCore.QSize(em * 80, em * 50)

    def minimumSizeHint(self) -> QtCore.QSize:
        em = self._em
        return QtCore.QSize(em * 25, em * 20)
