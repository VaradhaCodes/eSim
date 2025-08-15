# ngspiceSimulation/plot_window.py
"""
Plot Window Module

This module provides the main plotting window for NGSpice simulation results
with support for AC, DC, and Transient analysis visualization.
"""

from __future__ import division
import os
import sys
import json
import traceback
import logging
from pathlib import Path
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Tuple, Any, Union

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import Qt, QSettings, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QListWidget, QListWidgetItem, QPushButton,
                             QCheckBox, QRadioButton, QButtonGroup, QGroupBox,
                             QLabel, QLineEdit, QSlider, QDoubleSpinBox, QMenu,
                             QAction, QFileDialog, QColorDialog, QInputDialog,
                             QMessageBox, QErrorMessage, QStatusBar, QStyle,
                             QSplitter, QToolButton, QWidgetAction, QGridLayout,
                             QSpacerItem, QSizePolicy,QScrollArea)
from PyQt5.QtGui import (QColor, QBrush, QPalette, QKeySequence,
                         QPainter, QPixmap, QFont)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backend_bases import NavigationToolbar2
from matplotlib.figure import Figure
from matplotlib.widgets import Cursor
from matplotlib.lines import Line2D

from configuration.Appconfig import Appconfig
from .plotting_widgets import CollapsibleBox, MultimeterWidgetClass
from .data_extraction import DataExtraction

# Set up logging
logger = logging.getLogger(__name__)

# Constants
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

# Color Constants
VIBRANT_COLOR_PALETTE = [
    '#E53935',  # Vivid Red
    '#1E88E5',  # Strong Blue
    '#43A047',  # Rich Green
    '#FB8C00',  # Bright Orange
    '#8E24AA',  # Deep Purple
    '#00ACC1',  # Vibrant Teal
    '#D81B60',  # Strong Pink
    '#6D4C41',  # Earthy Brown
    '#FDD835',  # Visible Amber
    '#039BE5',  # Sky Blue
    '#C0CA33',  # Lime Green
    '#37474F'   # Dark Grey
]

# Time unit conversion thresholds
TIME_UNIT_THRESHOLD_NS = 1e-6
TIME_UNIT_THRESHOLD_US = 1e-3
TIME_UNIT_THRESHOLD_MS = 1

# Line style options
LINE_STYLES = [
    ('-', "Solid"),
    ('--', "Dashed"),
    (':', "Dotted"),
    ('steps-post', "Step (Post)")
]

# Thickness options
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
        """
        Initialize a trace object.

        Args:
            index: Trace index
            name: Trace name
            color: Trace color (hex string)
            thickness: Line thickness
            style: Line style
            visible: Whether trace is visible
        """
        self.index = index
        self.name = name
        self.color = color or VIBRANT_COLOR_PALETTE[0]
        self.thickness = thickness
        self.style = style
        self.visible = visible
        self.line_object: Optional[Line2D] = None

    def update_line(self, **kwargs) -> None:
        """
        Update line properties if line object exists.

        Args:
            **kwargs: Line properties to update (color, thickness, style)
        """
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
    """Custom QListWidget that handles selection without default styling."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """
        Initialize the custom list widget.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.setSelectionMode(QListWidget.MultiSelection)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """
        Override paint event to disable default selection painting.

        Args:
            event: Paint event
        """
        super().paintEvent(event)


class plotWindow(QWidget):  # Changed from QMainWindow to QWidget
    """
    Main plotting widget for NGSpice simulation results.

    This widget provides comprehensive plotting capabilities for AC, DC, and
    Transient analysis results with interactive features like cursors,
    zoom, and export functionality.
    """

    def __init__(self, file_path: str, project_name: str, parent=None) -> None:
        """
        Initialize the plot window.

        Args:
            file_path: Path to simulation data files
            project_name: Name of the project
            parent: Parent widget
        """
        super().__init__(parent)  # QWidget init

        self.file_path = file_path
        self.project_name = project_name

        # **CRITICAL FIX**: Set proper size policy for dock embedding
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)

        # Set minimum size to prevent weird behavior
        self.setMinimumSize(400, 300)

        self.obj_appconfig = Appconfig()

        logger.info(f"Complete Project Path: {self.file_path}")
        logger.info(f"Project Name: {self.project_name}")

        self.obj_appconfig.print_info(f'NGSpice simulation called: {self.file_path}')
        self.obj_appconfig.print_info(f'PythonPlotting called: {self.file_path}')

        # Initialize data structures
        self._initialize_data_structures()

        # Initialize configuration
        self._initialize_configuration()

        # Create UI (modified for QWidget)
        self.create_main_frame()

        # Load simulation data
        self.load_simulation_data()

        # Apply theme
        self.apply_theme()

    def _initialize_data_structures(self) -> None:
        """Initialize all data tracking structures."""
        self.active_traces: Dict[int, Line2D] = {}
        self.trace_visibility: Dict[int, bool] = {}
        self.trace_colors: Dict[int, str] = {}
        self.trace_thickness: Dict[int, float] = {}
        self.trace_style: Dict[int, str] = {}
        self.trace_names: Dict[int, str] = {}
        self.cursor_lines: List[Optional[Line2D]] = []
        self.cursor_positions: List[Optional[float]] = []
        self.timing_annotations: Dict[int, Any] = {}

        # Color management
        self.color_palette = VIBRANT_COLOR_PALETTE.copy()
        self.color: List[str] = []
        self.color_index = 0

        # Timing diagram settings
        self.logic_threshold: Optional[float] = None
        self.vertical_spacing = DEFAULT_VERTICAL_SPACING

    def _initialize_configuration(self) -> None:
        """Initialize configuration directories and files."""
        self.config_dir = Path.home() / '.pythonPlotting'
        self.config_file = self.config_dir / 'config.json'
        self.config: Dict[str, Any] = self.load_config()
        self.settings = QSettings('eSim', 'PythonPlotting')

    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.

        Returns:
            Configuration dictionary
        """
        try:
            self.config_dir.mkdir(exist_ok=True)
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as config_file:
                    config = json.load(config_file)
                    # Remove deprecated theme setting
                    if 'theme' in config:
                        del config['theme']
                    return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")

        # Return default configuration
        return {
            'trace_colours': {},
            'trace_thickness': {},
            'trace_style': {},
            'experimental_acdc': False
        }

    def save_config(self) -> None:
        """Save current configuration to file."""
        try:
            self.config_dir.mkdir(exist_ok=True)

            # Update configuration with current trace settings
            self.config['trace_colours'] = {
                self.trace_names.get(idx, self.obj_dataext.NBList[idx]): color
                for idx, color in self.trace_colors.items()
            }
            self.config['trace_thickness'] = {
                self.trace_names.get(idx, self.obj_dataext.NBList[idx]): thickness
                for idx, thickness in self.trace_thickness.items()
            }
            self.config['trace_style'] = {
                self.trace_names.get(idx, self.obj_dataext.NBList[idx]): style
                for idx, style in self.trace_style.items()
            }

            # Write to temporary file first, then replace
            temp_file = self.config_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as config_file:
                json.dump(self.config, config_file, indent=2)
            temp_file.replace(self.config_file)

        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """
        Handle widget close event.

        Args:
            event: Close event
        """
        self.save_config()

        # Clean up matplotlib resources
        if hasattr(self, 'canvas'):
            self.canvas.close()
        if hasattr(self, 'fig'):
            plt.close(self.fig)

        super().closeEvent(event)

    def apply_theme(self) -> None:
        """Apply clean theme with curved white background boxes only."""
        theme_stylesheet = """
        QMenuBar {
            border-radius: 8px;
            background-color: #FFFFFF;
            border: 1px solid #E0E0E0;
            padding: 2px;
        }
        QStatusBar {
            border-radius: 8px;
            background-color: #FFFFFF;
            border: 1px solid #E0E0E0;
            padding: 2px;
        }
        QWidget {
            background-color: #FFFFFF;
            color: #212121;
        }
        QListWidget {
            background-color: #FFFFFF;
            border: 1px solid #E0E0E0;
            padding: 2px;
            outline: none;
            selection-background-color: transparent;
            selection-color: inherit;
        }
        QListWidget::item {
            min-height: 32px;
            padding: 6px 8px;
            margin: 2px 4px;
            background-color: transparent;
            border: none;
        }
        QListWidget::item:selected {
            background-color: transparent;
            border: none;
        }
        QListWidget::item:hover {
            background-color: rgba(0, 0, 0, 0.04);
        }
        QListWidget::item:focus {
            outline: none;
        }
        QGroupBox {
            border: 1px solid #E0E0E0;
            margin-top: 0.5em;
            padding-top: 0.5em;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        QPushButton {
            background-color: #FFFFFF;
            border: 1px solid #E0E0E0;
            padding: 6px 12px;
            font-weight: 500;
        }
        QPushButton:hover {
            background-color: #F2F2F2;
            border-color: #1976D2;
        }
        QPushButton:pressed {
            background-color: #E0E0E0;
        }
        QCheckBox::indicator {
            width: 16px;
            height: 16px;
        }
        QMenu {
            background-color: #FFFFFF;
            border: 1px solid #E0E0E0;
        }
        QMenu::item:selected {
            background-color: #E3F2FD;
        }
        QLineEdit {
            border: 1px solid #E0E0E0;
            padding: 6px 12px;
            background-color: #FAFAFA;
        }
        QLineEdit:focus {
            border-color: #1976D2;
            background-color: #FFFFFF;
        }
        QSlider::groove:horizontal {
            border: 1px solid #E0E0E0;
            height: 4px;
            background: #E0E0E0;
        }
        QSlider::handle:horizontal {
            background: #1976D2;
            border: 1px solid #1976D2;
            width: 16px;
            height: 16px;
            margin: -6px 0;
        }
        QScrollBar:vertical {
            background-color: #F5F5F5;
            width: 8px;
            border: none;
            border-radius: 4px;
        }
        QScrollBar::handle:vertical {
            background-color: #BDBDBD;
            border-radius: 4px;
            min-height: 20px;
            margin: 2px;
        }
        QScrollBar::handle:vertical:hover {
            background-color: #9E9E9E;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
            background: transparent;
        }
        """
        self.setStyleSheet(theme_stylesheet)

    def create_main_frame(self) -> None:
        """Create the main application frame with proper size policies."""
        # Main layout for the widget
        main_widget_layout = QVBoxLayout(self)
        main_widget_layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins

        # Create menu bar as a widget menu bar
        self.menu_bar = QtWidgets.QMenuBar(self)
        main_widget_layout.addWidget(self.menu_bar)

        # Create main content widget with expanding policy
        content_widget = QWidget()
        content_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        main_layout = QHBoxLayout(content_widget)

        # Create splitter for resizable panels
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                  QtWidgets.QSizePolicy.Expanding)

        # Create main UI sections
        left_widget = self.create_waveform_list()
        self.splitter.addWidget(left_widget)

        center_widget = self.create_plot_area()
        self.splitter.addWidget(center_widget)

        right_widget = self.create_control_panel()

        # --- New code: wrap right_widget in a scroll area ---
        scroll_area = QScrollArea()
        scroll_area.setWidget(right_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        
        # **REMOVE HORIZONTAL SCROLLBAR COMPLETELY**
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)


        # Thin, minimal, rounded corner vertical scrollbar CSS
        scrollbar_style = '''
                QScrollBar:vertical {
                background-color: #F5F5F5;
                width: 8px;
                border: none;
                border-radius: 4px;
                }
                QScrollBar::handle:vertical {
                background-color: #BDBDBD;
                border-radius: 4px;
                min-height: 20px;
                margin: 2px;
                }
                QScrollBar::handle:vertical:hover {
                background-color: #9E9E9E;
                }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
}
                QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: transparent;
                }
                '''
        scroll_area.verticalScrollBar().setStyleSheet(scrollbar_style)

        # Now add the scroll area to the splitter, not the plain right_widget
        self.splitter.addWidget(scroll_area)
        # Set initial splitter sizes
        self.splitter.setSizes([280, 840, 280])

        main_layout.addWidget(self.splitter)
        main_widget_layout.addWidget(content_widget)

        # Create status bar as a widget
        self.status_bar = QStatusBar()
        self.coord_label = QLabel("X: --, Y: --")
        self.status_bar.addWidget(self.coord_label)
        self.measure_label = QLabel("")
        self.status_bar.addPermanentWidget(self.measure_label)
        main_widget_layout.addWidget(self.status_bar)

        # Create menus
        self.create_menu_bar()

        # Set window title (for dock title)
        self.setWindowTitle(f'Python Plotting - {self.project_name}')

    def create_waveform_list(self) -> QWidget:
        """
        Create the left panel with waveform list and controls.

        Returns:
            Widget containing the waveform list interface
        """
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Analysis type label
        self.analysis_label = QLabel()
        self.analysis_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        left_layout.addWidget(self.analysis_label)

        # Search functionality
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search waveforms...")
        self.search_box.textChanged.connect(self.filter_waveforms)
        left_layout.addWidget(self.search_box)

        # Waveform list with custom styling
        self.waveform_list = CustomListWidget()
        self.waveform_list.itemClicked.connect(self.on_waveform_toggle)
        self.waveform_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.waveform_list.customContextMenuRequested.connect(self.show_list_context_menu)

        # Add scrollbar styling
        self.waveform_list.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.waveform_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        left_layout.addWidget(self.waveform_list)

        # Selection buttons
        button_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.select_all_waveforms)
        self.deselect_all_btn = QPushButton("Deselect All")
        self.deselect_all_btn.clicked.connect(self.deselect_all_waveforms)
        button_layout.addWidget(self.select_all_btn)
        button_layout.addWidget(self.deselect_all_btn)
        left_layout.addLayout(button_layout)

        return left_widget

    def create_plot_area(self) -> QWidget:
        """
        Create the center panel with matplotlib plot and toolbar.

        Returns:
            Widget containing the plot area
        """
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)

        # Create matplotlib figure and canvas
        self.fig = Figure(figsize=DEFAULT_FIGURE_SIZE, dpi=DEFAULT_DPI)
        self.canvas = FigureCanvas(self.fig)

        # Create navigation toolbar
        self.nav_toolbar = NavigationToolbar(self.canvas, self)
        self.nav_toolbar.addSeparator()

        # Add custom figure options button
        fig_options_action = QAction('⚙', self.nav_toolbar)
        fig_options_action.triggered.connect(self.open_figure_options)
        fig_options_action.setToolTip('Figure Options (P)')
        self.nav_toolbar.addAction(fig_options_action)

        center_layout.addWidget(self.nav_toolbar)
        center_layout.addWidget(self.canvas)

        # Connect canvas events
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)

        self.canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self.show_canvas_context_menu)

        return center_widget

    def create_control_panel(self) -> QWidget:
        """
        Create the right panel with all control options.

        Returns:
            Widget containing the control panel
        """
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Display Options
        display_box = CollapsibleBox("Display Options")
        display_group = QWidget()
        display_layout = QVBoxLayout(display_group)

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
        display_layout.addWidget(self.autoscale_check)

        self.timing_check = QCheckBox("Digital Timing View")
        self.timing_check.stateChanged.connect(self.on_timing_view_changed)
        display_layout.addWidget(self.timing_check)

        display_box.addWidget(display_group)
        right_layout.addWidget(display_box)

        # Digital Timing Controls
        self.timing_box = CollapsibleBox("Digital Timing Controls")
        timing_group = QWidget()
        timing_layout = QVBoxLayout(timing_group)

        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Threshold:"))
        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setRange(-100, 100)
        self.threshold_spinbox.setDecimals(3)
        self.threshold_spinbox.setSingleStep(0.1)
        self.threshold_spinbox.setSuffix(" V")
        self.threshold_spinbox.setSpecialValueText("Auto")
        self.threshold_spinbox.valueChanged.connect(self.on_threshold_changed)
        threshold_layout.addWidget(self.threshold_spinbox)
        timing_layout.addLayout(threshold_layout)

        spacing_layout = QHBoxLayout()
        spacing_layout.addWidget(QLabel("Spacing:"))
        self.spacing_slider = QSlider(Qt.Horizontal)
        self.spacing_slider.setRange(100, 200)
        self.spacing_slider.setValue(120)
        self.spacing_slider.valueChanged.connect(self.on_spacing_changed)
        self.spacing_label = QLabel("1.2x")
        spacing_layout.addWidget(self.spacing_slider)
        spacing_layout.addWidget(self.spacing_label)
        timing_layout.addLayout(spacing_layout)

        self.timing_box.addWidget(timing_group)
        self.timing_box.content_area.setEnabled(False)
        right_layout.addWidget(self.timing_box)

        # Cursor Measurements
        cursor_box = CollapsibleBox("Cursor Measurements")
        cursor_group = QWidget()
        cursor_layout = QVBoxLayout(cursor_group)

        self.cursor1_label = QLabel("Cursor 1: Not set")
        self.cursor2_label = QLabel("Cursor 2: Not set")
        self.delta_label = QLabel("Delta: --")

        cursor_layout.addWidget(self.cursor1_label)
        cursor_layout.addWidget(self.cursor2_label)
        cursor_layout.addWidget(self.delta_label)

        self.clear_cursors_btn = QPushButton("Clear Cursors")
        self.clear_cursors_btn.clicked.connect(self.clear_cursors)
        cursor_layout.addWidget(self.clear_cursors_btn)

        cursor_box.addWidget(cursor_group)
        right_layout.addWidget(cursor_box)

        # Export Tools
        export_box = CollapsibleBox("Export Tools")
        export_group = QWidget()
        export_layout = QVBoxLayout(export_group)

        self.export_btn = QPushButton("Export Image")
        self.export_btn.clicked.connect(self.export_image)
        export_layout.addWidget(self.export_btn)

        self.func_input = QLineEdit()
        self.func_input.setPlaceholderText("e.g., v(in) + v(out)")
        export_layout.addWidget(self.func_input)

        self.plot_func_btn = QPushButton("Plot Function")
        self.plot_func_btn.clicked.connect(self.plot_function)
        export_layout.addWidget(self.plot_func_btn)

        self.multimeter_btn = QPushButton("Multimeter")
        self.multimeter_btn.clicked.connect(self.multi_meter)
        export_layout.addWidget(self.multimeter_btn)

        export_box.addWidget(export_group)
        right_layout.addWidget(export_box)

        right_layout.addStretch()

        return right_widget

    def create_menu_bar(self) -> None:
        """Create the menu bar with file and view menus."""
        # File menu
        file_menu = self.menu_bar.addMenu('File')

        export_action = QAction('Export Image...', self)
        export_action.triggered.connect(self.export_image)
        file_menu.addAction(export_action)

        # View menu
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

    def load_simulation_data(self) -> None:
        """Load and process simulation data from files."""
        # Get DataExtraction Details
        self.obj_dataext = DataExtraction()
        self.plot_type = self.obj_dataext.openFile(self.file_path)

        self.obj_dataext.computeAxes()
        self.data_info = self.obj_dataext.numVals()

        # Use the vibrant color palette for default assignment
        for i in range(0, self.data_info[0] - 1):
            color_idx = i % len(self.color_palette)
            self.color.append(self.color_palette[color_idx])

        # Total number of voltage source
        self.volts_length = self.data_info[1]

        # Set analysis type label
        if self.plot_type[0] == DataExtraction.AC_ANALYSIS:
            self.analysis_label.setText("AC Analysis")
        elif self.plot_type[0] == DataExtraction.TRANSIENT_ANALYSIS:
            self.analysis_label.setText("Transient Analysis")
        else:
            self.analysis_label.setText("DC Analysis")

        # Store trace names
        for i, name in enumerate(self.obj_dataext.NBList):
            self.trace_names[i] = name

        self.populate_waveform_list()

    def create_colored_icon(self, color: QColor, is_selected: bool) -> QtGui.QIcon:
        """
        Create a clean circular icon for list items.

        Args:
            color: Icon color
            is_selected: Whether the item is selected

        Returns:
            QIcon with colored circle
        """
        pixmap = QPixmap(18, 18)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        if is_selected:
            # Clean filled circle for selected items
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(1, 1, 16, 16)
        else:
            # Clean empty circle for unselected items
            painter.setBrush(Qt.NoBrush)
            pen = QtGui.QPen(QColor("#9E9E9E"))
            pen.setWidth(1)
            painter.setPen(pen)
            painter.drawEllipse(2, 2, 14, 14)

        painter.end()
        return QtGui.QIcon(pixmap)

    def populate_waveform_list(self) -> None:
        """Populate the waveform list with available signals."""
        self.waveform_list.clear()

        saved_colors = self.config.get('trace_colours', {})
        saved_thickness = self.config.get('trace_thickness', {})
        saved_style = self.config.get('trace_style', {})

        for i, node_name in enumerate(self.obj_dataext.NBList):
            item = QListWidgetItem()
            item.setData(Qt.UserRole, i)

            # Use saved colors if available, otherwise use generated colors
            if node_name in saved_colors:
                self.trace_colors[i] = saved_colors[node_name]
            elif i < len(self.color):
                self.trace_colors[i] = self.color[i]
            else:
                # If we run out of pre-assigned colors, use palette cycling
                color_idx = i % len(self.color_palette)
                self.trace_colors[i] = self.color_palette[color_idx]

            if node_name in saved_thickness:
                self.trace_thickness[i] = saved_thickness[node_name]
            else:
                self.trace_thickness[i] = DEFAULT_LINE_THICKNESS

            if node_name in saved_style:
                self.trace_style[i] = saved_style[node_name]
            else:
                self.trace_style[i] = '-'

            if i < self.obj_dataext.volts_length:
                item.setToolTip("Voltage signal")
            else:
                item.setToolTip("Current signal")

            # Initialize visibility state
            self.trace_visibility[i] = False

            # Add the item to the list FIRST
            self.waveform_list.addItem(item)

            # THEN update appearance
            self.update_list_item_appearance(item, i)

    def filter_waveforms(self, text: str) -> None:
        """
        Filter waveforms based on search text.

        Args:
            text: Search filter text
        """
        for i in range(self.waveform_list.count()):
            item = self.waveform_list.item(i)
            if item:
                item.setHidden(text.lower() not in item.text().lower())

    def on_waveform_toggle(self, item: QListWidgetItem) -> None:
        """
        Handle waveform selection toggle.

        Args:
            item: List widget item that was clicked
        """
        index = item.data(Qt.UserRole)

        # Toggle visibility based on selection state
        self.trace_visibility[index] = item.isSelected()

        if item.isSelected():
            # Assign color if not already assigned
            if index not in self.trace_colors:
                self.assign_trace_color(index)

        # Update appearance
        self.update_list_item_appearance(item, index)
        self.refresh_plot()

    def assign_trace_color(self, index: int) -> None:
        """
        Assign a unique color to a trace.

        Args:
            index: Trace index
        """
        # Get already used colors
        used_colors = set(self.trace_colors.values())

        # Find first unused color from palette
        available_colors = [color for color in self.color_palette if color not in used_colors]

        if available_colors:
            self.trace_colors[index] = available_colors[0]
        else:
            # If all palette colors are used, generate a unique color
            hue = (0.618033988749895 * len(self.trace_colors)) % 1.0
            color = QtGui.QColor.fromHsvF(hue, 0.7, 0.8)
            self.trace_colors[index] = color.name()

        self.save_config()

    def update_list_item_appearance(self, item: QListWidgetItem, index: int) -> None:
        """
        Update the visual appearance of a list item.

        Args:
            item: List widget item to update
            index: Trace index
        """
        node_name = self.trace_names.get(index, self.obj_dataext.NBList[index])
        is_selected = self.trace_visibility.get(index, False)

        # Create a widget to hold the custom layout
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(10)

        # Create icon label
        icon_label = QLabel()
        if is_selected and index in self.trace_colors:
            color = QColor(self.trace_colors[index])
            icon = self.create_colored_icon(color, True)
        else:
            color = QColor("#9E9E9E")
            icon = self.create_colored_icon(color, False)

        icon_label.setPixmap(icon.pixmap(18, 18))

        # Create text label with styling
        text_label = QLabel(node_name)

        if is_selected and index in self.trace_colors:
            text_label.setStyleSheet("color: #212121; font-weight: 500;")
        else:
            text_label.setStyleSheet("color: #757575; font-weight: normal;")

        layout.addWidget(icon_label)
        layout.addWidget(text_label)
        layout.addStretch()

        # Set the widget as the item widget
        self.waveform_list.setItemWidget(item, widget)

        # Store the text for search functionality
        item.setText(node_name)

    def select_all_waveforms(self) -> None:
        """Select all visible waveforms."""
        for i in range(self.waveform_list.count()):
            item = self.waveform_list.item(i)
            if item and not item.isHidden():
                item.setSelected(True)
                index = item.data(Qt.UserRole)
                self.trace_visibility[index] = True
                if index not in self.trace_colors:
                    self.assign_trace_color(index)
                self.update_list_item_appearance(item, index)
        self.refresh_plot()

    def deselect_all_waveforms(self) -> None:
        """Deselect all waveforms."""
        self.waveform_list.clearSelection()

        # Update visibility state for all items
        for index in self.trace_visibility:
            self.trace_visibility[index] = False

        # Update visual appearance for all items
        for i in range(self.waveform_list.count()):
            item = self.waveform_list.item(i)
            if item:
                index = item.data(Qt.UserRole)
                self.update_list_item_appearance(item, index)

        self.refresh_plot()

    def show_list_context_menu(self, position: QtCore.QPoint) -> None:
        """
        Show context menu for waveform list.

        Args:
            position: Menu position
        """
        item = self.waveform_list.itemAt(position)
        if not item:
            return

        clicked_index = item.data(Qt.UserRole)

        # Check if the clicked item is selected
        was_selected = item.isSelected()
        selected_items = self.waveform_list.selectedItems()

        # If clicked item is not selected, select only it
        if not was_selected:
            self.waveform_list.clearSelection()
            item.setSelected(True)
            selected_items = [item]

        menu = QMenu()

        # Color menu - only affects clicked item
        color_menu = menu.addMenu("Change colour ▶")
        self.populate_color_menu(color_menu, [item])

        # Thickness and style affect all selected items
        thickness_menu = menu.addMenu("Thickness ▶")
        for thickness, label in THICKNESS_OPTIONS:
            action = thickness_menu.addAction(label)
            action.triggered.connect(lambda checked, t=thickness: self.change_thickness(selected_items, t))

        style_menu = menu.addMenu("Style ▶")
        for style, label in LINE_STYLES:
            action = style_menu.addAction(label)
            action.triggered.connect(lambda checked, s=style: self.change_style(selected_items, s))

        menu.addSeparator()

        if len(selected_items) == 1:
            rename_action = menu.addAction("Rename...")
            rename_action.triggered.connect(lambda: self.rename_trace(selected_items[0]))

        index = item.data(Qt.UserRole)
        visible = False
        if index in self.active_traces and self.active_traces[index]:
            visible = self.active_traces[index].get_visible()
        hide_show_action = menu.addAction("Show" if not visible else "Hide")
        if visible:
            hide_show_action.setCheckable(True)
            hide_show_action.setChecked(True)
        hide_show_action.triggered.connect(lambda: self.toggle_trace_visibility(selected_items))

        menu.addSeparator()
        properties_action = menu.addAction("Figure Options...")
        properties_action.triggered.connect(self.open_figure_options)

        menu.exec_(self.waveform_list.mapToGlobal(position))

    def show_canvas_context_menu(self, position: QtCore.QPoint) -> None:
        """
        Show context menu for canvas.

        Args:
            position: Menu position
        """
        menu = QMenu()

        export_action = menu.addAction("Export Image...")
        export_action.triggered.connect(self.export_image)

        menu.addSeparator()

        clear_action = menu.addAction("Clear Plot")
        clear_action.triggered.connect(self.clear_plot)

        menu.exec_(self.canvas.mapToGlobal(position))

    def populate_color_menu(self, menu: QMenu, selected_items: List[QListWidgetItem]) -> None:
        """
        Populate color selection menu.

        Args:
            menu: Menu to populate
            selected_items: Selected list items
        """
        color_widget = QWidget()
        color_widget.setStyleSheet("background-color: #FFFFFF;")
        grid_layout = QGridLayout(color_widget)
        grid_layout.setSpacing(2)

        for i, color in enumerate(self.color_palette):
            btn = QPushButton()
            btn.setFixedSize(24, 24)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    border: 1px solid #E0E0E0;
                    border-radius: 2px;
                }}
                QPushButton:hover {{
                    border: 2px solid #212121;
                }}
            """)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked, c=color: self.change_color_and_close(selected_items, c, menu))
            grid_layout.addWidget(btn, i // 4, i % 4)

        widget_action = QWidgetAction(menu)
        widget_action.setDefaultWidget(color_widget)
        menu.addAction(widget_action)

        menu.addSeparator()

        more_action = menu.addAction("More...")
        more_action.triggered.connect(lambda: self.change_color_dialog(selected_items))

    def change_color_and_close(self, items: List[QListWidgetItem], color: str, menu: QMenu) -> None:
        """
        Change color and close menu.

        Args:
            items: Items to change color for
            color: New color
            menu: Menu to close
        """
        self.change_color(items, color)
        parent = menu.parent()
        while isinstance(parent, QMenu):
            parent.close()
            parent = parent.parent()

    def change_color(self, items: List[QListWidgetItem], color: str) -> None:
        """
        Change trace color for selected items.

        Args:
            items: Items to change color for
            color: New color
        """
        for item in items:
            index = item.data(Qt.UserRole)
            self.trace_colors[index] = color
            self.update_list_item_appearance(item, index)

            if index in self.active_traces and self.active_traces[index]:
                self.active_traces[index].set_color(color)

            # Update timing diagram colors
            if self.timing_check.isChecked() and hasattr(self, 'axes'):
                self.update_timing_tick_colors()
                if hasattr(self, 'timing_annotations') and index in self.timing_annotations:
                    self.timing_annotations[index].set_color(color)

        self.save_config()
        self.canvas.draw()

    def update_timing_tick_colors(self) -> None:
        """Update timing diagram tick label colors."""
        if not hasattr(self, 'axes'):
            return

        visible_indices = [i for i, v in self.trace_visibility.items() if v]
        ytick_labels = self.axes.get_yticklabels()

        for i, label in enumerate(ytick_labels):
            if i < len(visible_indices):
                idx = visible_indices[::-1][i]
                if idx in self.trace_colors:
                    label.set_color(self.trace_colors[idx])

    def change_color_dialog(self, items: List[QListWidgetItem]) -> None:
        """
        Show color picker dialog.

        Args:
            items: Items to change color for
        """
        color = QColorDialog.getColor()
        if color.isValid():
            self.change_color(items, color.name())

    def change_thickness(self, items: List[QListWidgetItem], thickness: float) -> None:
        """
        Change line thickness for selected items.

        Args:
            items: Items to change thickness for
            thickness: New thickness
        """
        for item in items:
            index = item.data(Qt.UserRole)
            self.trace_thickness[index] = thickness

            if index in self.active_traces and self.active_traces[index]:
                self.active_traces[index].set_linewidth(thickness)

        self.save_config()
        self.canvas.draw()

    def change_style(self, items: List[QListWidgetItem], style: str) -> None:
        """
        Change line style for selected items.

        Args:
            items: Items to change style for
            style: New line style
        """
        for item in items:
            index = item.data(Qt.UserRole)
            self.trace_style[index] = style

            if index in self.active_traces and self.active_traces[index]:
                if style == 'steps-post':
                    self.refresh_plot()
                    return
                else:
                    self.active_traces[index].set_linestyle(style)

        self.save_config()
        self.canvas.draw()

    def rename_trace(self, item: QListWidgetItem) -> None:
        """
        Rename a trace.

        Args:
            item: Item to rename
        """
        index = item.data(Qt.UserRole)
        current_name = self.trace_names.get(index, self.obj_dataext.NBList[index])

        new_name, ok = QInputDialog.getText(self, "Rename Trace",
                                            "New name:", text=current_name)

        if ok and new_name and new_name != current_name:
            self.trace_names[index] = new_name
            self.update_list_item_appearance(item, index)
            self.obj_dataext.NBList[index] = new_name

            if self.legend_check.isChecked():
                self.refresh_plot()

    def toggle_trace_visibility(self, items: List[QListWidgetItem]) -> None:
        """
        Toggle visibility of selected traces.

        Args:
            items: Items to toggle visibility for
        """
        any_visible = False
        for item in items:
            index = item.data(Qt.UserRole)
            if index in self.active_traces and self.active_traces[index]:
                if self.active_traces[index].get_visible():
                    any_visible = True
                    break

        for item in items:
            index = item.data(Qt.UserRole)
            if index in self.active_traces and self.active_traces[index]:
                self.active_traces[index].set_visible(not any_visible)

        self.canvas.draw()

    def open_figure_options(self) -> None:
        """Open figure options dialog."""
        try:
            # Check if we have the required matplotlib backend support
            if hasattr(self.fig.canvas, 'toolbar'):
                if hasattr(self.fig.canvas.toolbar, 'edit_parameters'):
                    self.fig.canvas.toolbar.edit_parameters()
                    return

            # Fallback to basic figure options dialog
            from matplotlib.backends.qt_compat import QtWidgets
            from matplotlib.backends.qt_editor import _formlayout

            if hasattr(_formlayout, 'FormDialog'):
                options = []
                options.append(('Title', self.fig.suptitle('').get_text()))

                if hasattr(self, 'axes'):
                    options.append(('X Label', self.axes.get_xlabel()))
                    options.append(('Y Label', self.axes.get_ylabel()))
                    options.append(('X Min', self.axes.get_xlim()[0]))
                    options.append(('X Max', self.axes.get_xlim()[1]))
                    options.append(('Y Min', self.axes.get_ylim()[0]))
                    options.append(('Y Max', self.axes.get_ylim()[1]))

                dialog = _formlayout.FormDialog(options, parent=self, title='Figure Options')
                if dialog.exec_():
                    results = dialog.get_results()
                    if results:
                        self.fig.suptitle(results[0])
                        if hasattr(self, 'axes') and len(results) > 1:
                            self.axes.set_xlabel(results[1])
                            self.axes.set_ylabel(results[2])
                            self.axes.set_xlim(results[3], results[4])
                            self.axes.set_ylim(results[5], results[6])
                        self.canvas.draw()
            else:
                QMessageBox.information(self, "Figure Options",
                                        "Figure options are limited in this environment.\n"
                                        "You can use the zoom and pan tools in the toolbar.")

        except ImportError:
            QMessageBox.information(self, "Feature Unavailable",
                                    "Figure options require additional matplotlib components.\n"
                                    "You can still use the zoom and pan tools in the toolbar.")
        except Exception as e:
            logger.error(f"Error opening figure options: {e}")
            QMessageBox.information(self, "Figure Options",
                                    "Basic figure editing is available through the toolbar.")

    def on_timing_view_changed(self, state: int) -> None:
        """
        Handle timing view checkbox state change.

        Args:
            state: Checkbox state
        """
        timing_enabled = state == Qt.Checked

        self.timing_box.content_area.setEnabled(timing_enabled)
        self.autoscale_check.setEnabled(not timing_enabled)

        self.refresh_plot()

    def refresh_plot(self) -> None:
        """Refresh the plot with current settings."""
        self.fig.clear()
        self.active_traces.clear()

        if self.timing_check.isChecked():
            self.axes = self.fig.add_subplot(111)
            self.plot_timing_diagram()
        else:
            # Use original plotting logic based on analysis type
            if self.plot_type[0] == DataExtraction.AC_ANALYSIS:
                if self.plot_type[1] == 1:
                    self.on_push_decade()
                else:
                    self.on_push_ac()
            elif self.plot_type[0] == DataExtraction.TRANSIENT_ANALYSIS:
                self.on_push_trans()
            else:
                self.on_push_dc()

        # Apply grid setting
        if hasattr(self, 'axes') and hasattr(self, 'grid_check'):
            self.axes.grid(self.grid_check.isChecked())

        # Apply legend
        if hasattr(self, 'axes'):
            if self.legend_check.isChecked():
                plt.subplots_adjust(top=0.85, bottom=0.1)
                self.position_legend()
            else:
                plt.subplots_adjust(top=0.95, bottom=0.1)

        self.canvas.draw()

    def position_legend(self) -> None:
        """Position legend at top for both normal and timing modes."""
        if hasattr(self, 'axes') and self.legend_check.isChecked():
            handles = []
            labels = []
            for idx in sorted(self.trace_visibility.keys()):
                if self.trace_visibility[idx] and idx in self.active_traces:
                    line = self.active_traces[idx]
                    if line:
                        handles.append(line)
                        labels.append(self.trace_names.get(idx, self.obj_dataext.NBList[idx]))

            if handles:
                num_traces = len(handles)
                if num_traces <= 6:
                    ncol = min(4, num_traces)
                else:
                    ncol = min(6, num_traces)

                legend = self.axes.legend(handles, labels,
                                            bbox_to_anchor=(0.5, 1.02), loc='lower center',
                                            ncol=ncol, frameon=True,
                                            fancybox=False, shadow=False, fontsize=LEGEND_FONT_SIZE,
                                            borderaxespad=0, columnspacing=1.5)

                legend.get_frame().set_facecolor('white')
                legend.get_frame().set_edgecolor('#E0E0E0')
                legend.get_frame().set_linewidth(1)
                legend.get_frame().set_alpha(0.95)

    def plot_timing_diagram(self) -> None:
        """Plot digital timing diagram."""
        visible_indices = [i for i, v in self.trace_visibility.items() if v]
        if not visible_indices:
            return

        voltage_indices = [i for i in visible_indices if i < self.obj_dataext.volts_length]

        if not voltage_indices:
            voltage_indices = visible_indices

        all_voltage_data = []
        for idx in voltage_indices:
            all_voltage_data.extend(self.obj_dataext.y[idx])

        all_voltage_data = np.array(all_voltage_data, dtype=float)
        vmin = np.min(all_voltage_data)
        vmax = np.max(all_voltage_data)

        if self.threshold_spinbox.value() == self.threshold_spinbox.minimum():
            self.logic_threshold = 0.7 * vmax
            self.threshold_spinbox.setSpecialValueText(f"Auto ({self.logic_threshold:.3f} V)")
        else:
            self.logic_threshold = self.threshold_spinbox.value()

        time_data = np.asarray(self.obj_dataext.x, dtype=float)

        self.axes.set_xlim(time_data[0], time_data[-1])

        spacing = self.vertical_spacing * vmax

        yticks, ylabels = [], []

        annotation_offset_base = 0.01 * (time_data[-1] - time_data[0])

        for rank, idx in enumerate(visible_indices[::-1]):
            raw_data = np.asarray(self.obj_dataext.y[idx], dtype=float)
            logic_data = np.where(raw_data > self.logic_threshold, vmax, 0)
            logic_offset = logic_data + rank * spacing

            color = self.trace_colors.get(idx, 'blue')
            thickness = self.trace_thickness.get(idx, DEFAULT_LINE_THICKNESS)
            label = self.trace_names.get(idx, self.obj_dataext.NBList[idx])

            line, = self.axes.step(time_data, logic_offset, where="post",
                                   linewidth=thickness, color=color, label=label)
            self.active_traces[idx] = line

            y_center = rank * spacing + vmax / 2
            yticks.append(y_center)
            ylabels.append(label)

            # Create annotation text with proper color
            final_voltage = f"{float(raw_data[-1]):.3f} V"
            annotation_offset = annotation_offset_base * (1 + 0.1 * (rank % 3))
            text_obj = self.axes.text(time_data[-1] + annotation_offset, y_center, f" {final_voltage}",
                                      va='center', fontsize=8, color=color)
            if not hasattr(self, 'timing_annotations'):
                self.timing_annotations = {}
            self.timing_annotations[idx] = text_obj

        if len(visible_indices) > 0:
            self.axes.set_ylim(-0.5 * vmax, len(visible_indices) * spacing + 0.1 * vmax)
        else:
            self.axes.set_ylim(-0.1, 1.1)

        self.axes.set_yticks(yticks)
        self.axes.set_yticklabels(ylabels, fontsize=8)

        self.update_timing_tick_colors()
        self.set_time_axis_label()

        if len(visible_indices) > 0:
            threshold_line = self.axes.axhline(y=self.logic_threshold, color='red',
                                               linestyle=':', alpha=THRESHOLD_ALPHA, linewidth=1)

        # Show title only when legend is NOT shown
        if not self.legend_check.isChecked():
            self.axes.set_title(f'Digital Timing Diagram (Threshold: {self.logic_threshold:.3f} V)',
                                fontsize=10, pad=10)

    def set_time_axis_label(self) -> None:
        """Set appropriate time axis label with unit scaling."""
        if not hasattr(self, 'axes') or not hasattr(self.obj_dataext, 'x'):
            return

        time_data = np.array(self.obj_dataext.x, dtype=float)
        if len(time_data) < 2:
            self.axes.set_xlabel('Time (s)', fontsize=10)
            return

        time_span = time_data[-1] - time_data[0]

        if time_span < TIME_UNIT_THRESHOLD_NS:
            unit = 'ns'
            scale = 1e9
        elif time_span < TIME_UNIT_THRESHOLD_US:
            unit = 'µs'
            scale = 1e6
        elif time_span < TIME_UNIT_THRESHOLD_MS:
            unit = 'ms'
            scale = 1e3
        else:
            unit = 's'
            scale = 1

        if scale != 1:
            scaled_x = time_data * scale
            for idx, line in self.active_traces.items():
                if line:
                    y_data = line.get_ydata()
                    line.set_xdata(scaled_x)

        self.axes.set_xlabel(f'Time ({unit})', fontsize=10)

    def on_threshold_changed(self, value: float) -> None:
        """
        Handle threshold value change.

        Args:
            value: New threshold value
        """
        if self.timing_check.isChecked():
            self.refresh_plot()

    def on_spacing_changed(self, value: int) -> None:
        """
        Handle spacing slider value change.

        Args:
            value: New spacing value (as percentage)
        """
        self.vertical_spacing = value / 100.0
        self.spacing_label.setText(f"{self.vertical_spacing:.1f}x")
        if self.timing_check.isChecked():
            self.refresh_plot()

    def on_canvas_click(self, event) -> None:
        """
        Handle canvas click events for cursor placement.

        Args:
            event: Mouse click event
        """
        if hasattr(self, 'axes') and event.inaxes == self.axes:
            axes = self.axes
        else:
            return

        if event.button == 1:  # Left click
            self.set_cursor(0, event.xdata, axes)
        elif event.button == 2:  # Middle click
            self.set_cursor(1, event.xdata, axes)

    def set_cursor(self, cursor_num: int, x_pos: float, axes) -> None:
        """
        Set cursor position and update display.

        Args:
            cursor_num: Cursor number (0 or 1)
            x_pos: X position for cursor
            axes: Matplotlib axes object
        """
        if cursor_num < len(self.cursor_lines):
            if self.cursor_lines[cursor_num]:
                self.cursor_lines[cursor_num].remove()

        color = 'red' if cursor_num == 0 else 'blue'
        line = axes.axvline(x=x_pos, color=color, linestyle='--', alpha=CURSOR_ALPHA)

        if cursor_num >= len(self.cursor_lines):
            self.cursor_lines.append(line)
            self.cursor_positions.append(x_pos)
        else:
            self.cursor_lines[cursor_num] = line
            self.cursor_positions[cursor_num] = x_pos

        if cursor_num == 0:
            self.cursor1_label.setText(f"Cursor 1: {x_pos:.6g}")
        else:
            self.cursor2_label.setText(f"Cursor 2: {x_pos:.6g}")

        if len(self.cursor_positions) >= 2 and all(p is not None for p in self.cursor_positions[:2]):
            delta = abs(self.cursor_positions[1] - self.cursor_positions[0])
            self.delta_label.setText(f"Delta: {delta:.6g}")

            # For frequency measurements in AC analysis
            if self.plot_type[0] == DataExtraction.AC_ANALYSIS:
                freq_delta = 1.0 / delta if delta != 0 else 0
                self.measure_label.setText(f"Freq: {freq_delta:.6g} Hz")

        self.canvas.draw()

    def clear_cursors(self) -> None:
        """Clear all cursors from the plot."""
        for line in self.cursor_lines:
            if line:
                line.remove()
        self.cursor_lines = []
        self.cursor_positions = []

        self.cursor1_label.setText("Cursor 1: Not set")
        self.cursor2_label.setText("Cursor 2: Not set")
        self.delta_label.setText("Delta: --")
        self.measure_label.setText("")

        self.canvas.draw()

    def on_mouse_move(self, event) -> None:
        """
        Handle mouse movement over canvas.

        Args:
            event: Mouse move event
        """
        if event.inaxes:
            self.coord_label.setText(f"X: {event.xdata:.6g}, Y: {event.ydata:.6g}")
        else:
            self.coord_label.setText("X: --, Y: --")

    def on_key_press(self, event) -> None:
        """
        Handle keyboard shortcuts.

        Args:
            event: Key press event
        """
        if event.key == 'g':
            self.grid_check.setChecked(not self.grid_check.isChecked())
        elif event.key == 'l':
            self.legend_check.setChecked(not self.legend_check.isChecked())
        elif event.key == 'p':
            self.open_figure_options()
        elif event.key == 'escape':
            self.clear_cursors()

    def on_scroll(self, event) -> None:
        """
        Handle mouse scroll events for zooming and panning.

        Args:
            event: Scroll event
        """
        if not event.inaxes:
            return

        xlim = event.inaxes.get_xlim()
        ylim = event.inaxes.get_ylim()

        zoom_factor = DEFAULT_ZOOM_FACTOR if event.button == 'up' else 1.1

        if event.key == 'control':
            # Zoom around mouse position
            x_center = event.xdata
            y_center = event.ydata

            x_range = (xlim[1] - xlim[0]) * zoom_factor
            y_range = (ylim[1] - ylim[0]) * zoom_factor

            x_ratio = (x_center - xlim[0]) / (xlim[1] - xlim[0])
            y_ratio = (y_center - ylim[0]) / (ylim[1] - ylim[0])

            event.inaxes.set_xlim(x_center - x_range * x_ratio,
                                  x_center + x_range * (1 - x_ratio))
            event.inaxes.set_ylim(y_center - y_range * y_ratio,
                                  y_center + y_range * (1 - y_ratio))
        elif event.key == 'shift':
            # Pan horizontally
            pan_distance = (xlim[1] - xlim[0]) * 0.1
            if event.button == 'up':
                pan_distance = -pan_distance
            event.inaxes.set_xlim(xlim[0] + pan_distance, xlim[1] + pan_distance)

        self.canvas.draw()

    def export_image(self) -> None:
        """Export the current plot as an image file."""
        file_name, file_filter = QFileDialog.getSaveFileName(
            self, "Export Image", "",
            "PNG Files (*.png);;SVG Files (*.svg);;All Files (*)")

        if file_name:
            try:
                if file_filter == "SVG Files (*.svg)" or file_name.endswith('.svg'):
                    self.fig.savefig(file_name, format='svg', dpi=DEFAULT_EXPORT_DPI, bbox_inches='tight')
                else:
                    if not file_name.endswith('.png'):
                        file_name += '.png'
                    self.fig.savefig(file_name, format='png', dpi=DEFAULT_EXPORT_DPI, bbox_inches='tight')

                self.status_bar.showMessage(f"Image exported to {file_name}", 3000)
            except Exception as e:
                logger.error(f"Error exporting image: {e}")
                QMessageBox.warning(self, "Export Error",
                                    f"Failed to export image: {str(e)}")

    def clear_plot(self) -> None:
        """Clear the plot and reset timing annotations."""
        if hasattr(self, 'timing_annotations'):
            self.timing_annotations.clear()
        self.deselect_all_waveforms()

    def zoom_in(self) -> None:
        """Zoom into the plot."""
        zoom_factor = DEFAULT_ZOOM_FACTOR

        if hasattr(self, 'axes'):
            xlim = self.axes.get_xlim()
            ylim = self.axes.get_ylim()

            x_center = (xlim[0] + xlim[1]) / 2
            y_center = (ylim[0] + ylim[1]) / 2

            x_range = (xlim[1] - xlim[0]) * zoom_factor
            y_range = (ylim[1] - ylim[0]) * zoom_factor

            self.axes.set_xlim(x_center - x_range/2, x_center + x_range/2)
            self.axes.set_ylim(y_center - y_range/2, y_center + y_range/2)

        self.canvas.draw()

    def zoom_out(self) -> None:
        """Zoom out of the plot."""
        zoom_factor = 1.1

        if hasattr(self, 'axes'):
            xlim = self.axes.get_xlim()
            ylim = self.axes.get_ylim()

            x_center = (xlim[0] + xlim[1]) / 2
            y_center = (ylim[0] + ylim[1]) / 2

            x_range = (xlim[1] - xlim[0]) * zoom_factor
            y_range = (ylim[1] - ylim[0]) * zoom_factor

            self.axes.set_xlim(x_center - x_range/2, x_center + x_range/2)
            self.axes.set_ylim(y_center - y_range/2, y_center + y_range/2)

        self.canvas.draw()

    def reset_view(self) -> None:
        """Reset the plot view to show all data."""
        visible_indices = [i for i, v in self.trace_visibility.items() if v]

        if not visible_indices:
            return

        if hasattr(self, 'axes'):
            x_min, x_max = float('inf'), float('-inf')
            y_min, y_max = float('inf'), float('-inf')

            for idx in visible_indices:
                x_data = np.array(self.obj_dataext.x, dtype=float)
                y_data = np.array(self.obj_dataext.y[idx], dtype=float)

                x_min = min(x_min, np.min(x_data))
                x_max = max(x_max, np.max(x_data))
                y_min = min(y_min, np.min(y_data))
                y_max = max(y_max, np.max(y_data))

            x_margin = (x_max - x_min) * 0.05
            y_margin = (y_max - y_min) * 0.1

            self.axes.set_xlim(x_min - x_margin, x_max + x_margin)

            if self.timing_check.isChecked():
                vmax = y_max
                self.axes.set_ylim(-0.5 * vmax, len(visible_indices) * self.vertical_spacing * vmax + 0.1 * vmax)
            else:
                self.axes.set_ylim(y_min - y_margin, y_max + y_margin)

        self.canvas.draw()

    def toggle_grid(self) -> None:
        """Toggle grid display."""
        if hasattr(self, 'axes'):
            self.axes.grid(self.grid_check.isChecked())
            self.canvas.draw()

    def toggle_legend(self) -> None:
        """Toggle legend display."""
        self.refresh_plot()

    def plot_function(self) -> None:
        """Plot a mathematical function of existing traces."""
        function_parts = str(self.func_input.text()).split(" ")

        if function_parts and function_parts[-1] == '':
            function_parts = function_parts[:-1]

        if len(function_parts) <= 2:
            QMessageBox.warning(self, "Warning!!",
                                "Too Few Arguments/SYNTAX Error!\nRefer Examples")
            return

        trace_indices = []
        final_result = []

        # Find matching trace indices
        for i in range(0, len(function_parts), 2):
            if i < len(function_parts):
                for j, trace_name in enumerate(self.obj_dataext.NBList):
                    if function_parts[i] == trace_name:
                        trace_indices.append(j)
                        break

        if len(trace_indices) != len(function_parts) // 2 + 1:
            QMessageBox.warning(self, "Warning!!",
                                "One of the operands doesn't belong to "
                                "the above list of Nodes!!")
            return

        # Handle vs plotting (x vs y)
        if len(function_parts) == 3 and function_parts[1] == 'vs':
            if hasattr(self, 'axes'):
                x_data = [float(val) for val in self.obj_dataext.y[trace_indices[0]]]
                y_data = [float(val) for val in self.obj_dataext.y[trace_indices[1]]]

                self.axes.plot(x_data, y_data, c='green',
                               label=f"{function_parts[0]} vs {function_parts[2]}")

                if max(trace_indices) < self.volts_length:
                    self.axes.set_ylabel('Voltage(V)-->')
                    self.axes.set_xlabel('Voltage(V)-->')
                else:
                    self.axes.set_ylabel('Current(I)-->')
                    self.axes.set_xlabel('Current(I)-->')

                if self.legend_check.isChecked():
                    self.position_legend()

                self.canvas.draw()
            return

        # Check for mixed voltage/current
        voltage_indices = [idx for idx in trace_indices if idx < self.volts_length]
        current_indices = [idx for idx in trace_indices if idx >= self.volts_length]

        if voltage_indices and current_indices:
            QMessageBox.warning(self, "Warning!!",
                                "Do not combine Voltage and Current!!")
            return

        # Evaluate mathematical expression
        combo_data = [self.obj_dataext.y[idx] for idx in trace_indices]

        for j in range(len(combo_data[0])):
            expression_parts = function_parts.copy()
            for i in range(0, len(expression_parts), 2):
                if i < len(expression_parts):
                    trace_idx = i // 2
                    if trace_idx < len(combo_data):
                        expression_parts[i] = str(combo_data[trace_idx][j])

            expression_string = " ".join(expression_parts)
            try:
                final_result.append(eval(expression_string))
            except (ArithmeticError, ZeroDivisionError):
                QMessageBox.warning(self, "Warning!!", "Division by zero or arithmetic error!")
                return

        if hasattr(self, 'axes'):
            label = " ".join(function_parts)
            self.axes.plot(self.obj_dataext.x, final_result, c='green',
                           label=label, linewidth=2)

            if self.legend_check.isChecked():
                self.position_legend()

            self.canvas.draw()

    def multi_meter(self) -> None:
        """Display multimeter widgets for selected traces."""
        visible_indices = [i for i, v in self.trace_visibility.items() if v]

        if not visible_indices:
            QMessageBox.warning(self, "Warning",
                                "Please select at least one waveform")
            return

        location_x = 300
        location_y = 300

        for idx in visible_indices:
            is_voltage = idx < self.obj_dataext.volts_length
            rms_value = self.get_rms_value(self.obj_dataext.y[idx])

            meter = MultimeterWidgetClass(
                self.trace_names.get(idx, self.obj_dataext.NBList[idx]),
                rms_value,
                location_x,
                location_y,
                is_voltage
            )

            # Add to dock if available
            if (hasattr(self.obj_appconfig, 'dock_dict') and
                    self.obj_appconfig.current_project['ProjectName'] in self.obj_appconfig.dock_dict):
                self.obj_appconfig.dock_dict[
                    self.obj_appconfig.current_project['ProjectName']
                ].append(meter)

            location_x += 50
            location_y += 50

    def get_rms_value(self, data_points: List) -> Decimal:
        """
        Calculate RMS value of data points.

        Args:
            data_points: List of data values

        Returns:
            RMS value as Decimal
        """
        getcontext().prec = 5
        return Decimal(str(np.sqrt(np.mean(np.square([float(x) for x in data_points])))))

    def redraw_cursors(self) -> None:
        """Redraw cursors after plot refresh."""
        new_cursor_lines = []
        for i, pos in enumerate(self.cursor_positions):
            if pos is not None:
                color = 'red' if i == 0 else 'blue'
                if hasattr(self, 'axes'):
                    line = self.axes.axvline(x=pos, color=color, linestyle='--', alpha=CURSOR_ALPHA)
                    new_cursor_lines.append(line)
        self.cursor_lines = new_cursor_lines

    def _plot_analysis_data(self, analysis_type: str) -> None:
        """
        Generic method to plot analysis data for any analysis type.
        
        Args:
            analysis_type: Type of analysis ('ac_linear', 'ac_log', 'transient', 'dc')
        """
        self.axes = self.fig.add_subplot(111)
        traces_plotted = 0

        for trace_index in self.trace_visibility:
            if not self.trace_visibility[trace_index]:
                continue
                
            traces_plotted += 1
            
            # Get trace properties
            color = self.trace_colors.get(trace_index, self.color_palette[trace_index % len(self.color_palette)])
            label = self.trace_names.get(trace_index, self.obj_dataext.NBList[trace_index])
            thickness = self.trace_thickness.get(trace_index, DEFAULT_LINE_THICKNESS)
            style = self.trace_style.get(trace_index, '-')

            # Plot based on analysis type and style
            if style == 'steps-post' and analysis_type in ['transient', 'dc']:
                line, = self.axes.step(
                    self.obj_dataext.x,
                    self.obj_dataext.y[trace_index],
                    where='post',
                    c=color,
                    label=label,
                    linewidth=thickness
                )
            elif analysis_type == 'ac_log':
                # Force solid line for log plots
                plot_style = '-' if style == 'steps-post' else style
                line, = self.axes.semilogx(
                    self.obj_dataext.x,
                    self.obj_dataext.y[trace_index],
                    c=color,
                    label=label,
                    linewidth=thickness,
                    linestyle=plot_style
                )
            else:
                # Linear plots (AC linear, transient, DC)
                plot_style = '-' if style == 'steps-post' else style
                line, = self.axes.plot(
                    self.obj_dataext.x,
                    self.obj_dataext.y[trace_index],
                    c=color,
                    label=label,
                    linewidth=thickness,
                    linestyle=plot_style
                )
            
            self.active_traces[trace_index] = line

        # Set axis labels based on analysis type
        if analysis_type in ['ac_linear', 'ac_log']:
            self.axes.set_xlabel('freq-->')
            # Set ylabel based on first visible trace type
            first_visible = next((i for i in self.trace_visibility if self.trace_visibility[i]), 0)
            if first_visible < self.volts_length:
                self.axes.set_ylabel('Voltage(V)-->')
            else:
                self.axes.set_ylabel('Current(I)-->')
        elif analysis_type == 'transient':
            self.axes.set_xlabel('time-->')
            first_visible = next((i for i in self.trace_visibility if self.trace_visibility[i]), 0)
            if first_visible < self.volts_length:
                self.axes.set_ylabel('Voltage(V)-->')
            else:
                self.axes.set_ylabel('Current(I)-->')
        elif analysis_type == 'dc':
            self.axes.set_xlabel('Voltage Sweep(V)-->')
            first_visible = next((i for i in self.trace_visibility if self.trace_visibility[i]), 0)
            if first_visible < self.volts_length:
                self.axes.set_ylabel('Voltage(V)-->')
            else:
                self.axes.set_ylabel('Current(I)-->')

        # Show message if no traces selected
        if traces_plotted == 0:
            self.axes.text(0.5, 0.5, 'Please select at least one waveform',
                           horizontalalignment='center',
                           verticalalignment='center',
                           transform=self.axes.transAxes)

    def on_push_decade(self) -> None:
        """Plot AC Analysis with decade (log scale)."""
        self._plot_analysis_data('ac_log')

    def on_push_ac(self) -> None:
        """Plot AC Analysis with linear scale."""
        self._plot_analysis_data('ac_linear')

    def on_push_trans(self) -> None:
        """Plot Transient Analysis."""
        self._plot_analysis_data('transient')

    def on_push_dc(self) -> None:
        """Plot DC Analysis."""
        self._plot_analysis_data('dc')

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """Handle resize events to maintain proper dock behavior."""
        super().resizeEvent(event)

        # Notify parent dock widget of size changes
        if self.parent():
            self.parent().updateGeometry()

        # Force canvas redraw if matplotlib is loaded
        if hasattr(self, 'canvas') and self.canvas:
            self.canvas.draw_idle()

    def sizeHint(self) -> QtCore.QSize:
        """Provide size hint for proper dock widget sizing."""
        return QtCore.QSize(1200, 800)

    def minimumSizeHint(self) -> QtCore.QSize:
        """Provide minimum size hint."""
        return QtCore.QSize(400, 300)
