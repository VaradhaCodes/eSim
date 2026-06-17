# =========================================================================
#             FILE: makerchip.py
#
#            USAGE: ---
#
#      DESCRIPTION: This defines all components of the Makerchip.
#
#          OPTIONS: ---
#     REQUIREMENTS: ---
#             BUGS: ---
#            NOTES: ---
#           AUTHOR: Sumanto Kar, sumantokar@iitb.ac.in, FOSSEE, IIT Bombay
# ACKNOWLEDGEMENTS: Rahul Paknikar, rahulp@iitb.ac.in, FOSSEE, IIT Bombay
#                Digvijay Singh, digvijay.singh@iitb.ac.in, FOSSEE, IIT Bombay
#                Prof. Maheswari R. and Team, VIT Chennai
#     GUIDED BY: Steve Hoover, Founder Redwood EDA
#                Kunal Ghosh, VLSI System Design Corp.Pvt.Ltd
#                Anagha Ghosh, VLSI System Design Corp.Pvt.Ltd
# OTHER CONTRIBUTERS:
#                Prof. Madhuri Kadam, Shree L. R. Tiwari College of Engineering
#                Rohinth Ram, Madras Institue of Technology
#                Charaan S., Madras Institue of Technology
#                Nalinkumar S., Madras Institue of Technology
#  ORGANIZATION: eSim Team at FOSSEE, IIT Bombay
#       CREATED: Monday 29, November 2021
#      REVISION: Tuesday 25, January 2022
# =========================================================================

# importing the files and libraries
import os
import sys
from PyQt6 import QtCore, QtWidgets
from . import Maker
from . import NgVeri

# The NGHDL GUI lives in-repo at <repo>/nghdl/src and uses flat imports, just
# like its standalone `nghdl` entry point. Put that directory on sys.path so it
# can be imported and embedded as a tab. Mutating sys.path is acceptable here
# because the names it exposes (ngspice_ghdl, Appconfig, model_generation,
# createKicadLibrary) do not collide with any eSim module.
_NGHDL_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'nghdl', 'src'))
if os.path.isdir(_NGHDL_SRC) and _NGHDL_SRC not in sys.path:
    sys.path.append(_NGHDL_SRC)

# filecount is used to count thenumber of objects created
filecount = 0


# This class creates objects for creating the Maker and the Ngveri tabs
class makerchip(QtWidgets.QWidget):

    # initialising the variables
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self)

        # filecount=int(open("a.txt",'r').read())
        print(filecount)
        # self.splitter.setOrientation(QtCore.Qt.Orientation.Vertical)
        print("==================================")
        print("Makerchip and Verilog to Ngspice Converter")
        print("==================================")
        self.createMainWindow()

    # Creating the main Window(Main tab)

    def createMainWindow(self):
        self.vbox = QtWidgets.QVBoxLayout()
        self.hbox = QtWidgets.QHBoxLayout()
        self.hbox.addStretch(1)
        self.vbox.addWidget(self.createWidget())
        self.vbox.addLayout(self.hbox)

        self.setLayout(self.vbox)
        self.setWindowTitle("Makerchip and Verilog to Ngspice Converter")
        self.show()

    # Creating the maker and ngveri widgets
    def createWidget(self):
        global obj_Maker
        global filecount
        self.convertWindow = QtWidgets.QWidget()

        self.MakerTab = QtWidgets.QScrollArea()
        obj_Maker = Maker.Maker(filecount)
        self.MakerTab.setWidget(obj_Maker)
        self.MakerTab.setWidgetResizable(True)

        global obj_NgVeri
        self.NgVeriTab = QtWidgets.QScrollArea()
        obj_NgVeri = NgVeri.NgVeri(filecount)
        self.NgVeriTab.setWidget(obj_NgVeri)
        self.NgVeriTab.setWidgetResizable(True)
        self.tabWidget = QtWidgets.QTabWidget()
        self.tabWidget.addTab(self.MakerTab, "Makerchip")
        self.tabWidget.addTab(self.NgVeriTab, "NgVeri")

        # NGHDL (VHDL -> ngspice digital model) tab. Wrapped in a QScrollArea
        # exactly like the other two tabs and built lazily + guarded, so a
        # missing/broken NGHDL install degrades to a placeholder and can never
        # take down the Makerchip dock.
        self.NgHdlTab = QtWidgets.QScrollArea()
        self.NgHdlTab.setWidgetResizable(True)
        self._nghdl_built = False
        self.nghdl_index = self.tabWidget.addTab(self.NgHdlTab, "NGHDL")
        self.tabWidget.currentChanged.connect(self.buildNgHdlTab)

        # The object refresh gets destroyed when Ngspice\
        # to verilog converter is called
        # so calling refresh_change to start toggling of refresh again
        self.tabWidget.currentChanged.connect(obj_Maker.refresh_change)
        self.mainLayout = QtWidgets.QVBoxLayout()
        self.mainLayout.addWidget(self.tabWidget)
        self.convertWindow.setLayout(self.mainLayout)
        self.convertWindow.show()
        # incrementing filecount for every new window
        filecount = filecount + 1
        return self.convertWindow

    def buildNgHdlTab(self, index):
        """Construct the embedded NGHDL widget the first time its tab is
        selected. Any failure (NGHDL not installed, no config.ini, import
        error) is contained here and shown as a placeholder."""
        if index != self.nghdl_index or self._nghdl_built:
            return
        self._nghdl_built = True
        try:
            from ngspice_ghdl import Mainwindow
            self.NgHdlTab.setWidget(Mainwindow(embedded=True))
        except Exception as e:
            print("NGHDL tab unavailable:", e)
            self.NgHdlTab.setWidget(self.nghdlPlaceholder(str(e)))

    def nghdlPlaceholder(self, reason):
        """A friendly stand-in shown when NGHDL cannot be loaded, so the
        Makerchip/NgVeri tabs keep working regardless."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        label = QtWidgets.QLabel(
            "<b>NGHDL is not available.</b><br/><br/>"
            "Make sure NGHDL is installed and configured "
            "(<code>~/.nghdl/config.ini</code>), then reopen Makerchip."
            "<br/><br/><small>Details: " + reason + "</small>")
        label.setWordWrap(True)
        label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        layout.addWidget(label)
        layout.addStretch(1)
        widget.setLayout(layout)
        return widget
