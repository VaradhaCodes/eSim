# ==============================================================================
#             FILE: createkicadCosim.py
#
#      DESCRIPTION: Creates the KiCad symbol + modelParamXML entry for a
#                   d_cosim (Icarus Verilog) digital block.
#
#                   Parallel to createkicad.AutoSchematic (legacy static
#                   Ngveri.cm). Emits model type "NgVeriCosim" into its own
#                   library, eSim_NgVeriCosim.kicad_sym, so d_cosim blocks and
#                   legacy Ngveri blocks coexist in the same schematic.
#
#  ORGANIZATION: eSim Team at FOSSEE, IIT Bombay
# ==============================================================================

import os
import xml.etree.cElementTree as ET
from PyQt6 import QtWidgets

from . import createkicad


class CosimSchematic(createkicad.AutoSchematic):
    '''
        Build a KiCad symbol and modelParamXML record for a Verilog block that
        is co-simulated through ngspice's d_cosim code model (Icarus engine).

        Reuses AutoSchematic.getPortInformation / createSym / removeOldLibrary
        (identical pin geometry), overriding only the target library file, the
        modelParamXML "type", and the parameter set.
    '''

    def init(self, modelname, modelpath, engine="icarus", sim_lib=""):
        super().init(modelname, modelpath)
        self.engine = engine
        # Absolute path of the compiled d_cosim artifact (Icarus vvp file).
        self.sim_lib = sim_lib
        if os.name == 'nt':
            eSim_src = self.App_obj.src_home
            inst_dir = eSim_src.replace('\\eSim', '')
            self.kicad_ngveri_sym = (
                inst_dir +
                '/KiCad/share/kicad/symbols/eSim_NgVeriCosim.kicad_sym')
        else:
            self.kicad_ngveri_sym = \
                '/usr/share/kicad/symbols/eSim_NgVeriCosim.kicad_sym'

    def createKicadSymbol(self):
        '''
            Same flow as AutoSchematic.createKicadSymbol but checks/uses the
            "NgVeriCosim" modelParamXML subdirectory and library file.
        '''
        target = os.path.join(self.xml_loc, 'NgVeriCosim')
        xmlFound = None
        for root, dirs, files in os.walk(self.xml_loc):
            if (str(self.modelname) + '.xml') in files:
                xmlFound = root
                break

        if xmlFound is None or xmlFound == target:
            if xmlFound == target:
                ret = QtWidgets.QMessageBox.warning(
                    None, "Warning",
                    "<b>d_cosim library files for this model already exist. "
                    "Do you want to overwrite them?</b>",
                    QtWidgets.QMessageBox.StandardButton.Ok,
                    QtWidgets.QMessageBox.StandardButton.Cancel)
                if ret != QtWidgets.QMessageBox.StandardButton.Ok:
                    return "Error"
            self.getPortInformation()
            self.createXML()
            self._ensure_lib_exists()
            if xmlFound == target:
                self.removeOldLibrary()
            self.createSym()
            return "No Error"

        QtWidgets.QMessageBox.critical(
            None, "Error",
            "<b>A different library already exists with this name.</b><br/>"
            "<b>Please change the name of your Verilog model and add "
            "it again.</b>",
            QtWidgets.QMessageBox.StandardButton.Ok)
        return "Error"

    def _ensure_lib_exists(self):
        '''
            createSym (inherited) opens the library file for reading first, so
            create an empty one on first use; createSym then writes the
            kicad_symbol_lib header into the empty file.
        '''
        if not os.path.isfile(self.kicad_ngveri_sym):
            open(self.kicad_ngveri_sym, 'w').close()

    def createXML(self):
        '''
            Write library/modelParamXML/NgVeriCosim/<model>.xml. The netlister
            (Convert.addCosimParameter) keys off type == "NgVeriCosim" and emits
            a d_cosim ".model" line rather than the generic param= form.
        '''
        cwd = os.getcwd()
        xmlDestination = os.path.join(self.xml_loc, 'NgVeriCosim')
        if not os.path.isdir(xmlDestination):
            os.makedirs(xmlDestination)

        self.splitText = ""
        for bit in self.portInfo[:-1]:
            self.splitText += bit + "-V:"
        self.splitText += self.portInfo[-1] + "-V"

        os.chdir(xmlDestination)
        root = ET.Element("model")
        ET.SubElement(root, "name").text = self.modelname
        ET.SubElement(root, "type").text = "NgVeriCosim"
        ET.SubElement(root, "node_number").text = str(len(self.portInfo))
        ET.SubElement(root, "title").text = (
            "Add parameters for " + str(self.modelname))
        ET.SubElement(root, "split").text = self.splitText
        param = ET.SubElement(root, "param")
        ET.SubElement(param, "instance_id", default="1").text = (
            "Enter Instance ID (Between 0-99)")
        # Bookkeeping consumed by the netlister; defaults carry the values.
        ET.SubElement(param, "engine", default=self.engine).text = (
            "Co-simulation engine")
        ET.SubElement(param, "sim_lib", default=self.sim_lib).text = (
            "Compiled d_cosim artifact path")
        tree = ET.ElementTree(root)
        tree.write(str(self.modelname) + '.xml')
        os.chdir(cwd)
