# ==============================================================================
#             FILE: createkicadCosim.py
#
#      DESCRIPTION: Build the KiCad symbol + modelParamXML record for a d_cosim
#                   (Icarus Verilog) digital block.
#
#                   Parallel to createkicad.AutoSchematic (legacy static
#                   Ngveri.cm). Emits model type "NgVeriCosim" into its own
#                   library, eSim_NgVeriCosim.kicad_sym, so d_cosim blocks and
#                   legacy Ngveri blocks coexist in the same schematic. Pin
#                   geometry, the shared-library writer, and port extraction are
#                   all inherited from AutoSchematic unchanged.
#
#  ORGANIZATION: eSim Team at FOSSEE, IIT Bombay
# ==============================================================================

import os
import xml.etree.cElementTree as ET
from PyQt6 import QtWidgets

from . import createkicad


class CosimSchematic(createkicad.AutoSchematic):
    '''
        KiCad symbol + modelParamXML record for a Verilog block co-simulated
        through ngspice's d_cosim code model (Icarus engine).

        Overrides only the target library file, the modelParamXML "type", and
        the parameter set; reuses AutoSchematic.getPortInformation / createSym /
        removeOldLibrary (identical pin geometry + atomic shared-lib writer).
    '''

    def init(self, modelname, modelpath, engine="icarus", sim_lib=""):
        super().init(modelname, modelpath)
        self.engine = engine
        # Absolute path of the compiled d_cosim artifact (Icarus vvp). Recorded
        # in the XML so the netlister never has to re-derive it.
        self.sim_lib = sim_lib
        # Mirror AutoSchematic.init's symbol-library path convention, but target
        # the separate eSim_NgVeriCosim library.
        if os.name == 'nt':
            inst_dir = self.App_obj.src_home.replace('\\eSim', '')
            self.kicad_ngveri_sym = (
                inst_dir +
                '/KiCad/share/kicad/symbols/eSim_NgVeriCosim.kicad_sym')
        else:
            self.kicad_ngveri_sym = \
                '/usr/share/kicad/symbols/eSim_NgVeriCosim.kicad_sym'

    def createKicadSymbol(self):
        '''
            Same flow/guards as AutoSchematic.createKicadSymbol, but keyed to the
            "NgVeriCosim" modelParamXML subdirectory and library. createSym ->
            _commit_block creates the library file on first use and replaces an
            existing block idempotently, so no pre-seed / removeOldLibrary needed.
        '''
        target = os.path.join(self.xml_loc, 'NgVeriCosim')
        xmlFound = None
        for root, dirs, files in os.walk(self.xml_loc):
            if (str(self.modelname) + '.xml') in files:
                xmlFound = root
                break

        if xmlFound is None:
            self.getPortInformation()
            self.createXML()
            self.createSym()
            return "No Error"

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
            self.createSym()
            return "No Error"

        QtWidgets.QMessageBox.critical(
            None, "Error",
            "<b>A different library already exists with this name.</b><br/>"
            "<b>Please change the name of your Verilog model and add "
            "it again.</b>",
            QtWidgets.QMessageBox.StandardButton.Ok)
        return "Error"

    def createXML(self):
        '''
            Write library/modelParamXML/NgVeriCosim/<model>.xml. Convert keys off
            type == "NgVeriCosim" to emit a d_cosim ".model" line instead of the
            generic param= form. The compiled vvp path is NOT stored here: both
            the build step and the netlister derive it from the nghdl config via
            CosimConfig.cosim_vvp_path(), so there is exactly one source of truth
            and nothing absolute/stale to carry in the schematic.
        '''
        cwd = os.getcwd()
        xmlDestination = os.path.join(self.xml_loc, 'NgVeriCosim')
        if not os.path.isdir(xmlDestination):
            os.makedirs(xmlDestination)

        # d_cosim has a fixed 2-port ifspec (one d_in vector, one d_out vector),
        # so ALL input bits collapse into one bracket group and ALL output bits
        # into another, regardless of how many separate Verilog ports were
        # declared. node_number=2 + this split drive Processing to emit exactly
        # two bracket groups on the a-device line.
        in_bits = sum(int(self.portInfo[i]) for i in range(self.input_length))
        out_bits = sum(int(self.portInfo[i])
                       for i in range(self.input_length, len(self.portInfo)))
        self.splitText = str(in_bits) + "-V:" + str(out_bits) + "-V"

        os.chdir(xmlDestination)
        root = ET.Element("model")
        ET.SubElement(root, "name").text = self.modelname
        ET.SubElement(root, "type").text = "NgVeriCosim"
        ET.SubElement(root, "node_number").text = "2"
        ET.SubElement(root, "title").text = (
            "Add parameters for " + str(self.modelname))
        ET.SubElement(root, "split").text = self.splitText
        param = ET.SubElement(root, "param")
        ET.SubElement(param, "instance_id", default="1").text = (
            "Enter Instance ID (Between 0-99)")
        tree = ET.ElementTree(root)
        tree.write(str(self.modelname) + '.xml')
        os.chdir(cwd)
