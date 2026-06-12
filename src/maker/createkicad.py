# ==============================================================================
#             FILE: createkicad.py
#
#            USAGE: ---
#
#      DESCRIPTION: This define all components of to create the Kicad Library.
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
#                Partha Singha Roy, Kalyani Government Engineering College
#  ORGANIZATION: eSim Team at FOSSEE, IIT Bombay
#       CREATED: Monday 29, November 2021
#      REVISION: Friday 16, June 2023
# ==============================================================================

from . import Appconfig
import re
import os
import tempfile
import xml.etree.cElementTree as ET
from PyQt6 import QtWidgets


# ── Robust S-expression helpers for the shared eSim_Ngveri.kicad_sym lib ──
# The library is one KiCad symbol file appended to by every NgVeri model. The
# original code mutated it with raw byte/line surgery (content[:-2],
# lines[0:-2], line.startswith("(symbol")) which, on repeated add/overwrite/
# delete, glued blocks together ("))(symbol ..."), shed a part's opening
# "(symbol" line, and left unbalanced parens. KiCad then rejects the whole file
# and the library disappears. These helpers parse the file into balanced
# top-level part blocks keyed by model name, so add/overwrite/delete are
# idempotent and always re-serialize a valid, balanced file (also healing an
# already-corrupted file on the next write).

_LIB_HEADER = ('(kicad_symbol_lib (version 20211014) '
               '(generator kicad_symbol_editor)')

# A *part* opener is distinctively '(symbol "<name>" (pin_names'. Sub-units
# '(symbol "<name>_0_1"(rectangle' / '(symbol "<name>_1_1"' never match this.
_PART_RE = re.compile(r'\(symbol\s+"([^"]+)"\s+\(pin_names')


def _balanced_end(text, start):
    '''Index just past the ")" that closes the "(" at text[start], honoring
       quoted strings. Returns -1 if it never balances (truncated/corrupt).'''
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == '\\':
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                return i + 1
    return -1


def _extract_parts(content):
    '''Parse a (possibly corrupt) kicad_sym into an ordered {name: block}.
       Only well-formed balanced part blocks are kept; orphaned/duplicate
       blocks are dropped and the last definition of a name wins (freshest).'''
    parts = {}
    for m in _PART_RE.finditer(content):
        end = _balanced_end(content, m.start())
        if end != -1:
            parts[m.group(1)] = content[m.start():end].strip()
    return parts


def _read_parts(path):
    try:
        with open(path) as f:
            content = f.read()
    except FileNotFoundError:
        content = ''
    return _extract_parts(content)


def _write_lib(path, parts):
    '''Serialize {name: block} back into a valid, balanced kicad_sym file.
       Written atomically (temp file in the same dir + os.replace) so a crash,
       full disk, or kill -9 mid-write can never leave the shared library
       truncated or empty — the failure mode this module exists to fix.'''
    out = [_LIB_HEADER, '']
    for block in parts.values():
        out.append(block)
        out.append('')
    out.append(')')
    data = '\n'.join(out) + '\n'
    directory = os.path.dirname(path) or '.'
    fd, tmp = tempfile.mkstemp(
        dir=directory, prefix='.eSim_Ngveri_', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)           # atomic on POSIX and Windows
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


class AutoSchematic:
    def init(self, modelname, modelpath):
        self.App_obj = Appconfig.Appconfig()
        self.modelname = modelname.split('.')[0]
        self.template = self.App_obj.kicad_sym_template.copy()
        self.xml_loc = self.App_obj.xml_loc
        self.lib_loc = self.App_obj.lib_loc
        self.modelpath = modelpath
        if os.name == 'nt':
            eSim_src = self.App_obj.src_home
            inst_dir = eSim_src.replace('\\eSim', '')
            self.kicad_ngveri_sym = \
                inst_dir + '/KiCad/share/kicad/symbols/eSim_Ngveri.kicad_sym'
        else:
            self.kicad_ngveri_sym = \
                '/usr/share/kicad/symbols/eSim_Ngveri.kicad_sym'
        # self.parser = self.App_obj.parser_ngveri

    def createKicadSymbol(self):
        '''
            creating KiCad library using this function
        '''
        xmlFound = None
        for root, dirs, files in os.walk(self.xml_loc):
            if (str(self.modelname) + '.xml') in files:
                xmlFound = root
                print(xmlFound)
                break

        if xmlFound is None:
            self.getPortInformation()
            self.createXML()
            self.createSym()

        elif (xmlFound == os.path.join(self.xml_loc, 'Ngveri')):
            print('Library already exists...')
            ret = QtWidgets.QMessageBox.warning(
                None, "Warning", '''<b>Library files for this model''' +
                ''' already exist. Do you want to overwrite it?</b><br/>
                If yes press ok, else cancel it and ''' +
                '''change the name of your verilog model.''',
                QtWidgets.QMessageBox.StandardButton.Ok, QtWidgets.QMessageBox.StandardButton.Cancel
            )

            if ret == QtWidgets.QMessageBox.StandardButton.Ok:
                print("Overwriting existing libraries")
                self.getPortInformation()
                self.createXML()
                self.removeOldLibrary()     # Removes the existing library
                self.createSym()
            else:
                print("Library Creation Cancelled")
                return "Error"

        else:
            print('Pre-existing library...')
            ret = QtWidgets.QMessageBox.critical(
                self.parent, "Error", '''<b>A standard library already ''' +
                '''exists with this name.</b><br/><b>Please change the ''' +
                '''name of your verilog model and add it again.</b>''',
                QtWidgets.QMessageBox.StandardButton.Ok
            )

    def getPortInformation(self):
        '''
            getting the port information here
        '''
        portInformation = PortInfo(self, self.modelpath)
        portInformation.getPortInfo()
        self.portInfo = portInformation.bit_list
        self.input_length = portInformation.input_len
        self.portName = portInformation.port_name

    def createXML(self):
        '''
            creating the XML files at `library/modelParamXML/Ngveri`
        '''
        cwd = os.getcwd()
        xmlDestination = os.path.join(self.xml_loc, 'Ngveri')
        self.splitText = ""
        for bit in self.portInfo[:-1]:
            self.splitText += bit + "-V:"
        self.splitText += self.portInfo[-1] + "-V"

        print("changing directory to ", xmlDestination)
        os.chdir(xmlDestination)

        root = ET.Element("model")
        ET.SubElement(root, "name").text = self.modelname
        ET.SubElement(root, "type").text = "Ngveri"
        ET.SubElement(root, "node_number").text = str(len(self.portInfo))
        ET.SubElement(root, "title").text = (
            "Add parameters for " + str(self.modelname))
        ET.SubElement(root, "split").text = self.splitText
        param = ET.SubElement(root, "param")
        ET.SubElement(param, "rise_delay", default="1.0e-9").text = (
            "Enter Rise Delay (default=1.0e-9)")
        ET.SubElement(param, "fall_delay", default="1.0e-9").text = (
            "Enter Fall Delay (default=1.0e-9)")
        ET.SubElement(param, "input_load", default="1.0e-12").text = (
            "Enter Input Load (default=1.0e-12)")
        ET.SubElement(param, "instance_id", default="1").text = (
            "Enter Instance ID (Between 0-99)")

        tree = ET.ElementTree(root)
        tree.write(str(self.modelname) + '.xml')
        print("Leaving the directory ", xmlDestination)
        os.chdir(cwd)

    def findBlockSize(self):
        '''
            Calculates the maximum between input and output ports
        '''
        ind = self.input_length
        return max(
            self.char_sum(self.portInfo[:ind]),
            self.char_sum(self.portInfo[ind:])
        )

    def char_sum(self, ls):
        return sum([int(x) for x in ls])

    def removeOldLibrary(self):
        '''
            Remove every block for this model name from the shared library and
            rewrite a clean, balanced file. Parse-based, so it correctly strips
            glued/duplicated blocks (which the old startswith() scan missed).
        '''
        parts = _read_parts(self.kicad_ngveri_sym)
        parts.pop(self.modelname, None)
        _write_lib(self.kicad_ngveri_sym, parts)

    def deleteKicadSymbol(self):
        '''
            Public entry point for the NgVeri "Remove Verilog Models" feature:
            drop this model's symbol from eSim_Ngveri.kicad_sym. Idempotent and
            safe to call when the symbol is absent.
        '''
        self.removeOldLibrary()

    def _commit_block(self, block):
        '''
            Idempotently insert/overwrite this model's symbol block in the
            shared library, then re-serialize a valid balanced file. The block
            is verified to be a single balanced s-expression first, so a
            malformed block is rejected instead of poisoning the shared file.
        '''
        block = block.strip()
        if _balanced_end(block, 0) != len(block):
            raise ValueError(
                "Refusing to write malformed symbol block for '" +
                str(self.modelname) + "': not a single balanced s-expression")
        parts = _read_parts(self.kicad_ngveri_sym)
        parts[self.modelname] = block
        _write_lib(self.kicad_ngveri_sym, parts)

    def createSym(self):
        '''
            Build this model's KiCad symbol block (pins snapped to the KiCad-6
            grid) and commit it idempotently to the shared library. The block
            is assembled as a balanced string and handed to _commit_block(),
            which replaces any existing block of the same name and
            re-serializes a valid file — no raw byte/line surgery, so the
            library can never be left glued or unbalanced.
        '''
        self.grid = 0.635
        self.dist_port = 4 * self.grid         # Distance between two ports # 100 mil (= 2.54 mm)
        self.inc_size = self.dist_port      # Increment size of a block (mil)
        def snap(val):
                snapped = round(float(val) / self.grid) * self.grid
                return f"{snapped:.3f}"

        block = []                          # lines of this one (symbol ...)

        line1 = self.template["start_def"]
        line1 = line1.split()
        line1 = [w.replace('comp_name', self.modelname) for w in line1]
        self.template["start_def"] = ' '.join(line1)

        block.append(self.template["start_def"])
        block.append(self.template["U_field"])

        line3 = self.template["comp_name_field"]
        line3 = line3.split()
        line3 = [w.replace('comp_name', self.modelname) for w in line3]
        self.template["comp_name_field"] = ' '.join(line3)

        block.append(self.template["comp_name_field"])

        line4 = self.template["blank_field"]
        line4_1 = line4[0]
        line4_2 = line4[1]
        line4_1 = line4_1.split()
        line4_1 = [w.replace('blank_quotes', '""') for w in line4_1]
        line4_2 = line4_2.split()
        line4_2 = [w.replace('blank_quotes', '""') for w in line4_2]
        line4[0] = ' '.join(line4_1)
        line4[1] = ' '.join(line4_2)
        self.template["blank_qoutes"] = line4

        block.append(line4[0])
        block.append(line4[1])

        draw_pos = self.template["draw_pos"]
        draw_pos = draw_pos.split()

        draw_pos = \
            [w.replace('comp_name', f"{self.modelname}_0_1") for w in draw_pos]
        draw_pos[8] = snap(float(draw_pos[8]) +           # previously it is (-)
                          float(self.findBlockSize() * self.inc_size))
        draw_pos_rec = draw_pos[8]

        self.template["draw_pos"] = ' '.join(draw_pos)

        block.append(self.template["draw_pos"])
        block.append(
            self.template["start_draw"] + " \"" + f"{self.modelname}_1_1\""
        )

        input_port = self.template["input_port"]
        input_port = input_port.split()
        output_port = self.template["output_port"]
        output_port = output_port.split()
        input_port[3] = snap(float(input_port[3]))
        output_port[3] = snap(float(output_port[3]))
        inputs = self.portInfo[0: self.input_length]
        outputs = self.portInfo[self.input_length:]
        inputName = []
        outputName = []

        for i in range(self.input_length):
            for j in range(int(inputs[i])):
                inputName.append(
                    self.portName[i] + str(int(inputs[i]) - j - 1))

        for i in range(self.input_length, len(self.portName)):
            for j in range(int(outputs[i - self.input_length])):
                outputName.append(
                    self.portName[i] +
                    str(int(outputs[i - self.input_length]) - j - 1))

        inputs = self.char_sum(inputs)
        outputs = self.char_sum(outputs)

        total = inputs + outputs

        port_list = []

        # Set input & output port
        input_port[4] = draw_pos_rec
        output_port[4] = draw_pos_rec

        j = 0
        for i in range(total):
            if (i < inputs):
                input_port[9] = f"\"{inputName[i]}\""
                input_port[13] = f"\"{str(i + 1)}\""
                input_port[4] = \
                    snap(float(input_port[4]) - float(self.dist_port))
                input_list = ' '.join(input_port)
                port_list.append(input_list)
                j = j + 1

            else:
                output_port[9] = f"\"{outputName[i - inputs]}\""
                output_port[13] = f"\"{str(i + 1)}\""
                output_port[4] = \
                    snap(float(output_port[4]) - float(self.dist_port))
                output_list = ' '.join(output_port)
                port_list.append(output_list)

        for ports in port_list:
            block.append(ports)
        block.append(self.template["end_draw"])      # "))" closes _1_1 + part

        self._commit_block('\n'.join(block))


class PortInfo:
    '''
        The class contains port information
    '''
    def __init__(self, model, modelpath):
        self.modelname = model.modelname
        self.bit_list = []
        self.port_name = []
        self.input_len = 0
        self.modelpath = modelpath

    def getPortInfo(self):
        '''
            getting the port information from `connection_info.txt`
        '''
        input_list = []
        output_list = []
        read_file = open(self.modelpath + 'connection_info.txt', 'r')
        data = read_file.readlines()
        # print(data)
        read_file.close()

        for line in data:
            if re.match(r'^\s*$', line):
                pass
            else:
                in_items = re.findall(
                    "INPUT", line, re.MULTILINE | re.IGNORECASE
                )
                inout_items = re.findall(
                    "INOUT", line, re.MULTILINE | re.IGNORECASE
                )

                out_items = re.findall(
                    "OUTPUT", line, re.MULTILINE | re.IGNORECASE
                )
            if in_items:
                input_list.append(line.split())
            if inout_items:
                input_list.append(line.split())
            if out_items:
                output_list.append(line.split())

        for in_list in input_list:
            self.bit_list.append(in_list[2])
            self.port_name.append(in_list[0])
        self.input_len = len(self.bit_list)
        for out_list in output_list:
            self.bit_list.append(out_list[2])
            self.port_name.append(out_list[0])
