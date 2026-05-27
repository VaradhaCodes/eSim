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
import xml.etree.cElementTree as ET
from PyQt6 import QtWidgets


_HEADER_RE = re.compile(r'\(kicad_symbol_lib[^\n]*\n')
_SYM_NAME_RE = re.compile(r'\(\s*symbol\s+"([^"]+)"')


def _parse_lib(content):
    """Split a kicad_sym file into (header, [blocks], footer).

    Blocks are top-level (symbol ...) s-expressions inside
    (kicad_symbol_lib ...).  Paren-balance counting is used so that
    blocks whose opener has been merged onto a previous block's closing
    `))` (as historically produced by createSym) are still recognised.

    Returns (header, blocks, footer) or None if the file is malformed.
    """
    m = _HEADER_RE.match(content)
    if not m:
        return None
    header = m.group(0)
    rest = content[len(header):].rstrip()
    if not rest.endswith(')'):
        return None
    body = rest[:-1].rstrip()

    blocks = []
    i = 0
    n = len(body)
    while i < n:
        while i < n and body[i] in ' \t\r\n':
            i += 1
        if i >= n:
            break
        if body[i] != '(':
            i += 1
            continue
        start = i
        depth = 0
        in_string = False
        while i < n:
            ch = body[i]
            if in_string:
                if ch == '"' and body[i - 1] != '\\':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
            i += 1
        blocks.append(body[start:i])
    return header, blocks, ')\n'


def _render_lib(header, blocks, footer):
    if not blocks:
        return header + '\n' + footer
    return header + '\n' + '\n\n'.join(blocks) + '\n' + footer


def _remove_blocks(content, modelname):
    """Return content with every top-level entry related to *modelname*
    removed.  In addition to the canonical `(symbol "modelname" ...)`
    block, this also strips:

      * sub-symbols (`modelname_0_1`, `modelname_1_1`, ...) that have
        bubbled up to the top level because a previous run wrote a
        block without its opener;
      * orphan top-level entries that are not `(symbol ...)` at all
        (stray `(property ...)` lines from the same corruption).

    If the header itself is malformed the content is returned
    unchanged."""
    parsed = _parse_lib(content)
    if parsed is None:
        return content
    header, blocks, footer = parsed
    sub_re = re.compile(r'^' + re.escape(modelname) + r'_\d+_\d+$')
    kept = []
    for blk in blocks:
        m = _SYM_NAME_RE.match(blk)
        if not m:
            continue
        name = m.group(1)
        if name == modelname or sub_re.match(name):
            continue
        kept.append(blk)
    return _render_lib(header, kept, footer)


def _append_block(content, new_block):
    """Insert new_block as the last top-level entry before the final `)`.

    Falls back to writing a fresh empty library if content is empty or
    malformed."""
    if not content.strip():
        header = ("(kicad_symbol_lib (version 20211014) "
                  "(generator kicad_symbol_editor)\n")
        return _render_lib(header, [new_block], ')\n')
    parsed = _parse_lib(content)
    if parsed is None:
        header = ("(kicad_symbol_lib (version 20211014) "
                  "(generator kicad_symbol_editor)\n")
        return _render_lib(header, [new_block], ')\n')
    header, blocks, footer = parsed
    blocks.append(new_block)
    return _render_lib(header, blocks, footer)


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
            # The kicad_sym file may still hold a stale block from an
            # earlier aborted run or a manually-deleted XML — clean
            # before appending so we don't accumulate duplicates.
            self.removeOldLibrary()
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
            Remove every top-level (symbol "<modelname>" ...) block from
            eSim_Ngveri.kicad_sym.  Uses paren-balanced s-expression
            parsing so it works regardless of how the block opener is
            laid out on the line (handles legacy `))(symbol "X"...`
            merged-line writes too).
        '''
        if not os.path.exists(self.kicad_ngveri_sym):
            return
        with open(self.kicad_ngveri_sym, 'r') as f:
            content = f.read()
        new_content = _remove_blocks(content, self.modelname)
        with open(self.kicad_ngveri_sym, 'w', newline='') as f:
            f.write(new_content)

    def createSym(self):
        '''
            Build the (symbol ...) block for this model in memory and
            insert it into eSim_Ngveri.kicad_sym before the file's
            closing `)`.  Pins are snapped to the KiCad-6 grid.
        '''
        self.grid = 0.635
        self.dist_port = 4 * self.grid         # 100 mil between ports
        self.inc_size = self.dist_port

        def snap(val):
            snapped = round(float(val) / self.grid) * self.grid
            return f"{snapped:.3f}"

        start_def = self.template["start_def"].replace(
            'comp_name', self.modelname)
        u_field = self.template["U_field"]
        comp_name_field = self.template["comp_name_field"].replace(
            'comp_name', self.modelname)
        blank_field_0 = self.template["blank_field"][0].replace(
            'blank_quotes', '""')
        blank_field_1 = self.template["blank_field"][1].replace(
            'blank_quotes', '""')

        draw_pos = self.template["draw_pos"].replace(
            'comp_name', f"{self.modelname}_0_1").split()
        draw_pos[8] = snap(
            float(draw_pos[8]) +
            float(self.findBlockSize() * self.inc_size)
        )
        draw_pos_rec = draw_pos[8]
        draw_pos_line = ' '.join(draw_pos)

        start_draw_line = (
            self.template["start_draw"] + " \"" + f"{self.modelname}_1_1\""
        )

        input_port = self.template["input_port"].split()
        output_port = self.template["output_port"].split()
        input_port[3] = snap(float(input_port[3]))
        output_port[3] = snap(float(output_port[3]))
        inputs = self.portInfo[0:self.input_length]
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
        input_port[4] = draw_pos_rec
        output_port[4] = draw_pos_rec

        for i in range(total):
            if i < inputs:
                input_port[9] = f"\"{inputName[i]}\""
                input_port[13] = f"\"{str(i + 1)}\""
                input_port[4] = snap(
                    float(input_port[4]) - float(self.dist_port))
                port_list.append(' '.join(input_port))
            else:
                output_port[9] = f"\"{outputName[i - inputs]}\""
                output_port[13] = f"\"{str(i + 1)}\""
                output_port[4] = snap(
                    float(output_port[4]) - float(self.dist_port))
                port_list.append(' '.join(output_port))

        block_lines = [
            start_def,
            u_field,
            comp_name_field,
            blank_field_0,
            blank_field_1,
            draw_pos_line,
            start_draw_line,
        ]
        block_lines.extend(port_list)
        block_lines.append(self.template["end_draw"])
        new_block = '\n'.join(block_lines)

        content = ''
        if os.path.exists(self.kicad_ngveri_sym):
            with open(self.kicad_ngveri_sym, 'r') as f:
                content = f.read()

        new_content = _append_block(content, new_block)
        with open(self.kicad_ngveri_sym, 'w', newline='') as f:
            f.write(new_content)


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
