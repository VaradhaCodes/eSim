# =========================================================================
#             FILE: ModelGeneration.py
#
#            USAGE: ---
#
#      DESCRIPTION: This define all model generation processes of NgVeri.
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
#      REVISION: Tuesday 2nd, September 2023
# =========================================================================


import re
import os
from PyQt6 import QtCore, QtWidgets
from configparser import ConfigParser
from configuration import Appconfig

from . import createkicad
from . import CosimConfig
from .CosimLogger import CosimLog
import hdlparse.verilog_parser as vlog


class ModelGeneration(QtWidgets.QWidget):
    '''
        Class is used to generate the Ngspice Model
    '''

    # Generous cap (ms) so big verilator/make builds are not guillotined at
    # the old 50 s limit, while a genuinely hung process is still killed and
    # reported instead of either freezing the GUI forever or silently
    # producing a half-built model.
    PROCESS_TIMEOUT = 600000        # 10 minutes

    def __init__(self, file, termedit):
        QtWidgets.QWidget.__init__(self)
        super().__init__()
        self.obj_Appconfig = Appconfig.Appconfig()
        print("Argument is : ", file)

        if os.name == 'nt':
            self.file = file.replace('\\', '/')
        else:
            self.file = file

        self.termedit = termedit
        # Dual-sink d_cosim logger: same events to the NgVeri GUI terminal
        # (this termedit) AND the OS terminal + ~/.esim/dcosim.log.
        self.clog = CosimLog(termedit)
        self.cur_dir = os.getcwd()
        self.fname = os.path.basename(file)
        self.fname = self.fname.lower()
        print("Verilog/SystemVerilog/TL Verilog filename is : ", self.fname)

        if os.name == 'nt':
            self.home = os.path.join('library', 'config')
        else:
            self.home = os.path.expanduser('~')

        self.parser = ConfigParser()
        self.parser.read(os.path.join(
            self.home, os.path.join('.nghdl', 'config.ini')))
        self.nghdl_home = self.parser.get('NGHDL', 'NGHDL_HOME')
        self.release_dir = self.parser.get('NGHDL', 'RELEASE')
        self.src_home = self.parser.get('SRC', 'SRC_HOME')
        self.licensefile = self.parser.get('SRC', 'LICENSE')
        self.digital_home = self.parser.get(
                            'NGHDL', 'DIGITAL_MODEL') + "/Ngveri"

    def _run(self, cmd, title, cwd=None):
        '''
            Run one shell command of the model-build pipeline, streaming its
            stdout and stderr live into the NgVeri terminal, and return True
            only when the process exits cleanly with code 0. The working
            directory is always restored afterwards (even on error/timeout),
            so a failed step can never leave eSim stuck inside a model
            sub-directory. This replaces the old "fire QProcess and assume it
            worked" pattern that made every compile failure look like success.
        '''
        prev_dir = os.getcwd()
        try:
            if cwd:
                os.chdir(cwd)
            self.process = QtCore.QProcess(self)
            self.process \
                .readyReadStandardOutput.connect(self.readAllStandard)
            self.process \
                .readyReadStandardError.connect(self.readAllStandard)
            self.termtitle(title)
            self.termtext("Current Directory: " + (cwd or prev_dir))
            self.termtext("Command: " + cmd)
            self.process.start('sh', ['-c', cmd])
            if not self.process.waitForFinished(self.PROCESS_TIMEOUT):
                self.process.kill()
                self.process.waitForFinished(2000)
                self.termtext("[NgVeri] '" + title +
                              "' timed out and was stopped.")
                return False
            ok = (self.process.exitStatus() ==
                  QtCore.QProcess.ExitStatus.NormalExit and
                  self.process.exitCode() == 0)
            if not ok:
                self.termtext("[NgVeri] '" + title + "' failed (exit code " +
                              str(self.process.exitCode()) + ").")
            return ok
        finally:
            os.chdir(prev_dir)

    def verilogfile(self):
        '''
            Reading the file and performing operations and
            copying it in the Ngspice folder
        '''
        Text = "<span style=\" font-size:25pt;\
         font-weight:1000; color:#008000;\" >"
        Text += ".................Running NgVeri..................."
        Text += "</span>"
        self.termedit.append(Text)

        with open(self.file, 'r') as read_verilog:
            verilog_data = read_verilog.readlines()
        self.modelpath = self.digital_home + \
            "/" + self.fname.split('.')[0] + "/"
        if not os.path.isdir(self.modelpath):
            os.mkdir(self.modelpath)

        if self.fname.split('.')[1] == "tlv":
            self.sandpiper()
            with open(self.modelpath + self.fname, 'r') as read_verilog:
                verilog_data = read_verilog.readlines()
        with open(self.modelpath + self.fname, 'w') as f:
            for item in verilog_data:
                if self.fname.split('.')[1] == "sv":
                    string = item.replace("top", self.fname.split('.')[0])
                else:
                    string = item
                f.write(string)
            f.write("\n")

    def sandpiper(self):
        '''
            This function calls the sandpiper to convert .tlv file to .sv file
        '''
        init_path = '../../'
        if os.name == 'nt':
            init_path = ''
        # Text="Running Sandpiper............"
        print("Running Sandpiper-Saas for TLV to SV Conversion")
        self.cmd = "cp " + init_path + "library/tlv/clk_gate.v " + \
                   init_path + "library/tlv/pseudo_rand.sv " + \
                   init_path + "library/tlv/sandpiper.vh " + \
                   init_path + "library/tlv/sandpiper_gen.vh " + \
                   init_path + "library/tlv/sp_default.vh " + \
                   init_path + "library/tlv/pseudo_rand_gen.sv " + \
                   init_path + "library/tlv/pseudo_rand.m4out.tlv " + \
                   self.file + " " + self.modelpath
        self._run(self.cmd, "COPY TLV FILES")
        print("Copied the files required for TLV successfully")

        print("Running Sandpiper............")
        self.cmd = "sandpiper-saas -i " + \
            self.fname.split('.')[0] + ".tlv -o "\
            + self.fname.split('.')[0] + ".sv"
        self._run(self.cmd, "RUN SANDPIPER-SAAS", cwd=self.modelpath)
        print("Ran Sandpiper successfully")
        self.fname = self.fname.split('.')[0] + ".sv"

    def verilogParse(self, make_symbol=True):
        '''
            This function parses the module name and
            input/output ports of verilog code using HDL parse
            and writes to the "connection_info.txt".

            make_symbol=False skips creating the legacy "Ngveri" KiCad symbol,
            so the d_cosim flow can reuse the port parsing and then create its
            own "NgVeriCosim" symbol instead.
        '''
        with open(self.modelpath + self.fname, 'rt') as fh:
            code = fh.read()

        code = code.replace("wire", " ")
        code = code.replace("reg", " ")
                
        header_re = re.compile(r'module\s+\w+\s*\((.*?)\)\s*;', re.S)
        def _split_ports(match):
            # add a newline after every comma that is inside the header
            return match.group(0).replace(',', ',\n')
        code = header_re.sub(_split_ports, code)
        vlog_ex = vlog.VerilogExtractor()
        vlog_mods = vlog_ex.extract_objects_from_source(code)
        with open(self.modelpath + "connection_info.txt", 'w') as f:
            for m in vlog_mods:
                if m.name.lower() == self.fname.split('.')[0]:
                    print(str(m.name) + " " + self.fname.split('.')[0])
                    for p in m.ports:
                        print(p.data_type)
                        if str(p.data_type).find(':') == -1:
                            p.port_number = "1"
                        else:
                            x = p.data_type.split(":")
                            print(x)
                            y = x[0].split("[")
                            z = x[1].split("]")
                            z = int(y[1]) - int(z[0])
                            p.port_number = z + 1

            for m in vlog_mods:
                if m.name.lower() == self.fname.split('.')[0]:
                    m.name = m.name.lower()
                    print('Module "{}":'.format(m.name))
                    for p in m.generics:
                        print('\t{:20}{:8}{}'.format(
                            p.name, p.mode, p.data_type))
                    print('  Ports:')
                    for p in m.ports:
                        print(
                            '\t{:20}{:8}{}'.format(
                                p.name, p.mode, p.port_number))
                        f.write(
                            '\t{:20}{:8}{}\n'.format(
                                p.name, p.mode, p.port_number))
                    break
        if m.name.lower() != self.fname.split(".")[0]:
            QtWidgets.QMessageBox.critical(
                None,
                "Error Message",
                "<b>Error: File name and module \
                name are not same. Please ensure that they are same</b>",
                QtWidgets.QMessageBox.StandardButton.Ok)

            self.obj_Appconfig.print_info(
                'NgVeri stopped due to file \
                name and module name not matching error')
            return "Error"
        if make_symbol:
            modelname = str(m.name)
            schematicLib = createkicad.AutoSchematic()
            schematicLib.init(modelname, self.modelpath)
            error = schematicLib.createKicadSymbol()
            if error == "Error":
                return "Error"
        return "No Error"

    def getPortInfo(self):
        '''
            This function is used to get the port information
            from "connection_info.txt"
        '''
        with open(self.modelpath + 'connection_info.txt', 'r') as readfile:
            data = readfile.readlines()
        self.input_list = []
        self.output_list = []
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
                self.input_list.append(line.split())
            if inout_items:
                self.input_list.append(line.split())
            if out_items:
                self.output_list.append(line.split())

        self.input_port = []
        self.output_port = []

        # creating list of input and output port with its weight
        for input in self.input_list:
            self.input_port.append(input[0] + ":" + input[2])
        for output in self.output_list:
            self.output_port.append(output[0] + ":" + output[2])

    def build_cosim(self, engine="icarus"):
        '''
            Build a d_cosim digital artifact for this Verilog model and return
            its absolute path (or "Error").

            Uses ngspice's upstream d_cosim code model (ngspice >= 44): the
            Verilog block is loaded at simulation time, so ngspice is never
            rebuilt -- unlike the legacy static Ngveri.cm flow that runs
            "make install".

            Icarus engine (default): iverilog compiles <model>.v to a vvp-format
            file named <model>. NO C/C++ compiler is needed on the user machine;
            at simulation time ngspice's ivlng adapter + libvvp run the vvp. The
            iverilog path is resolved via CosimConfig (env / config.ini / PATH),
            never hardcoded. Requires self.modelpath populated by verilogfile().
        '''
        import subprocess
        import tempfile
        import time
        import shlex

        log = self.clog
        log.phase("BUILD d_cosim MODEL (icarus)")

        if engine != "icarus":
            log.error("d_cosim engine '" + str(engine) + "' not supported. "
                      "Only the Icarus Verilog engine is available.")
            return "Error"

        # ----- [1/4] Resolve toolchain -----
        log.phase("[1/4] Resolve toolchain")
        iverilog = CosimConfig.iverilog_binary()
        if not iverilog or not CosimConfig.has_iverilog():
            log.error("d_cosim build FAILED: " +
                      (CosimConfig.missing_reason() or
                       "iverilog with libvvp not found."))
            log.fix("Install / rebuild Icarus Verilog with --enable-libvvp, "
                    "then retry.")
            return "Error"
        log.info("iverilog: " + iverilog)
        log.detail("version: " + self._tool_version(iverilog))

        model = self.fname.split('.')[0]
        src = os.path.abspath(os.path.join(self.modelpath, self.fname))
        # Build the vvp at the ONE canonical location the netlister also
        # derives (CosimConfig.cosim_vvp_path, keyed by the lowercased model
        # name). Decoupling it from modelpath's case is what stops the compiled
        # model from going missing at simulation time on case-sensitive
        # filesystems (build wrote <Model>/<Model>, lookup read <model>/<model>).
        out = CosimConfig.cosim_vvp_path(model.lower())
        if out:
            os.makedirs(os.path.dirname(out), exist_ok=True)
        else:
            out = os.path.abspath(os.path.join(self.modelpath, model.lower()))
        log.info("Model:       " + model)
        log.info("Source:      " + src)
        log.info("Output vvp:  " + out)

        if not os.path.isfile(src):
            log.error("d_cosim build FAILED: source Verilog not found at " +
                      src)
            log.fix("The model dir was not populated (or was removed after a "
                    "backend switch). Re-run the build; verilogfile() should "
                    "copy the .v in first.")
            return "Error"

        # eSim's port parser lumps `inout` into the input side, but d_cosim
        # keeps a separate d_inout group -- warn rather than emit a wrong
        # netlist silently. (Common no-inout modules are unaffected.)
        try:
            with open(os.path.join(self.modelpath, 'connection_info.txt')) as f:
                if 'inout' in f.read().lower():
                    log.warn("Module has inout port(s). d_cosim handling of "
                             "inout is limited; results may be wrong.")
                    log.fix("Use the legacy NgVeri flow if inout is required.")
        except OSError:
            pass

        try:
            # ----- [2/4] Prepare source -----
            log.phase("[2/4] Prepare source")
            # d_cosim/ivlng needs a `timescale to advance VVP ticks; without one
            # the tick length defaults to 1 second and combinational logic never
            # re-evaluates. Inject one transparently if the source lacks it.
            with open(src, 'r') as fh:
                verilog_text = fh.read()
            compile_src = src
            tmp_src = None
            if '`timescale' not in verilog_text:
                tmp_fd, tmp_src = tempfile.mkstemp(
                    suffix='.v', dir=os.path.abspath(self.modelpath))
                os.write(tmp_fd,
                         ('`timescale 1ns/1ps\n' + verilog_text).encode())
                os.close(tmp_fd)
                compile_src = tmp_src
                log.info("Injected `timescale 1ns/1ps (absent in source).")
            else:
                log.detail("`timescale present in source.")

            # ----- [3/4] Compile -----
            log.phase("[3/4] Compile")
            cmd = [iverilog, "-g2012", "-o", out, compile_src]
            log.info("$ " + " ".join(shlex.quote(c) for c in cmd))
            start = time.monotonic()
            try:
                proc = subprocess.run(
                    cmd, cwd=os.path.abspath(self.modelpath),
                    capture_output=True, text=True, timeout=300)
            finally:
                if tmp_src and os.path.isfile(tmp_src):
                    os.remove(tmp_src)
            elapsed = time.monotonic() - start
            log.output(proc.stdout, 'stdout')
            log.output(proc.stderr, 'stderr')
            log.info("iverilog exited rc=%d in %.2fs"
                     % (proc.returncode, elapsed))

            # ----- [4/4] Verify artifact -----
            log.phase("[4/4] Verify artifact")
            if proc.returncode != 0 or not os.path.isfile(out):
                log.error("d_cosim model build FAILED (rc=%d)."
                          % proc.returncode)
                log.fix("Check the compiler errors above (syntax, missing "
                        "module, or a construct Icarus -g2012 rejects).")
                return "Error"
            log.ok("Built d_cosim model: %s (%d bytes)"
                   % (out, os.path.getsize(out)))
            return out
        except subprocess.TimeoutExpired:
            log.error("iverilog timed out after 300s.")
            log.fix("Simplify the design or raise the build timeout.")
            return "Error"
        except Exception as e:
            log.error("d_cosim build error: " + str(e))
            return "Error"

    def _tool_version(self, binary):
        '''
            First line of "<binary> -V", or "unknown". Best-effort: identifies
            which compiler actually ran, and never raises.
        '''
        try:
            import subprocess
            res = subprocess.run([binary, "-V"], capture_output=True,
                                 text=True, timeout=10)
            lines = (res.stdout or res.stderr or "").strip().splitlines()
            return lines[0] if lines and lines[0].strip() else "unknown"
        except Exception:
            return "unknown"

    def cfuncmod(self):
        '''
            This function is used to create the "cfunc.mod" file
            in Ngspice folder automatically.
        '''

        # ############# Creating content for cfunc.mod file ############## #

        print("Starting With cfunc.mod file")
        cfunc = open(self.modelpath + 'cfunc.mod', 'w')
        print("Building content for cfunc.mod file")

        comment = '''/* This cfunc.mod file auto generated by gen_con_info.py
        Developed by Sumanto, Rahul at IIT Bombay */\n
                '''

        header = '''
        #include <stdio.h>
        #include <math.h>
        #include <string.h>
        #include "sim_main_''' + self.fname.split('.')[0] + '''.h"

        '''

        function_open = (
            '''void cm_''' + self.fname.split('.')[0] + '''(ARGS) \n{''')

        digital_state_output = []
        for item in self.output_port:
            digital_state_output.append(
                "Digital_State_t *_op_" + item.split(':')[0] +
                ", *_op_" + item.split(':')[0] + "_old;"
            )

        var_section = '''
    static int inst_count=0;
    int count=0;
        '''

        # Start of INIT function
        init_start_function = '''
    if(INIT)
    {
        inst_count++;
        PARAM(instance_id)=inst_count;
        foo_''' + self.fname.split('.')[0] + '''(0,inst_count);
        /* Allocate storage for output ports \
and set the load for input ports */

        '''
        port_init = []
        for i, item in enumerate(self.input_port + self.output_port):
            port_init.append(self.fname.split('.')[0] + '''_port_''' +
                             item.split(':')[0] + '''=PORT_SIZE(''' +
                             item.split(':')[0] + ''');
''')

        cm_event_alloc = []
        cm_count_output = 0
        for item in self.output_port:
            cm_event_alloc.append(
                "cm_event_alloc(" +
                str(cm_count_output) + "," + item.split(':')[1] +
                "*sizeof(Digital_State_t));"
            )
            cm_count_output = cm_count_output + 1

        load_in_port = []
        for item in self.input_port:
            load_in_port.append(
                "for(Ii=0;Ii<PORT_SIZE(" + item.split(':')[0] +
                ");Ii++)\n\t\t{\n\t\t\tLOAD(" + item.split(':')[0] +
                "[Ii])=PARAM(input_load); \n\t\t}"
            )

        cm_count_ptr = 0
        cm_event_get_ptr = []
        for item in self.output_port:
            cm_event_get_ptr.append(
                "_op_" + item.split(':')[0] + " = _op_" +
                item.split(':')[0] +
                "_old = (Digital_State_t *) cm_event_get_ptr(" +
                str(cm_count_ptr) + ",0);"
            )

            cm_count_ptr = cm_count_ptr + 1

        els_evt_ptr = []
        els_evt_count1 = 0
        els_evt_count2 = 0
        for item in self.output_port:
            els_evt_ptr.append("_op_" + item.split(":")[0] +
                               " = (Digital_State_t *) cm_event_get_ptr(" +
                               str(els_evt_count1) + "," +
                               str(els_evt_count2) + ");")
            els_evt_count2 = els_evt_count2 + 1
            els_evt_ptr.append("_op_" + item.split(":")[0] + "_old" +
                               " = (Digital_State_t *) cm_event_get_ptr(" +
                               str(els_evt_count1) + "," +
                               str(els_evt_count2) + ");")
            els_evt_count1 = els_evt_count1 + 1

        # Assign bit value to every input
        assign_data_to_input = []
        for item in self.input_port:
            assign_data_to_input.append("\
    for(Ii=0;Ii<PORT_SIZE(" + item.split(':')[0] + ");Ii++)\n\
    {\n\
        if( INPUT_STATE(" + item.split(':')[0] + "[Ii])==ZERO )\n\
        {\n\
            " + self.fname.split('.')[0] +
                "_temp_" + item.split(':')[0] + "[Ii]=0;\
            }\n\
        else\n\
        {\n\
            " + self.fname.split('.')[0] +
                "_temp_" + item.split(':')[0] + "[Ii]=1;\n\
        }\n\
            }\n")

        # Scheduling output event
        sch_output_event = []

        for item in self.output_port:
            sch_output_event.append(
                "\t/* Scheduling event and processing them */\n\
    for(Ii=0;Ii<PORT_SIZE(" + item.split(':')[0] + ");Ii++)\n\
    {\n\
        if(" + self.fname.split('.')[0] + "_temp_" +
                item.split(':')[0] + "[Ii]==0)\n\
        {\n\
            _op_" + item.split(':')[0] + "[Ii]=ZERO;\n\
            }\n\
        else if(" + self.fname.split('.')[0] +
                "_temp_" + item.split(':')[0] + "[Ii]==1)\n\
        {\n\
            _op_" + item.split(':')[0] + "[Ii]=ONE;\n\
            }\n\
        else\n\
        {\n\
            printf(\"Unknown value\\n\");\n\
                }\n\n\
        if(ANALYSIS == DC)\n\
        {\n\
            OUTPUT_STATE(" + item.split(':')[0] +
                "[Ii]) = _op_" + item.split(':')[0] + "[Ii];\n\
            }\n\
        else if(_op_" + item.split(':')[0] +
                "[Ii] != _op_" + item.split(':')[0] + "_old[Ii])\n\
        {\n\
            OUTPUT_STATE(" + item.split(':')[0] + "[Ii]) = _op_" +
                item.split(':')[0] + "[Ii];\n\
            OUTPUT_DELAY(" + item.split(':')[0] + "[Ii]) = ((_op_" +
                item.split(':')[0] +
                "[Ii] == ZERO) ? PARAM(fall_delay) : PARAM(rise_delay));\n\
            }\n\
        else\n\
        {\n\
            OUTPUT_CHANGED(" + item.split(':')[0] + "[Ii]) = FALSE;\n\
            }\n\
        OUTPUT_STRENGTH(" + item.split(':')[0] + "[Ii]) = STRONG;\n\
    }\n")

        # Writing content in cfunc.mod file
        cfunc.write(comment)
        cfunc.write(header)
        cfunc.write("\n")
        cfunc.write(function_open)
        cfunc.write("\n")

        # Adding digital state Variable
        for item in digital_state_output:
            cfunc.write("\t" + item + "\n")

        # Adding variable declaration section
        cfunc.write(var_section)

        # Adding INIT portion
        cfunc.write(init_start_function)
        for item in port_init:
            cfunc.write(item)
        for item in cm_event_alloc:
            cfunc.write(2 * "\t" + item)
            cfunc.write("\n")

        cfunc.write(2 * "\t" + "/* set the load for input ports. */")
        cfunc.write("\n")
        cfunc.write(2 * "\t" + "int Ii;")
        cfunc.write("\n")

        for item in load_in_port:
            cfunc.write(2 * "\t" + item)
            cfunc.write("\n")
        cfunc.write("\n")
        cfunc.write(2 * "\t" + "/*Retrieve Storage for output*/")
        cfunc.write("\n")
        for item in cm_event_get_ptr:
            cfunc.write(2 * "\t" + item)
            cfunc.write("\n")
        cfunc.write("\n")

        # if os.name == 'nt':
        #     digital_home = parser.get('NGHDL', 'DIGITAL_MODEL')
        #     msys_home = parser.get('COMPILER', 'MSYS_HOME')
        #     cmd_str2 = "/start_server.sh %d %s & read" + "\\" + "\"" + "\""
        #     cmd_str1 = os.path.normpath(
        #                         "\"" + digital_home + "/" +
        #                         fname.split('.')[0] + "/DUTghdl/"
        #     )
        #     cmd_str1 = cmd_str1.replace("\\", "/")

        #     cfunc.write(
        #         '\t\tsnprintf(command,1024, "start mintty.exe -t ' +
        #         '\\"VHDL-Testbench Logs\\" -h always bash.exe -c ' +
        #         '\\' + cmd_str1 + cmd_str2 + ', sock_port, my_ip);'
        #     )
        # else:
        #     cfunc.write(
        #         '\t\tsnprintf(command,1024,"' + home +
        #         '/nghdl-simulator/src/xspice/icm/ghdl/' +
        #         fname.split('.')[0] +
        #         '/DUTghdl/start_server.sh %d %s &", sock_port, my_ip);'
        #     )

        cfunc.write("\n\t}")
        cfunc.write("\n")
        cfunc.write("\telse\n\t{\n")

        for item in els_evt_ptr:
            cfunc.write(2 * "\t" + item)
            cfunc.write("\n")
        cfunc.write("\t}")
        cfunc.write("\n\n")

        cfunc.write("\t//Formating data for sending it to client\n")
        cfunc.write("\tint Ii;\n")
        cfunc.write("\tcount=(int)PARAM(instance_id);\n\n")
        for item in assign_data_to_input:
            cfunc.write(item)

        cfunc.write("\tfoo_" + self.fname.split('.')[0] + "(1,count);\n\n")

        for item in sch_output_event:
            cfunc.write(item)

        # Close cm_ function
        cfunc.write("\n}")
        cfunc.close()

    def ifspecwrite(self):
        '''
            This function creates the ifspec file
            automatically in Ngspice folder.
        '''
        print("Starting with ifspec.ifs file")
        ifspec = open(self.modelpath + 'ifspec.ifs', 'w')

        print("Gathering Al the content for ifspec file")

        ifspec_comment = '''
        /*
        SUMMARY: This file is auto generated and it contains the interface
         specification for the code model. */\n
        '''

        name_table = 'NAME_TABLE:\n\
        C_Function_Name: cm_' + self.fname.split('.')[0] + '\n\
        Spice_Model_Name: ' + self.fname.split('.')[0] + '\n\
        Description: "Model generated from ghdl code ' + self.fname + '" \n'

        # Input and Output Port Table
        in_port_table = []
        out_port_table = []

        for item in self.input_port:
            port_table = 'PORT_TABLE:\n'
            port_name = 'Port_Name:\t' + item.split(':')[0] + '\n'
            description = (
                'Description:\t"input port ' + item.split(':')[0] + '"\n'
            )
            direction = 'Direction:\tin\n'
            default_type = 'Default_Type:\td\n'
            allowed_type = 'Allowed_Types:\t[d]\n'
            vector = 'Vector:\tyes\n'
            vector_bounds = (
                'Vector_Bounds:\t[' + item.split(':')[1] +
                ' ' + item.split(":")[1] + ']\n'
            )
            null_allowed = 'Null_Allowed:\tno\n'

            # Insert detail in the list
            in_port_table.append(
                port_table + port_name + description +
                direction + default_type + allowed_type +
                vector + vector_bounds + null_allowed
            )

        for item in self.output_port:
            port_table = 'PORT_TABLE:\n'
            port_name = 'Port_Name:\t' + item.split(':')[0] + '\n'
            description = (
                'Description:\t"output port ' + item.split(':')[0] + '"\n'
            )
            direction = 'Direction:\tout\n'
            default_type = 'Default_Type:\td\n'
            allowed_type = 'Allowed_Types:\t[d]\n'
            vector = 'Vector:\tyes\n'
            vector_bounds = (
                'Vector_Bounds:\t[' + item.split(':')[1] +
                ' ' + item.split(":")[1] + ']\n'
            )
            null_allowed = 'Null_Allowed:\tno\n'

            # Insert detail in the list
            in_port_table.append(
                port_table + port_name + description +
                direction + default_type + allowed_type +
                vector + vector_bounds + null_allowed
            )

        parameter_table = '''

        PARAMETER_TABLE:
        Parameter_Name:     instance_id                  input_load
        Description:        "instance_id"                "input load value (F)"
        Data_Type:          real                         real
        Default_Value:      0                            1.0e-12
        Limits:             -                            -
        Vector:              no                          no
        Vector_Bounds:       -                           -
        Null_Allowed:       yes                          yes

        PARAMETER_TABLE:
        Parameter_Name:     rise_delay                  fall_delay
        Description:        "rise delay"                "fall delay"
        Data_Type:          real                        real
        Default_Value:      1.0e-9                      1.0e-9
        Limits:             [1e-12 -]                   [1e-12 -]
        Vector:              no                          no
        Vector_Bounds:       -                           -
        Null_Allowed:       yes                         yes

        '''

        # Writing all the content in ifspec file
        ifspec.write(ifspec_comment)
        ifspec.write(name_table + "\n\n")

        for item in in_port_table:
            ifspec.write(item + "\n")

        ifspec.write("\n")

        for item in out_port_table:
            ifspec.write(item + "\n")

        ifspec.write("\n")
        ifspec.write(parameter_table)
        ifspec.write("\n")
        ifspec.close()

    def sim_main_header(self):
        '''
            This function creates the header file of
            "sim_main" file automatically in Ngspice folder.
        '''
        print("Starting With sim_main_" + self.fname.split('.')[0] + ".h file")
        simh = open(
            self.modelpath +
            'sim_main_' +
            self.fname.split('.')[0] +
            '.h',
            'w')
        print("Building content for sim_main_" +
              self.fname.split('.')[0] + ".h file")
        simh.write("int foo_" + self.fname.split('.')[0] + "(int,int);")
        extern_var = []
        for i, item in enumerate(self.input_port + self.output_port):
            extern_var.append('''
        int ''' + self.fname.split('.')[0] + '''_temp_''' +
                              item.split(':')[0] + '''[1024];
        int ''' + self.fname.split('.')[0] + '''_port_''' +
                              item.split(':')[0] + ''';''')
        for item in extern_var:
            simh.write(item)
        simh.close()

    def sim_main(self):
        '''
            This function creates the "sim_main" file needed by verilator
            automatically in Ngspice folder.
        '''
        print(
            "Starting With sim_main_" +
            self.fname.split('.')[0] +
            ".cpp file")
        csim = open(
            self.modelpath +
            'sim_main_' +
            self.fname.split('.')[0] +
            '.cpp',
            'w')
        print(
            "Building content for sim_main_" +
            self.fname.split('.')[0] +
            ".cpp file")

        comment = \
            '''/* This is cfunc.mod file auto generated by gen_con_info.py
        Developed by Sumanto Kar at IIT Bombay */\n
        '''

        header = '''
        #include <memory>
        #include <verilated.h>
        #include "V''' + self.fname.split('.')[0] + '''.h"
        #include <stdio.h>
        #include <stdio.h>
        #include <fstream>
        #include <stdlib.h>
        #include <string>
        #include <iostream>
        #include <cstring>
        using namespace std;
        '''

        extern_var = []
        for i, item in enumerate(self.input_port + self.output_port):
            extern_var.append('''
        extern "C" int ''' + self.fname.split('.')[0] +
                              '''_temp_''' + item.split(':')[0] + '''[1024];
        extern "C" int ''' + self.fname.split('.')[0] +
                              '''_port_''' + item.split(':')[0] + ''';''')

        extern_var.append('''
        extern "C" int foo_''' + self.fname.split('.')[0] + '''(int,int);
        ''')
        convert_func = '''
        void int2arr''' + self.fname.split('.')[0] + \
            '''(int  num, int array[], int n)
        {
            for (int i = 0; i < n && num>=0; i++)
            {
                array[n-i-1] = num % 2;
                num /= 2;
                }
        }
        int arr2int''' + self.fname.split('.')[0] + '''(int array[],int n)
        {
            int i,k=0;
            for (i = 0; i < n; i++)
                k = 2 * k + array[i];
            return k;
        }
        '''
        foo_func = '''
        int foo_''' + self.fname.split('.')[0] + '''(int init,int count)
        {
            int argc=1;
            const char* argv[]={"fullverbose"};
            Verilated::commandArgs(argc, argv);
            static VerilatedContext* contextp = new VerilatedContext;
            static V''' + self.fname.split('.')[0] + "* " + \
            self.fname.split('.')[0] + '''[1024];
            count--;
            if (init==0)
            {
                if (''' + self.fname.split('.')[0] + '''[count] != nullptr) {
                    ''' + self.fname.split('.')[0] + '''[count]->final();
                    delete ''' + self.fname.split('.')[0] + '''[count];
                    ''' + self.fname.split('.')[0] + '''[count] = nullptr;
                }
                contextp->time(0);
                ''' + self.fname.split('.')[0] + '''[count]=new V''' + \
            self.fname.split('.')[0] + '''{contextp};
                contextp->traceEverOn(true);
            }
            else
            {
                contextp->timeInc(1);
                printf("=============''' + self.fname.split('.')[0] + \
            ''' : New Iteration===========");
                printf("\\nInstance : %d\\n",count);
                printf("\\nInside foo before eval.....\\n");
'''

        before_eval = []
        after_eval = []
        for i, item in enumerate(self.input_port + self.output_port):
            before_eval.append(
                '''\t\t\t\tprintf("''' +
                item.split(':')[0] +
                '''=%d\\n", ''' +
                self.fname.split('.')[0] +
                '''[count] ->''' +
                item.split(':')[0] +
                ''');\n''')
        for i, item in enumerate(self.input_port):

            before_eval.append(
                '''\t\t\t\t''' +
                self.fname.split('.')[0] +
                '''[count]->''' +
                item.split(':')[0] +
                ''' = arr2int''' +
                self.fname.split('.')[0] +
                '''(''' + self.fname.split('.')[0] + '''_temp_''' +
                item.split(':')[0] +
                ''', ''' + self.fname.split('.')[0] + '''_port_''' +
                item.split(':')[0] +
                ''');\n''')
        before_eval.append(
            "\t\t\t\t" +
            self.fname.split('.')[0] +
            "[count]->eval();\n")

        after_eval.append('''
                printf("\\nInside foo after eval.....\\n");\n''')
        for i, item in enumerate(self.input_port + self.output_port):
            after_eval.append(
                '''\t\t\t\tprintf("''' +
                item.split(':')[0] +
                '''=%d\\n", ''' +
                self.fname.split('.')[0] +
                '''[count] ->''' +
                item.split(':')[0] +
                ''');\n''')

        for i, item in enumerate(self.output_port):
            after_eval.append(
                "\t\t\t\tint2arr" +
                self.fname.split('.')[0] +
                "(" +
                self.fname.split('.')[0] +
                '''[count] -> ''' +
                item.split(':')[0] +
                ''', ''' + self.fname.split('.')[0] + '''_temp_''' +
                item.split(':')[0] +
                ''', ''' + self.fname.split('.')[0] + '''_port_''' +
                item.split(':')[0] +
                ''');\n''')
        after_eval.append('''
            }
            return 0;
        }''')

        csim.write(comment)
        csim.write(header)
        for item in extern_var:
            csim.write(item)
        csim.write(convert_func)
        csim.write(foo_func)

        for item in before_eval:
            csim.write(item)
        for item in after_eval:
            csim.write(item)
        csim.close()

    def modpathlst(self):
        '''
            This function creates modpathlst in Ngspice folder.
        '''
        print("Editing modpath.lst file")
        with open(self.digital_home + '/modpath.lst', 'r') as mod:
            text = mod.read()
        # Exact-line membership: a plain "in text" substring test wrongly
        # treats "divider" as already present because "divider_8bit" contains
        # it, which silently drops the shorter model from Ngveri.cm.
        modname = self.fname.split('.')[0]
        with open(self.digital_home + '/modpath.lst', 'a+') as mod:
            if modname not in text.split():
                mod.write(modname + "\n")
        # Self-heal: a stale entry whose build dir was deleted (e.g. the model
        # was later removed via the d_cosim path, which nuked the shared
        # <model>/ dir but not this list) makes cmpp abort the ENTIRE Ngveri.cm
        # build -- "Unable to open <model>/ifspec.ifs". Drop such ghosts now so
        # one dead entry can't take every other model down with it.
        self.prune_modpathlst()

    def prune_modpathlst(self):
        '''
            Rewrite modpath.lst keeping only entries whose build dir still has
            an ifspec.ifs (what cmpp needs), and de-duplicating. Returns the
            list of dropped (ghost / duplicate) names; logs each via clog.

            This is the guard that keeps a single orphaned model -- the usual
            fallout of switching a model between the d_cosim and legacy NgVeri
            flows -- from breaking the build for all the others.
        '''
        path = self.digital_home + '/modpath.lst'
        try:
            with open(path) as f:
                entries = [ln.strip() for ln in f]
        except OSError:
            return []

        kept, dropped, seen = [], [], set()
        for name in entries:
            if not name:
                continue
            if name in seen:
                dropped.append(name)        # duplicate line
                continue
            ifs = os.path.join(self.digital_home, name, 'ifspec.ifs')
            if os.path.isfile(ifs):
                kept.append(name)
                seen.add(name)
            else:
                dropped.append(name)        # ghost: dir/ifspec.ifs gone

        if dropped:
            with open(path, 'w') as f:
                for name in kept:
                    f.write(name + "\n")
            for name in dropped:
                self.clog.warn(
                    'Pruned stale model "' + name + '" from modpath.lst '
                    '(its build dir / ifspec.ifs is missing).')
        return dropped

    def run_verilator(self):
        '''
            This function is used to run the Verilator
            using the verilator commands.
        '''
        init_path = '../../'
        if os.name == 'nt':
            init_path = ''

        wno = " "
        with open(init_path + "library/tlv/lint_off.txt") as file:
            for item in file.readlines():
                if item and item.strip():
                    wno += " -Wno-" + item.strip("\n")

        print("Running Verilator.............")
        self.release_home = self.parser.get('NGHDL', 'RELEASE')
        # print(self.modelpath)

        if os.name == 'nt':
            self.msys_home = self.parser.get('COMPILER', 'MSYS_HOME')
            self.cmd = "export VERILATOR_ROOT=" + self.msys_home + "/mingw64; "
        else:
            self.cmd = ''

        # self.cmd = self.cmd + "verilator -Wall " + wno + " \
        # --cc --exe --no-MMD --Mdir . -CFLAGS -fPIC sim_main_" + \
        #    self.fname.split('.')[0] + ".cpp " + self.fname
        self.cmd = self.cmd + "verilator --stats -O3 -CFLAGS\
         -O3 -LDFLAGS \"-static\" --x-assign fast \
         --x-initial fast --noassert  --bbox-sys -Wall " + wno + "\
         --cc --exe --no-MMD --Mdir . -CFLAGS\
          -fPIC -output-split 0 sim_main_" + \
            self.fname.split('.')[0] + ".cpp --autoflush  \
            -DBSV_RESET_FIFO_HEAD -DBSV_RESET_FIFO_ARRAY  " + self.fname
        return self._run(self.cmd, "RUN VERILATOR", cwd=self.modelpath)

    def make_verilator(self):
        '''
            Running make verilator using this function
        '''
        print("Make Verilator.............")

        if os.path.exists(self.modelpath + "../verilated.o"):
            os.remove(self.modelpath + "../verilated.o")

        if os.name == 'nt':
            # path to msys home directory
            self.msys_home = self.parser.get('COMPILER', 'MSYS_HOME')
            self.cmd = self.msys_home + "/mingw64/bin/mingw32-make.exe"
        else:
            self.cmd = "make"

        self.cmd = self.cmd + " -f V" + self.fname.split('.')[0]\
            + ".mk V" + self.fname.split(
            '.')[0] + "__ALL.a sim_main_" \
            + self.fname.split('.')[0] + ".o ../verilated.o ../verilated_threads.o"
        return self._run(self.cmd, "MAKE VERILATOR", cwd=self.modelpath)

    def copy_verilator(self):
        '''
            This function copies the verilator files/object files from
            "src/xspice/icm/Ngveri/ to release/src/xspice/icm/Ngveri/"
        '''
        print("Copying the required files to Release Folder.............")
        self.release_home = self.parser.get('NGHDL', 'RELEASE')
        ngveri_icm = self.release_home + "/src/xspice/icm/Ngveri/"
        model = self.fname.split('.')[0]
        # Per-model dir; keep a trailing slash so the os.remove guards below
        # actually target real files. Without it the paths glued to
        # ".../Ngveri/<model>sim_main_..." (note the missing slash), never
        # existed, so the stale-artifact cleanup was a silent no-op.
        path_icm = ngveri_icm + model + "/"
        if not os.path.isdir(path_icm):
            os.mkdir(path_icm)
        if os.path.exists(path_icm + "sim_main_" + model + ".o"):
            os.remove(path_icm + "sim_main_" + model + ".o")
        if os.path.exists(ngveri_icm + "verilated.o"):
            os.remove(ngveri_icm + "verilated.o")
        if os.path.exists(ngveri_icm + "verilated_threads.o"):
            os.remove(ngveri_icm + "verilated_threads.o")
        if os.path.exists(path_icm + "V" + model + "__ALL.a"):
            os.remove(path_icm + "V" + model + "__ALL.a")
        # print(self.modelpath)
        self.cmd = "cp sim_main_" + model + ".o V" + \
            model + "__ALL.a " + path_icm
        ok1 = self._run(self.cmd, "COPYING FILES", cwd=self.modelpath)
        self.cmd = "cp ../verilated.o ../verilated_threads.o " + ngveri_icm
        ok2 = self._run(self.cmd, "COPYING FILES", cwd=self.modelpath)
        print("Copied the files")
        return ok1 and ok2

    def runMake(self):
        '''
            Running the make command for Ngspice
        '''
        print("run Make Called")
        self.release_home = self.parser.get('NGHDL', 'RELEASE')
        path_icm = os.path.join(self.release_home, "src/xspice/icm")

        if os.name == 'nt':
            # path to msys home directory
            self.msys_home = self.parser.get('COMPILER', 'MSYS_HOME')
            self.cmd = self.msys_home + "/mingw64/bin/mingw32-make.exe"
        else:
            self.cmd = "make"

        print("Running Make command in " + path_icm)
        return self._run(self.cmd, "MAKE COMMAND", cwd=path_icm)

    def runMakeInstall(self):
        '''
            Running the make install command for Ngspice
        '''
        print("run Make Install Called")
        self.release_home = self.parser.get('NGHDL', 'RELEASE')
        path_icm = os.path.join(self.release_home, "src/xspice/icm")

        if os.name == 'nt':
            self.msys_home = self.parser.get('COMPILER', 'MSYS_HOME')
            self.cmd = self.msys_home + \
                "/mingw64/bin/mingw32-make.exe install"
        else:
            self.cmd = "make install"
        print("Running Make Install")
        return self._run(self.cmd, "MAKE INSTALL COMMAND", cwd=path_icm)

    def addfile(self):
        '''
            This function is used to add additional files
            required by the verilog top module.
        '''
        print("Adding the files required by the top level module file")

        init_path = '../../'
        if os.name == 'nt':
            init_path = ''

        includefile = QtCore.QDir.toNativeSeparators(
            QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Open adding other necessary files to be included",
                init_path + "home")[0])

        if includefile == "":
            reply = QtWidgets.QMessageBox.critical(
                None, "Error Message",
                "<b>Error: No File Chosen. Please chose a file</b>",
                QtWidgets.QMessageBox.StandardButton.Ok | QtWidgets.QMessageBox.StandardButton.Cancel
            )

            if reply == QtWidgets.QMessageBox.StandardButton.Ok:
                self.addfile()

                if includefile == "":
                    return

                self.obj_Appconfig.print_info('Add Other Files Called')

            elif reply == QtWidgets.QMessageBox.StandardButton.Cancel:
                self.obj_Appconfig.print_info('No File Chosen')
                return

        filename = os.path.basename(includefile)
        self.modelpath = self.digital_home + \
            "/" + self.fname.split('.')[0] + "/"

        if not os.path.isdir(self.modelpath):
            os.mkdir(self.modelpath)
        with open(includefile) as fh:
            text = fh.read()
        text = text + '\n'
        with open(self.modelpath + filename, 'w') as f:
            for item in text:
                f.write(item)
            f.write("\n")
        print("Added the File:" + filename)
        self.termtitle("Added the File:" + filename)

    def addfolder(self):
        '''
            This function is used to add additional folder required
            by the verilog top module
        '''
        # self.cur_dir = os.getcwd()
        print("Adding the folder required by the top level module file")

        includefolder = QtCore.QDir.toNativeSeparators(
            QtWidgets.QFileDialog.getExistingDirectory(
                self, "open", "home"
            )
        )

        if includefolder == "":
            reply = QtWidgets.QMessageBox.critical(
                None, "Error Message",
                "<b>Error: No Folder Chosen. Please chose a folder</b>",
                QtWidgets.QMessageBox.StandardButton.Ok | QtWidgets.QMessageBox.StandardButton.Cancel
            )

            if reply == QtWidgets.QMessageBox.StandardButton.Ok:
                self.addfolder()

                if includefolder == "":
                    return

                self.obj_Appconfig.print_info('Add Folder Called')

            elif reply == QtWidgets.QMessageBox.StandardButton.Cancel:
                self.obj_Appconfig.print_info('No Folder Chosen')
                return

        self.modelpath = self.digital_home + \
            "/" + self.fname.split('.')[0] + "/"

        reply = QtWidgets.QMessageBox.question(
            None, "Message",
            '''<b>If you want only the contents\
             of the folder to be added press "Yes".\
                    If you want complete folder \
                    to be added, press "No". </b>''',
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self.cmd = "cp -a " + includefolder + "/. " + self.modelpath
            self.obj_Appconfig.print_info('Adding Contents of the Folder')
        elif reply == QtWidgets.QMessageBox.StandardButton.No:
            self.cmd = "cp -R " + includefolder + " " + self.modelpath
            self.obj_Appconfig.print_info('Adding the Folder')

        print("Adding the Folder:" + includefolder.split('/')[-1])
        self._run(self.cmd,
                  "Adding the Folder:" + includefolder.split('/')[-1])
        print("Added the folder")

    def termtitle(self, textin):
        '''
            This function is used to print the titles
            in the terminal of Ngveri tab.
        '''
        Text = "<span style=\" font-size:20pt; \
        font-weight:1000; color:#0000FF;\" >"
        Text += "<br>================================<br>"
        Text += textin
        Text += "<br>================================<br>"
        Text += "</span>"
        self.termedit.append(Text)

    def termtext(self, textin):
        '''
            This function is used to print the text/commands
            in the terminal of Ngveri tab.
        '''
        Text = "<span style=\" font-size:12pt;\
         font-weight:500; color:#000000;\" >"
        Text += textin
        Text += "</span>"
        self.termedit.append(Text)

    @QtCore.pyqtSlot()
    def readAllStandard(self):
        '''
            This function reads all the standard output data and
            the errors from the process that are being run.
        '''
        # self.termedit = termedit
        # self.termedit.append(str(self.process.readAll().data(),\
        # encoding='utf-8'))
        stdoutput = self.process.readAll()
        TextStdOut = "<span style=\" font-size:12pt;\
         font-weight:300; color:#000000;\" >"
        for line in str(stdoutput.data(), encoding='utf-8').split("\n"):
            TextStdOut += "<br>" + line
        TextStdOut += "</span>"
        self.termedit.append(TextStdOut)
        # print(str(self.process.readAll().data(), encoding='utf-8'))

        stderror = self.process.readAllStandardError()
        if stderror.toUpper().contains(b"ERROR"):
            self.errorFlag = True
        TextErr = "<span style=\" font-size:12pt; \
        font-weight:1000; color:#ff0000;\" >"
        for line in str(stderror.data(), encoding='utf-8').split("\n"):
            TextErr += "<br>" + line
        TextErr += "</span>"
        self.termedit.append(TextErr)

    # @QtCore.pyqtSlot()
    # def readAllStandard(self):
    #     #self.termedit = termedit
    #     self.termedit.append(str(self.process.\
    #         readAll().data(), encoding='utf-8'))

    #     print(str(self.process.readAll().data(), encoding='utf-8'))
    #     stderror = self.process.readAllStandardError()
    #     if stderror.toUpper().contains(b"ERROR"):
    #         self.errorFlag = True
    #     Text = "<span style=\" font-size:12pt;\
    # font-weight:1000; color:#ff0000;\" >"
    #     for line in str(stderror.data(), encoding='utf-8').split("\n"):
    #         Text += "<br>"+line+"<br>"
    #     Text += "</span>"
    #     self.termedit.append(Text+"\n")

    #     init_path = '../../'
    #     if os.name == 'nt':
    #         init_path = ''
    #     includefile = QtCore.QDir.toNativeSeparators(\
    #     QtWidgets.QFileDialog.getOpenFileName(
    #             self, "Open adding other necessary files to be included",
    #                 init_path + "home"
    #            )[0]
    #         )
    #     if includefile=="":
    #         reply=QtWidgets.QMessageBox.critical(
    #                 None, "Error Message",
    #                 "<b>Error: No File Chosen. Please chose a file</b>",
    #                 QtWidgets.QMessageBox.StandardButton.Ok | QtWidgets.QMessageBox.StandardButton.Cancel
    #             )
    #         if reply == QtWidgets.QMessageBox.StandardButton.Ok:
    #             self.addfile()
    #             self.obj_Appconfig.print_info('Add Other Files Called')

    #         elif reply == QtWidgets.QMessageBox.StandardButton.Cancel:
    #             self.obj_Appconfig.print_info('No File Chosen')
    #     filename = os.path.basename(includefile)
    #     self.modelpath=self.digital_home+"/"+self.fname.split('.')[0]+"/"

    #     if not os.path.isdir(self.modelpath):
    #         os.mkdir(self.modelpath)
    #     text = open(includefile).read()
    #     open(self.modelpath+filename,'w').write(text)
    #     includefile.close()
