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
import shutil
import subprocess
from PyQt6 import QtCore, QtWidgets
from configuration import Dialogs
from configuration import paths
from configparser import ConfigParser
from configuration import Appconfig

from . import createkicad
from . import CosimConfig
from .hdl import icarus
from .CosimLogger import CosimLog
import hdlparse.verilog_parser as vlog


class ModelGeneration(QtWidgets.QWidget):
    '''
        Class is used to generate the Ngspice Model
    '''

    # Generous cap (seconds) so big verilator/make builds are not guillotined
    # at the old 50 s limit, while a genuinely hung process is still killed and
    # reported instead of either freezing the GUI forever or silently
    # producing a half-built model. The whole legacy pipeline now runs off the
    # GUI thread (NgVeri.addverilog), so a slow build no longer freezes eSim;
    # this is only the per-step wall-clock safety net.
    PROCESS_TIMEOUT = 600           # 10 minutes, in seconds (subprocess.run)

    # Emitted for every line/block of build output. Connected to termedit in
    # __init__; because the connection is auto-typed, calls from the GUI thread
    # deliver synchronously while calls from the build worker thread are queued
    # back onto the GUI thread -- so the subprocess pipeline can stream output
    # without ever touching a widget off-thread.
    line = QtCore.pyqtSignal(str)

    def __init__(self, file, termedit):
        super().__init__()
        self.obj_Appconfig = Appconfig.Appconfig()
        print("Argument is : ", file)

        if os.name == 'nt':
            self.file = file.replace('\\', '/')
        else:
            self.file = file

        self.termedit = termedit
        # Route every termtext/termtitle/_run write through the line signal so
        # the same code path is safe whether it runs on the GUI thread (fast
        # file-generation steps) or the build worker thread (verilator/make).
        self.line.connect(self.termedit.append)
        # Dual-sink d_cosim logger: same events to the NgVeri GUI terminal
        # (this termedit) AND the OS terminal + ~/.esim/dcosim.log. Route the
        # GUI sink through the `line` signal (not termedit.append directly) so
        # build_cosim's log lines stay GUI-thread-safe when the build runs on a
        # worker thread.
        self.clog = CosimLog(termedit, sink=self.line.emit)
        self.cur_dir = os.getcwd()
        self.fname = os.path.basename(file)
        self.fname = self.fname.lower()
        print("Verilog/SystemVerilog/TL Verilog filename is : ", self.fname)

        # Keep a parser for the legacy build methods below, but all constructor
        # values are read through CosimConfig's missing-safe boundary. This is
        # crucial for d_cosim-only installs, which intentionally have no NGHDL
        # config file.
        self.parser = ConfigParser()
        self.parser.read(CosimConfig.nghdl_config_path())
        self.nghdl_home = CosimConfig.nghdl_cfg('NGHDL', 'NGHDL_HOME')
        self.release_dir = CosimConfig.nghdl_cfg('NGHDL', 'RELEASE')
        self.src_home = CosimConfig.nghdl_cfg('SRC', 'SRC_HOME')
        self.licensefile = CosimConfig.nghdl_cfg('SRC', 'LICENSE')
        self.digital_home = os.path.join(
            CosimConfig.digital_model_root(), 'Ngveri')

    def require_legacy_toolchain(self):
        """Report a missing legacy toolchain cleanly instead of crashing.

        Two layers: the cheap config check (NGHDL_HOME/RELEASE/SRC_HOME keys
        present) and the full doctor probe (verilator/make/gcc/ngspice all
        actually on disk), so a half-installed toolchain fails HERE with the
        exact missing tool + fix hint instead of exploding mid-pipeline in
        make."""
        if not (self.nghdl_home and self.release_dir and self.src_home):
            message = (
                "NGHDL/NgVeri toolchain not configured — install NGHDL or "
                "use Dual Co-sim."
            )
            self.termtext(message)
            self.obj_Appconfig.print_error(message)
            return False
        from . import ToolchainCheck
        message = ToolchainCheck.failure_message(ToolchainCheck.NGVERI)
        if message:
            self.termtext(message)
            self.obj_Appconfig.print_error(message)
            return False
        return True

    # ------------------------------------------------------------------ #
    #  Cross-platform build-tool resolution (single source for every step)
    # ------------------------------------------------------------------ #
    def _msys_home(self):
        return CosimConfig.nghdl_cfg('COMPILER', 'MSYS_HOME')

    def _nt_build_env(self):
        """Environment for build subprocesses on Windows: the MSYS2 mingw64
        and usr/bin dirs go FIRST on PATH (make's child gcc/g++/ar and the
        verilator wrapper resolve from there -- the eSim process itself never
        has them on PATH), plus VERILATOR_ROOT for the model Makefiles.
        Returns None on POSIX (inherit as-is)."""
        if os.name != 'nt':
            return None
        env = os.environ.copy()
        msys_home = self._msys_home()
        if msys_home:
            env["PATH"] = os.pathsep.join([
                os.path.join(msys_home, 'mingw64', 'bin'),
                os.path.join(msys_home, 'usr', 'bin'),
            ]) + os.pathsep + env.get("PATH", "")
            # MSYS2's verilator package keeps the runtime tree (include/
            # verilated.cpp, verilated_std.sv, lint waivers) under
            # share/verilator, not the mingw64 prefix itself -- pointing
            # VERILATOR_ROOT at the prefix makes every verilator run fail with
            # "Cannot find verilated_std_waiver.vlt".
            # Forward slashes: the value is spliced into verilator's generated
            # Makefile, where backslashes are escape characters (make eats
            # them, e.g. C:\FOSSEE\... becomes C:FOSSEE... and the includer
            # path collapses).
            env["VERILATOR_ROOT"] = os.path.join(
                msys_home, 'mingw64', 'share', 'verilator'
            ).replace('\\', '/')
        return env

    def _make_binary(self):
        """make (POSIX) / MSYS2 mingw32-make (Windows), or None with an
        actionable terminal message naming the exact probed path."""
        if os.name != 'nt':
            return "make"
        msys_home = self._msys_home()
        cand = (os.path.join(msys_home, 'mingw64', 'bin',
                             'mingw32-make.exe') if msys_home else '')
        if cand and os.path.isfile(cand):
            return cand
        self.termtext(
            "[NgVeri] mingw32-make not found (probed: " +
            (cand or "~/.nghdl/config.ini [COMPILER] MSYS_HOME unset") +
            "). Reinstall eSim with the HDL-toolchain (MSYS2) component.")
        return None

    def _verilator_binary(self):
        """verilator (POSIX) / the MSYS2 mingw64 verilator (Windows), or None
        with an actionable terminal message. On Windows the real binary is
        verilator_bin.exe (the `verilator` front-end is a perl script); with
        VERILATOR_ROOT set, invoking it directly is equivalent."""
        if os.name != 'nt':
            return "verilator"
        msys_home = self._msys_home()
        probed = []
        for name in ('verilator_bin.exe', 'verilator.exe'):
            cand = (os.path.join(msys_home, 'mingw64', 'bin', name)
                    if msys_home else '')
            probed.append(cand or name)
            if cand and os.path.isfile(cand):
                return cand
        self.termtext(
            "[NgVeri] Verilator not found (probed: " + ", ".join(probed) +
            "). Reinstall eSim with the HDL-toolchain (MSYS2) component.")
        return None

    def _run(self, cmd, title, cwd=None, env=None):
        '''
            Run one step of the model-build pipeline and return True only when
            the process exits cleanly with code 0.

            `cmd` is an argument LIST (never a shell string): the process is
            spawned directly, so a path with spaces or shell metacharacters can
            neither split into extra arguments nor be interpreted -- the old
            ``sh -c`` + string-concatenation was both fragile with spaced paths
            and an injection surface. The working directory is passed as
            ``cwd=`` instead of an ``os.chdir`` dance, so a failed step can
            never strand the whole app inside a model sub-directory and no
            longer races the CWD-relative paths elsewhere in eSim.

            Output is captured and streamed into the NgVeri terminal via the
            ``line`` signal, so this is safe to call from the build worker
            thread (the signal is queued back onto the GUI thread).
        '''
        self.termtitle(title)
        self.termtext("Current Directory: " + (cwd or os.getcwd()))
        self.termtext("Command: " + " ".join(cmd))
        try:
            res = subprocess.run(
                cmd, cwd=cwd, env=env, capture_output=True, text=True,
                timeout=self.PROCESS_TIMEOUT)
        except subprocess.TimeoutExpired:
            self.termtext("[NgVeri] '" + title +
                          "' timed out and was stopped.")
            return False
        except OSError as err:
            self.termtext("[NgVeri] '" + title + "' could not be started: " +
                          str(err))
            return False

        if res.stdout:
            self.termtext(res.stdout)
        if res.stderr:
            self._emit_error(res.stderr)
        if res.returncode != 0:
            self.termtext("[NgVeri] '" + title + "' failed (exit code " +
                          str(res.returncode) + ").")
            return False
        return True

    def _emit_error(self, textin):
        '''Append stderr text in red (theme-independent) to the terminal.'''
        Text = "<span style=\"font-size:12pt; font-weight:1000; " \
               "color:#ff0000;\">"
        for ln in textin.split("\n"):
            Text += "<br>" + ln
        Text += "</span>"
        self.line.emit(Text)

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
        modname = os.path.splitext(self.fname)[0]
        self.modelpath = self.digital_home + "/" + modname + "/"
        if not os.path.isdir(self.modelpath):
            os.mkdir(self.modelpath)

        # os.path.splitext keeps the true extension even for dotted/no-dot
        # names (the old .split('.')[1] IndexError'd on "counter" and read
        # "v" from "model.v.bak").
        if os.path.splitext(self.fname)[1] == ".tlv":
            self.sandpiper()
            # sandpiper() rewrote self.fname to "<model>.sv"
            modname = os.path.splitext(self.fname)[0]
            with open(self.modelpath + self.fname, 'r') as read_verilog:
                verilog_data = read_verilog.readlines()
        is_sv = os.path.splitext(self.fname)[1] == ".sv"
        with open(self.modelpath + self.fname, 'w') as f:
            for item in verilog_data:
                if is_sv:
                    # Rename the SV top module to the file's stem. A bare
                    # substring replace mangled any identifier CONTAINING
                    # "top" (stop, laptop, top_val); a word-boundary regex
                    # only touches the standalone token.
                    string = re.sub(r'\btop\b', modname, item)
                else:
                    string = item
                f.write(string)
            f.write("\n")

    def sandpiper(self):
        '''
            This function calls the sandpiper to convert .tlv file to .sv file
        '''
        # Text="Running Sandpiper............"
        print("Running Sandpiper-Saas for TLV to SV Conversion")
        tlv = paths.library_path("tlv")
        # Pure-Python copy: no sh quoting problem when tlv/ or the workspace
        # sits under a spaced path (e.g. a "VLSI Lab" username on MSYS).
        self.termtitle("COPY TLV FILES")
        tlv_files = ["clk_gate.v", "pseudo_rand.sv", "sandpiper.vh",
                     "sandpiper_gen.vh", "sp_default.vh", "pseudo_rand_gen.sv",
                     "pseudo_rand.m4out.tlv"]
        for name in tlv_files:
            shutil.copy2(os.path.join(tlv, name), self.modelpath)
        shutil.copy2(self.file, self.modelpath)
        print("Copied the files required for TLV successfully")

        print("Running Sandpiper............")
        model = os.path.splitext(self.fname)[0]
        self._run(["sandpiper-saas", "-i", model + ".tlv",
                   "-o", model + ".sv"],
                  "RUN SANDPIPER-SAAS", cwd=self.modelpath)
        print("Ran Sandpiper successfully")
        self.fname = model + ".sv"

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

        modname = os.path.splitext(self.fname)[0]
        # hdlparse returns nothing for an empty file, a syntax error or a
        # construct it cannot parse. The old code then indexed a loop variable
        # `m` that was never bound -> "NameError: m" instead of a useful
        # message. Bail early with a clear error.
        if not vlog_mods:
            Dialogs.critical(
                None, "Error Message",
                "<b>Error: No Verilog module could be parsed from " +
                self.fname + ". Check the file for syntax errors.</b>",
                QtWidgets.QMessageBox.StandardButton.Ok)
            self.obj_Appconfig.print_info(
                'NgVeri stopped: no parseable module in ' + self.fname)
            return "Error"

        matched = None
        with open(self.modelpath + "connection_info.txt", 'w') as f:
            for m in vlog_mods:
                if m.name.lower() == modname:
                    print(str(m.name) + " " + modname)
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
                if m.name.lower() == modname:
                    m.name = m.name.lower()
                    matched = m
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
        if matched is None:
            Dialogs.critical(
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
            modelname = str(matched.name)
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
                # Same iverilog invocation path as the Verilog Simulator IDE
                # (hdl.icarus), so both features stay byte-for-byte consistent.
                res = icarus.run_iverilog(
                    iverilog, [compile_src], out,
                    cwd=os.path.abspath(self.modelpath), timeout=300)
            finally:
                if tmp_src and os.path.isfile(tmp_src):
                    os.remove(tmp_src)
            elapsed = time.monotonic() - start
            log.output(res.stdout, 'stdout')
            log.output(res.stderr, 'stderr')
            log.info("iverilog exited rc=%d in %.2fs"
                     % (res.returncode, elapsed))

            # ----- [4/4] Verify artifact -----
            log.phase("[4/4] Verify artifact")
            if not res.ok:
                log.error("d_cosim model build FAILED (rc=%d)."
                          % res.returncode)
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
        wno = []
        try:
            with open(paths.library_path("tlv/lint_off.txt")) as file:
                for item in file.readlines():
                    if item and item.strip():
                        wno.append("-Wno-" + item.strip())
        except OSError:
            # A missing lint_off.txt should degrade to "no extra -Wno" rather
            # than crash the whole verilator build with a raw exception.
            wno = []

        print("Running Verilator.............")
        self.release_home = self.parser.get('NGHDL', 'RELEASE')
        # print(self.modelpath)

        # Windows: VERILATOR_ROOT/PATH go into the environment (a shell
        # `export` has no meaning for a direct exec) rather than being glued
        # onto a `sh -c` string.
        env = self._nt_build_env()
        verilator = self._verilator_binary()
        if not verilator:
            return False

        model = os.path.splitext(self.fname)[0]
        # -DVL_TIME_CONTEXT: verilated.o is (re)built by the generated .mk
        # with these CFLAGS. Without it verilated.cpp leaves the weak
        # sc_time_stamp() reference undefined, which a Linux .so tolerates but
        # the Windows Ngveri.cm DLL link rejects (undefined reference).
        cmd = [
            verilator, "--stats", "-O3",
            "-CFLAGS", "-O3", "-CFLAGS", "-DVL_TIME_CONTEXT",
            "-LDFLAGS", "-static", "--x-assign", "fast",
            "--x-initial", "fast", "--noassert", "--bbox-sys", "-Wall",
        ] + wno + [
            "--cc", "--exe", "--no-MMD", "--Mdir", ".", "-CFLAGS", "-fPIC",
            "-output-split", "0", "sim_main_" + model + ".cpp", "--autoflush",
            "-DBSV_RESET_FIFO_HEAD", "-DBSV_RESET_FIFO_ARRAY", self.fname,
        ]
        return self._run(cmd, "RUN VERILATOR", cwd=self.modelpath, env=env)

    def make_verilator(self):
        '''
            Running make verilator using this function
        '''
        print("Make Verilator.............")

        stale = os.path.join(self.modelpath, "..", "verilated.o")
        if os.path.exists(stale):
            os.remove(stale)

        make_bin = self._make_binary()
        if not make_bin:
            return False

        model = os.path.splitext(self.fname)[0]
        # Purge make-generated aggregates from any earlier (possibly failed)
        # build of this model: an interrupted verilator_includer leaves an
        # empty V<model>__ALL.cpp that make then treats as up to date, and the
        # resulting symbol-less archive only fails much later at the
        # Ngveri.cm link.
        for leftover in ("V" + model + "__ALL.cpp",
                         "V" + model + "__ALL.o",
                         "V" + model + "__ALL.a"):
            p = os.path.join(self.modelpath, leftover)
            if os.path.exists(p):
                os.remove(p)
        cmd = [make_bin, "-f", "V" + model + ".mk",
               "V" + model + "__ALL.a",
               "sim_main_" + model + ".o",
               "../verilated.o", "../verilated_threads.o"]
        return self._run(cmd, "MAKE VERILATOR", cwd=self.modelpath,
                         env=self._nt_build_env())

    def copy_verilator(self):
        '''
            This function copies the verilator files/object files from
            "src/xspice/icm/Ngveri/ to release/src/xspice/icm/Ngveri/"
        '''
        print("Copying the required files to Release Folder.............")
        self.release_home = self.parser.get('NGHDL', 'RELEASE')
        ngveri_icm = self.release_home + "/src/xspice/icm/Ngveri/"
        model = os.path.splitext(self.fname)[0]
        # Per-model dir; keep a trailing slash so the os.remove guards below
        # actually target real files. Without it the paths glued to
        # ".../Ngveri/<model>sim_main_..." (note the missing slash), never
        # existed, so the stale-artifact cleanup was a silent no-op.
        path_icm = ngveri_icm + model + "/"
        if not os.path.isdir(path_icm):
            os.makedirs(path_icm, exist_ok=True)
        for stale in (path_icm + "sim_main_" + model + ".o",
                      ngveri_icm + "verilated.o",
                      ngveri_icm + "verilated_threads.o",
                      path_icm + "V" + model + "__ALL.a"):
            if os.path.exists(stale):
                os.remove(stale)
        # shutil instead of `cp` via sh -c: no quoting hazard for spaced
        # release paths, and a copy failure raises here (-> False) instead of
        # a silent nonzero exit that the old success search would have missed.
        self.termtitle("COPYING FILES")
        try:
            shutil.copy2(os.path.join(
                self.modelpath, "sim_main_" + model + ".o"), path_icm)
            shutil.copy2(os.path.join(
                self.modelpath, "V" + model + "__ALL.a"), path_icm)
            shutil.copy2(os.path.normpath(os.path.join(
                self.modelpath, "..", "verilated.o")), ngveri_icm)
            shutil.copy2(os.path.normpath(os.path.join(
                self.modelpath, "..", "verilated_threads.o")), ngveri_icm)
        except OSError as err:
            self.termtext(
                "[NgVeri] Copying build artifacts failed: " + str(err))
            return False
        print("Copied the files")
        return True

    def runMake(self):
        '''
            Running the make command for Ngspice
        '''
        print("run Make Called")
        self.release_home = self.parser.get('NGHDL', 'RELEASE')
        path_icm = os.path.join(self.release_home, "src/xspice/icm")

        make_bin = self._make_binary()
        if not make_bin:
            return False

        print("Running Make command in " + path_icm)
        return self._run([make_bin], "MAKE COMMAND", cwd=path_icm,
                         env=self._nt_build_env())

    def runMakeInstall(self):
        '''
            Running the make install command for Ngspice
        '''
        print("run Make Install Called")
        self.release_home = self.parser.get('NGHDL', 'RELEASE')
        path_icm = os.path.join(self.release_home, "src/xspice/icm")

        make_bin = self._make_binary()
        if not make_bin:
            return False
        print("Running Make Install")
        cmd = [make_bin, "install"]
        if os.name == 'nt':
            # The configured tree bakes the BUILD machine's absolute prefix
            # into makedefs (pkglibdir/pkgdatadir), so a stock `make install`
            # on an end-user PC would write the rebuilt code models into the
            # packager's path instead of this install's install_dir. Override
            # both on the command line from ~/.nghdl/config.ini (forward
            # slashes: these values are make variables).
            try:
                nghdl_home = self.parser.get('NGHDL', 'NGHDL_HOME')
            except Exception:
                nghdl_home = ''
            if nghdl_home:
                inst = os.path.join(nghdl_home, 'install_dir').replace(
                    '\\', '/')
                cmd += ["pkglibdir=" + inst + "/lib/ngspice",
                        "pkgdatadir=" + inst + "/share/ngspice"]
        return self._run(cmd,
                         "MAKE INSTALL COMMAND", cwd=path_icm,
                         env=self._nt_build_env())

    def addfile(self):
        '''
            This function is used to add additional files
            required by the verilog top module.
        '''
        print("Adding the files required by the top level module file")

        includefile = QtCore.QDir.toNativeSeparators(
            QtWidgets.QFileDialog.getOpenFileName(
                Dialogs.resolve_parent(self),
                "Open adding other necessary files to be included",
                os.path.join(paths.repo_root(), "home"))[0])

        if includefile == "":
            reply = Dialogs.critical(
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

        # Esc / window-X on the dialog matches neither branch; without this
        # guard execution fell through with an empty path and wrote a blank
        # include file.
        if includefile == "":
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
                Dialogs.resolve_parent(self), "open", "home"
            )
        )

        if includefolder == "":
            reply = Dialogs.critical(
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

        # Esc / window-X matches neither branch; guard against falling through
        # with an empty path (would makedirs/copytree against a bad target).
        if includefolder == "":
            return

        self.modelpath = self.digital_home + \
            "/" + os.path.splitext(self.fname)[0] + "/"
        if not os.path.isdir(self.modelpath):
            os.makedirs(self.modelpath, exist_ok=True)

        reply = Dialogs.question(
            None, "Message",
            '''<b>If you want only the contents\
             of the folder to be added press "Yes".\
                    If you want complete folder \
                    to be added, press "No". </b>''',
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
        )
        foldername = os.path.basename(os.path.normpath(includefolder))
        self.termtitle("Adding the Folder: " + foldername)
        # shutil.copytree instead of `cp` via sh -c: a user-picked folder with
        # spaces/metacharacters can neither split nor execute. Esc / window-X
        # returns neither Yes nor No -> do nothing, instead of the old code
        # falling through and re-running a stale self.cmd from a prior action.
        try:
            if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                shutil.copytree(includefolder, self.modelpath,
                                dirs_exist_ok=True)
                self.obj_Appconfig.print_info('Adding Contents of the Folder')
            elif reply == QtWidgets.QMessageBox.StandardButton.No:
                shutil.copytree(
                    includefolder,
                    os.path.join(self.modelpath, foldername),
                    dirs_exist_ok=True)
                self.obj_Appconfig.print_info('Adding the Folder')
            else:
                self.obj_Appconfig.print_info('Add Folder cancelled')
                return
        except OSError as err:
            self.termtext("[NgVeri] Could not add folder '" +
                          foldername + "': " + str(err))
            return
        print("Added the folder")

    def termtitle(self, textin):
        '''
            This function is used to print the titles
            in the terminal of Ngveri tab. Emitted via the ``line`` signal so
            it is safe from the build worker thread.
        '''
        # No hardcoded colour: the old #0000FF blue was near-invisible on the
        # dark theme. Weight + rule bars carry the emphasis; the text inherits
        # the palette so it reads in both themes.
        Text = "<span style=\"font-size:20pt; font-weight:1000;\">"
        Text += "<br>================================<br>"
        Text += textin
        Text += "<br>================================<br>"
        Text += "</span>"
        self.line.emit(Text)

    def termtext(self, textin):
        '''
            This function is used to print the text/commands
            in the terminal of Ngveri tab. Emitted via the ``line`` signal so
            it is safe from the build worker thread.
        '''
        # No hardcoded colour (was #000000, invisible on dark): inherit the
        # palette. stderr keeps red via _emit_error.
        Text = "<span style=\"font-size:12pt; font-weight:500;\">"
        Text += textin
        Text += "</span>"
        self.line.emit(Text)
