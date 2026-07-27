"""Tests for the Qt-free Icarus backend (maker.hdl.icarus).

The file-writing / result-shaping logic is tested without any toolchain by
pointing the "compiler" at /bin/true. Real compile+simulate behaviour is
covered by integration tests that skip when iverilog/vvp are not installed.
"""
import os
import shutil

import pytest

from maker import CosimConfig
from maker.hdl import icarus

_IVERILOG = CosimConfig.iverilog_binary()
_VVP = CosimConfig.vvp_binary()
# Resolve the lib dir at import time, like the binaries above: at test run
# time the repo-wide isolated_user_home fixture has re-pointed HOME, so a
# run-time CosimConfig lookup finds no ~/.nghdl config and returns None --
# and a source-built vvp then fails to dlopen libvvp (no LD_LIBRARY_PATH).
_LIBDIR = CosimConfig.iverilog_libdir()
needs_iverilog = pytest.mark.skipif(
    not _IVERILOG, reason="iverilog not installed")
needs_sim = pytest.mark.skipif(
    not (_IVERILOG and _VVP), reason="iverilog+vvp not installed")

COUNTER = """\
module counter(input clk, input rst, output reg [3:0] count);
  always @(posedge clk or posedge rst)
    if (rst) count <= 0; else count <= count + 1;
endmodule
"""

TB = """\
`timescale 1ns/1ps
module tb_design;
  reg clk = 0, rst = 1;
  wire [3:0] count;
  counter uut(.clk(clk), .rst(rst), .count(count));
  always #5 clk = ~clk;
  initial begin
    $dumpfile("sim_out.vcd"); $dumpvars(0, tb_design);
    #20 rst = 0; #200 $finish;
  end
endmodule
"""

_TRUE = shutil.which("true") or "/bin/true"
needs_true = pytest.mark.skipif(
    not os.path.exists(_TRUE), reason="no /bin/true (POSIX-only test)")


@needs_true
def test_compile_writes_named_sources(tmp_path):
    # /bin/true exits 0 but produces no artifact -> ok is False because the
    # output binary is required, yet the sources are written under their names.
    res = icarus.compile_design(
        _TRUE, [("alu.v", "x"), ("tb_design.v", "y")], str(tmp_path))
    assert res.ok is False                      # no out.bin produced
    assert {os.path.basename(p) for p in res.written} == {"alu.v", "tb_design.v"}
    assert (tmp_path / "alu.v").read_text() == "x"
    assert res.out_path is None


def test_compile_passes_bare_names_so_diagnostics_stay_readable(
        tmp_path, monkeypatch):
    # iverilog echoes back whatever path it was handed. Handing it absolute
    # paths turned every error into
    #   C:\...\AppData\Local\Temp\tmp8f3k\tb_design.v:8: error: ...
    # which buries the line number -- the one actionable token -- behind a temp
    # path. Run from workdir with bare names instead; `written` stays absolute
    # for callers that need the files themselves.
    seen = {}

    def fake_run(cmd, cwd, timeout, cancel, env=None):
        seen['cmd'], seen['cwd'] = cmd, cwd
        return 1, "", ""

    monkeypatch.setattr(icarus, "_run_cmd", fake_run)
    res = icarus.compile_design(
        "iverilog", [("alu.v", "x"), ("tb_design.v", "y")], str(tmp_path))

    assert seen['cwd'] == str(tmp_path)
    assert seen['cmd'][-2:] == ["alu.v", "tb_design.v"]
    assert all(os.path.isabs(p) for p in res.written)


# --- diagnostic parsing --------------------------------------------------- #

_ELAB_LOG = """\
tb_design.v:8: error: Unknown module type: counter
2 error(s) during elaboration.
*** These modules were missing:
        counter referenced 1 times.
***
"""


def test_missing_modules_reads_unknown_module_errors():
    assert icarus.missing_modules(_ELAB_LOG) == ["counter"]
    assert icarus.missing_modules("") == []
    assert icarus.missing_modules("alu.v:3: error: syntax error") == []


def test_missing_modules_deduplicates_in_first_seen_order():
    log = ("tb.v:4: error: Unknown module type: alu\n"
           "tb.v:9: error: Unknown module type: mux\n"
           "tb.v:12: error: Unknown module type: alu\n")
    assert icarus.missing_modules(log) == ["alu", "mux"]


def test_unknown_module_sites_carry_the_file_that_asked():
    # Which file instantiated the missing module decides the wording: a stale
    # testbench is a different problem from a design missing a submodule.
    assert icarus.unknown_module_sites(_ELAB_LOG) == [
        ("tb_design.v", 8, "counter")]


def test_diagnostics_defaults_severity_for_bare_parse_errors():
    # iverilog's own parse errors carry no "error:" token at all. Requiring one
    # would silently drop the single most common failure.
    log = "or_gate.v:2: syntax error\nI give up.\n"
    assert icarus.diagnostics(log) == [("or_gate.v", 2, "error", "syntax error")]
    assert icarus.error_locations(log) == ["or_gate.v:2"]


def test_error_locations_skips_warnings_and_dedups():
    log = ("alu.v:3: warning: implicit definition of wire 'q'\n"
           "alu.v:7: error: unable to bind wire\n"
           "alu.v:7: error: another one on the same line\n")
    assert icarus.error_locations(log) == ["alu.v:7"]


def test_diagnostics_survive_an_absolute_windows_path():
    # Defence in depth: even if a diagnostic does come back fully qualified
    # (an include, a toolchain that ignores cwd), the bare name is recovered.
    log = (r"C:\Users\x\AppData\Local\Temp\tmpe4k\tb_design.v"
           ":8: error: Unknown module type: counter")
    assert icarus.unknown_module_sites(log) == [("tb_design.v", 8, "counter")]
    assert icarus.error_locations(log) == ["tb_design.v:8"]


@needs_iverilog
def test_real_diagnostics_are_not_prefixed_with_the_temp_path(tmp_path):
    res = icarus.compile_design(
        _IVERILOG, [("or_gate.v", "module or_gate(input a); endmodule"),
                    ("tb_design.v", TB)], str(tmp_path))
    assert res.ok is False
    assert str(tmp_path) not in res.stderr
    assert icarus.error_locations(res.output), res.stderr


@needs_iverilog
def test_compile_error_is_reported(tmp_path):
    res = icarus.compile_design(
        _IVERILOG, [("bad.v", "module bad(); syntax error endmodule")],
        str(tmp_path))
    assert res.ok is False
    assert res.returncode != 0
    assert res.stderr.strip()
    assert res.out_path is None


def test_vpi_load_failure_detected():
    # The exact shape Icarus emits when a .vpi's runtime DLL is missing
    # (fresh Windows machine, MinGW closure absent). Icarus exits 0 anyway,
    # so this detector is the only thing standing between that state and a
    # bogus "build succeeded".
    msg = ("error: Failed to open "
           r"'C:\FOSSEE\eSim\library\bin\iverilog\lib\ivl\system.vpi'"
           " because:\n     : The specified module could not be found.\n")
    assert icarus.vpi_load_failed(msg) is True
    assert icarus.vpi_load_failed("") is False
    assert icarus.vpi_load_failed("t.v:3: error: Failed to open include") is False
    assert icarus.vpi_load_failed("loaded system.vpi fine") is False


def test_vpi_load_failure_fails_compile(tmp_path, monkeypatch):
    # rc=0 AND artifact present AND vpi-load error in stderr -> ok must be
    # False (the artifact cannot simulate: the same load fails under vvp and
    # under ngspice's ivlng).
    out_path = tmp_path / "m.out"

    def fake_run(cmd, cwd, timeout, cancel, env=None):
        out_path.write_bytes(b"#! vvp\n")
        return 0, "", ("error: Failed to open 'x/system.vpi' because:\n"
                       "     : The specified module could not be found.\n")

    monkeypatch.setattr(icarus, "_run_cmd", fake_run)
    res = icarus.run_iverilog("iverilog", ["m.v"], str(out_path))
    assert res.ok is False
    assert res.returncode == 0
    assert res.out_path is None


@needs_iverilog
def test_compile_ok(tmp_path):
    res = icarus.compile_design(
        _IVERILOG, [("counter.v", COUNTER), ("tb_design.v", TB)],
        str(tmp_path), out_name="sim.out")
    assert res.ok is True
    assert res.out_path and os.path.isfile(res.out_path)


@needs_sim
def test_compile_and_simulate_produces_vcd(tmp_path):
    res = icarus.compile_design(
        _IVERILOG, [("counter.v", COUNTER), ("tb_design.v", TB)],
        str(tmp_path), out_name="sim.out")
    assert res.ok, res.stderr
    sim = icarus.simulate(
        _VVP, res.out_path, str(tmp_path),
        env=icarus.vvp_env(_VVP, libdir=_LIBDIR))
    assert sim.ok, sim.stderr
    assert sim.vcd_path and os.path.isfile(sim.vcd_path)


@needs_sim
def test_build_and_simulate_orchestration(tmp_path):
    run = icarus.build_and_simulate(
        _IVERILOG, _VVP, [("counter.v", COUNTER), ("tb_design.v", TB)],
        str(tmp_path), libdir=_LIBDIR)
    assert run.ok, run.compile.stderr + ((run.sim.stderr) if run.sim else "")
    # VCD content is read on the worker side so the GUI never touches tmpdir.
    assert run.vcd_content and "$var" in run.vcd_content


# --- CancelToken (tool-free, POSIX uses `sleep`) ------------------------- #
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

posix_only = pytest.mark.skipif(os.name == 'nt', reason="POSIX `sleep` test")


@posix_only
def test_cancel_token_kills_running_process():
    tok = icarus.CancelToken()
    out = {}

    def run():
        try:
            out['res'] = icarus._run_cmd(['sleep', '30'], None, None, tok)
        except Exception as exc:
            out['err'] = exc

    t = threading.Thread(target=run)
    t.start()
    time.sleep(0.3)
    tok.cancel()
    t.join(5)
    assert not t.is_alive()          # cancel unblocked the worker
    assert tok.cancelled


@posix_only
def test_cancel_before_bind_kills_immediately():
    tok = icarus.CancelToken()
    tok.cancel()                     # cancelled before a process is bound
    proc = subprocess.Popen(['sleep', '30'])
    tok.bind(proc)                   # bind must kill it right away
    proc.wait(5)
    assert proc.poll() is not None


# --- timeout: a runaway sim must not hang the worker forever ------------- #

@posix_only
def test_run_cmd_times_out_without_cancel():
    # subprocess.run path (no cancel token).
    with pytest.raises(subprocess.TimeoutExpired):
        icarus._run_cmd(['sleep', '5'], None, 0.2, None)


@posix_only
def test_run_cmd_times_out_with_cancel_and_reaps_proc():
    # Popen path (cancel token bound): the timeout branch kills + reaps.
    tok = icarus.CancelToken()
    with pytest.raises(subprocess.TimeoutExpired):
        icarus._run_cmd(['sleep', '5'], None, 0.2, tok)
