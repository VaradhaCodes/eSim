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


@needs_iverilog
def test_compile_error_is_reported(tmp_path):
    res = icarus.compile_design(
        _IVERILOG, [("bad.v", "module bad(); syntax error endmodule")],
        str(tmp_path))
    assert res.ok is False
    assert res.returncode != 0
    assert res.stderr.strip()
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
        env=icarus.vvp_env(_VVP, libdir=CosimConfig.iverilog_libdir()))
    assert sim.ok, sim.stderr
    assert sim.vcd_path and os.path.isfile(sim.vcd_path)


@needs_sim
def test_build_and_simulate_orchestration(tmp_path):
    run = icarus.build_and_simulate(
        _IVERILOG, _VVP, [("counter.v", COUNTER), ("tb_design.v", TB)],
        str(tmp_path), libdir=CosimConfig.iverilog_libdir())
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
