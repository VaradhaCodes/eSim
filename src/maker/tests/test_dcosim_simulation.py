"""End-to-end gate for the d_cosim backend: compile, simulate, decode.

The failure mode this whole area is being hardened against is *a model that
builds and runs and is wrong*, which no unit test over the generators can
catch. This one drives the real chain -- iverilog -> vvp-format artifact ->
ngspice + ivlng -> `print allv` -- and decodes the counter's output bus back
into integers, so a regression shows up as the wrong number rather than as a
plausible-looking waveform.

Skipped unless the machine has both Icarus (with libvvp) and an ngspice build
carrying the ivlng adapter, so it is a no-op on a bare CI box.
"""
import os
import re
import subprocess

import pytest

from maker import CosimConfig
from maker.hdl import icarus
from kicadtoNgspice.KicadtoNgspice import collapse_adc_band_for_hdl

# Resolved once, at import, and used for both the skip decision and the runs.
# CosimConfig reads ~/.nghdl/config.ini, and the suite's fixtures redirect HOME
# to a tmp dir -- resolving inside a test would silently fall back to a bare
# "ngspice" that is not on PATH, so the skip guard would pass and the run would
# then die on WinError 2.
NGSPICE = CosimConfig.ngspice_binary()
IVERILOG = CosimConfig.iverilog_binary()
CODEMODELS = CosimConfig.ngspice_codemodel_dir()
LOADER = CosimConfig.loader_path_var()
LOADER_DIRS = [d for d in (os.path.dirname(NGSPICE or ""),
                           CosimConfig.iverilog_libdir(),
                           os.path.dirname(CosimConfig.vvp_binary() or ""))
               if d and os.path.isdir(d)]


def _toolchain_present():
    """Every binary and adapter this test shells out to, actually on disk.

    has_dcosim() answers "is this install configured for d_cosim", which a
    source checkout can satisfy by pointing at a tools/ tree that was never
    built. Only files that exist can be run."""
    return bool(
        CosimConfig.has_dcosim(force=True)
        and os.path.isfile(NGSPICE or "")
        and os.path.isfile(IVERILOG or "")
        and CODEMODELS
        and os.path.isfile(os.path.join(CODEMODELS, "ivlng.vpi")))


pytestmark = pytest.mark.skipif(
    not _toolchain_present(),
    reason="needs iverilog with libvvp and an ngspice carrying ivlng")


# A 4-bit up counter with a one-clock wrap strobe: the strobe is the witness.
# A model whose outputs are naturally static cannot tell a frozen pin from a
# working one, which is why the counter is the test vehicle.
COUNTER_V = """\
`timescale 1ns / 1ps
module dccount (
    input clk,
    input rst,
    output [3:0] cnt,
    output wrap
);
    reg [3:0] cnt_reg;
    reg       wrap_reg;

    always @(posedge clk) begin
        wrap_reg <= 1'b0;
        if (rst) begin
            cnt_reg <= 4'd0;
        end else begin
            cnt_reg <= cnt_reg + 4'd1;
            if (cnt_reg == 4'd15)
                wrap_reg <= 1'b1;
        end
    end

    assign cnt  = cnt_reg;
    assign wrap = wrap_reg;
endmodule
"""

# 1 ms clock with a 1 us ramp -- the ramp is the point: it is what walks the
# adc_bridge through its unknown band. Reset clears at 1.9-2.0 ms, so the
# first counted edge is at 3 ms.
NETLIST = """\
* d_cosim end-to-end
v1 clk_a gnd pulse(0 5 0 1u 1u 0.5m 1m)
v2 rst_a gnd pwl(0 5 1.9m 5 2m 0)
a1 [clk_a rst_a] [clk_d rst_d] u1
a2 [clk_d rst_d] [c3_d c2_d c1_d c0_d w_d] u2
a3 [c3_d c2_d c1_d c0_d w_d] [c3 c2 c1 c0 w] u3
.model u1 adc_bridge(in_low=1.0 in_high=2.0 rise_delay=1.0e-9 \
fall_delay=1.0e-9 )
.model u2 d_cosim simulation="ivlng" lib_args=["libvvp", "ivlng"] \
sim_args=["dccount"]
.model u3 dac_bridge(out_low=0.0 out_high=5.0 out_undef=0.5 t_rise=1.0e-9 \
t_fall=1.0e-9 )
.control
set width=1000
tran 10e-06 20e-03 0e-00
print allv > OUTFILE
.endc
.end
"""


def read_columns(path):
    """``(names, rows)`` from an ngspice ``print allv`` dump."""
    names, rows = None, []
    with open(path, errors="replace") as fh:
        for line in fh:
            token = line.strip()
            if token.startswith("Index"):
                names = token.split()[1:]
                continue
            if names is None or not re.match(r"^\d+\s", token):
                continue
            values = token.split()[1:len(names) + 1]
            if len(values) == len(names):
                rows.append([float(v) for v in values])
    assert names, "no data block in " + path
    return names, rows


def sample(names, rows, when, signals):
    """The last sample at or before ``when``, decoded MSB-first as digital."""
    index = {n: i for i, n in enumerate(names)}
    latest = None
    for row in rows:
        if row[index["time"]] > when:
            break
        latest = row
    value = 0
    for name in signals:
        value = (value << 1) | (1 if latest[index[name]] > 2.5 else 0)
    return value


def simulate(tmp_path, netlist_lines, tag):
    """Run ``netlist_lines`` under the d_cosim ngspice; return its dump."""
    out = tmp_path / (tag + ".txt")
    cir = tmp_path / (tag + ".cir")
    cir.write_text("\n".join(netlist_lines).replace("OUTFILE", out.name))

    # ivlng resolves both the vvp and its VPI module relative to the working
    # directory, which is why the converter stages them beside the netlist.
    with open(os.path.join(CODEMODELS, "ivlng.vpi"), "rb") as fh:
        (tmp_path / "ivlng.vpi").write_bytes(fh.read())

    # libvvp and ngspice's own runtime DLLs are found through the loader path,
    # exactly as the eSim launcher sets it up.
    env = dict(os.environ)
    env[LOADER] = os.pathsep.join(LOADER_DIRS + [env.get(LOADER, "")])

    subprocess.run([NGSPICE, "-b", str(cir)],
                   cwd=str(tmp_path), env=env, capture_output=True,
                   text=True, timeout=300)
    assert out.is_file(), "ngspice produced no output for " + tag
    return read_columns(str(out))


@pytest.fixture(scope="module")
def counter_vvp(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("dccount")
    (workdir / "dccount.v").write_text(COUNTER_V)
    res = icarus.run_iverilog(
        IVERILOG, ["dccount.v"],
        str(workdir / "dccount"), cwd=str(workdir), timeout=300)
    assert res.ok, res.output
    return workdir / "dccount"


def stage(tmp_path, counter_vvp, lines):
    (tmp_path / "dccount").write_bytes(counter_vvp.read_bytes())
    return lines


BITS = ["c3", "c2", "c1", "c0"]

#: Reset clears between 1.9 and 2.0 ms, so the clock at 2 ms is the first one
#: counted; sample 50 us after each rise, well clear of the 1 us ramp.
def counts(names, rows, n):
    return [sample(names, rows, 2.05e-3 + k * 1e-3, BITS) for k in range(n)]


def test_uncollapsed_band_double_clocks_the_design(tmp_path, counter_vvp):
    """The defect itself, pinned: with adc_bridge's default 1.0/2.0 band the
    counter advances two per clock, because the x between the thresholds
    reaches Icarus as a second posedge. Guards against the fix being undone by
    a 'harmless' default change somewhere else."""
    lines = stage(tmp_path, counter_vvp, NETLIST.splitlines())
    names, rows = simulate(tmp_path, lines, "raw")
    assert counts(names, rows, 7) == [2, 4, 6, 8, 10, 12, 14]


def test_collapsed_band_advances_once_per_clock(tmp_path, counter_vvp):
    lines, collapsed = collapse_adc_band_for_hdl(
        stage(tmp_path, counter_vvp, NETLIST.splitlines()))
    assert [c[0] for c in collapsed] == ["u1"]

    names, rows = simulate(tmp_path, lines, "fixed")
    assert counts(names, rows, 9) == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_one_clock_strobe_is_one_clock_wide(tmp_path, counter_vvp):
    """The wrap strobe is the signal a frozen or double-clocked model gets
    wrong while the datapath still looks plausible."""
    lines, _ = collapse_adc_band_for_hdl(
        stage(tmp_path, counter_vvp, NETLIST.splitlines()))
    names, rows = simulate(tmp_path, lines, "strobe")

    index = {n: i for i, n in enumerate(names)}
    edges, previous = [], 0
    for row in rows:
        level = 1 if row[index["w"]] > 2.5 else 0
        if level != previous:
            edges.append(row[index["time"]])
            previous = level

    assert len(edges) == 2, "expected exactly one strobe, got %d edges" % len(
        edges)
    assert edges[1] - edges[0] == pytest.approx(1e-3, abs=5e-6)
