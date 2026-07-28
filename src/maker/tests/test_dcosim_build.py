"""d_cosim build-step guards: `timescale precision and multi-file resolution.

Both are silent failures at simulation time, not build time:

* ivlng advances VVP by ``(spice_time - vvp_time) / precision`` ticks and
  truncates, so a source declaring a precision coarser than one SPICE step
  never advances at all -- ngspice reports success and every output reads 0;
* dependency sources added through "Add dependency files/folder" land beside
  the top source but were never handed to iverilog, so any design with a
  submodule in another file died on "Unknown module type".

No iverilog needed: run_iverilog is stubbed so the tests assert on the command
eSim builds and on the source text it hands over.
"""
import importlib
import os

import pytest
from PyQt6 import QtWidgets

from maker import CosimConfig, ModelGeneration
from maker.ModelGeneration import normalise_timescale


# --------------------------------------------------------------------------- #
# normalise_timescale (pure)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("declared, expected, flagged", [
    # already fine enough -- untouched
    ("`timescale 1ns / 1ps", "`timescale 1ns / 1ps", []),
    ("`timescale 100ps / 10fs", "`timescale 100ps / 10fs", []),
    ("`timescale 1s/1fs", "`timescale 1s/1fs", []),
    # too coarse -- only the precision field moves
    ("`timescale 1ms / 1ms", "`timescale 1ms / 1ps", ["1ms/1ms"]),
    ("`timescale 1ns/1ns", "`timescale 1ns/1ps", ["1ns/1ns"]),
    ("`timescale 10us / 100ns", "`timescale 10us / 1ps", ["10us/100ns"]),
])
def test_precision_is_sharpened_and_the_unit_is_not(declared, expected,
                                                    flagged):
    text, coarse = normalise_timescale(declared + "\nmodule m; endmodule\n")
    assert text.splitlines()[0] == expected
    assert coarse == flagged


def test_source_without_a_directive_is_returned_unchanged():
    src = "module m; endmodule\n"
    assert normalise_timescale(src) == (src, [])


def test_every_directive_in_a_file_is_normalised():
    text, coarse = normalise_timescale(
        "`timescale 1us/1us\nmodule a; endmodule\n"
        "`timescale 1ns/1ps\nmodule b; endmodule\n"
        "`timescale 1ms/1ms\nmodule c; endmodule\n")
    assert text.count("1ps") == 3
    assert coarse == ["1us/1us", "1ms/1ms"]


# --------------------------------------------------------------------------- #
# build_cosim: what actually reaches iverilog
# --------------------------------------------------------------------------- #
class _Recorded:
    """Stands in for icarus.run_iverilog, capturing its arguments."""

    def __init__(self):
        self.calls = []

    def __call__(self, iverilog, srcs, out, **kwargs):
        self.calls.append({"srcs": list(srcs), "out": out, **kwargs})
        self.source_text = open(srcs[0]).read()
        with open(out, "w") as fh:            # a plausible vvp artifact
            fh.write("#! /usr/bin/vvp\n")
        return ModelGeneration.icarus.CompileResult(
            ok=True, returncode=0, stdout="", stderr="", out_path=out)


@pytest.fixture
def cosim(qapp, tmp_path, monkeypatch):
    """A ModelGeneration ready for build_cosim, with iverilog stubbed out."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    importlib.reload(CosimConfig)
    monkeypatch.setattr(CosimConfig, "iverilog_binary", lambda: "iverilog")
    monkeypatch.setattr(CosimConfig, "has_iverilog", lambda: True)

    model = ModelGeneration.ModelGeneration(
        str(tmp_path / "counter.v"), QtWidgets.QTextEdit())
    model.modelpath = str(tmp_path / "counter") + "/"
    os.makedirs(model.modelpath, exist_ok=True)
    monkeypatch.setattr(model, "_tool_version", lambda _b: "stub")

    recorded = _Recorded()
    monkeypatch.setattr(ModelGeneration.icarus, "run_iverilog", recorded)
    return model, recorded


def write_source(model, text):
    path = os.path.abspath(os.path.join(model.modelpath, "counter.v"))
    with open(path, "w") as fh:
        fh.write(text)
    return path


def test_dependency_directory_is_on_the_library_search_path(cosim):
    model, recorded = cosim
    write_source(model, "`timescale 1ns/1ps\nmodule counter; endmodule\n")
    assert model.build_cosim() != "Error"

    flags = recorded.calls[0]["extra_flags"]
    libdir = os.path.abspath(model.modelpath)
    assert flags[flags.index("-y") + 1] == libdir      # sibling modules
    assert flags[flags.index("-I") + 1] == libdir      # `include files
    assert flags[flags.index("-Y") + 1] == ".sv"       # SystemVerilog members


def test_a_good_timescale_is_compiled_from_the_users_own_file(cosim):
    model, recorded = cosim
    src = write_source(model,
                       "`timescale 1ns/1ps\nmodule counter; endmodule\n")
    assert model.build_cosim() != "Error"
    assert recorded.calls[0]["srcs"] == [src]         # no temp copy


def test_a_coarse_timescale_is_sharpened_before_compiling(cosim):
    model, recorded = cosim
    src = write_source(model,
                       "`timescale 1ms/1ms\nmodule counter; endmodule\n")
    assert model.build_cosim() != "Error"

    assert recorded.calls[0]["srcs"] != [src]         # compiled from a copy
    assert "`timescale 1ms/1ps" in recorded.source_text
    assert open(src).read().startswith("`timescale 1ms/1ms")   # user's intact
    assert "Sharpened" in model.clog.termedit.toPlainText()


def test_a_missing_timescale_is_still_injected(cosim):
    model, recorded = cosim
    write_source(model, "module counter; endmodule\n")
    assert model.build_cosim() != "Error"
    assert recorded.source_text.startswith("`timescale 1ns/1ps")


def test_the_temporary_copy_does_not_survive_the_build(cosim):
    model, recorded = cosim
    write_source(model, "`timescale 1ms/1ms\nmodule counter; endmodule\n")
    assert model.build_cosim() != "Error"
    assert not os.path.exists(recorded.calls[0]["srcs"][0])


# --------------------------------------------------------------------------- #
# inout: refused, not warned about
# --------------------------------------------------------------------------- #
def write_ports(model, lines):
    path = os.path.join(model.modelpath, "connection_info.txt")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def test_an_inout_port_is_refused_before_the_compiler_runs(cosim):
    """eSim's parser files inout under the inputs, so the netlist declares a
    plain d_in and d_cosim's d_inout group stays empty. ngspice then reports
    "mismatched ... input counts: 2/1" and carries on with the port indices
    shifted: measured on a probe module, the inout never left the simulation
    AND a sibling output declared `assign q = 1'b1;` toggled with the clock.
    Every port is wrong, so a warning is not enough."""
    model, recorded = cosim
    write_source(model, "`timescale 1ns/1ps\nmodule counter; endmodule\n")
    write_ports(model, ["clk input 1", "bus inout 2", "q output 1"])

    assert model.build_cosim() == "Error"
    assert recorded.calls == []                  # never reached iverilog
    log = model.clog.termedit.toPlainText()
    assert "REFUSED" in log and "bus" in log


def test_a_port_merely_named_inout_is_not_refused(cosim):
    """The direction is a field, not a substring: `inout_en` is an input."""
    model, recorded = cosim
    write_source(model, "`timescale 1ns/1ps\nmodule counter; endmodule\n")
    write_ports(model, ["inout_en input 1", "q output 1"])

    assert model.build_cosim() != "Error"
    assert len(recorded.calls) == 1


def test_a_module_with_no_port_file_still_builds(cosim):
    """connection_info.txt is written by verilogParse; a build driven without
    it must not be blocked by the inout gate."""
    model, recorded = cosim
    write_source(model, "`timescale 1ns/1ps\nmodule counter; endmodule\n")
    assert model.build_cosim() != "Error"
