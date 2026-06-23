"""S6: Verify-stage run-side hardening (GUI-wired, but iverilog-free).

Pins the behaviour that only shows up around the compile/sim/plot pipeline and
its async/cancel/lifecycle machinery -- none of which needs a toolchain to
verify, because the failures are in our own plumbing:

* compile filenames are sanitised so an error squiggle / jump survives a tab
  label the diagnostic regex can't carry (spaces, parens);
* a partial failure (sim exits nonzero but still wrote a VCD) still plots, yet
  does not report success;
* a cancelled run renders nothing;
* the per-run temp dir is reaped even when the stage is torn down mid-run.

Ungated -- constructs the widget headlessly.
"""
import re

import pytest

from PyQt6 import QtGui


@pytest.fixture
def verifier(qapp):
    from maker.VerilogVerifier import VerilogVerifier
    w = VerilogVerifier()
    w.unlock_ui()
    yield w
    w.deleteLater()


# --- error -> editor mapping survives messy tab labels ------------------- #

def test_safe_source_name_is_diagnostic_regex_safe():
    from maker.VerilogVerifier import VerilogVerifier as V
    assert V._safe_source_name("alu.v") == "alu.v"
    assert V._safe_source_name("counter") == "counter.v"      # extension added
    n = V._safe_source_name("foo (2).v")                       # S5 disambiguation
    assert n.endswith(".v")
    # the squiggle/jump regex is [\w./-]+ ; spaces and parens would break it
    assert re.fullmatch(r"[\w./-]+\.v", n)


def test_design_sources_names_are_unique_after_sanitising(verifier):
    # Two tabs whose sanitised names would collide must still map distinctly,
    # so no module silently overwrites another on disk at compile time.
    verifier.add_module_tab("a b.v", "module x; endmodule")
    verifier.add_module_tab("a_b.v", "module y; endmodule")
    verifier._design_sources()
    names = list(verifier._compile_editors)
    assert len(names) == len(set(names))                       # collision-free
    assert all(re.fullmatch(r"[\w./-]+\.(?:v|sv)", n) for n in names)


def test_error_squiggle_survives_spaced_duplicate_label(verifier):
    verifier.add_module_tab("alu.v", "module alu; endmodule")
    verifier.add_module_tab("alu.v", "module alu2; endmodule")   # -> 'alu (2).v'
    labels = [verifier.editor_tabs.tabText(i)
              for i in range(verifier.editor_tabs.count())]
    assert "alu (2).v" in labels

    verifier._design_sources()
    safe = [n for n in verifier._compile_editors if n != "alu.v"
            and verifier._compile_editors[n].toPlainText().startswith("module alu2")]
    assert safe, "duplicate tab should get its own sanitised compile name"
    name = safe[0]
    assert " " not in name and "(" not in name

    second = verifier._compile_editors[name]
    hits = []
    second.mark_error_line = lambda ln: hits.append(ln)
    verifier.highlight_errors_from_log(f"{name}:7: error: oops")
    assert hits == [7]


def test_jump_to_error_uses_compile_name_map(verifier):
    verifier.add_module_tab("alu.v", "module alu; endmodule")
    verifier._design_sources()
    ed = verifier._compile_editors["alu.v"]
    goto = []
    ed.mark_error_line = lambda ln: None
    ed.goto_line = lambda ln: goto.append(ln)
    verifier.jump_to_error(5, "alu.v")
    assert goto == [5]


# --- success / failure detection ----------------------------------------- #

def _run(compile_ok, sim_ok=None, vcd=None):
    from maker.hdl.icarus import RunResult, CompileResult, SimResult
    cres = CompileResult(ok=compile_ok, returncode=0 if compile_ok else 1,
                         stdout="", stderr="" if compile_ok else "boom",
                         out_path="x" if compile_ok else None)
    sres = None
    if sim_ok is not None:
        sres = SimResult(ok=sim_ok, returncode=0 if sim_ok else 1,
                         stdout="", stderr="", vcd_path="x" if vcd else None)
    return RunResult(compile=cres, sim=sres, vcd_content=vcd)


def test_partial_failure_still_plots_without_success_signal(verifier):
    vcd = ("$timescale 1ns $end\n$var wire 1 ! clk $end\n"
           "$enddefinitions $end\n#0\n0!\n#5\n1!\n")
    run = _run(compile_ok=True, sim_ok=False, vcd=vcd)
    plotted, fired = [], []
    verifier.render_waveform = lambda *a: plotted.append(a)
    verifier.simulationSucceeded.connect(lambda: fired.append(1))
    verifier._render_sim_result(run, cancelled=False)
    assert plotted              # nonzero sim exit but a VCD -> still plotted
    assert not fired            # run.ok is False -> no success report


def test_clean_run_emits_success(verifier):
    vcd = ("$timescale 1ns $end\n$var wire 1 ! clk $end\n"
           "$enddefinitions $end\n#0\n0!\n#5\n1!\n")
    run = _run(compile_ok=True, sim_ok=True, vcd=vcd)
    fired = []
    verifier.render_waveform = lambda *a: None
    verifier.simulationSucceeded.connect(lambda: fired.append(1))
    verifier._render_sim_result(run, cancelled=False)
    assert fired == [1]


def test_cancelled_run_renders_nothing(verifier):
    run = _run(compile_ok=False)
    plotted = []
    verifier.render_waveform = lambda *a: plotted.append(a)
    verifier._render_sim_result(run, cancelled=True)
    assert not plotted


# --- temp dir hygiene ---------------------------------------------------- #

def test_cleanup_tmpdir_is_idempotent(verifier, tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    verifier._active_tmpdir = str(d)
    verifier._cleanup_tmpdir()
    assert not d.exists()
    assert verifier._active_tmpdir is None
    verifier._cleanup_tmpdir()          # second call must be a harmless no-op


def test_closeevent_reaps_orphaned_tmpdir(verifier, tmp_path):
    # Simulate a stage torn down mid-run: the done/fail closures never fired,
    # so closeEvent is the only thing left to reap the temp dir.
    d = tmp_path / "orphan"
    d.mkdir()
    verifier._active_tmpdir = str(d)
    verifier.closeEvent(QtGui.QCloseEvent())
    assert not d.exists()
