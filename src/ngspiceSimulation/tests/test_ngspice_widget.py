"""Unit tests for the pure helpers on NgspiceWidget.

These avoid constructing the widget (which would launch ngspice); they exercise
the rawfile-path derivation in isolation.
"""
from ngspiceSimulation.NgspiceWidget import NgspiceWidget


# ── _raw_path ─────────────────────────────────────────────────────────────

def test_raw_path_strips_cir_out():
    assert NgspiceWidget._raw_path("/p/foo.cir.out") == "/p/foo.raw"


def test_raw_path_other_suffix_swaps_extension():
    assert NgspiceWidget._raw_path("/p/foo.cir") == "/p/foo.raw"


def test_raw_path_never_equals_netlist():
    # The old str.replace('.cir.out', '.raw') was a no-op here and let ngspice
    # overwrite the netlist via '-r'. The derived path must always differ.
    for netlist in ("/p/foo", "/p/foo.txt", "/p/bar.cir.out"):
        assert NgspiceWidget._raw_path(netlist) != netlist
