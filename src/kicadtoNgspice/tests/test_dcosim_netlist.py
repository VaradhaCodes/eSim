# ==============================================================================
#  test_dcosim_netlist.py -- unit tests for the d_cosim netlister helpers added
#  for Icarus Verilog co-simulation: _get_event_plot_nodes (which event/digital
#  nodes get an `eprint`) runs as a pure function on hand-built netlist lines, so
#  no kicad-cli / ngspice / schematic is required.
# ==============================================================================
import os
import sys

# KicadtoNgspice uses package-relative imports (`from . import Analysis`), so it
# must be imported as part of the `kicadtoNgspice` package, not top-level. Put
# `src` on the path so the package (and `from maker import ...` inside it)
# resolve, exactly as eSim's pathmagic does at runtime.
SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Headless Qt: KicadtoNgspice pulls in PyQt6 at import time.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from kicadtoNgspice.KicadtoNgspice import _get_event_plot_nodes  # noqa: E402


def test_dcosim_output_node_is_event():
    sch = [
        'adut [ a ] [ y ] dut',
        '.model dut d_cosim simulation="ivlng" sim_args=["inv"]',
    ]
    plot = ['plot v(y)']
    assert _get_event_plot_nodes(sch, plot) == ['y']


def test_plain_analog_node_excluded():
    # 'ain' is a normal analog node (a voltage source), never an event node.
    sch = [
        'vin ain 0 pulse(0 1 0 1u 1u 5u 12u)',
        'adut [ a ] [ y ] dut',
        '.model dut d_cosim simulation="ivlng" sim_args=["inv"]',
    ]
    plot = ['plot v(ain)', 'plot v(y)']
    assert _get_event_plot_nodes(sch, plot) == ['y']


def test_adc_bridge_only_output_group_is_event():
    # adc_bridge: input group is analog, output group is the digital/event side.
    sch = [
        'aadc [ ain ] [ d ] adc_b',
        '.model adc_b adc_bridge(in_low=0.4 in_high=0.6)',
    ]
    assert _get_event_plot_nodes(sch, ['plot v(ain)']) == []
    assert _get_event_plot_nodes(sch, ['plot v(d)']) == ['d']


def test_order_preserved_and_deduped():
    sch = [
        'adut [ a b ] [ y z ] dut',
        '.model dut d_cosim simulation="ivlng" sim_args=["m"]',
    ]
    plot = ['plot v(z)', 'plot v(y)', 'plot v(z)']
    assert _get_event_plot_nodes(sch, plot) == ['z', 'y']


def test_no_models_returns_empty():
    assert _get_event_plot_nodes(['r1 a b 1k'], ['plot v(a)']) == []
