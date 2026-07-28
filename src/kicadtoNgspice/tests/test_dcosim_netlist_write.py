"""Integration: the guards actually reach the netlist eSim writes.

test_dcosim_netlist.py proves the two transforms in isolation. This file drives
the real ``MainWindow.createNetlistFile`` -- the single funnel every conversion
goes through -- and reads the ``.cir.out`` it leaves on disk, so a fix that is
correct but not wired up cannot pass.

The instance is built with ``__new__`` and given only the seven attributes the
method touches (``optionInfo``, ``outputOption``, ``kicadFile``, ``infoline``,
``obj_appconfig``, ``modelList``, ``microcontrollerList``). That is the whole
input surface: no schematic parse, no Qt window, no dialogs.
"""
import pytest

from configuration.Appconfig import Appconfig
from kicadtoNgspice.KicadtoNgspice import MainWindow

# schematicInfo as Convert leaves it: a-devices, then the .model cards.
COSIM_NETLIST = [
    'v1 clk_a gnd pulse(0 5 0 1u 1u 0.5m 1m)',
    'a1 [clk_a rst_a] [clk_d rst_d] u1',
    'a2 [clk_d rst_d] [cnt0_d] u2',
    'a3 [cnt0_d] [cnt0] u3',
    '.model u1 adc_bridge(in_low=1.0 in_high=2.0 rise_delay=1.0e-9 ) ',
    '.model u2 d_cosim simulation="ivlng" sim_args=["counter"] ',
    '.model u3 dac_bridge(out_low=0.0 out_high=5.0 ) ',
]

#: modelList rows are [index, compline, modelname, card, comment, title, type,
#: params] -- only the card name and the type matter here.
def model_row(card, mtype, modelname='counter'):
    return [0, '', modelname, card, '', '', mtype, {}]


@pytest.fixture
def converter(tmp_path):
    """A MainWindow with just enough state to run createNetlistFile.

    No QApplication: the method is pure file work, which is exactly why it is
    the right seam to test the wiring at."""
    project = tmp_path / "tb.cir"
    project.write_text("* tb\n")
    (tmp_path / "analysis").write_text(".tran 10e-06 15e-03 0e-00\n")

    window = MainWindow.__new__(MainWindow)
    window.kicadFile = str(project)
    window.infoline = "* tb"
    window.optionInfo = []
    window.outputOption = []
    window.modelList = []
    window.microcontrollerList = []
    window.obj_appconfig = Appconfig()
    return window, project


def written(project):
    with open(str(project) + ".out") as fh:
        return fh.read()


def test_a_d_cosim_netlist_is_written_with_one_threshold(converter):
    window, project = converter
    window.createNetlistFile(list(COSIM_NETLIST), [])

    text = written(project)
    assert 'in_low=1.5 in_high=1.5' in text
    assert 'in_low=1.0' not in text
    assert 'rise_delay=1.0e-9' in text          # nothing else disturbed
    assert 'd_cosim' in text


def test_an_ngveri_block_named_in_modellist_is_collapsed_too(converter):
    """The Verilator backend cannot be spotted in the netlist text -- the card
    type is the HDL entity name -- so the pass reads modelList. If that wiring
    breaks, the analog half of the netlist quietly stops matching d_cosim."""
    window, project = converter
    netlist = [ln.replace('.model u2 d_cosim simulation="ivlng" '
                          'sim_args=["counter"] ',
                          '.model u2 counter(instance_id=1 ) ')
               for ln in COSIM_NETLIST]
    window.modelList = [model_row('u2', 'Ngveri')]
    window.createNetlistFile(netlist, [])

    assert 'in_low=1.5 in_high=1.5' in written(project)


def test_an_nghdl_block_is_collapsed_from_the_microcontroller_list(converter):
    """Nghdl rows are moved out of modelList during the parse; reading only
    modelList would miss every VHDL block."""
    window, project = converter
    netlist = [ln.replace('.model u2 d_cosim simulation="ivlng" '
                          'sim_args=["counter"] ',
                          '.model u2 counter(instance_id=1 ) ')
               for ln in COSIM_NETLIST]
    window.microcontrollerList = [model_row('u2', 'Nghdl')]
    window.createNetlistFile(netlist, [])

    assert 'in_low=1.5 in_high=1.5' in written(project)


def test_a_plain_analog_netlist_is_untouched(converter):
    window, project = converter
    netlist = [ln for ln in COSIM_NETLIST if 'd_cosim' not in ln]
    netlist = [ln for ln in netlist if not ln.startswith('a2')]
    window.createNetlistFile(netlist, [])

    assert 'in_low=1.0 in_high=2.0' in written(project)


def test_two_cosim_blocks_are_refused_before_anything_is_written(converter):
    """ngspice segfaults on the second libvvp engine, so the conversion has to
    stop here rather than hand the user a netlist that crashes the simulator.
    callConvert turns this into the "Conversion failed" dialog."""
    window, project = converter
    netlist = COSIM_NETLIST + ['a4 [clk_d rst_d] [cnt1_d] u2']

    with pytest.raises(RuntimeError) as excinfo:
        window.createNetlistFile(netlist, [])

    message = str(excinfo.value)
    assert "2 Verilog co-simulation" in message
    assert "NgVeri" in message                  # names the way out
    import os
    assert not os.path.exists(str(project) + ".out")


def test_the_analysis_card_still_moves_into_control_for_d_cosim(converter):
    """Pre-existing behaviour the collapse pass must not disturb: a d_cosim
    netlist runs its analysis once, inside .control, because the vvp cannot be
    re-run."""
    window, project = converter
    window.createNetlistFile(list(COSIM_NETLIST), [])

    text = written(project)
    assert '\ntran 10e-06 15e-03 0e-00' in text
    assert '\n.tran' not in text
