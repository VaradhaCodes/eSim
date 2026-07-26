# ==============================================================================
#  test_subcircuit_result.py -- what Convert reports after building a .sub.
#
#  The ngspice model is the entire point of the Subcircuit Builder, and its
#  creation was reported only to stdout. Worse, the generic "conversion
#  completed successfully!" dialog fired BEFORE the .sub was written, so a
#  schematic with no PORT element announced success and then immediately raised
#  an error about the file it had not produced.
# ==============================================================================
import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
SRC = os.path.dirname(PKG)
for _p in (SRC, PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from PyQt6 import QtWidgets                                      # noqa: E402
from kicadtoNgspice import KicadtoNgspice                        # noqa: E402
from configuration import Dialogs                                # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

CIR_OUT = """* half_adder
u1 x in_a in_b sum carry port
r1 in_a sum 1k
.tran 1m 10m
.end
"""


@pytest.fixture
def window():
    """An uninitialised MainWindow: createSubFile only needs the two
    attributes it sets itself."""
    return KicadtoNgspice.MainWindow.__new__(KicadtoNgspice.MainWindow)


@pytest.fixture
def shown(monkeypatch):
    """Capture every dialog raised, in order."""
    seen = []
    for kind in ('information', 'critical'):
        monkeypatch.setattr(
            Dialogs, kind,
            lambda parent, title, text, *a, **k: seen.append(
                (title, text, k.get('informative_text', ''))))
    return seen


def _stem(tmp_path, name='half_adder', body=CIR_OUT):
    base = os.path.join(str(tmp_path), name)
    with open(base + '.cir.out', 'w') as fh:
        fh.write(body)
    return base


# -- createSubFile reports its outcome ---------------------------------------

def test_a_written_model_reports_success(window, tmp_path, shown):
    base = _stem(tmp_path)
    assert window.createSubFile(base) is True
    assert os.path.isfile(base + '.sub')


def test_a_schematic_without_a_port_reports_failure(window, tmp_path, shown):
    base = _stem(tmp_path, body='r1 a b 1k\n.end\n')
    assert window.createSubFile(base) is False
    assert not os.path.isfile(base + '.sub')
    assert 'Subcircuit creation failed' in shown[0][0]


def test_a_missing_netlist_reports_failure(window, tmp_path, shown):
    assert window.createSubFile(os.path.join(str(tmp_path), 'absent')) is False
    assert 'Subcircuit creation failed' in shown[0][0]


# -- the report itself -------------------------------------------------------

def test_the_report_names_the_file_and_its_ports(window, tmp_path, shown):
    base = _stem(tmp_path)
    window.createSubFile(base)
    window._reportSubcircuitBuilt(base)

    title, text, detail = shown[-1]
    assert title == 'Subcircuit built'
    assert 'half_adder.sub' in text
    assert base + '.sub' in detail
    # The .subckt header is echoed so the port list can be checked against the
    # parent circuit's symbol without opening anything.
    assert '.subckt half_adder' in detail
    assert '4 ports' in detail
    assert 'in_a, in_b, sum, carry' in detail


def test_the_report_survives_a_deleted_model_file(window, tmp_path, shown):
    """The dialog is about reassurance; it must not become a second failure if
    the file vanished between writing and reporting."""
    base = _stem(tmp_path)
    window.createSubFile(base)
    os.remove(base + '.sub')
    window._reportSubcircuitBuilt(base)
    assert shown[-1][0] == 'Subcircuit built'


def test_a_single_port_is_not_pluralised(window, tmp_path, shown):
    base = _stem(tmp_path, name='one', body='u1 x only port\n.end\n')
    window.createSubFile(base)
    window._reportSubcircuitBuilt(base)
    assert '1 port:' in shown[-1][2]
