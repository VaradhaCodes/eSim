# ==============================================================================
#  test_rawfile_names.py -- unit test for DataExtraction._full_voltage_names,
#  which recovers full (untruncated) node names from the ASCII rawfile to undo
#  ngspice `print allv` column-name truncation. Pure parse, no Qt/ngspice.
# ==============================================================================
import os
import sys
import tempfile

SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ngspiceSimulation.data_extraction import DataExtraction  # noqa: E402


_RAW = """Title: relu32
Date: Sat Jun 13 13:51:22  2026
Plotname: Transient Analysis
Flags: real
No. Variables: 5
No. Points: 3
Variables:
\t0\ttime\ttime
\t1\tv(plot_vout_bit_9)\tvoltage
\t2\tv(plot_vout_bit_10)\tvoltage
\t3\tv(plot_vout_bit_31)\tvoltage
\t4\ti(v1)\tcurrent
Values:
0\t0.0
\t0.0
\t0.0
\t0.0
\t0.0
"""


def _names_from(raw):
    # _full_voltage_names is a staticmethod -> no DataExtraction() instance
    # (its __init__ builds a QWidget needing a QApplication).
    fd, path = tempfile.mkstemp(suffix=".raw")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(raw)
        return DataExtraction._full_voltage_names(path)
    finally:
        os.remove(path)


def test_recovers_full_voltage_names_in_order():
    # time + current skipped; v(...) unwrapped to the inner node name, in order.
    assert _names_from(_RAW) == [
        "plot_vout_bit_9", "plot_vout_bit_10", "plot_vout_bit_31"]


def test_missing_rawfile_returns_empty():
    assert DataExtraction._full_voltage_names("/no/such/file.raw") == []


def test_no_variables_section_returns_empty():
    assert _names_from("Title: x\nValues:\n0\t0.0\n") == []
