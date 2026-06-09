# ==============================================================================
#  test_netlister_unit.py -- unit tests for KicadNetlister internals that the
#  golden schematics do not exercise (none of the re-saved KiCad-9 examples carry
#  a Spice_Node_Sequence / Spice_Netlist_Enabled field, and none collide on net
#  names). These run the pure functions directly on hand-built kicadxml, so no
#  kicad-cli is required.
# ==============================================================================
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
for _p in (HERE, PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import KicadNetlister as K     # noqa: E402


def _lines_from_xml(xml):
    """Write kicadxml to a temp file and return the netlister's spice lines."""
    fd, path = tempfile.mkstemp(suffix=".xml")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(xml)
        return K.xml_to_spice_lines(path, title="unit")
    finally:
        os.remove(path)


def _comp_line(lines, ref):
    for ln in lines:
        if ln.split() and ln.split()[0] == ref:
            return ln
    return None


# -- sanitizer -----------------------------------------------------------------
def test_sanitize_lowercases_and_keeps_clean_names():
    assert K._sanitize_net("GND") == "gnd"
    assert K._sanitize_net("in") == "in"
    assert K._sanitize_net("V_Out") == "v_out"


def test_sanitize_maps_operator_chars_to_underscore():
    # '+', '-', '/' are operators inside ngspice v()/i(); must not survive.
    for raw in ("Net-(v1-+)", "Net-(C1-Pad1)", "/sheet1/clk", "+5V"):
        out = K._sanitize_net(raw)
        assert all(c.isalnum() or c == "_" for c in out), out
        assert not out.startswith("_") and not out.endswith("_"), out


# -- node ordering -------------------------------------------------------------
def test_pins_emitted_in_pin_number_order():
    xml = """<export><components>
      <comp ref="Q1"><value>eSim_NPN</value></comp>
    </components><nets>
      <net code="1" name="C"><node ref="Q1" pin="1"/></net>
      <net code="2" name="B"><node ref="Q1" pin="2"/></net>
      <net code="3" name="E"><node ref="Q1" pin="3"/></net>
    </nets></export>"""
    line = _comp_line(_lines_from_xml(xml), "q1")
    assert line == "q1 c b e eSim_NPN", line


def test_spice_node_sequence_reorders():
    # default order is c b e (pins 1,2,3); sequence 2,1,0 -> e b c
    xml = """<export><components>
      <comp ref="Q1"><value>eSim_NPN</value>
        <fields><field name="Spice_Node_Sequence">2,1,0</field></fields>
      </comp>
    </components><nets>
      <net code="1" name="C"><node ref="Q1" pin="1"/></net>
      <net code="2" name="B"><node ref="Q1" pin="2"/></net>
      <net code="3" name="E"><node ref="Q1" pin="3"/></net>
    </nets></export>"""
    line = _comp_line(_lines_from_xml(xml), "q1")
    assert line == "q1 e b c eSim_NPN", line


def test_node_sequence_field_name_is_case_insensitive():
    xml = """<export><components>
      <comp ref="Q1"><value>m</value>
        <fields><field name="SPICE_NODE_SEQUENCE">2,1,0</field></fields>
      </comp>
    </components><nets>
      <net code="1" name="a"><node ref="Q1" pin="1"/></net>
      <net code="2" name="b"><node ref="Q1" pin="2"/></net>
      <net code="3" name="c"><node ref="Q1" pin="3"/></net>
    </nets></export>"""
    line = _comp_line(_lines_from_xml(xml), "q1")
    assert line == "q1 c b a m", line


# -- component disable ---------------------------------------------------------
def test_spice_netlist_enabled_n_drops_component():
    xml = """<export><components>
      <comp ref="R1"><value>1k</value>
        <fields><field name="Spice_Netlist_Enabled">N</field></fields>
      </comp>
      <comp ref="R2"><value>2k</value></comp>
    </components><nets>
      <net code="1" name="a"><node ref="R1" pin="1"/><node ref="R2" pin="1"/></net>
      <net code="2" name="b"><node ref="R1" pin="2"/><node ref="R2" pin="2"/></net>
    </nets></export>"""
    lines = _lines_from_xml(xml)
    assert _comp_line(lines, "r1") is None
    assert _comp_line(lines, "r2") == "r2 a b 2k"


# -- collision safety ----------------------------------------------------------
def test_distinct_nets_never_collapse_to_same_node():
    # Two different raw names that both sanitize to "net__v1" must stay distinct,
    # otherwise the two resistors would be shorted.
    xml = """<export><components>
      <comp ref="R1"><value>1k</value></comp>
      <comp ref="R2"><value>2k</value></comp>
    </components><nets>
      <net code="7" name="Net-(v1-+)"><node ref="R1" pin="1"/></net>
      <net code="8" name="Net-(v1--)"><node ref="R2" pin="1"/></net>
      <net code="9" name="gnd"><node ref="R1" pin="2"/><node ref="R2" pin="2"/></net>
    </nets></export>"""
    lines = _lines_from_xml(xml)
    n1 = _comp_line(lines, "r1").split()[1]
    n2 = _comp_line(lines, "r2").split()[1]
    assert n1 != n2, (n1, n2)


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print("PASS  " + fn.__name__)
        except AssertionError as e:
            failed += 1
            print("FAIL  " + fn.__name__ + "  " + str(e))
    print("\n==== %d / %d PASS ====" % (len(fns) - failed, len(fns)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
