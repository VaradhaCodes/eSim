"""Tests for the Qt-free VCD parser (maker.hdl.vcd)."""
from maker.hdl.vcd import format_vcd_val, parse_vcd_for_plot


VCD = """\
$timescale 1ns $end
$var wire 1 ! clk $end
$var wire 4 # count $end
$enddefinitions $end
#0
0!
b0000 #
#5
1!
#10
0!
b0001 #
"""


def test_parse_basic_forward_fill():
    ts, signals, types, raw, timescale = parse_vcd_for_plot(VCD)
    assert ts == [0, 5, 10]
    # clk toggles 0 -> 1 -> 0 across the three recorded times
    assert signals['clk'] == [0, 1, 0]
    # count holds 0 until it changes to 1 at t=10 (forward fill, not reset)
    assert signals['count'] == [0, 0, 1]
    assert types == {'clk': 'wire', 'count': 'wire'}
    assert timescale == '1ns'


def test_parse_empty_returns_all_none():
    assert parse_vcd_for_plot("") == (None, None, None, None, None)


def test_format_single_bit_passthrough():
    assert format_vcd_val('0', 1) == '0'
    assert format_vcd_val('1', 1) == '1'


def test_format_xz_passthrough():
    assert format_vcd_val('x', 1) == 'x'
    assert format_vcd_val('z', 4) == 'z'


def test_format_multibit_is_hex():
    assert format_vcd_val('1010', 4) == '0xa'


def test_format_ascii_only_when_named_and_wide():
    # "Hey" = 0x48 0x65 0x79, 24 bits, var name hints a string -> decoded.
    bits = format(0x48, '08b') + format(0x65, '08b') + format(0x79, '08b')
    assert format_vcd_val(bits, 24, 'msg') == '"Hey"'
    # Same bits but a neutral counter name -> stays hex (no false positive).
    assert format_vcd_val(bits, 24, 'counter') == hex(int(bits, 2))
