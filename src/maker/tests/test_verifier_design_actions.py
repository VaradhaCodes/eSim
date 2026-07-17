"""S5: design-side actions wired through the live Verify widget.

The structural logic is unit-tested in test_hdl_structure.py; this file pins the
*wiring* -- that the widget actually feeds tab labels/code into order_modules,
disambiguates colliding tab names, generates a width-correct testbench end to
end, and survives a non-UTF8 source file.

Ungated -- headless widget, no iverilog needed.
"""
import pytest

from PyQt6 import QtCore


@pytest.fixture
def verifier(qapp):
    from maker.VerilogVerifier import VerilogVerifier
    w = VerilogVerifier()
    w.unlock_ui()
    yield w
    w.deleteLater()


def _hierarchy_names(w):
    return [w.hierarchy_list.item(i).data(QtCore.Qt.ItemDataRole.UserRole)
            for i in range(w.hierarchy_list.count())]


def test_duplicate_tab_names_are_disambiguated(verifier):
    # Default design tab is 'design.v'; loading another 'design.v' must not
    # collide (the hierarchy + serialiser key on the tab label).
    verifier.add_module_tab("design.v", "module a; endmodule")
    names = [verifier.editor_tabs.tabText(i)
             for i in range(verifier.editor_tabs.count())]
    assert names.count("design.v") == 1
    assert "design (2).v" in names


def test_auto_detect_orders_parent_before_child(verifier):
    verifier.add_module_tab("child.v", "module child(input c); endmodule")
    verifier.add_module_tab(
        "top.v", "module top(input c); child u0(.c(c)); endmodule")
    verifier.auto_detect_hierarchy()
    names = _hierarchy_names(verifier)
    assert names.index("top.v") < names.index("child.v")


def test_get_design_code_follows_detected_order(verifier):
    # Remove the default design tab, add two in reverse dependency order.
    verifier.close_tab(0)
    verifier.add_module_tab("child.v", "module child(input c); endmodule")
    verifier.add_module_tab(
        "top.v", "module top(input c); child u0(.c(c)); endmodule")
    verifier.auto_detect_hierarchy()
    code = verifier.get_design_code()
    assert code.index("module top") < code.index("module child")


def test_auto_generate_tb_widths_for_single_line_ansi(verifier):
    # The exact case the old path produced an empty, portless TB for.
    verifier.add_module_tab(
        "alu.v", "module alu(input [3:0] a, input [3:0] b, output [7:0] y);\nendmodule")
    verifier.editor_tabs.setCurrentIndex(
        verifier.editor_tabs.indexOf(verifier.design_views[-1]))
    verifier.auto_generate_tb()
    tb = verifier.tb_view.toPlainText()
    assert "module tb_alu;" in tb
    assert "reg [3:0] a;" in tb
    assert "wire [7:0] y;" in tb
    assert "alu uut (" in tb


def test_read_text_falls_back_on_non_utf8(tmp_path):
    from maker.VerilogVerifier import VerilogVerifier
    f = tmp_path / "legacy.v"
    # 0xE9 ('é' in latin-1) is invalid as standalone UTF-8.
    f.write_bytes(b"// design by Jos\xe9\nmodule m; endmodule\n")
    text = VerilogVerifier._read_text(str(f))
    assert "module m" in text
    assert "Jos" in text
