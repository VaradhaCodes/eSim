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


def test_tab_is_named_after_the_module_it_holds(verifier):
    """Tab labels come from the code, not from whatever the caller called the
    tab. The default tab used to read 'design.v' while holding `module
    counter`, which taught users that the file name and the module name were
    two things they had to keep in step by hand."""
    assert verifier.editor_tabs.tabText(0) == "counter.v"
    verifier.add_module_tab("ignored.v", "module nand_gate; endmodule")
    names = [verifier.editor_tabs.tabText(i)
             for i in range(verifier.editor_tabs.count())]
    assert "nand_gate.v" in names
    assert "ignored.v" not in names


def test_renaming_the_module_renames_its_tab(verifier):
    """The one way to rename anything is to rename the module in the code --
    and then it renames everything, tab included."""
    editor = verifier.design_views[0]
    editor.setText("module half_adder(input a, output s);\nendmodule\n")
    assert verifier.editor_tabs.tabText(
        verifier.editor_tabs.indexOf(editor)) == "half_adder.v"


def test_duplicate_tab_names_are_disambiguated(verifier):
    # Two tabs really can define the same module (the same design pasted
    # twice). The hierarchy + serialiser key on the tab label, so the second
    # one must not silently take the first one's identity.
    verifier.add_module_tab("a.v", "module a; endmodule")
    verifier.add_module_tab("a.v", "module a; endmodule")
    names = [verifier.editor_tabs.tabText(i)
             for i in range(verifier.editor_tabs.count())]
    assert names.count("a.v") == 1
    assert "a (2).v" in names


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
