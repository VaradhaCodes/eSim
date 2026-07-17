"""Syntax-highlighting tests for the shared code-editor theme + lexers.

These pin the S2 Verilog fixes and guard the shared blast radius:

* port declarations (`input wire clk`) colour the direction/net keywords as
  keywords and the signal name as a plain identifier -- not one black blob;
* modern SystemVerilog keywords (`logic`, `always_ff`, `typedef`, …) are
  recognised;
* system tasks (`$display`) read apart from control keywords;
* the dead branch of a `` `ifdef `` is muted rather than full strength;
* SPICE and VHDL classification is unchanged (the theme is shared).

``_classify``/``_mute`` are pure; the lexer/styling tests need a QApplication
(the ``qapp`` fixture) because they instantiate Qt objects.
"""
from PyQt6.QtGui import QColor

from codeEditor import lexers, theme

# Scintilla "style at position" query (stable ABI id).
_SCI_GETSTYLEAT = 2010

# QsciLexerVerilog style numbers exercised below.
_PRIMARY_KEYWORD = 5
_SYSTEM_TASK = 8
_IDENTIFIER = 11


# ── pure _classify / _mute (no Qt) ────────────────────────────────────
def test_system_task_reads_as_function_not_control_keyword():
    assert theme._classify("System task")[0] == theme.FUNCTION
    # control keywords keep the bold keyword colour
    assert theme._classify("Primary keywords and identifiers") == (
        theme.KEYWORD, True, False)


def test_inactive_style_is_muted_and_unbolded():
    base, base_bold, _ = theme._classify("Primary keywords and identifiers")
    colour, bold, _ = theme._classify(
        "Inactive primary keywords and identifiers")
    assert base_bold is True and bold is False      # dead code never shouts
    assert colour == theme._mute(base) != base      # faded, not full strength


def test_inactive_comment_keeps_italic():
    assert theme._classify("Inactive comment")[2] is True


def test_mute_blends_each_channel_toward_paper():
    muted = QColor(theme._mute(theme.KEYWORD, amount=0.5))
    kw, paper = QColor(theme.KEYWORD), QColor(theme.PAPER)
    for got, src, dst in (
        (muted.red(), kw.red(), paper.red()),
        (muted.green(), kw.green(), paper.green()),
        (muted.blue(), kw.blue(), paper.blue()),
    ):
        assert min(src, dst) <= got <= max(src, dst)


def test_spice_and_vhdl_classification_unchanged():
    # Guards the shared theme: the Verilog-only edits (inactive/system-task)
    # must not perturb how SPICE or VHDL styles are coloured.
    assert theme._classify("Directive") == (theme.KEYWORD, True, False)
    assert theme._classify("Command") == (theme.KEYWORD, True, False)
    assert theme._classify("Instance device") == (theme.INSTANCE, True, False)
    assert theme._classify("Node net") == (theme.NODE, False, False)
    assert theme._classify("Expression") == (theme.EXPRESSION, False, False)
    assert theme._classify("Keyword") == (theme.KEYWORD, True, False)
    assert theme._classify("Standard function") == (theme.FUNCTION, False, False)
    assert theme._classify("Comment") == (theme.COMMENT, False, True)


# ── lexer keyword set (needs QApplication) ────────────────────────────
def test_systemverilog_keywords_present_and_join_bug_repaired(qapp):
    words = set(lexers.VerilogLexer().keywords(1).split())
    assert {"logic", "bit", "byte", "int", "typedef", "enum", "struct",
            "always_ff", "always_comb", "always_latch"} <= words
    # stock QScintilla ships "endprimitiveendspecify" with no space.
    assert "endprimitiveendspecify" not in words
    assert {"endprimitive", "endspecify"} <= words


def test_make_lexer_uses_verilog_lexer_for_v_and_sv(qapp):
    font = theme.editor_font()
    assert isinstance(lexers.make_lexer("m.v", font), lexers.VerilogLexer)
    assert isinstance(lexers.make_lexer("m.sv", font), lexers.VerilogLexer)


# ── end-to-end styling (needs QApplication) ───────────────────────────
def _style_lookup(text, path="m.v"):
    from PyQt6.Qsci import QsciScintilla

    font = theme.editor_font()
    lexer = lexers.make_lexer(path, font)
    editor = QsciScintilla()
    editor.setLexer(lexer)
    theme.apply(editor, lexer, font)
    editor.setText(text)
    editor.recolor()

    def style_of(token):
        return editor.SendScintilla(_SCI_GETSTYLEAT, text.find(token))

    return style_of


def test_port_keywords_keep_keyword_style_names_stay_plain(qapp):
    style = _style_lookup(
        "module m (input wire clk, output reg dout, inout wire bidir);\n"
        "endmodule\n")
    for kw in ("input", "output", "inout", "wire", "reg"):
        assert style(kw) == _PRIMARY_KEYWORD, kw
    for name in ("clk", "dout", "bidir"):
        assert style(name) == _IDENTIFIER, name


def test_systemverilog_keyword_is_highlighted(qapp):
    style = _style_lookup(
        "module m;\n  logic ready;\n"
        "  always_ff @(posedge clk) ready <= 1'b1;\nendmodule\n")
    assert style("logic") == _PRIMARY_KEYWORD
    assert style("always_ff") == _PRIMARY_KEYWORD


def test_system_task_uses_distinct_style(qapp):
    style = _style_lookup('initial $display("hi");\n')
    # lexer tags it as a system task...
    assert style("$display") == _SYSTEM_TASK
    # ...and the theme colours that apart from control keywords.
    assert theme._classify("System task")[0] != theme.KEYWORD
