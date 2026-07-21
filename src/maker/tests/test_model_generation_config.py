import importlib
import os

from PyQt6 import QtWidgets

from maker import CosimConfig, ModelGeneration


def test_constructor_survives_missing_nghdl_config(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    importlib.reload(CosimConfig)

    terminal = QtWidgets.QTextEdit()
    model = ModelGeneration.ModelGeneration(
        str(tmp_path / "counter.v"), terminal)

    assert model.nghdl_home == ""
    assert model.release_dir == ""
    assert model.src_home == ""
    assert model.digital_home == os.path.join(
        str(tmp_path), ".nghdl", "DigitalModelLibrary", "Ngveri")
    assert model.require_legacy_toolchain() is False
    assert "toolchain not configured" in terminal.toPlainText()


def test_constructor_accepts_partial_nghdl_config(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    config_dir = tmp_path / ".nghdl"
    config_dir.mkdir()
    (config_dir / "config.ini").write_text(
        "[NGHDL]\nNGHDL_HOME=/opt/nghdl\n"
    )
    importlib.reload(CosimConfig)

    terminal = QtWidgets.QTextEdit()
    model = ModelGeneration.ModelGeneration(
        str(tmp_path / "counter.v"), terminal)

    assert model.nghdl_home == "/opt/nghdl"
    assert model.release_dir == ""
    assert model.require_legacy_toolchain() is False


def test_verilog_parse_keeps_wire_reg_in_identifiers(qapp, tmp_path,
                                                     monkeypatch):
    """M1: the wire/reg keyword strip must not punch holes in identifiers
    that merely CONTAIN 'wire'/'reg' (out_reg, wire_en, data_reg). A bare
    substring replace mangled the module name (parse then fails to match)
    and the port names; a word-boundary regex leaves them intact."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    importlib.reload(CosimConfig)

    # Module and port names each contain 'reg'/'wire' as a sub-token.
    verilog = (
        "module out_reg(clk, wire_en, data_reg, q);\n"
        "input clk, wire_en;\n"
        "output reg data_reg;\n"
        "output reg q;\n"
        "endmodule\n"
    )
    (tmp_path / "out_reg.v").write_text(verilog)

    terminal = QtWidgets.QTextEdit()
    model = ModelGeneration.ModelGeneration(
        str(tmp_path / "out_reg.v"), terminal)
    model.modelpath = str(tmp_path) + os.sep

    # Module name "out_reg" survives -> the parse matches -> "No Error".
    assert model.verilogParse(make_symbol=False) == "No Error"

    conn = (tmp_path / "connection_info.txt").read_text()
    # Every port identifier lands intact, un-mangled.
    for port in ("clk", "wire_en", "data_reg", "q"):
        assert port in conn
    # The old substring replace would have left the truncated stems instead.
    assert "data_ " not in conn
    assert "_en" not in conn.replace("wire_en", "")
