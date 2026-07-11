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
