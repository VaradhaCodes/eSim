# ==============================================================================
#  test_cosim_config.py -- unit tests for CosimConfig, the d_cosim toolchain
#  resolver. Verifies the no-hardcode resolution order (env override ->
#  config.ini -> PATH / derived) and the capability gates, using temp files and
#  a monkeypatched config path so the tests are hermetic and need no real
#  iverilog/ngspice install.
# ==============================================================================
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)                 # src/maker
for _p in (HERE, PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import CosimConfig as C     # noqa: E402


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    """Each test starts with a clean env and capability cache."""
    for var in ("ESIM_NGSPICE", "ESIM_IVERILOG", "ESIM_IVERILOG_LIB"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(C, "_dcosim_capable", None)
    yield


def _write(path, text=""):
    with open(path, "w") as fh:
        fh.write(text)
    return path


def _use_config(monkeypatch, tmp_path, body):
    cfg = tmp_path / "config.ini"
    cfg.write_text(body)
    monkeypatch.setattr(C, "_config_path", lambda: str(cfg))
    return cfg


# -- ngspice -------------------------------------------------------------------
def test_ngspice_env_override_wins(monkeypatch, tmp_path):
    ng = _write(str(tmp_path / "my_ngspice"))
    monkeypatch.setenv("ESIM_NGSPICE", ng)
    assert C.ngspice_binary() == ng


def test_ngspice_from_nghdl_home(monkeypatch, tmp_path):
    home = tmp_path / "nghdl-sim"
    bindir = home / "install_dir" / "bin"
    bindir.mkdir(parents=True)
    ng = _write(str(bindir / "ngspice"))
    _use_config(monkeypatch, tmp_path,
                "[NGHDL]\nNGHDL_HOME = %s\n" % home)
    assert C.ngspice_binary() == ng


def test_ngspice_falls_back_to_path(monkeypatch, tmp_path):
    _use_config(monkeypatch, tmp_path, "[NGHDL]\nNGHDL_HOME = /nope\n")
    monkeypatch.setattr(C.shutil, "which",
                        lambda name: "/usr/bin/ngspice")
    assert C.ngspice_binary() == "/usr/bin/ngspice"


# -- iverilog ------------------------------------------------------------------
def test_iverilog_env_override(monkeypatch, tmp_path):
    iv = _write(str(tmp_path / "iverilog"))
    monkeypatch.setenv("ESIM_IVERILOG", iv)
    assert C.iverilog_binary() == iv


def test_iverilog_from_config(monkeypatch, tmp_path):
    iv = _write(str(tmp_path / "iverilog"))
    _use_config(monkeypatch, tmp_path, "[COSIM]\nIVERILOG = %s\n" % iv)
    assert C.iverilog_binary() == iv


def test_iverilog_none_when_absent(monkeypatch, tmp_path):
    _use_config(monkeypatch, tmp_path, "[COSIM]\n")
    monkeypatch.setattr(C.shutil, "which", lambda name: None)
    assert C.iverilog_binary() is None


def test_libdir_derived_from_binary(monkeypatch, tmp_path):
    prefix = tmp_path / "iv"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "lib").mkdir()
    iv = _write(str(prefix / "bin" / "iverilog"))
    monkeypatch.setenv("ESIM_IVERILOG", iv)
    _use_config(monkeypatch, tmp_path, "")
    assert C.iverilog_libdir() == str(prefix / "lib")


# -- capability gates ----------------------------------------------------------
def test_has_iverilog_requires_libvvp(monkeypatch, tmp_path):
    prefix = tmp_path / "iv"
    (prefix / "bin").mkdir(parents=True)
    lib = prefix / "lib"
    lib.mkdir()
    iv = _write(str(prefix / "bin" / "iverilog"))
    monkeypatch.setenv("ESIM_IVERILOG", iv)
    _use_config(monkeypatch, tmp_path, "")

    assert C.has_iverilog() is False        # no libvvp yet
    _write(str(lib / "libvvp.so"))
    assert C.has_iverilog() is True


def test_cosim_vvp_path_from_digital_model(monkeypatch, tmp_path):
    _use_config(monkeypatch, tmp_path,
                "[NGHDL]\nDIGITAL_MODEL = /opt/icm\n")
    assert C.cosim_vvp_path("adder") == os.path.join(
        "/opt/icm", "Ngveri", "adder", "adder")


def test_cosim_vvp_path_none_without_config(monkeypatch, tmp_path):
    _use_config(monkeypatch, tmp_path, "[NGHDL]\n")
    assert C.cosim_vvp_path("adder") is None


def test_loader_path_var_posix():
    assert C.loader_path_var() == "LD_LIBRARY_PATH"


def test_missing_reason_no_iverilog(monkeypatch, tmp_path):
    _use_config(monkeypatch, tmp_path, "")
    monkeypatch.setattr(C.shutil, "which", lambda name: None)
    assert "iverilog not found" in C.missing_reason()
