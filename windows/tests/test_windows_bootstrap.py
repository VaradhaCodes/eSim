"""Tests for windows/windows_bootstrap.py.

The bootstrap is deliberately OS-independent stdlib code so these tests run
on Linux CI even though only Windows ships the script. HOME is redirected to
tmp_path throughout -- the real ~/.esim and ~/.config are never touched.
"""

import configparser
import os
import shutil
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))

import windows_bootstrap as wb  # noqa: E402


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    h.mkdir()
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: p.replace("~", str(h), 1))
    # On Windows _kicad_config_root() resolves via APPDATA; without this the
    # KiCad-table tests read -- and WRITE -- the developer's real
    # %APPDATA%\kicad. Redirect it so every test is hermetic on both OSes.
    appdata = h / "AppData"
    appdata.mkdir()
    monkeypatch.setenv("APPDATA", str(appdata))
    return h


@pytest.fixture
def root(tmp_path):
    """A minimal fake eSim install root."""
    r = tmp_path / "eSim"
    sym = r / "library" / "kicadLibrary" / "eSim-symbols"
    sym.mkdir(parents=True)
    for lib in ("eSim_Devices", "eSim_Ngveri", "eSim_NgVeriCosim",
                "eSim_Nghdl"):
        (sym / (lib + ".kicad_sym")).write_text(
            '(kicad_symbol_lib (version 20211014) '
            '(generator kicad_symbol_editor)\n)\n')
    # Real helper module, so registration logic is the shipped one.
    maker = r / "src" / "maker"
    maker.mkdir(parents=True)
    shutil.copy2(os.path.join(REPO, "src", "maker", "kicad_symlib.py"), maker)
    return r


def test_config_written_and_home_rewritten(home, root):
    path = wb.write_esim_config(str(root))
    cfg = configparser.ConfigParser()
    cfg.read(path)
    assert cfg.get("eSim", "eSim_HOME") == str(root)
    # A moved install self-heals on next launch.
    moved = str(root) + "-moved"
    cfg2 = configparser.ConfigParser()
    cfg2.read(wb.write_esim_config(moved))
    assert cfg2.get("eSim", "eSim_HOME") == moved


def test_seed_never_clobbers(home, root):
    dst = wb.seed_generated_symbols(str(root))
    seeded = os.path.join(dst, "eSim_Ngveri.kicad_sym")
    assert os.path.isfile(seeded)
    # Static libs are NOT copied to the per-user dir.
    assert not os.path.exists(os.path.join(dst, "eSim_Devices.kicad_sym"))
    with open(seeded, "w") as fh:
        fh.write("USER MODELS")
    wb.seed_generated_symbols(str(root))
    assert open(seeded).read() == "USER MODELS"


def test_nghdl_config_needs_some_toolchain(home, root):
    """No tools/nghdl and no tools/msys64 -> nothing to describe."""
    assert wb.write_nghdl_config(str(root)) is None


def test_nghdl_config_msys_only(home, root):
    """Compact-ish install: MSYS2 present, no built nghdl tree -> only the
    COMPILER section is written (no keys pointing at missing dirs)."""
    (root / "tools" / "msys64").mkdir(parents=True)
    path = wb.write_nghdl_config(str(root))
    cfg = configparser.ConfigParser()
    cfg.read(path)
    assert cfg.get("COMPILER", "MSYS_HOME") == str(root / "tools" / "msys64")
    assert not cfg.has_section("NGHDL")


def test_nghdl_config_full_install(home, root):
    """Full install: every key the nghdl python + CosimConfig read, with
    absolute paths mirroring the Ubuntu installer's config."""
    nghdl = root / "tools" / "nghdl"
    nghdl.mkdir(parents=True)
    (root / "tools" / "msys64").mkdir(parents=True)
    ivbin = root / "library" / "bin" / "iverilog" / "bin"
    ivbin.mkdir(parents=True)
    (ivbin / "iverilog.exe").write_text("")
    (ivbin / "vvp.exe").write_text("")
    (ivbin / "libvvp-2.dll").write_text("")   # mingw: DLLs land in bin/

    path = wb.write_nghdl_config(str(root))
    cfg = configparser.ConfigParser()
    cfg.read(path)
    assert cfg.get("NGHDL", "NGHDL_HOME") == str(nghdl)
    assert cfg.get("NGHDL", "DIGITAL_MODEL") == str(
        nghdl / "src" / "xspice" / "icm")
    assert cfg.get("NGHDL", "RELEASE") == str(nghdl / "release")
    assert cfg.get("SRC", "SRC_HOME") == str(root / "nghdl")
    assert cfg.get("SRC", "LICENSE") == str(root / "nghdl" / "LICENSE")
    assert cfg.get("COMPILER", "MSYS_HOME") == str(root / "tools" / "msys64")
    assert cfg.get("COSIM", "IVERILOG") == str(ivbin / "iverilog.exe")
    assert cfg.get("COSIM", "IVERILOG_LIB") == str(ivbin)


def test_fix_spinit_rewrites_codemodel_paths(home, root):
    """Build-machine absolute codemodel paths get repointed at THIS install,
    idempotently."""
    inst = root / "tools" / "nghdl" / "install_dir"
    scripts = inst / "share" / "ngspice" / "scripts"
    scripts.mkdir(parents=True)
    cmdir = inst / "lib" / "ngspice"
    cmdir.mkdir(parents=True)
    (scripts / "spinit").write_text(
        "* Standard ngspice init file\n"
        "codemodel C:/buildbot/stage/eSim/tools/nghdl/install_dir/lib/"
        "ngspice/analog.cm\n"
        "codemodel C:/buildbot/stage/eSim/tools/nghdl/install_dir/lib/"
        "ngspice/ghdl.cm\n"
        "set num_threads=2\n")
    changed = wb.fix_spinit(str(root))
    assert changed == 2
    text = (scripts / "spinit").read_text()
    want = str(cmdir).replace("\\", "/") + "/analog.cm"
    assert "codemodel " + want in text
    assert "buildbot" not in text
    assert "set num_threads=2" in text          # untouched lines survive
    # Second run: nothing left to rewrite.
    assert wb.fix_spinit(str(root)) == 0


def test_fix_spinit_absent_tree_is_noop(home, root):
    assert wb.fix_spinit(str(root)) == 0


def test_fix_spinit_rewrite_is_atomic(home, root, monkeypatch):
    """spinit is how ngspice locates EVERY .cm code model, so a rewrite that
    dies part-way leaves a truncated file and then every simulation fails with
    "codemodel not found" -- nowhere near this code. The rewrite goes through
    kicad_symlib._atomic_write (temp file + os.replace), so a failure must
    leave the previous spinit byte-for-byte intact and drop no scratch file."""
    inst = root / "tools" / "nghdl" / "install_dir"
    scripts = inst / "share" / "ngspice" / "scripts"
    scripts.mkdir(parents=True)
    (inst / "lib" / "ngspice").mkdir(parents=True)
    original = ("* Standard ngspice init file\n"
                "codemodel C:/buildbot/stage/lib/ngspice/analog.cm\n")
    (scripts / "spinit").write_text(original)

    def boom(src, dst):
        raise OSError("simulated crash before rename")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        wb.fix_spinit(str(root))

    assert (scripts / "spinit").read_text() == original
    assert [n for n in os.listdir(str(scripts))
            if n.startswith(".eSim_atomic_")] == []


def test_register_appends_and_repoints(home, root, kicad_config_root):
    # A KiCad user table with one stale eSim entry.
    kdir = kicad_config_root / "9.0"
    kdir.mkdir(parents=True)
    table = kdir / "sym-lib-table"
    table.write_text(
        '(sym_lib_table\n'
        '  (lib (name "eSim_Ngveri")(type "KiCad")'
        '(uri "${KICAD6_SYMBOL_DIR}/eSim_Ngveri.kicad_sym")'
        '(options "")(descr ""))\n'
        ')\n')
    wb.seed_generated_symbols(str(root))
    assert wb.register_kicad_libraries(str(root)) is True
    content = table.read_text()
    # Stale generated-lib uri repointed to ~/.esim/kicad_symbols.
    assert "${KICAD6_SYMBOL_DIR}/eSim_Ngveri" not in content
    assert os.path.join(".esim", "kicad_symbols",
                        "eSim_Ngveri.kicad_sym") in content
    # Static lib appended, referenced in place from the install dir.
    assert "eSim_Devices" in content
    assert str(root) in content
    # Idempotent: second run changes nothing.
    wb.register_kicad_libraries(str(root))
    assert table.read_text() == content


def test_register_no_kicad_config_is_noop(home, root):
    assert wb.register_kicad_libraries(str(root)) is False


@pytest.fixture
def kicad_config_root(home):
    """Where _kicad_config_root() resolves under the hermetic `home` fixture:
    APPDATA drives it on Windows, HOME/.config elsewhere."""
    if os.name == "nt":
        return home / "AppData" / "kicad"
    return home / ".config" / "kicad"


def test_kicad_config_dir_from_bundled_stamp(home, root, kicad_config_root):
    # No bundled KiCad (dev tree): no-op.
    assert wb.ensure_kicad_config_dir(str(root)) is None
    # Stamped bundled KiCad: %APPDATA%/kicad/<major.minor> gets created...
    kdir = root / "tools" / "kicad"
    kdir.mkdir(parents=True)
    (kdir / "KICAD-VERSION").write_text("9.0.3\n")
    made = wb.ensure_kicad_config_dir(str(root))
    assert made == str(kicad_config_root / "9.0")
    assert os.path.isdir(made)
    # ...so first-launch registration now has a version dir to seed.
    wb.seed_generated_symbols(str(root))
    assert wb.register_kicad_libraries(str(root)) is True
    table = kicad_config_root / "9.0" / "sym-lib-table"
    assert "eSim_Devices" in table.read_text()
    # Garbage stamp: no-op, never a crash at launch.
    (kdir / "KICAD-VERSION").write_text("not-a-version")
    assert wb.ensure_kicad_config_dir(str(root)) is None


def test_main_runs_everything(home, root):
    assert wb.main(["--esim-root", str(root)]) == 0
    assert (home / ".esim" / "config.ini").is_file()
    assert (home / ".esim" / "kicad_symbols" /
            "eSim_Nghdl.kicad_sym").is_file()
