# ==============================================================================
#  test_cosim_lifecycle.py -- unit tests for the d_cosim (Icarus) model
#  management fixes: backend-correct deletion, KiCad-symbol-library integrity
#  across repeated add/overwrite/delete, and the canonical (lowercased) name
#  contract that keeps the compiled vvp findable on case-sensitive filesystems.
#
#  All hermetic: they drive the on-disk symbol/XML/path logic with temp files
#  and manually-set attributes, so no Qt event loop, ngspice, or iverilog is
#  needed. CosimSchematic objects are built WITHOUT init() (which would read the
#  real eSim/KiCad install) -- only the three attributes the methods under test
#  touch are set.
# ==============================================================================
import os
import sys

# Package-relative imports (`from . import createkicad`) require importing as
# part of the `maker` package; put `src` on the path like eSim's pathmagic.
SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Headless Qt: createkicad / createkicadCosim pull in PyQt6 at import time.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from maker import createkicad, createkicadCosim   # noqa: E402
from maker import CosimConfig as C                 # noqa: E402


def _block(name):
    """A minimal but valid, balanced KiCad part block matching _PART_RE."""
    return '(symbol "%s" (pin_names (offset 1.016)) (in_bom yes))' % name


def _cosim(lib, xml_root, modelname=None):
    """A CosimSchematic with only the attributes the methods under test use,
    skipping init() so no real eSim/KiCad install is required."""
    obj = createkicadCosim.CosimSchematic()
    obj.kicad_ngveri_sym = str(lib)
    obj.xml_loc = str(xml_root)
    if modelname is not None:
        obj.modelname = modelname
    return obj


def _balanced(path):
    """True if the whole library file is a single balanced s-expression."""
    content = path.read_text()
    return createkicad._balanced_end(content, 0) == len(content.rstrip())


# --------------------------------------------------------------------------- #
#  Deletion targets the d_cosim backend ONLY
# --------------------------------------------------------------------------- #
def test_cosim_delete_removes_symbol_and_cosim_xml_only(tmp_path):
    xml_root = tmp_path / "modelParamXML"
    (xml_root / "NgVeriCosim").mkdir(parents=True)
    (xml_root / "Ngveri").mkdir(parents=True)
    cosim_xml = xml_root / "NgVeriCosim" / "foo.xml"
    cosim_xml.write_text("<model/>")
    # A legacy NgVeri model that happens to share the name must NOT be touched.
    legacy_xml = xml_root / "Ngveri" / "foo.xml"
    legacy_xml.write_text("<model/>")

    lib = tmp_path / "eSim_NgVeriCosim.kicad_sym"
    createkicad._write_lib(str(lib), {"foo": _block("foo"), "bar": _block("bar")})

    _cosim(lib, xml_root, "foo").deleteKicadSymbol()

    assert set(createkicad._read_parts(str(lib))) == {"bar"}  # foo symbol gone
    assert not cosim_xml.exists()                             # cosim xml gone
    assert legacy_xml.exists()                                # legacy untouched
    assert _balanced(lib)


def test_cosim_delete_is_idempotent(tmp_path):
    xml_root = tmp_path / "x"
    (xml_root / "NgVeriCosim").mkdir(parents=True)
    lib = tmp_path / "lib.kicad_sym"
    createkicad._write_lib(str(lib), {"foo": _block("foo")})

    obj = _cosim(lib, xml_root, "foo")
    obj.deleteKicadSymbol()
    obj.deleteKicadSymbol()      # second pass: no crash, nothing left behind

    assert createkicad._read_parts(str(lib)) == {}
    assert _balanced(lib)


# --------------------------------------------------------------------------- #
#  Symbol-library integrity across repeated add / overwrite / delete
# --------------------------------------------------------------------------- #
def test_cosim_lib_integrity_through_add_overwrite_delete(tmp_path):
    xml_root = tmp_path / "x"
    (xml_root / "NgVeriCosim").mkdir(parents=True)
    lib = tmp_path / "eSim_NgVeriCosim.kicad_sym"
    obj = _cosim(lib, xml_root)

    def commit(name):
        obj.modelname = name
        obj._commit_block(_block(name))

    def delete(name):
        obj.modelname = name
        obj.deleteKicadSymbol()

    commit("foo"); commit("foo"); commit("foo")   # repeated overwrite
    commit("bar")
    delete("foo")
    commit("foo")                                  # re-add after delete

    content = lib.read_text()
    # Exactly one block per name (no duplicates piling up), no glued blocks,
    # file always a valid balanced s-expression.
    assert content.count('(symbol "foo" (pin_names') == 1
    assert content.count('(symbol "bar" (pin_names') == 1
    assert set(createkicad._read_parts(str(lib))) == {"foo", "bar"}
    assert _balanced(lib)


def test_cosim_commit_rejects_unbalanced_block(tmp_path):
    xml_root = tmp_path / "x"
    (xml_root / "NgVeriCosim").mkdir(parents=True)
    lib = tmp_path / "lib.kicad_sym"
    obj = _cosim(lib, xml_root, "foo")
    try:
        obj._commit_block('(symbol "foo" (pin_names')   # truncated -> rejected
        assert False, "expected ValueError on malformed block"
    except ValueError:
        pass
    # A rejected write must not have created/poisoned the shared library.
    assert createkicad._read_parts(str(lib)) == {}


# --------------------------------------------------------------------------- #
#  Canonical-name contract: build and netlister must converge on one vvp path
# --------------------------------------------------------------------------- #
def test_cosim_vvp_path_case_contract(monkeypatch, tmp_path):
    cfg = tmp_path / "config.ini"
    cfg.write_text("[NGHDL]\nDIGITAL_MODEL = %s\n" % tmp_path)
    monkeypatch.setattr(C, "_config_path", lambda: str(cfg))

    # The path IS case-sensitive: a mixed-case name and its lowercase form map
    # to different files -- this is exactly why the build (which used the raw
    # filename) and the netlister (which uses the lowercased symbol value) used
    # to disagree and lose the vvp at simulation time.
    assert C.cosim_vvp_path("Adder") != C.cosim_vvp_path("adder")

    # The fix: both sides pass basename.lower(), so they converge on one path.
    assert C.cosim_vvp_path("Adder".lower()) == C.cosim_vvp_path("adder")
