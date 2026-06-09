# ==============================================================================
#  test_netlister_golden.py -- regression suite for the KiCad-9 netlister.
#
#  For every fixture under tests/golden/<example>/ that has both a KiCad-9
#  <name>.kicad_sch and the legacy ground-truth <name>.cir, this:
#     1. copies ONLY the .kicad_sch to a scratch dir (the netlister OVERWRITES
#        <name>.cir, so it must never run next to the golden .cir),
#     2. runs KicadNetlister.generate_netlist on the copy,
#     3. asserts the result is topologically equivalent to the golden .cir
#        (see netlist_compare for the equivalence definition).
#
#  Run with pytest:        pytest src/kicadtoNgspice/tests/
#  Or standalone (no deps): python3 src/kicadtoNgspice/tests/test_netlister_golden.py
#
#  Requires kicad-cli (KiCad >= 7) on PATH, or $ESIM_KICAD_CLI. Skips if absent.
# ==============================================================================
import os
import sys
import shutil
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)                       # .../src/kicadtoNgspice
for _p in (HERE, PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import KicadNetlister                              # noqa: E402
from netlist_compare import compare                # noqa: E402

GOLDEN_ROOT = os.environ.get("ESIM_NETLIST_GOLDENS",
                             os.path.join(HERE, "golden"))


def discover():
    """Yield (test_id, sch_path, cir_path) for every golden example."""
    cases = []
    for root, _dirs, files in os.walk(GOLDEN_ROOT):
        for f in files:
            if not f.endswith(".kicad_sch"):
                continue
            name = f[:-len(".kicad_sch")]
            cir = os.path.join(root, name + ".cir")
            if os.path.isfile(cir):
                tid = os.path.relpath(os.path.join(root, name), GOLDEN_ROOT)
                cases.append((tid, os.path.join(root, f), cir))
    return sorted(cases)


def run_one(sch_path, cir_path):
    """Generate from sch_path in a scratch dir and compare to the golden cir.
    Returns (equivalent: bool, detail: str)."""
    name = os.path.basename(sch_path)[:-len(".kicad_sch")]
    tmp = tempfile.mkdtemp(prefix="esim_netlist_")
    try:
        shutil.copy(sch_path, os.path.join(tmp, name + ".kicad_sch"))
        ok, msg = KicadNetlister.generate_netlist(tmp, name)
        if not ok:
            return False, "generation failed: " + msg
        generated = open(os.path.join(tmp, name + ".cir")).read()
        legacy = open(cir_path).read()
        result = compare(generated, legacy)
        if result["equivalent"]:
            return True, "ok"
        return False, ("topology mismatch: %s" % result)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# -- pytest entry point (parametrized; one case per example) -------------------
try:
    import pytest

    _CASES = discover()
    _kicad = KicadNetlister._kicad_cli()

    @pytest.mark.skipif(_kicad is None, reason="kicad-cli not found")
    @pytest.mark.skipif(not _CASES, reason="no golden fixtures found")
    @pytest.mark.parametrize(
        "sch_path,cir_path",
        [(c[1], c[2]) for c in _CASES],
        ids=[c[0] for c in _CASES],
    )
    def test_netlist_matches_golden(sch_path, cir_path):
        ok, detail = run_one(sch_path, cir_path)
        assert ok, detail

except ImportError:
    pass


# -- standalone runner (no pytest required) ------------------------------------
def main():
    if KicadNetlister._kicad_cli() is None:
        print("SKIP: kicad-cli not found (set $ESIM_KICAD_CLI).")
        return 0
    cases = discover()
    if not cases:
        print("SKIP: no golden fixtures under " + GOLDEN_ROOT)
        return 0
    npass = 0
    fails = []
    for tid, sch, cir in cases:
        ok, detail = run_one(sch, cir)
        if ok:
            npass += 1
            print("PASS  " + tid)
        else:
            fails.append((tid, detail))
            print("FAIL  " + tid + "  " + detail)
    print("\n==== %d / %d PASS ====" % (npass, len(cases)))
    for tid, detail in fails:
        print("  FAIL " + tid)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
