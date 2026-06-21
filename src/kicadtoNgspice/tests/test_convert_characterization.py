# ==============================================================================
#  test_convert_characterization.py -- the netlist guard for the grouping work.
#
#  Grouping only changes HOW the per-instance deviceModelTrack / subcircuitTrack
#  dicts get filled; Convert (which turns those dicts into netlist lines and
#  .include directives) is deliberately untouched. This pins Convert's output
#  for a known per-ref track, so if a future change to the grouping layer ever
#  feeds Convert something different, the emitted netlist change is caught here.
#
#  It exercises addSubcircuit end-to-end (line rewrite + .include + file copy),
#  the path most directly downstream of the grouped fan-out.
# ==============================================================================
import os
import sys
import shutil
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
SRC = os.path.dirname(PKG)
for _p in (SRC, PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from kicadtoNgspice import Convert, TrackWidget               # noqa: E402


def _sub_dir(name, nports):
    d = tempfile.mkdtemp(prefix="esim_char_")
    sub = os.path.join(d, name)
    os.makedirs(sub)
    ports = " ".join(str(i) for i in range(nports))
    with open(os.path.join(sub, name + ".sub"), "w") as f:
        f.write(".subckt %s %s\n.ends %s\n" % (name, ports, name))
    return sub


def _convert(schematic, kicad_file):
    conv = Convert.Convert(None, None, list(schematic), kicad_file)
    conv.obj_track = TrackWidget.TrackWidget()
    return conv


def test_grouped_subcircuit_fanout_produces_one_include_and_rewrites_lines():
    TrackWidget.TrackWidget.reset()
    sub = _sub_dir("myamp", 3)                       # dir basename != model tok
    proj = tempfile.mkdtemp(prefix="esim_proj_")
    kicad_file = os.path.join(proj, "proj.cir")
    try:
        schematic = ["x1 a b c lm_741", "x2 d e f lm_741"]
        # What the grouped tab fans out: same dir for both instances.
        TrackWidget.TrackWidget.subcircuitTrack = {"x1": sub, "x2": sub}
        # subcircuitList must match in length (Convert's "all specified" guard).
        TrackWidget.TrackWidget.subcircuitList = {
            "projx1": schematic[0].split(),
            "projx2": schematic[1].split(),
        }

        conv = _convert(schematic, kicad_file)
        out = conv.addSubcircuit(list(schematic), kicad_file)

        # Exactly one .include despite two instances (Convert dedups).
        includes = [ln for ln in out if ln.startswith(".include")]
        assert includes == [".include myamp.sub"], includes

        # Both instance lines had their model token rewritten to the dir name.
        body = [ln for ln in out if ln and not ln.startswith(".include")]
        assert body[0].split()[-1] == "myamp"
        assert body[1].split()[-1] == "myamp"
        assert body[0].split()[0] == "x1"
        assert body[1].split()[0] == "x2"

        # The subcircuit file was copied into the project dir.
        assert os.path.exists(os.path.join(proj, "myamp.sub"))
    finally:
        shutil.rmtree(sub, ignore_errors=True)
        shutil.rmtree(proj, ignore_errors=True)


# -- standalone runner ---------------------------------------------------------
def _main():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print("PASS  " + fn.__name__)
        except AssertionError as e:
            failed += 1
            print("FAIL  " + fn.__name__ + "  " + str(e))
        except Exception as e:                       # noqa: BLE001
            failed += 1
            import traceback
            print("ERROR " + fn.__name__ + "  " + repr(e))
            traceback.print_exc()
    print("\n==== %d / %d PASS ====" % (len(fns) - failed, len(fns)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
