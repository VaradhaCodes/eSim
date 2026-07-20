"""Round 3 fuzz harness: KicadtoNgspice.MainWindow against fuzzed .cir and
mismatched *_Previous_Values.xml (audit item 1a — prove/deny M1 empirically).

Run:  python audit_harness/fuzz_ktn.py
Isolation: HOME/USERPROFILE/APPDATA redirected to a temp dir so the real
~/.esim and real KiCad config are never touched (R2-5 lesson).
AUDIT ONLY — writes nothing into src/.
"""
import os
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SRC = os.path.join(REPO, "src")

# ---- isolate user profile BEFORE any esim import touches ~/.esim ----------
ISO = tempfile.mkdtemp(prefix="esim_fuzz_home_")
os.environ["HOME"] = ISO
os.environ["USERPROFILE"] = ISO
os.environ["APPDATA"] = os.path.join(ISO, "AppData", "Roaming")
os.environ["LOCALAPPDATA"] = os.path.join(ISO, "AppData", "Local")
os.makedirs(os.environ["APPDATA"], exist_ok=True)
os.makedirs(os.environ["LOCALAPPDATA"], exist_ok=True)
os.environ["QT_QPA_PLATFORM"] = "offscreen"

sys.path.insert(0, SRC)

from PyQt6 import QtWidgets  # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

# ---- neuter every modal dialog (record instead of blocking) ---------------
from configuration import Dialogs  # noqa: E402

DIALOGS = []


def _rec(kind):
    def f(parent, title, text, *a, **k):
        DIALOGS.append((kind, str(title), str(text)[:200]))
        return None
    return f


class _StubErrMsg:
    def __init__(self):
        self._msg = ""

    def setModal(self, *_):
        pass

    def setWindowTitle(self, t):
        self._t = t

    def showMessage(self, m):
        DIALOGS.append(("errormsg", getattr(self, "_t", ""), str(m)[:200]))

    def exec(self):
        return 0


Dialogs.information = _rec("information")
Dialogs.critical = _rec("critical")
Dialogs.warning = _rec("warning")
Dialogs.question = _rec("question")
Dialogs.make_error_message = lambda parent=None: _StubErrMsg()

from kicadtoNgspice import KicadtoNgspice  # noqa: E402
from projManagement.projectPaths import previous_values_path  # noqa: E402

RESULTS = []


def scenario(name):
    def deco(fn):
        RESULTS.append((name, fn))
        return fn
    return deco


def make_proj(cir_body):
    d = tempfile.mkdtemp(prefix="fuzzproj_")
    cir = os.path.join(d, "proj.cir")
    with open(cir, "w") as f:
        f.write(cir_body)
    return d, cir


RC_NETLIST = """* Sheet Name: /root
r1  net-in net-out 1k
c1  net-out 0 1u
v1  net-in 0 dc
.end
"""

SINE_NETLIST = """* Sheet Name: /root
r1  net-in net-out 1k
v1  net-in 0 sine
.end
"""


def build_window(cir, clarg2=None):
    DIALOGS.clear()
    w = KicadtoNgspice.MainWindow(cir, clarg2)
    _app.processEvents()
    return w


def write_prev(cir, xml_body):
    p = previous_values_path(cir)
    with open(p, "w") as f:
        f.write(xml_body)
    return p


@scenario("A. baseline RC + dc source, no prevvalues, full convert")
def s_a():
    d, cir = make_proj(RC_NETLIST)
    w = build_window(cir)
    assert w._netlistValid, "netlist should parse"
    w.callConvert()
    assert os.path.exists(cir + ".out"), ".cir.out must be written"
    return "convert ok, %d dialogs %r" % (len(DIALOGS), DIALOGS[:2])


@scenario("B. prevvalues XML exists but has NO <source> child -> callConvert")
def s_b():
    d, cir = make_proj(RC_NETLIST)
    write_prev(cir, "<KicadtoNgspice><analysis /></KicadtoNgspice>")
    w = build_window(cir)
    w.callConvert()
    return "survived, dialogs=%r" % (DIALOGS[:3],)


@scenario("C. source-type drift: prev XML says v1=pulse(7 fields), live v1=dc")
def s_c():
    d, cir = make_proj(RC_NETLIST)
    # authentic-shape pulse node with 7 grand children under <source>
    fields = "".join(
        '<field%d name="f">%d</field%d>' % (i, i, i) for i in range(1, 8))
    write_prev(cir,
               "<KicadtoNgspice><source>"
               '<v1 name="Source type">dc%s</v1>'
               "</source></KicadtoNgspice>" % fields)
    # child.tag == 'v1' matches wordv; child.text must equal words[-1] ('dc')
    w = build_window(cir)
    w.callConvert()
    return "survived, dialogs=%r" % (DIALOGS[:3],)


@scenario("D. sky130 netlist + empty <scmode1/> prevvalues node -> tab build")
def s_d():
    d, cir = make_proj("""* sky130 sheet
sc1  net-in net-out sky130_fd_pr__nfet_01v8
.end
""")
    write_prev(cir,
               "<KicadtoNgspice><devicemodel><scmode1 />"
               "</devicemodel></KicadtoNgspice>")
    w = build_window(cir)
    return "survived construction, dialogs=%r" % (DIALOGS[:3],)


@scenario("E. malformed (truncated) prevvalues XML -> construction+convert")
def s_e():
    d, cir = make_proj(RC_NETLIST)
    write_prev(cir, "<KicadtoNgspice><analysis>")
    w = build_window(cir)
    w.callConvert()
    return "survived, dialogs=%r" % (DIALOGS[:2],)


@scenario("F. createSubFile on .cir.out whose .control lost .endc (M2)")
def s_f():
    d, cir = make_proj(RC_NETLIST)
    with open(cir + ".out", "w") as f:
        f.write("* out\nu1 a b port\nr1 a b 1k\n.control\nrun\n")  # no .endc
    w = build_window(cir)
    w.createSubFile(os.path.splitext(cir)[0])
    return "survived, dialogs=%r" % (DIALOGS[:2],)


@scenario("G. prevvalues with <source> but ZERO grandchildren for live sine")
def s_g():
    d, cir = make_proj(SINE_NETLIST)
    write_prev(cir,
               "<KicadtoNgspice><source>"
               '<v1 name="Source type">sine</v1>'
               "</source></KicadtoNgspice>")
    w = build_window(cir)
    w.callConvert()
    return "survived, dialogs=%r" % (DIALOGS[:3],)


@scenario("H. prevvalues valid <source>, but live netlist source RENAMED v9")
def s_h():
    d, cir = make_proj("""* Sheet
r1 a b 1k
v9 a 0 dc
.end
""")
    write_prev(cir,
               "<KicadtoNgspice><source>"
               '<v1 name="Source type">dc<field1 name="Value">5</field1></v1>'
               "</source></KicadtoNgspice>")
    w = build_window(cir)
    w.callConvert()
    return "survived, dialogs=%r" % (DIALOGS[:3],)


@scenario("I. empty .cir file")
def s_i():
    d, cir = make_proj("")
    w = build_window(cir)
    return "survived, valid=%s dialogs=%r" % (w._netlistValid, DIALOGS[:2])


@scenario("J. .cir deleted between window open and Convert click")
def s_j():
    d, cir = make_proj(RC_NETLIST)
    w = build_window(cir)
    os.remove(cir)
    w.callConvert()
    return "survived, dialogs=%r" % (DIALOGS[:3],)


def main():
    print("prevvalues base:", os.path.join(ISO, ".esim", "prevvalues"))
    failures = 0
    for name, fn in RESULTS:
        DIALOGS.clear()
        try:
            note = fn()
            print("[PASS] %s :: %s" % (name, note))
        except Exception:
            failures += 1
            tb = traceback.format_exc().strip().splitlines()
            site = next((ln.strip() for ln in reversed(tb)
                         if 'File "' in ln), "?")
            print("[RAISE] %s\n        %s\n        %s"
                  % (name, tb[-1], site))
    print("\n%d/%d scenarios raised" % (failures, len(RESULTS)))


if __name__ == "__main__":
    main()
