"""Batch-3 fuzz: PrcocessNetlist parser guards (M3a/b/c + R2-4 h/f/transfo).

Feeds malformed netlist fragments straight into the parser methods and
asserts none raise. These crash sites were found by the Round-2 machine
sweep but are NOT reachable through fuzz_ktn (MainWindow swallows them in
_loadNetlist), so this harness exercises the methods directly.

Run:  python audit_harness/fuzz_processing.py
AUDIT / regression only — writes nothing into src/.
"""
import os
import sys
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SRC = os.path.join(REPO, "src")

# Isolate the user profile so importing configuration.paths never touches
# the real ~/.esim (R2-5 lesson), even though Processing itself is pure.
ISO = tempfile.mkdtemp(prefix="esim_fuzz_home_")
os.environ["HOME"] = ISO
os.environ["USERPROFILE"] = ISO
os.environ["APPDATA"] = os.path.join(ISO, "AppData", "Roaming")
os.environ["LOCALAPPDATA"] = os.path.join(ISO, "AppData", "Local")
os.makedirs(os.environ["APPDATA"], exist_ok=True)
os.makedirs(os.environ["LOCALAPPDATA"], exist_ok=True)
os.environ["QT_QPA_PLATFORM"] = "offscreen"

sys.path.insert(0, SRC)

from kicadtoNgspice.Processing import PrcocessNetlist  # noqa: E402

CASES = []


def scenario(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


@scenario("A. readParamInfo: '.param foo' (no '=') (M3a)")
def s_a():
    PrcocessNetlist().readParamInfo(['.param foo'])


@scenario("B. readParamInfo: bare '.param'")
def s_b():
    PrcocessNetlist().readParamInfo(['.param'])


@scenario("C. preprocessNetlist: leading '+' continuation first line (M3b)")
def s_c():
    PrcocessNetlist().preprocessNetlist(['+ 1 2 3', 'r1 1 0 1k'], {})


@scenario("D. preprocessNetlist: nothing but a stripped-empty netlist (M3b)")
def s_d():
    PrcocessNetlist().preprocessNetlist([], {})


@scenario("E. insertSpecialSourceParam: short h-source 'h1 1 2' (R2-4)")
def s_e():
    PrcocessNetlist().insertSpecialSourceParam(['h1 1 2'], [])


@scenario("F. insertSpecialSourceParam: short f-source 'f1 1 2' (R2-4)")
def s_f():
    PrcocessNetlist().insertSpecialSourceParam(['f1 1 2'], [])


@scenario("G. convertICintoBasicBlocks: short transfo 'u3 1 2 transfo' (R2-4)")
def s_g():
    PrcocessNetlist().convertICintoBasicBlocks(
        ['u3 1 2 transfo'], [], [], [])


@scenario("H. convertICintoBasicBlocks: 'u1 plot_v2' with no nodes (M3c)")
def s_h():
    PrcocessNetlist().convertICintoBasicBlocks(['u1 plot_v2'], [], [], [])


@scenario("I. convertICintoBasicBlocks: 'u1 n1 plot_i2' one node (M3c)")
def s_i():
    PrcocessNetlist().convertICintoBasicBlocks(['u1 n1 plot_i2'], [], [], [])


@scenario("J. single-node plot_v2 must NOT emit the type token as a node")
def s_j():
    plot_text = []
    PrcocessNetlist().convertICintoBasicBlocks(
        ['u1 n1 plot_v2'], [], [], plot_text)
    assert not any('plot_v2' in p for p in plot_text), \
        "bogus node name leaked: %r" % plot_text


def main():
    fails = 0
    for name, fn in CASES:
        try:
            fn()
            print("[PASS] %s" % name)
        except Exception:
            fails += 1
            tb = traceback.format_exc().strip().splitlines()
            site = next((ln.strip() for ln in reversed(tb)
                         if 'File "' in ln), "?")
            print("[RAISE] %s\n        %s\n        %s"
                  % (name, tb[-1], site))
    print("\n%d/%d scenarios raised" % (fails, len(CASES)))


if __name__ == "__main__":
    main()
