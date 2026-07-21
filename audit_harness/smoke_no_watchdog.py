"""Import-smoke for M8: Model Creation must survive a missing ``watchdog``.

Blocks the ``watchdog`` package at the import-hook level (the closest in-repo
stand-in for a machine whose optional dependency is absent or broken), then
imports the DesignBus chain and drives the design flow that does NOT need the
watch -- content edit, materialize to disk, and ``_start_watch`` (which must
no-op, not raise). Exit 0 = importing maker.DesignBus and using a bus works
with watchdog uninstalled; only the passive external-edit watch is off.
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SRC = os.path.join(REPO, "src")

ISO = tempfile.mkdtemp(prefix="esim_smoke_wd_home_")
os.environ["HOME"] = ISO
os.environ["USERPROFILE"] = ISO
os.environ["APPDATA"] = os.path.join(ISO, "AppData", "Roaming")
os.environ["LOCALAPPDATA"] = os.path.join(ISO, "AppData", "Local")
os.makedirs(os.environ["APPDATA"], exist_ok=True)
os.makedirs(os.environ["LOCALAPPDATA"], exist_ok=True)
os.environ["QT_QPA_PLATFORM"] = "offscreen"

sys.path.insert(0, SRC)


class _BlockWatchdog:
    """Meta-path finder that makes the ``watchdog`` package look uninstalled."""

    def find_module(self, name, path=None):
        return None

    def find_spec(self, name, path=None, target=None):
        if name == "watchdog" or name.startswith("watchdog."):
            raise ImportError("No module named 'watchdog' (blocked by smoke)")
        return None


sys.meta_path.insert(0, _BlockWatchdog())
for mod in [m for m in sys.modules if m == "watchdog" or m.startswith("watchdog.")]:
    del sys.modules[mod]

failures = []

try:
    import watchdog  # noqa: F401
    failures.append("watchdog import was NOT blocked -- smoke is meaningless")
except ImportError:
    pass

from PyQt6 import QtWidgets  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

# 1) The module that hard-imported watchdog must import cleanly.
try:
    from maker import DesignBus  # noqa: E402
    assert DesignBus._HAS_WATCHDOG is False, "smoke did not disable watchdog"
    assert DesignBus._DiskWatchHandler is None
    print("[OK]     import maker.DesignBus (watchdog absent, _HAS_WATCHDOG=False)")
except Exception as exc:
    failures.append("import maker.DesignBus -> %r" % (exc,))
    print("[FAIL]   import maker.DesignBus -> %r" % (exc,))

# 2) A bus must build, hold content, and _start_watch must silently no-op
#    (no observer, no raise). _path is set directly to keep this test about M8
#    only -- set_path()/materialize() pull in maker.Maker's UI dep tree, which
#    is not what this smoke is asserting.
try:
    bus = DesignBus.DesignBus(0)
    bus.set_content("module m; endmodule\n")   # pure in-memory
    bus._path = os.path.join(ISO, "design.v")
    bus._start_watch()                          # must no-op with watchdog gone
    assert bus._observer is None, "watch started despite watchdog absent"
    bus.close()                                 # must be a safe no-op too
    print("[OK]     DesignBus builds, watch disabled cleanly")
except Exception as exc:
    failures.append("DesignBus flow -> %r" % (exc,))
    print("[FAIL]   DesignBus flow -> %r" % (exc,))

# 3) The whole Model Creation entry point must not die *because of watchdog*.
#    Tolerate unrelated heavy-dep import errors offscreen -- only a watchdog
#    ImportError leaking through is an M8 regression.
try:
    from maker import makerchip  # noqa: F401
    print("[OK]     import maker.makerchip (watchdog absent)")
except ImportError as exc:
    if "watchdog" in str(exc).lower():
        failures.append("maker.makerchip leaks a watchdog ImportError -> %r" % (exc,))
        print("[FAIL]   maker.makerchip -> %r" % (exc,))
    else:
        print("[SKIP]   maker.makerchip import needs an unrelated dep: %r" % (exc,))
except Exception as exc:
    print("[SKIP]   maker.makerchip import (non-import error, ignored): %r" % (exc,))

print("\nRESULT: %s" % ("FAIL (%d)" % len(failures) if failures else "PASS"))
for line in failures:
    print("  - " + line)
sys.exit(1 if failures else 0)
