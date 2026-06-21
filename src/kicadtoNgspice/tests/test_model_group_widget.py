# ==============================================================================
#  test_model_group_widget.py -- behaviour tests for ModelGroupWidget.
#
#  Runs headless via the Qt 'offscreen' platform (no display needed). These
#  exercise the fan-out / inherit / override / reset / derive-on-reload logic
#  through the public API.
#
#  Run:  QT_QPA_PLATFORM=offscreen python3 \
#            src/kicadtoNgspice/tests/test_model_group_widget.py
#        (pytest also works; the platform is forced below.)
# ==============================================================================
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)                       # .../src/kicadtoNgspice
for _p in (HERE, PKG):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from PyQt6 import QtWidgets                        # noqa: E402
from ModelGroupWidget import ModelGroupWidget, InstanceRow  # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)


def _rows(refs, texts=None):
    rows = []
    for i, ref in enumerate(refs):
        edit = QtWidgets.QLineEdit()
        if texts:
            edit.setText(texts[i])
        rows.append(InstanceRow(ref, edit))
    return rows


def _widget(refs, texts=None, log=None):
    resolve = (lambda ref, path: log.append((ref, path))) if log is not None \
        else None
    return ModelGroupWidget("eSim_NPN (transistor)", _rows(refs, texts),
                            resolve_fn=resolve)


# -- fan-out -------------------------------------------------------------------

def test_group_path_fans_out_to_all_instances():
    log = []
    w = _widget(["q1", "q2", "q3"], log=log)
    w.set_group_path("/lib/bc547.lib")
    assert w.resolved() == {
        "q1": "/lib/bc547.lib",
        "q2": "/lib/bc547.lib",
        "q3": "/lib/bc547.lib",
    }
    # resolve_fn fired once per instance with the group path.
    assert set(log) == {("q1", "/lib/bc547.lib"),
                        ("q2", "/lib/bc547.lib"),
                        ("q3", "/lib/bc547.lib")}


def test_override_then_group_change_leaves_override_alone():
    w = _widget(["q1", "q2", "q3"])
    w.set_group_path("/lib/a.lib")
    w.override("q2", "/lib/special.lib")
    assert w.is_overridden("q2")
    # Change the group again: q1/q3 follow, q2 stays put.
    w.set_group_path("/lib/b.lib")
    assert w.resolved() == {
        "q1": "/lib/b.lib",
        "q2": "/lib/special.lib",
        "q3": "/lib/b.lib",
    }


def test_reset_reattaches_to_group():
    w = _widget(["q1", "q2"])
    w.set_group_path("/lib/a.lib")
    w.override("q1", "/lib/x.lib")
    w.reset_row_by_ref("q1")
    assert not w.is_overridden("q1")
    assert w.resolved()["q1"] == "/lib/a.lib"


def test_user_keystroke_marks_override():
    # textEdited is what the GUI emits on typing; simulate it directly.
    w = _widget(["q1", "q2"])
    w.set_group_path("/lib/a.lib")
    row1 = w._row("q1")
    row1.path_edit.setText("/typed/by/hand.lib")
    row1.path_edit.textEdited.emit("/typed/by/hand.lib")
    assert w.is_overridden("q1")
    # Group change no longer touches the hand-edited row.
    w.set_group_path("/lib/c.lib")
    assert w.resolved()["q1"] == "/typed/by/hand.lib"
    assert w.resolved()["q2"] == "/lib/c.lib"


def test_programmatic_settext_does_not_mark_override():
    # Fan-out uses setText(); that must NOT look like a manual override.
    w = _widget(["q1", "q2"])
    w.set_group_path("/lib/a.lib")
    assert not w.is_overridden("q1")
    assert not w.is_overridden("q2")


# -- derive initial state from restored values --------------------------------

def test_uniform_restored_values_become_group_default():
    w = _widget(["q1", "q2", "q3"],
                texts=["/lib/x.lib", "/lib/x.lib", "/lib/x.lib"])
    assert w.group_path() == "/lib/x.lib"
    assert not any(w.is_overridden(r) for r in ("q1", "q2", "q3"))
    assert not w.is_expanded()


def test_mixed_restored_values_show_overrides_and_expand():
    w = _widget(["q1", "q2", "q3"],
                texts=["/lib/x.lib", "/lib/y.lib", ""])
    assert w.group_path() == ""
    assert w.is_overridden("q1")
    assert w.is_overridden("q2")
    assert not w.is_overridden("q3")     # blank -> still inheriting
    assert w.is_expanded()               # user must see the conflict


def test_all_blank_restored_is_clean_inherit():
    w = _widget(["q1", "q2"], texts=["", ""])
    assert w.group_path() == ""
    assert not w.is_overridden("q1")
    assert not w.is_expanded()


def test_expand_collapse_toggle():
    w = _widget(["q1", "q2"])
    assert not w.is_expanded()
    w.set_expanded(True)
    assert w.is_expanded()
    w.set_expanded(False)
    assert not w.is_expanded()


# -- standalone runner ---------------------------------------------------------
def _main():
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
            print("ERROR " + fn.__name__ + "  " + repr(e))
    print("\n==== %d / %d PASS ====" % (len(fns) - failed, len(fns)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
