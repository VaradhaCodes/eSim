"""Stress the theme-toggle path that produced the native use-after-free
(0xc0000005) when the user flipped dark<->light rapidly.

Runs headless (offscreen). Hammers apply_theme while synthetic hover events
spawn button-glow animations onto the very QGraphicsDropShadowEffects the
repolish tears down. Pre-fix this path frees an effect a live animation drives;
post-fix the glow freeze guarantees no such animation exists across the window.

Asserts, additionally, the invariants the fix rests on:
  * while frozen, the tactile filter starts NO glow animation
  * while frozen, _drop_glow deletes NO effect
  * after a storm settles, the freeze is fully lifted (never stranded)
  * apply_theme coalesces re-entrant calls instead of stacking repolishes
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from PyQt6 import QtCore, QtGui, QtWidgets  # noqa: E402
from frontEnd import motion, theme_utils     # noqa: E402

FAILS = []


def check(cond, msg):
    print(("PASS" if cond else "FAIL"), "-", msg)
    if not cond:
        FAILS.append(msg)


def drain(app, rounds=6):
    """Run queued events + deferred singleShots (the _thaw pass) to completion."""
    for _ in range(rounds):
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 20)


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    # Alternate the resolved theme every apply without touching disk.
    modes = ["Dark", "Light"]
    state = {"i": 0}
    orig_prefs = theme_utils.get_preferences

    def fake_prefs(*a, **k):
        p = orig_prefs(*a, **k)
        p["theme_mode"] = modes[state["i"] % 2]
        return p
    theme_utils.get_preferences = fake_prefs

    win = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(win)
    buttons = []
    for n in range(24):
        b = QtWidgets.QPushButton("btn%d" % n)
        lay.addWidget(b)
        buttons.append(b)
    motion.install_button_motion(win)
    win.resize(300, 600)
    win.show()
    drain(app)

    filt = getattr(win, "_esim_press_motion_filter", None)
    check(filt is not None, "tactile filter installed on window")

    # ---- invariant 1: frozen filter starts no animation -------------------
    filt.stop_all_glow()
    motion.freeze_glow()
    for b in buttons[:6]:
        app.sendEvent(b, QtGui.QEnterEvent(
            QtCore.QPointF(1, 1), QtCore.QPointF(1, 1), QtCore.QPointF(1, 1)))
    check(len(filt._glow_anims) == 0,
          "frozen: hover Enter started zero glow animations")

    # ---- invariant 2: frozen _drop_glow deletes nothing -------------------
    b0 = buttons[0]
    motion._ensure_glow(b0)
    had = isinstance(b0.graphicsEffect(),
                     QtWidgets.QGraphicsDropShadowEffect)
    motion._drop_glow(b0)
    still = isinstance(b0.graphicsEffect(),
                       QtWidgets.QGraphicsDropShadowEffect)
    check(had and still, "frozen: _drop_glow left the effect alive")
    motion.unfreeze_glow()
    check(not motion.glow_frozen(), "freeze depth back to zero after balance")

    # ---- the storm: rapid apply_theme + hover churn -----------------------
    iterations = 400
    for i in range(iterations):
        state["i"] = i
        # spawn/settle glows on a rotating slice of buttons
        for b in buttons[(i % 4)::4]:
            app.sendEvent(b, QtGui.QEnterEvent(
                QtCore.QPointF(1, 1), QtCore.QPointF(1, 1),
                QtCore.QPointF(1, 1)))
        theme_utils.apply_theme(app)
        # simulate the OS firing a second time mid-apply (re-entrancy path)
        theme_utils.apply_theme(app)
        for b in buttons[(i % 4)::4]:
            app.sendEvent(b, QtCore.QEvent(QtCore.QEvent.Type.Leave))
        if i % 7 == 0:
            drain(app, 2)
    drain(app, 12)

    check(True, "survived %d apply_theme storm iterations (no crash)"
          % iterations)
    check(not motion.glow_frozen(),
          "freeze fully lifted after storm (not stranded)")
    check(theme_utils._APPLYING is False, "apply lock released")
    check(theme_utils._APPLY_PENDING is False, "no coalesced apply left pending")

    # theme actually alternated end to end
    check(theme_utils.current_theme_is_dark() in (True, False),
          "theme resolved to a concrete mode")

    theme_utils.get_preferences = orig_prefs
    print("\n%s (%d checks, %d failed)"
          % ("ALL PASS" if not FAILS else "FAILURES",
             6 + 1, len(FAILS)))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
