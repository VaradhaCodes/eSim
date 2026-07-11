"""P1.1 regression: install_button_motion must be idempotent.

apply_fullscreen_feature calls install_button_motion(self) on EVERY dock open.
It used to build a fresh TactileButtonFilter each call and installEventFilter it
onto every button under the root; installing a *different* filter object stacks
(it never replaces), so after N dock opens each button carried N filters and
every hover fired N glow animations -- the "gets laggy over the session" bug.

These pin that a second call reuses the one filter and re-wires nothing.
"""
from PyQt6 import QtWidgets

from frontEnd import motion


def test_install_button_motion_reuses_filter_and_skips_wired_buttons(qapp):
    motion._motion_enabled_cache = True          # force glows on, no file read
    try:
        root = QtWidgets.QWidget()
        btn = QtWidgets.QPushButton("go", root)

        motion.install_button_motion(root)
        filt1 = root._esim_press_motion_filter
        eff1 = btn.graphicsEffect()
        assert btn.property("_esim_motion_installed") is True
        assert filt1 is not None

        # A second dock-open equivalent must NOT stack a new filter or a new
        # shadow effect onto the already-wired button.
        motion.install_button_motion(root)
        assert root._esim_press_motion_filter is filt1   # same filter reused
        assert btn.graphicsEffect() is eff1              # shadow not recreated
    finally:
        motion._motion_enabled_cache = None


def test_motion_disabled_installs_nothing(qapp):
    motion._motion_enabled_cache = False
    try:
        root = QtWidgets.QWidget()
        btn = QtWidgets.QPushButton("go", root)
        motion.install_button_motion(root)
        assert not btn.property("_esim_motion_installed")
        assert getattr(root, "_esim_press_motion_filter", None) is None
    finally:
        motion._motion_enabled_cache = None


def test_invalidate_motion_cache_forces_reread(qapp):
    motion._motion_enabled_cache = True
    motion.invalidate_motion_cache()
    assert motion._motion_enabled_cache is None


def test_motion_default_is_platform_specific(qapp, monkeypatch, tmp_path):
    # With no preferences file, motion_enabled() falls back to the platform
    # default: OFF on Windows (CPU-blurred effects are the biggest drag there),
    # ON elsewhere.
    import os
    from configuration import paths
    monkeypatch.setattr(
        paths, "esim_config_path", lambda *p: str(tmp_path / "absent.json"))
    motion.invalidate_motion_cache()
    try:
        assert motion.motion_enabled() is (os.name != "nt")
    finally:
        motion.invalidate_motion_cache()
