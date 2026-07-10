"""The zoom pill and theme toggle must track the toolbar's zoom.

build_qss() scales the px metrics *inside* the stylesheet, but sizes set from
Python are invisible to it. The zoom pill used to be pinned at
setMinimumWidth(132) and the theme toggle at a QSS min-width/min-height of
32px, so zooming out shrank the "60%" text while the box around it stayed put,
and the toggle never matched the file / workspace icon buttons at any zoom.
"""
import os
import sys

import pytest

from PyQt6 import QtCore, QtWidgets

# Application.py does a bare ``import pathmagic``, so its own directory has to
# be importable -- the launcher gets this for free by chdir'ing into frontEnd.
_FRONTEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _FRONTEND not in sys.path:
    sys.path.insert(0, _FRONTEND)


class _StubToolbar:
    """Stands in for a QToolBar: records the icon size, has no home button."""

    def __init__(self):
        self.icon_size = QtCore.QSize(28, 28)

    def setIconSize(self, size):
        self.icon_size = size

    def widgetForAction(self, action):
        return None


class _Stub:
    """Minimum surface _apply_view_control_metrics() touches, so the test
    doesn't have to build the whole Application window."""

    def __init__(self):
        self.topToolbar = _StubToolbar()
        self.lefttoolbar = _StubToolbar()
        self.zoom_container = QtWidgets.QWidget()
        self.zoom_layout = QtWidgets.QHBoxLayout(self.zoom_container)
        self.zoom_label = QtWidgets.QLabel()
        self.theme_toggle_btn = QtWidgets.QToolButton()


def _metrics(stub, zoom):
    from frontEnd.Application import Application
    Application._apply_view_control_metrics(stub, zoom)


@pytest.fixture
def stub(qapp):
    return _Stub()


@pytest.mark.parametrize("zoom", [50, 60, 100, 200, 300])
def test_pill_and_toggle_scale_with_zoom(stub, zoom):
    _metrics(stub, zoom)
    scale = zoom / 100.0

    assert stub.zoom_container.minimumWidth() == round(132 * scale)
    assert stub.zoom_label.minimumWidth() == round(50 * scale)
    assert stub.topToolbar.icon_size.width() == round(28 * scale)
    assert stub.lefttoolbar.icon_size.width() == round(40 * scale)


def test_toggle_is_square_and_pill_height_matches_it(stub):
    _metrics(stub, 100)
    box = stub.theme_toggle_btn.height()
    assert stub.theme_toggle_btn.width() == box
    # The two view controls read as a matched pair with the icon buttons.
    assert stub.zoom_container.height() == box
    # The toggle carries a real icon, sized like a toolbar icon.
    assert stub.theme_toggle_btn.iconSize().width() == 28


def test_zooming_out_then_in_restores_the_original_box(stub):
    _metrics(stub, 100)
    before = (stub.zoom_container.minimumWidth(),
              stub.theme_toggle_btn.width())
    _metrics(stub, 60)
    assert stub.zoom_container.minimumWidth() < before[0]
    assert stub.theme_toggle_btn.width() < before[1]
    _metrics(stub, 100)
    assert (stub.zoom_container.minimumWidth(),
            stub.theme_toggle_btn.width()) == before


def test_theme_toggle_icon_renders_non_empty(qapp):
    from frontEnd.icon_paths import theme_toggle_icon
    icon = theme_toggle_icon()
    assert not icon.isNull()
    # Rasterised well above the 24px viewBox so 300% zoom stays crisp.
    pixmap = icon.pixmap(QtCore.QSize(256, 256))
    assert pixmap.width() >= 256
    assert not pixmap.toImage().allGray()  # actually painted something
