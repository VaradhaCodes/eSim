"""Welcome screen — the empty-state dashboard shown when no project is open.

Built with native Qt widgets (no HTML) so it inherits the active Qt
Style Sheet, plays nicely with High-DPI scaling, and exposes proper
keyboard / focus navigation. Ported from the Aurora design; the tool
tiles dispatch to the real QActions on our MainWindow, so the card
``attr_name`` values must match the attribute names in Application.py.
"""
from PyQt6 import QtCore, QtGui, QtWidgets
import os

try:
    from frontEnd.elevation import elevate
    from frontEnd.widgets import HoverSurfaceMixin, accent_color
except Exception:  # pragma: no cover - flat sys.path (script / harness run)
    from elevation import elevate
    from widgets import HoverSurfaceMixin, accent_color


# Repo root: src/browser/Welcome.py -> ../.. -> repo root, where images/ lives.
_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
INIT_PATH = _BASE_DIR + os.sep


# --------------------------------------------------------------------- helpers
def _rounded_pixmap(path: str, size: int, radius_ratio: float = 0.18) -> QtGui.QPixmap:
    """Load an icon and return it as a rounded pixmap.

    Falls back to a 1x1 transparent pixmap when the source file is
    missing, so a missing asset never crashes the welcome screen.
    """
    raw = QtGui.QPixmap(path)
    if raw.isNull():
        return QtGui.QPixmap(size, size)

    scaled = raw.scaled(
        size, size,
        QtCore.Qt.AspectRatioMode.KeepAspectRatio,
        QtCore.Qt.TransformationMode.SmoothTransformation,
    )
    canvas = QtGui.QPixmap(size, size)
    canvas.fill(QtCore.Qt.GlobalColor.transparent)

    painter = QtGui.QPainter(canvas)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    radius = int(size * radius_ratio)
    path_obj = QtGui.QPainterPath()
    path_obj.addRoundedRect(0, 0, size, size, radius, radius)
    painter.setClipPath(path_obj)
    x = (size - scaled.width()) // 2
    y = (size - scaled.height()) // 2
    painter.drawPixmap(x, y, scaled)
    painter.end()
    return canvas


# --------------------------------------------------------------------- card
class ToolCard(HoverSurfaceMixin, QtWidgets.QFrame):
    """A tool tile. Hover is painted here and ONLY here — the QSS
    ``QFrame#welcomeCard:hover`` rule was deleted in S2 because the two
    highlight systems were stacking (UI_AUDIT §2.2)."""

    def __init__(
        self,
        name: str,
        icon_path: str,
        description: str,
        attr_name: str,
        trigger_callback,
    ):
        super().__init__()
        self.attr_name = attr_name
        self.trigger_callback = trigger_callback

        self.setObjectName('welcomeCard')
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setMinimumHeight(96)
        self._press_origin = None
        self._press_anim = None
        self._release_anim = None

        self._init_hover_anim()

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(14)

        icon_label = QtWidgets.QLabel()
        icon_label.setPixmap(_rounded_pixmap(icon_path, 56, 0.18))
        icon_label.setFixedSize(56, 56)
        icon_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignCenter
        )
        icon_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout.addWidget(icon_label, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)

        text = QtWidgets.QVBoxLayout()
        text.setSpacing(2)
        title = QtWidgets.QLabel(name)
        title.setProperty('cssClass', 'welcomeTitle')
        title.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        desc = QtWidgets.QLabel(description)
        desc.setProperty('cssClass', 'welcomeDesc')
        desc.setWordWrap(True)
        desc.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        text.addWidget(title)
        text.addWidget(desc)
        text.addStretch(1)
        layout.addLayout(text, 1)

        chevron = QtWidgets.QLabel('>')
        chevron.setProperty('cssClass', 'subtle')
        chevron.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout.addWidget(chevron, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._hover_progress <= 0:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        # The wash is the theme's accent, not dark-theme cyan: on the light
        # sheet this tile is white, and #53D7FF over white is a different
        # brand colour than the one the rest of the light UI paints.
        painter.setBrush(accent_color(self, int(42 * self._hover_progress)))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 16, 16)

    # ------------------------------------------------------------------ events
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._press_origin = self.pos()
            anim = QtCore.QPropertyAnimation(self, b"pos", self)
            anim.setDuration(70)
            anim.setEndValue(self.pos() + QtCore.QPoint(0, 1))
            anim.setEasingCurve(QtCore.QEasingCurve.Type.OutQuad)
            self._press_anim = anim
            anim.start()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        fire = False
        if hasattr(self, "_press_origin") and self._press_origin:
            anim = QtCore.QPropertyAnimation(self, b"pos", self)
            anim.setDuration(150)
            anim.setEndValue(self._press_origin)
            anim.setEasingCurve(QtCore.QEasingCurve.Type.OutBack)
            self._release_anim = anim
            anim.start()
            # Fire only if the release lands inside the card (drag-away cancels),
            # and only on left-button release.
            if (event.button() == QtCore.Qt.MouseButton.LeftButton
                    and self.rect().contains(event.position().toPoint())):
                fire = True
        super().mouseReleaseEvent(event)
        if fire:
            self.trigger_callback(self.attr_name)

    def keyPressEvent(self, event):
        if event.key() in (
            QtCore.Qt.Key.Key_Return,
            QtCore.Qt.Key.Key_Enter,
            QtCore.Qt.Key.Key_Space,
        ):
            self.trigger_callback(self.attr_name)
        else:
            super().keyPressEvent(event)


# --------------------------------------------------------------------- hero
class HeroBanner(QtWidgets.QFrame):
    """A wide, gradient hero panel used at the top of the welcome page."""

    # Must track the border-radius of QFrame#welcomeHero in the .qss files,
    # otherwise the orb overlay paints square corners over the rounded panel.
    RADIUS = 18

    def __init__(self):
        super().__init__()
        self.setObjectName('welcomeHero')
        self.setMinimumHeight(160)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(20)

        # eSim logo (if available), otherwise a soft monogram
        logo_label = QtWidgets.QLabel()
        logo_path = os.path.join(_BASE_DIR, 'images', 'logo.png')
        if os.path.exists(logo_path):
            logo_pix = QtGui.QPixmap(logo_path).scaled(
                64, 64,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            logo_label.setPixmap(logo_pix)
        logo_label.setFixedSize(64, 64)
        layout.addWidget(logo_label, 0, QtCore.Qt.AlignmentFlag.AlignTop)

        # Title + subtitle
        text = QtWidgets.QVBoxLayout()
        text.setSpacing(6)
        # Formal solid-colour heading (no gradient text) — colour comes from
        # the #welcomeHeroTitle QSS rule so it stays theme-correct.
        title = QtWidgets.QLabel('Welcome to eSim')
        title.setObjectName('welcomeHeroTitle')
        subtitle = QtWidgets.QLabel(
            'Open-source EDA for circuit design, simulation, and PCB layout.'
        )
        subtitle.setObjectName('welcomeHeroSubtitle')
        subtitle.setWordWrap(True)

        about_text = (
            "It is an integrated tool built using open source softwares such as "
            "<a href='https://www.kicad.org/'>KiCad</a>, "
            "<a href='https://ngspice.sourceforge.io/'>Ngspice</a>, "
            "<a href='http://ghdl.free.fr'>GHDL</a>, "
            "<a href='https://www.veripool.org/verilator/'>Verilator</a>, "
            "<a href='https://www.makerchip.com/'>Makerchip IDE</a>, and "
            "<a href='https://skywater-pdk.rtfd.io/'>SkyWater SKY130 PDK</a>. "
            "eSim source is released under GNU General Public License.<br><br>"
            "Developed by the eSim Team at FOSSEE, IIT Bombay. "
            "Visit: <a href='https://esim.fossee.in/'>esim.fossee.in</a> | "
            "Discuss: <a href='https://forums.fossee.in/'>forums.fossee.in</a>"
        )
        about_label = QtWidgets.QLabel(about_text)
        about_label.setWordWrap(True)
        about_label.setOpenExternalLinks(True)
        about_label.setContentsMargins(0, 8, 0, 0)
        about_label.setProperty("cssClass", "muted")

        text.addWidget(title)
        text.addWidget(subtitle)
        text.addWidget(about_label)
        text.addStretch(1)
        layout.addLayout(text, 1)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        dark = self.palette().color(QtGui.QPalette.ColorRole.Window).lightness() < 128
        # Light needs the weaker wash: the same alpha that reads as a glow on
        # #050812 reads as a stain on #F3F7FC.
        accent = accent_color(self, 64 if dark else 34, "accent")
        violet = accent_color(self, 54 if dark else 28, "accent_2")
        orb = QtGui.QRadialGradient(QtCore.QPointF(rect.right() - 40, rect.top() + 30), max(rect.width(), rect.height()) * 0.65)
        orb.setColorAt(0.0, accent)
        orb.setColorAt(0.45, violet)
        orb.setColorAt(1.0, QtCore.Qt.GlobalColor.transparent)
        clip = QtGui.QPainterPath()
        clip.addRoundedRect(QtCore.QRectF(rect), self.RADIUS, self.RADIUS)
        painter.setClipPath(clip)
        painter.fillRect(rect, orb)
        painter.end()
        super().paintEvent(event)


# --------------------------------------------------------------------- section
class SectionHeader(QtWidgets.QLabel):
    """Small uppercase section header used between welcome cards."""

    def __init__(self, text: str):
        super().__init__(text)
        self.setProperty('cssClass', 'caps')
        self.setContentsMargins(4, 8, 0, 4)


# --------------------------------------------------------------------- page
class Welcome(QtWidgets.QWidget):
    """The Welcome screen that fills the dock area on first launch."""

    def __init__(self):
        super().__init__()

        # Tiles are elevated on first show, not here — see _elevate_later.
        self._pending_elevation = []

        # Outer container that scrolls so smaller docks still work
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Scroll area in case the dock is short
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setObjectName('welcomeScroll')
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        outer.addWidget(scroll, 1)

        # Inner widget with generous padding so drop shadows don't clip
        inner = QtWidgets.QWidget()
        scroll.setWidget(inner)
        inner_layout = QtWidgets.QVBoxLayout(inner)
        inner_layout.setContentsMargins(28, 22, 28, 24)
        inner_layout.setSpacing(8)

        # ---- Hero ----
        hero = HeroBanner()
        inner_layout.addWidget(hero)
        self._elevate_later(hero, "e3")
        inner_layout.addSpacing(6)

        # ---- Quick-start actions ----
        inner_layout.addWidget(SectionHeader('Quick start'))
        quick = QtWidgets.QGridLayout()
        quick.setSpacing(12)
        quick.setContentsMargins(4, 0, 4, 0)

        # attr names must match the QAction attributes on our MainWindow
        # (Application.py): new/open project are ``newproj`` / ``openproj``.
        quick_actions = [
            ('New project', 'newProject.png', 'Create a new eSim project', 'newproj'),
            ('Open project', 'openProject.png', 'Open an existing project', 'openproj'),
        ]
        for n, (name, icon, desc, attr) in enumerate(quick_actions):
            card = ToolCard(
                name,
                os.path.join(_BASE_DIR, 'images', icon),
                desc,
                attr,
                self.trigger_action,
            )
            quick.addWidget(card, 0, n)
            self._elevate_later(card, ToolCard.REST_LEVEL)
            quick.setColumnStretch(n, 1)
        inner_layout.addLayout(quick)

        # ---- Tools grid ----
        inner_layout.addWidget(SectionHeader('Tools'))
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(12)
        grid.setContentsMargins(4, 0, 4, 0)

        # Tiles follow the rail's pipeline order: Design -> Simulate -> Model
        # -> Convert. NGHDL lives as the VHDL path inside Makerchip; its tile
        # uses the dedicated ``nghdl`` action that jumps straight to that stage.
        tools = [
            ('Open Schematic', 'kicad.png', 'Design your circuit in KiCad', 'kicad'),
            ('Convert to Ngspice', 'ki-ng.png', 'Generate Ngspice netlist', 'conversion'),
            ('Simulate', 'ngspice.png', 'Run circuit simulation', 'ngspice'),
            ('Model Editor', 'model.png', 'Create or edit SPICE models', 'model'),
            ('Subcircuit', 'subckt.png', 'Build reusable subcircuits', 'subcircuit'),
            ('Makerchip', 'makerchip.png', 'Verilog via Makerchip', 'makerchip'),
            ('NGHDL', 'nghdl.png', 'Add VHDL digital models', 'nghdl'),
            ('Modelica', 'omedit.png', 'Convert to Modelica format', 'omedit'),
            ('OM Optimisation', 'omoptim.png', 'Open OpenModelica optimiser', 'omoptim'),
            ('Schematic Converter', 'icon.png', 'Import PSpice / LTspice files', 'conToeSim'),
        ]

        cols = 3
        for i, (name, icon_file, desc, attr) in enumerate(tools):
            row = i // cols
            col = i % cols
            card = ToolCard(
                name,
                os.path.join(_BASE_DIR, 'images', icon_file),
                desc,
                attr,
                self.trigger_action,
            )
            grid.addWidget(card, row, col)
            self._elevate_later(card, ToolCard.REST_LEVEL)
            grid.setColumnStretch(col, 1)

        inner_layout.addLayout(grid)

        inner_layout.addStretch(1)

    def _elevate_later(self, widget, level):
        """Queue a tile for elevation on first show.

        Elevating during __init__ would have to ask a half-built, unshown
        widget which theme it is in — and reading a palette on this path is
        also what tips a latent fault in the style repolish (see showEvent).
        Deferring is the more correct answer anyway: at first show the page is
        parented, polished and carries the theme it will actually be seen in.
        """
        self._pending_elevation.append((widget, level))

    def showEvent(self, event):
        super().showEvent(event)
        # One-shot: the queue empties, so a hide/show does not re-elevate.
        # A theme change re-tints these through elevation.retint(), driven by
        # apply_theme's effect sweep — not from here.
        pending, self._pending_elevation = self._pending_elevation, []
        for widget, level in pending:
            elevate(widget, level)

    # ------------------------------------------------------------------ dispatch
    def trigger_action(self, attr_name):
        """Resolve `attr_name` against the top-level MainWindow and trigger it."""
        app = self.window()
        if not app:
            return
        target = getattr(app, attr_name, None)
        if target is None:
            return
        if hasattr(target, 'trigger'):
            target.trigger()
        elif callable(target):
            target()
