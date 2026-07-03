from PyQt6 import QtWidgets, QtGui, QtCore
import os


def _prepare_msg(msg, kind):
    msg.setObjectName("esimMessageBox")
    msg.setProperty("messageKind", kind)
    try:
        from frontEnd.motion import apply_popup_depth
        apply_popup_depth(msg)
    except Exception:
        pass


def show_error(parent, title, message):
    msg = QtWidgets.QMessageBox(parent)
    msg.setIcon(QtWidgets.QMessageBox.Icon.Critical)
    msg.setWindowTitle(title)
    msg.setText(message)
    _prepare_msg(msg, "error")
    msg.exec()


def show_warning(parent, title, message):
    msg = QtWidgets.QMessageBox(parent)
    msg.setIcon(QtWidgets.QMessageBox.Icon.Warning)
    msg.setWindowTitle(title)
    msg.setText(message)
    _prepare_msg(msg, "warning")
    msg.exec()


def show_info(parent, title, message):
    msg = QtWidgets.QMessageBox(parent)
    msg.setIcon(QtWidgets.QMessageBox.Icon.Information)
    msg.setWindowTitle(title)
    msg.setText(message)
    _prepare_msg(msg, "info")
    msg.exec()


def ask_question(parent, title, message):
    reply = QtWidgets.QMessageBox.question(
        parent, title, message,
        QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        QtWidgets.QMessageBox.StandardButton.No
    )
    return reply == QtWidgets.QMessageBox.StandardButton.Yes


ESIM_VERSION = "2.5.0"


def _about_palette(parent):
    """Theme-aware colour set for the About surfaces. Restrained, on-brand —
    a single quiet accent, neutral chip for the (warm bronze) logo, no loud
    full-saturation gradient."""
    dark = parent.palette().color(
        QtGui.QPalette.ColorRole.Window).lightness() < 128
    if dark:
        return dict(
            dark=True,
            page="#0E1728", header="#121E33",
            chip="#0A1220", chip_border="rgba(255,255,255,0.10)",
            title="#F4F8FF", muted="#9FB1CC", subtle="#6C7F99",
            accent="#53D7FF", link="#8BEAFF",
            sep="rgba(255,255,255,0.08)",
            pill_bg="rgba(83,215,255,0.14)", pill_fg="#8BEAFF",
        )
    return dict(
        dark=False,
        page="#FFFFFF", header="#F5F8FC",
        chip="#FFFFFF", chip_border="rgba(20,32,51,0.12)",
        title="#142033", muted="#5A6E89", subtle="#9AAABE",
        accent="#0077A8", link="#0077A8",
        sep="rgba(20,32,51,0.08)",
        pill_bg="rgba(0,119,168,0.10)", pill_fg="#0077A8",
    )


def _logo_chip(icon_path, c, size=76, logo_px=48):
    """Bronze eSim coin inside a clean neutral rounded chip so its warm tone
    reads cleanly on any theme — fixes the coin-on-cyan clash + opaque band."""
    chip = QtWidgets.QLabel()
    chip.setFixedSize(size, size)
    chip.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    chip.setStyleSheet(
        f"background: {c['chip']}; border: 1px solid {c['chip_border']}; "
        f"border-radius: {size // 2}px;")
    if os.path.exists(icon_path):
        pix = QtGui.QPixmap(icon_path).scaled(
            logo_px, logo_px, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation)
        chip.setPixmap(pix)
    return chip


def _about_header(icon_path, c, rounded=False):
    """Quiet header band: thin accent hairline, logo chip, wordmark, tagline.
    No gradient — structure comes from a subtle raised surface + accent rule.
    ``rounded`` frames it as a standalone card (used in the Preferences tab)."""
    header = QtWidgets.QWidget()
    header.setObjectName("aboutHeader")
    if rounded:
        header.setStyleSheet(
            f"#aboutHeader {{ background: {c['header']}; "
            f"border: 1px solid {c['sep']}; border-radius: 12px; }}")
    else:
        header.setStyleSheet(
            f"#aboutHeader {{ background: {c['header']}; "
            f"border-bottom: 1px solid {c['sep']}; }}")
    outer = QtWidgets.QVBoxLayout(header)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)

    rule = QtWidgets.QWidget()
    rule.setFixedHeight(3)
    rule.setStyleSheet(f"background: {c['accent']};")
    outer.addWidget(rule)

    body = QtWidgets.QVBoxLayout()
    body.setContentsMargins(28, 22, 28, 20)
    body.setSpacing(8)
    body.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)

    chip = _logo_chip(icon_path, c)
    body.addWidget(chip, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)

    title = QtWidgets.QLabel("eSim")
    title.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
    tf = title.font()
    tf.setPointSize(22)
    tf.setWeight(QtGui.QFont.Weight.Black)
    title.setFont(tf)
    title.setStyleSheet(f"color: {c['title']}; background: transparent;")
    body.addWidget(title)

    tagline = QtWidgets.QLabel("Open Source EDA for Circuit Design")
    tagline.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
    gf = tagline.font()
    gf.setPointSize(9)
    tagline.setFont(gf)
    tagline.setStyleSheet(f"color: {c['muted']}; background: transparent;")
    body.addWidget(tagline)

    outer.addLayout(body)
    return header


def show_about_dialog(parent):
    """About eSim — a calm, on-brand card: neutral logo chip, quiet accent
    header (no loud gradient), correct version, product blurb and credits."""
    dlg = QtWidgets.QDialog(parent)
    dlg.setWindowTitle("About eSim")
    dlg.setFixedSize(440, 500)
    dlg.setObjectName("aboutDialog")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    icon_path = os.path.join(base_dir, 'images', 'logo.png')
    if os.path.exists(icon_path):
        dlg.setWindowIcon(QtGui.QIcon(icon_path))

    c = _about_palette(parent)
    dlg.setStyleSheet(f"#aboutDialog {{ background: {c['page']}; }}")

    root = QtWidgets.QVBoxLayout(dlg)
    root.setContentsMargins(0, 0, 0, 0)
    root.setSpacing(0)

    root.addWidget(_about_header(icon_path, c))

    # ── Content ──────────────────────────────────────────────────
    content = QtWidgets.QWidget()
    content_layout = QtWidgets.QVBoxLayout(content)
    content_layout.setContentsMargins(28, 22, 28, 20)
    content_layout.setSpacing(12)

    version = QtWidgets.QLabel(f"Version {ESIM_VERSION}")
    version.setStyleSheet(
        f"color: {c['pill_fg']}; background: {c['pill_bg']}; "
        f"border-radius: 10px; padding: 4px 12px; font-weight: 700;")
    version.setSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum,
                          QtWidgets.QSizePolicy.Policy.Fixed)
    content_layout.addWidget(version, 0, QtCore.Qt.AlignmentFlag.AlignLeft)

    desc = QtWidgets.QLabel(
        "Circuit design, simulation, analysis and PCB layout in a single "
        "integrated environment — built on KiCad and ngspice.")
    desc.setWordWrap(True)
    desc.setStyleSheet(f"color: {c['muted']}; font-size: 13px;")
    content_layout.addWidget(desc)

    content_layout.addSpacing(2)

    sep = QtWidgets.QWidget()
    sep.setFixedHeight(1)
    sep.setStyleSheet(f"background: {c['sep']};")
    content_layout.addWidget(sep)

    content_layout.addSpacing(2)

    org = QtWidgets.QLabel("FOSSEE, IIT Bombay")
    org.setStyleSheet(
        f"color: {c['title']}; font-size: 14px; font-weight: 650; "
        "background: transparent;")
    content_layout.addWidget(org)

    website = QtWidgets.QLabel(
        f'<a href="https://esim.fossee.in" '
        f'style="color: {c["link"]}; text-decoration: none;">esim.fossee.in</a>')
    website.setOpenExternalLinks(True)
    website.setStyleSheet("font-size: 12px; background: transparent;")
    content_layout.addWidget(website)

    content_layout.addSpacing(4)

    credits = QtWidgets.QLabel(
        "Originally created by Fahim Khan\n"
        "Maintained by the eSim team at FOSSEE")
    credits.setWordWrap(True)
    credits.setStyleSheet(
        f"color: {c['subtle']}; font-size: 11px; background: transparent;")
    content_layout.addWidget(credits)

    content_layout.addStretch()

    btn_row = QtWidgets.QHBoxLayout()
    btn_row.addStretch()
    close_btn = QtWidgets.QPushButton("Close")
    close_btn.setProperty("cssClass", "secondary")
    close_btn.setFixedWidth(100)
    close_btn.clicked.connect(dlg.accept)
    btn_row.addWidget(close_btn)
    content_layout.addLayout(btn_row)

    root.addWidget(content, 1)

    for fn in ("apply_popup_depth", "install_button_motion"):
        try:
            mod = __import__("frontEnd.motion", fromlist=[fn])
            getattr(mod, fn)(dlg)
        except Exception:
            pass

    dlg.exec()
