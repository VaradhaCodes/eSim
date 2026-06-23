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


def show_about_dialog(parent):
    """Premium About eSim dialog — gradient background, rich typography,
    logo display, and depth shadow for an elevated, polished feel."""
    dlg = QtWidgets.QDialog(parent)
    dlg.setWindowTitle("About eSim")
    dlg.setFixedSize(480, 520)
    dlg.setObjectName("aboutDialog")

    # Icon
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    icon_path = os.path.join(base_dir, 'images', 'logo.png')
    if os.path.exists(icon_path):
        dlg.setWindowIcon(QtGui.QIcon(icon_path))

    # Use system palette to detect theme
    dark = parent.palette().color(QtGui.QPalette.ColorRole.Window).lightness() < 128

    # Gradient background and accent colors
    if dark:
        bg_top = QtGui.QColor("#0A1122")
        bg_bot = QtGui.QColor("#050812")
        card_bg = QtGui.QColor(13, 23, 40, 210)
        accent_a = QtGui.QColor("#53D7FF")
        accent_b = QtGui.QColor("#9B7CFF")
        title_color = "#F8FBFF"
        muted_color = "#94A8C3"
        subtle_color = "#5F728D"
        border_color = "rgba(83,215,255,0.20)"
    else:
        bg_top = QtGui.QColor("#F0F5FB")
        bg_bot = QtGui.QColor("#F3F7FC")
        card_bg = QtGui.QColor(255, 255, 255, 230)
        accent_a = QtGui.QColor("#0077A8")
        accent_b = QtGui.QColor("#6D5DF6")
        title_color = "#142033"
        muted_color = "#6B7F99"
        subtle_color = "#9AAABE"
        border_color = "rgba(0,119,168,0.14)"

    dlg.setStyleSheet(f"""
        #aboutDialog {{
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 {bg_top.name()}, stop:1 {bg_bot.name()});
        }}
    """)

    root = QtWidgets.QVBoxLayout(dlg)
    root.setContentsMargins(0, 0, 0, 0)
    root.setSpacing(0)

    # ── Hero gradient strip ──────────────────────────────────────
    hero_strip = QtWidgets.QWidget()
    hero_strip.setFixedHeight(140)
    hero_strip.setStyleSheet(f"""
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
            stop:0 {accent_a.name()}, stop:0.4 {accent_a.name()},
            stop:0.6 {accent_b.lighter(120).name()}, stop:1 {accent_b.name()});
    """)

    hero_layout = QtWidgets.QVBoxLayout(hero_strip)
    hero_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

    logo_lbl = QtWidgets.QLabel()
    if os.path.exists(icon_path):
        pix = QtGui.QPixmap(icon_path).scaled(
            64, 64, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation
        )
        logo_lbl.setPixmap(pix)
    logo_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    hero_layout.addWidget(logo_lbl)

    hero_title = QtWidgets.QLabel("eSim")
    hero_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    hero_title.setStyleSheet(
        "font-size: 28px; font-weight: 850; color: #FFFFFF; "
        "letter-spacing: -0.5px; background: transparent;"
    )
    hero_layout.addWidget(hero_title)

    tagline = QtWidgets.QLabel("Open Source EDA for Circuit Design")
    tagline.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    tagline.setStyleSheet(
        "font-size: 12px; color: rgba(255,255,255,0.85); "
        "background: transparent; letter-spacing: 0.4px;"
    )
    hero_layout.addWidget(tagline)

    root.addWidget(hero_strip)

    # ── Content card ─────────────────────────────────────────────
    content = QtWidgets.QWidget()
    content_layout = QtWidgets.QVBoxLayout(content)
    content_layout.setContentsMargins(32, 28, 32, 24)
    content_layout.setSpacing(12)

    version_lbl = QtWidgets.QLabel("Version 2.4")
    version_lbl.setStyleSheet(
        f"color: {title_color}; font-size: 18px; font-weight: 750;"
    )
    content_layout.addWidget(version_lbl)

    desc = QtWidgets.QLabel(
        "Circuit design, simulation, analysis, and PCB layout — "
        "all in one integrated environment. Built on KiCad and "
        "ngspice with a modern, fast UI."
    )
    desc.setWordWrap(True)
    desc.setStyleSheet(f"color: {muted_color}; font-size: 13px; line-height: 1.5;")
    content_layout.addWidget(desc)

    content_layout.addSpacing(8)

    # Separator with accent gradient
    sep = QtWidgets.QWidget()
    sep.setFixedHeight(2)
    sep.setStyleSheet(f"""
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {accent_a.name()}, stop:0.5 {accent_b.lighter(130).name()},
            stop:1 {accent_b.name()});
        border-radius: 1px;
    """)
    content_layout.addWidget(sep)

    content_layout.addSpacing(4)

    org = QtWidgets.QLabel("FOSSEE, IIT Bombay")
    org.setStyleSheet(f"color: {title_color}; font-size: 14px; font-weight: 650;")

    website = QtWidgets.QLabel(
        '<a href="https://esim.fossee.in" style="color: {acc}; '
        'text-decoration: none;">esim.fossee.in</a>'.format(
            acc=accent_a.name()
        )
    )
    website.setOpenExternalLinks(True)
    website.setStyleSheet(f"font-size: 12px;")

    credits = QtWidgets.QLabel(
        "Originally created by Fahim Khan\n"
        "Maintained by the eSim Team at FOSSEE"
    )
    credits.setStyleSheet(f"color: {subtle_color}; font-size: 11px;")
    credits.setWordWrap(True)

    content_layout.addWidget(org)
    content_layout.addWidget(website)
    content_layout.addSpacing(6)
    content_layout.addWidget(credits)
    content_layout.addStretch()

    # Close button
    btn_row = QtWidgets.QHBoxLayout()
    btn_row.addStretch()
    close_btn = QtWidgets.QPushButton("Close")
    close_btn.setFixedWidth(100)
    close_btn.clicked.connect(dlg.accept)
    btn_row.addWidget(close_btn)
    content_layout.addLayout(btn_row)

    root.addWidget(content, 1)

    # Depth shadow
    try:
        from frontEnd.motion import apply_popup_depth
        apply_popup_depth(dlg)
    except Exception:
        pass

    try:
        from frontEnd.motion import install_button_motion
        install_button_motion(dlg)
    except Exception:
        pass

    dlg.exec()
