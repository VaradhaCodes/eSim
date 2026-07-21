from PyQt6 import QtWidgets, QtGui, QtCore
import os

try:
    from frontEnd import tokens
except Exception:  # pragma: no cover - import when run as a script
    import tokens


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
    # Build the box the same way show_warning/show_info do (QMessageBox(parent)
    # + _prepare_msg) rather than the static QMessageBox.question() — the
    # static call is a parentless top-level window and is banned by the
    # dialog-hygiene guard (configuration/tests/test_dialog_hygiene.py).
    msg = QtWidgets.QMessageBox(parent)
    msg.setIcon(QtWidgets.QMessageBox.Icon.Question)
    msg.setWindowTitle(title)
    msg.setText(message)
    msg.setStandardButtons(
        QtWidgets.QMessageBox.StandardButton.Yes
        | QtWidgets.QMessageBox.StandardButton.No)
    msg.setDefaultButton(QtWidgets.QMessageBox.StandardButton.No)
    _prepare_msg(msg, "question")
    return msg.exec() == QtWidgets.QMessageBox.StandardButton.Yes


# Persisted across runs in the shared QSettings store. When the user ticks
# "Remember my choice" the popup is skipped on every later simulation and the
# saved answer is reused. It can be re-enabled from Preferences ▸ Simulation
# ("Ask before generating Ngspice plots").
NGSPICE_REMEMBER_KEY = "ngspicePlots/remember"
NGSPICE_FLAG_KEY = "ngspicePlots/flag"


def ask_ngspice_plots(parent):
    """Return whether to also generate native ngspice plots for this run.

    If the user previously chose "Remember my choice", reuse the stored answer
    silently. Otherwise show the Yes/No popup (with a remember checkbox) and
    persist the decision when the box is ticked. Shared by
    Application.plotFlagPopBox and TerminalUi's redo path so the popup and its
    QSettings keys live in exactly one place.
    """
    settings = QtCore.QSettings('eSim', 'eSim')
    if settings.value(NGSPICE_REMEMBER_KEY, False, type=bool):
        return settings.value(NGSPICE_FLAG_KEY, False, type=bool)

    msg_box = QtWidgets.QMessageBox(parent)
    msg_box.setWindowTitle("Ngspice Plots")
    msg_box.setText("Do you want Ngspice plots?")
    # Widen the message label so the window title is not clipped behind the
    # close/minimise/maximise buttons on tight window managers. QMessageBox
    # ignores setMinimumWidth, so size via the label.
    msg_box.setStyleSheet("QLabel { min-width: 360px; }")

    remember_cb = QtWidgets.QCheckBox(
        "Remember my choice (re-enable in Preferences ▸ Simulation)")
    msg_box.setCheckBox(remember_cb)

    yes_button = msg_box.addButton(
        "Yes", QtWidgets.QMessageBox.ButtonRole.YesRole)
    msg_box.addButton("No", QtWidgets.QMessageBox.ButtonRole.NoRole)

    msg_box.exec()
    flag = msg_box.clickedButton() == yes_button

    if remember_cb.isChecked():
        settings.setValue(NGSPICE_REMEMBER_KEY, True)
        settings.setValue(NGSPICE_FLAG_KEY, flag)

    return flag


ESIM_VERSION = "2.6.0"


def _about_palette(parent):
    """Theme-aware colour set for the About surfaces. Restrained, on-brand —
    a single quiet accent, neutral chip for the (warm bronze) logo, no loud
    full-saturation gradient.

    Every value is derived from ``tokens`` rather than hand-copied beside it
    (UI_AUDIT §2.4): this dict used to hold ~14 literals per theme, three of
    which had already drifted off the sheets. The SHAPE is unchanged — both
    consumers (``show_about_dialog`` and ``PreferencesDialog._build_about_page``)
    read the same keys.
    """
    dark = parent.palette().color(
        QtGui.QPalette.ColorRole.Window).lightness() < 128
    t = tokens.theme(dark)
    text_rgb = tokens.hex_to_rgb(t["text"])
    acc_rgb = tokens.hex_to_rgb(t["accent"])

    def rgba(rgb, a):
        return "rgba(%d,%d,%d,%s)" % (rgb[0], rgb[1], rgb[2], a)

    return dict(
        dark=dark,
        # The card, and the band across its top, one surface step apart.
        page=t["surface"], header=t["surface_2"],
        # Dark drops the logo chip BELOW the card so the bronze coin sits in a
        # well; light has nowhere lower to go than the card itself.
        chip=t["bg_raise"] if dark else t["surface"],
        chip_border=rgba(text_rgb, "0.12"),
        title=t["text"], muted=t["text_muted"], subtle=t["text_subtle"],
        accent=t["accent"],
        # The link steps AWAY from the page: brightest accent on dark, and on
        # light the base accent, which is already the darkest readable one
        # (4.6:1 on #FFFFFF — accent_hi would be 3.0:1).
        link=t["accent_hi"] if dark else t["accent"],
        sep=rgba(text_rgb, "0.08"),
        # Same tint, deeper on dark where a 10% wash disappears.
        pill_bg=rgba(acc_rgb, "0.14" if dark else "0.10"),
        pill_fg=t["accent_hi"] if dark else t["accent"],
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
    # Minimum, not fixed (UI_AUDIT §C7): build_qss scales every px metric by
    # the zoom preference, so at 150% the type inside this dialog grew while a
    # fixed 440x500 frame did not — the credits clipped. adjustSize() below,
    # once the content exists, lets the frame track what it actually holds.
    dlg.setMinimumSize(440, 500)
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
        f"color: {c['title']}; font-size: 14px; font-weight: 700; "
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

    # Grow to the content (never below the minimum above) so a zoomed sheet
    # gets the frame it needs instead of a scrollbar-less clip.
    dlg.adjustSize()
    dlg.exec()
