#!/bin/bash
#=============================================================================
#          FILE: install-eSim.sh
#
#         USAGE: ./install-eSim.sh --install
#                ./install-eSim.sh --uninstall
#                ./install-eSim.sh --dry-run     (preview profile + actions)
#
#   DESCRIPTION: Unified installer for the eSim EDA Suite on Ubuntu.
#                ONE script, version-aware via detect_profile(). Replaces the
#                old per-version install-eSim-scripts/install-eSim-XX.04.sh set
#                that drifted out of sync (only 24.04 was maintained).
#
#                Supported: Ubuntu 23.04 / 24.04 / 25.04 / 26.04
#                Target app: current master (PyQt6 + QScintilla code editor,
#                            kicadLibrary as dir-or-tarball, KiCad 8/9).
#
#                Idempotent: safe to re-run. --install cleans eSim-owned bits
#                first, so reinstall over ANY prior eSim version is clean.
#
#       AUTHORS: Fahim Khan, Rahul Paknikar, Saurabh Bansode, Sumanto Kar,
#                Partha Singha Roy, Jayanth Tatineni, Anshul Verma,
#                Shiva Krishna Sangati, Harsha Narayana P
#  ORGANIZATION: eSim Team, FOSSEE, IIT Bombay
#       CREATED: Wednesday 15 July 2015 15:26
#      REVISION: June 2026 — unified version-profile rewrite
#=============================================================================

# NOTE: `set -e`/`set -E` + an ERR trap are enabled only inside --install
# (see Main). We deliberately do NOT use `set -u`/`pipefail` globally: `set -u`
# breaks `source <venv>/bin/activate`, and `pipefail` would abort harmless
# `ls | grep` detections when grep matches nothing. All profile vars are
# explicitly initialised below instead.

#-----------------------------------------------------------------------------
# Globals
#-----------------------------------------------------------------------------
config_dir="$HOME/.esim"
config_file="config.ini"
eSim_Home=""        # resolved by resolve_esim_home()

# Profile vars (filled by detect_profile)
UBUNTU_VER=""
KICAD_SOURCE=""     # ppa | universe
KICAD_PPA=""
KICAD_MIN_MAJOR=""
QT_PKGS="python3-pyqt6 python3-pyqt6.qtsvg pyqt6-dev-tools"
QSCI_PKG="python3-pyqt6.qsci"

#-----------------------------------------------------------------------------
# Helpers
#-----------------------------------------------------------------------------
log()  { echo -e "\n>>> $*"; }
warn() { echo -e "[WARN] $*" >&2; }
die()  { echo -e "\n[ERROR] $*\n" >&2; exit 1; }

error_exit() {
    echo -e "\n\nError! Kindly resolve the above error(s) and try again."
    echo -e "Aborting installation...\n"
}

# Locate the eSim root (the dir holding VERSION + src/ + library/). Works when
# run from the repo root, from Ubuntu/, or from an extracted release zip root.
resolve_esim_home() {
    local d
    for d in "$(pwd)" \
             "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" \
             "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; do
        if [ -f "$d/VERSION" ] && [ -d "$d/src" ] && [ -d "$d/library" ]; then
            eSim_Home="$d"
            return 0
        fi
    done
    die "Cannot locate the eSim root (need VERSION + src/ + library/).
       cd into the top-level eSim directory and run ./install-eSim.sh again."
}

# Fail fast on insufficient disk BEFORE any apt work, with a message that
# says exactly how much to free. A fresh install needs roughly (measured on
# real 24.04/26.04 installs):
#   /      ~3.5 GB  apt payloads (KiCad ~0.5 GB, GHDL+LLVM ~0.7 GB, Qt/python/
#                   build tools, deb download cache) + SKY130 PDK ~0.7 GB
#   $HOME  ~1.5 GB  eSim tree + venv + nghdl build scratch (~1 GB peak)
# On a RE-install most of that is already on disk, so low space only warns.
# Without this check the install used to run for 20 minutes and then quietly
# skip the simulator build — see installNghdl(), which is now fatal instead.
preflightDisk() {
    if [ -n "${ESIM_SKIP_DISK_CHECK:-}" ]; then
        warn "ESIM_SKIP_DISK_CHECK set — skipping the disk-space preflight."
        return 0
    fi

    local root_fs home_fs free_root free_home short=""
    root_fs=$(df -Pk /       | awk 'NR==2 {print $1}')
    home_fs=$(df -Pk "$HOME" | awk 'NR==2 {print $1}')
    free_root=$(df -Pk /       | awk 'NR==2 {print $4}')
    free_home=$(df -Pk "$HOME" | awk 'NR==2 {print $4}')
    # df failed or gave nothing parseable — don't block the install on it.
    [ -n "$free_root" ] && [ -n "$free_home" ] || return 0

    if [ "$root_fs" = "$home_fs" ]; then
        [ "$free_home" -lt 5000000 ] && short="$((free_home / 1024)) MB free \
on $HOME — a fresh install needs ~5 GB (apt packages + PDKs + simulator build)"
    else
        [ "$free_root" -lt 3500000 ] && short="$((free_root / 1024)) MB free \
on / — apt packages + PDKs need ~3.5 GB"
        [ -z "$short" ] && [ "$free_home" -lt 1500000 ] && \
            short="$((free_home / 1024)) MB free on $HOME — the simulator \
build needs ~1.5 GB"
    fi
    [ -z "$short" ] && return 0

    if [ -f "$config_dir/$config_file" ]; then
        warn "Low disk space: $short."
        warn "Continuing because an existing eSim install was found (re-installs"
        warn "reuse most of it), but watch for out-of-space failures below."
        return 0
    fi
    die "Not enough disk space: $short.
       Free up space and re-run ./install-eSim.sh --install.
       (ESIM_SKIP_DISK_CHECK=1 bypasses this check.)"
}

# Set per-version profile variables. Single source of truth for what differs
# between Ubuntu releases.
detect_profile() {
    if [ ! -r /etc/os-release ]; then
        die "/etc/os-release not readable — cannot detect Ubuntu version."
    fi
    UBUNTU_VER=$(grep '^VERSION_ID=' /etc/os-release | cut -d '"' -f 2)

    case "$UBUNTU_VER" in
        23.04)
            # Lunar is EOL; best-effort. KiCad 9 PPA may not publish for lunar,
            # installKicad falls back to universe automatically.
            KICAD_SOURCE="ppa"; KICAD_PPA="kicad/kicad-9.0-releases"; KICAD_MIN_MAJOR=7
            warn "Ubuntu 23.04 is end-of-life. Support is best-effort."
            ;;
        24.04)
            KICAD_SOURCE="ppa"; KICAD_PPA="kicad/kicad-9.0-releases"; KICAD_MIN_MAJOR=9
            ;;
        25.04)
            # KiCad 9 PPA is unreliable on plucky; universe ships KiCad 8.
            KICAD_SOURCE="universe"; KICAD_MIN_MAJOR=8
            ;;
        26.04)
            KICAD_SOURCE="universe"; KICAD_MIN_MAJOR=9
            ;;
        22.04)
            die "Ubuntu 22.04 LTS is not supported.
       Its apt repositories ship Verilator 4 and KiCad 6, but this eSim needs
       Verilator 5 and KiCad >= 7 (PyQt6 GUI, Verilator-5 NgVeri build, and the
       KiCad-9 custom netlister). Please use Ubuntu 24.04 LTS or 26.04 LTS."
            ;;
        *)
            die "Unsupported Ubuntu version: $UBUNTU_VER
       Supported: 23.04, 24.04, 25.04, 26.04 (24.04 LTS / 26.04 LTS recommended)"
            ;;
    esac

    log "Profile: Ubuntu $UBUNTU_VER | KiCad via $KICAD_SOURCE (min major $KICAD_MIN_MAJOR)"
}

#-----------------------------------------------------------------------------
# Install steps
#-----------------------------------------------------------------------------
createConfigFile() {
    log "Writing $config_dir/$config_file"
    mkdir -p "$config_dir"
    rm -f "$config_dir/$config_file"
    {
        echo "[eSim]"
        echo "eSim_HOME = $eSim_Home"
        echo "LICENSE = %(eSim_HOME)s/LICENSE"
        echo "KicadLib = %(eSim_HOME)s/library/kicadLibrary"
        echo "IMAGES = %(eSim_HOME)s/images"
        echo "VERSION = %(eSim_HOME)s/VERSION"
        echo "MODELICA_MAP_JSON = %(eSim_HOME)s/library/ngspicetoModelica/Mapping.json"
    } >> "$config_dir/$config_file"
}

# Sweep artifacts of OLD eSim releases (2.x and earlier) so installing on
# top of any prior eSim just works. Each install step below already cleans
# and regenerates its own bits (config, venv, launcher, symbols, ngspice,
# SKY130); this handles only what old releases left OUTSIDE those paths.
# Everything touched here is eSim-owned — user files (old extracted eSim
# directories, zips) are reported, never deleted.
cleanLegacyEsim() {
    log "Checking for artifacts of an older eSim install"

    # Legacy-format symbol libs: this eSim ships only .kicad_sym. The install
    # rsync never deletes, so stale eSim_*.lib from old releases would linger
    # in /usr/share forever (uninstall already removes them).
    sudo rm -f /usr/share/kicad/symbols/eSim_*.lib 2>/dev/null || true

    # Old eSim's KiCad PPA entries (kicad-6.0 era). On a newer/upgraded
    # Ubuntu those PPAs publish no Release file, which makes every
    # `apt-get update` below fail hard. installKicad re-adds the right
    # source for this Ubuntu version, so dropping them is always safe.
    if ls /etc/apt/sources.list.d/kicad* &>/dev/null; then
        warn "Removing old KiCad apt sources left by a previous eSim:"
        ls /etc/apt/sources.list.d/kicad* | sed 's/^/       /'
        sudo rm -f /etc/apt/sources.list.d/kicad*
    fi

    # Old nghdl (eSim <= 2.5) compiled GHDL from source into /usr/local —
    # typically the mcode backend. /usr/local/bin shadows /usr/bin in PATH,
    # so it would hijack the apt ghdl-llvm/gcc this installer sets up and
    # break VHDL co-simulation (mcode cannot link the nghdl socket server).
    if [ -x /usr/local/bin/ghdl ]; then
        warn "Found /usr/local/bin/ghdl: $(/usr/local/bin/ghdl --version 2>/dev/null | head -n 1)"
        warn "It shadows the GHDL this installer sets up (old eSim/nghdl builds put it there)."
        read -rp "Remove the /usr/local GHDL? (y/n): " g
        if [[ "$g" =~ ^[Yy] ]]; then
            sudo rm -rf /usr/local/bin/ghdl /usr/local/bin/ghdl1-* \
                        /usr/local/lib/ghdl /usr/local/include/ghdl*
            log "/usr/local GHDL removed"
        else
            warn "Keeping it — VHDL co-simulation will likely use THIS ghdl (PATH order) and fail."
        fi
    fi

    # Old standalone nghdl GUI launcher; nghdl is embedded in eSim now and
    # the tree it pointed at is usually gone. (Uninstall removes it too.)
    sudo rm -f /usr/local/bin/nghdl 2>/dev/null || true

    # Purely informational — old trees that are no longer referenced once
    # this install repoints the launcher and ~/.esim/config.ini. Dirs only
    # (the glob would otherwise flag the user's freshly-downloaded release
    # .zip/.sha256), and never the tree we are installing FROM: the release
    # zip extracts to eSim-2.x/, so $eSim_Home itself matches the glob —
    # telling the user their live install is "safe to delete" broke installs.
    local d
    for d in "$HOME/ngspice-nghdl" "$HOME"/eSim-2.* "$HOME"/Downloads/eSim-2.*; do
        [ -d "$d" ] || continue
        [ "$(realpath "$d" 2>/dev/null)" = "$(realpath "$eSim_Home" 2>/dev/null)" ] && continue
        warn "Old eSim-era directory found: $d (unused after this install — safe to delete)"
    done
    return 0
}

installDependency() {
    log "Updating apt index"
    set +e; trap "" ERR
    sudo apt-get update
    set -e; trap error_exit ERR

    log "Installing base system packages"
    sudo apt-get install -y \
        python3-full python3-venv python3-virtualenv python3-pip \
        xterm xz-utils unzip rsync git build-essential \
        python3-psutil python3-setuptools \
        python3-matplotlib python3-numpy python3-scipy

    log "Creating virtualenv (with system site-packages so apt Qt is visible)"
    rm -rf "$config_dir/env"
    if command -v virtualenv &>/dev/null; then
        virtualenv --system-site-packages "$config_dir/env"
    else
        python3 -m venv --system-site-packages "$config_dir/env"
    fi
    # shellcheck disable=SC1091
    source "$config_dir/env/bin/activate"
    pip install --upgrade pip

    # eSim-specific pure-python deps not packaged in apt. Heavy/native deps
    # (PyQt6, matplotlib, numpy, scipy) come from apt above to stay mutually
    # consistent — pinning them via pip here is what made the old scripts fragile.
    log "Installing eSim Python deps (watchdog, makerchip, sandpiper, hdlparse, volare)"
    set +e; trap "" ERR
    pip install watchdog
    pip install "https://github.com/hdl/pyhdlparser/tarball/master" || warn "hdlparse (github) failed"
    pip install makerchip-app   || warn "makerchip-app failed"
    pip install sandpiper-saas  || warn "sandpiper-saas failed"
    pip install volare          || warn "volare failed"
    set -e; trap error_exit ERR
}

installQt() {
    log "Installing PyQt6 + QScintilla (apt)"
    sudo apt-get install -y $QT_PKGS
    sudo apt-get install -y "$QSCI_PKG" \
        || warn "$QSCI_PKG unavailable on this release — code editor will be disabled"

    # Verify the GUI toolkit actually imports inside the venv.
    # shellcheck disable=SC1091
    source "$config_dir/env/bin/activate"
    python3 -c "import PyQt6.QtWidgets" 2>/dev/null \
        || warn "PyQt6 import failed — eSim GUI will not start"
    python3 -c "import PyQt6.QtSvg" 2>/dev/null \
        || warn "PyQt6.QtSvg import failed — eSim GUI will not start (install python3-pyqt6.qtsvg)"
    python3 -c "import PyQt6.Qsci" 2>/dev/null \
        || warn "PyQt6.Qsci import failed — the code editor will be unavailable"
}

installKicad() {
    log "Installing KiCad (target major >= $KICAD_MIN_MAJOR)"

    if dpkg -s kicad &>/dev/null; then
        local cur
        cur=$(dpkg-query -W -f='${Version}' kicad | grep -oP '^\d+')
        if [ "${cur:-0}" -ge "$KICAD_MIN_MAJOR" ]; then
            log "KiCad $cur already installed (>= $KICAD_MIN_MAJOR) — keeping it"
            return 0
        fi
        warn "KiCad $cur installed but older than target major $KICAD_MIN_MAJOR"
        read -rp "Remove it and install KiCad >= $KICAD_MIN_MAJOR? (y/n): " r
        [[ "$r" =~ ^[Yy] ]] || die "Keeping existing KiCad $cur; aborting install."
        sudo apt-get remove --purge -y kicad kicad-footprints kicad-libraries kicad-symbols kicad-templates
        sudo apt-get autoremove -y
    fi

    sudo apt-get install -y software-properties-common

    if [ "$KICAD_SOURCE" = "ppa" ]; then
        if ! grep -rq "$KICAD_PPA" /etc/apt/sources.list /etc/apt/sources.list.d/ 2>/dev/null; then
            log "Adding PPA ppa:$KICAD_PPA"
            set +e; trap "" ERR
            sudo add-apt-repository -y "ppa:$KICAD_PPA"
            local ppa_rc=$?
            set -e; trap error_exit ERR
            if [ $ppa_rc -ne 0 ]; then
                warn "KiCad PPA add failed (release may be unsupported) — falling back to universe"
                sudo add-apt-repository -y universe
            fi
        fi
    else
        sudo add-apt-repository -y universe
    fi

    sudo apt-get update
    sudo apt-get install -y --no-install-recommends \
        kicad kicad-footprints kicad-libraries kicad-symbols kicad-templates

    local newmajor
    newmajor=$(dpkg-query -W -f='${Version}' kicad 2>/dev/null | grep -oP '^\d+')
    if [ "${newmajor:-0}" -ge "$KICAD_MIN_MAJOR" ]; then
        log "KiCad $newmajor installed"
    else
        warn "Installed KiCad major ${newmajor:-unknown} is below target $KICAD_MIN_MAJOR (eSim may still work)"
    fi
}

copyKicadLibrary() {
    log "Installing eSim KiCad symbol library"

    local libdir
    if [ -d "$eSim_Home/library/kicadLibrary" ]; then
        libdir="$eSim_Home/library/kicadLibrary"
    elif [ -f "$eSim_Home/library/kicadLibrary.tar.xz" ]; then
        tar -xJf "$eSim_Home/library/kicadLibrary.tar.xz" -C "$eSim_Home/library"
        libdir="$eSim_Home/library/kicadLibrary"
    else
        warn "No kicadLibrary (dir or tarball) found — skipping symbol library"
        return 0
    fi

    # The 3 libraries eSim rewrites at runtime when users build HDL models.
    local gen_libs=(eSim_Ngveri eSim_NgVeriCosim eSim_Nghdl) g

    # --- Static libs: the 14 eSim never rewrites go into KiCad's own dir and
    #     stay ROOT-OWNED. No chown here: the old chown -R of the whole symbols
    #     dir hijacked ownership of KiCad's standard libraries.
    # --chown/--chmod: plain `sudo rsync -a` preserves the SOURCE owner/mode,
    # i.e. the user's repo checkout — leaving user-owned, group-writable files
    # in /usr/share. Force the root-owned 644 a system dir is supposed to have.
    sudo mkdir -p /usr/share/kicad/symbols
    sudo rsync -a --chown=root:root --chmod=D755,F644 \
        --exclude='eSim_Ngveri.kicad_sym' \
        --exclude='eSim_NgVeriCosim.kicad_sym' \
        --exclude='eSim_Nghdl.kicad_sym' \
        "$libdir/eSim-symbols/" /usr/share/kicad/symbols/

    # --- Generated libs: live in the user's own ~/.esim/kicad_symbols so the
    #     app never needs write access to /usr/share. Seed each ONCE ( -n ) —
    #     a reinstall must not clobber a user's accumulated models.
    local gendir="$config_dir/kicad_symbols"
    mkdir -p "$gendir"
    for g in "${gen_libs[@]}"; do
        cp -n "$libdir/eSim-symbols/$g.kicad_sym" "$gendir/" 2>/dev/null || true
    done

    # Seed sym-lib-table into the KiCad per-user config dir that actually exists
    # (do NOT hard-code 6.0 — the old scripts did and broke under KiCad 8/9).
    local cfg="$HOME/.config/kicad" ver
    ver=$(ls "$cfg" 2>/dev/null | grep -E '^[0-9]+\.[0-9]+$' | sort -V | tail -n 1)
    if [ -z "$ver" ]; then
        local major
        major=$(dpkg-query -W -f='${Version}' kicad 2>/dev/null | grep -oP '^\d+')
        ver="${major:-9}.0"
    fi
    mkdir -p "$cfg/$ver"
    log "Using KiCad config dir: $cfg/$ver"

    # Rewrite the 3 generated-lib uris from ${KICAD6_SYMBOL_DIR}/<lib> to their
    # ~/.esim absolute path on a TEMP copy of the template (never modify the
    # file inside $libdir), then install it.
    local tmptable
    tmptable=$(mktemp)
    cp "$libdir/template/sym-lib-table" "$tmptable"
    for g in "${gen_libs[@]}"; do
        sed -i "s|\${KICAD6_SYMBOL_DIR}/$g.kicad_sym|$gendir/$g.kicad_sym|g" \
            "$tmptable"
    done
    cp "$tmptable" "$cfg/$ver/sym-lib-table" 2>/dev/null \
        || warn "sym-lib-table copy failed (eSim registers libs at runtime anyway)"
    rm -f "$tmptable"

    log "Static eSim symbols -> /usr/share/kicad/symbols (root-owned)"
    log "Generated symbols   -> $gendir (user-writable)"

    # The extracted copy is only needed during install.
    [ -f "$eSim_Home/library/kicadLibrary.tar.xz" ] && rm -rf "$libdir"
    return 0
}

installNghdl() {
    log "Installing NGHDL (GHDL/Verilator co-simulation)"

    # Release ships nghdl as nghdl.zip; dev repo ships it as a nghdl/ dir.
    if [ -f "$eSim_Home/nghdl.zip" ] && [ ! -d "$eSim_Home/nghdl" ]; then
        unzip -o "$eSim_Home/nghdl.zip" -d "$eSim_Home" >/dev/null
    fi

    # NGHDL is NOT optional: it is the only step that provides ngspice (no
    # apt line in this script installs a simulator), so an eSim without it
    # launches and draws schematics but cannot simulate ANYTHING. It used to
    # be treated as a skippable extra — which shipped exactly that broken
    # install, with a success banner and exit 0. Fail loudly instead.
    local nd="$eSim_Home/nghdl"
    if [ ! -d "$nd" ]; then
        die "nghdl not found (no nghdl/ dir, no nghdl.zip) in $eSim_Home.
       NGHDL provides ngspice itself — without it eSim cannot simulate.
       The download/zip is incomplete; re-download and re-run."
    fi
    if [ ! -f "$nd/nghdl-simulator-source.tar.xz" ]; then
        die "nghdl/nghdl-simulator-source.tar.xz missing from $eSim_Home.
       NGHDL provides ngspice itself — without it eSim cannot simulate.
       The download/zip is incomplete; re-download and re-run."
    fi

    chmod +x "$nd/install-nghdl.sh" 2>/dev/null || true

    # Run without the ERR trap so a failure reaches the clear message below
    # instead of the generic abort text.
    set +e; trap "" ERR
    ( cd "$nd" && ./install-nghdl.sh --install )
    local rc=$?
    set -e; trap error_exit ERR

    if [ $rc -eq 0 ]; then
        log "NGHDL installed"
    else
        die "NGHDL install failed (exit $rc).
       NGHDL provides ngspice — without it eSim cannot run any simulation,
       so this install is aborted rather than shipped broken. Fix the error
       above (usually disk space or network), then re-run
       ./install-eSim.sh --install — re-running is safe and keeps everything
       already installed."
    fi
}

installSky130Pdk() {
    log "Installing SKY130 PDK"
    local t="$eSim_Home/library/sky130_fd_pr.tar.xz"
    if [ ! -f "$t" ]; then
        warn "library/sky130_fd_pr.tar.xz missing — skipping SKY130 PDK"
        return 0
    fi
    sudo rm -rf /usr/share/local/sky130_fd_pr
    rm -rf "$eSim_Home/library/sky130_fd_pr"
    tar -xJf "$t" -C "$eSim_Home/library"
    sudo mkdir -p /usr/share/local
    sudo mv "$eSim_Home/library/sky130_fd_pr" /usr/share/local/
    sudo chown -R "$USER:$USER" /usr/share/local/sky130_fd_pr
}

installIhpPdk() {
    local script="$eSim_Home/ihp/ihp-install-script.sh"
    [ -f "$script" ] || { warn "IHP install script not found — skipping"; return 0; }

    read -rp "Install IHP Open PDK for analog IC design? (y/n): " ans
    if [[ ! "$ans" =~ ^[Yy] ]]; then
        log "Skipping IHP Open PDK"
        return 0
    fi
    chmod +x "$script"
    set +e; trap "" ERR
    ( cd "$eSim_Home/ihp" && ./ihp-install-script.sh --install )
    set -e; trap error_exit ERR
}

createDesktopStartScript() {
    log "Creating launcher (esim) + desktop entry"

    # The application anchors all resources to __file__ (configuration.paths),
    # so the launcher no longer needs to cd into src/frontEnd or source the
    # venv — invoking the venv's python directly is equivalent and quoting-safe.
    #
    # Default to software OpenGL (llvmpipe). Inside VirtualBox / headless / broken
    # -GPU-driver setups Mesa's hardware path fails with
    #   libEGL warning: failed to get driver name for fd -1
    #   MESA: error: ZINK: failed to choose pdev / egl: failed to create dri2 screen
    # eSim is a Qt Widgets + matplotlib app that needs no GPU, so forcing the
    # software path is free on real hardware and removes those errors in VMs.
    # All three vars are defaulted-not-forced (`:=`), so a user can still override.
    {
        echo '#!/bin/bash'
        echo ': "${LIBGL_ALWAYS_SOFTWARE:=1}"; export LIBGL_ALWAYS_SOFTWARE'
        echo ': "${QT_OPENGL:=software}"; export QT_OPENGL'
        echo ': "${EGL_LOG_LEVEL:=fatal}"; export EGL_LOG_LEVEL'
        printf 'exec "%s/env/bin/python3" "%s/src/frontEnd/Application.py" "$@"\n' \
            "$config_dir" "$eSim_Home"
    } > esim-start.sh
    sudo chmod 755 esim-start.sh
    sudo cp -p esim-start.sh /usr/bin/esim
    rm -f esim-start.sh

    cat > esim.desktop << EOF
[Desktop Entry]
Version=1.0
Name=eSim
Comment=EDA Tool
GenericName=eSim
Keywords=eda-tools
Exec=esim %u
Terminal=true
X-MultipleArgs=false
Type=Application
Icon=$config_dir/logo.png
Categories=Development;
StartupNotify=true
EOF
    sudo chmod 755 esim.desktop
    sudo cp -p esim.desktop /usr/share/applications/
    mkdir -p "$HOME/Desktop"
    cp -p esim.desktop "$HOME/Desktop/"

    set +e; trap "" ERR
    gio set "$HOME/Desktop/esim.desktop" "metadata::trusted" true
    chmod a+x "$HOME/Desktop/esim.desktop"
    rm -f esim.desktop
    set -e; trap error_exit ERR

    cp -p "$eSim_Home/images/logo.png" "$config_dir" 2>/dev/null || true
}

#-----------------------------------------------------------------------------
# Uninstall (version-agnostic, idempotent, best-effort throughout)
#-----------------------------------------------------------------------------
uninstall_eSim() {
    log "Removing eSim application files"
    # ~/.esim now also holds the generated KiCad symbol libs (kicad_symbols/),
    # so this one removal covers them too.
    sudo rm -rf "$HOME/.esim" "$HOME/Desktop/esim.desktop" \
                /usr/bin/esim /usr/share/applications/esim.desktop

    log "Removing KiCad (any version) + eSim symbols"
    sudo apt-get purge -y kicad kicad-footprints kicad-libraries kicad-symbols kicad-templates 2>/dev/null || true
    sudo apt-get autoremove -y 2>/dev/null || true
    # Remove ONLY eSim's own files. /usr/share/kicad belongs to the kicad apt
    # package (purged above); never rm -rf the whole dir — that nuked KiCad's
    # standard libraries. Generated symbol libs now live under ~/.esim and are
    # already cleared by the "$HOME/.esim" removal at the top of this function.
    sudo rm -f /usr/share/kicad/symbols/eSim_*.kicad_sym \
               /usr/share/kicad/symbols/eSim_*.lib 2>/dev/null || true
    # Drop the now-empty dirs our mkdir -p created; --ignore-fail-on-non-empty
    # keeps this a no-op if KiCad (or anything else) still owns files there.
    sudo rmdir --ignore-fail-on-non-empty /usr/share/kicad/symbols \
               /usr/share/kicad 2>/dev/null || true
    sudo rm -f /etc/apt/sources.list.d/kicad* 2>/dev/null || true

    log "Removing SKY130 PDK"
    sudo rm -rf /usr/share/local/sky130_fd_pr

    log "Removing NGHDL"
    if [ -f "$eSim_Home/nghdl/install-nghdl.sh" ]; then
        ( cd "$eSim_Home/nghdl" && chmod +x install-nghdl.sh && ./install-nghdl.sh --uninstall ) || true
    else
        # nghdl never extracted, or release form — remove known artifacts.
        sudo rm -rf "$HOME/nghdl-simulator" "$HOME/.nghdl" \
                    /usr/local/bin/nghdl /usr/bin/ngspice 2>/dev/null || true
    fi

    log "Removing IHP Open PDK (if present)"
    if [ -f "$eSim_Home/ihp/ihp-install-script.sh" ]; then
        ( cd "$eSim_Home/ihp" && chmod +x ihp-install-script.sh && ./ihp-install-script.sh --uninstall ) || true
    fi

    # Clear runtime-generated NGHDL/NgVeri model XML so a later reinstall is clean.
    rm -rf "$eSim_Home"/library/modelParamXML/Nghdl/* \
           "$eSim_Home"/library/modelParamXML/Ngveri/* 2>/dev/null || true

    log "eSim uninstalled."
}

#-----------------------------------------------------------------------------
# Post-install self-check: the same toolchain doctor the app ships
# (Help menu / `esim --doctor`). Non-fatal: a red row here is exactly the
# actionable report we want the user to see, not an abort.
#-----------------------------------------------------------------------------
runToolchainDoctor() {
    log "Running the simulation-toolchain doctor (esim --doctor)"
    set +e; trap "" ERR
    "$config_dir/env/bin/python3" "$eSim_Home/src/frontEnd/Application.py" --doctor
    local rc=$?
    set -e; trap error_exit ERR
    if [ $rc -ne 0 ]; then
        warn "Toolchain doctor reported missing pieces (see report above)."
        warn "eSim will run; the affected flows will explain what to fix."
    fi
}

#-----------------------------------------------------------------------------
# Proxy prompt (optional)
#-----------------------------------------------------------------------------
setupProxy() {
    read -rp "Is your internet connection behind a proxy? (y/n): " getProxy
    if [[ "$getProxy" =~ ^[Yy] ]]; then
        read -rp  "Proxy hostname: " proxyHostname
        read -rp  "Proxy port: "     proxyPort
        read -rp  "Username: "       proxyUser
        read -rsp "Password: "       proxyPass; echo
        local url="http://$proxyUser:$proxyPass@$proxyHostname:$proxyPort"
        export http_proxy="$url" https_proxy="$url" ftp_proxy="$url"
        export HTTP_PROXY="$url" HTTPS_PROXY="$url" FTP_PROXY="$url"
        log "Proxy configured"
    else
        log "Installing without proxy"
    fi
}

#-----------------------------------------------------------------------------
# Dry-run preview
#-----------------------------------------------------------------------------
print_plan() {
    cat << EOF

================ eSim install plan (dry run) ================
 eSim root        : $eSim_Home
 Ubuntu version   : $UBUNTU_VER
 KiCad source     : $KICAD_SOURCE ${KICAD_PPA:+(ppa:$KICAD_PPA)}  min major: $KICAD_MIN_MAJOR
 Qt packages      : $QT_PKGS $QSCI_PKG
 Virtualenv       : $config_dir/env (--system-site-packages)
 nghdl input      : $( [ -d "$eSim_Home/nghdl" ] && echo "nghdl/ dir" || ( [ -f "$eSim_Home/nghdl.zip" ] && echo "nghdl.zip" || echo "MISSING" ) )
 kicadLibrary     : $( [ -d "$eSim_Home/library/kicadLibrary" ] && echo "dir" || ( [ -f "$eSim_Home/library/kicadLibrary.tar.xz" ] && echo "tarball" || echo "MISSING" ) )
 sky130 PDK       : $( [ -f "$eSim_Home/library/sky130_fd_pr.tar.xz" ] && echo "present" || echo "MISSING" )
 kicad symbols    : 14 static -> /usr/share/kicad/symbols (root); 3 generated -> $config_dir/kicad_symbols (user)
 Steps            : config -> deps+venv -> PyQt6+QScintilla -> KiCad -> kicadLib
                    -> nghdl(required: provides ngspice) -> sky130 -> ihp(prompt) -> launcher
=============================================================

EOF
}

#=============================================================================
# Main
#=============================================================================
if [ "$#" -ne 1 ]; then
    echo "USAGE:"
    echo "  ./install-eSim.sh --install"
    echo "  ./install-eSim.sh --uninstall"
    echo "  ./install-eSim.sh --dry-run"
    exit 1
fi
option="$1"

resolve_esim_home
detect_profile

case "$option" in
    --dry-run)
        print_plan
        ;;

    --install)
        # Tee everything to a log so a failed install is fully diagnosable
        # (handy when iterating on fresh VMs).
        exec > >(tee "$HOME/eSim-install.log") 2>&1
        echo ">>> Logging to $HOME/eSim-install.log"
        set -e; set -E; trap error_exit ERR
        preflightDisk
        setupProxy
        cleanLegacyEsim
        createConfigFile
        installDependency
        installQt
        installKicad
        copyKicadLibrary
        installNghdl
        installSky130Pdk
        installIhpPdk
        createDesktopStartScript
        runToolchainDoctor
        echo
        echo "----------------- eSim installed successfully -----------------"
        echo 'Type "esim" in a terminal to launch it, or use the desktop icon.'
        ;;

    --uninstall)
        read -rp "This removes eSim, KiCad, NGHDL, SKY130 PDK and their models. Continue? (y/n): " c
        if [[ "$c" =~ ^[Yy] ]]; then
            uninstall_eSim
        else
            log "Uninstall cancelled."
        fi
        ;;

    *)
        echo "Invalid argument: $option"
        echo "Usage: $0 --install | --uninstall | --dry-run"
        exit 1
        ;;
esac
