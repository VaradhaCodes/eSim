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
QT_PKGS="python3-pyqt6 pyqt6-dev-tools"
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
    sudo mkdir -p /usr/share/kicad/symbols
    sudo rsync -a \
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

    local nd="$eSim_Home/nghdl"
    if [ ! -d "$nd" ]; then
        warn "nghdl not found (no nghdl/ dir, no nghdl.zip) — skipping co-simulation"
        return 0
    fi
    if [ ! -f "$nd/nghdl-simulator-source.tar.xz" ]; then
        warn "nghdl/nghdl-simulator-source.tar.xz missing — skipping co-simulation"
        return 0
    fi

    chmod +x "$nd/install-nghdl.sh" 2>/dev/null || true

    # NGHDL is optional: never abort the whole eSim install if it fails.
    set +e; trap "" ERR
    ( cd "$nd" && ./install-nghdl.sh --install )
    local rc=$?
    set -e; trap error_exit ERR

    if [ $rc -eq 0 ]; then
        log "NGHDL installed"
    else
        warn "NGHDL install failed (exit $rc) — GHDL co-simulation unavailable; eSim still works"
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
    printf '#!/bin/bash\nexec "%s/env/bin/python3" "%s/src/frontEnd/Application.py" "$@"\n' \
        "$config_dir" "$eSim_Home" > esim-start.sh
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
                    -> nghdl(optional) -> sky130 -> ihp(prompt) -> launcher
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
        setupProxy
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
