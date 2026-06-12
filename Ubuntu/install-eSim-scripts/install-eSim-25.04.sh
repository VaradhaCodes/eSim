#!/bin/bash
#=============================================================================
#          FILE: install-esim-ubuntu.sh
#
#         USAGE: ./install-esim-ubuntu.sh --install
#                ./install-esim-ubuntu.sh --uninstall
#
#   DESCRIPTION: Standalone installer for eSim master branch on Ubuntu.
#                Clones eSim from GitHub, installs all dependencies, sets
#                up KiCad 8, SKY130 PDK, NGHDL tools, and creates a desktop
#                launcher. Supports Ubuntu 22.04, 23.04, 24.04, and 25.xx.
#
#  REQUIREMENTS: Ubuntu 22.04+ (tested on 25.04), internet connection, sudo
#        AUTHOR: Improved by community for Ubuntu 25.xx + eSim master
#      REVISION: 2026-06-12
#=============================================================================

set -eE
set -o pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
ESIM_REPO="https://github.com/FOSSEE/eSim.git"
ESIM_BRANCH="master"
ESIM_INSTALL_DIR="$HOME/eSim"
CONFIG_DIR="$HOME/.esim"
CONFIG_FILE="config.ini"
NGHDL_CONFIG_DIR="$HOME/.nghdl"

# ─── Colour helpers ───────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()     { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ─── Error trap ───────────────────────────────────────────────────────────────
error_exit() {
    echo -e "\n${RED}Installation failed.${NC} Resolve the error above and re-run.\n"
    exit 1
}
trap error_exit ERR

# ─── Ubuntu version detection ────────────────────────────────────────────────
detect_ubuntu_version() {
    if [ ! -f /etc/os-release ]; then
        die "Cannot detect OS. /etc/os-release not found."
    fi
    . /etc/os-release
    if [ "$ID" != "ubuntu" ]; then
        die "This script is for Ubuntu only. Detected: $ID"
    fi
    UBUNTU_VERSION="$VERSION_ID"
    UBUNTU_MAJOR="${VERSION_ID%%.*}"
    UBUNTU_MINOR="${VERSION_ID##*.}"
    info "Detected Ubuntu $UBUNTU_VERSION ($VERSION_CODENAME)"
}

check_ubuntu_supported() {
    # Supported: 22.04, 23.04, 24.04, 25.xx and later
    case "$UBUNTU_VERSION" in
        22.04|23.04|24.04) ;;
        *)
            if [ "$UBUNTU_MAJOR" -ge 25 ] 2>/dev/null; then
                : # supported
            else
                die "Unsupported Ubuntu version: $UBUNTU_VERSION. Supported: 22.04, 23.04, 24.04, 25.xx+"
            fi
            ;;
    esac
}

# ─── Prerequisite tools check ────────────────────────────────────────────────
check_prerequisites() {
    local missing=()
    for cmd in git curl sudo python3; do
        command -v "$cmd" &>/dev/null || missing+=("$cmd")
    done
    if [ ${#missing[@]} -gt 0 ]; then
        info "Installing missing prerequisites: ${missing[*]}"
        sudo apt-get update -qq
        sudo apt-get install -y "${missing[@]}"
    fi
}

# ─── Clone eSim master ────────────────────────────────────────────────────────
clone_esim() {
    if [ -d "$ESIM_INSTALL_DIR/.git" ]; then
        info "eSim directory already exists at $ESIM_INSTALL_DIR — pulling latest master."
        git -C "$ESIM_INSTALL_DIR" fetch origin
        git -C "$ESIM_INSTALL_DIR" checkout "$ESIM_BRANCH"
        git -C "$ESIM_INSTALL_DIR" pull origin "$ESIM_BRANCH"
    else
        info "Cloning eSim $ESIM_BRANCH branch to $ESIM_INSTALL_DIR ..."
        git clone --branch "$ESIM_BRANCH" --depth 1 "$ESIM_REPO" "$ESIM_INSTALL_DIR"
    fi
    success "eSim source ready at $ESIM_INSTALL_DIR"
}

# ─── Config file ─────────────────────────────────────────────────────────────
create_esim_config() {
    info "Creating eSim config file..."
    mkdir -p "$CONFIG_DIR"
    cat > "$CONFIG_DIR/$CONFIG_FILE" <<EOF
[eSim]
eSim_HOME = $ESIM_INSTALL_DIR
LICENSE = %(eSim_HOME)s/LICENSE
KicadLib = %(eSim_HOME)s/library/kicadLibrary.tar.xz
IMAGES = %(eSim_HOME)s/images
VERSION = %(eSim_HOME)s/VERSION
MODELICA_MAP_JSON = %(eSim_HOME)s/library/ngspicetoModelica/Mapping.json
EOF
    success "Config written to $CONFIG_DIR/$CONFIG_FILE"
}

# ─── APT dependencies ─────────────────────────────────────────────────────────
install_apt_deps() {
    info "Updating apt index..."
    set +e; sudo apt-get update; set -e

    info "Installing system packages..."
    sudo apt-get install -y \
        python3-venv \
        python3-pip \
        python3-pyqt6 \
        python3-matplotlib \
        python3-setuptools \
        python3-psutil \
        xterm \
        unzip \
        rsync \
        libcanberra-gtk3-module

    success "APT dependencies installed."
}

# ─── Python virtual environment ───────────────────────────────────────────────
setup_virtualenv() {
    info "Creating Python virtual environment at $CONFIG_DIR/env ..."
    rm -rf "$CONFIG_DIR/env"
    python3 -m venv --system-site-packages "$CONFIG_DIR/env"

    # shellcheck source=/dev/null
    source "$CONFIG_DIR/env/bin/activate"

    info "Upgrading pip inside venv..."
    pip install --upgrade pip --quiet

    info "Installing Python packages from requirements.txt ..."
    # Use requirements.txt from master but allow pip to pick compatible versions
    # for Python 3.13 (pinned old versions in requirements.txt lack py3.13 wheels)
    pip install --upgrade \
        "PyQt6>=6.5.0" \
        "PyQt6-Qt6>=6.5.0" \
        "PyQt6-sip>=13.5.0" \
        "matplotlib>=3.8.0" \
        "numpy>=1.26.0" \
        "scipy>=1.11.0" \
        "watchdog>=4.0.0" \
        "makerchip-app" \
        "sandpiper-saas" \
        "pillow>=10.0.0" \
        "packaging" \
        "sphinx>=5.0.0" \
        "sphinx_rtd_theme>=1.0.0" || warn "Some pip packages failed — eSim core will still work."

    # hdlparse 1.0.4 on PyPI uses deprecated use_2to3; install from GitHub fork that fixes it
    pip install --upgrade "git+https://github.com/hdl/pyhdlparser.git" || \
        warn "hdlparse install failed — NgVeri VHDL parsing will not be available."

    # volare: PDK version manager used for SKY130 installation
    pip install volare || warn "volare install failed — SKY130 PDK may need manual setup."

    deactivate
    success "Python virtual environment ready."
}

# ─── KiCad 8 installation ─────────────────────────────────────────────────────
install_kicad() {
    info "Checking KiCad installation..."

    if dpkg -s kicad &>/dev/null; then
        installed_ver=$(dpkg-query -W -f='${Version}' kicad | cut -d'.' -f1)
        if [ "$installed_ver" = "8" ]; then
            success "KiCad 8 already installed — skipping."
            return
        else
            warn "KiCad $installed_ver is installed. eSim needs KiCad 8."
            read -rp "Remove KiCad $installed_ver and install KiCad 8? (y/N): " ans
            [[ "$ans" =~ ^[Yy]$ ]] || die "Aborting: KiCad 8 is required."
            sudo apt-get remove --purge -y kicad kicad-footprints kicad-libraries \
                kicad-symbols kicad-templates || true
            sudo apt-get autoremove -y
        fi
    fi

    # On Ubuntu 25.04+ (plucky/oracular), KiCad 8 is in universe — no PPA needed.
    # On 22.04/23.04/24.04 we add the official KiCad PPA.
    if [ "$UBUNTU_MAJOR" -lt 25 ] 2>/dev/null; then
        local ppa="kicad/kicad-8.0-releases"
        if ! grep -qr "^deb.*$ppa" /etc/apt/sources.list /etc/apt/sources.list.d/ 2>/dev/null; then
            info "Adding KiCad 8.0 PPA for Ubuntu $UBUNTU_VERSION ..."
            sudo add-apt-repository -y "ppa:$ppa"
            sudo apt-get update -qq
        fi
    fi

    info "Installing KiCad 8..."
    sudo apt-get install -y --no-install-recommends \
        kicad kicad-footprints kicad-libraries kicad-symbols kicad-templates
    success "KiCad installed."
}

# ─── KiCad library setup ──────────────────────────────────────────────────────
install_kicad_library() {
    # In the master branch the kicadLibrary/ directory is already checked in;
    # there is no tarball to extract. Support both cases.
    local lib_dir="$ESIM_INSTALL_DIR/library/kicadLibrary"
    local lib_archive="$ESIM_INSTALL_DIR/library/kicadLibrary.tar.xz"

    if [ ! -d "$lib_dir" ]; then
        if [ -f "$lib_archive" ]; then
            info "Extracting KiCad library from tarball..."
            tar -xJf "$lib_archive" -C "$ESIM_INSTALL_DIR/library"
        else
            die "KiCad library not found (expected $lib_dir or $lib_archive)"
        fi
    fi

    info "Installing KiCad library..."

    # Detect the KiCad config version directory (8.0, 7.0, 6.0 …)
    local kicad_cfg="$HOME/.config/kicad"
    mkdir -p "$kicad_cfg"
    local kicad_ver
    kicad_ver=$(ls "$kicad_cfg" 2>/dev/null | grep -E '^[0-9]+\.[0-9]+$' | sort -V | tail -n1)
    if [ -z "$kicad_ver" ]; then
        kicad_ver="8.0"
        mkdir -p "$kicad_cfg/$kicad_ver"
        info "Created KiCad config dir: $kicad_cfg/$kicad_ver"
    fi
    info "Using KiCad config version: $kicad_ver"

    # Copy sym-lib-table
    if [ -f "$lib_dir/template/sym-lib-table" ]; then
        cp "$lib_dir/template/sym-lib-table" "$kicad_cfg/$kicad_ver/"
        success "Copied sym-lib-table to $kicad_cfg/$kicad_ver/"
    fi

    # Copy eSim custom symbols
    local sym_dir="/usr/share/kicad/symbols"
    sudo mkdir -p "$sym_dir"
    sudo rsync -a "$lib_dir/eSim-symbols/" "$sym_dir/"
    sudo chown -R "$USER:$USER" "$sym_dir"
    success "Copied eSim symbols to $sym_dir"
}

# ─── SKY130 PDK (via volare — matches 24.04 installer approach) ──────────────
# Pinned commit: same version used by eSim 2.5 / 24.04 official installer
SKY130_COMMIT="0fe599b2afb6708d281543108caf8310912f54af"

install_sky130_pdk() {
    info "Installing SKY130 PDK via volare..."
    sudo rm -rf /usr/share/local/sky130_fd_pr
    sudo mkdir -p /usr/share/local

    # Run volare inside the eSim venv (it was installed there)
    source "$CONFIG_DIR/env/bin/activate"
    set +e
    volare enable --pdk sky130 --pdk-root /usr/share/local/ "$SKY130_COMMIT"
    local rc=$?
    set -e
    deactivate

    if [ $rc -ne 0 ]; then
        warn "volare download failed — falling back to bundled tarball."
        local pdk_archive="$ESIM_INSTALL_DIR/library/sky130_fd_pr.tar.xz"
        if [ -f "$pdk_archive" ]; then
            local tmp_dir; tmp_dir=$(mktemp -d)
            tar -xJf "$pdk_archive" -C "$tmp_dir"
            sudo mv "$tmp_dir/sky130_fd_pr" /usr/share/local/
            rm -rf "$tmp_dir"
        else
            warn "No SKY130 tarball found either — skipping SKY130 PDK."
            return
        fi
    else
        # Move from volare's cache to the expected path and clean up
        local volare_path="/usr/share/local/volare/sky130/versions/$SKY130_COMMIT/sky130A/libs.ref/sky130_fd_pr"
        [ -d "$volare_path" ] && sudo mv "$volare_path" /usr/share/local/
        sudo rm -rf /usr/share/local/volare
    fi

    sudo chown -R "$USER:$USER" /usr/share/local/sky130_fd_pr/ 2>/dev/null || true
    success "SKY130 PDK installed to /usr/share/local/sky130_fd_pr/"
}

# ─── IHP Open PDK (optional) ──────────────────────────────────────────────────
install_ihp_pdk() {
    read -rp "Install IHP Open PDK for analog IC design? (y/N): " ans
    [[ "$ans" =~ ^[Yy]$ ]] || { info "Skipping IHP PDK."; return; }

    local ihp_script="$ESIM_INSTALL_DIR/ihp/ihp-install-script.sh"
    if [ -f "$ihp_script" ]; then
        info "Installing IHP Open PDK..."
        chmod +x "$ihp_script"
        set +e; trap "" ERR
        bash "$ihp_script" --install
        trap error_exit ERR; set -e
    else
        warn "IHP install script not found ($ihp_script) — skipping."
    fi
}

# ─── NGHDL ────────────────────────────────────────────────────────────────────
install_nghdl() {
    info "Installing NGHDL dependencies (ghdl, verilator, build tools)..."
    # Use apt ghdl + verilator — avoids building from source tarballs entirely.
    # The custom nghdl-simulator (patched ngspice) still needs a source build.
    sudo apt-get install -y \
        ghdl verilator \
        make autoconf g++ flex bison \
        zlib1g-dev libxaw7 libxaw7-dev

    # Write ~/.nghdl/config.ini (updated by build_nghdl_from_source if tarballs present)
    mkdir -p "$NGHDL_CONFIG_DIR"
    cat > "$NGHDL_CONFIG_DIR/$CONFIG_FILE" <<EOF
[NGHDL]
NGHDL_HOME = $ESIM_INSTALL_DIR/nghdl
DIGITAL_MODEL = /usr/lib/ngspice
RELEASE = /usr/bin
[SRC]
SRC_HOME = $ESIM_INSTALL_DIR/nghdl/src
LICENSE = $ESIM_INSTALL_DIR/nghdl/LICENSE
EOF

    local nghdl_py="$ESIM_INSTALL_DIR/nghdl/src/ngspice_ghdl.py"
    if [ -f "$nghdl_py" ]; then
        sudo chmod 755 "$nghdl_py"
        sudo ln -sf "$nghdl_py" /usr/local/bin/nghdl
    fi

    # Build custom nghdl-simulator if tarballs are present.
    # Tarballs are NOT in git — get them from the full eSim release zip and
    # drop them into $ESIM_INSTALL_DIR/nghdl/src/, then re-run --install.
    local src_dir="$ESIM_INSTALL_DIR/nghdl/src"
    if [ -f "$src_dir/nghdl-simulator-source.tar.xz" ]; then
        build_nghdl_from_source "$src_dir"
    else
        sudo apt-get install -y ngspice
        warn "nghdl-simulator tarballs not found — using system ngspice."
        warn "Drop nghdl-simulator-source.tar.xz into $src_dir and re-run for full NGHDL."
    fi

    success "NGHDL installed."
}

build_nghdl_from_source() {
    local src_dir="$1"
    info "Building nghdl-simulator (custom ngspice) from source..."

    tar -xJf "$src_dir/nghdl-simulator-source.tar.xz" -C "$HOME"
    local nghdl_home="$HOME/nghdl-simulator"
    [ -d "$HOME/nghdl-simulator-source" ] && mv "$HOME/nghdl-simulator-source" "$nghdl_home"

    mkdir -p "$nghdl_home/install_dir" "$nghdl_home/release"
    pushd "$nghdl_home/release" > /dev/null
    chmod +x ../configure
    # ngspice source has 'typedef int bool' which is illegal in C23 (GCC 15+
    # default). Force C11 so the typedef is valid. One flag, no other changes.
    ../configure \
        --enable-xspice --disable-debug \
        --prefix="$nghdl_home/install_dir/" \
        --exec-prefix="$nghdl_home/install_dir/" \
        CFLAGS="-std=gnu11"
    make -j"$(nproc)"
    make install
    popd > /dev/null

    sudo chmod 755 "$nghdl_home/install_dir/bin/ngspice"
    set +e; sudo apt-get purge -y ngspice 2>/dev/null; sudo rm -f /usr/bin/ngspice; set -e
    sudo ln -sf "$nghdl_home/install_dir/bin/ngspice" /usr/bin/ngspice

    cat > "$NGHDL_CONFIG_DIR/$CONFIG_FILE" <<EOF
[NGHDL]
NGHDL_HOME = $nghdl_home
DIGITAL_MODEL = %(NGHDL_HOME)s/src/xspice/icm
RELEASE = %(NGHDL_HOME)s/release
[SRC]
SRC_HOME = $src_dir
LICENSE = $src_dir/../LICENSE
EOF
    success "nghdl-simulator built and installed."
}

# ─── Desktop launcher ─────────────────────────────────────────────────────────
create_launcher() {
    info "Creating eSim launcher..."

    # esim wrapper script
    local launcher
    launcher=$(mktemp)
    cat > "$launcher" <<EOF
#!/bin/bash
cd "$ESIM_INSTALL_DIR/src/frontEnd"
source "$CONFIG_DIR/env/bin/activate"
exec python3 Application.py "\$@"
EOF
    sudo install -m 755 "$launcher" /usr/bin/esim
    rm "$launcher"

    # Copy logo
    [ -f "$ESIM_INSTALL_DIR/images/logo.png" ] && \
        cp "$ESIM_INSTALL_DIR/images/logo.png" "$CONFIG_DIR/"

    # .desktop file
    local desktop
    desktop=$(mktemp)
    cat > "$desktop" <<EOF
[Desktop Entry]
Version=1.0
Name=eSim
Comment=EDA Tool — Free/Open Source Electronic Design Automation
GenericName=eSim
Keywords=eda-tools;spice;kicad;simulation
Exec=esim %u
Terminal=true
Type=Application
Icon=$CONFIG_DIR/logo.png
Categories=Development;Electronics;
StartupNotify=true
EOF
    sudo install -m 644 "$desktop" /usr/share/applications/esim.desktop

    # Copy to Desktop only if the directory exists
    if [ -d "$HOME/Desktop" ]; then
        install -m 755 "$desktop" "$HOME/Desktop/esim.desktop"
        set +e
        gio set "$HOME/Desktop/esim.desktop" "metadata::trusted" true 2>/dev/null
        chmod a+x "$HOME/Desktop/esim.desktop" 2>/dev/null
        set -e
    fi
    rm "$desktop"
    success "Launcher installed: run 'esim' in a terminal or use the desktop icon."
}

# ─── Install ──────────────────────────────────────────────────────────────────
do_install() {
    detect_ubuntu_version
    check_ubuntu_supported
    check_prerequisites
    clone_esim

    # From here on we work relative to the cloned repo
    cd "$ESIM_INSTALL_DIR"

    install_apt_deps
    create_esim_config
    setup_virtualenv
    install_kicad
    install_kicad_library
    install_nghdl
    install_sky130_pdk
    install_ihp_pdk
    create_launcher

    echo ""
    echo -e "${GREEN}══════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  eSim master installed successfully!              ${NC}"
    echo -e "${GREEN}══════════════════════════════════════════════════${NC}"
    echo ""
    echo "  Run:  esim"
    echo "  Or double-click the eSim icon on your desktop."
    echo ""
    echo "  eSim home: $ESIM_INSTALL_DIR"
    echo "  Config:    $CONFIG_DIR"
    echo ""
    echo -e "${YELLOW}Note:${NC} Full NGHDL digital co-simulation (VHDL inside ngspice)"
    echo "      requires building the custom nghdl-simulator from source."
    echo "      System ngspice + ghdl are installed for basic support."
}

# ─── Uninstall ────────────────────────────────────────────────────────────────
do_uninstall() {
    echo -n "Remove eSim completely (including KiCad, SKY130 PDK, NGHDL tools)? (y/N): "
    read -r ans
    [[ "$ans" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }

    info "Removing eSim files..."
    sudo rm -f /usr/bin/esim /usr/local/bin/nghdl \
               /usr/share/applications/esim.desktop
    rm -f "$HOME/Desktop/esim.desktop"
    rm -rf "$CONFIG_DIR" "$NGHDL_CONFIG_DIR"

    info "Removing KiCad..."
    sudo apt-get purge -y kicad kicad-footprints kicad-libraries \
        kicad-symbols kicad-templates || true
    sudo rm -rf /usr/share/kicad
    sudo rm -f /etc/apt/sources.list.d/kicad*.list \
               /etc/apt/sources.list.d/kicad*.sources 2>/dev/null || true
    sudo apt-get autoremove -y

    info "Removing SKY130 PDK..."
    sudo rm -rf /usr/share/local/sky130_fd_pr

    info "Removing NGHDL tools..."
    sudo apt-get purge -y ngspice ghdl verilator libxaw7-dev \
        gnat llvm llvm-dev || true
    sudo apt-get autoremove -y

    warn "The eSim source directory ($ESIM_INSTALL_DIR) was NOT removed."
    echo  "Remove it manually if desired:  rm -rf $ESIM_INSTALL_DIR"
    success "eSim uninstalled."
}

# ─── Entry point ─────────────────────────────────────────────────────────────
case "${1:-}" in
    --install)   do_install   ;;
    --uninstall) do_uninstall ;;
    --help|-h)
        echo "Usage: $0 --install | --uninstall"
        echo ""
        echo "  --install    Clone eSim master and install everything"
        echo "  --uninstall  Remove eSim and all its dependencies"
        ;;
    *)
        echo "Usage: $0 --install | --uninstall"
        exit 1
        ;;
esac
