#!/bin/bash
#==========================================================
#          FILE: install-nghdl.sh
#
#         USAGE: ./install-nghdl.sh --install
#                ./install-nghdl.sh --uninstall
#
#   DESCRIPTION: Unified installer for NGHDL — the eSim custom ngspice
#                (nghdl-simulator) plus GHDL, Verilator and Icarus Verilog
#                (with libvvp, for d_cosim), for Ubuntu 24.04 / 25.x / 26.04.
#                (22.04 is unsupported: apt ships Verilator 4 + KiCad 6, too old
#                 for the current PyQt6 / Verilator-5 / KiCad-9 application.)
#
#                ONE script, version-aware via detect_profile(). Replaces the
#                old per-version install-nghdl-scripts/install-nghdl-XX.04.sh set.
#
#                Key design:
#                - GHDL + Verilator come from apt (no source builds). apt
#                  verilator is >= 5 on all supported releases, and ghdl-llvm
#                  (or the gcc backend) is available.
#                - Icarus Verilog is built from source with --enable-libvvp:
#                  ngspice's ivlng adapter dlopens libvvp at run time and apt
#                  iverilog ships WITHOUT it. Source-built (not a committed
#                  binary) so it tracks the distro toolchain and stays ABI-
#                  matched to the nghdl-simulator. Non-fatal: on failure we fall
#                  back to apt iverilog (Verilog Verifier still works; only
#                  d_cosim is lost). See library/cosim/INSTALL_SUBSTRATE.md.
#                - Both the nghdl-simulator (custom ngspice) and Icarus Verilog
#                  are built from source; everything else comes from apt.
#                - GHDL MUST use the llvm or gcc backend, NEVER mcode: nghdl
#                  links its VHDL socket server with `ghdl -e -Wl,...`, which
#                  mcode rejects -> server never builds -> "Simulation Failed".
#                  See GHDL-BACKEND-26.04.md in install-nghdl-scripts/.
#
#        AUTHOR: Fahim Khan, Rahul Paknikar, Sumanto Kar, Harsha Narayana P,
#                Jayanth Tatineni, Anshul Verma, Shiva Krishna Sangati
#  ORGANIZATION: eSim, FOSSEE group at IIT Bombay
#       CREATED: Tuesday 02 December 2014 17:01
#      REVISION: June 2026 — unified version-profile rewrite
#==========================================================

# `set -e`/`set -E` + ERR trap are enabled only inside --install (see Main).
# No global `set -u`/`pipefail` (breaks venv activate / aborts benign pipelines).

nghdl="nghdl-simulator"
config_dir="$HOME/.nghdl"
config_file="config.ini"
src_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Profile vars
UBUNTU_VER=""
NGHDL_CFLAGS=""     # extra CFLAGS for the ngspice build
GTK_CANBERRA=""     # gtk2+gtk3 vs gtk3-only (libcanberra-gtk-module dropped in 26.04)

# Icarus Verilog (d_cosim substrate). Built into an eSim-owned prefix UNDER the
# nghdl tree so --uninstall (which removes $HOME/$nghdl) cleans it up too.
ICARUS_PREFIX="$HOME/$nghdl/iverilog"
# Pinned, validated upstream commit (post-v12, compatible with ngspice's ivlng
# adapter — see library/cosim/INSTALL_SUBSTRATE.md). Bump deliberately and re-run
# the d_cosim smoke test from that doc before shipping a new ref.
ICARUS_REF="de415b2f03c1b41ab5b46faa9632716d98c1cd86"
ICARUS_OK=0        # set to 1 only when the libvvp build truly succeeds

log()  { echo -e "\n>>> $*"; }
warn() { echo -e "[WARN] $*" >&2; }

error_exit() {
    echo -e "\n\nError! Kindly resolve the above error(s) and try again."
    echo -e "Aborting NGHDL installation...\n"
}

detect_profile() {
    UBUNTU_VER=$(grep '^VERSION_ID=' /etc/os-release | cut -d '"' -f 2)

    # GCC 14/15 (24.04+) default to C23 where 'bool' is a keyword; ngspice has
    # `typedef int bool`. Force gnu11 to restore pre-C23 compatibility.
    NGHDL_CFLAGS="-std=gnu11"

    case "$UBUNTU_VER" in
        26.04)
            GTK_CANBERRA="libcanberra-gtk3-module"   # gtk2 variant gone in 26.04
            ;;
        23.04|24.04|25.04)
            GTK_CANBERRA="libcanberra-gtk-module libcanberra-gtk3-module"
            ;;
        *)
            GTK_CANBERRA="libcanberra-gtk3-module"
            warn "Ubuntu $UBUNTU_VER not explicitly profiled — using defaults."
            ;;
    esac
    log "NGHDL profile: Ubuntu $UBUNTU_VER | CFLAGS='$NGHDL_CFLAGS'"
}

# Fail fast on the classic fresh-VM traps (no disk, no network, no apt)
# BEFORE an hour of compiling, with messages that say exactly what to fix.
preflight() {
    log "Preflight checks"

    # ~6 GB: ngspice source tree + build objects + iverilog build. df -P for
    # portable output; check the filesystem $HOME lives on (build target).
    local free_kb need_kb=6000000
    free_kb=$(df -Pk "$HOME" | awk 'NR==2 {print $4}')
    if [ -n "$free_kb" ] && [ "$free_kb" -lt "$need_kb" ]; then
        echo "ERROR: less than 6 GB free on $HOME ($((free_kb / 1024)) MB)."
        echo "       The nghdl-simulator + Icarus builds need ~6 GB. Free up"
        echo "       space and re-run."
        exit 1
    fi

    if ! command -v apt-get &>/dev/null; then
        echo "ERROR: apt-get not found - this installer supports Ubuntu only."
        exit 1
    fi

    # apt must be able to fetch package lists (proxy/offline detection).
    # Non-fatal only if the cache is already primed; a hard failure here
    # otherwise surfaces as cryptic per-package errors later.
    if ! sudo apt-get update -qq 2>/dev/null; then
        warn "apt-get update failed - check network/proxy. Continuing with"
        warn "the existing package cache; installs below may fail."
    fi

    if [ ! -f "$src_dir/${nghdl}-source.tar.xz" ]; then
        echo "ERROR: $src_dir/${nghdl}-source.tar.xz not found."
        echo "       The nghdl simulator source tarball must ship with eSim."
        exit 1
    fi
}

installDependency() {
    log "Installing build dependencies"
    sudo apt-get install -y \
        make autoconf g++ flex bison \
        git gperf \
        zlib1g-dev libreadline-dev \
        libxaw7 libxaw7-dev \
        $GTK_CANBERRA

    log "Installing Verilator (apt)"
    sudo apt-get install -y verilator
    if command -v verilator &>/dev/null; then
        log "verilator: $(verilator --version 2>/dev/null | head -n1)"
    else
        warn "verilator not found after install"
    fi

    installGHDL
}

# Install GHDL with a non-mcode backend and prove it.
installGHDL() {
    log "Installing GHDL (llvm/gcc backend; never mcode)"

    # Prefer llvm, fall back to gcc. NEVER install the 'ghdl' meta-package: its
    # Depends line is `ghdl-mcode | ghdl-gcc | ghdl-llvm` and apt picks the first
    # available alternative (mcode), which cannot link the nghdl socket server.
    if ! sudo apt-get install -y ghdl-llvm; then
        warn "ghdl-llvm unavailable — trying ghdl-gcc"
        sudo apt-get install -y ghdl-gcc || warn "ghdl-gcc unavailable too"
    fi

    # The /usr/bin/ghdl wrapper (ghdl-common) selects mcode -> gcc -> llvm unless
    # $GHDL_BACKEND is set, so an installed mcode always wins. Keep it out.
    sudo apt-get purge -y ghdl-mcode 2>/dev/null || true

    if dpkg -l ghdl-mcode 2>/dev/null | grep -q '^ii'; then
        echo "ERROR: ghdl-mcode is installed. It cannot link the nghdl socket"
        echo "       server (-Wl) and breaks nghdl simulation silently."
        echo "       Run: sudo apt-get purge -y ghdl-mcode"
        exit 1
    fi

    if ! command -v ghdl &>/dev/null; then
        echo "ERROR: ghdl not found after install."
        exit 1
    fi
    # Confirm the active backend is NOT mcode.
    if ghdl --version 2>/dev/null | grep -qi 'mcode'; then
        echo "ERROR: active GHDL backend is mcode (JIT). nghdl needs llvm/gcc."
        echo "       Installed: $(dpkg -l | grep -E '^ii.*ghdl' | awk '{print $2}' | tr '\n' ' ')"
        echo "       Try: export GHDL_BACKEND=llvm  (or install ghdl-llvm/ghdl-gcc)"
        exit 1
    fi
    log "ghdl: $(ghdl --version 2>/dev/null | head -n1)"
}

installNGHDL() {
    log "Building nghdl-simulator (custom ngspice) from source"

    cd "$src_dir"
    tar -xJf "${nghdl}-source.tar.xz" -C "$HOME"
    rm -rf "$HOME/$nghdl"
    mv "$HOME/${nghdl}-source" "$HOME/$nghdl" 2>/dev/null || true
    log "Extracted to $HOME/$nghdl"

    cd "$HOME/$nghdl"

    # Verilator-5 / GCC-14+ link fixes (verilated_threads.o, V<mod>__ALL.a with
    # --whole-archive, ghdl.cm/Ngveri.cm in spinit). --forward makes re-runs
    # idempotent (an already-applied patch is skipped, not re-applied).
    if [ -f "$src_dir/nghdl-simulator.patch" ]; then
        log "Applying nghdl-simulator.patch"
        patch -p1 --forward < "$src_dir/nghdl-simulator.patch" \
            || warn "patch did not apply cleanly (already applied?) — continuing"
    else
        warn "nghdl-simulator.patch not found — Verilator-5/GCC-14+ links may fail"
    fi

    mkdir -p install_dir release
    cd release

    log "Configuring nghdl-simulator"
    chmod +x ../configure
    ../configure \
        --enable-xspice \
        --disable-debug \
        --prefix="$HOME/$nghdl/install_dir/" \
        --exec-prefix="$HOME/$nghdl/install_dir/" \
        CFLAGS="$NGHDL_CFLAGS"

    make -j"$(nproc)"
    make install
    sudo chmod 755 "$HOME/$nghdl/install_dir/bin/ngspice"

    log "Replacing system ngspice with nghdl-simulator"
    set +e; trap "" ERR
    sudo apt-get purge -y ngspice 2>/dev/null || true
    sudo rm -f /usr/bin/ngspice
    set -e; trap error_exit ERR
    sudo ln -sf "$HOME/$nghdl/install_dir/bin/ngspice" /usr/bin/ngspice
    log "nghdl-simulator installed, symlinked to /usr/bin/ngspice"
}

# Build Icarus Verilog with libvvp for d_cosim. ngspice's ivlng adapter dlopens
# libvvp at run time; distro/apt iverilog is built WITHOUT it, so a source build
# is the one piece d_cosim needs. Installs into an eSim-owned prefix and records
# the paths in config.ini [COSIM]. Entirely non-fatal: any failure falls back to
# apt iverilog so the Verilog Verifier still works (only d_cosim is unavailable),
# and the app surfaces a clear reason via CosimConfig.missing_reason().
installIcarus() {
    log "Building Icarus Verilog with libvvp (d_cosim co-simulation)"

    local build="$HOME/${nghdl}-iverilog-build"
    local tarball="$src_dir/iverilog-source.tar.xz"

    # Run the whole build in a subshell so a failure cannot abort the eSim
    # install (ERR trap is suspended around it).
    set +e; trap "" ERR
    (
        set -e
        rm -rf "$build" "$ICARUS_PREFIX"
        # Prefer a shipped source tarball (offline + reproducible release zip);
        # otherwise clone the pinned upstream commit.
        if [ -f "$tarball" ]; then
            mkdir -p "$build"
            tar -xJf "$tarball" -C "$build" --strip-components=1
        else
            git clone https://github.com/steveicarus/iverilog.git "$build"
            git -C "$build" checkout "$ICARUS_REF"
        fi
        cd "$build"
        sh autoconf.sh
        ./configure --prefix="$ICARUS_PREFIX" --enable-libvvp

        # Cap parallel jobs by RAM (~1.5 GB/job). Upstream make at -j$(nproc)
        # OOM-kills on small VMs (a ~6 GB box dies around -j7); fresh test VMs
        # are exactly that small.
        local mem_kb jobs nj
        mem_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo 2>/dev/null || echo 2000000)
        jobs=$(( mem_kb / 1500000 )); [ "$jobs" -lt 1 ] && jobs=1
        nj=$(nproc 2>/dev/null || echo 1); [ "$jobs" -gt "$nj" ] && jobs=$nj
        make -j"$jobs"
        make install
    )
    local rc=$?
    set -e; trap error_exit ERR
    rm -rf "$build"

    if [ $rc -eq 0 ] && [ -x "$ICARUS_PREFIX/bin/iverilog" ] \
       && ls "$ICARUS_PREFIX"/lib/libvvp* >/dev/null 2>&1; then
        ICARUS_OK=1
        log "Icarus Verilog (with libvvp) installed at $ICARUS_PREFIX"
    else
        warn "Icarus source build failed — falling back to apt iverilog."
        warn "The Verilog Verifier will still work; d_cosim co-simulation will be"
        warn "unavailable (apt iverilog has no libvvp). Re-run --install to retry."
        sudo apt-get install -y iverilog || warn "apt iverilog install also failed"
    fi
}

createConfigFile() {
    mkdir -p "$config_dir"
    rm -f "$config_dir/$config_file"
    {
        echo "[NGHDL]"
        echo "NGHDL_HOME = $HOME/$nghdl"
        echo "DIGITAL_MODEL = %(NGHDL_HOME)s/src/xspice/icm"
        echo "RELEASE = %(NGHDL_HOME)s/release"
        echo "[SRC]"
        echo "SRC_HOME = $src_dir"
        echo "LICENSE = %(SRC_HOME)s/LICENSE"
    } >> "$config_dir/$config_file"

    # Record the source-built Icarus paths so CosimConfig resolves d_cosim
    # without relying on PATH. Appended verbatim (NOT via configparser): the
    # [NGHDL] section above uses %(NGHDL_HOME)s interpolation that a configparser
    # rewrite would mangle. Only written when the libvvp build succeeded; on the
    # apt fallback, CosimConfig finds iverilog on PATH (and reports d_cosim off).
    if [ "$ICARUS_OK" -eq 1 ]; then
        {
            echo ""
            echo "[COSIM]"
            echo "IVERILOG = $ICARUS_PREFIX/bin/iverilog"
            echo "VVP = $ICARUS_PREFIX/bin/vvp"
            echo "IVERILOG_LIB = $ICARUS_PREFIX/lib"
        } >> "$config_dir/$config_file"
    fi
}

# Post-install self-check: prove the pieces THIS script owns actually landed
# (the full app-level doctor runs from install-eSim.sh afterwards).
selfCheck() {
    log "NGHDL post-install self-check"
    local bad=0

    if [ -x "$HOME/$nghdl/install_dir/bin/ngspice" ]; then
        log "ngspice: $HOME/$nghdl/install_dir/bin/ngspice"
    else
        warn "ngspice binary missing at $HOME/$nghdl/install_dir/bin/ngspice"
        bad=1
    fi

    local cmdir="$HOME/$nghdl/install_dir/lib/ngspice"
    for cm in ghdl.cm ivlng; do
        if ls "$cmdir"/${cm}* >/dev/null 2>&1; then
            log "code model present: ${cm}*"
        else
            warn "code model MISSING: $cmdir/${cm}* (VHDL/d_cosim will fail)"
            bad=1
        fi
    done

    if ghdl --version 2>/dev/null | grep -qi 'mcode'; then
        warn "GHDL backend is mcode - nghdl simulation will fail (see"
        warn "install-nghdl-scripts/GHDL-BACKEND-26.04.md)"
        bad=1
    fi

    if [ "$ICARUS_OK" -eq 1 ]; then
        log "Icarus Verilog with libvvp: $ICARUS_PREFIX"
    else
        warn "Icarus libvvp build fell back to apt iverilog - the Verilog"
        warn "Verifier works, d_cosim does not. Re-run --install to retry."
    fi

    if [ $bad -ne 0 ]; then
        warn "Self-check found problems (above). Run 'esim --doctor' after"
        warn "the eSim install finishes for the full actionable report."
    else
        log "Self-check passed."
    fi
}

createSoftLink() {
    sudo chmod 755 "$src_dir/src/ngspice_ghdl.py"
    cd /usr/local/bin
    [ -L nghdl ] && sudo unlink nghdl
    sudo ln -sf "$src_dir/src/ngspice_ghdl.py" nghdl
    log "Softlink created: /usr/local/bin/nghdl -> $src_dir/src/ngspice_ghdl.py"
}

#####################################################################
# Main
#####################################################################
if [ "$#" -ne 1 ]; then
    echo "USAGE: ./install-nghdl.sh --install | --uninstall"
    exit 1
fi
option="$1"

detect_profile

case "$option" in
    --install)
        set -e; set -E; trap error_exit ERR
        preflight
        installDependency
        installNGHDL
        installIcarus
        createConfigFile
        createSoftLink
        selfCheck
        log "NGHDL installed successfully on Ubuntu $UBUNTU_VER"
        ;;

    --uninstall)
        log "Removing NGHDL"
        sudo rm -rf "$HOME/$nghdl" "$HOME/.nghdl" \
                    /usr/share/kicad/library/eSim_Nghdl.lib \
                    /usr/local/bin/nghdl /usr/bin/ngspice 2>/dev/null || true
        # GHDL/Verilator are apt packages shared with other tools — purge
        # best-effort only.
        sudo apt-get purge -y ghdl-llvm ghdl-gcc verilator 2>/dev/null || true
        sudo apt-get autoremove -y 2>/dev/null || true
        log "NGHDL uninstalled."
        ;;

    *)
        echo "Please select a valid operation: --install | --uninstall"
        exit 1
        ;;
esac
