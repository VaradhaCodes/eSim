#!/bin/bash
#==========================================================
#          FILE: install-nghdl-26.04.sh
#
#         USAGE: ./install-nghdl.sh --install
#                            OR
#                ./install-nghdl.sh --uninstall
#
#   DESCRIPTION: NGHDL installer for Ubuntu 26.04 LTS (Resolute Raccoon)
#
#         NOTES: Changes from 24.04:
#                - GHDL: built from source (ghdl-4.1.0) on 24.04 via LLVM.
#                  On 26.04 the system LLVM is v21 — incompatible with GHDL
#                  4.1.0's build system. Use apt ghdl 5.0.1 instead.
#                - Verilator: built from source (4.210) on 24.04.
#                  Use apt verilator 5.032 instead on 26.04.
#                - nghdl-simulator (custom ngspice): still built from source
#                  as it is the eSim-specific fork.
#        AUTHOR: Fahim Khan, Rahul Paknikar, Sumanto Kar,
#                Harsha Narayana P, Jayanth Tatineni, Anshul Verma
#  ORGANIZATION: eSim, FOSSEE group at IIT Bombay
#       CREATED: Tuesday 02 December 2014 17:01
#      REVISION: Sunday 08 June 2026 — Ubuntu 26.04 support
#==========================================================

nghdl="nghdl-simulator"
config_dir="$HOME/.nghdl"
config_file="config.ini"
src_dir=$(pwd)

error_exit() {
    echo -e "\n\nError! Kindly resolve above error(s) and try again."
    echo -e "\nAborting Installation...\n"
}


function installDependency
{
    echo "Installing build dependencies.........................."

    sudo apt-get install -y \
        make autoconf g++ flex bison \
        zlib1g-dev \
        libreadline-dev \
        libcanberra-gtk3-module \
        libxaw7 libxaw7-dev

    echo "Installing GHDL (LLVM backend) from Ubuntu 26.04 repos (5.0.1)."
    # IMPORTANT: install ghdl-llvm EXPLICITLY, never the 'ghdl' meta-package.
    # The meta-package's Depends line is:
    #     ghdl-mcode | ghdl-gcc | ghdl-llvm
    # and apt picks the FIRST available alternative -> ghdl-mcode (JIT).
    # nghdl builds the VHDL socket server with `ghdl -e -Wl,ghdlserver.o ...`;
    # the mcode backend rejects -Wl ("option -Wl is not available when ghdl is
    # not configured with gcc or llvm"), so the server binary never builds, it
    # never listens on the socket, and the ngspice A-device client gives up with
    # "Connect Error ... giving up" => "Simulation Failed". Only gcc/llvm link.
    # See GHDL-BACKEND-26.04.md in this directory for the full writeup.
    sudo apt-get install -y ghdl-llvm

    # The /usr/bin/ghdl wrapper (owned by ghdl-common) selects the backend in
    # order mcode -> gcc -> llvm unless $GHDL_BACKEND is set, so if mcode is ever
    # present it wins. Keep it out of the box.
    sudo apt-get purge -y ghdl-mcode 2>/dev/null || true

    # Fail loudly if mcode slipped back in: a working box must NOT have it.
    if dpkg -l ghdl-mcode 2>/dev/null | grep -q '^ii'; then
        echo "ERROR: ghdl-mcode is installed; it cannot link the nghdl socket"
        echo "       server (-Wl) and will break nghdl simulation silently."
        echo "       Run: sudo apt-get purge -y ghdl-mcode"
        exit 1
    fi

    echo "Installing Verilator from Ubuntu 26.04 repos (5.032).."
    sudo apt-get install -y verilator

    echo "Dependencies installed."
}


function installNGHDL
{
    echo "Building nghdl-simulator (custom ngspice) from source.."

    cd "$src_dir"
    tar -xJf "$nghdl-source.tar.xz" -C "$HOME"
    mv "$HOME/${nghdl}-source" "$HOME/$nghdl" 2>/dev/null || true

    echo "nghdl-simulator extracted to $HOME/$nghdl"
    cd "$HOME/$nghdl"

    # Apply toolchain-compatibility patch (Verilator 5.x / GCC 14+):
    # links verilated_threads.o, uses V<mod>__ALL.a archives with
    # --whole-archive, and loads ghdl.cm / Ngveri.cm in spinit. Without
    # this the Ngveri code models fail to link/load. Fail loudly if the
    # patch is present but does not apply (a silent skip = broken build).
    if [ -f "$src_dir/nghdl-simulator.patch" ]; then
        echo "Applying nghdl-simulator.patch........................."
        patch -p1 --forward < "$src_dir/nghdl-simulator.patch"
    else
        echo "WARNING: nghdl-simulator.patch not found in $src_dir —"
        echo "         Verilator-5 / GCC-14+ builds will fail to link."
    fi

    mkdir -p install_dir
    mkdir -p release
    cd release

    echo "Configuring nghdl-simulator............................"
    chmod +x ../configure
    # GCC 15 (Ubuntu 26.04) defaults to C23 where 'bool' is a keyword.
    # ngspice source has 'typedef int bool' which breaks under C23.
    # Force C11 to restore pre-C23 compatibility.
    ../configure \
        --enable-xspice \
        --disable-debug \
        --prefix="$HOME/$nghdl/install_dir/" \
        --exec-prefix="$HOME/$nghdl/install_dir/" \
        CFLAGS="-std=gnu11"

    make -j$(nproc)
    make install

    sudo chmod 755 "$HOME/$nghdl/install_dir/bin/ngspice"

    set +e
    trap "" ERR
    echo "Removing system ngspice if present (replaced by nghdl-simulator).."
    sudo apt-get purge -y ngspice 2>/dev/null || true
    sudo rm -f /usr/bin/ngspice 2>/dev/null || true
    set -e
    trap error_exit ERR

    sudo ln -sf "$HOME/$nghdl/install_dir/bin/ngspice" /usr/bin/ngspice
    echo "nghdl-simulator installed, symlinked to /usr/bin/ngspice"
}


function createConfigFile
{
    if [ -d "$config_dir" ]; then
        rm -f "$config_dir/$config_file" && touch "$config_dir/$config_file"
    else
        mkdir -p "$config_dir" && touch "$config_dir/$config_file"
    fi

    echo "[NGHDL]" >> "$config_dir/$config_file"
    echo "NGHDL_HOME = $HOME/$nghdl" >> "$config_dir/$config_file"
    echo "DIGITAL_MODEL = %(NGHDL_HOME)s/src/xspice/icm" >> "$config_dir/$config_file"
    echo "RELEASE = %(NGHDL_HOME)s/release" >> "$config_dir/$config_file"
    echo "[SRC]" >> "$config_dir/$config_file"
    echo "SRC_HOME = $src_dir" >> "$config_dir/$config_file"
    echo "LICENSE = %(SRC_HOME)s/LICENSE" >> "$config_dir/$config_file"
}


function createSoftLink
{
    sudo chmod 755 "$src_dir/src/ngspice_ghdl.py"

    cd /usr/local/bin
    if [[ -L nghdl ]]; then
        sudo unlink nghdl
    fi
    sudo ln -sf "$src_dir/src/ngspice_ghdl.py" nghdl
    echo "Softlink created: nghdl -> $src_dir/src/ngspice_ghdl.py"
}


#####################################################################
#                   Script starts here                              #
#####################################################################

if [ "$#" -ne 1 ]; then
    echo "USAGE: ./install-nghdl.sh --install | --uninstall"
    exit 1
fi

option=$1

if [ "$option" == "--install" ]; then

    set -e
    set -E
    trap error_exit ERR

    installDependency
    installNGHDL
    createConfigFile
    createSoftLink

    echo "NGHDL installed successfully on Ubuntu 26.04"

elif [ "$option" == "--uninstall" ]; then
    sudo rm -rf "$HOME/$nghdl" "$HOME/.nghdl" \
        /usr/share/kicad/library/eSim_Nghdl.lib \
        /usr/local/bin/nghdl \
        /usr/bin/ngspice
    echo "NGHDL uninstalled."
else
    echo "Please select the proper operation: --install | --uninstall"
fi
