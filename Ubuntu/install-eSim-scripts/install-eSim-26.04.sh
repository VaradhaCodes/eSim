#!/bin/bash
#=============================================================================
#          FILE: install-eSim-26.04.sh
#         USAGE: ./install-eSim.sh --install | --uninstall
#   DESCRIPTION: eSim installer for Ubuntu 26.04 LTS (Resolute Raccoon)
#                Designed for the master branch source structure.
#         NOTES: - PyQt6 (master branch requires it)
#                - kicadLibrary handled as a directory (no tarball)
#                - KiCad 9 from ubuntu universe repo (no PPA)
#                - NGHDL: apt ghdl/verilator + nghdl-simulator from source
#                  with CFLAGS=-std=gnu11 (GCC 15 / C23 fix)
#                - libcanberra-gtk-module gone in 26.04, use gtk3 variant
#                - installNghdl truly non-fatal (set +e not just trap)
#  ORGANIZATION: eSim Team, FOSSEE, IIT Bombay
#      REVISION: Sunday 08 June 2026
#=============================================================================

config_dir="$HOME/.esim"
config_file="config.ini"
eSim_Home=$(pwd)
ngspiceFlag=0

error_exit() {
    echo -e "\n\nError! Kindly resolve above error(s) and try again."
    echo -e "\nAborting Installation...\n"
}

function createConfigFile {
    if [ -d "$config_dir" ]; then
        rm -f "$config_dir/$config_file" && touch "$config_dir/$config_file"
    else
        mkdir -p "$config_dir" && touch "$config_dir/$config_file"
    fi
    echo "[eSim]"                                                             >> "$config_dir/$config_file"
    echo "eSim_HOME = $eSim_Home"                                            >> "$config_dir/$config_file"
    echo "LICENSE = %(eSim_HOME)s/LICENSE"                                   >> "$config_dir/$config_file"
    echo "KicadLib = %(eSim_HOME)s/library/kicadLibrary"                     >> "$config_dir/$config_file"
    echo "IMAGES = %(eSim_HOME)s/images"                                     >> "$config_dir/$config_file"
    echo "VERSION = %(eSim_HOME)s/VERSION"                                   >> "$config_dir/$config_file"
    echo "MODELICA_MAP_JSON = %(eSim_HOME)s/library/ngspicetoModelica/Mapping.json" >> "$config_dir/$config_file"
}

function installDependency {
    set +e; trap "" ERR
    echo "Updating apt index files..................."
    sudo apt-get update
    set -e; trap error_exit ERR

    sudo apt-get install -y python3-full python3-venv python3-virtualenv python3-pip
    sudo apt-get install -y xterm python3-psutil python3-matplotlib python3-setuptools

    echo "Installing PyQt6 (system package)........."
    sudo apt-get install -y python3-pyqt6 pyqt6-dev-tools

    echo "Creating virtualenv with system-site-packages..."
    rm -rf "$config_dir/env"
    if command -v virtualenv &>/dev/null; then
        virtualenv --system-site-packages "$config_dir/env"
    else
        python3 -m venv --system-site-packages "$config_dir/env"
    fi

    source "$config_dir/env/bin/activate"
    pip install --upgrade pip
    pip install watchdog

    set +e; trap "" ERR
    pip install https://github.com/hdl/pyhdlparser/tarball/master || echo "[WARN] hdlparse failed"
    pip install makerchip-app || echo "[WARN] makerchip-app failed"
    pip install sandpiper-saas || echo "[WARN] sandpiper-saas failed"

    # Install pinned deps from requirements.txt, skipping PyQt6/matplotlib/sphinx
    if [ -f "$eSim_Home/requirements.txt" ]; then
        echo "Installing from requirements.txt (non-Qt deps)..."
        grep -vE "PyQt6|matplotlib|sphinx|hdlparse" "$eSim_Home/requirements.txt" \
            | grep -v "^#" \
            | grep -v "^$" \
            | xargs pip install --ignore-installed 2>/dev/null \
            || echo "[WARN] Some requirements.txt packages failed"
    fi
    set -e; trap error_exit ERR
}

function installKicad {
    echo "Installing KiCad..........................."
    if dpkg -s kicad &>/dev/null; then
        installed_version=$(dpkg-query -W -f='${Version}' kicad | cut -d'.' -f1)
        if [[ "$installed_version" != "9" ]]; then
            read -p "KiCad $installed_version installed. Replace with KiCad 9? (yes/no): " response
            if [[ "$response" =~ ^([Yy][Ee][Ss]|[Yy])$ ]]; then
                sudo apt-get remove --purge -y kicad kicad-footprints kicad-libraries kicad-symbols kicad-templates
                sudo apt-get autoremove -y
            else
                echo "Keeping existing KiCad. Exiting."; exit 1
            fi
        else
            echo "KiCad 9 already installed."; return 0
        fi
    fi
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y universe && sudo apt-get update
    sudo apt-get install -y --no-install-recommends \
        kicad kicad-footprints kicad-libraries kicad-symbols kicad-templates
    echo "KiCad 9 installed!"
}

function copyKicadLibrary {
    kicad_lib_src="$eSim_Home/library/kicadLibrary"
    if [ ! -d "$kicad_lib_src" ]; then
        echo "[WARN] $kicad_lib_src not found — skipping library copy"; return 0
    fi

    kicad_config_dir="$HOME/.config/kicad"
    latest_version=$(ls "$kicad_config_dir" 2>/dev/null | grep -E "^[0-9]+\.[0-9]+$" | sort -V | tail -n 1)
    if [ -z "$latest_version" ]; then
        kicad_pkg_ver=$(dpkg-query -W -f='${Version}' kicad 2>/dev/null | grep -oP '^\d+')
        latest_version="${kicad_pkg_ver:-9}.0"
        mkdir -p "$kicad_config_dir/$latest_version"
    fi
    echo "Using KiCad config dir: $kicad_config_dir/$latest_version"

    cp "$kicad_lib_src/template/sym-lib-table" "$kicad_config_dir/$latest_version/"
    sudo mkdir -p /usr/share/kicad/symbols
    sudo rsync -a "$kicad_lib_src/eSim-symbols/" /usr/share/kicad/symbols/
    sudo chown -R "$USER:$USER" /usr/share/kicad/symbols
    echo "KiCad library configured!"
}

function installNghdl {
    echo "Installing NGHDL..........................."
    if [ ! -f "nghdl/nghdl-simulator-source.tar.xz" ]; then
        echo "[WARN] nghdl/nghdl-simulator-source.tar.xz not found."
        echo "[WARN] Skipping NGHDL. Copy nghdl-simulator-source.tar.xz into nghdl/ to enable."
        return 0
    fi
    cd nghdl/
    chmod +x install-nghdl.sh
    chmod +x install-nghdl-scripts/*.sh 2>/dev/null || true

    set +e; trap "" ERR
    ./install-nghdl.sh --install
    nghdl_exit=$?
    set -e; trap error_exit ERR

    if [ $nghdl_exit -ne 0 ]; then
        echo "[WARN] NGHDL failed (exit $nghdl_exit) — GHDL co-simulation unavailable."
    else
        ngspiceFlag=1; echo "NGHDL installed."
    fi
    cd ../
}

function installSky130Pdk {
    echo "Installing SKY130 PDK......................"
    tar -xJf library/sky130_fd_pr.tar.xz
    sudo rm -rf /usr/share/local/sky130_fd_pr
    sudo mkdir -p /usr/share/local/
    sudo mv sky130_fd_pr /usr/share/local/
    sudo chown -R "$USER:$USER" /usr/share/local/sky130_fd_pr/
}

function createDesktopStartScript {
    printf '#!/bin/bash\ncd %s/src/frontEnd\nsource %s/env/bin/activate\npython3 Application.py\n' \
        "$eSim_Home" "$config_dir" > esim-start.sh
    sudo chmod 755 esim-start.sh
    sudo cp -vp esim-start.sh /usr/bin/esim
    rm esim-start.sh

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
    sudo cp -vp esim.desktop /usr/share/applications/
    cp -vp esim.desktop "$HOME/Desktop/"

    set +e; trap "" ERR
    gio set "$HOME/Desktop/esim.desktop" "metadata::trusted" true
    chmod a+x "$HOME/Desktop/esim.desktop"
    rm esim.desktop
    set -e; trap error_exit ERR

    cp -vp images/logo.png "$config_dir"
}

####################################################################

if [ "$#" -ne 1 ]; then
    echo "USAGE: ./install-eSim.sh --install | --uninstall"; exit 1
fi
option=$1

if [ "$option" == "--install" ]; then
    set -e; set -E; trap error_exit ERR

    echo -n "Is your internet connection behind proxy? (y/n): "
    read getProxy
    if [ "$getProxy" == "y" ] || [ "$getProxy" == "Y" ]; then
        echo -n "Proxy Hostname: "; read proxyHostname
        echo -n "Proxy Port: ";     read proxyPort
        echo -n "Username: ";       read proxyUser
        echo -n "Password: ";       read -s proxyPass; echo
        export http_proxy="http://$proxyUser:$proxyPass@$proxyHostname:$proxyPort"
        export https_proxy="$http_proxy" HTTP_PROXY="$http_proxy" HTTPS_PROXY="$http_proxy"
    else
        echo "Install without proxy"
    fi

    createConfigFile
    installDependency
    installKicad
    copyKicadLibrary
    installNghdl
    installSky130Pdk
    createDesktopStartScript

    echo "-----------------eSim Installed Successfully-----------------"
    echo "Type \"esim\" in Terminal to launch it"

elif [ "$option" == "--uninstall" ]; then
    echo -n "Remove eSim completely? (y/n): "; read getConfirmation
    if [ "$getConfirmation" == "y" ] || [ "$getConfirmation" == "Y" ]; then
        sudo rm -rf "$HOME/.esim" "$HOME/Desktop/esim.desktop" /usr/bin/esim /usr/share/applications/esim.desktop
        sudo apt purge -y kicad kicad-footprints kicad-libraries kicad-symbols kicad-templates 2>/dev/null || true
        sudo rm -rf /usr/share/kicad
        rm -rf "$HOME/.config/kicad/9.0"
        sudo rm -rf "$config_dir/env" /usr/share/local/sky130_fd_pr
        [ -f "nghdl/install-nghdl.sh" ] && (cd nghdl && ./install-nghdl.sh --uninstall) || true
        echo "eSim uninstalled."
    fi
else
    echo "Use --install or --uninstall"
fi
