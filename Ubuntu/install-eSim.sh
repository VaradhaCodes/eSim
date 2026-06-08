#!/bin/bash
#=============================================================================
#          FILE: install-eSim.sh
#
#         USAGE: ./install-eSim.sh --install
#                            OR
#                ./install-eSim.sh --uninstall
#
#   DESCRIPTION: Installation script for eSim EDA Suite.
#                Detects the Ubuntu version and delegates to the
#                appropriate version-specific installer in
#                install-eSim-scripts/.
#
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#       AUTHORS: Fahim Khan, Rahul Paknikar, Saurabh Bansode,
#                Sumanto Kar, Partha Singha Roy, Jayanth Tatineni,
#                Anshul Verma, Shiva Krishna Sangati, Harsha Narayana P
#  ORGANIZATION: eSim Team, FOSSEE, IIT Bombay
#       CREATED: Wednesday 15 July 2015 15:26
#      REVISION: Sunday 08 June 2026 — added Ubuntu 26.04 (Resolute Raccoon) support
#=============================================================================

# Detect Ubuntu version
get_ubuntu_version() {
    VERSION_ID=$(grep "^VERSION_ID" /etc/os-release | cut -d '"' -f 2)
    # Try 3-part version (e.g. 24.04.2), fall back to 2-part (e.g. 26.04)
    FULL_VERSION=$(lsb_release -d | grep -oP '\d+\.\d+\.\d+')
    if [ -z "$FULL_VERSION" ]; then
        FULL_VERSION=$(lsb_release -d | grep -oP '\d+\.\d+')
    fi
    echo "Detected Ubuntu Version: $FULL_VERSION"
}

# Choose and run the appropriate version-specific script
run_version_script() {
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/install-eSim-scripts"

    case $VERSION_ID in
        "22.04")
            if [[ "$FULL_VERSION" == "22.04.4" ]]; then
                SCRIPT="$SCRIPT_DIR/install-eSim-22.04.sh"
            else
                SCRIPT="$SCRIPT_DIR/install-eSim-23.04.sh"
            fi
            ;;
        "23.04")
            SCRIPT="$SCRIPT_DIR/install-eSim-23.04.sh"
            ;;
        "24.04")
            SCRIPT="$SCRIPT_DIR/install-eSim-24.04.sh"
            ;;
        "25.04")
            SCRIPT="$SCRIPT_DIR/install-eSim-25.04.sh"
            ;;
        "26.04")
            SCRIPT="$SCRIPT_DIR/install-eSim-26.04.sh"
            ;;
        *)
            echo "Unsupported Ubuntu version: $VERSION_ID ($FULL_VERSION)"
            exit 1
            ;;
    esac

    if [[ -f "$SCRIPT" ]]; then
        echo "Running script: $SCRIPT $ARGUMENT"
        bash "$SCRIPT" "$ARGUMENT"
    else
        echo "Installer script not found: $SCRIPT"
        exit 1
    fi
}

####################################################################
#                   MAIN START FROM HERE                           #
####################################################################

if [ "$#" -eq 1 ]; then
    ARGUMENT=$1
else
    echo "USAGE : "
    echo "./install-eSim.sh --install"
    echo "./install-eSim.sh --uninstall"
    exit 1
fi

if [[ "$ARGUMENT" != "--install" && "$ARGUMENT" != "--uninstall" ]]; then
    echo "Invalid argument: $ARGUMENT"
    echo "Usage: $0 --install | --uninstall"
    exit 1
fi

get_ubuntu_version
run_version_script
