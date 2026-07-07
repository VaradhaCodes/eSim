#!/bin/sh
# Copy the transitive MinGW runtime-DLL closure of the custom eSim ngspice
# (exe + code models + ivlng) into install_dir/bin, so the simulator runs
# WITHOUT tools\msys64 on PATH (Compact installs, plain simulations) -- the
# same trick the official ngspice Windows zip uses. Run inside MINGW64.
#
#   usage: msys-dll-closure.sh <nghdl-root>   (the dir holding install_dir/)
set -e
root="$1"
[ -n "$root" ] || { echo "usage: $0 <nghdl-root>" >&2; exit 2; }
BIN="$root/install_dir/bin"
CM="$root/install_dir/lib/ngspice"

deps() { objdump -p "$@" 2>/dev/null | grep 'DLL Name' | awk '{print $3}' | sort -u; }

seen=""
queue=$(deps "$BIN/ngspice.exe" "$CM/ivlng.dll" "$CM"/*.cm)
while [ -n "$queue" ]; do
    next=""
    for d in $queue; do
        case " $seen " in *" $d "*) continue ;; esac
        seen="$seen $d"
        if [ -f "/mingw64/bin/$d" ]; then
            cp -n "/mingw64/bin/$d" "$BIN/" && echo "staged runtime DLL: $d"
            next="$next $(deps /mingw64/bin/$d)"
        fi
    done
    queue=$(echo "$next" | tr ' ' '\n' | sort -u | grep . || true)
done
