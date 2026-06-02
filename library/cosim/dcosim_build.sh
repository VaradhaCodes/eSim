#!/bin/bash
# Build a d_cosim Verilator shared library directly (no ngspice vlnggen).
#
# Why: ngspice's bundled `vlnggen` is an interpreter script, and this ngspice
# build lowercases sourced-script tokens, so `--Mdir`->`--mdir` and the
# `--prefix Vlng` value ->`vlng`, breaking Verilator + the shim (which hardcodes
# class `Vlng`). Driving Verilator + g++ directly gives full case control.
#
# Usage: dcosim_build.sh <model.v> [extra verilator args, e.g. --timing]
# Output: <base>.so in the current directory.
set -e

NGINST="${NGSPICE_HOME:-$HOME/ngspice46}"
SHIMDIR="$NGINST/share/ngspice/scripts/src"
VERILATOR="${VERILATOR:-verilator}"

SRC="$1"; shift
base="$(basename "$SRC" .v)"
obj="${base}_obj_dir"

# Detect --timing to add the matching shim define / extra object.
timing_def=""
timing_obj=""
for a in "$@"; do
    if [ "$a" = "--timing" ]; then
        timing_def="--CFLAGS -DWITH_TIMING"
        timing_obj="$obj/verilated_timing.o"
    fi
done

rm -rf "$obj" "${base}.so"

# Pass 1: Verilog -> C++. prefix MUST be Vlng to match verilator_shim.cpp.
"$VERILATOR" --Mdir "$obj" --prefix Vlng --CFLAGS -fpic --cc "$@" "$SRC"

# Generate the port tables consumed by verilator_shim.cpp from Vlng.h.
# Format in Vlng.h:  VL_IN8(&name,msb,lsb);  ->  VL_DATA(8,name,msb,lsb)
# INOUT must be matched before IN (VL_INOUT contains VL_IN).
hdr() { printf '/* Generated code: do not edit. */\n' > "$obj/$1"; }
hdr inputs.h; hdr outputs.h; hdr inouts.h
sed -n 's/.*VL_INOUT[A-Z]*\([0-9]\+\)(&\([^;]*\);.*/VL_DATA(\1,\2/p' "$obj/Vlng.h" >> "$obj/inouts.h"
sed -n '/VL_INOUT/d; s/.*VL_IN[A-Z]*\([0-9]\+\)(&\([^;]*\);.*/VL_DATA(\1,\2/p' "$obj/Vlng.h" >> "$obj/inputs.h"
sed -n 's/.*VL_OUT[A-Z]*\([0-9]\+\)(&\([^;]*\);.*/VL_DATA(\1,\2/p' "$obj/Vlng.h" >> "$obj/outputs.h"

# Pass 2: compile shim + main + verilated model into objects/archive.
"$VERILATOR" --Mdir "$obj" --prefix Vlng --CFLAGS "-I$SHIMDIR" --CFLAGS -fpic $timing_def \
    --cc --build --exe "$@" "$SHIMDIR/verilator_main.cpp" "$SHIMDIR/verilator_shim.cpp" "$SRC"

# Link the shared library loaded by the d_cosim code model (main.o excluded).
# Glob verilated*.o: older Verilator omits verilated_threads.o; --timing adds
# verilated_timing.o. Vlng__ALL.a is the model archive.
g++ --shared "$obj/verilator_shim.o" "$obj"/verilated*.o \
    "$obj/Vlng__ALL.a" -pthread -lpthread -o "${base}.so"

echo "Built ${base}.so"
