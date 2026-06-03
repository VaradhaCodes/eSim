# HANDOFF — eSim Verilog co-simulation via ngspice `d_cosim` (Icarus)

**Date:** 2026-06-02 · **Branch:** `feature/ngveri-dcosim` · **Machine:** Ubuntu VM (sudo pw: `7102006`)

## GOAL
Add a NEW Verilog↔Ngspice co-simulation backend to eSim using ngspice's
upstream **`d_cosim`** code model with **Icarus Verilog** (NOT Verilator):
- No C/C++ compiler on the user machine, no ngspice rebuild per model (unlike
  legacy NgVeri static `.cm`).
- Must work for **ANY** Verilog block / circuit, cross-platform, foolproof.
- Legacy NgVeri/NGHDL flows kept in parallel (not deprecated).
Verilator was explicitly REJECTED (needs a C++ compiler, duplicates legacy).

---

## SUBSTRATE (built on this VM; NOT in the eSim repo — rebuild elsewhere)
- **ngspice 46** built from source → `~/ngspice46` (`--enable-xspice
  --enable-osdi`, XSPICE/OSDI default-on). Source: `~/src/ngspice-46`.
  Has `d_cosim` (in `digital.cm`) + `ivlng.so` + `ivlng.vpi`.
  System ngspice is **v35** (`/usr/bin/ngspice`, NO d_cosim) — kept as fallback.
- **iverilog 14 built `--enable-libvvp`** → `~/iverilog` (gives
  `~/iverilog/lib/libvvp.so`). **apt/distro iverilog LACKS libvvp** → source
  build is MANDATORY (needs `gperf`, `bison`, `flex`, `autoconf`). ivlng dlopens
  libvvp at runtime.
- verilator 4.210 present but UNUSED (dropped).

### Canonical d_cosim Icarus workflow (proven)
```
iverilog -g2012 -o <model> <model>.v          # tiny vvp, no C compiler
# netlist:
a<id> [<inputs>] [<outputs>] <inst>            # 2 bracket groups ONLY
.model <inst> d_cosim simulation="ivlng" sim_args=["<model>"]
# run (vvp/ivlng need libvvp on the loader path, and cwd must contain the vvp):
cd <projdir> && LD_LIBRARY_PATH=$HOME/iverilog/lib ~/ngspice46/bin/ngspice -b <netlist>
```

---

## eSIM CODE CHANGES (all committed on `feature/ngveri-dcosim`)
Commits (newest first): `2d15d4c3 bb00b4cb eee28fbd 031b71b4 3548965d de7e2588
dd5da306 92f722b0 fcbff8d7`.

- **`src/maker/ModelGeneration.py`** — `build_cosim(engine="icarus")`: compiles
  `<model>.v` → vvp via `iverilog -g2012` (prefers `~/iverilog/bin/iverilog` or
  `$ESIM_IVERILOG`). `verilogParse(make_symbol=False)` reused to get ports.
- **`src/maker/createkicadCosim.py`** (NEW) — `CosimSchematic`: writes
  `library/modelParamXML/NgVeriCosim/<m>.xml` (`<type>NgVeriCosim</type>`) +
  appends symbol to `eSim_NgVeriCosim.kicad_sym`. **split = `"<in_bits>-V:<out_bits>-V"`**
  (2 groups — see d_cosim ifspec) + `node_number 2`.
- **`src/maker/NgVeri.py`** — button "Convert Verilog to Ngspice (d_cosim,
  Icarus)" + `addverilog_cosim()` handler.
- **`src/kicadtoNgspice/Convert.py`** — `addModelParameter` special-cases
  `line[6]=="NgVeriCosim"` → `_cosim_model_line()`: emits `.model <inst> d_cosim
  simulation="ivlng" sim_args=["<model>"]` and copies the vvp into the project
  dir (sim_args is cwd-relative).
- **`src/kicadtoNgspice/KicadNetlister.py`** (NEW) — generates `<proj>.cir` from
  `<proj>.kicad_sch` via `kicad-cli ... --format kicadxml` (KiCad 8's `spice`
  export is broken for eSim symbols). Called from `Kicad.openKicadToNgspice`.
- **`src/projManagement/Kicad.py`** — `openKicadToNgspice` calls
  `KicadNetlister.generate_netlist` before reading the `.cir`.
- **`src/ngspiceSimulation/NgspiceWidget.py`** — `_ngspice_binary(netlist)`:
  returns `~/ngspice46/bin/ngspice` when netlist contains `d_cosim`, else system
  `ngspice` (override `$ESIM_NGSPICE`). `_add_iverilog_libpath()` sets
  `LD_LIBRARY_PATH=~/iverilog/lib`. Passes binary to TerminalUi. Prints
  `eSim: launching ngspice -> <path>`.
- **`src/frontEnd/TerminalUi.py`** — `__init__(...,ngspice_bin='ngspice')`; the
  rerun/redo button uses `self.ngspice_bin` (was hardcoded bare `ngspice`=v35).
- **`src/kicadtoNgspice/KicadtoNgspice.py`** `createNetlistFile` — for d_cosim
  netlists, run the analysis ONCE inside `.control` (strip the `.tran` card +
  the `run`) because ivlng's vvp is one-shot.
- KiCad lib registered: `eSim_NgVeriCosim` added to
  `~/.config/kicad/{6.0,8.0}/sym-lib-table` + `library/kicadLibrary/template/sym-lib-table`;
  empty seed lib at `library/kicadLibrary/eSim-symbols/eSim_NgVeriCosim.kicad_sym`.

---

## VERIFIED WORKING
- Icarus d_cosim end-to-end on ngspice 46: inverter + NE555 example + the user's
  `relu16` (ml_act_relu_16bit_q8_8). relu computes correctly: `sign_in=1 ->
  outputs 0`; `sign_in=0 -> pass-through`. `plot_bit_*` toggle 0↔5 V.
- KiCad 8 netlist (kicadxml) → full connectivity, all 22 components + 17 plot
  markers. eSim launches v46 for d_cosim (confirmed in log).
- Single-run fix removes the `16/0` / "already run" error.

## KNOWN REMAINING ISSUES (next work)
1. **Digital-node plotting** (`plot_v_in_msb`): digital/event nodes are NOT in
   `print allv` (analog only), so they don't appear in eSim's Python plots; and
   batch-mode native `plot` is ignored. Pre-existing for ALL eSim mixed-signal
   (nghdl too), not d_cosim-specific. Fix = capture event nodes (e.g. `eprvcd`/
   per-node event print) and extend `src/ngspiceSimulation/data_extraction.py`.
   Don't change `print allv` globally (regression risk on analog flow).
2. **vvp one-shot:** ivlng can't reset; only ONE analysis run per ngspice
   invocation works. Single-run fix handles `-b`. Multiple analyses in one run
   (e.g. `.op` + `.tran`, or `.ac`+`.tran`) would still reuse → fail. Truly
   foolproof = make ivlng reload/reset per run (upstream C: `icarus_shim.c`).
3. **`inout` ports:** eSim's PortInfo lumps `inout` into inputs; d_cosim has a
   separate `d_inout` group. Fine for no-inout modules; needs handling otherwise.
4. **No-input modules** (e.g. pure generators): d_cosim wants `null` for an empty
   input group; Processing emits `[]`. Untested.
5. Windows packaging: documented only, not built/tested.

---

## TEST RECIPES (reproduce / verify)
```bash
# Substrate smoke test (inverter), no eSim:
mkdir -p /tmp/t && cd /tmp/t
printf '`timescale 1ns/1ps\nmodule inv(input a,output reg y);always @(*) y=~a;endmodule\n' > inv.v
~/iverilog/bin/iverilog -g2012 -o inv inv.v
cat > t.cir <<'EOF'
* inv d_cosim
.model adc_b adc_bridge(in_low=0.4 in_high=0.6)
.model dac_b dac_bridge(out_low=0 out_high=1 t_rise=1n t_fall=1n)
vin ain 0 pulse(0 1 0 1u 1u 5u 12u)
aadc [ain] [a] adc_b
ainv [a] [y] dut
.model dut d_cosim simulation="ivlng" sim_args=["inv"]
adac [y] [yout] dac_b
.control
tran 0.5u 40u
wrdata out.txt v(ain) v(yout)
.endc
.end
EOF
LD_LIBRARY_PATH=$HOME/iverilog/lib ~/ngspice46/bin/ngspice -b t.cir   # yout = ~ain

# relu project (after re-Convert in eSim):
cd /home/varadha/Downloads/relu/relu16/relu16
LD_LIBRARY_PATH=$HOME/iverilog/lib ~/ngspice46/bin/ngspice -b relu16.cir.out
# want: NO "mismatched ... 16/0", NO "already run"; plot_data_v.txt has 0 and 5V
```
eSim run: `esim` (launcher at `/usr/bin/esim` → runs `~/eSim/src/frontEnd`,
venv `~/eSim/.venv`). Watch terminal for `eSim: launching ngspice -> .../ngspice46/...`.

## GUI FLOW (the feature)
Makerchip/NgVeri tab → pick `.v` → "Convert Verilog to Ngspice (d_cosim,
Icarus)" → builds vvp + `NgVeriCosim` symbol → place symbol from
`eSim_NgVeriCosim` lib in eeschema → Convert KiCad to Ngspice → Simulate.

## KEY GOTCHAS (learned the hard way)
- ngspice's bundled `vlnggen` (Verilator helper) is BROKEN on this build: the
  sourced-script tokenizer lowercases everything (`--Mdir`→`--mdir`). Another
  reason Verilator was dropped.
- KiCad 8 `--format spice` emits `<ref> __<REF>` with no nodes for eSim symbols
  (no sim model) → use `kicadxml`.
- d_cosim a-device MUST be exactly `[d_in][d_out]` (+ optional `[d_inout]`);
  one bracket group per Verilog port FAILS ("Too many connections").
- eSim doesn't generate the `.cir` — historically the user exports it; now
  `KicadNetlister` auto-generates it in `openKicadToNgspice`.

## REFERENCES
- Full architecture: `~/eSim/ARCHITECTURE_NGVERI_DCOSIM.md`
- ngspice cosim source: `~/src/ngspice-46/src/xspice/verilog/` (icarus_shim.c,
  vpi.c) + `src/xspice/icm/digital/d_cosim/ifspec.ifs`
- ngspice examples: `~/src/ngspice-46/examples/xspice/icarus_verilog/`

## SESSION UPDATE: Digital Node Plotting, UI Fixes, & The Timescale Bug

**Date:** 2026-06-03 · **Branch:** `feature/ngveri-dcosim`

### 1. FEATURE: Digital/Event Node Plotting in eSim Waveform Viewer
**The Problem:** eSim's Python plotting relies on `plot_data_v.txt` generated via the `print allv` command. However, `allv` only captures *analog* nodes. Pure digital/event nodes (like `adc_bridge` outputs, e.g., `plot_v_in_msb`) were invisible in the standard eSim plots, and batch mode ignores native `plot` commands.
**The Fix:** Implemented a focused, two-phase capture system to explicitly extract event nodes without breaking the global analog flow:
* **Netlist Generation (`src/kicadtoNgspice/KicadtoNgspice.py`):** Added `_get_event_plot_nodes()` to scan the schematic info for `.model` definitions (`adc_bridge`, `d_cosim`, `dac_bridge`) and isolate explicitly probed event nodes. It automatically injects an `eprint` command (e.g., `eprint plot_v_in_msb > plot_data_event.txt`) into the `.control` block.
* **Data Extraction (`src/ngspiceSimulation/data_extraction.py`):** Added `_parse_event_file()`. This parses the `eprint` output, maps digital logic states to analog voltages (`0s`/`0u` $\rightarrow$ 0V, `1s`/`1u` $\rightarrow$ 5V, `Us`/`Xu` $\rightarrow$ 2.5V), and step-hold resamples the data onto the analog transient time axis (`tran_x`). These arrays are merged seamlessly into the standard node lists.
* **Commit:** `62b101cc`

### 2. BUG FIX: Native "Ngspice Plots" UI Button Crash
**The Problem:** Clicking the "Ngspice Plots" button in the eSim GUI opened an xterm session using the system's legacy `ngspice` (v35) binary, which immediately crashed when encountering `d_cosim` elements.
**The Fix:**
* Patched `open_ngspice_plots()` in `src/ngspiceSimulation/NgspiceWidget.py`. 
* It now dynamically uses `self.ngspice_bin` (resolving to v46 for `d_cosim` netlists).
* Injected `LD_LIBRARY_PATH=$HOME/iverilog/lib` into the xterm execution string so the bundled ngspice46 can successfully `dlopen` `libvvp.so` (the Icarus engine).
* **Commit:** `a7c169d7`

### 3. BUG FIX: `d_cosim` Icarus Flatlining (The `half_adder` 0V Bug)
**The Problem:** A standard `half_adder.v` module run through `d_cosim` was outputting a continuous 0V on both `sum` and `carry`, despite valid logic transitions on the input pins. (Note: The C-source of ngspice/xspice/vpi was deeply debugged to find this. Modifying `icarus_shim.c` to use `After_input` or forcing `advance()` at `t=0` in `cfunc.mod` caused timing regressions and were strictly reverted. The ngspice-46 C-code remains pristine).
**The Root Causes:**
1.  **Missing Timescale (Primary Blocker):** The user's `half_adder.v` lacked a ``timescale` directive. Without this, Icarus (`ivlng`) computes `tick_length = pow(10, vpiTimeUnit)`. With no timescale, `vpiTimeUnit=0`, resulting in a `tick_length` of 1 full second. Because the eSim transient simulation executes in milliseconds, the calculated VVP `ticks` equaled 0. The VVP engine never advanced, the active-event queue froze, and `cbValueChange` callbacks never fired to update the outputs.
2.  **Input Threshold Mismatch:** `v1` PWL peaked at `1v` (`pwl(0ms 0v 4ms 0v 5ms 1v)`). Because the `adc_bridge` defines `in_high=2.0V`, the input bit never registered as Logic-1.
3.  **Initial State Static High:** `v2` PWL started HIGH at `t=0` (`pwl(0ms 5v...)`). `d_cosim` records the initial state but does not execute the Verilog logic until an actual input *transition* forces the VVP to advance ticks.

**The Fixes:**
* **eSim Backend Patch:** Modified `src/maker/ModelGeneration.py` (`build_cosim`). The compiler now auto-scans the source `.v` file. If ``timescale` is missing, it creates a temporary file, seamlessly injects ``timescale 1ns/1ps\n`, and compiles the temp file. This guarantees VVP ticks advance properly for all future models.
* **Commit:** `1919cf54`

### REQUIRED USER ACTIONS FOR `half_adder` TESTBED:
To achieve a successful simulation on the `half_adder` project, the following manual schematic updates are required:
1.  **Fix V1 (Voltage level):** Update PWL to reach 5V $\rightarrow$ `pwl(0ms 0v 4ms 0v 5ms 5v)`
2.  **Fix V2 (Force transition):** Force a start at 0V with a rapid rise to 5V $\rightarrow$ `pwl(0ms 0v 0.01ms 5v 4ms 5v 5ms 0v)`
3.  **Re-Compile:** Click "Convert Verilog to Ngspice (d_cosim)" in eSim again so the backend auto-injects the timescale fix.
