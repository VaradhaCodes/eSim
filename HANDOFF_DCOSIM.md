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
