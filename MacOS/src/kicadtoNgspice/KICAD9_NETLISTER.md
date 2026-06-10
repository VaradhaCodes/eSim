# KiCad-9 Netlister — Status & Checkpoint

Checkpoint of the KiCad-9 → ngspice netlister work. This is the feature that
replaces KiCad's broken `--format spice` export. If it produces a wrong netlist,
every downstream simulation is wrong, so it is held to a measured bar: it must
reproduce the known-good legacy `.cir` for every example.

**Bottom line: 101/101 golden tests pass. The netlister now handles the full
known eSim example set plus 33 additional student project schematics.** The most
recent session found and fixed a subcircuit-prefix bug (§2.4) and extended the
golden suite from 68 to 101 fixtures by testing against 46 KiCad-9 schematics
from a new batch of student projects.

---

## 1. Why this exists

eSim simulates with ngspice. KiCad ≥ 7 rebuilt `kicad-cli ... --format spice`
around its Simulation-Model system; any symbol without a `Sim.*` model — which is
*every* eSim symbol (plots, behavioural u-blocks, sources, custom subckts) — is
exported with its connectivity stripped (`U2 __U2`, `v3 __v3`). Unrecoverable.

`KicadNetlister.py` sidesteps this: it runs `kicad-cli sch export netlist
--format kicadxml` (the generic netlist that always lists every component, every
net, and every pin→net mapping regardless of simulation models) and rewrites it
into the flat-spice form eSim expects:

```
<ref> <net-per-pin-in-node-order> <value>
```

Node order = pin-number order (eSim symbols number pins in spice node order); a
`Spice_Node_Sequence` field reorders when present; `Spice_Netlist_Enabled=N`
drops a component.

**Ground truth** = the legacy `<name>.cir` that shipped with each example
(produced by the old KiCad-5/6 flow, known to work). The KiCad-9 `.kicad_sch`
files were re-saved by hand; the generated `.cir` must match the legacy one
**functionally** (same circuit graph), not literally (net names, case and
component order differ harmlessly).

---

## 2. Changes made in this checkpoint

### 2.1 `KicadNetlister.py` hardening (+39 / −9)
- **`_sanitize_net`**: now maps every character that is not `[a-z0-9_]` to `_`
  (was: only stripped `()` and spaces). KiCad-9 auto-net names embed `+`, `-`,
  `/` (from pin names like `v1-+` and hierarchical paths). Those are arithmetic
  operators *inside* ngspice `v()`/`i()` references, so an internal node such as
  `Net-(v1-+)` could simulate but never be plotted by name. Now `v()`-safe.
  (Legacy was no better here — it used hyphens too — but eSim's default control
  uses `print allv`, so it never hit it. This is a strict improvement.)
- **Collision-safe net naming**: distinct KiCad nets that sanitize to the same
  string are disambiguated by the net's unique `code`, so two different nets can
  never collapse into one node and short the circuit.
- **Case-insensitive `Spice_*` field lookups** (`Spice_Node_Sequence`,
  `Spice_Netlist_Enabled`).
- **Broader disable values**: `Spice_Netlist_Enabled` ∈ {n, no, false, 0}.

No signature changes; the call site in `projManagement/Kicad.py` is untouched.

### 2.4 Subcircuit X-prefix fix (commit `3522d456b`)

**Bug:** When a schematic component had a `U`-prefixed ref (e.g. `U1`) and a
subcircuit value (e.g. `lm_741`), the netlister emitted `u1 ... lm_741`. The
correct SPICE form is `xu1 ... lm_741` — ngspice only recognises subcircuit
instantiation from lines starting with `X`.

This also broke the eSim GUI's **SubcircuitTab**, which detects subcircuit lines
solely by `eachline[0] == 'x'`. A `u1 ...` line is invisible to the tab — the
user could never select the `.sub` path for it.

The established eSim convention is to use `X1` in the schematic ref for opamp
subcircuits. The original 52 examples all did this. Student schematics in the
new batch used `U1` for the same opamps, exposing the gap.

**Fix:** Added `_is_subcircuit_value(value, proj_dir)` and `_esim_subckt_lib()`
to `KicadNetlister.py`. Before emitting a component line, if the ref does not
already start with `x` and the value has a matching `.sub` file (in the project
directory or in `library/SubcircuitLibrary/<value>/<value>.sub`), `x` is
prepended. Behavioural eSim blocks (`plot_v1`, `PORT`, `adc_bridge_1`, `d_dff`,
etc.) have no `.sub` file so their `U` prefix is unchanged.

All 68 pre-existing golden tests still pass after this change.

### 2.2 Regression test suite — `src/kicadtoNgspice/tests/`
- **`test_netlister_golden.py`** — 52 in-repo fixtures under `golden/<example>/`,
  each holding the KiCad-9 `.kicad_sch` + the legacy `.cir`. Generates from the
  schematic in a scratch dir and asserts topological equivalence to the legacy
  netlist. **52/52 PASS.**
- **`test_netlister_unit.py`** — 7 pure-function tests (no kicad-cli) for the
  sanitizer, collision safety, `Spice_Node_Sequence` reordering, and
  `Spice_Netlist_Enabled` dropping — features the golden schematics don't carry.
  **7/7 PASS.**
- **`netlist_compare.py`** — the topology-equivalence definition (see §4).
- **`README.md`** — how to run.

Runs under `pytest src/kicadtoNgspice/tests/` or standalone
(`python3 .../test_netlister_golden.py`). Needs `kicad-cli` (or
`$ESIM_KICAD_CLI`); skips cleanly if absent.

### 2.3 One schematic defect fixed — `Precision_Rectifiers_using_LM741`
This was the only example that failed. Cause: in the re-saved KiCad-9 schematic,
opamp **X3's V+ pin (pin 7) physically sat on the inverting-input rail**
(both at y=114.3), shorting the +12 V supply to the inverting input. A re-save
layout error, *not* a netlister bug — the netlister faithfully reported the
broken schematic. Fix: removed the two horizontal wires passing through pin 7
and rerouted the inverting-input rail around it (detour at y=111.76). Verified
the regenerated netlist is now topology-exact to the legacy ground truth.

> **Where the fix lives:** the schematic edited is the user's workspace copy at
> `~/eSim-Workspace/Precision_Rectifiers_using_LM741/Precision_Rectifiers_using_LM741.kicad_sch`
> (outside this repo). The **fixed** schematic is captured here as the test
> fixture `tests/golden/Precision_Rectifiers_using_LM741/`. The **original**
> (broken) schematic is backed up at
> `~/netlister_goldens_backup/Precision_Rectifiers_using_LM741.kicad_sch.orig`.
> The detour wire is electrically correct but cosmetically crosses the opamp
> symbol body; tidy it in the KiCad GUI if desired (no effect on simulation).

---

## 3. How good is it — evidence

- **Topology: 101/101** golden fixtures pass (52 original eSim examples + 16
  subcircuit-workspace fixtures + 33 new student-project fixtures). Covers R/L/C,
  diodes, BJT (C/B/E order), JFET/MOSFET, 8-pin LM555 and 14-pin 4023 subckts
  (exact pin order), adc/dac bridges, 4-node transformer, 8-pin opamp DIP with
  offset-null/NC pins, PS/2 protocol, DALI protocol, FSK lm741 sub-circuits,
  3x8 decoder, full/half adder, VCO-ADC sub-circuits, sinc3 filter, and more.
- **Simulation-equivalence** (ngspice, generated vs legacy at shared named nodes):
  | example | analysis | result |
  |---|---|---|
  | RC | `.tran` | exact, Δ = 0 |
  | FET_Characteristic | `.dc` (66-pt sweep) | exact, Δ = 0 |
  | Differentiator | `.tran` (opamp subckt + floating pins) | 0.24 % (adaptive-timestep noise on a spiky output; last sample matches) |
- **ngspice version independence**: verified identical results on the installed
  ngspice-35 **and** on ngspice-45.2 (Ubuntu 26.04 apt). The netlister emits
  plain flat spice and uses no version-specific feature.

### Confirmed-inert cosmetic differences (do not affect simulation)
- Net-name spelling / lowercase (ngspice is case-insensitive; aliases `gnd`→0).
- Component order in the file (ngspice is order-independent).
- Unconnected pins: legacy lumps all as `?`, KiCad-9 emits distinct
  `unconnected-*` nodes — each a distinct floating node, simulates the same.
- Model/value spelling (`eSim_NJF` vs `NJF`): resolved by the eSim **DeviceModel
  GUI step** downstream; both become the real model (e.g. `J2N3819`).
- Source values: the netlister emits type placeholders (`DC`, `sine`, `pwl`,
  `pulse`) exactly like legacy; the kicadtoNgspice GUI injects the real
  parameters from `<proj>_Previous_Values.xml`.

---

## 4. Equivalence definition (why a text diff is useless)

KiCad-9 and the legacy exporter name auto-nets differently, lowercase
differently, and order components differently — none of which changes the
circuit. Two netlists are equivalent when they describe the same graph:
1. same set of component refs (case-insensitive),
2. same node count per ref,
3. same connectivity up to net renaming — for every net, the set of
   `(ref, pin_position)` it touches matches. Refs are the anchors.

Pin-order rules: symmetric 2-terminal **R/L/C** are order-insensitive (terminal
swap is electrically identical); every other device (D, Q, M, J, V, I, X, U) is
order-sensitive — this is what catches a real node-order bug on a transistor or
subcircuit. Unconnected pins are distinct floating nodes.

---

## 5. Student-project batch test (2026-06-10)

Tested against 46 KiCad-9 schematics (from 60 student projects in
`esim_valid_projects/`) that had both a `.kicad_sch` (version ≥ 20250114) and a
legacy `.cir` ground-truth. Result: **33 PASS, 13 FAIL**.

The X-prefix fix (§2.4) was discovered and applied during this run; without it,
several additional cases would have failed.

### 33 new golden fixtures added
PS2_PROTOCOL, DALI_Protocol_Model, ClassABAmplifier_eSim, transimp2, sinc3,
Design_of_3x8_Decoder_Using_2x4_Decoders, Design_of_Full_subtractor_using_Nand_gates,
Design_of_Xor_gate_using_Nand_gates, Design_of_Half_Adder (2 sub-circuits),
counter, latch_block, latch_sch, latch_test, full_adder, half_adder,
4017, lm555n, Digital_Dice_4017, Digital_Dice_lm555n,
lm_741 (7 project-specific variants), 3_and (4 project-specific variants),
2x4_decoder (2 project-specific variants), FSK_Transceiver_lm_741.

### 13 failures — all schematic-level issues, not netlister bugs

| # | Schematic(s) | Root cause |
|---|---|---|
| 1 | `LM393.kicad_sch` (×4 projects) | Unannotated refs (`D?`, `Q?`, etc.) — KiCad-9 re-save stripped instance numbers from the sub-circuit definition. Fix: Tools → Annotate Schematic in KiCad. |
| 2 | `FSK_Transceiverrrrrr` | Refs now correct (`xu1`–`xu11` ✓) but golden has a **space inside a net name** (`DATA_OUT label`). Old netlister emitted literal spaces in net names — invalid SPICE. New netlister correctly sanitizes to `data_out_label`; the golden is the broken artifact here. |
| 3 | `FSK_Transceiver` | Golden was hand-edited to contain behavioral voltage sources (`BFSK`, `BBP10K`, `BBP5K`) not present in the schematic. Schematic and golden describe different circuits. |
| 4 | `VCO_ADC`, `dff`, `Sub1v_CMOS`, `scr` | Topology changed during KiCad-9 re-save (e.g. C5 re-connected to GND, PORT block wiring changed). Netlister is correct; the schematic was altered. |
| 5 | `I2S_Protocol_Simulation` | DAC bridge blocks (U12–U19) and 8 termination resistors removed from schematic during re-save. |
| 6 | `shunt_res` | `Vv1` → `v1` voltage source prefix change + R5 missing. Same class of issue as the subcircuit-workspace failures in the prior session. |
| 7 | `CA3140_BIMOS_op-amp` | Schematic redesigned with different model names (`pmos4`/`eSim_Diode` vs `CMOSP`/`1N4148`). Ground-truth and schematic describe different implementations. |

---

## 6. Known gaps / not-yet-covered (for a future session)

The 52 examples don't exercise everything. Untested edge cases worth hardening:
- **Multi-unit symbols** (e.g. quad opamp units A/B/C/D sharing power pins).
- **Hierarchical sheets / buses / power symbols** beyond what the examples use.
- **KiCad-9 native `Sim.*` fields** — none of the re-saved examples carry any
  `Spice_*` or `Sim.*` property, so those code paths are covered only by the
  synthetic unit tests, not by a real schematic.
- **Raw spice directives** (`.model`, `.include`, `.param`) carried on symbols.

## 7. ngspice / simulator note (out of scope here)

The netlister is **decoupled** from the ngspice version — it makes a standard
`.cir`. eSim launches simulation via `process.start('ngspice', ...)`, i.e. it
runs whatever `ngspice` resolves to on `PATH`. On this machine that is
`/usr/bin/ngspice` → a **custom nghdl-patched ngspice-35** (for digital
co-simulation). Ubuntu 26.04 ships mainline **ngspice-45.2** via apt. Mixing
"newer mainline ngspice" with "nghdl digital co-sim" is a separate
simulator/build concern and does **not** involve this netlister.

---

## 8. Local artifacts (not in repo)

- `~/netlister_plan.md` — original phased plan.
- `~/netlister_diff.py` — standalone topology-diff harness (seed of the test).
- `~/netlister_goldens_backup/` — backup of all 81 legacy `.cir` + the original
  (broken) Precision `.kicad_sch`.
