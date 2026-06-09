# KiCad-9 netlister tests

Regression + unit tests for `KicadNetlister.py`, which generates the eSim spice
netlist (`<proj>.cir`) from a KiCad-9 `.kicad_sch` via `kicad-cli ... --format
kicadxml`. This is the path that replaces KiCad's own spice export, so it must not
regress: if it produces a wrong netlist, every downstream simulation is wrong.

## What is here

- **`test_netlister_golden.py`** — for every example under `golden/`, generates a
  netlist from the KiCad-9 `.kicad_sch` and asserts it is *topologically*
  equivalent to the legacy ground-truth `.cir` (52 examples, all passing).
- **`test_netlister_unit.py`** — pure-function tests (no kicad-cli) for the
  sanitizer, net-name collision safety, `Spice_Node_Sequence` reordering, and
  `Spice_Netlist_Enabled` dropping — features the golden schematics do not carry.
- **`netlist_compare.py`** — the equivalence definition (see its header). Net
  names, lowercase and component order are ignored; the connectivity *graph*
  (and pin order on non-symmetric devices) is what is compared.
- **`golden/<example>/`** — fixtures: each holds the KiCad-9 `<name>.kicad_sch`
  and the legacy `<name>.cir`. kicad-cli needs only the `.kicad_sch`; the `.cir`
  is the ground truth to compare against.

## Running

```sh
# with pytest (one test per example):
pytest src/kicadtoNgspice/tests/

# or standalone, no dependencies beyond Python + kicad-cli:
python3 src/kicadtoNgspice/tests/test_netlister_golden.py
python3 src/kicadtoNgspice/tests/test_netlister_unit.py
```

Requires `kicad-cli` (KiCad >= 7) on `PATH`, or set `$ESIM_KICAD_CLI`. The golden
test skips cleanly if kicad-cli is absent. Point it at a different fixture set
with `$ESIM_NETLIST_GOLDENS`.

## Why topology, not text

The KiCad-9 netlister and the legacy KiCad-5/6 exporter name auto-nets
differently (`net__q1_c` vs `Net-(C1-Pad1)`), lowercase differently, and emit
components in a different order — none of which changes the circuit. A text diff
would be all false positives. Two netlists are equivalent when they describe the
same graph: same component refs, same node count per ref, and the same pin↔net
connectivity up to renaming, with pin *order* significant on diodes, transistors,
sources, subcircuits and behavioural blocks but not on symmetric R/L/C.

## Ground-truth safety

`generate_netlist()` overwrites `<proj>.cir`. The tests always copy the
`.kicad_sch` to a scratch dir before generating, so the golden `.cir` is never
touched. Never run the netlister in-place next to a golden.
