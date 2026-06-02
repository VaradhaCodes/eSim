# eSim Verilog Co-Simulation — Architecture: Legacy NgVeri vs. `d_cosim`

**Status:** Proposal / design doc
**Date:** 2026-06-02
**Scope:** Replace/augment the Verilog↔Ngspice co-simulation backend.
**Target ngspice:** v46 (released 2026-03-29).

---

## 1. Executive summary

eSim's current Verilog mixed-signal backend (**NgVeri**) compiles each Verilog
block into a statically-linked XSPICE code model (`Ngveri.cm`) and **rebuilds +
reinstalls the ngspice code-model library on every model add** (`make install`).
This requires shipping a full ngspice **source tree** (~118 MB of build-only
files), Verilator, and a C/C++ compiler (gcc on Linux, MinGW/MSYS2 on Windows).
It is heavy, slow, and brittle.

ngspice ≥ v42 ships an upstream code model, **`d_cosim`**, purpose-built for this
exact job. It compiles a Verilog block into a **dynamically-loaded shared
library** (or runs it through an interpreter adapter) that ngspice loads at
netlist-parse time. **No ngspice rebuild. No source tree. No `make install`.**

**Proposal:** add a parallel backend using `d_cosim` with two engines:

| Engine | ngspice support | User toolchain | Role |
|---|---|---|---|
| **Icarus Verilog** (`ivlng` adapter) | v44+ | `iverilog` only — **no C++ compiler** | **default** (portable, tiny) |
| **Verilator** (`vlnggen` → `.so`) | v42+ | Verilator ≥4.210 + C++ compiler | opt-in ("fast" runtime) |

Legacy NgVeri static-`.cm` flow is **kept in parallel** (not deprecated). The
"mixed schematic" requirement is satisfied **for free**: legacy `.cm` blocks,
new `d_cosim` blocks, and optional C `d_process` blocks are all XSPICE `A`
devices sharing ngspice's single event-driven node engine.

---

## 2. Legacy architecture (today)

eSim ships **two** independent mixed-signal backends.

### 2.1 NgVeri (Verilog) — static link, rebuild-per-model

Build orchestration: `src/maker/NgVeri.py:120-146` → `src/maker/ModelGeneration.py`.

```
.v ─┐
    │  verilogfile / verilogParse / getPortInfo            (ModelGeneration.py)
    │  cfuncmod / ifspecwrite / sim_main_header / sim_main  ← generate XSPICE
    │                                                          wrapper + sim_main
    ▼
 run_verilator()      ModelGeneration.py:834   verilator --cc --exe -O3 ...
 make_verilator()     ModelGeneration.py:886   make -f V<mod>.mk  → .o
 copy_verilator()     ModelGeneration.py:923   cp *.o  release/src/xspice/icm/Ngveri/
 runMake()            ModelGeneration.py:983   make            (relink Ngveri.cm)
 runMakeInstall()     ModelGeneration.py:1017  make install    (REINSTALL into ngspice lib)
    │                                          [Windows: copy Ngveri.cm by hand, NgVeri.py:138]
    ▼
 install_dir/lib/ngspice/Ngveri.cm   ← one fat code model holding ALL Verilog blocks
```

Runtime: `src/frontEnd/Application.py:474 open_ngspice()` runs `<proj>.cir.out`
through ngspice; ngspice loads `Ngveri.cm` via `codemodel`.

**Properties**
- Linking model: every Verilog module's Verilated `.o` is linked into a *single*
  `Ngveri.cm`. Adding/changing one block relinks + reinstalls the whole library.
- Hard deps: full ngspice **source tree** (`~/.nghdl/config.ini` →
  `RELEASE=~/nghdl-simulator/release`), **Verilator**, **gcc / MinGW (MSYS2)**.
- Known failure modes: `QProcess.start("make install")` hangs; hardcoded
  contributor paths; Windows path divergence (`NgVeri.py:134-146`).

### 2.2 NGHDL (VHDL) — dynamic, socket IPC

`nghdl/src/ghdlserver/ghdlserver.c`: each VHDL block is compiled by GHDL into a
standalone **TCP socket server**. The fixed `ghdl.cm` code model spawns the
server (`start_server.sh`, `DUTghdl/`) and exchanges port values over sockets.
**No ngspice rebuild.** Netlister routes these via
`src/kicadtoNgspice/KicadtoNgspice.py:111-114` (`if line[6] == "Nghdl"`).

> NGHDL already proves eSim can do dynamic, event-driven, process-based co-sim
> without rebuilding ngspice. `d_cosim` is the cleaner upstream generalization of
> this idea, and (since v45) also subsumes the GHDL/VHDL path.

### 2.3 Measured footprint (this machine, `~/nghdl-simulator`)

| Part | Size | Runtime-needed? |
|---|---|---|
| `src/` (ngspice source) | 35 MB | No (build-only) |
| `release/` (relink tree, 1423 `.o`, 22 MB obj bloat) | **83 MB** | No (build-only) |
| `install_dir/` (prebuilt ngspice + libs) | **7.3 MB** | **Yes** |
| `*.cm` code models (Ngveri.cm 110 KB, ghdl.cm 69 KB, …) | 656 KB | Yes |
| **Total** | **145 MB** | ~7.3 MB actually used |

**~118 MB exists only to support NgVeri's rebuild-the-code-model design.**

---

## 3. New architecture (`d_cosim`)

### 3.1 Mechanism

ngspice doc: *"The Verilog code is compiled with Verilator. Its resulting C code
then is compiled with some wrapper code (delivered by ngspice) into a shared
library, which is then loaded by the new code model `d_cosim`. The I/O ports of
the Verilog module are then directly accessible to the event based fast digital
ngspice/XSPICE simulator, or may directly connect to analog parts."*

Two engines, one code model:

```
                       eSim NgVeri tab  → engine select
                       ┌───────────────┴───────────────┐
                  Icarus Verilog                    Verilator ≥4.210
                       │                                 │
   iverilog -o m.vvp m.v   (bytecode, no C++)    vlnggen m.v
   + ivlng adapter (ships w/ ngspice)            → C++ + ngspice wrapper → m.so
                       │                                 │
                       ▼                                 ▼
            built ONCE per model, written to the PROJECT directory
                       └────────────────┬────────────────┘
                                        ▼
   netlister emits:
     a<id> [ <inputs> ] [ <outputs> ] [ <inouts> ] null <model>
     .model <model> d_cosim simulation="ivlng" sim_args=["<proj>/m"]     ; Icarus
     .model <model> d_cosim simulation="<proj>/m.so"                      ; Verilator
                                        ▼
                          ngspice ≥46  (event-driven node bus)
                                        ▼
                              <proj>.cir.out  →  open_ngspice()  (unchanged)
```

### 3.2 Netlist syntax (canonical)

```
* d_cosim Verilog DUT, Icarus engine
adut [ Clk Comp Start ] [ Sample Valid ~d5 ~d4 ~d3 ~d2 ~d1 ~d0 ] null dut
.model dut d_cosim simulation="ivlng" sim_args=["/path/proj/adc"]
```

Rules the netlister MUST honor:
- Bracket order: `[inputs] [outputs] [inouts]` then `null` then model name.
- Nodes matched **left-to-right against Verilog port declaration order**;
  vector port bits **MSB → LSB**.
- Analog↔digital coupling: ngspice **auto-inserts a bridge when an analog node
  and a digital node share the same name**. Explicit `dac_bridge` / `adc_bridge`
  remain available when level/threshold control is needed.

### 3.3 Footprint added

| Component | Linux | Windows | Note |
|---|---|---|---|
| Icarus Verilog (`iverilog`+`vvp`) | **6.5 MB** (2.1 MB compressed) | ~20–25 MB | default engine |
| `d_cosim` + `ivlng` adapter | ~0.2 MB | ~0.2 MB | inside ngspice build |
| ngspice ≥46 runtime | ~0 (already shipped) | ~0 | replaces current 7.3 MB |
| **Added (Icarus default)** | **~7 MB** | **~20–25 MB** | |

Per-model artifacts (`.vvp` ~10–50 KB, Verilator `.so` ~0.1–1 MB) live in the
**project dir, not the package** — they replace today's per-model `.o` bloat.

Verilator as opt-in (NOT bundled): +28 MB on Linux **plus** system g++; on
Windows MSYS2/MINGW64 ≈ 300 MB–1 GB+. This is why Icarus is the default and
Verilator is bring-your-own.

---

## 4. Old vs. New — side by side

| Axis | Legacy NgVeri (static `.cm`) | New `d_cosim` (Icarus default) |
|---|---|---|
| Per-model artifact | object linked into one fat `Ngveri.cm` | standalone `.vvp`/`.so` in project dir |
| Rebuild ngspice per model | **Yes — `make install`** | **No — runtime load** |
| ngspice source tree shipped | **Yes (~118 MB)** | **No** |
| User C/C++ compiler required | **Yes** (gcc / MinGW) | **No** (Icarus) / yes (Verilator opt-in) |
| Build steps | 5 stages (verilator→make→copy→make→install) | 1 subprocess (`iverilog`/`vlnggen`) |
| Windows pain | MSYS2 + manual `.cm` copy | iverilog prebuilt; no compiler |
| Runtime speed | compiled (fast) | Icarus interpreted (analog solve dominates) / Verilator fast |
| Maintenance | eSim-maintained static-link hack | tracks upstream ngspice |
| Failure modes | `make install` hang, path bugs | mostly packaging (lib path) |
| Net package size | baseline 145 MB | **−110 MB (lean)** or **+7 MB (full-parallel)** |

---

## 5. Mixed-schematic ("holy grail") — how coexistence works

All digital blocks become XSPICE **`A` devices** on ngspice's single embedded
**event-driven node engine**, regardless of how they were produced. One netlist:

```
* legacy NgVeri block — loaded via  codemodel Ngveri.cm
a1 [in0 in1] [out0] legacy_nand
.model legacy_nand d_cosim ...        ; (or legacy static model line)

* NEW d_cosim block — loaded via runtime shared lib / adapter
adut [ Clk Comp Start ] [ Sample Valid ~d5 ~d4 ~d3 ~d2 ~d1 ~d0 ] null dut
.model dut d_cosim simulation="ivlng" sim_args=["<proj>/adc"]

* optional C firmware block — d_process, pipe IPC
amcu [ ... ] [ ... ] null fw
.model fw d_process process="<proj>/firmware"
```

ngspice loads `Ngveri.cm` (legacy, via `codemodel`) **and** the `d_cosim`
shared lib/adapter (via `.model` at parse) **and** the `d_process` exe — all feed
the same event node bus and sync to the analog solver. **eSim adds no
orchestration**; ngspice schedules everything in one `.cir` run. This is the
core reason `d_cosim` satisfies the requirement with near-zero extra machinery.

---

## 6. How we build it — roadmap

Parallel, additive. Legacy code paths stay until explicitly retired.

### Phase 0 — Packaging
- Bundle prebuilt **ngspice ≥46** built with XSPICE + cosim enabled, plus the
  `ivlng` (and `vlng`) adapter libs, for Linux + Windows.
- Bundle **Icarus Verilog** (`iverilog`, `vvp`) for both OSes.
- Fix the known `ivlng.so` hardcoded-path issue (searches
  `/usr/local/lib/ngspice` regardless of `--prefix`) via launcher env var /
  symlink. (ngspice bug #772.)
- **Decision gate:** keep full legacy NgVeri *build* path (size +7 MB), or
  ship only prebuilt `Ngveri.cm` for legacy *simulation* and drop `src/` +
  `release/` (size −110 MB). Recommended: the lean option.

### Phase 1 — Model build (`src/maker/`)
- New path parallel to `ModelGeneration.run_verilator/make/copy/runMake/runMakeInstall`.
  - **Icarus:** `iverilog -o <proj>/<model>.vvp <model>.v` (+ supporting files).
  - **Verilator:** invoke `vlnggen` → `<proj>/<model>.so`.
  - Output goes to the **project directory**, not the ngspice tree.
  - One `QProcess` call; **no `make install`** → removes the hang bug class.
- `src/maker/NgVeri.py`: add engine selector to the tab —
  `[ Icarus (default) | Verilator (fast) ]`. Route to the new build path.
- Reuse existing port-extraction (`getPortInfo`) to record port order for the
  netlister.

### Phase 2 — Netlister (`src/kicadtoNgspice/`)
- `KicadtoNgspice.py:111` — add a branch parallel to `line[6] == "Nghdl"`, e.g.
  `line[6] == "NgVeriCosim"`, routing these blocks to the new emitter.
- `Convert.py` (model-line emission, ~`:370+`): emit
  - device line: `a<id> [ins] [outs] [inouts] null <model>`
  - model line: `.model <model> d_cosim simulation="ivlng" sim_args=["<proj>/<model>"]`
    (or `simulation="<proj>/<model>.so"` for Verilator).
- Honor MSB→LSB vector ordering; rely on auto-bridge, keep explicit bridge option.

### Phase 3 — Orchestration
- **Unchanged.** `Application.py:474 open_ngspice()` still runs `<proj>.cir.out`.
  ngspice dlopens the `.so` / loads the `ivlng` adapter at parse. Legacy
  `Ngveri.cm` still `codemodel`-loaded → coexists (Section 5).

### Phase 4 — KiCad symbols / library
- Generate `eSim_NgVeriCosim.kicad_sym` analogous to `eSim_Ngveri.kicad_sym`
  using the existing s-expression symbol writer.

### Phase 5 — Docs + steering
- Both flows remain. Default new users to `d_cosim` + Icarus; document Verilator
  opt-in; keep legacy `.cm` for existing projects.

---

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| `ivlng.so` hardcoded lib path (bug #772) | launcher env/symlink in Phase 0 |
| Verilator ≥4.210 hard requirement | gate Verilator engine on version check |
| Windows Verilator/MSYS2 fragility (bugs #706, #776) | Verilator opt-in only; Icarus default |
| v43 had broken d_cosim examples | pin to v46 (matured) |
| Vector port bit-order mismatch | enforce MSB→LSB in emitter + test |
| Bundled ngspice missing cosim/`ivlng` | verify in Phase 0 build QA |

---

## 8. References

- ngspice [release news (v42–v46)](https://ngspice.sourceforge.io/news.html) ·
  [special features: d_cosim / d_process / OSDI](https://ngspice.sourceforge.io/extras.html)
- Giles Atkinson, *Co-simulation Verilog – SAR ADC* ([PDF](https://ngspice.sourceforge.io/docs/others/Verilog-CoSim.pdf))
- d_process: ISOTEL [mixed-sim](https://www.isotel.eu/mixedsim/embedded/motorforce/) ·
  [merge request #6](https://sourceforge.net/p/ngspice/ngspice/merge-requests/6/)
- FOSDEM 2024 Holger Vogt, *ngspice + KiCad* ([slides](https://archive.fosdem.org/2024/events/attachments/fosdem-2024-2834-ngspice-circuit-simulator-stand-alone-and-embedded-into-kicad/slides/22676/ngspice-HolgerVogt_tEfhemB.pdf))
- Integration/packaging gotchas: [v43 d_cosim examples](https://sourceforge.net/p/ngspice/discussion/127605/thread/7eeb569dc8/) ·
  [bug #772 ivlng.so path](https://sourceforge.net/p/ngspice/bugs/772/) ·
  [#776 vlnggen/Verilator MSYS2](https://sourceforge.net/p/ngspice/bugs/776/) ·
  [#706 mingw install](https://sourceforge.net/p/ngspice/bugs/706/)
- ngspice mixed-signal syntax thread: [ngspice-tips](https://sourceforge.net/p/ngspice/discussion/ngspice-tips/thread/6a5b9dd2/)

### Code references (eSim, this repo)
- `src/maker/NgVeri.py:120-146` — legacy build orchestration
- `src/maker/ModelGeneration.py:834 / 886 / 923 / 983 / 1017` — verilator→make→install chain
- `src/kicadtoNgspice/KicadtoNgspice.py:111-114` — `line[6]` model-type branch
- `src/kicadtoNgspice/Convert.py:~370` — `.model` line emission
- `src/frontEnd/Application.py:474` — `open_ngspice()` runs `.cir.out`
- `nghdl/src/ghdlserver/ghdlserver.c` — legacy NGHDL socket server
