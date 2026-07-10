# eSim Packaging

How eSim is packaged, why it is packaged that way, and the exact commands per
target. For the step-by-step release cookbook, see
[MAINTAINERS-PACKAGING.md](MAINTAINERS-PACKAGING.md).

## Targets

eSim ships exactly **two** targets. Fewer working, documented targets beat
many broken ones (see [Retired targets](#retired-targets)).

| Target | Artifact | Built by | Supports |
|---|---|---|---|
| Ubuntu | `dist/eSim-<VERSION>-ubuntu.zip` | `./make-release.sh` | Ubuntu 24.04 LTS, 26.04 LTS (23.04 / 25.04 best-effort) |
| Windows | `windows/dist/eSim-<VERSION>-installer.exe` | `windows\build-windows.ps1` | Windows 10 / 11, 64-bit |

Ubuntu 22.04 is **intentionally unsupported**: its archives ship Verilator 4
and KiCad 6, but eSim needs Verilator 5 (NgVeri build) and KiCad ≥ 7
(PyQt6-era netlister). Do not re-add it.

---

## Ubuntu target

### Build

```bash
./make-release.sh          # freezes the CURRENT working tree -> dist/
```

The script snapshots the working tree (committed + uncommitted state, minus
VCS/build cruft and regeneratable simulation outputs), flattens
`Ubuntu/install-eSim.sh` to the release root, packs `nghdl/` → `nghdl.zip`
and `library/kicadLibrary/` → `kicadLibrary.tar.xz`, writes a provenance
`RELEASE` stamp (version, commit, dirty flag, date), and emits a zip + sha256.

### Install

One unified, version-aware installer — `install-eSim.sh` — detects the Ubuntu
release via `detect_profile()` and adjusts only two things per release: where
KiCad comes from (PPA vs universe) and the minimum KiCad major. Everything
else is identical across releases. `--dry-run` previews the full plan;
`--install` is idempotent (reinstall over any prior eSim is clean);
`--uninstall` is version-agnostic.

Key design decisions (do not regress these):

* **apt for heavy native deps** (PyQt6, QScintilla, matplotlib, numpy,
  scipy) so they stay ABI-consistent with the system Qt. pip-pinning them is
  what made the pre-2026 per-version scripts fragile. The venv is created
  `--system-site-packages`; pip adds only pure-python extras (watchdog,
  hdlparse, makerchip-app, sandpiper-saas, volare).
* **KiCad symbol split**: eSim's 14 static symbol libraries go to
  `/usr/share/kicad/symbols/` **root-owned**; the 3 libraries eSim rewrites
  at runtime (`eSim_Ngveri`, `eSim_NgVeriCosim`, `eSim_Nghdl`) live in
  `~/.esim/kicad_symbols/` (registered in the user `sym-lib-table` with
  absolute paths; runtime code in `src/maker/kicad_symlib.py` lazily migrates
  pre-relocation user models and repairs stale table entries). **Never chown
  or delete anything belonging to KiCad's own packages.**
* **The simulation toolchain is wired in `nghdl/install-nghdl.sh`**, not the
  main installer. It source-builds the custom ngspice (d_cosim + ivlng) and
  Icarus Verilog — apt's iverilog lacks `libvvp`, which ngspice's `ivlng`
  adapter `dlopen`s at runtime — and installs `ghdl-llvm` (NOT the `ghdl`
  meta-package, which pulls the mcode backend and makes nghdl VHDL
  simulation fail silently; see
  `nghdl/install-nghdl-scripts/GHDL-BACKEND-26.04.md`). It also carries C23
  `bool` CFLAGS fixes. **Refine it, don't gut it, and keep its comments.**
  It now runs a preflight (disk/network/apt/tarball) before building and a
  self-check after (binary + `ghdl.cm`/`ivlng` + backend), and
  `install-eSim.sh` finishes by running the app's own toolchain doctor.

### The toolchain doctor

`src/maker/ToolchainCheck.py` probes every external tool the simulation
flows need (ngspice + code models + `ivlng`/`ghdl.cm`, iverilog/vvp/libvvp,
verilator, make, gcc, ghdl **backend included**, the nghdl tree, MSYS2 on
Windows) and reports found/missing with the exact probed path and a fix
hint. Three surfaces, one truth:

* `esim --doctor` (Ubuntu) / `esim.bat --doctor` (Windows) — headless report,
  exit code 1 when anything is missing (used by installer self-checks and CI).
* Help menu → *Check Simulation Toolchain* — the same report in a dialog.
* Pre-flow gates — NgVeri builds, d_cosim and the NGHDL tab check their own
  dependency subset before shelling out, so a missing tool produces an
  actionable message instead of a mid-build explosion or a silent no-op.

The probing logic is unit-tested on Linux with fake install trees
(`src/maker/tests/test_toolchain_check.py`), including the Windows-only
branches.

---

## Windows target

### Why this design

The previous Windows installer was a ~478 MB NSIS blob built off-repo from
tribal knowledge — unmaintainable by definition. The replacement lives
entirely in `windows/`:

| File | Role |
|---|---|
| `build-windows.ps1` | The one build command. Downloads pinned deps, builds the sim toolchain from source inside MSYS2, stages the tree, compiles the installer. |
| `deps-manifest.json` | Every third-party download: URL + version + sha256 + why. The build refuses hash mismatches. |
| `requirements-windows.txt` | The pip wheel set for the bundled Python. |
| `installer.iss` | Inno Setup script (readable, diffable — the reason for Inno over NSIS). |
| `esim.bat` | Relocatable launcher: prepends bundled tool paths (custom ngspice first) + `SPICE_LIB_DIR`, runs the bootstrap, starts the GUI. `esim.bat --doctor` prints the toolchain report. |
| `windows_bootstrap.py` | Per-user every-launch setup (`~/.esim/config.ini`, symbol seeding, the full `~/.nghdl/config.ini`, ngspice `spinit` relocation, KiCad `sym-lib-table` registration). Pure stdlib, OS-independent, unit-tested on Linux (`windows/tests/`). |
| `collect-logs.ps1` | Debug bundle for the shakedown loop: doctor report + `~/.esim` + `~/.nghdl` + spinit + code-model inventory → one zip on the Desktop. |

Design decisions:

* **Private bundled Python** (nuget full CPython) + pip wheels. On Windows
  there is no system Qt to stay consistent with — the PyQt6 wheel carries its
  own Qt — so wheels are the right source there, unlike Ubuntu.
* **KiCad is bundled, pruned, reproducibly.** Stage-Kicad extracts the
  pinned official KiCad installer's payload (7z reads the NSIS exe directly),
  drops what eSim never touches -- 3D models (784 of the 1057 MB!), demos,
  translations, python plugin extras -- and stages it at `tools\kicad`. One
  eSim exe therefore carries the whole tool; nothing else to download.
  This supersedes the earlier ship-alongside design. The old objection
  ("a private KiCad copy is what made the pre-2026 blob rot") indicted a
  HAND-maintained off-repo repack; this one is a manifest bump + rebuild,
  and the build hard-verifies the pruned tree (kicad-cli version + a real
  netlist export) every time. The bundled copy is private to eSim:
  esim.bat prepends `tools\kicad\bin` to PATH, no registry / file
  associations / global env vars, so a user's own KiCad install coexists
  untouched.
* **Per-user state happens at launch, not install** (`windows_bootstrap.py`
  runs from `esim.bat` every start, idempotently). Multi-user machines and
  upgrades self-heal, and the logic is testable.
* **Space-free default install root** (`C:\FOSSEE\eSim`): the MSYS2/mingw
  toolchain and code-model paths break subtly under `Program Files`. The
  installer grants `users-modify` on the tree because HDL model builds write
  into it by design (exactly like the Ubuntu install owns
  `$HOME/nghdl-simulator`).
* **The custom eSim ngspice is built from source on Windows too.**
  `Stage-SimToolchain` compiles `nghdl/nghdl-simulator-source.tar.xz`
  (ngspice-45.2 + the nghdl delta baked in — no separate patch) inside
  the staged MSYS2 (mingw64) with the same flags as
  `install-nghdl.sh` on Ubuntu, into `tools\nghdl\{src,release,install_dir}` —
  the exact `$HOME/nghdl-simulator` layout, so every `~/.nghdl/config.ini`
  key means the same thing on both OSes. The official ngspice zip is staged
  at `tools\ngspice` only as the Compact flavour's plain-simulation fallback
  (it lacks `ghdl.cm`/`Ngveri.cm`, so VHDL/NgVeri co-sim need the custom
  build). It is a **console** build (no `--with-wingui`), like Ubuntu: eSim
  drives ngspice through QProcess and parses stdout, which the wingui build
  hijacks into its own window.
* **Icarus Verilog is built from the pinned source with `--enable-libvvp`**
  (the same `ICARUS_REF` commit `install-nghdl.sh` uses). No prebuilt Windows
  Icarus ships `libvvp`, and ngspice's `ivlng` adapter dlopens it — this is
  the one piece d_cosim cannot live without. The Bleyer prebuilt is only the
  `-SkipSimBuild` fallback (Verifier works, d_cosim doesn't).
* **GHDL comes from MSYS2's `mingw-w64-x86_64-ghdl-llvm`** — pinned to the
  llvm backend explicitly; the mcode trap from `GHDL-BACKEND-26.04.md`
  applies on Windows too, and the build hard-fails if the staged ghdl
  reports mcode.
* **Two installer flavours** via Inno components: *Full* (MSYS2
  gcc/make/verilator/ghdl + the nghdl src/release trees → NgVeri builds,
  d_cosim, NGHDL VHDL co-sim) and *Compact* (simulation/verifier only; much
  smaller). The custom ngspice runtime (`tools\nghdl\install_dir`) ships in
  **both** — all simulation runs through it.
* **Every stage hard-verifies its output**: ngspice must answer
  `--version`, carry `analog/digital/ghdl/Ngveri.cm` + the `ivlng` adapter
  and pass a trivial `.cir` batch run; iverilog must stage `iverilog.exe`,
  `vvp.exe` **and** `libvvp`; ghdl must not be mcode. A dependency problem
  kills the build with the exact missing path, not the user's install.

### Build

On a Windows machine (or `windows-latest` CI runner) with 7-Zip installed:

```powershell
git clone <repo> ; cd eSim
powershell -ExecutionPolicy Bypass -File windows\build-windows.ps1
# variants:
#   -SkipMsys           no MSYS2 component at all (implies -SkipSimBuild)
#   -SkipSimBuild       skip the source builds; official-ngspice shim +
#                       Bleyer iverilog (NO d_cosim / VHDL co-sim). For
#                       quick packaging iterations only.
#   -AcceptNewHashes    record hashes for newly-bumped manifest entries
#   -Clean              rebuild staging from scratch
```

Artifacts land in `windows\dist\`: ONE eSim installer exe (with the pruned
KiCad bundled inside at `tools\kicad`) and its `.sha256`.

### What works / what doesn't on Windows

Status legend: ✅ = verified on a clean Windows VM; 🧪 = fully scripted and
Linux-unit-tested, **awaiting the first Windows VM shakedown** (rows flip to
✅ only after checklist W1–W13 in MAINTAINERS-PACKAGING.md passes). Run
`esim.bat --doctor` (or Help → *Check Simulation Toolchain*) on any install
to see the live truth for that machine.

| Feature | Status | Why |
|---|---|---|
| Schematic + ngspice simulation + plotting | 🧪 | Custom eSim ngspice (console build) at `tools\nghdl\install_dir`; official zip only as Compact fallback |
| Verilog Verifier (iverilog) | 🧪 | Source-built Icarus staged at `library/bin/iverilog/` where `CosimConfig` probes first |
| NgVeri code-model builds | 🧪 *Full* flavour only | mingw gcc/make/verilator via `MSYS_HOME`; doctor-gated with per-tool errors |
| NgVeri d_cosim (Icarus) flow | 🧪 *Full* | Custom ngspice carries `d_cosim`/`ivlng`; iverilog built `--enable-libvvp` |
| NGHDL / GHDL VHDL co-simulation | 🧪 *Full* | Custom ngspice (`ghdl.cm`) + MSYS2 `ghdl-llvm` + staged `nghdl/src` python/ghdlserver (`_WIN32` socket code already in-tree); Winsock now linked via `-lws2_32` |
| SKY130 / IHP PDKs | ❌ not shipped | Analog PDK flows are Ubuntu-only today; deliberately lower priority than the HDL toolchain (revisit after W1–W13 pass) |

### Windows shakedown (first VM run)

Copy-pasteable loop for the first build + test round on a Windows 10/11 box
with 7-Zip and git:

```powershell
git clone <repo> ; cd eSim
powershell -ExecutionPolicy Bypass -File windows\build-windows.ps1 -AcceptNewHashes
# (verify the recorded hashes in windows\deps-manifest.json against upstream,
#  then commit them)
windows\dist\eSim-<VERSION>-installer.exe     # choose Full
C:\FOSSEE\eSim\esim.bat --doctor              # must be all-OK
C:\FOSSEE\eSim\esim.bat                       # then W1-W13 from MAINTAINERS-PACKAGING.md
```

When anything fails, run
`powershell -ExecutionPolicy Bypass -File C:\FOSSEE\eSim\windows\collect-logs.ps1`
and send the zip it drops on the Desktop.

---

## Dependency matrix

| Dependency | Ubuntu source | Windows source | Why |
|---|---|---|---|
| Python 3 | system (`python3-full`, venv `--system-site-packages`) | bundled private CPython (nuget) | Ubuntu: one Python, ABI-consistent with apt Qt. Windows: no system Python to rely on. |
| PyQt6 + QScintilla | apt `python3-pyqt6`, `python3-pyqt6.qsci` | pip wheels `PyQt6`, `PyQt6-QScintilla` | Apt keeps Qt ABI-consistent on Ubuntu; wheels bundle their own Qt on Windows. |
| matplotlib / numpy / scipy / psutil | apt | pip wheels | Same reasoning as Qt. |
| watchdog, hdlparse, makerchip-app, sandpiper-saas, volare | pip (venv) | pip (bundled Python) | Pure-python / not packaged by distros. |
| KiCad 8/9 | apt (PPA on 24.04, universe on 26.04) | official installer payload, pruned by Stage-Kicad, bundled at `tools\kicad` | Ubuntu: never touch KiCad's packages. Windows: reproducible prune of the pinned official payload (no 3D models/demos/translations); KiCad's own libraries inside it stay untouched. |
| ngspice (d_cosim + ivlng + ghdl.cm) | source-built by `nghdl/install-nghdl.sh` | source-built by `Stage-SimToolchain` inside MSYS2 (same tarball, same patch, same flags) → `tools\nghdl` | The ONLY ngspice with the eSim co-sim bridges; both OSes now run the identical custom build. Official Windows zip ships at `tools\ngspice` purely as the Compact fallback. |
| Icarus Verilog (`libvvp`) | source-built by `nghdl/install-nghdl.sh` (apt's lacks `libvvp`) | source-built by `Stage-SimToolchain` at the SAME pinned commit, `--enable-libvvp`, staged under `library/bin/iverilog/` | `ivlng` dlopens `libvvp` at runtime on both OSes; no prebuilt Windows Icarus ships it (Bleyer = `-SkipSimBuild` fallback only). |
| Verilator 5 | apt | MSYS2 `mingw-w64-x86_64-verilator` (Full flavour) | NgVeri model builds. |
| GHDL | apt `ghdl-llvm` (**never** the `ghdl` meta → mcode) | MSYS2 `mingw-w64-x86_64-ghdl-llvm` (Full flavour); build hard-fails on an mcode backend | See `GHDL-BACKEND-26.04.md` — the trap is identical on both OSes. |
| gcc/make | apt `build-essential` | MSYS2 mingw64 (Full flavour) | Runtime code-model compilation. |
| nghdl python + ghdlserver sources | shipped in `nghdl/` (softlinked `nghdl` launcher) | staged at `<install>\nghdl\` (`SRC_HOME`), embedded in the Makerchip VHDL tab | `ngspice_ghdl.py` copies `src/ghdlserver/*` into each model's `DUTghdl/` at build time. |
| SKY130 / IHP PDK | bundled tarball / `ihp/` script | not shipped | Ubuntu-only flows. |

---

## Retired targets

`flatpak/`, `snap/`, `appimage/` and `docker-launcher/` were removed in 2026
(git history has them). All four targeted the pre-2026 PyQt5 application and
each was broken or unmaintainable in its own way:

* **snap** — a copy of KiCad's own snapcraft.yaml (description included),
  `core22` base (= the unsupported 22.04 toolchain), PyQt5. Never started in
  earnest.
* **flatpak** — PyQt5, pip-pinned native deps; and fundamentally at odds with
  eSim, which compiles code models at runtime with gcc/make/verilator —
  inside a strict sandbox that means shipping a whole SDK.
* **appimage** — a 7,875-line build script embedding patched application code
  inline. Unmaintainable by construction.
* **docker-launcher** — `ubuntu:22.04` base, PyQt5-era pins, plus a separate
  PyInstaller launcher app to maintain and sign.

If someone wants one of these back, the bar is: it must install the PyQt6
app, carry the full simulation toolchain (including runtime compilation), and
come with a named maintainer. Start from the git history, not from scratch.
