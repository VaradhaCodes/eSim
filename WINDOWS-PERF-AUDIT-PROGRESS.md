# Windows Perf Audit — Implementation Progress

Branch `windows-test`, base `b019d5fe`. 15 commits, one item per commit, no mixed
scope. Every code change ran the full pytest suite; the Windows baseline held at
**28 pre-existing failures** (POSIX-path/HOME + iverilog-toolchain flakiness)
throughout — **zero new failures** introduced. ~30 new tests added.

Test interpreter: `C:\FOSSEE\eSim\python\python.exe` (3.12, pytest 9.1.1),
`QT_QPA_PLATFORM=offscreen`.

## Done and verified here (unit + suite green)

| Audit | Commit | What |
|-------|--------|------|
| P0.4 | `b5e8cc38` | convertICintoBasicBlocks iterates a snapshot (was mutating list under iteration) |
| P0.2 | `9de920d2` | kicad-cli netlist export runs on a BackgroundJob, not the GUI thread (+ folded the dirty `_no_window_kwargs` STARTUPINFO fix) |
| P1.1 | `83340c97` | button-motion filters no longer stack on every dock open; motion_enabled() cached |
| P1.2 | `598f88ae` | app-wide motion filters gated on event type + merged into one |
| P1.4 | `e5d179d2` | sim console → bounded QPlainTextEdit, writes coalesced on a 50 ms timer |
| P0.3 | `39d645a1` | child-process shutdown batched under one shared wait budget |
| P1.6 | `c87fc644` | modelParamXML indexed once per conversion, O(1) lookups |
| P1.7 | `f7ff37c0` | matplotlib import made lazy for the Verilog verifier + background prewarm |
| P2.4 | `a183f301` | main window maximized when revealed, not built+discarded in __init__ |
| P3.3/P3.4 | `2b8cd2cb` | save_preferences merges (no key loss); all config writes atomic |

## Done but NEEDS YOUR VERIFICATION (live GUI / installer rebuild)

Unit-tested where possible, but the acceptance criterion can only be confirmed on
a headed Windows box or after an installer rebuild.

- **P0.1 — teardown crash** `e750a621`. Added ordered `app.aboutToQuit` cleanup
  (stop glows → disconnect colorSchemeChanged → stop widget timers → close mpl
  figures). **Verify:** 5 open-tools-then-exit cycles, `Get-WinEvent -LogName
  Application` shows no new Event 1000/1001 for pythonw at offset `0x10496`.
- **P2.1 — single interpreter** `71d83032`. Env setup + bootstrap now run
  in-process (`frontEnd/launcher_windows.py`); `esim.bat` drops the bootstrap
  interpreter and sets `ESIM_ENV_READY=1`. **Verify:** launch via the installed
  shortcut; eeschema/ngspice/kicad-cli still resolve, symbols still seed,
  `startup.log` shows one fewer interpreter start. **Follow-up (not done):**
  switch installer `[Icons]` to `pythonw Application.py` directly to also kill the
  cmd window — kept on the proven `esim.bat` until launcher_windows is verified.
- **P2.2/P2.3 — installer** `f4436824`. `[Run]` compileall + default-checked
  `[Tasks]` Defender exclusion (`Add-MpPreference`) + `[UninstallRun]` removal.
  **Verify:** rebuild with `windows/build-windows.ps1` (needs ISCC), silent-install,
  confirm `__pycache__` under `{app}/src`, `Get-MpPreference` lists `{app}`,
  uninstall removes it, cold `startup.log` below baseline.
- **P1.3 stage 1 — motion off on Windows** `0c890281`. `enable_motion` now
  defaults False on `nt`. **Verify:** app is snappier; Preferences toggle re-enables.
  **Stages 2–3 (not done):** replace per-button `QGraphicsDropShadowEffect` with
  QSS hover/press, keep ~10 real shadows, then delete `EffectShowRefreshFilter` +
  the `_refresh_graphics_effects` double-pass. Visual — needs a headed box.
- **P3.6 — AppUserModelID** `76c6d017`. Taskbar identity set before QApplication.
  Deferred P3.6 items (single-instance `QLocalServer`, `.proj` association,
  multi-size `.ico`) change launch behaviour — need live verification.

## NOT done — blocked / needs a real run or a product decision

- **P1.5 — two-phase plot load off the GUI thread. BLOCKED for blind
  implementation.** The heavy parse (`DataExtraction.openFile` +
  `computeAxes`) is the right worker payload, BUT `openFile` calls
  `Dialogs.make_error_message(None)` on its missing-file/error path — creating a
  QWidget on a worker thread is a crash hazard. Doing P1.5 safely first requires
  refactoring `data_extraction.py` to return an error status instead of showing
  dialogs, then splitting `plotWindow.load_simulation_data` into `_parse()`
  (worker) + `_apply(dataext)` (GUI), with a loading placeholder, a
  close-before-finish guard, and a synchronous fallback. Its acceptance (≥100 MB
  plot_data appears instantly, stays interactive) can only be validated with a
  real simulation. Left untouched rather than ship a threading hazard to the
  primary waveform feature. `VcdPlotWindow` overrides `load_simulation_data`
  wholesale (in-memory, synchronous), so it is unaffected by whatever the base
  class does.
- **P3.1** (single `procutil` spawn helper), **P3.2** (God-window decomposition),
  **P3.7** (skip-mark the 28 baseline failures) — broad mechanical refactors;
  P3.7 in particular risks masking real Windows bugs, so it needs a per-test
  judgement, not a blanket skip.
- **P3.5 — drop QWebEngine.** The audit says flag, don't do silently — it's a
  product decision (Makerchip/user-manual UX vs ~180 MB installer + RAM). Your call.

## Working-tree note

`src/maker/ModelGeneration.py` and `src/maker/NgVeri.py` remain modified — that's
a **separate NgVeri live-build-progress-bar feature** already in the tree, NOT
part of this audit. Deliberately left uncommitted; commit it on its own.
