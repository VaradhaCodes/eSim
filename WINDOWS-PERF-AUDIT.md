# eSim Windows Performance & Stability Audit

**Date:** 2026-07-11 · **Branch:** `windows-test` @ `b019d5fe` · **Scope:** full `src/` + `windows/` runtime paths.

This is a prescriptive work plan. Each item has: the problem, the evidence (file:line), the fix, and an acceptance test. Items are ordered by priority — do P0 first, then P1, etc. Within a priority band, order is by (impact ÷ effort).

**Ground rules for the implementing model:**
- One item = one commit. Never mix items in a commit.
- Never add `Co-Authored-By: Claude` trailers to commits (user rule).
- After every change, run the relevant tests plus a real GUI smoke launch (`windows\build\eSim` staged tree or the repo tree via `src/frontEnd/Application.py`). The repo has pytest suites under `src/*/tests/`; 27–28 pre-existing failures on Windows are POSIX-path/HOME mismatches — the baseline. Do not "fix" those as a side effect; compare fail lists before/after.
- The installed tree at `C:\FOSSEE\eSim` must be kept in sync when patching (or reinstalled from a rebuilt installer). The current `windows\dist\eSim-2.5-installer.exe` (659 MB) predates commit `b019d5fe`; any installer rebuild must include everything on `windows-test` HEAD.

---

## P0 — Stability (crashes, hangs, GUI-thread blocking)

### P0.1 Recurring exit crash: 0xc0000005 in `PyQt6\sip.cp312-win_amd64.pyd`, offset `0x10496`
WER shows this crash on 07-08, 07-09, 07-10 (×2), 07-11 — same offset every time, always around app teardown. It is currently invisible to users (process is already closing) but it is a real use-after-free and will eventually corrupt state *before* the window closes.

**Diagnosis plan (do this before fixing):**
1. Reproduce headless: script that builds `Application`, opens a plot dock + a Model Creation dock, then closes via `appView.close()` and lets `app.exec()` return naturally. Run under `python -X faulthandler`; capture the faulting Python frame.
2. Prime suspects, in order:
   - **Animations/effects outliving widgets:** `motion.py` runs `QVariantAnimation`s with `DeleteWhenStopped` against `QGraphicsDropShadowEffect`s; at quit, widgets die while animations may still tick. `theme_utils.apply_theme` already had to add `stop_all_glow()` for exactly this class of segfault (`src/frontEnd/theme_utils.py:219-226`). Nothing calls `stop_all_glow()` at quit.
   - **`colorSchemeChanged` connection** to `_apply_theme` (`Application.py:1582`) firing during teardown.
   - **matplotlib figures** in plot docks not closed when quitting with docks open (closeEvent of plotWindow handles it, but only if the dock close path actually runs before interpreter teardown).
   - **`WorkerThread.__del__` calling `self.wait()`** during interpreter shutdown (`src/projManagement/Worker.py:108-122`) — QThread destruction at exit is a classic sip-crash source.
3. **Fix pattern:** add a single `app.aboutToQuit` handler installed in `main()` that, in order: calls `motion.stop_all_glow()`, disconnects `colorSchemeChanged`, stops all `plotWindow` timers/figures by iterating `app.topLevelWidgets()` + dock children, and asks the DockArea to `closeDock()` for every project. Then verify the WER crash stops recurring across ≥5 launch/exit cycles (check `Get-WinEvent -LogName Application` for Event 1000/1001 with pythonw).

**Acceptance:** 5 consecutive open-tools-then-exit cycles with zero new WER entries; pytest baseline unchanged.

### P0.2 `kicad-cli` netlist export runs synchronously on the GUI thread
`Kicad.openKicadToNgspice()` (`src/projManagement/Kicad.py:202`) calls `KicadNetlister.generate_netlist()`, which does `subprocess.run(..., timeout=120)` (`src/kicadtoNgspice/KicadNetlister.py:274`). On a cold Windows boot, `kicad-cli` start (Defender scan + KiCad DLL load) can take 5–15 s; the entire eSim UI is frozen ("Not Responding" — exactly the user complaint) for that time, and up to 120 s worst case.

**Fix:** run `generate_netlist` on a worker (`QThread` or `concurrent.futures` + `QMetaObject.invokeMethod`/signal back). UI flow: disable the Convert action, show status-bar "Generating KiCad netlist…" (`Appconfig.print_info` already mirrors to the status bar), then on completion continue into `validateCir` + `kicadToNgspiceEditor` **on the GUI thread**. On failure, show the existing error dialog. Keep the synchronous path only for the `__main__` CLI mode of KicadNetlister.

Also: fold the uncommitted `_no_window_kwargs()` STARTUPINFO fix (currently dirty in the working tree) into the same commit — it is correct, but see P3.1 for where it should live.

**Acceptance:** clicking Convert with a cold kicad-cli never blocks the event loop > 100 ms (verify: QElapsedTimer log around the handler, or click-drag the window during conversion — it must stay movable). Existing converter tests pass.

### P0.3 Blocking child-process shutdown in `closeEvent`
`Application.closeEvent` (`src/frontEnd/Application.py:954-957`) and `Worker.terminate_handle` (`src/projManagement/Worker.py:36-61`) do `terminate → waitForFinished(2000) → kill → waitForFinished(1000)` / `wait(timeout=2)` **per child, serially, on the GUI thread**. With eeschema + ngspice_gui + a plot session open, exit can freeze the UI for 3 s × N children — users read that as a hang and force-kill (which is plausibly feeding P0.1's teardown crashes).

**Fix:** in `closeEvent`, first send `terminate()` to *all* handles without waiting, then do one bounded wait pass (e.g. total budget 2 s across all children), then `kill()` stragglers. Simplest: move the wait/kill escalation into a small helper that takes a list and a total deadline.

**Acceptance:** with 3 external windows open, exit completes < 2.5 s wall and the UI repaints during it.

### P0.4 `Processing.convertICintoBasicBlocks` mutates the list it iterates
`src/kicadtoNgspice/Processing.py:270-280`: `for compline in schematicInfo: ... schematicInfo.remove(compline)`. Removing during iteration **skips the element after every removed one** — two adjacent `u`-prefixed components means the second is silently not processed (wrong netlist, no error). Classic latent correctness bug; also `index = schematicInfo.index(compline)` is O(n) and wrong after mutations.

**Fix:** iterate over a copy (`for compline in list(schematicInfo):`) or build a new list of survivors. Add a characterization test with two consecutive `u` components (the existing `tests/test_convert_characterization.py` is the right home).

**Acceptance:** new test proves both adjacent ICs are processed; full kicadtoNgspice suite passes.

---

## P1 — Responsiveness (the "laggy" complaints)

### P1.1 Event-filter accumulation: every dock open re-installs button-glow filters on ALL dock buttons
`DockArea.apply_fullscreen_feature` calls `install_button_motion(self)` on **every dock open** (`src/frontEnd/DockArea.py:268-269`). `install_button_motion` (`src/frontEnd/motion.py:224-249`) creates a **new** `TactileButtonFilter` each call and installs it on every `QPushButton`/`QToolButton` found under the DockArea. `installEventFilter` with a *different* filter object stacks — it does not replace. After opening N docks in a session, **every button in the dock area has N event filters**, each Enter/Leave spawns N glow animations fighting over one shadow effect. This degrades steadily over a session — the exact "gets laggy over time" signature — and the DockArea lives for the whole app lifetime, so it never resets.

**Fix (pick one, first is best):**
1. Install **one** app-level or DockArea-level `TactileButtonFilter` once in `DockArea.__init__`, and in `apply_fullscreen_feature` only install the filter on the *new* dock's buttons (`card.findChildren(...)`), tagging each button with a dynamic property (e.g. `_esim_motion_installed`) and skipping tagged ones.
2. Keep per-call installation but reuse the same filter object stored on the root, and still skip already-tagged buttons (needed so shadows aren't recreated either).

Also fix `motion_enabled()` (`motion.py:205-221`): it re-reads `preferences.json` from disk on every call — cache it in a module global invalidated by `apply_theme`/Preferences save.

**Acceptance:** open 10 docks, then hover a button: exactly one animation starts (instrument `_animate_glow` with a counter). No perceptible hover lag after 20 dock opens.

### P1.2 Two app-wide Python event filters see every event in the process
`install_popup_motion(app)` and `install_effect_refresh(app)` (`Application.py:1598-1600`, `motion.py:348-390`) install `QApplication`-level filters. Every event for every object — including every `MouseMove`, `Paint`, `Timer` — crosses C++→Python twice and runs isinstance checks. On Windows software rendering with big widget trees this is a constant tax on everything (dragging, resizing, typing).

**Fix:**
- `PopupMotionFilter` only cares about `Polish` and `Show` on menus/combo containers/treeviews. First statement should be a cheap event-type gate: `if et not in (Polish, Show): return False` before *any* isinstance work.
- `EffectShowRefreshFilter` only cares about `Show`: same gate (it already checks type first — keep it first, and merge the two filters into one object so each event crosses the boundary once, not twice).
- Longer-term: drop `EffectShowRefreshFilter` entirely once P1.3 reduces where drop shadows exist.

**Acceptance:** with a profiler (py-spy sampling while wiggling the mouse over the main window for 10 s) time spent in `eventFilter` drops to < 1% of samples.

### P1.3 QGraphicsDropShadowEffect on nearly every button/panel = CPU-blur repaints
`install_button_motion` gives every non-toolbar button its own `QGraphicsDropShadowEffect` (`motion.py:240-249`); toolbars, project tree, Welcome cards, popups get more (`apply_toolbar_depth`, `elevate`, `apply_panel_depth`). Each effect forces render-to-pixmap + gaussian blur **on the CPU** every time the source repaints. This is the single biggest structural drag on Windows (no compositor help for widget effects), and it's why the app feels heavier than stock Qt apps. It also interacts with the P0.1 teardown crash and required the whole effect-revalidation machinery (`_refresh_graphics_effects`, `EffectShowRefreshFilter`).

**Fix (staged):**
1. Change the **default** of `enable_motion` to `False` on Windows (keep True elsewhere if desired): one-line in `Appconfig.load_preferences` + `motion_enabled()`. Users get a visibly snappier app; the Preferences toggle still opts in.
2. Replace "shadow on every button" with QSS-only hover/press styling (background/border color shifts — already largely present in the QSS). Keep real shadows only on: toolbars (2), menus/popups, and the Welcome cards. That's ~10 effects instead of hundreds.
3. Once (2) lands, delete `EffectShowRefreshFilter` and the deferred `_refresh_graphics_effects` double-pass in `apply_theme` (`theme_utils.py:325-326`) for widgets that no longer carry effects.

**Acceptance:** window resize of a session with plot + editor docks stays smooth (no visible tearing/lag); py-spy during resize shows blur/effect frames gone.

### P1.4 Simulation console: unbounded rich-text QTextEdit fed per-chunk
`TerminalUi.simulationConsole` is a `QTextEdit` (`src/frontEnd/TerminalUi.py:42-45`) and `NgspiceWidget._handle_stdout/_handle_stderr` (`src/ngspiceSimulation/NgspiceWidget.py:281-309`) call `insertPlainText` on every `readyRead` burst. A chatty simulation (long transient with prints, or ngspice warnings per timestep) produces thousands of inserts into an unbounded rich-text document → document relayout each time → GUI stalls during simulation and memory grows without limit.

**Fix:**
1. Swap the console to `QPlainTextEdit` in `TerminalUi.ui` + set `maximumBlockCount` (e.g. 20 000 lines). `QPlainTextEdit` is designed for logs and is dramatically cheaper. The three `append(...)` HTML status lines (`SUCCESS_FORMAT` etc.) become `appendHtml`.
2. Coalesce writes: buffer decoded text in a list; flush on a 50 ms `QTimer` (started on first chunk, single-shot) with one `insertPlainText` per flush. Auto-scroll only when the scrollbar is already at bottom (preserve user scroll-back).

**Acceptance:** run a netlist that produces ≥ 50 000 output lines; UI stays interactive during the run; memory of the dock stable; cancel button responsive.

### P1.5 Plot data load (`np.loadtxt` over plot_data files) on the GUI thread
`plotWindow.__init__` → `load_simulation_data()` → `DataExtraction.openFile()` + `computeAxes()` (`src/ngspiceSimulation/plot_window.py:133,795-802`; `data_extraction.py:286`) reads and parses the whole `plot_data_*.txt` set synchronously while `DockArea.plottingEditor` is constructing the dock. For long transients (10⁵–10⁷ rows × 45 columns as in the 7805 tests) the app freezes for the whole parse, right after the user was told "Simulation Completed".

**Fix:** two-phase construction. `plotWindow` builds its (empty) UI immediately with a "Loading simulation data…" placeholder (a `_empty_placeholder` already exists), then parses in a `QThread`/`QThreadPool` worker that hands back the numpy arrays via a signal; `populate_waveform_list` + first draw run on the GUI thread on arrival. Guard: if the widget is closed before the worker finishes, discard the result (weakref or a cancelled flag checked in the slot).

**Acceptance:** simulate a netlist whose plot_data is ≥ 100 MB: the plot dock appears instantly with the loading state; main window stays interactive; waveforms appear when ready. Existing plot tests (69 dock/plot tests) pass.

### P1.6 `Processing` model lookup walks the whole modelParamXML tree per component
`src/kicadtoNgspice/Processing.py:296-302`: for **every** `u*` component, `os.walk` over `library/modelParamXML` plus `os.listdir` per subdirectory. A schematic with 20 ICs re-walks the tree 20 times; on cold NTFS with Defender each walk is disk-bound. This is inside the converter-open path (GUI thread, after P0.2's kicad-cli step).

**Fix:** build the index once per conversion: one `os.walk` producing `{filename: [paths]}`, then O(1) lookups per component. ~10 lines. (Note `modelxmlDIR` is a class attribute — keep the index per-call, not cached across calls, so newly created models are seen.)

**Acceptance:** converter opens with the same grouping results (characterization tests pass); conversion of a 20-IC schematic does exactly one tree walk (assert via monkeypatched `os.walk` counter in a test).

### P1.7 First click on Makerchip/NgVeri still pays the full matplotlib import
Known from the b019d5fe work: `maker/VerilogVerifier.py:18` imports `plotWindow` (→ matplotlib + numpy) at module level because `VcdPlotWindow(plotWindow)` subclasses it at line 64. So the first Model Creation open freezes for the matplotlib import (seconds, cold).

**Fix:** make the subclass lazy — factory function `make_vcd_plot_window(...)` that does `from ngspiceSimulation.plot_window import plotWindow` inside, defines the subclass once (module-level cache), and returns an instance. `VerilogVerifier` only instantiates plots in the Verify stage, so authoring/UI opens pay nothing.

Additionally, pre-warm the heavy imports in the background after startup settles: in `main()` after `app.exec()` is entered, `QTimer.singleShot(3000, ...)` → `threading.Thread(target=lambda: __import__('matplotlib.pyplot'))` (daemon). Python's import lock makes this safe; the first plot click then finds matplotlib already in `sys.modules`. Gate it behind `if os.name == 'nt'` if desired.

**Acceptance:** `python -c "import maker.VerilogVerifier"` (with src on path) does NOT pull matplotlib (assert `'matplotlib' not in sys.modules` in a test — same pattern as the existing lazy-import tests); Makerchip dock opens < 1 s warm.

---

## P2 — Startup time

Current: ~5 s warm, ~19 s+ cold (Defender). The lazy-import work (b019d5fe) already fixed the worst. Remaining levers:

### P2.1 Kill the double interpreter + the .bat entirely
`windows\esim.bat` starts `python.exe windows_bootstrap.py` **synchronously** (line 31), then `start pythonw.exe Application.py` (line 44). That's two full interpreter cold-starts per launch, plus a cmd window (currently hidden by the minimized-shortcut workaround).

**Fix:** move the launcher logic into Python:
1. `Application.main()` (or a tiny `src/frontEnd/launcher_windows.py` it calls first, `if os.name == 'nt'`) sets the PATH/SPICE_LIB_DIR env vars exactly as esim.bat does (bundled ngspice → nghdl install_dir → system KiCad scan → bundled KiCad; `os.environ` changes propagate to all children).
2. Run `windows_bootstrap.main()` **in-process** (it's already importable, idempotent, pure-stdlib) — before the QApplication is created, or even in a background thread since nothing it does is needed until a tool launches (symbol seeding must complete before eeschema starts; a `threading.Thread` joined lazily by `Kicad.openSchematic` covers that).
3. Shortcuts then point at `pythonw.exe ...\Application.py` directly (installer.iss `[Icons]` change) — no cmd window exists at all, `runminimized` hack gone, one interpreter start instead of two.
4. Keep `esim.bat` as a thin `--doctor`/`--debug` convenience wrapper.

**Acceptance:** warm launch wall time measured by `startup.log` drops ≥ 1 s; no console window appears even with default shortcut flags; `esim.bat --doctor` still works.

### P2.2 Precompile bytecode at install time
Nothing runs `compileall` at install (checked `installer.iss`, `build-windows.ps1`). First cold launch compiles every `.py` on import (and Defender scans each fresh `.pyc` write).

**Fix:** installer `[Run]` entry: `python.exe -m compileall -q -j 0 src windows` (and site-packages if not already compiled) after file copy. Or run it in `build-windows.ps1` at stage time so the `.pyc`s ship inside the installer (better: scanned once at install, no post-install step).

**Acceptance:** fresh install → first launch `startup.log` total noticeably below previous cold baseline; `__pycache__` dirs exist post-install.

### P2.3 Offer the Defender exclusion from the installer
The single biggest cold-start lever (measured this week: AppHangB1s disappeared after `Add-MpPreference -ExclusionPath 'C:\FOSSEE\eSim'`). Today it was added manually.

**Fix:** installer.iss `[Tasks]` checkbox (default **checked**, clearly labelled "Exclude eSim from Microsoft Defender real-time scanning (recommended, improves start-up)") + `[Run]` PowerShell `Add-MpPreference -ExclusionPath "{app}"` guarded by the task; `[UninstallRun]` `Remove-MpPreference`. The installer already runs elevated (`PrivilegesRequired=admin`). Users who decline just get slower cold starts.

**Acceptance:** fresh install with the box checked → `Get-MpPreference | Select -Expand ExclusionPath` contains the install dir; uninstall removes it.

### P2.4 Don't build+show the maximized main window before it's needed
`Application.__init__` calls `self.showMaximized()` (`Application.py:142`), then `main()` immediately `appView.hide()`s it (`Application.py:1679`) before the workspace flow decides what to show. That's a full layout+paint of the maximized window thrown away behind the splash (and it's one of the transient 'eSim' windows the flash-watcher used to see).

**Fix:** remove `showMaximized()` from `__init__`; let the workspace path (`defaultWorkspace()` / picker accept) call it. Verify with the existing `watch_windows.ps1` that no extra top-level map happens.

**Acceptance:** startup shows only splash → main window; `startup.log` 'main window built' stage shrinks; dock/lifecycle tests pass.

### P2.5 Trim `_refresh_recent_projects_menu` disk hits
`Application.py:647`: `os.path.isdir(p)` for every remembered project on every menu rebuild — fine on SSD, but stalls on dead network paths (UNC of an unplugged drive can block seconds). Build the menu without the check and validate on activation (the `_open_recent_project` error path already handles missing dirs), or check with a 0-cost `os.scandir` of the workspace once.

---

## P3 — Architecture & code health (do after P0–P2)

### P3.1 One subprocess-spawning helper for the whole codebase
There are ≥ 10 scattered spawn sites each hand-rolling `creationflags=CREATE_NO_WINDOW` (Worker.py:177, ModelGeneration.py:235, hdl/icarus.py:96, converters ×4, UserManual.py:47, nghdl's ngspice_ghdl.py, ToolchainCheck.py:91, KicadNetlister.py:274…). The new uncommitted KicadNetlister fix proves the pattern is *still* incomplete (STARTUPINFO+SW_HIDE needed on top). Every new call site is a fresh chance to regress the console-flash bug class.

**Fix:** create `src/configuration/procutil.py`:
```python
def hidden_kwargs() -> dict        # STARTUPINFO(SW_HIDE) + CREATE_NO_WINDOW on nt, {} elsewhere
def run(cmd, **kw)                 # subprocess.run(cmd, **hidden_kwargs() | kw)
def popen(cmd, **kw)               # same for Popen
```
Migrate every site mechanically; add a repo test that greps `src/` for raw `subprocess.(run|Popen|call)` outside `procutil.py` and fails on new offenders (allowlist the vendored `nghdl/src` copy if byte-identity matters — check the drift-guard test first; if `nghdl/src/kicad_symlib.py` is the only byte-locked file, ngspice_ghdl.py can migrate too, else leave vendored files and note it).

### P3.2 Decompose the God-window couplings
Not urgent, but the recurring bug pattern (converter globals, TrackWidget class-state, `_closeExistingConverters` single-instance hack in DockArea.py:635) traces to shared mutable class-level state as a data bus. When touching KicadtoNgspice next, move per-conversion state into an object created per dock and passed down. Do **not** attempt a big-bang rewrite; convert one consumer at a time behind the existing characterization tests.

### P3.3 Preferences write path loses keys
`Appconfig.save_preferences` (`src/configuration/Appconfig.py:267-282`) rewrites `preferences.json` with only 4 keys — dropping `zoom_level`, `enable_motion`, and any future key — while three other sites (`Application._toggle_theme`, `change_zoom`, `TerminalUi._cycle_theme`) each hand-roll read-modify-write with `json.dump`. PreferencesDialog partially compensates (it re-writes `existing` afterwards at lines 356-366/434-447), but the raw method is a data-loss trap.

**Fix:** single `save_preferences(**updates)` that reads existing, merges, writes atomically (tmp + `os.replace`, like modelCache does); migrate the 4 call sites. Add a test: set zoom → toggle theme → zoom preserved.

### P3.4 Config/JSON writes are scattered and non-atomic
`preferences.json`, `last_project.json`, `.projectExplorer.txt`, plot config all use bare `open(w)+json.dump` (crash mid-write = corrupt file; the workspace.txt truncation bug already bit once — `Application.py:1684-1690`). Standardize on one `write_json_atomic(path, data)` in `configuration/paths.py` and use it everywhere. plot_window.save_config already does tmp+replace — reuse the pattern.

### P3.5 Consider dropping QWebEngine (size + memory + startup of Makerchip docks)
`browser/UserManual.py` and the Makerchip dock use `QWebEngineView` — that's a full Chromium: ~180 MB of the installer, ~150–300 MB RAM per live view, slow first spawn, and its GPU process is another Defender cold-scan victim. Makerchip is a **website** (makerchip.com IDE) and the user manual is static HTML.

**Options:** (a) open both in the system browser (`webbrowser.open`) and shrink the dock to a "launch + instructions" panel — saves installer size, RAM, and removes a whole process family; (b) keep as is. This is a product decision — flag to the user, don't do it silently. If (a): remove PyQt6-WebEngine from `requirements-windows.txt` + stage prune, expect ~180 MB installer reduction.

### P3.6 Windows platform polish (small, high perceived quality)
- **AppUserModelID:** call `ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("FOSSEE.eSim.2.5")` before the QApplication is created, and set the same AppId on installer shortcuts — fixes taskbar grouping/pinning showing the generic pythonw icon.
- **Window icon on taskbar** then follows the `.ico` (installer already ships one for shortcuts; make sure `setWindowIcon` uses a multi-size `.ico`, not the png, for crisp taskbar rendering).
- **`.proj` file association** (optional, installer task): double-click opens eSim with that project (`Application` already restores by path — add `argv` handling).
- **Single-instance guard:** second launch currently starts a second full app; a `QLocalServer`/named-mutex check that focuses the existing window instead is ~30 lines and prevents "I clicked it twice because nothing appeared" (the exact cold-start scenario that produced the AppHangB1 reports).

### P3.7 Test hygiene on Windows
The 27–28 permanent test failures (POSIX path / HOME assumptions) mask real regressions — every session has to re-derive "is this failure pre-existing?". Fix or skip-mark them (`@pytest.mark.skipif(os.name == 'nt', reason=...)` where genuinely POSIX-only, real fixes where the product code is expected to work on Windows). Target: `pytest` green on Windows so CI can gate.

---

## Measurement appendix (do these before/after each P1 item)

- **Startup:** `~/.esim/startup.log` already timestamps stages. Add stages around `initToolBar`, `MainView`, `loadProjects`.
- **Event-loop stalls:** temporary watchdog — a `QTimer` at 100 ms that logs when the gap between fires exceeds 250 ms (detects any GUI-thread block, catches P0.2/P1.4/P1.5 classes in the wild). Ship it behind `ESIM_PERF=1`.
- **Sampling profiler:** `py-spy record -o profile.svg --pid <pythonw pid>` while reproducing lag (hover storm, resize, long sim). py-spy works on Windows and needs no code changes.
- **Paint cost:** `QT_LOGGING_RULES=qt.widgets.painting=true` or simply py-spy — blur frames from `QGraphicsDropShadowEffect` are unmistakable.
- **Cold-start realism:** test with `Remove-MpPreference` temporarily, or on a VM snapshot, since the dev machine now has the exclusion.

## Suggested execution order for the implementing model

1. P0.4 (tiny, isolated, test-first) → 2. P0.2 (+ commit the dirty KicadNetlister fix) → 3. P1.1 → 4. P1.2 → 5. P1.4 → 6. P0.3 → 7. P1.6 → 8. P1.7 → 9. P2.1 → 10. P2.2 + P2.3 (installer pass, one rebuild) → 11. P2.4 → 12. P1.5 (largest refactor, do when the rest is stable) → 13. P0.1 (diagnose after teardown paths simplified by earlier items — it may already be fixed by P1.1/P0.3) → 14. P1.3 staged rollout → 15. P3.x as capacity allows.

After item 10, rebuild the installer once (`windows\build-windows.ps1`), silent-install (`/VERYSILENT /NORESTARTAPPLICATIONS`, kill `C:\FOSSEE` python first), re-run doctor + convert→sim→plot smoke, and confirm `startup.log` deltas on a cold VM.
