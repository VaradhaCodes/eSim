# eSim Crash-Hardening Audit

**Scope:** `C:\Users\itsva\eSim-dev-src` @ `e05e89d6` (branch `dev`), full `src/` tree (~48.7k lines).
**Method:** manual read of every core subsystem — frontEnd (Application, DockArea, ProjectExplorer, Workspace, TerminalUi, FullScreen, theme), projManagement (Worker, Kicad, Validation, projectPaths, new/openProject), ngspiceSimulation (NgspiceWidget, plot_window, data_extraction), kicadtoNgspice (MainWindow, Processing, KicadNetlister), modelEditor, subcircuit, maker (DesignBus, jobs, ModelGeneration, Maker, CosimConfig), converters, ngspicetoModelica, codeEditor, browser, configuration — plus pattern sweeps for subprocess/thread/chdir/loop hazards.
**Status:** AUDIT ONLY. No file modified except this report.

**Context the fixer must know:** the app already ships three global safety nets —
1. a custom `sys.excepthook` (Application.py:1479) that stops PyQt6's qFatal-on-slot-exception abort and shows a dialog instead;
2. `faulthandler` → `~/.esim/crash.log` (Application.py:1533) for native deaths;
3. ordered teardown `_app_teardown` (Application.py:1560) against the sip use-after-free at exit.

So most "unhandled exception" findings below degrade to an error dialog rather than a process death. The findings are ranked by *what the user experiences*: BLOCKER = native crash / permanent freeze / app unusable; HIGH = feature dies or state corrupts in a way a user will hit; MEDIUM = plausible failure with confusing dialog / stranded state; LOW = edge cases.

---

## BLOCKER

### B1. Excepthook creates QWidgets on non-GUI threads → native crash risk
- **File:** `src/frontEnd/Application.py:1508-1523` (the `hook` function calling `Dialogs.critical`)
- **Root cause:** `sys.excepthook` runs on whichever thread raised; the hook constructs and `exec()`s a `QMessageBox` from that thread — creating/showing QWidgets off the GUI thread is undefined behavior in Qt and can crash natively (no Python traceback, window just dies).
- **Trigger:** any unhandled exception in a worker thread. A concrete, easy one exists: see B2. Others: an exception escaping `BackgroundJob.run` is caught, but `WorkerThread.run` and the watchdog observer thread are not.
- **Fix direction:** in the hook, check `QtCore.QThread.currentThread() is app.thread()`; if not on the GUI thread, log to error.log only and marshal the dialog to the GUI thread (queued signal or `QTimer.singleShot(0, …)` on a GUI-thread receiver).

### B2. `Worker.call_system` Popen is unguarded — missing external tool raises on the worker thread
- **File:** `src/projManagement/Worker.py:226-228`
- **Root cause:** `subprocess.Popen(shlex.split(command))` has no try/except; if the binary is not on PATH (`eeschema`, `OMEdit`, `OMOptim`), `FileNotFoundError` escapes `WorkerThread.run` — landing in the excepthook **on the worker thread** (B1's exact trigger). Best case: dialog-from-wrong-thread works by luck; worst case: native crash with no traceback.
- **Trigger:** user clicks **Open Schematic** (Ctrl+K) on a machine where KiCad is not installed / PATH not set (e.g. launched some way that skipped `launcher_windows.setup_environment`). `openSchematic` validates only that a project is open (`Validation.validateKicad`, Validation.py:84-101 — it never checks the tool exists).
- **Fix direction:** wrap the Popen in try/except; on `OSError` emit the existing `errorOccurred` signal (already delivered queued to the GUI thread) with a "KiCad/eeschema not found" message. Optionally `shutil.which()` preflight in `Kicad.openSchematic`.

### B3. Closing the Simulation tab mid-run permanently disables the toolbar
- **File:** `src/ngspiceSimulation/NgspiceWidget.py` (no `closeEvent`; signal wiring at 119-126) + `src/frontEnd/Application.py:1176-1180` (buttons disabled) + `src/frontEnd/DockArea.py:222-243` (`handle_tab_close` → `_destroy_dock`)
- **Root cause:** `open_ngspice` disables **Simulate, Convert, Close Project, Workspace** and re-enables them only in `plotSimulationData`, which fires from `simulationEndSignal` inside `finish_simulation`. If the user closes the Simulation dock's tab while ngspice runs, the dock is destroyed, the parented `QProcess` is destructed (killing ngspice) and `finished` is never delivered to the already-torn-down widget — `sim_end_signal.emit` never happens. Four core toolbar actions stay greyed out until app restart. To a user this *is* a frozen app.
- **Trigger:** run any longer transient, click the ✕ on the "Simulation-…" tab while it is still running.
- **Fix direction:** give `NgspiceWidget` a `closeEvent`/`destroyed` handler that terminates the process and emits `sim_end_signal` (CrashExit) — or have Application re-enable buttons on dock destruction; alternatively veto tab-close while `process.state() != NotRunning` and offer Cancel.

---

## HIGH

### H1. Background netlist job resumes against `current_project` instead of the captured project
- **File:** `src/frontEnd/DockArea.py:769-770` (`kicadToNgspiceEditor`: `projDir = self.obj_appconfig.current_project["ProjectName"]; projName = os.path.basename(projDir)`), reached from `src/projManagement/Kicad.py:263-272`
- **Root cause:** `Kicad.openKicadToNgspice` correctly captures `projDir` for the 5-15 s background `kicad-cli` export, and `_continueKicadToNgspice(projDir, projName)` carries it — but `kicadToNgspiceEditor` re-reads the *live* `current_project`. If the project was closed meanwhile it is `None` → `os.path.basename(None)` TypeError (excepthook dialog, converter dead). If a *different* project was opened meanwhile, the converter dock is registered and labeled under the wrong project.
- **Trigger:** click **Convert to Ngspice**, then close the project (or open another) during the export delay (long on cold Windows boots — the code itself documents 5-15 s).
- **Fix direction:** thread the captured `projDir`/`projName` all the way into `kicadToNgspiceEditor` (add parameters); guard `None` with a "project was closed" info dialog.

### H2. Fullscreen panel + dock destroyed = RuntimeError on deleted QDockWidget, panel content lost
- **File:** `src/frontEnd/FullScreen.py:88-99` (`_make_close_handler` → `self._dock.setWidget(...)`)
- **Root cause:** while a panel is fullscreened, its content is reparented out of the dock. The empty dock's tab can still be closed (`DockArea.handle_tab_close`) and **Close Project** (`closeDock`) destroys it too. Exiting fullscreen then calls `setWidget`/`show`/`raise_` on a deleted wrapper → RuntimeError (excepthook dialog inside a closeEvent) and the tool widget is orphaned parentless — visually the panel vanishes.
- **Trigger:** fullscreen the plotting panel (or converter), then Close Project (or close its now-empty tab), then press Esc.
- **Fix direction:** in the close handler, check `sip.isdeleted(self._dock)`; if dead, just close and drop the content (`deleteLater`), or block Close Project/tab close while a child panel is fullscreen.

### H3. `finish_simulation` runs twice when ngspice crashes
- **File:** `src/ngspiceSimulation/NgspiceWidget.py:119-126` (both `finished` and `errorOccurred` are connected to `finish_simulation`) and 372-435
- **Root cause:** a crashed child fires `errorOccurred(Crashed)` **and** `finished(...)`. The only dedupe guard is for the cancel path (`simulationCancelled`). Both invocations run the full UI path: two failure dialogs stacked, double `_unregister_process`, and `sim_end_signal` emitted twice → `plotSimulationData` twice.
- **Trigger:** any ngspice hard crash (bad codemodel, d_cosim vvp mismatch — common with NGHDL models).
- **Fix direction:** one-shot `self._finished_handled` flag set at the top of `finish_simulation`.

### H4. Model Creation dock is never registered per-project → survives Close Project with a live watchdog thread
- **File:** `src/frontEnd/DockArea.py:875-963` (`makerchip()` — the only tool opener with **no** `dock_dict.setdefault(temp,[]).append(...)` block; compare 727, 795, 846)
- **Root cause:** Close Project (`closeDock`, DockArea.py:1100-1129) only destroys docks registered in `dock_dict`. The Model Creation dock (Flow Navigator + DesignBus `watchdog.observers.Observer` thread + Verify-stage state, DesignBus.py:197-201) stays alive and bound to the closed project's paths. Subsequent actions in that dock (Save, Convert, external-edit popups) act on a project the app considers closed; repeated open/close cycles leak an OS observer thread each.
- **Trigger:** open Model Creation, Close Project, keep using the still-visible tab.
- **Fix direction:** add the same `dock_dict` registration as every sibling opener (and ensure DesignBus.close() runs from the dock teardown — closeDock's docstring already assumes it does).

### H5. ModelEditor writes into the install tree with `os.chdir` and no error handling
- **File:** `src/modelEditor/ModelEditor.py:695-830` (`createmodelfile`) and 1025-1046 (`converttoxml` tail)
- **Root cause (three stacked):**
  1. writes `.lib`/`.xml` into `library/deviceModelLibrary/...` — the *install* directory. Standard Windows installs under `C:\Program Files` (or Linux system installs) are not user-writable → `PermissionError`.
  2. uses `os.chdir(savepath)` and only restores CWD at the end (`ModelEditor.py:830`, `:1046`) with no try/finally — any exception mid-way strands the **whole process CWD** inside the library folder, silently breaking every later CWD-relative operation (e.g. `ModelicaUI.callConverter`'s chdir dance, relative file dialogs).
  3. `txtfile.close()` at :823 references a variable bound only inside the `if <radio>.isChecked()` branches → `NameError` if state ever allows no radio checked; earlier branches also leak open handles.
- **Trigger:** Model Editor → New → save a model on a machine where eSim is installed read-only (very common: Program Files, lab machines).
- **Fix direction:** write with absolute paths (no chdir at all); wrap in try/except with a clear "library not writable" dialog; consider redirecting user models to `~/.esim` like `prevvalues` already does.

### H6. `renameProject` revert path is itself unguarded — a failed revert half-renames the project
- **File:** `src/frontEnd/ProjectExplorer.py:644-662` (the revert loop: `os.rename` at :656 and :659 outside any try)
- **Root cause:** if renaming the project's files fails mid-way (Windows file lock — schematic open in eeschema is the norm), the code reverts prior renames; but the revert `os.rename` calls can fail for the same lock reason, escaping to the excepthook and leaving the project in a mixed old/new-stem state that no longer resolves (`resolve_stem` mismatch → tools can't find files).
- **Trigger:** right-click → Rename Project while the schematic is open in KiCad on Windows.
- **Fix direction:** preflight-check locks (try opening files exclusively) or wrap revert in per-file try/except and report what stayed renamed; better, refuse rename while `proc_dict[project]` has live children.

### H7. Corrupt `workspace.txt` check-token aborts startup
- **File:** `src/frontEnd/Workspace.py:104-106` (`QtCore.Qt.CheckState(int(self.obj_appconfig.workspace_check))`) with the value sourced from `paths.read_workspace()` (paths.py:86-96)
- **Root cause:** `read_workspace` returns the first token as-is. `Application.__init__` → `Workspace()` then does `Qt.CheckState(int(token))`. A non-numeric token raises `ValueError` at `int()`; a numeric one outside {0,1,2} raises at the enum constructor. This happens inside `Application()` construction in `main()` — the top-level handler prints and exits: **eSim never starts** until the user manually deletes `~/.esim/workspace.txt`.
- **Trigger:** hand-edited/corrupted `workspace.txt` such as `5 C:\ws` or `x C:\ws` (main()'s own startup guard at Application.py:1663-1669 catches `ValueError` from `int(check)` only for the *usability reset*, then re-reads later).
- **Fix direction:** sanitize in `read_workspace` (fall back to "0" unless token in {"0","2"}), or clamp in Workspace with a try/except defaulting to Unchecked.

---

## MEDIUM

### M1. `callConvert`'s XML serialization runs outside the failure surface
- **File:** `src/kicadtoNgspice/KicadtoNgspice.py:475-955` — the try/except with the friendly "Conversion failed" dialog starts only at :957.
- **Root cause:** hundreds of lines of index-driven serialization run bare: `self.obj_track.op_check[-1]` (:545 — IndexError if the Analysis tab never populated it), `entry_var_keys[count]` (:609-755 — IndexError whenever the XML's source structure and the live tab's entry count drift, e.g. prev-values file from an older schematic revision), `model_entry_var[i]` (:789-816). Any of these kills Convert with a raw excepthook dialog and can leave the mkstemp tmp file behind.
- **Trigger:** click **Convert** after editing the schematic's sources when a stale `*_Previous_Values.xml` exists (the mtime guard at :457 only catches *netlist* changes, not prev-values drift).
- **Fix direction:** wrap the whole serialization block in the same `_surface_conversion_failure` path; length-guard the parallel index walks.

### M2. `createSubFile` — unterminated `.control` block raises StopIteration
- **File:** `src/kicadtoNgspice/KicadtoNgspice.py:1239-1245` (`while words[0] != ".endc": eachline = next(netlist)`)
- **Root cause:** `next()` on exhausted iterator when `.endc` is missing → StopIteration escapes (and since this isn't a generator, it propagates as-is to the excepthook).
- **Trigger:** subcircuit-convert a hand-edited `.cir.out` whose `.control` block lost its `.endc`.
- **Fix direction:** iterate with a for-loop + flag, break at EOF.

### M3. Netlist parser index errors on malformed input (three spots)
- **File:** `src/kicadtoNgspice/Processing.py`
  - `:61-63` `readParamInfo`: `.param foo` (no `=`) → `paramList[1]` IndexError.
  - `:95` `preprocessNetlist`: netlist beginning with a `+` continuation line → `netlist.pop()` on empty list.
  - `:467-488` plot branches: `plot_v2`/`plot_i2` lines with missing node words → `words[2]` IndexError (this branch is *outside* the model try/except at :322-451).
- **Trigger:** hand-edited `.cir`, or a KiCad export interrupted mid-write (cloud-synced project folder).
- **Root cause:** parser trusts field counts.
- **Fix direction:** length-guard each `words[n]` access; report the offending line via the existing `_surface_conversion_failure`.

### M4. Double-clicking a stale project poisons `current_project`
- **File:** `src/frontEnd/ProjectExplorer.py:387-408` (`openProject`) and `:488-494` (`refreshProject`)
- **Root cause:** `openProject` calls `refreshProject(filePath)`, ignores its `False` (folder missing → error dialog shown), then **still** runs `set_current_project(filePath)` on the nonexistent path. Every subsequent tool click then operates on a dead path (eeschema on a missing file, "netlist not found" dialogs, `NgspiceWidget.setWorkingDirectory` on a ghost). Additionally, in `refreshProject` the `parentnode = self.treewidget.currentItem()` can be None when reached with a filePath but no selection → `parentnode.childCount()` AttributeError.
- **Trigger:** project folder on an unplugged USB/network drive; double-click its (stale) tree node.
- **Fix direction:** return early when `refreshProject` fails; None-guard `parentnode`.

### M5. Schematic converters: unguarded `getsize`/`copytree` around the parser call
- **File:** `src/converter/pspiceToKicad.py:30` (`os.path.getsize(file_path)` — FileNotFoundError if the user typed a path) and `:61-63` (`shutil.copytree` inside the try but only `CalledProcessError` is caught — copytree's FileNotFoundError/PermissionError escapes). Same shape in `ltspiceToKicad.py`, `libConverter.py`, `LtspiceLibConverter.py`.
- **Trigger:** type a nonexistent path into the Schematic Converter box and click Convert; or parser succeeds but output dir missing/locked (OneDrive).
- **Fix direction:** `os.path.isfile` preflight; broaden the except to `(subprocess.CalledProcessError, OSError)`.

### M6. Subcircuit upload writes into the install-dir library, unguarded
- **File:** `src/subcircuit/uploadSub.py:74-82` (`os.makedirs(subcircuit_path)` + `shutil.copy`)
- **Root cause:** same install-dir-writability class as H5 — PermissionError escapes to the excepthook on protected installs.
- **Trigger:** Subcircuit → Upload on a Program Files install.
- **Fix direction:** try/except with a clear dialog; long-term move user subcircuits under `~/.esim`.

### M7. `NewProjectInfo.createProject` — `.proj` write and registry save outside the guarded region
- **File:** `src/projManagement/newProject.py:88-99` (`f = open(...)` is inside the try, but `f.write`/`f.close` at :98-99 are **outside**) and `:118-120` (`save_project_explorer` can raise — the workspace may have become read-only after the earlier probe).
- **Trigger:** disk-full or workspace ACL change between probe and write; New Project then dies with a raw excepthook dialog after the folder was already created (half-project on disk).
- **Fix direction:** extend the try to cover write+close+registry save; on failure remove the half-created folder.

### M8. `DesignBus` hard-imports the `watchdog` package at module scope
- **File:** `src/maker/DesignBus.py:29-30`
- **Root cause:** if the optional `watchdog` pip package is missing/broken, importing `maker.makerchip` (→ DesignBus) raises ImportError → the entire Model Creation feature dies with a generic excepthook dialog (contrast with QScintilla, which EditorWindow degrades gracefully — EditorWindow.py:22-27).
- **Trigger:** partial install / user-managed venv without `watchdog`.
- **Fix direction:** guarded import; run without the external-edit watch when unavailable.

### M9. `Appconfig.print_*` is thread-unsafe by contract
- **File:** `src/configuration/Appconfig.py:206-226` (`_append_note` on a QTextEdit + `statusbar.showMessage`)
- **Root cause:** after the GUI attaches, `noteArea['Note']` is a QTextEdit and `_echo_status` touches the QStatusBar directly. Any current-or-future caller on a worker thread (the codebase currently avoids it, but nothing enforces it — e.g. a future `BackgroundJob` fn calling `print_info`) corrupts Qt state natively.
- **Fix direction:** route both through a queued signal on a small GUI-thread QObject; assert-thread in debug.

### M10. `validateSub` / `validateSubcir` unguarded file I/O
- **File:** `src/projManagement/Validation.py:146-148` (`open(lookSub)` bare) and `:208` (`os.stat(projDir)` on a path that may vanish between checks)
- **Trigger:** .sub file locked by a sync client or deleted between the exists-check and open.
- **Fix direction:** try/except OSError returning the existing "DIREC" / False codes.

### M11. `ModelEditor.converttoxml` parser assumptions
- **File:** `src/modelEditor/ModelEditor.py:941-942` (`filedata[modelcount]` IndexError when the .lib has no `.model` line), `:1027` (`os.chdir` into `User Libraries` that may not exist → FileNotFoundError, and any exception before :1046 strands CWD — same class as H5.2), `:1021-1022` (`ET.SubElement(param, tags)` with a parameter name that is not a valid XML tag → ValueError).
- **Trigger:** Upload .lib with a nonstandard or comment-only library file.
- **Fix direction:** validate `.model` presence, drop chdir, sanitize tag names.

### M12. Whole-app modal error dialogs from mid-teardown contexts
- **File:** `src/frontEnd/Application.py:1508-1521` — the hook `exec()`s an application-modal dialog even when raised inside paint/close/teardown handlers.
- **Root cause:** re-entering the event loop from inside a closeEvent/paintEvent via `exec()` can re-enter the very code that raised; the `seen_sites` dedupe stops *infinite dialog storms* but not the first re-entrancy.
- **Fix direction:** replace `exec()` with non-blocking `show()` for hook dialogs, or post them via `QTimer.singleShot(0, ...)`.

---

## LOW

### L1. `plotSimulationData` after crash-exit shows no dialog path issue — but double-status flip
`Application.py:1115-1142` — with H3 unfixed, status dot flips ok→failed→failed; benign once H3 is fixed.

### L2. Splash pixmap missing → null-pixmap warnings
`Application.py:1752-1768` — `splash_screen_esim.png` missing produces a null QPixmap; Qt warns, splash is invisible but startup continues. Cosmetic; guard `isNull()`.

### L3. `plot_window._spin_arrow_icon` caches PNGs in the shared temp dir
`plot_window.py:305-326` — if temp is cleaned while running, `QPixmap.save` fails silently and the QSS points at a missing file; arrows disappear (no crash). Regenerate on failure.

### L4. `WorkerThread.__del__` calls `self.wait()`
`Worker.py:157-171` — a blocking wait during GC; retention list makes this near-unreachable, but a wait() during interpreter shutdown could stall exit. Consider `wait(2000)`.

### L5. `Kicad.openSchematic` never checks the schematic file exists
`Kicad.py:64-81` — `main_schematic` returns a best-guess path (projectPaths.py:249); eeschema is launched on a possibly nonexistent file. eeschema handles it (creates/errors), so user-visible weirdness, not a crash.

### L6. `pspiceToKicad` blocks the GUI thread on `subprocess.run`
`pspiceToKicad.py:48` (and siblings) — the parser runs synchronously on the GUI thread; a big schematic = "(Not Responding)" spell. Move behind `BackgroundJob` like the KiCad netlister already does.

### L7. `ModelicaUI.callConverter` mutates process CWD on the GUI thread
`ModelicaUI.py:215-378` — chdir is try/finally-restored (good), but while it runs, any concurrent CWD-relative code (H5's chdir, relative dialogs) races. Eliminate the chdir (the converter takes paths).

### L8. `export_image` filename with dot-containing directory
`plot_window.py:1078` — `'.' not in basename` check is fine; no issue found. (Verified — listed to show it was checked.)

### L9. Cloud-synced folders (OneDrive/Dropbox)
No single crash line, but three interaction surfaces worth a defensive pass: `QFileSystemWatcher` refresh storms on sync (`ProjectExplorer.handleDirectoryChanged`, ProjectExplorer.py:179-184 — rebuilds children on every event, GUI-thread), transient `PermissionError` on `os.replace` while the sync client holds the file (`save_project_explorer` — caught, good), and ngspice writing `plot_data_*.txt` while sync locks them (surfaces as parse failure → handled dialog in `data_extraction.openFile`).

### L10. `Welcome`/`UserManual`/`ToolchainCheck`/`KicadNetlister`/`ModelGeneration`/`SnapshotStore`/`EditorWindow`/`DataExtraction`/`plot_window` render+parse paths
Read and probed; these are defensively written (timeouts, OSError catches, atomic writes, ragged-row retry, decoder resets, recursion-loop guards documented in-line). No unguarded crash path found beyond what is listed above.

---

## Systemic notes for the fix session

1. **Thread → dialog rule.** Every dialog must be born on the GUI thread. B1/B2/M9 are the same disease. One queued-signal "error reporter" object fixes the class.
2. **Install-dir writes.** ModelEditor (H5), UploadSub (M6), and `library/modelParamXML` writers assume a writable install. On Program Files installs these all fail. One shared "user library root under ~/.esim with fallback read from install" policy kills the class.
3. **`os.chdir` must die.** Three remaining users (ModelEditor ×2, ModelicaUI). Everything else in the codebase already passes `cwd=` explicitly.
4. **Async completion vs. mutated globals.** H1 is the shape to grep for: any `BackgroundJob`/`WorkerThread` continuation that re-reads `Appconfig.current_project` instead of using captured values.
5. **Sip/teardown crashes** (the historical 0xc0000005): the mitigations in `_app_teardown`, `plot_window.closeEvent`, `_applying_theme` guards and the `FigureCanvas.sizeHint` override are load-bearing. Fixers must not "simplify" them; each carries a regression comment explaining the crash it pins down.

---

## Round 2 — machine-assisted sweep (static analysis + import smoke + test suite + parser/widget fuzzing)

Round 2 stopped trusting eyes and ran the code: `ruff` (F821/F811/bugbear) over all of `src/`, an import-smoke of all 113 modules under `QT_QPA_PLATFORM=offscreen`, the project's own pytest suite (387 passed / 23 skipped on this machine), and purpose-built fuzz harnesses against `Processing`, `DataExtraction` and a live offscreen `plotWindow`. Everything below carries **dynamic or tool evidence**, not just reading.

### R2-1. BLOCKER — the QScintilla "graceful fallback" is broken: missing Qsci kills the whole app at startup
- **Files:** `src/codeEditor/theme.py:12` (`from PyQt6.Qsci import QsciScintilla`, unguarded) and `src/codeEditor/lexers.py` (same), imported unconditionally by `src/codeEditor/PlainEditor.py:13` (**the fallback editor itself**) and `src/codeEditor/EditorWindow.py:17,20` — which `src/frontEnd/ProjectExplorer.py:10` and therefore `src/frontEnd/Application.py:37` import at startup.
- **Evidence:** import-smoke and pytest both die on the exact chain: `Application → ProjectExplorer → EditorWindow → theme → PyQt6.Qsci → ModuleNotFoundError`. The `try/except ImportError` in EditorWindow (:22-27) guards only `CodeEditor`; the design comment "a missing optional dependency never bricks the project explorer" is currently false.
- **Trigger:** any install where QScintilla is absent **or fails to load** — partial installs, user venvs, a quarantined/corrupt `Qsci.pyd`, wrong VC runtime. Result: eSim window never appears.
- **Fix direction:** guard the Qsci import in `theme.py` and `lexers.py` (theme already tolerates a missing constant via `getattr(QsciScintilla, "SCI_SETPROPERTY", 4004)` — it only needs the import to be optional); keep `PlainEditor` genuinely Qsci-free; add a CI job that import-smokes `frontEnd.Application` with QScintilla uninstalled.

### R2-2. HIGH (functional, tool-confirmed) — Microcontroller previous-values restore is dead code
- **File:** `src/kicadtoNgspice/Microcontroller.py:162` and `:226` — `for child in root:` where `root` is **never defined** in the method (ruff F821).
- **Evidence:** ruff F821; the surrounding `except Exception: print("Passes previous values")` swallows the NameError on **every field**, so the Microcontroller tab never restores saved values and burns an exception per widget while looking fine.
- **Fix direction:** parse the prev-values XML into a real `root` (mirror the Model tab), or delete the dead block.

### R2-3. MEDIUM (fuzz-confirmed) — Digital Timing view raises on any zero-length trace
- **File:** `src/ngspiceSimulation/_render_mixin.py:470` (`np.min(raw_data)` / `np.max(raw_data)` inside the per-trace loop)
- **Evidence:** offscreen harness: header-only `plot_data_v.txt` (failed/empty run, or all rows dropped by the ragged-row defense) constructs and renders fine in Standard and Stacked, then dies in Timing with `ValueError: zero-size array to reduction operation minimum which has no identity`. In-app this becomes an excepthook dialog on **every** timing refresh. (The `spans` guard at :444 filters empty arrays for the threshold sync but the loop at :460-470 does not.)
- **Fix direction:** `if n == 0: continue` (or park as constant 0.5) before :470.

### R2-4. MEDIUM (fuzz-confirmed) — concrete crash inputs for the netlist pre-processor
Harness fed 15 malformed `.cir` bodies through the full `PrcocessNetlist` pipeline; four raised:
| Input | Raise site |
|---|---|
| `.param foo` (no `=`) | `Processing.py:63` IndexError (confirms M3a) |
| leading `+` continuation as first line | `Processing.py:95` pop from empty list (confirms M3b) |
| `h`/`f` component with < 6 fields (`h1 1 2`) | `Processing.py:214-231` `words[3..5]` IndexError — **new** |
| `transfo` with < 5 fields (`u3 1 2 transfo`) | `Processing.py:494-505` `words[4]` IndexError — **new** |
All are contained by `MainWindow._loadNetlist`'s catch (dialog, converter dead) — fix is per-branch length guards. Also fuzz-observed: `plot_v2` with only one node silently emits a bogus `plot v(5,plot_v2)` line — wrong output, no error at all.

### R2-5. MEDIUM — running the test suite on Windows corrupts the developer's REAL KiCad config
- **Files:** `src/maker/tests/test_kicad_symlib_paths.py` (redirects `HOME` only) vs `src/maker/kicad_symlib.py:178-180` (`_kicad_config_dir` uses `%APPDATA%` on Windows).
- **Evidence:** after one pytest run, this machine's real `%APPDATA%\kicad\9.0\sym-lib-table` contained `(lib (name "eSim_Ngveri") … (uri "/home/u/.esim/x.kicad_sym"))` — a bogus test entry pointing at a fake path. (Cleaned during the audit; backup left at `sym-lib-table.bak-audit`.) The same mismatch is why 3 of these tests "fail" on Windows: they assert against the tmp table while the code wrote the real one.
- **Fix direction:** tests must monkeypatch `APPDATA` (or `_kicad_config_dir` should honour an env override the tests set). Until fixed, treat pytest-on-Windows as unsafe on machines with a real KiCad profile.

### R2-6. LOW (test-infra) — 17 order-dependent test errors in the full-suite run
`ngspiceSimulation` list/perf/theme tests error at *setup* in a full run but pass 67/67 when the folder runs alone — cross-test global state (matplotlib rcParams / QApplication / module singletons) leaks between test modules. Makes CI flaky and can mask real regressions. Fix: isolate via fixtures resetting `plt.rcParams` and Appconfig class state.

### R2-7. LOW — `pspiceToKicad.py:8` imports `frontEnd.ProjectExplorer` and never uses it
Drags the entire editor/Qsci chain into the Schematic Converter import (part of why R2-1 has such a wide blast radius). Delete the import.

### R2-8. Notes for the record
- `converter/LTSpiceToKiCadConverter/src/*/{sch,lib}_LTspice2Kicad.py` index `sys.argv` **at module level** (import = IndexError). Safe today only because they are exclusively run as subprocess scripts; any future `import` of them kills the importer.
- `hdlparse` (bundled dep for maker/) does not pip-install on Python 3.13 — an unmaintained upstream; a future interpreter bump breaks Model Creation packaging.
- `codeEditor/theme.py` F821 storm from ruff is a **false positive** (palette names injected via `globals().update(LIGHT)`, both dicts complete) — do not "fix" it by renaming.
- **Verified robust under fuzz:** `DataExtraction.openFile` survived all 12 malformed-file cases (missing/garbage `analysis`, ragged rows, NaN/Inf, AC comma format, malformed event files, mismatched rawfile) — worst case is an error dialog and an empty plot; `plotWindow` survived empty/single-point/NaN-Inf/all-zero/constant-x data through Standard and Stacked modes, cursors, zoom and CSV collect; `ToolchainCheck` decodes subprocess output with `errors='replace'`.

---

## Round 3 — gap-list execution (widget fuzz + process chaos + soak/leak + full read of the remaining ~10k lines)

Round 3 executed gap items 1, 2, 3 and 6 from the Round-2 list. All harnesses live in `audit_harness/` (`fuzz_ktn.py`, `fuzz_modeleditor.py`, `fuzz_ngmo.py`, `fuzz_subcircuit.py`, `chaos_ngspice.py` + `chaos_stub.py`, `soak_leak.py`, `probe_registry.py`) with captured outputs (`*_out.txt`). Everything below marked **[EMPIRICAL]** was reproduced offscreen on this machine (Python 3.13.2, PyQt6 6.9.1, Windows 11); the rest comes from the full read of the 16 previously-unread files (~10.1k lines — every file on the item-6 list is now fully read).

### R3-1. HIGH [EMPIRICAL] — every simulation leaks one dead QProcess pair into the shared registries (Windows)
- **Files:** `src/ngspiceSimulation/NgspiceWidget.py:202` (`_start_process` registers unconditionally) vs `:170-171` (`_on_process_started` re-register with a `not in` guard); consumed by `Application.py:1049` (Close Project) and `:953-955` (exit `terminate_all`).
- **Evidence:** `soak_leak.py`: after 25 completed stub runs, `Appconfig.process_obj` and `proc_dict[proj]` each hold **25** dead entries. `probe_registry.py` pins the mechanism: at first unregister the registries already hold **(2,2)** for a single run, and after `_unregister_process` removes one, `self.process in process_obj` is **still True** — the same QProcess was appended twice. On Windows, `QProcess.start()` delivers `started` synchronously (CreateProcess succeeds inside `start()`), so `_on_process_started` runs its guarded register FIRST (registry empty → appends), then `_start_process:202` appends again with no guard. `finish_simulation`'s single unregister removes one of the two; net +1 forever. On Linux `started` is queued, the order flips, and the guard works — this is a **Windows-only** leak, invisible in Linux CI.
- **Consequence:** unbounded registry growth over a session; Close Project / exit then iterate dead sip wrappers (`terminate_handle` swallows the RuntimeError, so no crash — but the stale-handle sweep gets slower and the parented QProcess wrappers pin their widgets' Python side alive).
- **Fix direction:** make registration idempotent at BOTH sites (or drop the `:202` call and rely solely on `_on_process_started`, which fires on every platform and on redo).

### R3-2. H4 CONFIRMED [EMPIRICAL] — abandoned DesignBus leaks 2 OS threads per open/close cycle
- **Evidence:** `soak_leak.py`: 30 `DesignBus` instances abandoned without `close()` (exactly what happens because the Makerchip dock is not in `dock_dict` — H4) leave the process at **61 threads from a baseline of 1 (leak = 60, i.e. 2 per bus: watchdog observer + emitter)**. Control run WITH `close()`: leak = 0. H4's fix (register the dock so `closeDock` reaches `FlowNavigator.closeEvent` → `bus.close()`) is now empirically certified as both necessary and sufficient.

### R3-3. Chaos matrix results — B3 and H3 empirically proven; FailedToStart and garbage-bytes paths OK [EMPIRICAL]
`chaos_ngspice.py` drives a real `NgspiceWidget` (stub child via `chaos_stub.py`) through 7 process lifecycles (`chaos_out.txt`):
| Case | Result |
|---|---|
| A. instant non-zero exit | 1 finish, 1 error dialog — correct |
| B. hard crash (abort) | **finish_calls=2, 2 stacked error dialogs** — H3 live, exactly as read |
| C. garbage bytes both channels | decoded with no crash, no bogus dialog — decoder robust |
| D. dies mid multibyte char | **finish_calls=2 again** (H3), decoder survives the split rune |
| E. binary missing (FailedToStart) | 1 finish, 1 error dialog — handled |
| F. clean success | 1 finish, no dialog — correct |
| G. hang + user closes dock mid-run | **finish_calls=0, emits=0, widget destroyed** — B3 live: no completion signal ever reaches the app, toolbar stays dead |

### R3-4. KicadtoNgspice previous-values fuzz — M1 proven, two new crash sites [EMPIRICAL]
`fuzz_ktn.py` builds the real `KicadtoNgspice.MainWindow` offscreen against crafted `.cir` + `*_Previous_Values.xml` (10 cases, `fuzz_ktn_out.txt`):
- **B:** prevvalues XML present but no `<source>` child → **`UnboundLocalError: attr_source` at `KicadtoNgspice.py:613`** in `callConvert` (M1 empirically confirmed — Convert dies via excepthook dialog).
- **C:** source-type drift (XML remembers `v1`=pulse/7 fields, live netlist has `v1`=dc) → **IndexError at `KicadtoNgspice.py:619`**.
- **D:** sky130 netlist + empty `<scmode1/>` node → **IndexError "child index out of range" at `DeviceModel.py:119`** during tab construction (window fails to open at all).
- **F:** `.cir.out` whose `.control` block lost `.endc` → **StopIteration at `KicadtoNgspice.py:1241`** in `createSubFile` (M2 confirmed).
- A/E/G/H/I/J passed: truncated/malformed XML is caught, renamed sources degrade gracefully, empty `.cir` gets the new proper dialog, `.cir` deleted after window open still converts from memory (by design).

### R3-5. ModelEditor fuzz — five concrete crash inputs, one stranded-CWD repro [EMPIRICAL]
`fuzz_modeleditor.py` (10 cases, `fuzz_modeleditor_out.txt`):
- **B/C/D:** uploading a `.lib` with no `.model` line, an empty `.lib`, or `.model` as the last token → **IndexError at `ModelEditor.py:941`** (`converttoxml`).
- **F:** library name containing `"`/`<`/`>`/`|` → **OSError from `ElementTree` write** AND the harness caught the process **CWD stranded** in `deviceModelLibrary/User Libraries` (the `os.chdir` class from the systemic notes, now reproduced).
- **H:** Edit a `.lib` whose sibling `.xml` lacks `ref_model`, then Save → **AttributeError `'ModelEditorclass' object has no attribute 'ref_model'` at `ModelEditor.py:860`**.
- **I:** Edit valid lib → New → Save without picking a device type → **UnboundLocalError `txtfile` at `ModelEditor.py:823`**.
- A/E/G/J passed (invalid XML tag names, missing/truncated sibling XML are survived).

### R3-6. Subcircuit fuzz — M10 race proven at two sites [EMPIRICAL]
`fuzz_subcircuit.py` (7 cases): file deleted between selection and validation → **FileNotFoundError at `Validation.py:208`** (`validateSubcir`) and **`Validation.py:146`** (`validateSub`). Binary-garbage `.sub`, name-mismatch, duplicate-upload and empty-dir cases all produce proper dialogs.

### R3-7. NgspicetoModelica fuzz — fully contained, but empty input reports FALSE SUCCESS [EMPIRICAL]
`fuzz_ngmo.py` (12 cases through the real `ModelicaUI` path): every malformed netlist (continuation-first-line, short `.model`, `f`-source, missing `.sub`, single-node `r`, unknown model ref…) lands in the catch → error dialog, converter survives. **But cases K/L: an empty `.cir.out` (or comments-only) produces the "Ngspice netlist successfully converted to OpenModelica netlist" dialog** and a junk `.mo` — false success where an "empty netlist" error belongs. Functional, not a crash.

### R3-8. MEDIUM (read) — second unguarded index in `eSim_sky130`, same class as the fuzz-proven D
- **File:** `src/kicadtoNgspice/DeviceModel.py:157-164` — the corner-field restore loop does `child[1].text` (and on success `path_name = child[0].text`) with no IndexError guard, mirroring the `:119` crash the fuzz proved. Any `<scmode1>` node with exactly one child crashes tab build. The IHP branch (`:384-462`) guards all three indices with `except (IndexError, AttributeError)` — copy that pattern.
- Also `:187`: the designator whitelist check indexes `eachline[0]` — an empty line in `schematicInfo` would IndexError (Processing currently strips these; latent).

### R3-9. MEDIUM (read) — Operating-Point analysis restore is doubly dead code
- **File:** `src/kicadtoNgspice/Analysis.py:601-603` — `str(root[1][4].text())` **calls** the `.text` string property → TypeError on every run, swallowed → `op_check` always seeded `'0'`. And `:648` — `if root[1][4].text == 1:` compares str to int, never true → the OP checkbox is never restored from previous values. Two independent bugs, either alone kills the feature. (Same family as the ruff-found Microcontroller R2-2.) Also: every `createAnalysisWidget`/`createACgroup`… `open()` of the analysis/prevvalues files leaks its handle, and `setflag` appends to `op_check` forever (Convert reads `[-1]`, so growth is harmless but unbounded).

### R3-10. MEDIUM (read) — Convert step dies raw when a tracked library/subcircuit path vanishes
- **File:** `src/kicadtoNgspice/Convert.py:763` and `:814` (`shutil.copy2(libAbsPath, projpath)` uncaught), `:869-873` (`os.listdir(src)` + per-file copy uncaught in `addSubcircuit`).
- **Trigger:** a device-model `.lib` or subcircuit dir picked in the tab, then deleted/renamed (or on an unplugged network drive) before Convert → FileNotFoundError propagates to `callConvert`'s generic handler; conversion dead with a raw path error. The tab-restore guards (`_restore_device` drops dead paths) protect *remembered* values only, not the live-session race. Also `:754`: MOSFET branch indexes `tempStr[1]` — any `deviceModelTrack['m…']` entry without the `:W=… L=…` suffix (possible via the legacy `textChange` else-branch writing a bare `self.libfile`) IndexErrors.

### R3-11. MEDIUM (read) — `createkicad.py` is a fourth `os.chdir` user with the full H5 failure signature
- **File:** `src/maker/createkicad.py:153-183` (`createXML`): `os.chdir(xmlDestination)` → ET write → `os.chdir(cwd)` with **no try/finally** — a PermissionError on a read-only install (it writes into `library/modelParamXML/Ngveri`, install-dir class) or a missing dir both raise AND strand the process CWD, corrupting later relative-path operations exactly like H5. The systemic note "os.chdir must die — three remaining users" undercounts: this is a fourth.
- Same file, `PortInfo.getPortInfo:403-421`: a **leading blank line** in `connection_info.txt` hits the `pass` branch and then reads `in_items` before assignment → **UnboundLocalError**; a missing `connection_info.txt` raises FileNotFoundError at `:398` uncaught; and `createXML:156-158` does `self.portInfo[-1]` → IndexError when the port list parses empty. All three fire from NgVeri "Add Verilog model" with a malformed/empty connection file.

### R3-12. MEDIUM (read) — VerilogVerifier worker-thread teardown relies on `closeEvent`, which dock destruction skips
- **File:** `src/maker/VerilogVerifier.py:1496-1505`. The `closeEvent` cancel+`wait(3000)` is correct for an explicit close, but when the Makerchip dock is torn down by parent destruction (tab close → `deleteLater` of the whole tree) Qt does **not** deliver `closeEvent` — a running `BackgroundJob` (QThread, parented to the widget) is then destroyed while running → "QThread: Destroyed while thread is still running" → native-crash territory. Same blind spot as B3's. Also: `render_waveform` creates the `VcdPlotWindow` parentless and emits `waveformReady`; if no host is connected (verifier used standalone) the window leaks unparented.
- **Fix direction:** duplicate the join into a `destroyed`-connected teardown or parent the job to something that outlives the dock; guard the emit.

### R3-13. LOW (read) — latent dead code / contained leftovers from the full read
- `src/kicadtoNgspice/Model.py:168-171`: `add_hex_btn` connects `self.addHex`, which does not exist on `Model` (it lives on `Microcontroller`) → AttributeError if this currently-dead method is ever wired.
- `src/codeEditor/CodeEditor.py:124-136`: `reload()` re-reads via `_read_bytes()` with no guard — file deleted externally (watcher fires, host prompts, user clicks Reload after the delete) → FileNotFoundError through the reload path.
- `src/kicadtoNgspice/DeviceModel.py:864-993` (`GenerateSOCbutton`): both file handles never closed; output flushed only at GC.
- `Source.py`/`Model.py`/`Analysis.py` share the `root`-may-be-unbound restore pattern; all uses are inside `try/except` so restores silently skip — contained, but any future code touching `root` outside a try re-creates the fuzz-proven R3-4-B crash shape.
- **Read clean (no findings):** `motion.py` (extensively hardened, every RuntimeError guarded), `widgets.py`, `FindBar.py`, `Welcome.py`, `modelCache.py` (atomic writes), `SubcircuitTab.py` (guards its restores properly), `hdl/jobs.py`, `FlowNavigator.py` (beyond the H4/R3-2 close-path already covered; minor: `_set_panel` removes old widgets without `deleteLater`).

---

## What still remains after Round 3

1. ~~Per-widget fuzz harnesses~~ — **DONE** (R3-4…R3-7); NgVeri terminal paths still unfuzzed (its crash sites are now statically mapped in R3-11).
2. ~~Process-lifecycle chaos~~ — **DONE** (R3-3). Reusable: point `chaos_ngspice.py`'s stub at Worker/ModelGeneration to extend.
3. ~~Soak/leak~~ — **DONE for the two prime suspects** (R3-1, R3-2). Still open: theme-toggle ×100 and dock-open/close ×100 GDI/RSS soak (needs the full Application boot offscreen).
4. **Exit-path chaos — partially covered** (case G = close-dock-mid-run). Still open: whole-app close during sim/fullscreen/modal/snapshot-restore, logoff/taskkill, next-boot cleanliness. Needs a bootable offscreen `Application` fixture first.
5. **Environment matrix (VM-level)** — unchanged, cannot be simulated in-repo: read-only Program Files, non-ASCII username, OneDrive sync, disk-full, Qt 6.4 vs 6.8, `pythonw`, Defender cold-boot, DPI/monitor hot-unplug.
6. ~~Full read of the remaining ~12k lines~~ — **DONE** (all 16 files; findings R3-8…R3-13).
7. **CI gates** — unchanged, for the fix session: ruff F821+bugbear required; import-smoke without QScintilla/watchdog/hdlparse (R2-1/M8); Windows+Linux offscreen pytest with the R2-5 fix; commit `audit_harness/` fuzz corpora as regression tests. Note for CI authors: `hdlparse==1.0.4` from PyPI does not install on Python 3.13 — use the `https://github.com/hdl/pyhdlparser` tarball as `windows/requirements-windows.txt` already does.

---

## Platform note for the fix session (audit ran on Windows — what that means for Ubuntu)

- **~90% of the findings are platform-independent Python logic bugs** (IndexError/UnboundLocalError crash sites, dead XML restores, `os.chdir` without try/finally, missing thread joins, double-`finished` signal handling). They exist identically on Ubuntu; the prescribed fixes are additive guards and apply universally. Nothing in the fix directions is a Windows-specific rewrite.
- **Explicitly Windows-only findings:** R3-1 (registry leak — on Linux `started` is delivered queued, so the `:170` guard works and the leak does not occur; the idempotent-registration fix is a no-op there and safe) and R2-5 (`%APPDATA%` test pollution). **When fixing R3-1, do NOT "fix" it by reordering calls under an assumed signal timing — Windows delivers `started` synchronously inside `start()`, Linux queues it. The only ordering-proof fix is idempotence at both register sites.**
- **What this audit could not see for Ubuntu** (no Linux runtime here): Qt 6.4 (Ubuntu 24.04's version) vs 6.9 behavioral differences, the Wayland dock-drag/`startSystemMove` paths in `frontEnd/widgets.py` (read clean, never executed), launcher/packaging differences. Mitigation is cheap: every harness in `audit_harness/` is offscreen and path-portable — run them plus pytest once on Ubuntu (or in the item-7 CI Linux job) after the fixes land.

## Post-fix verification checklist (rerun on both OSes; every command is repo-root relative)

Expected results AFTER the fixes — any deviation = fix incomplete:
1. `python audit_harness/chaos_ngspice.py` → every case exactly **finish_calls=1, emits=1**; case G (dock closed mid-run) emits a completion signal so the toolbar revives (B3), no case shows 2 dialogs (H3).
2. `python audit_harness/soak_leak.py` → DesignBus no-close leak **0** (H4 fix registers the dock / closes the bus), registry counts **0/0** after 25 runs (R3-1).
3. `python audit_harness/fuzz_ktn.py` / `fuzz_modeleditor.py` / `fuzz_subcircuit.py` → **zero [RAISE] lines**; every former raise becomes a user-readable dialog or a silent-skip restore. `fuzz_modeleditor` case F must additionally report no `[CWD STRANDED]`.
4. `python audit_harness/fuzz_ngmo.py` → cases K/L (empty/comment-only netlist) now produce an **error** dialog, not "successfully converted".
5. Full pytest offscreen on Windows AND Linux; Windows run must not touch the real `%APPDATA%\kicad` (R2-5).
6. Import-smoke `frontEnd.Application` in a venv WITHOUT QScintilla installed → app must still reach the window (R2-1).

---

## Top 3 most likely to fail in front of a live user

1. **B3 — closing the Simulation tab mid-run bricks the toolbar.** Zero exotic preconditions: one impatient click on a tab ✕ during a slow simulation, and Simulate/Convert/Close Project/Workspace are dead until restart. In a demo this reads as a total freeze.
2. **B2 (+B1) — missing/unfindable eeschema on "Open Schematic".** First button a new user presses on a machine where KiCad isn't installed exactly as the launcher expects; raises on a worker thread where the crash net itself becomes the hazard (dialog created off the GUI thread → possible silent native death).
3. **H5 — Model Editor save on a read-only install.** Lab/demo machines install eSim under Program Files; the very first "Save model" throws PermissionError and strands the process CWD inside the library folder, quietly corrupting later file operations for the rest of the session.

**Honourable mention after Round 2: R2-1.** On any machine where the QScintilla module is missing or fails to load, eSim does not start *at all* — no window, no dialog. Bundled installs carry it, so a live demo on the official installer survives; every other distribution path (venv, partial install, AV-quarantined DLL) is one import away from a black screen.

---

## FIX LOG

### Batch 1 — crash net + ngspice process lifecycle (all 3 BLOCKERs + R2-1) — DONE
Theme: the "thread → dialog" disease (systemic note 1), the ngspice signal/registry lifecycle, and the startup-killer optional import. Verified by `chaos_ngspice.py`, `soak_leak.py`, `smoke_no_qsci.py`.

- **B1** — `Application.py` excepthook now checks `QThread.currentThread() is app.thread()`; off the GUI thread it logs (already done) and marshals the dialog via `QTimer.singleShot(0, app, show)` instead of building a QWidget on the worker thread.
- **B2** — `Worker.call_system` wraps `subprocess.Popen` in `try/except OSError`; a missing eeschema/OMEdit/OMOptim now emits the queued `errorOccurred` signal (dialog on the GUI thread) instead of raising on the worker thread (which was B1's trigger).
- **B3** — `NgspiceWidget` gains `_make_abandon_reporter` wired to `destroyed`: closing the Simulation dock mid-run now emits `sim_end_signal(CrashExit, -1)` so Simulate/Convert/Close Project/Workspace re-enable. Closure holds only plain Python objects (never the dying widget). `chaos_ngspice.py` case G: emits 0→1.
- **H3** — one-shot `_run_state['finished']` flag at the top of `finish_simulation`, plus disconnecting both process signals on first finalize. A hard crash firing both `finished`+`errorOccurred` now runs the UI path once. `chaos_ngspice.py` B/D: 2 dialogs→1, `finish_calls` 2→1.
- **R3-1** — `_register_process` is now idempotent at both sites (membership check before append). Windows' synchronous `started` no longer double-registers. `soak_leak.py` registry: (25,25)→(0,0) after 25 runs. **Do not** reorder the register calls — only idempotence is ordering-proof across Windows/Linux `started` delivery.
- **R2-1** — `codeEditor/theme.py` and `codeEditor/lexers.py` now guard the `PyQt6.Qsci` import; `make_lexer` returns `None` without Qsci so `PlainEditor` stays the genuine fallback. New `smoke_no_qsci.py` (Qsci blocked at the import hook): 7 FAIL→PASS; `frontEnd.Application` reaches its window with QScintilla absent.

**Harness note:** `chaos_ngspice.py` case G now flushes `DeferredDelete` (`sendPostedEvents(None, DeferredDelete)`) after `deleteLater` — a bare `processEvents()` never deletes the widget, so the pre-fix harness only *labelled* it destroyed while it stayed alive. The real top-level event loop does flush it, which is exactly when `destroyed` fires.

### Batch 2 — install-dir writes + `os.chdir` death (H5, M6, M11, R3-11, L7) — DONE
Theme: systemic notes 2 ("install-dir writes") and 3 ("os.chdir must die"). Every remaining `os.chdir` in a GUI/write path is gone; every install-tree write is now an absolute-path write guarded by `try/except (OSError, ValueError)` that degrades to a dialog instead of a raw `PermissionError` on read-only installs. Verified by `fuzz_modeleditor.py` (10/10 PASS, 0 raises, no `[CWD STRANDED]`) and `fuzz_subcircuit.py` (M6 paths A/B/E/F/G PASS). ngspicetoModelica tests 2/2; maker kicad/ports/cosim 29/30 (the 1 fail, `test_vvp_falls_back_next_to_iverilog`, is a pre-existing iverilog-not-installed env failure, unrelated).

- **H5** — `ModelEditor.createXML` rewritten: the six near-identical `os.chdir(subfolder)` write blocks collapse to one absolute-path write (`os.makedirs(exist_ok) + open(libpath) + tree.write(xmlpath)`) wrapped in `try/except (OSError, ValueError)` → "library may be read-only" dialog. No `os.chdir` remains, so a failed write can no longer strand the process CWD. The no-device-selected fall-through (which hit `txtfile.close()` on an unbound name — R3-5 case I) is now a guard-and-return at the top. `converttoxml` write tail likewise de-chdir'd + guarded (R3-5 case F: illegal-filename write → dialog, **no CWD strand**).
- **M11** — `ModelEditor.converttoxml`: the `.model` parse now tracks a `found` flag and length-guards `filedata[modelcount(+1)]`, so no-`.model` / empty / `.model`-last-token files (R3-5 B/C/D) show "no valid .model definition" instead of IndexError. File is read ONCE via `with open(...)` (two leaked handles + the char-by-char re-read gone); `stringof = list(content)`. `os.chdir` into a possibly-missing `User Libraries` replaced by `os.makedirs(exist_ok) + tree.write(abspath)`.
- **R3-5 case H** — `createtable` seeds `self.ref_model = self.model_name = None` before the XML `iter()` loops (kills stale-value carryover); `savethefile` reads them via `getattr(..., None) or filename` and wraps its `.lib`/`.xml` writes in the same guard — a sibling `.xml` with no `<ref_model>` no longer AttributeErrors.
- **M6** — `subcircuit/uploadSub.upload`: the `os.makedirs + shutil.copy` into `library/SubcircuitLibrary` is wrapped in `try/except OSError` → "library may be read-only" dialog + return.
- **R3-11** — `maker/createkicad.createXML`: `os.chdir(xmlDestination)`/`os.chdir(cwd)` removed; writes `os.path.join(xmlDestination, name+'.xml')` after `os.makedirs(exist_ok)`. Empty `portInfo` (would IndexError on `portInfo[-1]`) raises a clear `ValueError` the NgVeri caller already logs. `PortInfo.getPortInfo`: blank lines `continue` (kills the leading-blank-line UnboundLocalError), and a missing/unreadable `connection_info.txt` raises a clear `ValueError ... from e` instead of a bare FileNotFoundError. **createkicadCosim.createXML** got the identical de-chdir treatment (a fourth chdir user, same class).
- **L7** — `ngspicetoModelica/ModelicaUI.callConverter`: `os.chdir(dir_name)` + the `finally: os.chdir(cwd)` both removed. Verified the converter takes every path explicitly (`dir_name` is threaded into `compInit`/`procesSubckt`/`getSubParamLine`; the `.mo` is written via `os.path.dirname(self.ngspiceNetlist)`; `map_json` is absolute), so the chdir bought nothing and only raced concurrent CWD-relative code on the GUI thread.

**Scope note:** `fuzz_subcircuit.py` cases C/D still raise — those are **M10** (`Validation.validateSub`/`validateSubcir` file-I/O race), a different theme (file-I/O hardening), not part of this batch. The converter subprocess scripts under `converter/**` keep their module-level `os.chdir` (run only as standalone scripts, R2-8) — out of scope. `~/.esim` relocation of user models/subcircuits (the long-term half of systemic note 2) is deliberately deferred: it changes where models are *read* from across the app and carries real regression risk; batch 2 only converts the crash into a clean dialog.

### Batch 3 — parser / index guards (M1, M2, M3, R2-3, R2-4, R3-4, R3-8, R3-9, R3-10) — DONE
Theme: every raw `words[n]` / `child[n]` / `entry_var[n]` index walk in the netlist → prevvalues-XML → convert pipeline, plus the two dead XML-restore comparisons. All fixes are additive length/None guards — a malformed netlist, a drifted `*_Previous_Values.xml`, or a vanished library now degrades to a graceful skip or a readable dialog instead of a raw excepthook. Verified by `fuzz_ktn.py` (4/10 raises → **0/10**: B/C/D/F now PASS), a new `fuzz_processing.py` (10/10, direct parser calls), `fuzz_ngmo.py` unchanged (0 raises — regression gate), and `kicadtoNgspice` + `ngspiceSimulation` pytest (**123 passed**).

- **M1 / R3-4-B,C** — `KicadtoNgspice.callConvert` prevvalues serialization: `attr_source` is now always bound (a cache with no `<source>` node creates a fresh one instead of `UnboundLocalError`, fuzz B); the source grand-child restore breaks once `entry_var_keys` is exhausted (source-type drift no longer `IndexError`s, fuzz C); the model + microcontroller field walks length-guard `model_entry_var[i]`/`microcontroller_var[i]`; `op_check[-1]` guards the empty list.
- **M2 / R3-4-F** — `createSubFile`: the `.control` skip replaced `next(netlist)` (raised `StopIteration` on a missing `.endc`) with a `for ctrl_line in netlist` over the shared iterator that simply exhausts at EOF (fuzz F).
- **M3 / R2-4** — `Processing`: `readParamInfo` skips a `.param` token with no `=` (a); `preprocessNetlist` guards `netlist.pop()`/`netlist[0]` on a leading-`+`-first-line or empty netlist (b); the plot branch rebuilds nodes as `words[1:-1]`, killing both the `plot_v2`/`plot_i2` `IndexError` and the single-node-`plot_v2` bogus-node output (c); `h`/`f` (CCVS/CCCS) length-guard `words[1..5]` and `transfo` length-guards `words[1..4]` — the four fuzz-found parser raises.
- **R2-3** — `_render_mixin` Digital Timing: `if n == 0: continue` before `np.min`/`np.max`, so a zero-length trace (header-only / all-rows-dropped run) no longer throws "zero-size array to reduction" on every timing refresh.
- **R3-8** — `DeviceModel` sky130 `scmode1` restore: both `child[0]`/`child[1]` reads wrapped in `try/except (IndexError, AttributeError)` (empty `<scmode1/>` node → default library path, fuzz-KTN D); a blank `schematicInfo` line `continue`s before the `eachline[0]` designator test.
- **R3-9** — `Analysis` DC group: `root[1][4].text()` → `.text` (the method-call `TypeError` seeded `op_check` `'0'` every run) and the `== 1` int compare → `== '1'` string compare — Operating Point Analysis now actually restores from previous values.
- **R3-10** — `Convert`: `tempStr[1]` MOSFET-dimension and scmode-corner accesses length-guarded; the three `shutil.copy2`/`os.listdir` device-library + subcircuit copies wrapped in `try/except OSError` → a "moved/renamed/unavailable drive" `FileNotFoundError` message instead of a raw path error through `callConvert`'s handler.

**Scope note:** `fuzz_ngmo.py` cases K/L (empty / comment-only netlist reporting **false success**) are **R3-7** — the ngspicetoModelica converter has its own parser (no `Processing` import), so this batch cannot reach it; its clean fix lives in the ngmo / `ModelicaUI` area and is left for a later batch. This batch used `fuzz_ngmo` only as a regression gate (still 0 raises). The `root`-may-be-unbound restore pattern in Source/Model/Analysis (R3-13) stays inside its existing `try/except` and was not touched.

### Batch 7 — startup + optional-dependency resilience (H7, M8, R2-7) — DONE
Theme: nothing an end user can hand-edit into a config file, or leave uninstalled, should stop eSim from starting or brick a whole feature. Verified by a new `audit_harness/verify_batch7.py` (H7 clamp + dialog build, M8 watchdog-present path, R2-7 — all PASS), a new `audit_harness/smoke_no_watchdog.py` (M8 watchdog-ABSENT import-smoke, PASS), the `smoke_no_qsci.py` regression (PASS), and configuration + maker pytest (35 passed). ruff F821+bugbear clean on all four touched files.

- **H7** — a corrupt `workspace.txt` check-token no longer aborts startup. `paths.read_workspace` now clamps the first token to the only two values ever written — `"0"` (Unchecked) / `"2"` (Checked) — falling back to the default otherwise. This is the single source every consumer reads (`Appconfig.load_workspace`, both `main()` startup guards, `PspiceConverter`), so a token like `5 C:\ws` or `x C:\ws` can no longer reach `Qt.CheckState(int(token))`. Defense-in-depth at the actual crash line: `Workspace.__init__` no longer feeds an unvalidated int into `Qt.CheckState()` — it decides the two-state box explicitly (`== 2` → Checked else Unchecked). This matters because the enum constructor does **not** behave as the audit assumed on every build: on PyQt6 6.9.1 `Qt.CheckState(5)` does not raise, it silently coerces to Checked; the explicit compare is the only version-proof guard.
- **M8** — `DesignBus`' module-scope `import watchdog.{events,observers}` is now guarded behind `_HAS_WATCHDOG`; `_DiskWatchHandler` (a `watchdog.events.PatternMatchingEventHandler` subclass) is defined only when the package imports, and `_start_watch` early-returns when it is absent (`close`/`_observer` were already None-safe). Without watchdog the entire design flow — Author→Verify→Convert navigation, explicit Save, the lazy materialize before Convert — is unchanged; only the passive external-edit watch is disabled, mirroring the graceful QScintilla fallback in `codeEditor.EditorWindow`. `smoke_no_watchdog.py` confirms `maker.DesignBus` **and** `maker.makerchip` import with watchdog blocked.
- **R2-7** — deleted the unused `from frontEnd import ProjectExplorer` in `converter/pspiceToKicad.py`, cutting the whole editor/Qsci import chain out of the Schematic Converter's import blast radius (part of why R2-1 reached so far).

**Numbering note:** this entry follows the "batch 7" theme label the task used; the FIX LOG skips 4–6 because batch 4 (project/dock state + thread teardown) landed in code at `b93c97b3` without its own log entry and the R2-5 test-pollution batch is still in progress — the gap is bookkeeping, not missing work in *this* batch.
