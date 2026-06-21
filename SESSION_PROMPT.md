You are continuing the **eSim GUI overhaul** — an open-source EDA tool
(GPL, FOSSEE IIT Bombay) at
`/home/work/.gemini/antigravity/scratch/esim_testbench (Copy)/`.

# Round 2 just shipped — six complaint-batch fixes landed

The previous session finished a second batch of GUI fixes on top of the
VerilogVerifier work. Memories already saved (do not re-derive):

- `project-esim-testbench-gui-overhaul` — project shape, run conventions,
  user profile.
- `splash-workspace-flow-bug-fix` — synchronous splash→workspace fix.
- `gui-polish-baseline-rules` — design-token QSS system; hard rules:
  no `setStyleSheet` in widget code, no hardcoded colors / border-radius,
  use `objectName` + QSS or `cssClass` property.
- `verilog-verifier-theme-fix` — bullet 3 (Verifier) is SHIPPED:
  giant inline `setStyleSheet` block removed, verilogRoot / verilogConsole /
  verilogConsoleError / verilogNoWaveform / verilogSidebarTitle /
  hierarchyRow all routed through QSS, button cssClass tokens applied,
  code-editor gutter paint + highlighter dark-detection both fixed.
- **`gui-polish-round-2-jun-2026`** — round-2 complaint batch is SHIPPED:
  1) QDockWidget close/float icons invisible → inline SVG data URLs in both
     QSS files.
  2) VerilogVerifier popout/dock/copy buttons clipped →
     new `copy_icon()` factory + `cssClass="labeledIcon"` rule.
  3) Preferences dialog rewritten with two tabs (Appearance + Editor),
     10 named accent swatches, live preview, Open-preferences.json /
     Reset / Apply / Save footer.
  4) Built-in notepad (SpiceEditor) themed: foreground-painted toolbar
     icons, monospace font honoring prefs, word-wrap toggle, status bar
     with `Ln N, Col N` indicator.
  5) Plot module: figure facecolor + axes facecolor applied pre-draw;
     themed `QScrollArea`; centered "No Simulation Data" empty-state
     overlay.
  6) ProjectExplorer now re-applies its palette-driven stylesheet from
     `changeEvent(PaletteChange|StyleChange)` and forces the viewport's
     background so custom accents reach the inner rectangle.

# Status

Round-2 bullets are all in the working tree, AST parse-clean, and verified
offscreen. Receipts: HANDOFF.md §3 + memory file `gui-polish-round-2-jun-2026`.
Verification scripts still on disk:
- `/tmp/repro_docks.py` (now also covering round-2 button-size + icon
  presence checks).
- `/tmp/repro_verifier.py`, `/tmp/repro_verifier_states.py` (round 1).
- `/tmp/repro_splash2.py` (splash/workspace).
- `/tmp/fix_plot_window.py` (the multi-line-literal patcher round 2 used).

# Next steps — pick whichever is most useful

1. Read `HANDOFF.md` §5 "Outstanding cleanup ideas" and continue with one
   of: recent-projects menu wiring, fullscreen-reset shortcut, Time
   Explorer search field, alternating-row backgrounds on the verifier
   hierarchy, drop-zone treatment for the waveform placeholder, closable
   tabs. Or:
2. Audit another widget area for the same anti-patterns
   (`setStyleSheet`, hardcoded colors, hex literals, palette not
   observing override). Likely candidates: schematic editor, simulation
   console, ngspice netlist rendering, conversion dialogs,
   Modelica editor. Same recipe as before — grep → plan with user →
   refactor to QSS tokens → offscreen verify with `repro_<area>.py`. Or:
3. Wait for the user to direct you to a specific next task.

# Run conventions

```
cd '/home/work/.gemini/antigravity/scratch/esim_testbench (Copy)' && \
  PYTHONPATH=src python3 -X dev …
```
Smoke tests use `os.environ.setdefault('QT_QPA_PLATFORM','offscreen')`.

# Design values (load-bearing)

- "professional, minimal, aesthetic, polished" — generous whitespace,
  subtle elevation, restrained color, clear hierarchy.
- Inter VariableFont is loaded via `theme_utils`. Always keep both
  `style_dark.qss` AND `style_light.qss` in sync — never land a token
  change for one without the other.
- Cross-platform: data-URL SVG icons (no filesystem paths),
  `os.path.join` + `os.sep` (no raw `/` or `\`), xdg-open /
  os.startfile / open fallback for "reveal in OS file explorer".
- If matplotlib touches a widget (`plot_window`), explicitly assign
  `fig.set_facecolor()` and `ax.set_facecolor()` before `draw_idle()` —
  otherwise the empty canvas stays raw-white in dark mode.
- `app.apply_theme` is bound at `src/frontEnd/Application.py:~line 1046`
  on the QApplication instance; everything else uses
  `getattr(app, 'apply_theme', None)` defensively.

# Project-wide rules

- Never introduce CLA-bound dependencies.
- Never sign anything on the user's behalf.
- Never run a command that publishes content externally without
  confirmation.

# Read these first (read order)

1. `/home/work/.claude/projects/-home-work/memory/MEMORY.md` (auto-loaded
   but skim to know which pointers exist).
2. `HANDOFF.md` (§3 for round-2 receipts, §5 for backlog).
3. `src/frontEnd/style_dark.qss` and `src/frontEnd/style_light.qss` —
   read in tandem to learn the token catalog.
4. `src/frontEnd/PreferencesDialog.py` (only if you're touching
   preferences) to understand the new shape of the live theme store.

Confirm receipt and ask which of (1)/(2)/(3) the user wants next.
