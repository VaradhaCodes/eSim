# eSim GUI Overhaul — Session Handoff

You are continuing a UI / UX overhaul of **eSim**, an open-source EDA tool
(GPL, FOSSEE IIT Bombay) for circuit design, simulation, and PCB layout.
The user has set the effort to `xhigh` and asked us to bring the GUI at
parity with professional desktop applications.

> Run from `bash` with `cd /home/work/.gemini/antigravity/scratch/'esim_testbench (Copy)' && PYTHONPATH=src python3 -X dev …`.
> Smoke tests must be **offscreen**: `os.environ.setdefault('QT_QPA_PLATFORM','offscreen')`.

---

## 1. Where things live

```
src/frontEnd/
  Application.py        # main window / lifecycle
  Workspace.py          # workspace selector dialog  ← JUST FIXED
  Welcome.py            # empty-state dashboard (hero + cards)  ← rebuilt
  MainView.py, ProjectExplorer.py, TimeExplorer.py    # refreshed
  style_dark.qss        # dark theme
  style_light.qss       # light theme
  theme_utils.py        # theme switcher + Inter font loader
  icon_paths.py         # inline SVG icon factory
src/main.py             # standard entry: creates QApplication, calls Application()
src/browser/            # dock-area widgets (Welcome, etc.)
src/ngspicetoModelica/, src/converter/, src/projManagement/    # backend modules (bugs fixed previously)
```

---

## 2. What's already done in this session

### 2.1 Splash → Workspace → Main-window flow  *(USER CONFIRMED WORKING)*
- **Bug:** After splash and workspace click, the main window never showed.
- **Cause:** `Workspace._refresh_app_after_workspace_change` deferred work
  via `QTimer.singleShot(80, _finish_workspace_change)`. In the splash
  flow (where `defaultWorkspace()` is called without prior `show()`), the
  deferred timer never fired before app-exit.
- **Fix:** Replaced with two synchronous methods, called directly from
  `defaultWorkspace()` and `createWorkspace()`:
  - `_refresh_project_explorer()` — exception-guarded tree rebuild.
  - `_finish_workspace_change()` — splash close (guarded) + `view.show()`.
  File: `src/frontEnd/Workspace.py`. **Do not re-introduce a singleShot
  between `accept()` and `view.show()`.**

### 2.2 Hardcoded `BaseException` → `Exception`
- 99 instances across 24 files replaced via `sed`.
- Hungry `except` blocks no longer swallow `KeyboardInterrupt` /
  `SystemExit`.

### 2.3 Hardcoded QSS / theme bypasses removed
- `ProjectExplorer.py`: hardcoded `border-radius:15px;border:1px solid gray`
  → replaced with `#projectTree` object-name-scoped rule in both QSS files.
- `TimeExplorer.py`: full rewrite, no inline `setStyleSheet`.

### 2.4 Welcome screen
- Hero banner, scroll area, section headers (caps), tool cards (icon + title
  + description + chevron) with click + keyboard activation.

### 2.5 Workspace selector (`Workspace.py`)
- Now a proper `QDialog` (`setModal(True)`, not the broken
  `WindowModality.ApplicationModal`).
- QFormLayout rows, group box, button-box layout.
- Sync flow (see 2.1).

### 2.6 Menu bar / Toolbar
- File / Edit / View / Tools / Help menu in Application.
- New/open/save actions. Includes recent-projects submenu.

### 2.7 Backend crash fixes (already verified)
- `NgspicetoModelica.py`: `if '0' in n or 'gnd' in n` (operator-precedence fix).
- `NgspicetoModelica.py`: `mappingData["Devices"]...` (dict-access fix).
- `ltspiceToKicad.py`: `merge_copytree(src, dst, item)` argument fix.
- `projManagement/Worker.py`: `proc_dict` KeyError guard.
- `projManagement/openProject.py`: walk loop replaced with root-targeted walk.

### 2.8 Design-token system established
- Label `cssClass`: `title`, `heading`, `caps`, `muted`, `subtle`,
  `error`, `warning`, `success`.
- Button `cssClass`: `secondary`, `tertiary`, `icon`, `danger`.
- Surface tiers, border tiers, focus rings, hover states, status bar, etc.
- Inter VariableFont loaded via `theme_utils.apply_theme()`.

---

## 3. Polish round 2 — six GUI complaints  *(SHIPPED)*

> A second user request bundled six regressions / missing-polish items.
> All shipped in one session. Receipts live in memory:
> [`gui-polish-round-2-jun-2026`](file:///home/work/.claude/projects/-home-work/memory/gui-polish-round-2-jun-2026.md).

### 3.1 QDockWidget close / float icons invisible
- `QDockWidget::close-button` QSS rules sized 22×22 but never declared an
  `image:`, so Qt fell back to whatever the platform painted (often
  transparent). Added explicit cross-platform **inline SVG `data:` URLs**
  for both close (two-cross) and float (outward-arrow) glyphs in
  `style_dark.qss` and `style_light.qss`. Stroke color matches the title
  bar foreground.

### 3.2 VerilogVerifier popout/dock/copy buttons clipped
- Buttons carried 10-char labels (`"🗗 Fullscreen"`, `"📋 Copy"`, `"🡮 Dock
  to IDE"`) but `cssClass="icon"` pinned `min/max-width: 32px`, so they
  clipped. Unicode glyphs also rendered unreliably across fonts.
- Added `copy_icon()` to `frontEnd/icon_paths.py`. `VerilogVerifier.py`
  imports the icon factories, drops the glyphs, switches to
  `cssClass="labeledIcon"`. New `QPushButton[cssClass="labeledIcon"]`
  rule in both QSS files: `min-height 30/32 px`, `padding: 0 12px`,
  theme-aware hover/pressed, `qproperty-iconSize: 16px 16px`.

### 3.3 Preferences dialog — friendlier + feature-rich
- `src/frontEnd/PreferencesDialog.py` was a tiny dialog with only three
  color buttons, a hardcoded `#777` border that disappeared in dark
  mode, no live preview, no way to open the underlying JSON.
- Rewrote with two tabs:
  - **Appearance** — "Follow OS / Always light / Always dark" combo;
    row of **10 named accent swatches** (Default, Indigo, Sky, Teal,
    Emerald, Amber, Crimson, Magenta, Slate, Charcoal); custom
    `QColorDialog`; secondary background + internal surface pickers;
    live preview strip with `border-left: 6px solid <accent>` and
    theme-aware text.
  - **Editor** — `QFontComboBox` (Monospace hint) + 8–24 pt size spin
    + 2–8 space tab width.
  - Footer: **Open preferences.json** (`xdg-open` / `os.startfile` /
    `open` fallback for cross-platform), **Reset to Defaults**, **Apply**
    (live), **Cancel**, **Save**. `_border_for_theme()` replaces the
    hardcoded `#777`. **Apply** calls `app.apply_theme()` (bound in
    `Application.py:~line 1046`) and re-runs `update_theme_styles()` on
    every open `SpiceEditor` window.

### 3.4 Built-in notepad (txt /.cir /.cir.out) themed + featureful
- `src/frontEnd/SpiceEditor.py` used `QStyle.StandardPixmap` icons that
  disappear on dark canvas; font family / tab width were hardcoded; no
  status bar; no word-wrap toggle.
- New helpers `_paint_with_foreground()` + `_paint_with_foreground(
  Save|FindGlyph)()` paint toolbar icons with the live palette
  foreground so they're visible in both themes.
- `update_theme_styles()` now reads `editor_font_family`,
  `editor_font_size`, `editor_tab_width` from prefs; sets a
  `StyleHint.Monospace` `QFont`; applies `setTabStopDistance()`.
- Added `Word wrap` checkable action, themed scrollbar rules, and a
  status bar with live `Ln N, Col N` + UTF-8 indicators.

### 3.5 Plot module frame + empty state themed
- `src/ngspiceSimulation/plot_window.py`'s matplotlib `Figure` held its
  initial white facecolor until the first draw, and the surrounding
  `QScrollArea` had `frameShape(NoFrame)` with no themed background,
  leaving the empty canvas in raw-white-on-dark contrast.
- `apply_theme()` now walks `self.fig` + every `ax`, assigns palette
  `bg` / `axes_face`, then calls `canvas.draw_idle()`. The empty-state
  bug is closed.
- `QScrollArea` chrome now themed with `border_strong` /
  `axes_face`.
- New centered **plotEmptyState** QLabel overlay — "No Simulation Data
  / Run a transient, AC, or DC sweep…", themed with dashed
  `border_strong` + `axes_face` so it disappears once data arrives.
  `_show_empty_state(volts_length == 0)` toggles it during
  `load_simulation_data()`; `_layout_empty_overlay()` resizes it on
  every canvas resize / show.
- Edits had to be done through a Python patcher (`/tmp/fix_plot_window.py`)
  because the multi-line literal syntax (`</div>"`) historically
  tripped the structured-Edit tool. AST-parse the file before
  declaring success.

### 3.6 ProjectExplorer follows accent selection
- `src/frontEnd/ProjectExplorer.py` installed its inline stylesheet
  exactly once in `__init__()`; `palette(...)` role references only
  re-resolve on `setStyleSheet()`. Viewport (the inner rectangle)
  keeps a default white background.
- Pulled the inline stylesheet into `_apply_tree_stylesheet()`, re-call
  it from `changeEvent(PaletteChange|StyleChange)`. Also sets
  `treewidget.viewport().setStyleSheet("QWidget { background-color:
  palette(base); }")` so the inner rect keeps the theme color.
- Imported `QtGui` at module level; stored `self._base_dir` on
  `__init__`. Verified dark → `palette(base) = #1f2937`,
  `palette(highlight) = #3b82f6`, and that an override palette
  re-paints immediately.

### Files touched (round 2)

```
src/frontEnd/icon_paths.py        +  copy_icon()
src/frontEnd/PreferencesDialog.py ~  full rewrite
src/frontEnd/SpiceEditor.py        +  toolbar icon helpers, status bar, word wrap
src/frontEnd/ProjectExplorer.py    +  _apply_tree_stylesheet(), changeEvent
src/frontEnd/style_dark.qss        +  QDockWidget X/float image: rules
src/frontEnd/style_dark.qss        +  QPushButton[cssClass="labeledIcon"]
src/frontEnd/style_light.qss       +  mirror image: rules
src/frontEnd/style_light.qss       +  mirror labeledIcon rules
src/maker/VerilogVerifier.py       +  import icon factories; remove glyphs; flip cssClass
src/ngspiceSimulation/plot_window.py +  fig.set_facecolor, empty overlay, themed scroll area
```

### Verified offscreen (round 2)

- `/tmp/repro_docks.py` still passes; QSS ok, icon factories valid,
  screenshots saved under both themes.
- AST parse on all 7 touched files clean.
- `PreferencesDialog(...)` instantiates offscreen with 10 swatches and
  live preview.
- `ProjectExplorer` in dark theme shows `#1f2937` base / `#3b82f6`
  highlight and re-paints when palette changes.
- `plot_window` `apply_theme` sets figure facecolor pre-draw.

---

## 4. Smoke-test recipes

```bash
# Splash / workspace flow (verified working)
QT_QPA_PLATFORM=offscreen timeout 8 python3 -u /tmp/repro_splash2.py

# Welcome / main-window construction
QT_QPA_PLATFORM=offscreen timeout 10 python3 -c "
import os, sys
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
sys.path.insert(0,'src/frontEnd'); sys.path.insert(0,'src')
os.chdir('src/frontEnd')
import pathmagic
from PyQt6 import QtWidgets
from Application import Application
app = QtWidgets.QApplication(sys.argv)
av = Application()
print('OK', av.windowTitle())
"

# Theme toggle (verifier should look identical in both modes)
# Verify widgets by inspecting sets:
#   [w.cssClass() for w in av.findChildren(QtWidgets.QWidget) if w.property('cssClass')]
```

---

## 5. Outstanding cleanup ideas (not yet assigned)

- Recent-projects menu refresh (added but not wired to a settings hook).
- Fullscreen toggle (F11) is wired; consider adding Ctrl+Shift+F to reset UI.
- Time-explorer: alternating rows + filters are present but could use a
  search field for parity with VS Code's timeline.

---

## 6. Style guide for future changes

- `cpp|py` files: indent 4 spaces, `snake_case` for variables/functions,
  `PascalCase` for classes — match whatever file you're editing.
- New widgets: never `setStyleSheet`. Use `cssClass`.
- New icons: add to `icon_paths.py` as `inline_svg(name, path)`.
- Theme tokens: edit `style_dark.qss` AND `style_light.qss` together.
- The user values **"professional, minimal, aesthetic, polished"** — reach
  for generous whitespace, subtle elevation, restrained color, and
  clear hierarchy.

Good luck.
