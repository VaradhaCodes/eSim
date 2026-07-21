# UI_AUDIT.md — Aurora Design-System Audit (FINAL — Rounds 1+2 complete)

**Scope:** the complete new UI layer ("Aurora" theme) — `src/frontEnd/style_dark.qss`, `style_light.qss`, `tokens.py`, `theme_utils.py`, `widgets.py`, `motion.py`, `elevation.py`, `tooltips.py`, `icon_paths.py`, `dialogs.py`, `PreferencesDialog.py`, plus every consumer of them across `browser/`, `codeEditor/`, `kicadtoNgspice/`, `ngspiceSimulation/`, `maker/`, `modelEditor/`, `subcircuit/`.

**Audited tree:** `eSim-dev-src`, branch `dev` @ `e05e89d6` (2026-07-20).
Note: the font-architecture commit `110228cc` (JetBrains Mono bundling + `fonts.py`) is NOT in this tree — findings §C6 assume current tree state; re-check after that commit lands.

**How to use this file (for the fixing session):** every finding has a severity, receipts (`file:line`), and a **Fix:** paragraph describing the intended approach. Work top to bottom within each severity. After each themed-area change, run the existing guards:
`pytest src/frontEnd/tests/test_theme_qss_cache.py src/frontEnd/tests/test_view_control_metrics.py src/frontEnd/tests/test_motion_idempotent.py src/ngspiceSimulation/tests/test_plot_window_theme.py` — all must stay green. Verify visually in BOTH themes (toggle via the toolbar moon button) at 100% and 150% zoom.

Severity: **P1** = visible defect / broken styling · **P2** = design inconsistency users can see · **P3** = dead weight & hygiene (safe deletes) · **P4** = architecture / nice-to-have.

---

## P1 — Visible defects

### 1.1 ModelEditor inline-editor stylesheet is invalid QSS (silently discarded)
`src/modelEditor/ModelEditor.py:48-54` builds the delegate editor's sheet with `%`-formatting but kept f-string escaped braces: the emitted string is `QLineEdit{{margin:0;...}}`. Qt's QSS parser rejects it (console warning "Could not parse stylesheet"), so the whole rule is dropped and the editor silently falls back to the app-sheet rule.
**Fix:** the app sheet already styles inline cell editors (`QAbstractItemView QLineEdit` block, `style_dark.qss:888-900` and light mirror) — delete the entire per-delegate `setStyleSheet` block in `createEditor` rather than fixing the braces. If the 2px border look is genuinely wanted, fix braces to single `{`/`}` — but prefer deletion; one source of truth.

### 1.2 TerminalUi.ui ships pre-Aurora hardcoded styles that override both themes
`src/frontEnd/TerminalUi.ui:63` — progress chunk `rgb(54,158,225)` (old blue, not the Aurora cyan `#53D7FF`/`#18A8D8` gradient every other progress bar gets).
`src/frontEnd/TerminalUi.ui:140` — console `QPlainTextEdit { background-color: rgb(36,31,49); color: white; }` — an off-palette purple-black that persists even in LIGHT theme (widget-level sheets beat the app sheet).
**Fix:** delete both `styleSheet` properties from the `.ui` file. Give the console widget `setObjectName("simulationConsole")` in `TerminalUi.py` if it doesn't already flow through that name — the app sheets already carry a complete `QPlainTextEdit#simulationConsole` rule (dark `style_dark.qss:1346-1358`, light mirror). The progress bar falls back to the global `QProgressBar` rule, which is correct in both themes.

### 1.3 Simulation status dot uses dark-theme colors in light theme
`src/frontEnd/Application.py:818-827` — `_set_sim_status` hardcodes `#5F728D / #FACC15 / #42E6A4 / #FB7185` (the DARK token set). In light theme, mint `#42E6A4` and yellow `#FACC15` on the `#F3F7FC` status bar are near-invisible.
**Fix:** import `tokens` and pick per-theme: dark uses `DARK["text_subtle"/"warning"/"success"/"danger"]`, light uses `LIGHT` equivalents (`#9AAABE / #D97706 / #059669 / #E11D48`). Re-tint on theme change — call `_set_sim_status(self._last_sim_state)` from the palette-change path (store the last state string on the instance; currently only the color is stored implicitly in the stylesheet).

### 1.4 Legacy gray group boxes in KicadToNgspice screens defeat Aurora styling
13 call sites re-apply the ancient `QGroupBox { border: 1px solid gray; border-radius: 9px; ... }` sheet, overriding the themed group-box design on the converter windows in both themes:
- `src/kicadtoNgspice/Source.py:116, 159, 203, 246, 286, 329`
- `src/kicadtoNgspice/Analysis.py:321, 613, 796`
- `src/kicadtoNgspice/Model.py:136, 176`
- `src/kicadtoNgspice/Microcontroller.py:215, 249`
**Fix:** delete every one of these `setStyleSheet` blocks (they are identical copy-paste, marked `# CSS`). The global `QGroupBox` rule already provides the border/radius/margin these were approximating, themed correctly. If a section needs the stronger "card" look, use `box.setProperty("cssClass", "themedGroupBox")` like the rest of the app. No layout changes needed — the QSS `margin-top` handling is equivalent.

### 1.5 Contradictory duplicated QDockWidget button rules
`src/frontEnd/style_dark.qss:617-662` (and the light mirror): the first block zeroes out `close-button`/`float-button` (`width:0; height:0; image:none` — the deliberate "chrome moved to the card header" decision), then FIVE later blocks re-assign images, hover backgrounds and margins to the same subcontrols. Later rules win per-property, so the buttons end up 0×0 but with images and hover paint assigned — harmless today only because width 0 hides them, and confusing to every future editor.
**Fix:** keep only the hiding block (`:617-624`); delete `:626-662` in dark and the mirrored range in light. Visual result unchanged.

### 1.6 Dark/light rule parity gaps (found by structural diff)
The two sheets are 1:1 except:
- `QPushButton:disabled` — light adds `font-style: italic` (`style_light.qss:295`), dark doesn't; `QLabel#verilogNoWaveform` italic exists in both, fine. An italic disabled state in only one theme is an inconsistency, and it forces the awkward `font-style: normal` patch on light primary buttons (`style_light.qss:307`).
- `QCheckBox::indicator:hover` — dark tints the box background (`style_dark.qss:979-983`), light only recolors the border.
- `QPushButton:default:pressed` — dark sets `border-color`, light doesn't.
- `QPushButton[cssClass="verifierPrimary"]:disabled` — dark sets `border-color`, light doesn't.
- `QLabel[cssClass="heroGradient"]` exists only in dark (`style_dark.qss:1221-1226`) — dead anyway, see §P3.1.
**Fix:** drop the italic + its `font-style: normal` counter-patch from light (match dark); add the missing hover background / border-color lines to light using its palette (`#F6F9FD`-family tints). Goal: the structural diff of the two sheets returns empty except intentional palette values.

### 1.7 Editor window light chrome is a different design language than the app
`src/codeEditor/EditorWindow.py:30-93` (`STYLE_LIGHT`) is GitHub-flavored: `#F6F8FA` surfaces, `#0366D6` blue accent, `#D0D7DE` borders. The app's light theme is Aurora: `#F3F7FC` bg, `#0077A8` cyan accent, `#DCE6F1` strokes. `STYLE_DARK` (`:100-174`) was already re-tuned to Aurora — the light half never was, so the floating editor looks like a different product in light mode.
**Fix:** rewrite `STYLE_LIGHT`'s colors onto the Aurora light palette (reference values in `tokens.py LIGHT`): surfaces `#F3F7FC`/`#FFFFFF`, text `#142033`/`#5A6E89`, accent `#0077A8`, focus border `#0077A8`, borders `#DCE6F1`, find-bar noMatch red `#E11D48` family. Keep the structure/selectors identical. The InfoBar amber block may stay warm but should use the light warning tone (`#D97706` family) rather than the GitHub yellows.

### 1.8 Plot-window trace-color menus are light-only design
`src/ngspiceSimulation/_list_mixin.py:318,324` and `_func_trace_mixin.py:44,50-52`: color-picker panels hardcode `background-color:#FFFFFF`, swatch hover `border:2px solid #212121`, hidden-trace text `#757575` (Material grey). In dark theme this is a glaring white popup, and `#757575` is used for "hidden" in both themes regardless of contrast.
**Fix:** both mixins live under the plotting tree, which already has a theme source — `ngspiceSimulation/_palette.py current_palette()`. Use `p["panel"]` for the popup background, `p["border_strong"]` for swatch borders, `p["text"]` for hover border, `p["text_subtle"]` for the hidden-trace label. No new imports from frontEnd needed (keeps the leaf-module rule).

### 1.9 The console/log rich-text layer never got a theme pass (systemic — Round 2 finding)
Every console that colors its output via inline HTML or QTextCharFormat uses fixed light-web colors, so dark theme gets invisible or neon text. This is one class of bug across six files:
- `src/maker/VerilogVerifier.py:1045-1052` — `_LOG_COLORS` is a GitHub-light palette; `'output': "#24292E"` (near-black) is unreadable on the dark `#0E1728` console card; `'info': "#57606A"` marginal.
- `src/maker/CosimLogger.py:104-114` — `_COLOR` table: `'info': '#000000'` (pure black → **invisible** on the dark console), `'phase': '#0000FF'` pure blue on dark.
- `src/ngspiceSimulation/NgspiceWidget.py:25-28` — `SUCCESS_FORMAT`/`FAILURE_FORMAT`: neon `#00ff00`/`#ff3333` at `font-size:26px` — off-palette in dark, unreadable green-on-white in light.
- `src/frontEnd/TerminalUi.py:129` — cancel banner `#FF8624` at 26px, off-palette both themes.
- `src/maker/NgVeri.py:220, 225, 243` — `#00FF00` / `#00AA00` / `#FF0000` neon status lines.
- `src/maker/ModelGeneration.py:311-324, 1587, 1604` — `#ff0000`/`#008000`, plus invalid `font-weight:1000` (max meaningful value is 900/Black — clamps silently; use 800).
**Fix:** add one shared helper — suggest `frontEnd/console_colors.py` (or a function in `tokens.py`): `console_colors(is_dark) -> dict` with semantic keys `info/ok/warn/error/head/output/detail`, mapping to tokens (dark: `text_muted / success #42E6A4 / warning #FACC15 / danger #FB7185 / accent #53D7FF / text / text_subtle`; light: the LIGHT-side equivalents `#5A6E89 / #059669 / #D97706 / #E11D48 / #0077A8 / #142033 / #9AAABE`). Then: `VerilogVerifier._LOG_COLORS` and `CosimLogger._COLOR` become calls to it (re-read on theme change — both classes already have changeEvent/reconstruction paths; CosimLogger can resolve at emit time); the f-string/`.format` HTML sites interpolate from it. While there, shrink the 26px banners to 15-16px weight 700 — a status line four times body size is shouting, not hierarchy. Note: consoles keep their own dark-ish backgrounds in some designs — pick colors against the ACTUAL console background per theme, not the window background.

### 1.10 Plot cursor readout mixin hardcodes dark-on-light text (extends §2.6)
`src/ngspiceSimulation/_cursor_mixin.py:59-69, 123-137, 178-192, 361` — pervasive `color:#333`, `#555`, `#999`, `#aaa` in the cursor value HTML. In dark theme these sit on the dark plot panel → `#333` values effectively invisible. `_palette.py` already defines `cursor_dim`, `cursor_chrome`, `cursor_disabled`, `stats_text` for exactly these roles.
**Fix:** thread `self._palette` into the mixin's HTML f-strings (the mixin is mixed into plotWindow, which owns `self._palette`) — replace each literal with the matching palette key. Same one-move fix as §2.6; do them together.

---

## P2 — Design inconsistencies

### 2.1 The elevation system exists but almost nothing uses it (unfinished port)
`src/frontEnd/elevation.py` was written to replace "scattered per-widget shadow magic numbers" (its own docstring) — light-mode shadows tinted blue-grey via `tokens.shadow_rgb` so they read as ambient occlusion instead of invisible black. Adoption: **one** call site (`ProjectExplorer.py:112`). Everything else still hand-rolls black shadows:
- `browser/Welcome.py:415-422` `_apply_tile_shadow` — black, alpha 48-68 (invisible on light).
- `frontEnd/motion.py:68-90` `set_shadow` / `apply_panel_depth` / `apply_popup_depth` — black default.
- `frontEnd/motion.py:643-665` `apply_toolbar_depth` — black.
- `frontEnd/tooltips.py:85-89` — black 160.
- `frontEnd/widgets.py FloatingDockHost` reads tokens directly (fine, but it's dead code — §P3.2).
**Fix:** route all of the above through `elevation.elevate(widget, level)` (Welcome tiles → `e2`, hero → `e3`, toolbars → `e3`, popups/messageboxes → `e4`, tooltip card → `e3` with its existing blur kept if `e3` looks too heavy — adjust the `_SCALE` table rather than bypassing it). `motion.set_shadow` itself can stay as the low-level primitive but should default its color from `tokens.theme(is_dark)["shadow_rgb"]` instead of black, which fixes every remaining caller in one move. Delete the then-redundant `Welcome._apply_tile_shadow`.

### 2.2 Welcome page hover/hero glow double-paints and hardcodes accents
- Double hover: `QFrame#welcomeCard:hover` in QSS tints the card (`style_dark.qss:1270-1273`) AND `ToolCard.paintEvent` paints a second translucent cyan fill (`browser/Welcome.py:165-174`) driven by the hover animation. Two overlapping highlight systems.
- Hardcoded accents: `Welcome.py:77-79` (glow colors), `:171` (fill `83,215,255`), `:290-291` (hero orb `#53D7FF`/`#9B7CFF` literals).
**Fix:** delete the QSS `:hover`/`:focus` blocks for `#welcomeCard` (both sheets) and let the animated painter own hover entirely (it's the richer effect). Replace literal colors with `tokens.theme(dark)["accent"]` / `["accent_2"]`. Also delete the unused `GradientLabel` import at `Welcome.py:12-15`.

### 2.3 FlowNavigator duplicates tokens while claiming to use them
`src/maker/FlowNavigator.py:221-250` `_pill_tokens` — comment says "values are Aurora's own tokens (frontEnd/tokens.py)" but every value is a hand-copied literal. Any future palette tweak drifts silently.
**Fix:** import `tokens` (guarded try/except like other modules) and build the dict from `tokens.theme(dark)`: `bar_bg`→`bg_raise`, `bar_border`/`seg_border`→`stroke`, `seg_bg`→`surface`, `seg_fg`/`stage_fg`→`text_muted`, hover fg→`text`, `accent`→`accent`, `accent_fg`→`text_invert`, checked bg→accent at 0.18/0.13 alpha (compute rgba string from `hex_to_rgb`), reload tones→`warning`. Keep the dict shape so `_apply_pill_theme` is untouched.

### 2.4 About surfaces duplicate the palette instead of reading tokens
`src/frontEnd/dialogs.py:109-133` `_about_palette` hand-copies ~14 values per theme (most match `tokens.py`, which is exactly why it will drift). Consumed by both `show_about_dialog` and `PreferencesDialog._build_about_page`.
**Fix:** derive the dict from `tokens.theme(dark)` (`page`→`surface`, `title`→`text`, `muted`→`text_muted`, `subtle`→`text_subtle`, `accent`/`link`→`accent`/`accent_hi`, pills computed from accent rgb). Keep the function and its shape; only change where values come from.

### 2.5 FullScreenToggle still uses native OS icons
`src/frontEnd/FullScreen.py:41-44` uses `SP_TitleBarMaxButton`/`SP_TitleBarNormalButton` — the platform-divergent look `icon_paths.py` was explicitly created to eliminate (its `folder_icon`/`file_icon` docstrings state that rationale). Meanwhile `icon_paths.fullscreen_icon` and `dock_back_icon` sit unused (`icon_paths.py:77, 93`) — they are visibly the intended pair.
**Fix:** in `_set_state`, `self.setIcon(dock_back_icon() if full else fullscreen_icon())`. Icons bake in the theme fg color, so also refresh on `PaletteChange` (a tiny `changeEvent` override calling `_set_state(self._full)`; track the bool).

### 2.6 Plot-window cursor readouts hardcode colors that `_palette` already defines
`src/ngspiceSimulation/plot_window.py:710-716` — HTML labels hardcode `#e53935`, `#1976d2`, `#e65100`, `#aaa`; `:890` hardcodes `#444444`. `_palette.py` defines the same values as `cursor1/cursor2/cursor_delta/cursor_disabled/stats_text` tokens.
**Fix:** interpolate `self._palette[...]` into those f-strings/labels so a palette change (or the dark palette, which likely defines different cursor chrome) actually reaches them.

### 2.7 QSS font-weight scale is arbitrary
Dark sheet uses weights 600, 650, 700, 750, 800, 820, 850, 860 (`grep font-weight style_dark.qss`); light is the same ±1. Eight ad-hoc steps is noise, and with Inter as a *variable* font on Qt < 6.7 the intermediate weights may not even instantiate (Qt registers a single instance; weights then quantize) — so 650 vs 700 vs 750 likely renders identically on some installs while implying intent.
**Fix:** normalize to a 3-step scale: 600 (medium emphasis), 700 (headings/buttons), 800 (caps-labels/hero). Pure find-and-replace in both sheets: 650→600, 750→700, 820/850/860→800. Verify nothing visually regresses at 100% zoom.

### 2.8 `messageKind` property is set but never styled
`src/frontEnd/dialogs.py:7` sets `msg.setProperty("messageKind", kind)` ("error"/"warning"/"info"/"question") — no QSS selector `[messageKind=...]` exists in either sheet, so all message boxes look identical.
**Fix (pick one):** (a) intended-but-unfinished: add a subtle per-kind accent to `QMessageBox#esimMessageBox[messageKind="error"]` etc. — e.g. a left `border-left: 3px solid` in `danger`/`warning`/`accent` tokens, both sheets; or (b) delete the property line. (a) is a tasteful, cheap differentiator; prefer (a).

### 2.9 Hardcoded font families bypass the app font stack (Round 2 finding)
- `src/maker/VerilogVerifier.py:526` — `QFont("Segoe UI", 10, ...)` for the sidebar label: Windows-only family, overrides the app-wide Inter stack; on Linux silently falls back. Delete the setFont (label inherits the QSS `QWidget` font) or set only size/weight on `self.font()`.
- `src/maker/VerilogVerifier.py:788` — `QFont("Consolas", 11)` for the console: Windows-only; should be the mono stack. Use `widgets.mono_family()` (kept alive per §3.2) or, once commit `110228cc` lands, `fonts.MONO_FAMILY`.
- `src/codeEditor/PlainEditor.py:26` — `QFont("Monospace", 11)`: an X11 alias; the `setStyleHint` makes it degrade acceptably, but align it with the same mono resolver for an identical face on all platforms.

---

## P3 — Dead code, dead rules, dead assets (safe deletes)

Everything here was verified to have **zero** non-test consumers in the tree (grep receipts noted). Deleting is behavior-neutral.

### 3.1 Dead QSS rule groups — remove from BOTH sheets
| Selector group | Why dead |
|---|---|
| `QLabel[cssClass="gradientTitle"]`, `QLabel[cssClass="heroGradient"]` | superseded by painted `GradientLabel` (itself unused); no `setProperty` sites |
| `QPushButton[cssClass="labeledIcon"]` | no setters |
| `QPushButton[cssClass="swatch"]`, `[cssClass="colorPick"]`, `QLabel#preferencesPreviewStrip` / `[cssClass="previewStrip"]`, `QFrame#themePreview`, `QTabWidget#preferencesTabs` | the old Preferences design (color swatches / preview strip / tabs); current dialog uses nav-rail + segment control only |
| `QLabel[cssClass="error"]`, `[cssClass="warning"]`, `[cssClass="success"]` | no setters (status colors are set in Python) — OR keep and adopt in §1.3/§2.8 work; decide once, be consistent |
| `#editorFindBar`, `QToolBar#spiceEditorToolbar`, `QStatusBar#spiceEditorStatus` (dark `style_dark.qss:1280-1343`, light mirror) | the old SpiceEditor; replaced by `codeEditor/` module which styles itself |
| `QPlainTextEdit#verilogCodeEditor` (dark `:1568-1582`) | verifier editor is QScintilla now; objectName never set |
| `QPushButton[dockPopButton="true"]` all 4 blocks incl. `[isPoppedOut]`/`[isCloseBtn]` (dark `:709-741`) | no Python sets `dockPopButton` — also delete the matching special-case branches in `motion.py:192,263-279` |
| `#dockFloatHost`, `#dockTitleBar`, `#dockCardTitle`, `#dockDropOverlay` | only ever created by dead `widgets.py` classes (§3.2) |
| `QWidget#dockToolChrome` | no setter anywhere |

Keep the file-level comments that explain live decisions; delete comments describing deleted rules.

### 3.2 `widgets.py` — ~80% of the module is unreferenced
Used: nothing. (`GradientLabel` is imported by Welcome but never instantiated.) Unused classes: `GradientLabel`, `AuroraHeroFrame`, `RailDragGrip`, `DockTitleBar` (+ its `MIME` drag protocol), `DockDropOverlay`, `FloatingDockHost`, and `mono_family()`. These represent a serious dock-drag UX investment (Wayland-safe undock/redock) that the current `DockArea` implementation does not mount — probably superseded by the card/tab design in `DockArea.apply_fullscreen_feature`.
**Fix:** confirm the current dock UX is the keeper (it is — DockArea's tabified cards + `FullScreenToggle` are what ships), then delete the dead classes. **Exception — keep `mono_family()` and adopt it** (§C6): it solves the real "Consolas is Windows-only" problem for the editors. If deleting `DockTitleBar` feels risky given the engineering notes in its docstrings, move the file wholesale to `docs/` or a git tag instead of keeping dead code importable — history preserves it either way.

### 3.3 `motion.py` dead entry points
`install_popup_motion` (`:556`), `install_effect_refresh` (`:595`), `install_menu_depth` (`:668`) — superseded by `install_app_motion`; zero callers. `EffectShowRefreshFilter` is still referenced by `AppWideMotionFilter._revalidate` — keep the class, delete only `install_effect_refresh`. Also the `dockPopButton` branches (see §3.1).

### 3.4 `icon_paths.py` dead factories
`backup_icon`, `close_proj_icon`, `copy_icon`, `close_icon` — zero callers. `fullscreen_icon`, `dock_back_icon` — currently zero callers but become live via §2.5. Delete the first four (with their SVG constants) after double-checking no dynamic `getattr` dispatch (none found).

### 3.5 Orphaned SVG assets in `frontEnd/images/`
`text_find_*.svg`, `text_save_*.svg`, `text_save_as_*.svg`, `text_wrap_*.svg` (6 files — old SpiceEditor toolbar), `dock_pop_*.svg` (4 files — dead dockPopButton). No `.py`/`.qss` references. Delete.

### 3.6 Misc hygiene
- `src/frontEnd/Application.py:304-320` — commented-out SoC button block with its own commented stylesheet. Delete (git keeps it).
- `src/frontEnd/PreferencesDialog.py:480-488` — `import json as _json` + `json_load`/`json_dump` wrappers at the bottom of the file, below the class that uses them. Move to a normal top `import json` and call `json.load/json.dump` directly; also `reject()`'s second write (`:362-366`) should use `paths.write_json_atomic` like `_apply_preferences` does, not a bare `open(...,"w")`.
- `src/frontEnd/theme_utils.py:455-459` comment references "SpiceEditor / ProjectExplorer" changeEvent re-styling — SpiceEditor no longer exists; update comment while in there.

---

## P4 — Architecture / consolidation (do after P1-P3)

### C1 Token single-source-of-truth is aspirational, not real
`tokens.py` declares "Nothing else in the codebase should hard-code a brand hex" — today there are **six** parallel palette definitions: `tokens.py`, `theme_utils.py` (QPalette blocks `:393-428` + titlebar colors `:80-81` + the three `*_TOKENS` replace-lists), `dialogs._about_palette`, `FlowNavigator._pill_tokens`, `codeEditor/EditorWindow.STYLE_*`, `ngspiceSimulation/_palette.py`. §2.3/§2.4 fix two of them. For the rest:
- `theme_utils.apply_theme` palette blocks: build from `tokens.theme(is_dark)` (Window→`bg_raise`? — careful: current palette Window is `#050812`, tokens `bg` is `#05070F`; **decide the canonical value first**, see C2). Titlebar caption colors likewise.
- `_palette.py` deliberately avoids frontEnd imports (headless tests). Keep that, but its light defaults are Tailwind-grey (`#1F2937/#6B7280/#165982`) — visibly not Aurora next to the rest of the app. Retint its `_LIGHT_DEFAULTS`/`_DARK_DEFAULTS` to the Aurora values (text `#142033`, muted `#5A6E89`, primary `#0077A8`, surfaces `#F3F7FC/#FFFFFF`; dark: text `#F4F8FF`, surfaces `#05070F/#0E1728`), and add a small test asserting the shared keys match `tokens.py` so drift becomes a test failure instead of a slow visual rot.

### C2 tokens.py vs theme_utils vs QSS value drift
Concrete mismatches to reconcile when doing C1: tokens `DARK.bg #05070F` vs palette Window/QSS `#050812`; tokens `DARK.text #F4F8FF` vs palette/QSS `#F8FBFF`; tokens `LIGHT.bg #EEF3FA` vs palette Window `#F3F7FC` (tokens calls that `bg_raise`). The QSS files are the visually-tuned truth — update `tokens.py` to match the sheets, not the other way around.

### C3 The accent/secondary/internal customization engine is vestigial
`PreferencesDialog` pins `accent_color: "default"`, `secondary_accent_color: "system"`, `internal_bg_color: "system"` (`_collect_prefs`, `:420-430`) — the UI cannot set anything else, so `ACCENT_TOKENS`/`SECONDARY_TOKENS`/`INTERNAL_TOKENS`, `recolor_accent_rgba`, `DEFAULT_ACCENT_RGB` and the accent parameters of the cache key only serve preference files written by older builds. It's ~80 lines of find-and-replace machinery.
**Recommendation:** keep it (it's tested by `test_theme_qss_cache.py` and harmless) but add one sentence to the `tokens.py` docstring saying custom accents are currently write-locked in Preferences, so the next reader doesn't assume it's reachable. Alternative (bolder): strip it and simplify `build_qss` to `(qss_name, zoom)` keys. Only do the bold version if the FOSSEE upstream confirms custom accents are permanently out.

### C4 `is_dark` predicate duplicated in 11 files
`palette().color(Window).lightness() < 128` re-implemented in `Welcome.py, codeEditor/theme.py, dialogs.py, elevation.py, icon_paths.py, motion.py, theme_utils.py, tooltips.py, widgets.py, FlowNavigator.py, _palette.py`.
**Fix:** low priority; when touching any of these files for other findings, switch to a single helper (`elevation.is_dark(widget)` for widget-scoped, `theme_utils.current_theme_is_dark()` for app-scoped). Do NOT do a big-bang sweep — no behavior gain, churn risk. `_palette.py` stays self-contained (leaf rule).

### C5 Welcome hover system duplicated hover machinery
`HoverSurfaceMixin` (`Welcome.py:56-107`) is a nice generic mixin but is used by exactly one class in one file. If §2.2's cleanup keeps it, fine; if a second card-like surface ever appears (e.g. Workspace picker), move the mixin into `widgets.py` (which after §3.2 becomes the small, honest "live reusable widgets" module).

### C6 Monospace / font stack robustness
QSS mono stacks say `"JetBrains Mono", "Cascadia Mono", "Consolas", monospace` (`style_dark.qss:1354, 1448, 1459, 1577`) but JetBrains Mono is **not bundled in this tree** (only `images/fonts/Inter-VariableFont_opsz,wght.ttf` exists) — silent fallback per-platform, and `widgets.mono_family()` (the portable resolver) is dead code. The proper bundling lives in unpushed commit `110228cc` on another branch.
**Fix:** after that commit lands in `dev`, verify the QSS families match the bundled font's real family name and that registration happens before the first sheet build. If it doesn't land, either bundle JetBrains Mono here or drop it from the stacks so the declared stack is honest. Also verify Inter *variable* font actually yields bold weights on the minimum supported Qt (6.4 on Ubuntu 24.04) — if headings render regular-weight there, ship the static Inter weights (Regular/SemiBold/Bold) instead of the variable file.

### C7 Zoom scaling reaches QSS but not local sheets
`build_qss` scales px metrics; every locally-set sheet (editor chrome §1.7, FlowNavigator pills, About dialog `setFixedSize(440,500)` at `dialogs.py:210`, tooltip card) ignores `zoom_level`. At 150%+ the main UI grows while these stay small.
**Fix:** pragmatic scope — leave text-driven surfaces alone (they follow font metrics), but replace the About dialog's `setFixedSize` with `setMinimumSize` + `adjustSize` so it tracks content, and let `_apply_view_control_metrics`-style scaling stay toolbar-only. Full zoom propagation to local sheets is not worth the complexity.

---

---

## FIX LOG

### S1 — 2026-07-22 · C2, C1 (`_palette.py` retint half), 1.1, 1.2, 1.4, 1.5

Verifier: `audit_harness/verify_ui_s1.py` (11/11 offscreen). Guards green:
`test_theme_qss_cache` · `test_view_control_metrics` · `test_motion_idempotent` ·
`test_plot_window_theme` · `test_console_migration` + the new drift test = 69 passed.
Ruff `F,E9,B,E501,W291,W293` on every touched file: **no new findings** (one
pre-existing `F401 QtGui imported but unused` cleared by §1.1). Both sheets
re-parse through Qt with no "Could not parse stylesheet".

**C2 — tokens.py reconciled to the sheets.** The sheets are the visually-tuned
truth, so `tokens.py` moved, not them. What the audit listed:
- `DARK.bg` `#05070F` → **`#050812`** — the value `QMainWindow`/`QDialog`/`QFrame`
  paint and `QPalette.Window` carries (6 receipts in `style_dark.qss`).
- `DARK.text` `#F4F8FF` → **`#F8FBFF`** — the old value appeared in *neither*
  sheet; `#F8FBFF` appears 40×.
- `LIGHT.bg` `#EEF3FA` → **`#F3F7FC`** — same window/palette rule as dark.

Found by the same method and fixed with it (the audit spot-checked; this is the
rest of the same drift class):
- `text_muted` was a phantom in both themes — `#9FB1CC`/`#5A6E89` appear 0×/1×.
  Now **`#94A8C3`**/**`#6B7F99`**, the literal `QLabel[cssClass="muted"]` value,
  which is *also* what `theme_utils` sets `QPalette.PlaceholderText` to. This
  mattered before §2.3/§2.4/§1.9 consume `text_muted` — they would have
  introduced a new off-palette grey into three more files.
- Two keys added rather than values lost: **`bg_sunken`** (`#05070F`/`#EEF3FA`)
  keeps the old `bg` pair alive under an honest name — it is the dock workspace
  floor, 3 receipts each at the same three selectors; **`text_dim`**
  (`#D3DEEF`/`#405168`) names the most-used text tier after `text` (11 receipts
  each, exact 1:1 mirror: menubar, toolbuttons, consoles) which §1.9 needs.
- `LIGHT.bg_raise` moved `#F3F7FC` → **`#F8FBFF`** since `bg` took its old value.
- Docstring now spells out the background ladder
  (`bg_sunken < bg < bg_raise < surface < surface_2 < surface_3`) so the next
  reader does not have to re-derive it.

Blast radius was zero: nothing in the tree read `bg`/`text`/`text_muted` yet
(only `shadow_rgb` and `DEFAULT_ACCENT_RGB` have consumers), which is exactly
why the values had rotted. **For §C1's remaining half: `QPalette.Window` maps to
`bg` in both themes now, not `bg_raise`.** Note for §2.3: FlowNavigator's
`bar_bg` (`#0A1020`/`#F3F7FC`) is `bg_raise` in dark but `bg` in light — the
light chrome strips sit flat on the window by design; don't "fix" that to
`bg_raise` or light chrome shifts.

**C1 (retint half) — `ngspiceSimulation/_palette.py` is Aurora now.** Both
`_LIGHT_DEFAULTS`/`_DARK_DEFAULTS` rebuilt off the corrected tokens: surfaces,
borders, text tiers, brand, overlays, and the matplotlib axes/legend/grid roles.
The Tailwind greys the audit named (`#1F2937`/`#6B7280`/`#165982`, plus
`#9CA3AF`/`#E5E7EB`) are gone. Two deliberate calls:
- The plot canvas now reads as a **card on the window backdrop** in both themes
  (`figure.facecolor` = `bg`, `axes.facecolor` = `surface`) instead of the flat
  white-on-white / same-on-same it was.
- **Cursor marker hues (`cursor1`/`cursor2`/`cursor_delta`) were left alone** —
  same rule as `VIBRANT_COLOR_PALETTE`: they are data identity, not chrome. Only
  the grey cursor *chrome* (`cursor_chrome`/`cursor_dim`/`cursor_disabled`)
  moved onto the Aurora text ladder.
- The leaf rule holds — `_palette.py` still imports nothing from `frontEnd`.
  New `src/ngspiceSimulation/tests/test_palette_tokens_match.py` pins the 20
  genuinely-shared keys to `tokens.py` in both themes, asserts both themes
  define the same key set, and rejects named colors (`"gray"`, `"white"`). Drift
  is a test failure from here on.
- `theme_utils.apply_theme`'s palette blocks were **not** touched (that is §C1's
  other half, out of this session's scope) — but its literals now equal the
  tokens they mirror, which the verifier asserts.

**1.1 — ModelEditor delegate.** Deleted the whole per-delegate `setStyleSheet`
*and* `createEditor`, which existed only to carry it, *and* `_is_dark`, which
existed only to feed it. The app sheet's `QAbstractItemView QLineEdit` rule was
already doing the work (the doubled-brace sheet was discarded by Qt every time).
Class docstring rewritten to say where the color comes from and why there is no
local sheet. `updateEditorGeometry` + `setEditorData` — the parts that actually
fixed the clipped/select-all editor — are untouched.

**1.2 — TerminalUi.** Both `styleSheet` properties deleted from
`TerminalUi.ui`; the console already carried `objectName="simulationConsole"`, so
`QPlainTextEdit#simulationConsole` reaches it and the progress bar falls back to
the global `QProgressBar` rule. Verified live in both themes: console background
resolves to `#08111F` dark / `#FBFDFF` light. Added a comment in `TerminalUi.py`
warning that a Designer-set `styleSheet` beats the app sheet in every theme —
that is precisely how the purple-black console survived a switch to light mode.
(§1.9's 26px `#FF8624` cancel banner in `TerminalUi.py` is untouched, still open.)

**1.4 — 12 legacy gray group boxes deleted** (`Source.py` ×6, `Analysis.py` ×3,
`Microcontroller.py` ×2, `Model.py` ×1 — the audit said 13 and named two sites in
`Model.py`; there is only one). `kicadtoNgspice/` now contains **zero**
`setStyleSheet` calls. No layout change: the global `QGroupBox` rule already
supplies border/radius plus a larger `margin-top` + `padding`, verified by
asserting a themed box still gets non-zero contents margins in both sheets.

**1.5 — dock title-bar button rules collapsed to one.** Deleted `style_dark.qss`
`626-662` and the light mirror `633-669`; only the 0×0 hiding block remains, and
the comment now records why nothing may follow it. Both sheets had the identical
range removed, so the structural diff stays clean (§1.6's goal). Side effect for
§3.5: `images/dock_fullscreen_{dark,light}{,_hover}.svg` (4 files) are now
orphaned — `dock_close_*` are **still live** via `QTabBar::close-button`, do not
delete those.

**Not done in S1, still open in the areas touched:** §1.3, §1.6, §1.7, §1.8,
§1.9, §1.10, all of P2/P3, and §C1's `theme_utils`/`recolor` half.

---

## Verification checklist for the fixing session

1. `pytest src/frontEnd/tests src/ngspiceSimulation/tests/test_plot_window_theme.py -q` green.
2. Launch app (`python src/frontEnd/Application.py` per repo run docs), toggle Light/Dark:
   - KicadToNgspice window: group boxes themed (no gray borders) — §1.4.
   - Simulation: status dot readable in light — §1.3; console + progress themed in both — §1.2.
   - Model Editor: edit a cell — editor styled, no "Could not parse stylesheet" on stderr — §1.1.
   - Plot window: trace color menu themed in dark — §1.8; cursor labels legible — §2.6.
   - Editor window (open a .cir/.sub): light chrome matches app light theme — §1.7.
   - Verilog flow: pills/tabs still correct after tokenization — §2.3.
   - Welcome: single clean hover highlight, shadows visible in light theme — §2.1/§2.2.
   - Fullscreen toggle icon is the custom SVG in both themes — §2.5.
   - Verilog verifier console + NgVeri/cosim logs: every line readable in dark theme — §1.9.
   - Plot cursors: place C1/C2, readout values legible in dark — §1.10.
3. Zoom 100/150/200%: pill + theme toggle stay aligned (existing `test_view_control_metrics` covers the math).
4. `grep -c "dockPopButton\|labeledIcon\|swatch\|previewStrip\|spiceEditor" src/frontEnd/*.qss` → 0 after §3.1.
5. Structural sheet diff (the parser used for this audit): selectors and property keys identical between dark/light except palette values — §1.6.

## Round 2 sweep — coverage statement (audit is COMPLETE)

A second pass was run specifically over areas the first pass covered only by grep. Results:
- **Repo-wide `.ui` files:** `TerminalUi.ui` is the only one; already covered (§1.2).
- **Rich-text/HTML color literals repo-wide:** produced §1.9 and §1.10 — that was the one systematic gap.
- **Hardcoded `QFont` families repo-wide:** produced §2.9. Only 3 live sites (plus one already-commented-out line in Source.py).
- **`DockArea.py` full styling scan:** clean — no color literals, no inline sheets beyond what §P3.1 already covers.
- **`ProjectExplorer.py` full styling scan:** clean — uses `icon_paths` + `elevate(e2)`; the stale-project `QColor('gray')` italic treatment reads fine on both themes, leave it.
- **`CosimLogger`, `NgspiceWidget`, `TerminalUi`, `NgVeri`, `ModelGeneration` consoles:** covered by §1.9.
- Not audited by design (out of scope, don't chase): QScintilla lexer style-by-style correctness in `codeEditor/theme.py` (palettes reviewed, classification logic not — it has its own structure and no reported defects); Makerchip/UserManual QWebEngineView inner content (web pages, not our chrome); Ubuntu-specific WM behavior (needs a live Linux box); pixel-perfect spacing/rhythm review (needs eyes on a running app — do opportunistically while fixing).

No further audit sessions needed. Everything actionable is in this file; remaining work is execution.

## Explicitly NOT findings (checked, fine — don't "fix")
- QSS image url rewriting to absolute paths (`build_qss`) — correct and CWD-independent.
- `_QSS_CACHE` unbounded dict — key space is tiny by construction.
- `ComboPopupStyle` proxy + polish recursion guard — correct, documented, keep.
- Windows DWM titlebar/rounding paths, translucency vs mask decisions in `motion.py` — carefully reasoned, verified against the comments; keep exactly as is.
- Double `_refresh_graphics_effects` (sync + next-tick) — deliberate, documented, needed.
- Subcircuit tile labels hardcoding white text — they sit over image thumbnails, theme-independent by design.
- `QTreeWidget#projectTree` branch/selection rule pile — verbose but each state is load-bearing (gutter transparency).
