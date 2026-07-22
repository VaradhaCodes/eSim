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

### S2 — 2026-07-22 · 1.6, 1.7, 2.7, 2.8, 2.2 (QSS half)

Verifier: `audit_harness/verify_ui_s2.py` (16/16 offscreen). Guards green:
`test_theme_qss_cache` · `test_view_control_metrics` · `test_motion_idempotent` ·
`test_plot_window_theme` · `test_palette_tokens_match` = 66 passed. Both sheets
and both editor sheets re-parse through Qt with no "Could not parse stylesheet".

**1.6 — the structural diff of the two sheets is now empty.** The verifier
parses both sheets into `selector -> ordered property keys` and asserts the maps
are identical; that assertion is the finding's stated goal, and it is now a
test. Closed the five gaps the audit named: light's `QPushButton:disabled`
`font-style: italic` deleted **with** the `font-style: normal` counter-patch it
forced onto light primaries; light gained `QCheckBox/QGroupBox::indicator:hover`
`background: #F6F9FD`, `QPushButton:default:pressed` `border-color: #0077A8`,
and `verifierPrimary:disabled` `border-color: #DCE6F1`; dark-only
`QLabel[cssClass="heroGradient"]` deleted (§3.1 lists it as dead — no
`setProperty` site, and `GradientLabel` supersedes it). Two more the diff caught
that the audit's spot-check missed: light `dockPopButton:hover` had no `color`
where dark did, and its `isCloseBtn:hover` said `background` where dark said
`background-color` — same paint, different property key, so the parity test saw
them as different rules. Also normalized property *order* in two hover rules so
the ordered comparison holds.

**2.7 — eight ad-hoc font weights collapsed to three.** 650→600, 750→700,
820/850/860→800, applied to both sheets: 6 sites each. Weights now read as
600 medium / 700 headings+buttons / 800 caps-labels+hero. The verifier asserts
the union of weights across both sheets is exactly `{600,700,800}` *and* that
the two sheets' weight histograms match, so a future 650 in one sheet fails.
This also removes the Qt-variable-font trap the audit flagged: 650 vs 700 vs 750
quantize to the same instance on some installs, so the old scale implied
distinctions the renderer never drew.

**2.8 — `messageKind` is styled (option (a)).** Four
`QMessageBox#esimMessageBox[messageKind="..."]` rules per sheet, a 3px
`border-left` in `danger`/`warning`/`accent`/`accent_2` — the audit specified
the first three; question got `accent_2` (violet) so it is not a second info
box. Verified against the *setter*, not the docs: the verifier greps
`dialogs.py` for every literal passed to `setProperty("messageKind", …)` and
asserts each one has a rule in both sheets, so a new kind added in Python fails
the test instead of silently rendering unstyled. Also asserts the stripe
actually renders (box polished offscreen in both themes, left edge sampled).

**2.2 (QSS half only) — the double hover is gone.** Deleted
`QFrame#welcomeCard:hover` and `:focus` from both sheets; `ToolCard.paintEvent`
now owns hover alone, as the audit directed (it is the richer effect — animated
wash + shadow glow off one progress value). The `:focus` rule only restated
`border: none`, which every state already has. Comment left at the surviving
resting-state rule saying why nothing may follow it. **The Python half of §2.2
is NOT done** — `Welcome.py:77-79, 171, 290-291` still hardcode accent literals
and the unused `GradientLabel` import is still there.

**1.7 — `EditorWindow.STYLE_LIGHT` is Aurora, and the pair is a strict
mirror.** Every GitHub value is gone (`#F6F8FA`/`#0366D6`/`#D0D7DE`/`#E1E4E8`/
`#57606A`/`#E1604D` and the `#FCE5C0` amber family). Light now takes its values
straight from `tokens.LIGHT` — surfaces `#F3F7FC`/`#F8FBFF`/`#FFFFFF`, text
`#142033`/`#405168`/`#6B7F99`, accent `#0077A8`, focus `#0077A8`, no-match
`#E11D48` — and dark was corrected in the same pass (`#E6EDF7` → the real
`text` token `#F8FBFF`, twice; `selection-color` added so a selected find-field
string is not dark-on-dark). Structure is identical between the two, which the
verifier checks the same way it checks the app sheets.

Three deliberate non-mirrors, all documented in the header comment: the
no-match field, `#infoBar`, and `#findClose:hover` cannot share alphas, because
what is an opaque wash on dark must be a tint on light — at the dark alpha 0.24
the light close glyph lands at 4.28:1, and 0.14 lifts it to 5.0. The InfoBar
stays warm in both themes but drops GitHub's yellows for the Aurora warning hue
`#D97706` as the wash, with text at amber-800/900 (`#92400E`/`#78350F`). **That
is the one place this sheet leaves the LIGHT token set**, and it is deliberate:
no Aurora light token is dark enough to put body text on a warm tint at 4.5:1,
where that ramp reaches 5.9:1 and 7.6:1. The verifier computes WCAG contrast for
every tinted light surface (composited over its real backdrop, not assumed) and
requires AA.

**Not done in S2, still open:** §1.3, §1.8, §1.9, §1.10, §2.1, §2.2 (Python
half), §2.3, §2.4, §2.5, §2.6, §2.9, all of P3, §C1's `theme_utils` half,
§C3-C7.

---

### S2 — 2026-07-22 · 1.6, 2.7, 2.8, 2.2 (QSS half), 1.7

Verifier: `audit_harness/verify_ui_s2.py` (16/16 offscreen). S1's verifier
re-run on this tree: still 11/11. Guards green: `src/frontEnd/tests` +
`test_plot_window_theme` + `test_palette_tokens_match` +
`maker/test_highlighting` = **138 passed, 18 skipped, 0 failed**. Ruff
`F,E9,B,E501,W291,W293` on every touched file: clean. Both app sheets and both
editor sheets re-parse through Qt with no "Could not parse stylesheet".

**1.6 — the structural diff of the two sheets is now EMPTY.** That is the
audit's own success criterion, and it is a test from here on
(`p16_structural_diff_is_empty` compares selector order, property-key order and
every non-color value). The seven gaps the audit named:
- light `QPushButton:disabled` lost `font-style: italic`, and
  `QPushButton:default` lost the `font-style: normal` counter-patch it only
  existed to undo.
- light `QPushButton:default:pressed` gained `border-color: #0077A8` — dark
  paints the border one step *lighter* than the pressed fill (`accent` over
  `accent_lo`), so light paints it one step darker the same way.
- light `QPushButton[cssClass="verifierPrimary"]:disabled` gained
  `border-color: #DCE6F1` (`LIGHT.stroke`, mirroring dark's `DARK.stroke`).
- light `QCheckBox/QGroupBox::indicator:hover` gained `background: #F6F9FD` —
  dark tints the box one surface step up on hover, light now does too.
- the two `dockPopButton` hover rules: light gained the missing `color` and its
  `background` became `background-color` to match dark. These rules are **dead**
  (§3.1 deletes them together with `motion.py:192,263-279`) — they were made
  parallel rather than deleted so 1.6's guard can be absolute without pulling
  §3.1's Python half into this session.
- dark-only `QLabel[cssClass="heroGradient"]` deleted (dead per §3.1, and the
  last selector-level asymmetry). Its sibling `gradientTitle` is equally dead
  but symmetric, so it stays for §3.1.

Two more the audit did not have, surfaced by tightening the diff from *key sets*
to *declaration order*: `QPushButton[cssClass="secondary"]:hover` and
`[cssClass="tertiary"]:hover` each ordered `color`/`border-color` differently
between sheets — and inconsistently with each other *within* dark. All four
normalized to `background; color; border-color;`. The sheets are now diffable
line-for-line, which is what makes the guard cheap to keep.

**2.7 — weights normalized to 600/700/800.** `650→600` (×4), `750→700`,
`820→800`, `850→800`, `860→800` in both sheets; the histograms are now
identical (7/12/12) and asserted equal, so the two sheets cannot drift on
weight either. The audit's concern was real: on Qt < 6.7 the variable Inter
registers one instance and the intermediate steps quantize, so 650 vs 700 was
implying an intent the renderer never honoured.

**2.8 — `messageKind` styles for the first time (option (a)).** Four rules per
sheet give the box a 3px left severity spine: `error`→`danger`,
`warning`→`warning`, `info`→`accent`, `question`→`accent_2`. Verified by
*rendering*, not by grep — `p28_stripe_renders_in_both_themes` grabs each box
and asserts the three leftmost pixels at mid-height are exactly the expected
token, all 4 kinds × both themes. A second check parses `dialogs.py` for the
kinds `_prepare_msg` can actually emit and fails if the sheets ever cover a
different set. 3px is the widest stripe the 16px radius still curves cleanly
around; the rest of the border keeps its gradient.

**2.2 (QSS half) — one hover system on the Welcome cards.** `#welcomeCard:hover`
and `:focus` deleted from both sheets; the animated `ToolCard.paintEvent` (wash
+ shadow glow off one progress value) now owns hover alone. `:focus` was a
no-op restating the base `border: none`. Note this removes *nothing* visible in
light mode that the painter was not already drawing — the painter's cyan fill
was always painting there too, which is exactly the double-paint the finding is
about. **Still open (Python half):** `Welcome.py:77-79, 171, 290-291` keep
literal accents, so the light-theme hover wash is still dark-theme cyan; that
and the `GradientLabel` import are the rest of §2.2.

**1.7 — the editor window is Aurora in light too.** `STYLE_LIGHT` rewritten from
its GitHub palette (`#F6F8FA`/`#0366D6`/`#D0D7DE`) onto `tokens.LIGHT`, and the
two editor sheets are now **strict mirrors**: same selectors in the same order,
same property keys, same alphas — a guard asserts it, and a second one asserts
every hex in `STYLE_LIGHT` is a `LIGHT` token. Consequences worth recording:
- The emphasis tone steps *away from the background* in each theme, so dark's
  `accent_hi #8BEAFF` maps to light's `accent_lo #005E86`, not to `accent_hi`.
- `QMenu::separator` existed only in dark; light gained the mirror.
- `#findBar QLineEdit` gained `selection-color: #FFFFFF` in **both** sheets.
  Light was about to put `#142033` text on a `#0077A8` selection; the app sheets
  have paired `selection-background-color` with `selection-color` since S1, and
  the editor chrome was the one place that had not.
- Dark's `#E6EDF7` (a phantom — in no token and in neither sheet) became
  `text #F8FBFF`, the same drift class §C2 cleared.
- Contrast was computed, not eyeballed (offscreen Qt on this box exposes no
  fonts). Everything the sheet invents clears WCAG AA and is pinned by
  `p17_light_tinted_surfaces_are_legible`. Two values moved *because* of that
  measurement: the InfoBar text ramp went one step darker than first written
  (amber-800 `#92400E` / amber-900 `#78350F`, 5.9:1 and 7.6:1 — `#B45309` was
  4.2:1), and `#findClose:hover`'s wash dropped 0.24→0.14 so the *token*
  `danger_lo` clears 5.0:1 instead of 4.28:1 and light needs no invented red.
  The amber pair is the single place `STYLE_LIGHT` leaves the token set, and the
  comment above the sheets says why: no Aurora light token is dark enough to put
  body text on a warm tint at 4.5:1.
- The muted roles (tab labels, status bar, find count) land at ~4.0:1. That is
  `LIGHT.text_muted` behaving the same way it does across the whole light theme
  — a token-level call, deliberately not made here.

**Not done in S2, still open in the areas touched:** §2.2's Python half (above),
§1.3, §1.8, §1.9, §1.10, the rest of P2 (§2.1, §2.3, §2.4, §2.5, §2.6, §2.9),
all of P3, and §C1's `theme_utils`/`recolor` half. Observation for whoever takes
P3: with `:focus` gone from `#welcomeCard`, a keyboard-focused tile has no
visible ring at all (`* { outline: 0 }` kills the native one). That predates
this session — the deleted rule was `border: none` — but it is a real a11y gap
and wants its own finding rather than a quiet re-add.

---

### S3 — 2026-07-22 · 1.9, 1.3, 1.8, 1.10, 2.6

Verifier: `audit_harness/verify_ui_s3.py` (22/22 offscreen). S1 and S2 re-run on
this tree: still 11/11 and 16/16. Guards green: `src/frontEnd/tests` +
`src/ngspiceSimulation/tests` + `test_cosim_logger` + `test_verifier_layout` +
`test_verifier_run` + `test_highlighting` = **240 passed, 18 skipped**. Full
suite **581 passed / 9 failed / 22 skipped** — the 9 are the documented
pre-existing Windows env failures (`test_toolchain_check` ×6, `test_cosim_config`
vvp, `test_nghdl_embed` ×2), byte-identical to the baseline set. Ruff
`F,E9,B,E501,W291,W293` diffed against the pre-session tree for all 14 touched
files: **zero new findings, 15 pre-existing cleared**.

**1.9 — the console layer has one palette, and it is measured.** New
`src/frontEnd/console_colors.py`: `console_colors(is_dark)` returns seven
semantic levels — `info / detail / ok / warn / error / head / output` — plus
`current_console_colors()`, which resolves the live theme from the QPalette that
`apply_theme` installs (falling back to `current_theme_is_dark()`, then light,
and never raising: two of the six callers run on worker threads and one is
imported by a test with no QApplication at all).

The audit's suggested map was followed with three measured corrections, all
against the **console's own** background rather than the window's
(`#0E1728`/`#08111F` dark, `#FFFFFF`/`#FBFDFF` light — the real values from the
two sheets, and what the verifier composites against):
- light `ok` and `warn` are **not** `LIGHT.success`/`LIGHT.warning`. Those are
  tuned to sit *beside* light body text; as body text on white they measure
  3.8:1 and 3.2:1. They take the next step down each ramp — `#047857`
  (emerald-700, 5.5:1) and `#B45309` (amber-700, 5.1:1). Same call `STYLE_LIGHT`
  made for its InfoBar in S2, same reason.
- light `detail` is `#5A6E89`, not `LIGHT.text_muted` (4.1:1). `#5A6E89` is the
  darker grey the light sheet already carries a receipt for. S2 accepted ~4.0:1
  for muted *chrome*; a log line is not chrome.
- `info` is `text_dim`, not `text_muted` — `text_dim` is literally what
  `QPlainTextEdit#simulationConsole` and `QTextEdit` paint, so uncoloured
  chatter and `info` land on the same tone by construction. `text_subtle` never
  became a level: at 2.4:1 on the light console it is not a colour, it is a
  disappearance.

Every value in both themes clears WCAG AA (4.5:1) on both of its backdrops, and
`p19_console_colors_clear_wcag_aa` recomputes that, so a retint that breaks it
fails instead of shipping. A second guard asserts no level is the same hue in
both themes — the failure mode where one side quietly never got retuned.

The six consumers:
- `VerilogVerifier._LOG_COLORS` is now a **property**, so all five existing call
  sites are untouched and each read resolves live. Its GitHub-light table is
  gone (`'output': #24292E`, near-black on the dark console card).
- **`VerilogVerifier` also re-colours its backlog.** `_append_console` bakes the
  colour into each fragment's `QTextCharFormat`, so a mid-session toggle used to
  leave a whole scrollback in the outgoing theme. New `changeEvent` →
  `_retheme_console()` walks the document and remaps every fragment through
  `old_theme[level] → new_theme[level]`; nothing extra had to be remembered
  because *the colour is the level*. Verified by rendering: the harness writes
  all seven levels in dark, flips, and asserts the document's fragment-colour
  set is exactly the light set with the text undisturbed.
- `CosimLogger._COLOR` became `_LEVEL` (level → semantic key) + a `_color()`
  classmethod resolving at emit time. `None` still means "emit no colour",
  preserving the existing split. `fix` moved from magenta `#B30086` to `head`
  (accent): it is actionable advice, not a failure.
- `NgspiceWidget.SUCCESS_FORMAT`/`FAILURE_FORMAT` → `_banner_format(level)`;
  `TerminalUi`'s cancel banner, `NgVeri`'s three build banners and
  `ModelGeneration._emit_error` / `verilogfile` likewise.
- All banners dropped from 26px/25pt to 16px (`BANNER_PX`) / 14pt, and every
  `font-weight:1000` became 800 — Qt clamps rich text at 900, so 1000 was
  asking for a weight that does not exist. Both are guarded.

**Known limit, deliberately not chased:** the five HTML consoles bake their
colour into the document too, but unlike `QTextCharFormat` there is no reliable
way back from rendered HTML to the level that produced it. Lines written *after*
a toggle are correct (that is what emit-time resolution buys); lines already on
screen keep their old tone until the next run clears the terminal. Only the
verifier console, which owns real char formats, re-colours retroactively.

**1.3 — the status dot follows the theme, and re-tints on the toggle.**
`_set_sim_status` reads `tokens.theme(current_theme_is_dark())` and records
`_last_sim_state`; new `_retint_sim_status()` replays it. `apply_theme`'s
top-level-widget sweep calls the hook (guarded by `hasattr` + try/except, like
its neighbours) — `_CURRENT_DARK` is assigned ~100 lines earlier, so the sweep
already sees the new theme. The dot's `padding` moved out of the Python sheet
entirely: `QLabel#simStatusDot` already carries it in both sheets. Verified
against the **real** `apply_theme`, not a stub: the dot goes `#FACC15` → `#D97706`
across a live Dark→Light apply with nothing else touching it.

**1.8 — the trace-colour popups are themed, and so are hidden rows.**
`populate_color_menu` and `_populate_func_color_menu` read `self._palette`
(`panel` for the card, `border_strong` for swatch borders, `text` for the hover
border) instead of `#FFFFFF`/`#E0E0E0`/`#212121`. While there, the same finding's
other half: the hidden-trace ring (`#9E9E9E`) and label (`#757575`) — Material
greys for a white list — both became `cursor_dim`, so a switched-off row dims as
one object. New `refresh_list_theme()` rebuilds every row on a theme change;
each row is a hand-built widget with an inline sheet and a painted swatch, none
of which a stylesheet swap can reach.

**Two `_palette` keys were deliberately left unconsumed, and both are documented
in place.** `cursor_disabled` and `cursor_chrome` read like text roles but hold
`border_strong` and `text_subtle` — 1.75:1 on the dark plot panel and 2.4:1 on
the light one. Adopting them would have swapped one invisible grey for another,
which is the finding, not the fix. `p110_no_readout_paints_text_with_a_border_tone`
keeps them out.

**1.10 / 2.6 — the cursor readouts, in one move as the audit asked.** Every
`#333`/`#555`/`#999`/`#aaa` in `_cursor_mixin` and every `#e53935`/`#1976d2`/
`#e65100`/`#aaa` in `plot_window` now comes from `self._palette`
(`stats_text` for values, `cursor_dim` for the secondary tier, `cursor1/2/delta`
for the marker hues, which stay data identity per S1's rule). Three things fell
out of doing it properly:
- The drag path and the full-readout path were building the same "C1 @ 1.234 ms"
  string with the same four literals written twice. They share
  `_cursor_head_html` now; `_delta_html` and `_cursor_placeholder_html` collapse
  four more copies.
- **The drawn cursor line did not match its own label.** `set_cursor` drew bare
  `'red'`/`'blue'` while the readout announced `#e53935`/`#1976d2` — two
  different reds for one cursor, and pure blue is poor on the dark panel (which
  is exactly why the dark palette brightens both). Both now take `_cursor_hue`.
- `_make_focus_icon` stopped being a `@staticmethod`: it painted `#444444`, a
  near-black glyph on the dark toolbar, i.e. an invisible button. It needs the
  palette, so it takes `self`.

New `_retint_painted_chrome()`, called at the end of `_apply_theme_impl`, drives
the three surfaces a sheet cannot reach — list rows, cursor readouts, focus icon
— each step independently guarded because it also runs from `__init__`, before
the later widgets exist (there is a test for exactly that partial-window case).

**§2.6 extended past its receipts, same finding class.** `_render_mixin.py` held
five more Material literals painting the *matplotlib* chrome: legend
`'white'`/`#E0E0E0` (which overrode the themed `legend.facecolor` that
`matplotlib_rc_overrides` had already installed — a white card on the dark plot),
the timing-unavailable message `#757575`, the stacked-pane stat titles `#444444`
×2, and the inner-pane divider spine `#BDBDBD`. `_palette` defines
`legend_face`, `legend_edge`, `stats_text` and `spine_separator` for precisely
these and had **zero consumers** for any of them; it does now.

**Tests.** `maker/tests/test_cosim_logger.py` stopped pinning literals — it now
asserts against `console_colors(False)` (no QApplication in that module, so the
logger resolves light) and gained `test_semantic_colors_track_the_theme`, which
is the property the old literals could not express. Its two original intents —
"no line is invisible", "meaningful lines keep a colour" — are unchanged.

**Not done in S3, still open:** §2.1, §2.2 (Python half), §2.3, §2.4, §2.5,
§2.9, all of P3, §C1's `theme_utils`/`recolor` half, §C3–C7. Also still open is
the a11y gap S2 flagged: a keyboard-focused Welcome tile has no visible ring.

---

### S4 — 2026-07-22 · 2.1, 2.3, 2.4, 2.5, 2.9, 2.2 (Python half), C7, C5

Verifier: `audit_harness/verify_ui_s4.py` (43/43 offscreen). S1/S2/S3 re-run on
this tree: 11/11, 16/16, 22/22. `smoke_no_qsci` and `smoke_no_watchdog`: PASS.
Full suite **581 passed / 9 failed / 22 skipped — byte-identical to the
pre-session baseline** (the 9 are the documented Windows env failures:
`test_toolchain_check` ×6, `test_cosim_config` vvp, `test_nghdl_embed` ×2).
Ruff `F,E9,B,E501,W291,W293` diffed against HEAD for all 14 touched files:
**zero new findings, 2 pre-existing cleared**.

**2.1 — the elevation scale went from one call site to all of them.** The
finding was adoption, so the work is mostly deletion: `Welcome._apply_tile_shadow`,
the tooltip card's shadow block, `apply_toolbar_depth`'s per-bar magic numbers,
`install_menu_depth`'s, and (same finding class, one file past its receipts)
`EditorWindow`'s find-bar shadow are all gone. `p21_no_consumer_hand_rolls_a_shadow`
asserts no consumer constructs a `QGraphicsDropShadowEffect` at all any more;
`p21_no_black_shadow_literals_remain` asserts none of them names black.

`motion.set_shadow` keeps its job as the low-level primitive but defaults its
colour from `tokens.theme(is_dark)["shadow_rgb"]`, which is the audit's own
one-move fix: every frame-by-frame caller inherits the tint without knowing it
exists. `apply_panel_depth` / `apply_popup_depth` became `elevate(w, "e2")` /
`elevate(w, "e4")` outright — the blur/y/alpha kwargs `VerilogVerifier` was
passing are deleted, not re-passed.

Three things fell out of doing it properly:
- **`elevate` bakes the colour in, so nothing ever re-tinted.** A toolbar
  elevated at startup kept its dark-theme shadow after a switch to light —
  which makes the whole light track (blue-grey at half the alpha) unreachable
  in the one direction users actually travel. New `elevation.retint(w)` reads
  the level back off a `_esim_elevation` property, and `theme_utils.
  _refresh_graphics_effects` — a walk over every widget of every window that
  already ran on each theme change — drives it. One walk, not two.
- **The toolbars keep their offsets and lose everything else.** `(0,5)` and
  `(4,6)` are load-bearing (the left rail's blur must not bleed up into the
  inverted-L joint), so `elevate` gained an `offset=` override that changes
  direction only. Blur, alpha and tint now come off `e3` for both bars.
- **The tooltip pad was too small for e3.** 34/2 + 10 = 27px of reach against
  a 16px transparent margin would clip the shadow square — the exact artefact
  that widget exists to avoid. `_PAD` is 28 and the verifier computes the
  requirement rather than hardcoding it. The card is also re-elevated in
  `show_text`, because that window is built once and reused all session.

**Welcome's tiles are elevated on first show, not in `__init__` — and that is
worth reading before someone "simplifies" it.** Elevating during construction
means asking an unshown, unpolished, unparented widget which theme it is in.
On this tree that palette read *on that path* tips a latent fault: the full
suite dies with an access violation inside `apply_theme`'s `setStyleSheet`
during `test_widget_event_loops[MainView]`. It is **not** this session's bug —
the identical crash reproduces on an otherwise-pristine HEAD with nothing added
but an inert `widget.palette().color(Window).lightness()` inside the old
`_apply_tile_shadow` (bisected in seven full-suite runs: the crash follows the
palette read, not the colour, the alpha, the blur or the dynamic property).
Deferring to `showEvent` is the correct design independently — at first show
the page is parented, polished and carries the theme it will be seen in — and
it puts the suite back on its exact baseline. **The latent fault itself is still
there and wants its own finding:** something in the configuration + frontEnd
test sequence leaves state that makes a MainView re-theme fragile; no single
configuration test file triggers it, only the set.

**2.2 (Python half) — the Welcome page has no colour literals left.** The hover
glow, the hover wash and the hero orb read `tokens` through a new
`widgets.accent_color(widget, alpha, key)`; the dead `GradientLabel` import is
gone. `p22_hover_wash_differs_between_themes` proves it by rendering: it grabs
the tile at hover 0 and hover 1 and asserts the second pixel is *this* theme's
accent composited over the first at 42/255 — so the check needs no assumption
about what the backdrop is, and a light-theme tile washed with dark-theme cyan
fails it. That was the actual defect: `#53D7FF` over white, in a UI whose light
accent is `#0077A8`.

**C5 — `HoverSurfaceMixin` moved into `widgets.py`.** The audit left this
conditional ("if a second card-like surface ever appears"), and §2.1 answered
it: the mixin now interpolates an *elevation level* into an *accent token*, so
it is design-system machinery that happens to have one consumer, not page code
that happens to be generic. It also stops re-inventing the resting shadow —
the old copy faded from black-alpha-48 upward, a value that existed nowhere
else and was invisible in light mode; it now starts from exactly what
`elevate(w, REST_LEVEL)` painted. `widgets.py` gains its first live class,
which is where §3.2 wants that file to end up.

**2.3 — FlowNavigator's `_pill_tokens` reads the tokens it claimed to.** Same
method as §2.4 below; the dict shape is untouched, so `_apply_pill_theme` did
not change. Four mappings are deliberately asymmetric and each carries its
reason in place: `bar_bg` is `bg_raise` on dark but `bg` on light (S1's note —
light chrome strips sit flat on the window, and "fixing" that shifts the whole
header); hover moves one step *away* from the strip in the direction of
contrast, so dark lifts to `surface_2` and light sinks to `bg_sunken`, which is
what the light sheet's own hover rules already paint; the checked tint needs
0.18 alpha on dark and 0.13 on light; and `reload_fg` is `warning` on dark but
amber-800 `#92400E` on light, because `#D97706` on its own 10% wash is 3.2:1.
That amber is the one value in the file that is not a token, it is measured
(5.9:1) and pinned, and it is the same call `STYLE_LIGHT`'s InfoBar made in S2.
Three of the four dark values it used to copy were phantoms anyway
(`#9FB1CC`/`#F4F8FF` appear in neither sheet — the drift class §C2 cleared).

While in the file: `_placeholder`'s error HTML hardcoded `#8a939b` (3.0:1 on
the light page, in neither theme's palette) — same finding class, now
`text_muted`. Inline HTML cannot inherit a QSS colour, so it has to name one;
it does not have to invent one.

**2.4 — `_about_palette` derives all 14 values.** Shape and consumers
unchanged (`p24_about_keys_are_unchanged` pins the key set, since
`PreferencesDialog._build_about_page` indexes the same dict). `sep` and
`chip_border` are built from the theme's `text` rgb, `pill_bg` from its
`accent` rgb, and the verifier checks the *components*, not the strings. Two
asymmetries kept and documented: the logo chip drops to `bg_raise` on dark
(a well for the bronze coin) where light has nowhere below the card to go, and
`link`/`pill_fg` take `accent_hi` on dark but `accent` on light — the tone that
steps away from the page. On white, `accent_hi` is 3.0:1; `accent` is 4.6:1,
and it is also the pixel this dialog already shipped. Also caught here: an
inline `font-weight: 650`, left behind by S2's sweep because that sweep only
covered the .qss files.

**2.5 — the fullscreen toggle uses eSim's own SVGs.** `SP_TitleBarMaxButton` /
`SP_TitleBarNormalButton` resolve to the platform's title-bar glyphs — a
Windows chrome square on one OS, an icon-theme arrow on another — which is the
divergence `icon_paths` was created to end, and `fullscreen_icon` /
`dock_back_icon` were sitting there unused for exactly this. The icons bake the
theme's foreground into the raster, so the button re-renders on `PaletteChange`.

**That re-render must be deferred, and the first version of it was a hard
crash.** `PaletteChange` is delivered from inside the polish that
`setStyleSheet`/`setPalette` is running; rasterising an SVG and calling
`setIcon` there re-enters that polish, which re-delivers `PaletteChange` — with
one of these toggles in every docked panel (KicadToNgspice, the Makerchip flow
strip, the plot toolbar) it is unbounded recursion, i.e. a C-stack overflow.
The refresh now runs on the next tick behind a pending flag, which also
coalesces the burst a single toggle produces into one re-render.
`p25_retint_never_runs_inside_the_handler` asserts both halves: nothing renders
synchronously, and five queued events produce exactly one re-render.

**2.9 — one mono resolver, and it actually reaches the widget.** The audit's
premise turned out to be half right, and the measurement changed the fix:
- `VerilogVerifier`'s `QFont("Segoe UI", 10)` and `QFont("Consolas", 11)` were
  **dead**, not live. With an app-level sheet installed, the SHEET wins: both
  `QLabel#verilogSidebarTitle` and `QTextEdit#verilogConsole` resolve to the
  sheet's values with the `setFont` in place, verified by rendering
  (`p29_sheet_owns_the_verifier_console_font`). They are deleted as the
  misleading dead weight they were — a reader who saw `QFont("Consolas")` had
  every reason to think the console was Consolas on Windows only.
- The *live* half is the opposite defect: a `QPlainTextEdit` that is **not**
  styled by an objectName rule cannot hold a mono face through `setFont` at
  all, because the app sheet's `QWidget` font rule beats it. So the toolchain
  doctor's column-aligned report and the QScintilla-less fallback editor were
  both rendering in **Inter**. Both now set a widget-level sheet, the only
  thing that outranks the app sheet — new `theme.mono_font_css()`, with the
  measurement written down where the next reader will need it.
- Resolution is one chain: `codeEditor.theme.editor_font` delegates to
  `frontEnd.widgets.mono_family()` (its local `_FONT_PREFS` stays as the
  fallback for a `codeEditor` imported without `frontEnd`), so the QScintilla
  editor, the plain editor, the doctor and the QSS consoles land on the same
  face. That adopts `mono_family` per §3.2/§C6 instead of deleting it, and it
  head-aligns the resolver with what the sheets DECLARE (`JetBrains Mono`,
  `Cascadia Mono`, `Consolas`). Consequence worth recording: the code editor's
  first preference moves from Cascadia *Code* to Cascadia *Mono*, i.e. no
  ligatures — the declared stack wins over an undeclared local preference.
  `mono_family` also stopped caching its failure: before a QGuiApplication
  exists `families()` is empty, and caching that pinned every later caller to
  the generic.

**C7 — the About dialog tracks its content.** `setFixedSize(440, 500)` →
`setMinimumSize` + `adjustSize()` after the content exists. `build_qss` scales
every px metric by the zoom preference, so at 150–200% the type grew inside a
frame that could not: the credits clipped. `pc7_about_grows_with_the_zoom_level`
builds the real dialog at 100% and 200% (patching `exec`, not reimplementing
the dialog) and asserts the frame is at least its own size hint. The rest of
C7 is deliberately not done, per the finding: text-driven surfaces follow font
metrics already, and full zoom propagation into local sheets is not worth the
complexity.

**Not done in S4, still open:** all of P3, §C1's `theme_utils`/`recolor` half,
§C3, §C4, §C6's bundling question, and the remainder of §C7. Two findings this
session *created* rather than closed, both recorded above: the latent
MainView/`apply_theme` access violation (pre-existing, now reproducible on
demand) and — still, from S2 — the missing focus ring on a keyboard-focused
Welcome tile.

---

### S5 — 2026-07-22 · 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, C3 — **P3 is closed**

Verifier: `audit_harness/verify_ui_s5.py` (28/28 offscreen). S1/S2/S3/S4 re-run
on this tree: 11/11, 16/16, 22/22, 43/43. `smoke_no_qsci` and
`smoke_no_watchdog`: PASS. Ruff `F,E9,B,E501,W291,W293` diffed against HEAD for
all 8 touched source files: **zero new findings, 6 pre-existing cleared** (all
of them E501s inside deleted code); the new verifier is clean on the same set.

Full suite **579 passed / 9 failed / 18 skipped**. The 9 failures are the
documented Windows env set, unchanged (`test_toolchain_check` ×6,
`test_cosim_config` vvp, `test_nghdl_embed` ×2). The counts moved from the
581/9/22 baseline by exactly 6, and every one is accounted for: `test_widget_
event_loops` discovers widget classes by introspection, so the six deleted
classes stop being parametrised — `GradientLabel` and `AuroraHeroFrame` were
no-arg constructible (−2 passed) and `DockTitleBar`/`DockDropOverlay`/
`FloatingDockHost`/`RailDragGrip` took constructor arguments and were being
skipped (−4 skipped). No test lost coverage of anything that still exists.

**Deletion sessions verify vacuously unless you close the loop.** Every
absence the audit asked for is trivially assertable, and trivially wrong the
next time someone adds a rule. So three of this session's checks are sweeps
over the whole tree rather than greps for the names §3.1's table happens to
list: every `#name` and `[prop="value"]` selector in either sheet must have a
setter in some `.py`/`.ui`; every image file must be referenced **and** every
referenced image must exist; every public name in `widgets.py` and every
`*_icon` factory must have a consumer. A future dead rule now fails on the day
it is written.

That paid for itself immediately: the selector sweep found a **40th finding the
audit's table missed** — `QLabel#verilogNoWaveform` (dark `:1253`, light
`:1262`), an objectName rule with zero setters, in both sheets. §1.6 had even
looked straight at it ("italic exists in both, fine") without asking whether
anything wore it. Deleted with the rest, symmetrically.

**3.1 — nine dead selector groups out of both sheets, plus that one.** Removed
in mirrored ranges, so S2's empty-structural-diff guard stays absolute and is
re-asserted here on the trimmed sheets. Both sheets still re-parse through Qt
with no "Could not parse stylesheet". The one judgement call the audit left
open — `QLabel[cssClass="error"/"warning"/"success"]`, "delete OR adopt in
§1.3/§2.8, decide once" — is **delete**, because the decision was already made
by the shipped work: §1.3's status dot has to *re-tint* on a theme toggle,
which is why S3 gave it `tokens` + a replay hook rather than a static class,
and §2.8 went to `messageKind`. Nothing was ever going to set them.

What deliberately survived the cut, since the deletion ran straight through
that region: `QFrame#dockCard` (live — `DockArea.apply_fullscreen_feature`
sets it, and the verifier asserts both halves), `QDockWidget::title` and the
0×0 close/float hiding block S1 collapsed, and `QWidget#verilogFindBar`.
`dock_close_*` stay live via `QTabBar::close-button`.

**3.2 — `widgets.py` is all-live for the first time.** Seven names gone
(`GradientLabel`, `AuroraHeroFrame`, `RailDragGrip`, `DockTitleBar`,
`DockDropOverlay`, `FloatingDockHost`, `_is_wayland`), ~580 lines, leaving
`mono_family` / `accent_color` / `HoverSurfaceMixin` — each with a real
consumer, each still exercised by rendering rather than by grep. The module
docstring now states the rule the file is finally keeping.

**The dock-drag code went; its findings did not.** The audit offered "move the
file to docs/ or a git tag if deleting feels risky given the engineering notes
in its docstrings" — but a file in `docs/` is not where the next person to
attempt drag-out docking will be standing. They will be in
`DockArea.apply_fullscreen_feature`, which is where the four findings now live,
with a pointer to `git log -- src/frontEnd/widgets.py` for the implementation:
reparent-then-move races the compositor's async window-map on Wayland (so
undock must be DnD, which never reparents mid-gesture); `startSystemMove()` is
the only top-level move Wayland honours; `setFloating` inside a mouse/drag
handler re-enters Qt's drag state machine and freezes the window; and Wayland
delivers no events during a compositor move, so hover-to-redock is undetectable
there and double-click is the only universal gesture. The verifier fails if
that write-up is removed.

**3.3 — three dead installers and every `dockPopButton` branch.**
`install_popup_motion` / `install_effect_refresh` / `install_menu_depth` are
gone, and the property is now absent from executable code tree-wide (it was
read in `rest_alpha` and in four `if not is_dock:` guards wrapping every branch
of `TactileButtonFilter.eventFilter`; nothing has set it since the dock
prototype).

**One deviation from the finding's letter, and the reason.** The audit said
"keep the class, delete only `install_effect_refresh`" — but with the installer
gone, `EffectShowRefreshFilter.eventFilter` cannot fire from anywhere (the only
consumer, `test_app_motion_filter`, exercises `AppWideMotionFilter`), so
keeping it would have left exactly the dead weight this section exists to
remove. The **class** is kept, as instructed, because it is the home of
`_revalidate` and of the only written explanation of why a re-shown widget's
shadow cache goes stale; its uninstallable `eventFilter` is not. The docstring
now says plainly that this is a helper, not a filter.

Removing four guards from a hot event path is a behaviour change, so it is
verified as one, not read: a `dockPopButton` button now rests at the same alpha
as any other (the property is inert), `noMotion` — the live opt-out — still
returns 0, and an Enter event still builds the drop shadow and starts the glow
animation.

**3.4 — four icon factories and their SVG constants.** `backup_icon`,
`close_proj_icon`, `copy_icon`, `close_icon`. `_theme_icon_color`'s `"danger"`
role stays: `trash_icon` still uses it. Every surviving factory is asserted to
rasterise to a real pixmap, which is how a deleted constant that some other
factory shared would surface — as an empty icon rather than an ImportError.

**3.5 — 16 orphaned assets, not the 6 the audit counted.** `text_find_*`,
`text_save_*`, `text_save_as_*`, `text_wrap_*` are 8 files, not 6 (each name
ships a dark **and** a light variant); `dock_pop_*` is 4; and `dock_fullscreen_*`
is 4 more, orphaned by S1's own §1.5 collapse and noted there. Twelve SVGs
remain in `images/`, and all twelve are referenced — asserted in both
directions now, which is what stops this from silently re-accumulating.

**3.6 — the three hygiene items.** The commented-out SoC toolbutton (with its
own commented stylesheet, and a `connect` to a `showSoCRelease` that does not
exist) is gone. `PreferencesDialog`'s `import json as _json` + `json_load` /
`json_dump` wrappers — which sat *below* the class that called them — are a
plain top-level `import json` and direct calls. `reject()`'s second write was
the finding's real point and it is now `paths.write_json_atomic`: it is the
Cancel path, i.e. it runs precisely while the user is closing things down, and
a bare `open(w)` there leaves a truncated preferences.json. The verifier walks
the module's AST and fails on *any* `open(..., "w")`, not just that one.
`theme_utils`' comment stopped naming SpiceEditor; it names the handlers that
actually re-style on `PaletteChange` today, and the verifier checks each named
class exists and really has a `changeEvent`, so the comment cannot rot again.

**C3 — kept and documented, per the audit's recommendation.** The custom-accent
machinery is reachable only from a hand-edited or older-build preferences.json,
because `_collect_prefs` pins all three keys to sentinels. `tokens.py`'s
docstring now says so, `theme_utils.ACCENT_TOKENS` carries a pointer to it, and
— the part that makes it stay true — a check asserts the **docstring and the
code agree**: it parses `_collect_prefs` and fails if the three keys stop being
pinned, so re-enabling accent picking breaks the test that guards the note
rather than quietly making the note a lie. A third check builds a sheet with a
custom accent and requires it to still reach both the hex literals and the
`rgba()` glows, since "kept" has to mean "works".

**A pre-existing test-isolation bug, surfaced but not caused.**
`configuration/tests/test_dialogs.py`'s two `_resolve_parent` tests assume
exactly one visible `QMainWindow` exists, which nothing in the suite
guarantees; they fail if a leaked window is still alive when they run. Deleting
six widget classes removes six `_exercise()` cycles (each of which drains the
`deleteLater` queue), which changes whether the leak has been collected by
then. **This reproduces on a pristine HEAD** by merely deselecting those six
params — no source change at all — and it does *not* fire in the full suite in
either tree. It wants its own finding: the fixture should assert or enforce a
clean top-level set, rather than depending on how many widgets ran before it.

**Not done in S5, still open:** §C1's `theme_utils`/`recolor` half, §C4 (which
the audit explicitly wants done opportunistically, never as a sweep), §C6's
bundling question (blocked on commit `110228cc`), and the remainder of §C7
(deliberately out of scope per the finding). Also still open, both created by
earlier sessions rather than by the audit: the latent MainView/`apply_theme`
access violation S4 made reproducible, and the missing focus ring on a
keyboard-focused Welcome tile from S2. **P1, P2 and P3 are now fully closed.**

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
