; =============================================================================
; eSim Windows installer (Inno Setup 6).
;
; Compiled by build-windows.ps1, which passes:
;   /DAppVersion=<VERSION>   /DStageDir=<staged tree>   /DOutDir=<dist>
;
; Design notes (see PACKAGING.md for the full rationale):
;  * Install root defaults to C:\FOSSEE\eSim -- deliberately SPACE-FREE:
;    the optional MSYS2/mingw toolchain (NgVeri model builds) and ngspice
;    code-model paths break subtly under paths with spaces.
;  * Everything per-user (~/.esim, ~/.nghdl, KiCad sym-lib-table) is done by
;    windows\windows_bootstrap.py on every launch, NOT here -- so multi-user
;    machines and upgrades self-heal, and the logic is unit-tested.
;  * KiCad IS bundled, at tools\kicad: the official installer's payload,
;    pruned for eSim by build-windows.ps1's Stage-Kicad (no 3D models/demos/
;    translations). Private to eSim: esim.bat prepends tools\kicad\bin to
;    PATH; no registry entries, file associations or global env vars, so a
;    system-wide KiCad install coexists untouched. One exe = the whole tool.
; =============================================================================

#ifndef AppVersion
  #define AppVersion "0.0"
#endif
#ifndef StageDir
  #define StageDir "build\eSim"
#endif
#ifndef OutDir
  #define OutDir "dist"
#endif

[Setup]
AppId={{7E63A0F2-2C6B-4D3B-9E3E-ESIM0000FOSS}
AppName=eSim
AppVersion={#AppVersion}
AppPublisher=FOSSEE, IIT Bombay
AppPublisherURL=https://esim.fossee.in/
DefaultDirName={sd}\FOSSEE\eSim
DisableProgramGroupPage=yes
UninstallFilesDir={app}
UninstallDisplayIcon={app}\eSim.exe
OutputDir={#OutDir}
OutputBaseFilename=eSim-{#AppVersion}-installer
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
WizardStyle=modern

; License agreement page. eSim is GPLv3; the repo's root LICENSE is the same
; text shipped in the install tree. Pointing LicenseFile at it makes Inno add
; the standard "License Agreement" page (accept/decline) before folder select,
; matching the old NSIS installer's first step. Path is relative to this .iss.
; Inno reads it as plain text (no .txt extension needed; RTF is auto-detected).
LicenseFile=..\LICENSE

; Branding. One identity end to end: the setup exe carries the SAME icon as
; eSim.exe, the Start-menu/desktop shortcuts and the uninstall entry, so the
; downloaded file is recognisable in Explorer and the download bar before it
; is ever run. The wizard panels (eSim mark + FOSSEE logo) are generated from
; the repo's own logo assets by branding\make-wizard-images.py -- one file per
; DPI scaling step (100/125/150/200%); Inno picks the closest match, which is
; why they are listed rather than wildcarded (a wildcard would also sweep the
; small images into the big-panel slot).
SetupIconFile=..\images\esim.ico
WizardImageFile=branding\wizard-164x314.bmp,branding\wizard-205x393.bmp,branding\wizard-246x471.bmp,branding\wizard-328x628.bmp
WizardSmallImageFile=branding\wizard-small-55x58.bmp,branding\wizard-small-69x73.bmp,branding\wizard-small-83x87.bmp,branding\wizard-small-110x116.bmp

[Types]
Name: "full";    Description: "Full (HDL toolchain: NgVeri, d_cosim, NGHDL/GHDL VHDL co-sim)"
Name: "compact"; Description: "Compact (analog simulation only; no HDL model builds)"
Name: "custom";  Description: "Custom"; Flags: iscustom

[Components]
Name: "core"; Description: "eSim application, ngspice, Icarus Verilog"; \
    Types: full compact custom; Flags: fixed
Name: "hdl";  Description: "HDL toolchain: MSYS2 mingw gcc/make/verilator/GHDL (NgVeri code-model builds + NGHDL VHDL co-simulation)"; \
    Types: full

[Dirs]
; eSim's HDL model builds WRITE inside the install tree by design -- exactly
; like the Ubuntu install owns $HOME/nghdl-simulator. Setup runs as admin but
; eSim runs as the user, so the written dirs carry a users-modify ACE, which
; NTFS inheritance then extends to everything [Files] installs into them
; (Inno processes [Dirs] before [Files]).
;
; The grant used to sit on {app} itself. That also handed every local user
; write access to python\, eSim.exe, tools\kicad\bin and tools\msys64\ -- i.e.
; the ability to swap a binary the NEXT user of the machine runs. On the
; shared lab machines eSim targets that is a local tamper vector, so the ACE
; is scoped to the two trees the running app genuinely writes:
;
;   tools\nghdl            new model sources land in src\xspice\icm, the .cm
;                          rebuild runs in release\, `make install` writes
;                          install_dir\, and windows_bootstrap.fix_spinit
;                          rewrites install_dir\share\ngspice\scripts\spinit
;                          on first launch (it ships with build-machine paths
;                          baked into its codemodel lines).
;   library\modelParamXML  createkicad / createkicadCosim / NgVeri /
;                          model_teardown add and remove <model>.xml under
;                          Ngveri, NgVeriCosim and Nghdl.
;
; Read-only at runtime -- verified against the source, deliberately NOT
; granted (if a future change starts writing to one of these, widen the list
; here rather than moving the ACE back up to {app}):
;   library\kicadLibrary   static symbol libs are referenced in place; the
;                          libs eSim REWRITES moved to ~\.esim\kicad_symbols
;                          (kicad_symlib.generated_symlib_path) long ago, and
;                          the install path survives only as a legacy READ
;                          probe for migrating older installs.
;   library\bin\iverilog   ensure_unversioned_libvvp() copies libvvp.dll only
;                          when the build did not -- and Stage-SimToolchain
;                          stages it (the Bleyer fallback has no libvvp to
;                          copy at all, so the write never happens either way).
;   tools\msys64           used purely as a toolchain: mingw32-make, gcc,
;                          verilator and ghdl are invoked by full path with
;                          cwd= the model dir, and the generated cfunc's
;                          scratch file goes to %LOCALAPPDATA%\Temp.
;   src, windows, python   bytecode is precompiled by [Run] below, as admin.
;
; KNOWN RESIDUAL -- this scoping removes eSim's OWN over-broad grant, and on
; its own it does NOT finish the job at the default install root. Windows
; ships C:\ with `NT AUTHORITY\Authenticated Users:(OI)(CI)(IO)(M)`, an
; inherit-only Modify that propagates into every folder created under it, so a
; tree at {sd}\FOSSEE\eSim inherits user-writable ACLs from the drive root no
; matter what this section says. Verified on a real install:
;   C:\FOSSEE\eSim\python -> BUILTIN\Users:(I)(OI)(CI)(M)      [this section]
;                            NT AUTHORITY\Authenticated Users:(I)(M)  [from C:\]
; Closing it needs the install root's INHERITANCE broken as well, e.g. a [Run]
; step after the files land:
;   icacls "{app}" /inheritance:r ^
;     /grant *S-1-5-32-544:(OI)(CI)F /grant *S-1-5-18:(OI)(CI)F ^
;     /grant *S-1-5-32-545:(OI)(CI)RX
; (well-known SIDs, not names -- localized Windows renames these groups). That
; is a bigger behavioural change than a permissions narrowing: it must be
; validated by a Full install followed by an NgVeri + d_cosim + NGHDL model
; build from a SECOND, standard-user account, since getting it wrong leaves a
; tree the user can neither build in nor delete. Deliberately left for that
; test rather than shipped unverified.
Name: "{app}\tools\nghdl";           Permissions: users-modify
Name: "{app}\library\modelParamXML"; Permissions: users-modify

[Files]
; Core = everything except the HDL build toolchain. NOTE tools\nghdl\
; install_dir (the custom eSim ngspice runtime: d_cosim + ivlng + ghdl.cm,
; plus ngspice_gui.exe, the wingui twin that draws the interactive plots)
; IS core -- every flavour simulates with it; only the trees used to BUILD
; new HDL models (MSYS2, ngspice sources, configured build tree) are the
; optional 'hdl' component. release_gui is a throwaway build dir; the build
; script deletes it, and this excludes an interrupted build's leftovers.
; tools\nghdl.old: a hand-made backup of an earlier simulator tree that
; survives in the stage across incremental builds (nothing in the pipeline
; creates or purges it) and rode into the last release as 127 MB of dead
; weight. Belt and braces -- the build script now deletes it as well.
;
; Ship-size excludes (the stage keeps all of this -- source rebuilds there
; need the full toolchain; only the SHIPPED tree is trimmed):
;  * tools\nghdl\{examples,tests,autom4te.cache,visualc,man}: ngspice source
;    tarball baggage; runtime model builds touch only src\ headers, release\
;    and install_dir\.
;  * tools\ngspice\{examples,docs}: official-zip baggage, ~26 MB.
;  * pip/pytest and friends: installed into the stage python for build-time
;    testing; the app imports none of them. NOTE the installed tree therefore
;    has no pip/pytest -- run the suite against an install with the STAGE
;    python (same version, same wheel set).
;  * scipy: nothing in the tree imports it (also gone from
;    requirements-windows.txt; this catches stages provisioned before that).
;  * developer/repo files a user install has no use for (leading backslash =
;    rooted at {app}, so e.g. \scripts does NOT touch python\Scripts, and
;    \docs does not shadow anything deeper):
;      - build/porting notes + packaging docs (WINDOWS-*.md, PACKAGING.md,
;        MAINTAINERS-PACKAGING.md), contributor/repo docs (CONTRIBUTION.md,
;        SECURITY.md, INSTALL -- Ubuntu instructions), Sphinx sources
;        (\code, \conf.py, \index.rst), Linux-only trees (\ihp, \patches,
;        \scripts, \docs, make-release.sh), and pip metadata for the bundled
;        interpreter (setup.py, requirements.txt, python-wheels.lock).
;      - unrooted `tests`: every dir named tests outside msys64 is dev-only
;        (src\*\tests, site-packages numpy/matplotlib/colorama tests, kicad
;        stdlib tests) -- verified by find before adding. `test` likewise
;        (kicad stdlib ctypes/tkinter/unittest test dirs). msys64 and the
;        nghdl src/release trees have their OWN [Files] entries, so these
;        unrooted patterns cannot reach the toolchain.
;    windows\windows_bootstrap.py STAYS: launcher_windows imports it on
;    every launch. README.md, LICENSE, VERSION, RELEASE stay (user-facing).
Source: "{#StageDir}\*"; DestDir: "{app}"; Components: core; \
    Excludes: "tools\msys64\*,tools\nghdl\src\*,tools\nghdl\release\*,tools\nghdl\release_gui\*,tools\nghdl.old\*,tools\nghdl\examples,tools\nghdl\tests,tools\nghdl\autom4te.cache,tools\nghdl\visualc,tools\nghdl\man,tools\ngspice\examples,tools\ngspice\docs,python\Lib\site-packages\pip,python\Lib\site-packages\pip-*,python\Lib\site-packages\pytest,python\Lib\site-packages\_pytest,python\Lib\site-packages\pytest-*,python\Lib\site-packages\pytest_timeout*,python\Lib\site-packages\iniconfig*,python\Lib\site-packages\pluggy*,python\Lib\site-packages\scipy,python\Lib\site-packages\scipy.libs,python\Lib\site-packages\scipy-*,python\Scripts\pip.exe,python\Scripts\pip3.exe,python\Scripts\pip3.12.exe,python\Scripts\pytest.exe,python\Scripts\py.test.exe,python\Scripts\f2py.exe,python\Scripts\numpy-config.exe,\PACKAGING.md,\MAINTAINERS-PACKAGING.md,\CONTRIBUTION.md,\SECURITY.md,\INSTALL,\make-release.sh,\setup.py,\conf.py,\index.rst,\requirements.txt,\python-wheels.lock,\code,\docs,\ihp,\patches,\scripts,\src\conftest.py,tests,test"; \
    Flags: recursesubdirs createallsubdirs ignoreversion
; MSYS2 ship-size excludes, verified against what runtime model builds use:
;  * pacman download cache (var\cache) and locale/man/doc/info trees.
;  * gdb (debugger; nothing in a user install debugs). mingw64's python3.14
;    deliberately STAYS despite arriving as gdb's scripting dep: verilator's
;    verilated.mk hardcodes `PYTHON3 = python3` and every NgVeri model make
;    runs $(PYTHON3) verilator_includer -- pruning it broke NgVeri with
;    "Error 127" (caught by the counter e2e on the pruned test install).
;  * gnat*.exe + adainclude: the Ada COMPILER that built ghdl. ghdl.exe is
;    statically linked (imports only system DLLs) and never compiles Ada at
;    runtime. adalib STAYS: lib\ghdl\grt.lst passes -L...\adalib\ at VHDL
;    elaboration, so NGHDL links against it on user machines.
;  * autotools (autoconf x6, automake x8, aclocal): configure ran at package
;    time; runtime model builds only ever run make. perl STAYS -- the
;    verilator driver is a perl script.
;  * verilator debug twins (only `verilator --debug` loads them).
Source: "{#StageDir}\tools\msys64\*"; DestDir: "{app}\tools\msys64"; Components: hdl; \
    Excludes: "var\cache,usr\share\locale,usr\share\man,usr\share\doc,usr\share\info,usr\share\bash-completion,mingw64\share\man,mingw64\share\doc,mingw64\share\info,mingw64\share\locale,mingw64\share\gtk-doc,mingw64\share\gdb,gdb.exe,gdbserver.exe,gdb-add-index,gnat*.exe,adainclude,verilator_bin_dbg.exe,verilator_coverage_bin_dbg.exe,usr\bin\autoconf*,usr\bin\autoheader*,usr\bin\autom4te*,usr\bin\automake*,usr\bin\autoreconf*,usr\bin\autoscan*,usr\bin\autoupdate*,usr\bin\autopoint,usr\bin\aclocal*,usr\bin\ifnames,usr\share\autoconf*,usr\share\automake*,usr\share\aclocal*"; \
    Flags: recursesubdirs createallsubdirs ignoreversion skipifsourcedoesntexist
Source: "{#StageDir}\tools\nghdl\src\*"; DestDir: "{app}\tools\nghdl\src"; Components: hdl; \
    Flags: recursesubdirs createallsubdirs ignoreversion skipifsourcedoesntexist
Source: "{#StageDir}\tools\nghdl\release\*"; DestDir: "{app}\tools\nghdl\release"; Components: hdl; \
    Flags: recursesubdirs createallsubdirs ignoreversion skipifsourcedoesntexist

[Icons]
; eSim.exe is a real GUI executable (windows\launcher) with the icon and
; version info embedded: proper identity in Windows search and the taskbar.
; By default it spawns eSim under python.exe in its own log console (the
; stdout/stderr stream, Linux-like); pass --no-console for the silent pythonw
; launch. esim.bat remains in {app} for terminal use (--no-console, --doctor).
; AppUserModelID matches SetCurrentProcessExplicitAppUserModelID in
; Application.py: the running python windows group under this shortcut, so
; the taskbar shows the eSim icon (and pinning works) instead of a blank
; python entry.
Name: "{autoprograms}\eSim"; Filename: "{app}\eSim.exe"; WorkingDir: "{app}"; AppUserModelID: "FOSSEE.eSim.2.5"
Name: "{autodesktop}\eSim";  Filename: "{app}\eSim.exe"; WorkingDir: "{app}"; AppUserModelID: "FOSSEE.eSim.2.5"

[Run]
; Precompile bytecode at install time so the first cold launch doesn't compile
; every .py on import (and Defender doesn't scan each freshly written .pyc).
Filename: "{app}\python\python.exe"; \
    Parameters: "-m compileall -q -j 0 ""{app}\src"" ""{app}\windows"""; \
    WorkingDir: "{app}"; StatusMsg: "Precompiling eSim modules..."; \
    Flags: runhidden
; NOTE: no Defender exclusion here, deliberately. An installer that whitelists
; its own directory from the system antivirus is indistinguishable from
; malware behaviour and erodes user trust, whatever the cold-start win.
; Cold-start cost is addressed the legitimate ways instead: precompiled
; bytecode (above) and background import prewarm in the app.
; runasoriginaluser: the installer is elevated but eSim must run as the real
; user -- an elevated first run would write root-owned files into ~/.esim and
; the workspace that later non-elevated launches cannot touch.
Filename: "{app}\eSim.exe"; Description: "Launch eSim"; \
    Flags: postinstall nowait skipifsilent runasoriginaluser

[UninstallRun]
; Clean up the exclusion earlier installer versions added (current installer
; adds none). Harmless no-op when absent.
Filename: "powershell.exe"; RunOnceId: "RemoveEsimDefenderExclusion"; \
    Parameters: "-NoProfile -ExecutionPolicy Bypass -Command ""Remove-MpPreference -ExclusionPath '{app}' -ErrorAction SilentlyContinue"""; \
    Flags: runhidden

[UninstallDelete]
; Compiled artifacts the app writes inside its own tree at runtime.
Type: filesandordirs; Name: "{app}\library\modelParamXML\Ngveri"
Type: filesandordirs; Name: "{app}\library\modelParamXML\NgVeriCosim"
Type: filesandordirs; Name: "{app}\library\modelParamXML\Nghdl"
; The nghdl tree accumulates user-built model sources and build objects
; (src\xspice\icm\{Ngveri,ghdl}\<model>, release tree .o) on top of the
; installed files; sweep the whole thing so an uninstall is clean.
Type: filesandordirs; Name: "{app}\tools\nghdl"
; Bundled KiCad writes caches inside its own tree at runtime (fp-info-cache,
; regenerated .pyc under bin\Lib); sweep it too.
Type: filesandordirs; Name: "{app}\tools\kicad"
; NOTE: per-user state (~/.esim, ~/.nghdl) is deliberately NOT deleted here:
; the uninstaller runs as one user but state exists per user; and models the
; user built (kicad_symbols) may be wanted across reinstalls.

; No [Code] KiCad detection anymore: KiCad ships inside this installer at
; tools\kicad (see the design notes above), so there is nothing to detect,
; offer or download after this exe finishes.
