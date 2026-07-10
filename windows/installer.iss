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
UninstallDisplayIcon={app}\images\workspace.ico
OutputDir={#OutDir}
OutputBaseFilename=eSim-{#AppVersion}-installer
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
WizardStyle=modern

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
; eSim's HDL model builds WRITE inside the install tree by design (new model
; sources land in tools\nghdl\src\xspice\icm, the .cm rebuild runs in
; tools\nghdl\release, XML metadata in library\modelParamXML) -- exactly like
; the Ubuntu install owns $HOME/nghdl-simulator. The install runs as admin
; but eSim runs as the user, so grant users modify rights on the tree.
Name: "{app}"; Permissions: users-modify

[Files]
; Core = everything except the HDL build toolchain. NOTE tools\nghdl\
; install_dir (the custom eSim ngspice runtime: d_cosim + ivlng + ghdl.cm,
; plus ngspice_gui.exe, the wingui twin that draws the interactive plots)
; IS core -- every flavour simulates with it; only the trees used to BUILD
; new HDL models (MSYS2, ngspice sources, configured build tree) are the
; optional 'hdl' component. release_gui is a throwaway build dir; the build
; script deletes it, and this excludes an interrupted build's leftovers.
Source: "{#StageDir}\*"; DestDir: "{app}"; Components: core; \
    Excludes: "tools\msys64\*,tools\nghdl\src\*,tools\nghdl\release\*,tools\nghdl\release_gui\*"; \
    Flags: recursesubdirs createallsubdirs ignoreversion
Source: "{#StageDir}\tools\msys64\*"; DestDir: "{app}\tools\msys64"; Components: hdl; \
    Flags: recursesubdirs createallsubdirs ignoreversion skipifsourcedoesntexist
Source: "{#StageDir}\tools\nghdl\src\*"; DestDir: "{app}\tools\nghdl\src"; Components: hdl; \
    Flags: recursesubdirs createallsubdirs ignoreversion skipifsourcedoesntexist
Source: "{#StageDir}\tools\nghdl\release\*"; DestDir: "{app}\tools\nghdl\release"; Components: hdl; \
    Flags: recursesubdirs createallsubdirs ignoreversion skipifsourcedoesntexist

[Icons]
; runminimized: esim.bat runs in a cmd console; minimized keeps it as a
; taskbar blip instead of flashing a black console window at every launch.
Name: "{autoprograms}\eSim"; Filename: "{app}\esim.bat"; \
    IconFilename: "{app}\images\workspace.ico"; WorkingDir: "{app}"; \
    Flags: runminimized
Name: "{autodesktop}\eSim";  Filename: "{app}\esim.bat"; \
    IconFilename: "{app}\images\workspace.ico"; WorkingDir: "{app}"; \
    Flags: runminimized

[Run]
Filename: "{app}\esim.bat"; Description: "Launch eSim"; \
    Flags: postinstall nowait skipifsilent

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
