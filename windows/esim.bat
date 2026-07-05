@echo off
rem =========================================================================
rem eSim Windows launcher. Installed to <install>\esim.bat by installer.iss;
rem the Start-menu / desktop shortcuts point here.
rem
rem %~dp0 = the install root (this file lives at its top level), so the
rem install is fully relocatable and nothing here needs registry lookups.
rem =========================================================================
setlocal
set "ESIM_HOME=%~dp0"
rem strip trailing backslash
if "%ESIM_HOME:~-1%"=="\" set "ESIM_HOME=%ESIM_HOME:~0,-1%"

rem Bundled tools first on PATH: the custom eSim ngspice (d_cosim/ivlng/
rem ghdl.cm) wins over the official fallback copy; KiCad CLI tools if the
rem official installer put them in the default location (no-op otherwise).
set "PATH=%ESIM_HOME%\tools\ngspice\bin;%PATH%"
if exist "%ESIM_HOME%\tools\nghdl\install_dir\bin\ngspice.exe" (
    set "PATH=%ESIM_HOME%\tools\nghdl\install_dir\bin;%PATH%"
    rem Where this ngspice finds spinit (and through it the .cm code models,
    rem whose paths windows_bootstrap.py rewrites for this install).
    set "SPICE_LIB_DIR=%ESIM_HOME%\tools\nghdl\install_dir\share\ngspice"
)
for /d %%K in ("%ProgramFiles%\KiCad\*") do if exist "%%K\bin\eeschema.exe" set "PATH=%%K\bin;%PATH%"

rem Per-user setup (config.ini, symbol seeding, sym-lib-table registration).
rem Idempotent + best-effort: a failure must not stop the GUI from starting.
"%ESIM_HOME%\python\python.exe" "%ESIM_HOME%\windows\windows_bootstrap.py" --esim-root "%ESIM_HOME%"

rem Toolchain doctor: print the dependency report in this console and exit.
if "%~1"=="--doctor" (
    "%ESIM_HOME%\python\python.exe" "%ESIM_HOME%\src\frontEnd\Application.py" --doctor
    exit /b %errorlevel%
)

rem pythonw = no console window. Fall back to python.exe for a visible
rem traceback by running: esim.bat --debug
if "%~1"=="--debug" (
    "%ESIM_HOME%\python\python.exe"  "%ESIM_HOME%\src\frontEnd\Application.py"
) else (
    start "" "%ESIM_HOME%\python\pythonw.exe" "%ESIM_HOME%\src\frontEnd\Application.py"
)
endlocal
