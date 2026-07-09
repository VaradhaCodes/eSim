<#
=============================================================================
          FILE: build-windows.ps1

         USAGE: powershell -ExecutionPolicy Bypass -File windows\build-windows.ps1
                    [-SkipMsys] [-AcceptNewHashes] [-Clean]

   DESCRIPTION: Reproducible Windows packaging for eSim. Runs on a Windows
                build machine (or windows-latest CI runner). Produces:
                    windows\dist\eSim-<VERSION>-installer.exe   (Inno Setup)
                    windows\dist\kicad-<ver>-x86_64.exe         (pass-through)
                    windows\dist\*.sha256

                Every third-party download is pinned in deps-manifest.json
                (URL + sha256). Nothing is fetched from a moving 'latest'.

                Pipeline:
                  1. verify build tools (7z; Inno Setup auto-installed)
                  2. download + hash-check every dep into windows\downloads\
                  3. stage windows\build\eSim\        (app tree, filtered like
                     make-release.sh, INCLUDING nghdl\src for VHDL co-sim)
                  4. stage private Python + pip install requirements-windows
                  5. stage tools\msys64 (MSYS2 + mingw64 gcc/make/verilator/
                     ghdl-llvm; the HDL-toolchain installer component AND the
                     build substrate for step 6)
                  6. Stage-SimToolchain: build the CUSTOM eSim ngspice
                     (nghdl-simulator: d_cosim + ivlng + ghdl.cm) from
                     nghdl\nghdl-simulator-source.tar.xz inside MSYS2 into
                     tools\nghdl\{src,release,install_dir}, and Icarus
                     Verilog from the pinned source WITH libvvp into
                     library\bin\iverilog -- both hard-verified (binaries
                     run, code models present, a trivial .cir simulates)
                  7. stage tools\ngspice (official build; the Compact
                     flavour's plain-simulation fallback)
                  8. compile installer.iss with ISCC

        -SkipMsys        build without the HDL toolchain component (smaller
                         installer; NgVeri/NGHDL builds unavailable).
                         Implies -SkipSimBuild.
        -SkipSimBuild    skip the MSYS2 source builds (step 6): ngspice falls
                         back to the official zip shim and iverilog to the
                         Bleyer setup (NO libvvp -> d_cosim and VHDL co-sim
                         will not work). For quick packaging iterations only.
        -AcceptNewHashes record sha256 for manifest entries whose hash is
                         blank, rewriting deps-manifest.json. Verify recorded
                         hashes against upstream before committing!
        -Clean           delete build\ and dist\ first (downloads\ is kept;
                         it is content-addressed by the manifest hashes)

  ORGANIZATION: eSim Team, FOSSEE, IIT Bombay
=============================================================================
#>
[CmdletBinding()]
param(
    [switch]$SkipMsys,
    [switch]$SkipSimBuild,
    [switch]$AcceptNewHashes,
    [switch]$Clean
)
if ($SkipMsys) { $SkipSimBuild = $true }   # no MSYS2 -> nothing to build with

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # 10x faster Invoke-WebRequest

$WinDir   = $PSScriptRoot
$RepoRoot = Split-Path $WinDir -Parent
$Downloads = Join-Path $WinDir 'downloads'
$Build     = Join-Path $WinDir 'build'
$Stage     = Join-Path $Build 'eSim'
$Dist      = Join-Path $WinDir 'dist'
$Version   = (Get-Content (Join-Path $RepoRoot 'VERSION') -Raw).Trim()

function Log([string]$msg) { Write-Host ">>> $msg" -ForegroundColor Cyan }
function Die([string]$msg) { Write-Error $msg; exit 1 }

# ---------------------------------------------------------------- tools ----
function Resolve-7z {
    foreach ($c in @('7z', "$env:ProgramFiles\7-Zip\7z.exe")) {
        if (Get-Command $c -ErrorAction SilentlyContinue) { return $c }
    }
    Die '7-Zip is required (winget install 7zip.7zip) - extracts .7z/.tar.xz/.nupkg'
}

function Resolve-Iscc {
    # NB: ${env:ProgramFiles(x86)} MUST be brace-wrapped -- "$env:ProgramFiles(x86)"
    # parses as $env:ProgramFiles + literal "(x86)" -> "C:\Program Files(x86)" (no
    # space), a path that never exists, so the check always failed.
    foreach ($c in @('iscc',
                     "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
                     "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe")) {
        if (Get-Command $c -ErrorAction SilentlyContinue) { return $c }
    }
    # Auto-install from the pinned manifest entry (build-machine tool only).
    Log 'Inno Setup not found - installing from pinned manifest entry'
    $setup = Get-Dep 'innosetup'
    & $setup /VERYSILENT /SUPPRESSMSGBOXES /NORESTART | Out-Null
    $c = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
    if (-not (Test-Path $c)) { Die 'Inno Setup install failed' }
    return $c
}

# ------------------------------------------------------------- manifest ----
$ManifestPath = Join-Path $WinDir 'deps-manifest.json'
$Manifest = Get-Content $ManifestPath -Raw | ConvertFrom-Json

function Get-Dep([string]$name) {
    <# Download (once) + sha256-verify a manifest entry; returns local path. #>
    $e = $Manifest.$name
    if (-not $e) { Die "deps-manifest.json has no entry '$name'" }
    New-Item -ItemType Directory -Force -Path $Downloads | Out-Null
    $file = Join-Path $Downloads $e.filename
    if (-not (Test-Path $file)) {
        Log "Downloading $name $($e.version)"
        Invoke-WebRequest -Uri $e.url -OutFile $file -UserAgent 'eSim-build'
    }
    $hash = (Get-FileHash $file -Algorithm SHA256).Hash.ToLower()
    if ([string]::IsNullOrEmpty($e.sha256)) {
        if (-not $AcceptNewHashes) {
            Die "$name has no pinned sha256. Re-run with -AcceptNewHashes, then verify the recorded hash against upstream's published checksum before committing."
        }
        $e.sha256 = $hash
        $Manifest | ConvertTo-Json -Depth 5 | Set-Content $ManifestPath -Encoding UTF8
        Log "$name sha256 recorded: $hash  (VERIFY against upstream!)"
    }
    elseif ($hash -ne $e.sha256.ToLower()) {
        Die "$name hash mismatch!`n  expected $($e.sha256)`n  got      $hash`nDelete $file if upstream re-released, and re-verify."
    }
    return $file
}

# ---------------------------------------------------------------- stage ----
function Stage-App {
    Log 'Staging eSim tree (working-tree snapshot, filtered like make-release.sh)'
    New-Item -ItemType Directory -Force -Path $Stage | Out-Null
    # Same exclusion set as make-release.sh: VCS/build cruft + regeneratable
    # simulation outputs. robocopy /MIR keeps re-runs incremental. nghdl/ IS
    # staged (src/ = the VHDL co-sim python + ghdlserver sources that
    # ngspice_ghdl.py needs at runtime via [SRC] SRC_HOME, Example/ = test
    # designs); only its 11 MB simulator source tarball stays out -- the
    # BUILT tree ships at tools\nghdl instead (Stage-SimToolchain).
    # NOTE: library\SubcircuitLibrary IS staged (make-release.sh excludes it,
    # but the app needs it at runtime: the subcircuit browser opens there and
    # the netlister resolves X-instances against its .sub files -- without it
    # every schematic using a library IC fails to convert). Its bulk was sim
    # outputs; those are stripped by the /XF filters below instead.
    $xd = @('.git', 'dist', '__pycache__', '.pytest_cache', '.mypy_cache',
            'node_modules', 'Ubuntu',
            'windows', 'flatpak', 'snap', 'appimage', 'docker-launcher') |
          ForEach-Object { Join-Path $RepoRoot $_ }
    # Stage-side dirs the LATER stages create (private Python, built
    # toolchains): excluded so /MIR's purge pass cannot delete them on an
    # incremental re-run -- rebuilding the ngspice toolchain costs ~an hour.
    $xd += @('python', 'tools', 'library\bin\iverilog') |
           ForEach-Object { Join-Path $Stage $_ }
    robocopy $RepoRoot $Stage /MIR /NFL /NDL /NJH /NJS /XD @xd `
        /XF '*.pyc' '*.pyo' '*.raw' 'plot_data_*.txt' '.DS_Store' `
            'fp-info-cache' `
            'nghdl-simulator-source.tar.xz' `
            'install-nghdl.sh' | Out-Null
    if ($LASTEXITCODE -ge 8) { Die "robocopy failed ($LASTEXITCODE)" }

    # kicadLibrary must be a real dir at runtime on Windows (symbols are
    # referenced in place); expand the tarball form if that's what the tree has.
    $lib = Join-Path $Stage 'library'
    if ((Test-Path "$lib\kicadLibrary.tar.xz") -and -not (Test-Path "$lib\kicadLibrary")) {
        & $7z x "$lib\kicadLibrary.tar.xz" "-o$lib" -y | Out-Null
        & $7z x "$lib\kicadLibrary.tar"    "-o$lib" -y | Out-Null
        Remove-Item "$lib\kicadLibrary.tar"
    }

    Set-Content (Join-Path $Stage 'RELEASE') @"
eSim release (Windows)
version    : $Version
built      : $(Get-Date -AsUTC -Format yyyy-MM-dd) (UTC)
installer  : Inno Setup (windows/installer.iss)
"@

    # The repo windows\ dir is excluded wholesale above (build cruft:
    # build-windows.ps1, installer.iss, downloads\, dist\). But two files in
    # it are RUNTIME launchers the shipped tree needs:
    #   * esim.bat            -> stage ROOT: its ESIM_HOME=%~dp0 must resolve to
    #                            the install root so %ESIM_HOME%\src, \python,
    #                            \tools, \windows\windows_bootstrap.py all hit.
    #   * windows_bootstrap.py -> stage windows\: esim.bat runs it every launch
    #                            for per-user setup (config.ini, sym-lib-table).
    # Stage them back explicitly so installer.iss (StageDir\* -> {app}) ships them.
    Copy-Item (Join-Path $WinDir 'esim.bat') (Join-Path $Stage 'esim.bat') -Force
    $stageWin = Join-Path $Stage 'windows'
    New-Item -ItemType Directory -Force -Path $stageWin | Out-Null
    Copy-Item (Join-Path $WinDir 'windows_bootstrap.py') (Join-Path $stageWin 'windows_bootstrap.py') -Force
}

function Stage-Python {
    Log 'Staging private Python + wheel set'
    $pkg = Get-Dep 'python'
    $pydir = Join-Path $Stage 'python'
    if (-not (Test-Path "$pydir\python.exe")) {
        $tmp = Join-Path $Build 'python-nupkg'
        & $7z x $pkg "-o$tmp" -y | Out-Null          # .nupkg is a zip
        New-Item -ItemType Directory -Force -Path $pydir | Out-Null
        Copy-Item "$tmp\tools\*" $pydir -Recurse -Force
        Remove-Item $tmp -Recurse -Force
    }
    & "$pydir\python.exe" -m pip install --upgrade pip --quiet
    & "$pydir\python.exe" -m pip install --quiet `
        -r (Join-Path $WinDir 'requirements-windows.txt')
    if ($LASTEXITCODE -ne 0) { Die 'pip install failed' }
    # Record the exact resolved set for the release notes / reproducibility.
    & "$pydir\python.exe" -m pip freeze |
        Set-Content (Join-Path $Stage 'python-wheels.lock')
    # Sanity: the GUI toolkit must import.
    & "$pydir\python.exe" -c 'import PyQt6.QtWidgets, PyQt6.Qsci'
    if ($LASTEXITCODE -ne 0) { Die 'PyQt6/Qsci import check failed' }
}

function Stage-Ngspice {
    Log 'Staging ngspice (official Windows build; Compact-flavour fallback)'
    $arc = Get-Dep 'ngspice'
    $dst = Join-Path $Stage 'tools\ngspice'
    if (-not (Test-Path $dst)) {
        $tmp = Join-Path $Build 'ngspice-x'
        & $7z x $arc "-o$tmp" -y | Out-Null
        # Archive root is Spice64/: bin/ngspice.exe + lib/ngspice/*.cm
        $root = Get-ChildItem $tmp -Directory | Select-Object -First 1
        New-Item -ItemType Directory -Force -Path $dst | Out-Null
        Copy-Item "$($root.FullName)\*" $dst -Recurse -Force
        Remove-Item $tmp -Recurse -Force
    }
    if (-not (Test-Path "$dst\bin\ngspice.exe")) { Die 'ngspice.exe not staged' }
    if ($SkipSimBuild) {
        # No custom build: CosimConfig resolves
        # <NGHDL_HOME>\install_dir\bin\ngspice.exe and the bootstrap points
        # NGHDL_HOME at tools\nghdl, so shim the official build there. The
        # official build has NO ivlng/ghdl.cm -> d_cosim/VHDL co-sim stay off
        # (and the in-app doctor says so).
        $shim = Join-Path $Stage 'tools\nghdl\install_dir'
        if (-not (Test-Path $shim)) {
            New-Item -ItemType Directory -Force -Path (Split-Path $shim) | Out-Null
            Copy-Item $dst $shim -Recurse
        }
    }
}

function Stage-Iverilog {
    <# -SkipSimBuild fallback only: Bleyer's prebuilt Icarus. It has NO
       libvvp, so the Verilog Verifier works but d_cosim cannot run. The
       real path is the libvvp source build in Stage-SimToolchain. #>
    if (-not $SkipSimBuild) { return }
    Log 'Staging Icarus Verilog (Bleyer fallback; NO libvvp -> no d_cosim)'
    $setup = Get-Dep 'iverilog'
    $dst = Join-Path $Stage 'library\bin\iverilog'
    if (-not (Test-Path $dst)) {
        $tmp = Join-Path $Build 'iverilog-x'
        # Bleyer's setup is Inno-based; 7z extracts its {app} payload.
        & $7z x $setup "-o$tmp" -y | Out-Null
        New-Item -ItemType Directory -Force -Path $dst | Out-Null
        $payload = if (Test-Path "$tmp\{app}") { "$tmp\{app}" } else { $tmp }
        Copy-Item "$payload\*" $dst -Recurse -Force
        Remove-Item $tmp -Recurse -Force
    }
    if (-not (Test-Path "$dst\bin\iverilog.exe")) { Die 'iverilog.exe not staged' }
}

# MSYS2 package sets. mingw64: the toolchain the APP invokes at model-build
# time (ModelGeneration/nghdl: mingw32-make, gcc, verilator, ghdl) -- these
# SHIP in the 'hdl' installer component. msys: build-only utilities that
# Stage-SimToolchain needs to compile ngspice/iverilog from source; they ride
# along in the shipped tree (a few MB) and keep user-side model rebuilds
# self-sufficient.
$MingwPkgs = 'mingw-w64-x86_64-gcc mingw-w64-x86_64-make ' +
             'mingw-w64-x86_64-verilator mingw-w64-x86_64-ghdl-llvm ' +
             # dlfcn-win32 provides libdl (-ldl): the ngspice XSPICE code-model
             # link (src/xspice/icm/makedefs) hard-codes -ldl, a Linux-ism.
             # mingw has no libdl in its base -> spice2poly.cm et al. fail with
             # "cannot find -ldl". dlfcn-win32 is a Win32-API-backed libdl.
             'mingw-w64-x86_64-dlfcn'
$MsysPkgs  = 'make autoconf automake libtool bison flex gperf patch tar'

function Set-MsysEnv {
    <# Make every staged-MSYS2 bash call HERMETIC. The staged bash runs as a
       login shell (-l): it sources /etc/profile (good -- that puts /mingw64/bin
       and /usr/bin on PATH) but THEN ~/.bash_profile + ~/.bashrc. The stock
       nsswitch `db_home: cygwin desc` resolves HOME to the real Windows user
       profile, so a personal ~/.bashrc (e.g. one doing `export PATH=...:$PATH`
       that drops /usr/bin) silently wrecks the build -- cygpath/head/gcc vanish
       and commands fail with "cd: null directory". Point HOME at a clean
       build-local dir so NO user dotfiles are sourced; MSYS2 honors an
       inherited HOME. #>
    $env:MSYSTEM = 'MINGW64'
    $env:CHERE_INVOKED = '1'
    $env:HOME = Join-Path $Stage 'tools\msys64\home\builder'
    New-Item -ItemType Directory -Force -Path $env:HOME | Out-Null
}

function Invoke-MsysBash([string]$cmd, [string]$errmsg) {
    <# Run one command line in the staged MSYS2's bash as a MINGW64 login
       shell (login -> /etc/profile puts /mingw64/bin on PATH; CHERE_INVOKED
       keeps the caller's working directory). Dies with $errmsg on failure. #>
    $bash = Join-Path $Stage 'tools\msys64\usr\bin\bash.exe'
    if (-not (Test-Path $bash)) { Die "MSYS2 bash not staged ($bash)" }
    Set-MsysEnv
    & $bash -lc $cmd
    if ($LASTEXITCODE -ne 0) { Die "$errmsg (bash rc=$LASTEXITCODE)" }
}

function Stage-Msys {
    if ($SkipMsys) { Log 'Skipping MSYS2 toolchain (-SkipMsys)'; return }
    Log 'Staging MSYS2 + mingw gcc/make/verilator/ghdl-llvm (HDL toolchain)'
    $arc = Get-Dep 'msys2'
    $dst = Join-Path $Stage 'tools\msys64'
    # Set the hermetic MINGW64 login environment for EVERY raw `& $bash` call in
    # this function -- not just the staging block below. On a re-run (msys64
    # already staged) that block is skipped, but the ghdl backend check further
    # down still needs /usr/bin on PATH (else `head` and friends are not found
    # and the check silently reports nothing).
    Set-MsysEnv
    if (-not (Test-Path $dst)) {
        $tmp = Join-Path $Build 'msys-x'
        & $7z x $arc "-o$tmp" -y | Out-Null            # .tar.xz -> .tar
        & $7z x (Get-ChildItem "$tmp\*.tar").FullName "-o$tmp" -y | Out-Null
        New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
        Move-Item "$tmp\msys64" $dst
        Remove-Item $tmp -Recurse -Force
        # First run initializes; then install the toolchain NgVeri/NGHDL
        # invoke (ModelGeneration: mingw64\bin\mingw32-make.exe, verilator,
        # VERILATOR_ROOT=<msys>\mingw64; ngspice_ghdl: gcc, ghdl) plus the
        # source-build utilities. ghdl-llvm EXPLICITLY: the plain ghdl
        # alternative can resolve to mcode, which cannot link the nghdl
        # socket server (see nghdl/install-nghdl-scripts/GHDL-BACKEND-26.04.md).
        # MSYS2 provisioning MUST be multi-pass, each in its OWN login shell.
        # A core-system `pacman -Syu` (pacman, bash, msys2-runtime, filesystem)
        # DELIBERATELY terminates every MSYS2 process when done -- the shell
        # dies with a nonzero code BY DESIGN, and this can happen on more than
        # one consecutive pass (runtime, then keyring/db settle). So:
        #   * chaining `pacman -Syu && pacman -S ...` never reaches the install;
        #   * gating `Die` on a -Syu exit code would abort a healthy machine.
        # Run -Syu repeatedly, IGNORING its exit code, until a pass makes no
        # changes (rc 0 with no self-terminate), THEN install the toolchain and
        # gate only on the install + the binary/ghdl checks below.
        $bash = "$dst\usr\bin\bash.exe"
        & $bash -lc 'true'                                    # trigger first-run init (keyring, post-install)
        for ($i = 1; $i -le 4; $i++) {                        # -Syu may self-terminate several times; rc ignored on purpose
            & $bash -lc 'pacman -Syu --noconfirm'
            if ($LASTEXITCODE -eq 0) { break }                # a clean pass (no core update left) => settled
        }
        & $bash -lc "pacman -S --noconfirm --needed $MingwPkgs $MsysPkgs"
        if ($LASTEXITCODE -ne 0) { Die 'MSYS2 toolchain provisioning failed' }
        & $bash -lc 'pacman -Scc --noconfirm'
    }
    if (-not (Test-Path "$dst\mingw64\bin\mingw32-make.exe")) {
        Die 'mingw32-make.exe missing after MSYS2 provisioning'
    }
    # Prove the GHDL backend is NOT mcode before anything gets built with it.
    $ghdlver = & "$dst\usr\bin\bash.exe" -lc '/mingw64/bin/ghdl --version 2>&1 | head -n2'
    if ($ghdlver -match 'mcode') {
        Die "MSYS2 ghdl reports an mcode backend ($ghdlver). nghdl needs llvm/gcc; pin mingw-w64-x86_64-ghdl-llvm."
    }
    Log "ghdl: $($ghdlver -join ' ')"
}

function Stage-SimToolchain {
    <# The heart of Windows sim parity: build the CUSTOM eSim ngspice
       (nghdl-simulator -- ngspice + d_cosim + the ivlng Icarus bridge +
       ghdl.cm/Ngveri.cm) and a libvvp-enabled Icarus Verilog inside the
       staged MSYS2, mirroring what nghdl/install-nghdl.sh does on Ubuntu.

       Ships THREE trees, matching the Ubuntu $HOME/nghdl-simulator layout so
       every config key means the same thing on both OSes:
         tools\nghdl\src         ngspice sources (runtime model builds write
                                 new models into src\xspice\icm -- that is
                                 config DIGITAL_MODEL)
         tools\nghdl\release     the CONFIGURED build tree (runtime
                                 mingw32-make in release\src\xspice\icm
                                 rebuilds Ngveri.cm/ghdl.cm -- config RELEASE)
         tools\nghdl\install_dir the ngspice prefix (bin\ngspice.exe,
                                 lib\ngspice\*.cm -- what CosimConfig runs)
       Console build (no --with-wingui), exactly like Ubuntu: eSim drives
       ngspice through QProcess and parses stdout; the wingui build hijacks
       stdout into its own window. That console binary has no graphics device
       on Windows, so a second wingui-only exe is built alongside it purely for
       the interactive plot window (see below). #>
    if ($SkipSimBuild) {
        Log 'Skipping sim-toolchain source builds (-SkipSimBuild)'
        return
    }

    $nghdlDst  = Join-Path $Stage 'tools\nghdl'
    $ngspiceExe = Join-Path $nghdlDst 'install_dir\bin\ngspice.exe'
    $tarball = Join-Path $RepoRoot 'nghdl\nghdl-simulator-source.tar.xz'
    if (-not (Test-Path $tarball)) { Die "missing $tarball" }
    $toolsU = (Join-Path $Stage 'tools') -replace '\\', '/'
    $dllShU = (Join-Path $RepoRoot 'windows\msys-dll-closure.sh') -replace '\\', '/'

    # --- custom ngspice ----------------------------------------------------
    if (Test-Path $ngspiceExe) {
        Log 'Custom ngspice already staged (delete tools\nghdl to rebuild)'
    }
    else {
        Log 'Building custom eSim ngspice (nghdl-simulator, ngspice-45.2 base) in MSYS2 - this takes a while'
        $tarU   = $tarball -replace '\\', '/'
        # Fresh tree every time; configure a clean build dir. The tarball is the
        # eSim nghdl-simulator: ngspice-45.2 + the nghdl delta (ghdl/Ngveri icm
        # model dirs, outitf.c ghdlserver close hook, spinit/makedefs wiring,
        # Verilator-5 link rules) baked in -- no separate patch step anymore.
        # ngspice >= 42 also brings d_cosim + the ivlng Icarus bridge, which the
        # old ngspice-35 tarball could never provide.
        Invoke-MsysBash (
            "set -e; cd `"`$(cygpath -u '$toolsU')`" && " +
            "rm -rf nghdl nghdl-simulator-source && " +
            "tar -xJf `"`$(cygpath -u '$tarU')`" && " +
            "mv nghdl-simulator-source nghdl && cd nghdl && " +
            "rm -rf release && " +
            # Verilator runtime objects for Ngveri.cm: the icm makefile lists them
            # as link inputs but nothing compiles them (verilated_threads.o is a
            # Verilator-5 split). -DVL_TIME_CONTEXT drops the weak sc_time_stamp()
            # that a Windows DLL, unlike a Linux .so, will not leave undefined.
            "g++ -DVL_TIME_CONTEXT -I/mingw64/share/verilator/include -I/mingw64/share/verilator/include/vltstd -std=gnu++17 -O2 -fPIC -c /mingw64/share/verilator/include/verilated.cpp -o src/xspice/icm/Ngveri/verilated.o && " +
            "g++ -DVL_TIME_CONTEXT -I/mingw64/share/verilator/include -I/mingw64/share/verilator/include/vltstd -std=gnu++17 -O2 -fPIC -c /mingw64/share/verilator/include/verilated_threads.cpp -o src/xspice/icm/Ngveri/verilated_threads.o && " +
            "mkdir -p install_dir release && cd release && " +
            "../configure --enable-xspice --disable-debug " +
            "--prefix=`"`$(cygpath -am ../install_dir)`" " +
            "--exec-prefix=`"`$(cygpath -am ../install_dir)`" " +
            # -fno-strict-aliasing: kept from the ngspice-35 era (GCC-16 -O2
            # miscompiled its strcat paths); semantically safe on 45.2.
            # LIBS: ngspice gates Winsock behind --with-wingui (off in this console
            # build) but the nghdl socket client (frontend/outitf.c) needs ws2_32.
            "CFLAGS='-m64 -O2 -fno-strict-aliasing' LDFLAGS='-m64 -s' " +
            "LIBS='-lws2_32 -lpsapi -lshlwapi' && " +
            # -j2 NOT -j`$(nproc): MSYS2's fork() emulation collapses under heavy
            # parallel libtool/sh spawning (dofork: child died 0xC0000142, EAGAIN).
            "make -j2 && make install && " +
            # ngspice.exe (and the .cm/ivlng DLLs it loads) depend on MinGW
            # runtime DLLs (gomp/readline/termcap/stdc++/gcc_s/winpthread).
            # Ship the transitive closure next to the exe so the simulator runs
            # WITHOUT tools\msys64 on PATH (Compact installs, plain sims).
            "sh `"`$(cygpath -u '$dllShU')`" `"`$(cygpath -u '$toolsU')/nghdl`""
        ) 'custom ngspice build failed'
    }
    if (-not (Test-Path $ngspiceExe)) { Die 'ngspice.exe missing after custom build' }

    <# --- wingui ngspice, for interactive plots only -------------------------
       The console build above has no graphics device at all (config.h:
       X_DISPLAY_MISSING, no HAS_WINGUI), so `plot` in it can only ever answer
       "Can't open viewport for graphics" -- on Ubuntu the same console build
       plots because it links X11. Build the source a second time with
       --with-wingui and keep just the exe: NgspiceWidget.open_ngspice_plots
       runs it for the plot session while the batch run keeps using the console
       twin, whose stdout it parses. On Windows ngspice finds its lib dir
       relative to the exe (src/misc/ivars.c: dirname(argv0)/../share/ngspice),
       so the exe copied into install_dir\bin picks up the console build's
       installed spinit and its absolute codemodel paths; no `make install`,
       so nothing in install_dir is overwritten. #>
    $ngspiceGuiExe = Join-Path $nghdlDst 'install_dir\bin\ngspice_gui.exe'
    if (Test-Path $ngspiceGuiExe) {
        Log 'wingui ngspice already staged (delete tools\nghdl to rebuild)'
    }
    else {
        Log 'Building wingui ngspice (interactive plot window) in MSYS2'
        Invoke-MsysBash (
            "set -e; cd `"`$(cygpath -u '$toolsU')/nghdl`" && " +
            "rm -rf release_gui && mkdir -p release_gui && cd release_gui && " +
            "../configure --with-wingui --enable-xspice --disable-debug " +
            "--prefix=`"`$(cygpath -am ../install_dir)`" " +
            "--exec-prefix=`"`$(cygpath -am ../install_dir)`" " +
            "CFLAGS='-m64 -O2 -fno-strict-aliasing' LDFLAGS='-m64 -s' " +
            "LIBS='-lws2_32 -lpsapi -lshlwapi' && " +
            "make -j2 && " +
            "cp src/ngspice.exe ../install_dir/bin/ngspice_gui.exe && " +
            # The build tree is throwaway: only `release` is shipped (runtime
            # model rebuilds use it) and a second one would near-double it.
            "cd .. && rm -rf release_gui"
        ) 'wingui ngspice build failed'
    }
    if (-not (Test-Path $ngspiceGuiExe)) { Die 'ngspice_gui.exe missing after wingui build' }

    # Always ensure the MinGW runtime-DLL closure is present (idempotent, and
    # the "already staged" fast path must not skip it): without it the
    # verification below hard-hangs on Windows' missing-DLL error dialog.
    Invoke-MsysBash (
        "sh `"`$(cygpath -u '$dllShU')`" `"`$(cygpath -u '$toolsU')/nghdl`""
    ) 'runtime DLL closure staging failed'

    # Hard verification: version answers, the code models d_cosim/NGHDL need
    # exist, and a trivial transient actually simulates.
    $ver = & $ngspiceExe --version 2>&1 | Select-String 'ngspice' | Select-Object -First 1
    if (-not $ver) { Die 'staged ngspice does not answer --version' }
    Log "ngspice: $ver"
    $cmdir = Join-Path $nghdlDst 'install_dir\lib\ngspice'
    foreach ($cm in @('analog.cm', 'digital.cm', 'tlines.cm', 'ghdl.cm', 'Ngveri.cm')) {
        if (-not (Test-Path (Join-Path $cmdir $cm))) {
            Die "code model missing after build: $cmdir\$cm"
        }
    }
    # ngspice-45.2 ships the d_cosim code model (inside digital.cm's model set)
    # plus the ivlng Icarus bridge (ivlng.dll + ivlng.vpi, installed by
    # src/xspice/verilog). eSim's Icarus d_cosim flow and the toolchain doctor
    # both require it -- hard-fail if the build dropped it.
    foreach ($f in @('ivlng.dll', 'ivlng.vpi')) {
        if (-not (Test-Path (Join-Path $cmdir $f))) {
            Die "ivlng (Icarus d_cosim bridge) missing after build: $cmdir\$f"
        }
    }
    $smoke = Join-Path $Build 'smoke.cir'
    Set-Content $smoke "smoke test`nv1 1 0 dc 1`nr1 1 0 1k`n.op`n.end"
    & $ngspiceExe -b $smoke | Out-Null
    if ($LASTEXITCODE -ne 0) { Die "staged ngspice failed a trivial .cir run (rc=$LASTEXITCODE)" }
    Log 'ngspice smoke simulation OK'

    # --- Icarus Verilog with libvvp -----------------------------------------
    $ivDst = Join-Path $Stage 'library\bin\iverilog'
    $ivExe = Join-Path $ivDst 'bin\iverilog.exe'
    $haveLibvvp = @(Get-ChildItem "$ivDst\bin", "$ivDst\lib" -Filter 'libvvp*' `
                    -ErrorAction SilentlyContinue).Count -gt 0
    if ((Test-Path $ivExe) -and $haveLibvvp) {
        Log 'libvvp iverilog already staged (delete library\bin\iverilog to rebuild)'
    }
    else {
        Log 'Building Icarus Verilog with libvvp in MSYS2 (pinned source)'
        $src = Get-Dep 'iverilog_source'
        $srcU = $src -replace '\\', '/'
        $bldU = (Join-Path $Build 'iverilog-src') -replace '\\', '/'
        $dstU = $ivDst -replace '\\', '/'
        Invoke-MsysBash (
            "set -e; rm -rf `"`$(cygpath -u '$bldU')`" && mkdir -p `"`$(cygpath -u '$bldU')`" && " +
            "cd `"`$(cygpath -u '$bldU')`" && " +
            "tar -xzf `"`$(cygpath -u '$srcU')`" --strip-components=1 && " +
            "sh autoconf.sh && " +
            "./configure --prefix=`"`$(cygpath -m '$dstU')`" --enable-libvvp && " +
            "make -j`$(nproc) && make install"
        ) 'iverilog (libvvp) build failed'
    }
    if (-not (Test-Path $ivExe)) { Die 'iverilog.exe missing after build' }
    if (-not (Test-Path (Join-Path $ivDst 'bin\vvp.exe'))) { Die 'vvp.exe missing after build' }
    if (-not (Get-ChildItem "$ivDst\bin", "$ivDst\lib" -Filter 'libvvp*' -ErrorAction SilentlyContinue)) {
        Die "libvvp missing under $ivDst - the whole point of the source build. Check --enable-libvvp support at the pinned ref."
    }
    # ngspice's ivlng adapter does LoadLibrary("libvvp") -- the UNVERSIONED
    # name (eSim netlists pass no lib_args). The MinGW build emits only the
    # versioned libvvp-1.dll, so d_cosim dies with "Cannot open DLL libvvp"
    # unless a plain libvvp.dll sits beside it. Keep both.
    $vvpVersioned = Get-ChildItem "$ivDst\bin" -Filter 'libvvp-*.dll' -ErrorAction SilentlyContinue |
        Select-Object -First 1
    $vvpPlain = Join-Path $ivDst 'bin\libvvp.dll'
    if ($vvpVersioned -and -not (Test-Path $vvpPlain)) {
        Copy-Item $vvpVersioned.FullName $vvpPlain
        Log "staged libvvp.dll (copy of $($vvpVersioned.Name)) for ngspice ivlng"
    }
    if (-not (Test-Path $vvpPlain)) {
        Die "libvvp.dll (unversioned, dlopen-ed by ngspice ivlng) missing under $ivDst\bin"
    }
    Log 'Icarus Verilog (with libvvp) staged'
}

# ----------------------------------------------------------------- main ----
if ($Clean) { Remove-Item $Build, $Dist -Recurse -Force -ErrorAction SilentlyContinue }
$7z = Resolve-7z
New-Item -ItemType Directory -Force -Path $Build, $Dist | Out-Null

Stage-App
Stage-Python
Stage-Msys           # must precede Stage-SimToolchain (it builds inside MSYS2)
Stage-SimToolchain   # custom ngspice + libvvp iverilog (Full flavour)
Stage-Ngspice        # official build: Compact fallback (+ shim on -SkipSimBuild)
Stage-Iverilog       # Bleyer fallback, only on -SkipSimBuild

Log 'Compiling installer (Inno Setup)'
$Iscc = Resolve-Iscc
& $Iscc /Qp "/DAppVersion=$Version" "/DStageDir=$Stage" "/DOutDir=$Dist" `
    (Join-Path $WinDir 'installer.iss')
if ($LASTEXITCODE -ne 0) { Die 'ISCC failed' }

# KiCad ships alongside (never repacked); the eSim installer offers to run it.
Copy-Item (Get-Dep 'kicad_installer') $Dist -Force

Get-ChildItem $Dist -File | Where-Object Extension -ne '.sha256' | ForEach-Object {
    "$((Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower())  $($_.Name)" |
        Set-Content "$($_.FullName).sha256"
}

Log "Done. Artifacts in $Dist"
Get-ChildItem $Dist | Format-Table Name, @{n='MB';e={[math]::Round($_.Length/1MB,1)}}
