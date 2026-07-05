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
    foreach ($c in @('iscc',
                     "$env:ProgramFiles(x86)\Inno Setup 6\ISCC.exe",
                     "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe")) {
        if (Get-Command $c -ErrorAction SilentlyContinue) { return $c }
    }
    # Auto-install from the pinned manifest entry (build-machine tool only).
    Log 'Inno Setup not found - installing from pinned manifest entry'
    $setup = Get-Dep 'innosetup'
    & $setup /VERYSILENT /SUPPRESSMSGBOXES /NORESTART | Out-Null
    $c = "$env:ProgramFiles(x86)\Inno Setup 6\ISCC.exe"
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
    $xd = @('.git', 'dist', '__pycache__', '.pytest_cache', '.mypy_cache',
            'node_modules', 'library\subcircuitLibrary', 'Ubuntu',
            'windows', 'flatpak', 'snap', 'appimage', 'docker-launcher') |
          ForEach-Object { Join-Path $RepoRoot $_ }
    # Stage-side dirs the LATER stages create (private Python, built
    # toolchains): excluded so /MIR's purge pass cannot delete them on an
    # incremental re-run -- rebuilding the ngspice toolchain costs ~an hour.
    $xd += @('python', 'tools', 'library\bin\iverilog') |
           ForEach-Object { Join-Path $Stage $_ }
    robocopy $RepoRoot $Stage /MIR /NFL /NDL /NJH /NJS /XD @xd `
        /XF '*.pyc' '*.pyo' '*.raw' 'plot_data_*.txt' '.DS_Store' `
            'nghdl-simulator-source.tar.xz' 'nghdl-simulator.patch' `
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
             'mingw-w64-x86_64-verilator mingw-w64-x86_64-ghdl-llvm'
$MsysPkgs  = 'make autoconf automake libtool bison flex gperf patch tar'

function Invoke-MsysBash([string]$cmd, [string]$errmsg) {
    <# Run one command line in the staged MSYS2's bash as a MINGW64 login
       shell (login -> /etc/profile puts /mingw64/bin on PATH; CHERE_INVOKED
       keeps the caller's working directory). Dies with $errmsg on failure. #>
    $bash = Join-Path $Stage 'tools\msys64\usr\bin\bash.exe'
    if (-not (Test-Path $bash)) { Die "MSYS2 bash not staged ($bash)" }
    $env:MSYSTEM = 'MINGW64'
    $env:CHERE_INVOKED = '1'
    & $bash -lc $cmd
    if ($LASTEXITCODE -ne 0) { Die "$errmsg (bash rc=$LASTEXITCODE)" }
}

function Stage-Msys {
    if ($SkipMsys) { Log 'Skipping MSYS2 toolchain (-SkipMsys)'; return }
    Log 'Staging MSYS2 + mingw gcc/make/verilator/ghdl-llvm (HDL toolchain)'
    $arc = Get-Dep 'msys2'
    $dst = Join-Path $Stage 'tools\msys64'
    if (-not (Test-Path $dst)) {
        $tmp = Join-Path $Build 'msys-x'
        & $7z x $arc "-o$tmp" -y | Out-Null            # .tar.xz -> .tar
        & $7z x (Get-ChildItem "$tmp\*.tar").FullName "-o$tmp" -y | Out-Null
        Move-Item "$tmp\msys64" $dst
        Remove-Item $tmp -Recurse -Force
        # First run initializes; then install the toolchain NgVeri/NGHDL
        # invoke (ModelGeneration: mingw64\bin\mingw32-make.exe, verilator,
        # VERILATOR_ROOT=<msys>\mingw64; ngspice_ghdl: gcc, ghdl) plus the
        # source-build utilities. ghdl-llvm EXPLICITLY: the plain ghdl
        # alternative can resolve to mcode, which cannot link the nghdl
        # socket server (see nghdl/install-nghdl-scripts/GHDL-BACKEND-26.04.md).
        & "$dst\usr\bin\bash.exe" -lc 'true'
        & "$dst\usr\bin\bash.exe" -lc `
            "pacman -Syu --noconfirm && pacman -S --noconfirm --needed $MingwPkgs $MsysPkgs && pacman -Scc --noconfirm"
        if ($LASTEXITCODE -ne 0) { Die 'MSYS2 toolchain provisioning failed' }
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
       stdout into its own window. Interactive plots run via mintty. #>
    if ($SkipSimBuild) {
        Log 'Skipping sim-toolchain source builds (-SkipSimBuild)'
        return
    }

    $nghdlDst  = Join-Path $Stage 'tools\nghdl'
    $ngspiceExe = Join-Path $nghdlDst 'install_dir\bin\ngspice.exe'
    $tarball = Join-Path $RepoRoot 'nghdl\nghdl-simulator-source.tar.xz'
    $patch   = Join-Path $RepoRoot 'nghdl\nghdl-simulator.patch'
    if (-not (Test-Path $tarball)) { Die "missing $tarball" }

    # --- custom ngspice ----------------------------------------------------
    if (Test-Path $ngspiceExe) {
        Log 'Custom ngspice already staged (delete tools\nghdl to rebuild)'
    }
    else {
        Log 'Building custom eSim ngspice (nghdl-simulator) in MSYS2 - this takes a while'
        $toolsU = (Join-Path $Stage 'tools') -replace '\\', '/'
        $tarU   = $tarball -replace '\\', '/'
        $patchU = $patch   -replace '\\', '/'
        # Fresh tree every time: the tarball carries a PREBUILT LINUX release/
        # (ELF .o/.so) that would poison a Windows make; wipe it and configure
        # a clean build dir. Mirrors install-nghdl.sh incl. the C23 'bool'
        # CFLAGS fix (-std=gnu11) and the Verilator-5/GCC-14 link patch.
        Invoke-MsysBash (
            "set -e; cd `"`$(cygpath -u '$toolsU')`" && " +
            "rm -rf nghdl nghdl-simulator-source && " +
            "tar -xJf `"`$(cygpath -u '$tarU')`" && " +
            "mv nghdl-simulator-source nghdl && cd nghdl && " +
            "rm -rf release && " +
            "{ patch -p1 --forward < `"`$(cygpath -u '$patchU')`" || echo '[warn] patch skipped (already applied?)'; } && " +
            "mkdir -p install_dir release && cd release && " +
            "../configure --enable-xspice --disable-debug " +
            "--prefix=`"`$(cygpath -am ../install_dir)`" " +
            "--exec-prefix=`"`$(cygpath -am ../install_dir)`" " +
            "CFLAGS='-std=gnu11 -m64 -O2' LDFLAGS='-m64 -s' && " +
            "make -j`$(nproc) && make install"
        ) 'custom ngspice build failed'
    }
    if (-not (Test-Path $ngspiceExe)) { Die 'ngspice.exe missing after custom build' }

    # Hard verification: version answers, the code models d_cosim/NGHDL need
    # exist, and a trivial transient actually simulates.
    $ver = & $ngspiceExe --version 2>&1 | Select-String 'ngspice' | Select-Object -First 1
    if (-not $ver) { Die 'staged ngspice does not answer --version' }
    Log "ngspice: $ver"
    $cmdir = Join-Path $nghdlDst 'install_dir\lib\ngspice'
    foreach ($cm in @('analog.cm', 'digital.cm', 'ghdl.cm', 'Ngveri.cm')) {
        if (-not (Test-Path (Join-Path $cmdir $cm))) {
            Die "code model missing after build: $cmdir\$cm"
        }
    }
    if (-not (Get-ChildItem $cmdir -Filter 'ivlng*' -ErrorAction SilentlyContinue)) {
        Die "ivlng adapter missing in $cmdir - d_cosim would be dead. Check release\src\xspice\verilog build output."
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
