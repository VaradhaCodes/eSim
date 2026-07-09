# eSim Windows build (`windows-test`) — porting defects found by build-testing

Testing `windows\build-windows.ps1` end-to-end on a real Windows 11 machine.
The branch's ngspice/Icarus **source build had never worked end-to-end on
Windows**; every failure below is a genuine Linux→Windows gap, not machine
noise. **All 14 defects are now fixed in-tree and the full script runs
end-to-end to a finished installer** (verified 2026-07-06):

    dist\eSim-2.5-installer.exe    520.5 MB   (+ .sha256)
    dist\kicad-9.0.3-x86_64.exe   1056.1 MB   (+ .sha256, shipped alongside)

1–5 + 8 applied earlier; 6/7 (Verilator objects), 9 (`-fno-strict-aliasing`),
10 (`-j2` cap) folded into `Stage-SimToolchain`; 11–14 (below) surfaced only
once the full script reached its later stages. A clean full run exercised
every fix: custom ngspice built + **smoke sim passed** (no segfault),
iverilog+libvvp built, official ngspice/KiCad staged, Inno installer compiled.
**Nothing is committed.** Remaining before merge: re-commit the recovered
tarball (bug 3); verify the `-AcceptNewHashes`-recorded sha256s
(ngspice/innosetup/kicad) against upstream.

## Prereqs surfaced
- **PowerShell 7 required.** The script uses `Get-Date -AsUTC` (PS7-only); stock
  Windows PowerShell 5.1 fails instantly. Either fix the script
  (`(Get-Date).ToUniversalTime()`) or document the pwsh-7 requirement.

## Bugs

1. **`Stage-Msys` pacman chain never installs the toolchain.**
   `pacman -Syu && pacman -S <toolchain>` in one shell: MSYS2's core `-Syu`
   *deliberately self-terminates* its shell (can happen on several passes), so
   the `&&` chain dies before the install. → loop `-Syu` ignoring exit code,
   gate only on the `pacman -S`. **Fixed.**

2. **`MSYSTEM` set inside the staging `if`-block.** On a re-run (msys64 already
   staged) that block is skipped, so the ghdl-backend check ran with no
   `/usr/bin` on PATH (`head` not found, check silently blank). → hoisted.
   **Fixed.**

3. **`nghdl/nghdl-simulator-source.tar.xz` missing from the branch.** The 4.3 MB
   ngspice source was *added then deleted* in git history (`git log --all`);
   `install-nghdl.sh` and `build-windows.ps1` both require it in-tree. Recovered
   blob `a32ac1da8a09e84f08d99a7c214af705864dd74e` via `git cat-file`.
   **Must be re-committed to the branch.**

4. **Build is not hermetic — sources the user's `~/.bashrc`.** Staged bash runs
   as a login shell with `HOME` = real Windows profile. A personal
   `~/.bashrc` (here: `export PATH="/c/Program Files/Python313:...:$PATH"` from an
   old `C:\FOSSEE` install) that drops `/usr/bin`+`/mingw64/bin` makes
   cygpath/head/gcc vanish → `cd: null directory`. → `Set-MsysEnv` points HOME
   at a clean build-local dir (`tools\msys64\home\builder`). **Fixed.** This
   would break on *any* dev machine with a customized bashrc.

5. **Code models need `-ldl`, absent in base mingw.** `spice2poly.cm` et al.
   link `-ldl` (Linux-ism); mingw has no libdl. → add
   `mingw-w64-x86_64-dlfcn`. **Fixed.**

6/7. **`Ngveri.cm`: Verilator runtime objects not built + weak-symbol DLL gap.**
   The patch adds `verilated.o`+`verilated_threads.o` as link inputs but nothing
   compiles them (tarball ships a stale 2022 *Linux* `verilated.o`;
   `verilated_threads.o` — Verilator-5 split — never made). And Verilator's weak
   `sc_time_stamp()` may be left undefined in a Linux `.so` but **not** in a
   Windows DLL. → compile both from `/mingw64/share/verilator/include` as COFF
   with **`-DVL_TIME_CONTEXT`**. **Verified manually; needs folding into
   `Stage-SimToolchain`.**

8. **`ngspice.exe` final link misses Winsock.** ngspice gates
   `-lpsapi -lshlwapi -lws2_32` behind an automake conditional that is *off* in
   the console build (no `--with-wingui`), but the nghdl socket server
   (`frontend/outitf.c`) needs them → `undefined reference to __imp_WSAStartup`
   etc. → add `LIBS='-lws2_32 -lpsapi -lshlwapi'` to the ngspice `configure`.
   **Verified:** ngspice.exe (6.6 MB) links, installs, loads all 8 `.cm`.

9. **Built ngspice segfaulted on every analysis — strict-aliasing miscompile.**
   `--version` worked and all 8 code models loaded, but `.op`/`.dc`/`.tran` all
   SIGSEGV'd in `strcat()` (empty netlist was fine). ngspice-35 (2020) under
   GCC 16 `-O2`. → build with **`-fno-strict-aliasing`**. **Fixed & verified:**
   diagnostic rebuild with the flag runs `.op` (divider `out=2.5V`), `.dc`, and
   `.tran` (RC charge curve) cleanly — no segfault. Folded into
   `Stage-SimToolchain` CFLAGS.

10. **MSYS2 `fork()` collapses under `make -j$(nproc)`.** A clean full compile
    triggers `dofork: child died 0xC0000142 / Resource temporarily unavailable`
    (EAGAIN) once many libtool/`sh` forks run at once (aggravated by BLODA —
    Cloudflare WARP / AV). → **cap parallelism at `make -j2`.** **Fixed:** the
    `-j2` diag rebuild forked clean start-to-finish. Folded into the script
    (was `-j$(nproc)`).

11. **Verification `Die`d on a missing `ivlng` (Icarus) code model.** The old
    committed tarball was an **ngspice-35** fork, and ivlng/d_cosim only exist
    upstream from ngspice ≥ 42 — so the check could never pass. Interim fix
    downgraded the `Die` to a `Log`. **Superseded 2026-07-07:** the tarball was
    rebuilt on **ngspice-45.2** (official source + the full nghdl delta:
    ghdl/Ngveri icm model dirs, `outitf.c` ghdlserver close hook,
    spinit/makedefs wiring, Verilator-5 link rules baked in — the separate
    `nghdl-simulator.patch` is gone). That base ships `d_cosim` + `ivlng.dll` +
    `ivlng.vpi`, so the ivlng check is a hard `Die` again, `tlines.cm` joined
    the verified model list, and the `-std=gnu11` C23 pin (a 35-ism) was
    dropped from both installers. Notes 9 (`-fno-strict-aliasing`, kept as a
    harmless safety) and 8/10 still apply.

12. **Official ngspice 45.2 download 404s.** SourceForge moved 45.2 to
    `ng-spice-rework/old-releases/45.2/`; the manifest URL fetched a 404 HTML
    page. → repoint `deps-manifest.json` `ngspice.url` to the `old-releases`
    path (same file, sha256 `6a0c4405…`). **Fixed.**

13. **Inno Setup 6.3.3 download 404s.** jrsoftware pruned
    `files.jrsoftware.org/is/6/`; older point releases now live on GitHub. →
    repoint `innosetup.url` to
    `github.com/jrsoftware/issrc/releases/download/is-6_3_3/innosetup-6.3.3.exe`
    (sha256 `0bcb2a40…`). **Fixed.**

14. **`Resolve-Iscc` never finds ISCC — PowerShell env-var syntax bug.**
    `"$env:ProgramFiles(x86)\..."` parses as `$env:ProgramFiles` + literal
    `(x86)` → `C:\Program Files(x86)\...` (no space), a path that never exists,
    so the post-install `Test-Path` always failed with "Inno Setup install
    failed" even though ISCC installed fine. → brace-wrap:
    `${env:ProgramFiles(x86)}` (both the candidate list and the post-install
    check). **Fixed.**

15. **"Do you want Ngspice plots?" → Yes opened a window that could never draw.**
    eSim's ngspice is configured without `--with-wingui` on purpose: eSim drives
    it through `QProcess` and parses its stdout, which a wingui binary swallows
    into its own window. But that leaves the binary with *no* graphics device on
    Windows (`config.h`: `X_DISPLAY_MISSING 1`, no `HAS_WINGUI`) — the same
    console build plots on Ubuntu only because it links X11. So the mintty plot
    session printed `Warning: no graphics interface!` and answered every `plot`
    line from the netlist's `.control` block with `Can't open viewport for
    graphics`. → build the source a **second** time with `--with-wingui`, same
    `--prefix` and no `make install`, and keep only the exe as
    `install_dir\bin\ngspice_gui.exe`; `open_ngspice_plots` runs that for the
    plot session (it owns its console window, so mintty — and therefore the HDL
    component — is no longer needed for plots). **Fixed.**

## Also
- `-AcceptNewHashes` filled blank `deps-manifest.json` sha256s with whatever
  downloaded — **verify against upstream checksums before merge.**
- `iverilog_source` is a manifest download (steveicarus `de415b2f`), fine.
