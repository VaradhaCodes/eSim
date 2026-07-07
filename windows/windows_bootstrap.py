# =========================================================================
#             FILE: windows_bootstrap.py
#
#      DESCRIPTION: First-run (and every-run, idempotent) setup for the
#                   Windows install of eSim. Invoked by the esim.bat launcher
#                   BEFORE Application.py, with the bundled Python.
#
#                   Everything here is per-user state, which is why it runs at
#                   launch time and not from the machine-wide installer: a
#                   multi-user PC gets a correct ~/.esim for EVERY user who
#                   launches eSim, and reinstalls/upgrades self-heal.
#
#                   Responsibilities:
#                     1. ~/.esim/config.ini          (eSim_HOME + resource
#                        keys, mirrors install-eSim.sh createConfigFile)
#                     2. ~/.esim/kicad_symbols/      seed the 3 generated
#                        symbol libraries once (never clobber user models)
#                     3. ~/.nghdl/config.ini         NGHDL_HOME/DIGITAL_MODEL/
#                        RELEASE/SRC_HOME/MSYS_HOME/[COSIM] for whichever
#                        toolchain pieces this install actually has
#                     4. spinit                      rewrite the bundled
#                        ngspice's codemodel paths for this install location
#                     5. KiCad sym-lib-table         register eSim libraries
#                        (static -> install dir, generated -> ~/.esim) in the
#                        user's %APPDATA%/kicad/<ver>/sym-lib-table
#
#                   Pure stdlib + eSim's own kicad_symlib helpers; no Qt. The
#                   logic is OS-independent on purpose so the test suite covers
#                   it on Linux CI even though only Windows ships it.
#
#            USAGE: python windows_bootstrap.py [--esim-root DIR]
#                   (esim.bat passes the install root; default = two levels up)
#
#     ORGANIZATION: eSim Team at FOSSEE, IIT Bombay
# =========================================================================

import argparse
import configparser
import os
import re
import shutil
import sys


def esim_root_default():
    """Install root = the dir holding VERSION + src/ (this file lives in
    <root>/windows/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _user_dir(*parts):
    return os.path.join(os.path.expanduser('~'), *parts)


GENERATED_LIBS = ("eSim_Ngveri", "eSim_NgVeriCosim", "eSim_Nghdl")


def write_esim_config(esim_root):
    """~/.esim/config.ini — same keys the Ubuntu installer writes."""
    cfg_dir = _user_dir('.esim')
    os.makedirs(cfg_dir, exist_ok=True)
    path = os.path.join(cfg_dir, 'config.ini')
    cfg = configparser.ConfigParser()
    cfg.read(path)
    if not cfg.has_section('eSim'):
        cfg.add_section('eSim')
    # eSim_HOME is authoritative for this install; rewrite it every launch
    # so a moved/upgraded install self-heals. Keys mirror install-eSim.sh.
    cfg.set('eSim', 'eSim_HOME', esim_root)
    cfg.set('eSim', 'LICENSE', '%(eSim_HOME)s/LICENSE')
    cfg.set('eSim', 'KicadLib', '%(eSim_HOME)s/library/kicadLibrary')
    cfg.set('eSim', 'IMAGES', '%(eSim_HOME)s/images')
    cfg.set('eSim', 'VERSION', '%(eSim_HOME)s/VERSION')
    cfg.set('eSim', 'MODELICA_MAP_JSON',
            '%(eSim_HOME)s/library/ngspicetoModelica/Mapping.json')
    with open(path, 'w') as fh:
        cfg.write(fh)
    return path


def seed_generated_symbols(esim_root):
    """Copy the 3 runtime-rewritten symbol libs to ~/.esim/kicad_symbols once.
    Never overwrite: the user's built models accumulate inside these files."""
    src_dir = os.path.join(esim_root, 'library', 'kicadLibrary',
                           'eSim-symbols')
    dst_dir = _user_dir('.esim', 'kicad_symbols')
    os.makedirs(dst_dir, exist_ok=True)
    for lib in GENERATED_LIBS:
        src = os.path.join(src_dir, lib + '.kicad_sym')
        dst = os.path.join(dst_dir, lib + '.kicad_sym')
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
    return dst_dir


def write_nghdl_config(esim_root):
    """~/.nghdl/config.ini — every key the simulation flows read, mirroring
    what nghdl/install-nghdl.sh writes on Ubuntu (absolute paths instead of
    %()s interpolation so a configparser round-trip stays lossless):

      [NGHDL]    NGHDL_HOME    tools/nghdl (the built nghdl-simulator tree)
                 DIGITAL_MODEL <NGHDL_HOME>/src/xspice/icm  (model sources
                               land here at build time)
                 RELEASE       <NGHDL_HOME>/release (configured build tree
                               where runtime `make` rebuilds the .cm)
      [SRC]      SRC_HOME      <root>/nghdl (ngspice_ghdl.py resolves its
                               ghdlserver sources under SRC_HOME/src/)
                 LICENSE       <SRC_HOME>/LICENSE
      [COMPILER] MSYS_HOME     tools/msys64 (mingw32-make/gcc/verilator/ghdl)
      [COSIM]    IVERILOG/VVP/IVERILOG_LIB — the bundled libvvp Icarus

    Sections are written only for the pieces actually installed, so the
    in-app doctor reports the truth for Compact installs."""
    nghdl_home = os.path.join(esim_root, 'tools', 'nghdl')
    msys_home = os.path.join(esim_root, 'tools', 'msys64')
    if not (os.path.isdir(nghdl_home) or os.path.isdir(msys_home)):
        return None
    cfg_dir = _user_dir('.nghdl')
    os.makedirs(cfg_dir, exist_ok=True)
    path = os.path.join(cfg_dir, 'config.ini')
    cfg = configparser.ConfigParser()
    cfg.read(path)

    if os.path.isdir(nghdl_home):
        for section in ('NGHDL', 'SRC'):
            if not cfg.has_section(section):
                cfg.add_section(section)
        cfg.set('NGHDL', 'NGHDL_HOME', nghdl_home)
        cfg.set('NGHDL', 'DIGITAL_MODEL',
                os.path.join(nghdl_home, 'src', 'xspice', 'icm'))
        cfg.set('NGHDL', 'RELEASE', os.path.join(nghdl_home, 'release'))
        src_home = os.path.join(esim_root, 'nghdl')
        cfg.set('SRC', 'SRC_HOME', src_home)
        cfg.set('SRC', 'LICENSE', os.path.join(src_home, 'LICENSE'))

    if os.path.isdir(msys_home):
        if not cfg.has_section('COMPILER'):
            cfg.add_section('COMPILER')
        cfg.set('COMPILER', 'MSYS_HOME', msys_home)

    iv_root = os.path.join(esim_root, 'library', 'bin', 'iverilog')
    iv_bin = os.path.join(iv_root, 'bin', 'iverilog.exe')
    if os.path.isfile(iv_bin):
        if not cfg.has_section('COSIM'):
            cfg.add_section('COSIM')
        cfg.set('COSIM', 'IVERILOG', iv_bin)
        cfg.set('COSIM', 'VVP', os.path.join(iv_root, 'bin', 'vvp.exe'))
        # mingw puts runtime DLLs in bin/, POSIX-style builds in lib/; record
        # whichever holds libvvp so ngspice's ivlng adapter can dlopen it.
        for sub in ('bin', 'lib'):
            libdir = os.path.join(iv_root, sub)
            try:
                names = os.listdir(libdir)
            except OSError:
                continue
            if any(n.startswith('libvvp') or n == 'vvp.dll' for n in names):
                cfg.set('COSIM', 'IVERILOG_LIB', libdir)
                break

    with open(path, 'w') as fh:
        cfg.write(fh)
    return path


def fix_spinit(esim_root):
    """Self-heal the bundled ngspice's spinit for THIS install location.

    spinit's `codemodel` lines carry absolute paths baked in at build time
    (the build machine's staging dir). ngspice loads .cm code models ONLY
    through these lines, so after installation they must point at
    <esim_root>/tools/nghdl/install_dir/lib/ngspice. Idempotent: rewriting an
    already-correct file is a no-op. Returns the number of rewritten lines."""
    spinit = os.path.join(esim_root, 'tools', 'nghdl', 'install_dir',
                          'share', 'ngspice', 'scripts', 'spinit')
    cmdir = os.path.join(esim_root, 'tools', 'nghdl', 'install_dir',
                         'lib', 'ngspice')
    if not (os.path.isfile(spinit) and os.path.isdir(cmdir)):
        return 0
    pattern = re.compile(
        r'^(\s*codemodel\s+).*[/\\]([^/\\]+\.cm)\s*$', re.IGNORECASE)
    changed = 0
    out_lines = []
    with open(spinit, 'r', errors='replace') as fh:
        for line in fh:
            m = pattern.match(line)
            if m:
                target = os.path.join(cmdir, m.group(2)).replace('\\', '/')
                new = '%s%s\n' % (m.group(1), target)
                if new != line:
                    changed += 1
                line = new
            out_lines.append(line)
    if changed:
        with open(spinit, 'w') as fh:
            fh.writelines(out_lines)
    return changed


def ensure_unversioned_libvvp(esim_root):
    """Self-heal the libvvp DLL name for ngspice's d_cosim ivlng adapter.

    ivlng does LoadLibrary("libvvp") -- the UNVERSIONED name (eSim netlists
    pass no lib_args) -- but the MinGW iverilog build ships only
    libvvp-<N>.dll. Keep a plain libvvp.dll copy beside it. Idempotent;
    returns True when the plain name exists afterwards."""
    bindir = os.path.join(esim_root, 'library', 'bin', 'iverilog', 'bin')
    plain = os.path.join(bindir, 'libvvp.dll')
    if os.path.isfile(plain):
        return True
    if not os.path.isdir(bindir):
        return False
    for name in sorted(os.listdir(bindir)):
        if name.startswith('libvvp-') and name.endswith('.dll'):
            shutil.copy2(os.path.join(bindir, name), plain)
            return True
    return False


def _kicad_config_root():
    if os.name == 'nt':
        return os.path.join(os.environ.get('APPDATA', ''), 'kicad')
    return _user_dir('.config', 'kicad')


def register_kicad_libraries(esim_root):
    """Register every eSim library in the user's KiCad sym-lib-table(s):
    static libs -> <install>/library/kicadLibrary/eSim-symbols, generated
    libs -> ~/.esim/kicad_symbols. Reuses kicad_symlib.ensure_lib_registered
    (append-or-rewrite, idempotent, best-effort).

    On Windows eSim does NOT copy static libs into KiCad's own install tree
    (that was the old tribal-installer behaviour); they are referenced in
    place from the eSim install dir. KiCad's standard libraries are never
    touched.

    If KiCad has never been started (%APPDATA%/kicad absent) there is nothing
    to register into yet; runtime ensure_lib_registered calls in the model
    builders cover the generated libs later, and this function runs again on
    every eSim launch anyway."""
    sys.path.insert(0, os.path.join(esim_root, 'src', 'maker'))
    try:
        from kicad_symlib import ensure_lib_registered  # noqa: E402
    except ImportError:
        return False
    finally:
        sys.path.pop(0)

    base = _kicad_config_root()
    if not os.path.isdir(base):
        return False

    static_dir = os.path.join(esim_root, 'library', 'kicadLibrary',
                              'eSim-symbols')
    gen_dir = _user_dir('.esim', 'kicad_symbols')
    if os.path.isdir(static_dir):
        for fname in sorted(os.listdir(static_dir)):
            m = re.fullmatch(r'(eSim_\w+)\.kicad_sym', fname)
            if not m:
                continue
            lib = m.group(1)
            target = (os.path.join(gen_dir, fname) if lib in GENERATED_LIBS
                      else os.path.join(static_dir, fname))
            ensure_lib_registered(lib, target)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--esim-root', default=esim_root_default())
    args = ap.parse_args(argv)
    root = os.path.abspath(args.esim_root)

    write_esim_config(root)
    seed_generated_symbols(root)
    write_nghdl_config(root)
    fix_spinit(root)
    ensure_unversioned_libvvp(root)
    register_kicad_libraries(root)
    return 0


if __name__ == '__main__':
    sys.exit(main())
