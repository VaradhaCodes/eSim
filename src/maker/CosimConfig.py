# ==============================================================================
#             FILE: CosimConfig.py
#
#      DESCRIPTION: Locate the d_cosim (Icarus Verilog) co-simulation toolchain
#                   without any hardcoded paths, on Linux and Windows, wherever
#                   eSim happens to be installed.
#
#                   d_cosim needs two things at simulation time:
#                     1. an ngspice >= 44 built with the `ivlng` adapter (eSim's
#                        bundled nghdl-simulator already ships ivlng.so/.vpi);
#                     2. iverilog built with libvvp (`--enable-libvvp`), which
#                        ngspice's ivlng adapter dlopens at runtime, plus the
#                        `iverilog` compiler used to build the per-model vvp.
#
#                   Every path is resolved at call time in this order:
#                       explicit env override
#                     -> ~/.nghdl/config.ini  ([COSIM] / [NGHDL])
#                     -> PATH / standard install location relative to the binary
#                     -> None (caller shows a clear "not installed" message).
#
#                   Nothing here is eSim-install-location specific: the installer
#                   records the real paths in config.ini, and developers can
#                   override with ESIM_NGSPICE / ESIM_IVERILOG / ESIM_IVERILOG_LIB.
#
#  ORGANIZATION: eSim Team at FOSSEE, IIT Bombay
# ==============================================================================

import os
import shutil
from configparser import ConfigParser, Error as ConfigError

_WIN = os.name == 'nt'
_EXE = '.exe' if _WIN else ''

# Cached capability probe (filesystem checks only; safe to memoize per process).
_dcosim_capable = None


def _config_path():
    """Path to eSim's nghdl config.ini, mirroring ModelGeneration/NgVeri:
    Windows keeps it under library/config, POSIX under the user home."""
    home = os.path.join('library', 'config') if _WIN else os.path.expanduser('~')
    return os.path.join(home, '.nghdl', 'config.ini')


def _cfg_get(section, key):
    """Read one config.ini value, or None if absent/unreadable."""
    parser = ConfigParser()
    try:
        parser.read(_config_path())
        value = parser.get(section, key)
        return value or None
    except (ConfigError, OSError):
        return None


def _prefix_of(binary):
    """<prefix> for a <prefix>/bin/<tool> path, following symlinks first so a
    /usr/bin/ngspice symlink resolves to the real install tree."""
    real = os.path.realpath(binary)
    return os.path.dirname(os.path.dirname(real))


# --------------------------------------------------------------------------- #
#  ngspice (the eSim / nghdl-simulator build that carries d_cosim + ivlng)
# --------------------------------------------------------------------------- #
def ngspice_binary():
    """Resolve eSim's ngspice executable. Used for ALL simulations (the bundled
    nghdl-simulator build is the one that has d_cosim + ivlng)."""
    env = os.environ.get('ESIM_NGSPICE')
    if env and os.path.isfile(env):
        return env
    nghdl_home = _cfg_get('NGHDL', 'NGHDL_HOME')
    if nghdl_home:
        cand = os.path.join(nghdl_home, 'install_dir', 'bin', 'ngspice' + _EXE)
        if os.path.isfile(cand):
            return cand
    return shutil.which('ngspice') or 'ngspice'


def ngspice_codemodel_dir():
    """Directory holding ngspice's *.cm code models + the ivlng adapter
    (<prefix>/lib/ngspice), or None."""
    cand = os.path.join(_prefix_of(ngspice_binary()), 'lib', 'ngspice')
    return cand if os.path.isdir(cand) else None


def cosim_vvp_path(model_name):
    """Canonical location of a model's compiled Icarus vvp, derived from the
    nghdl config so the build step (ModelGeneration.build_cosim) and the
    netlister (Convert) agree on ONE path without storing it anywhere. Mirrors
    ModelGeneration's per-model store: <DIGITAL_MODEL>/Ngveri/<model>/<model>.
    Returns None if the config is unavailable."""
    digital_model = _cfg_get('NGHDL', 'DIGITAL_MODEL')
    if not digital_model:
        return None
    return os.path.join(digital_model, 'Ngveri', model_name, model_name)


# --------------------------------------------------------------------------- #
#  iverilog (with libvvp)
# --------------------------------------------------------------------------- #
def iverilog_binary():
    """Resolve the iverilog compiler, or None if not installed."""
    env = os.environ.get('ESIM_IVERILOG')
    if env and os.path.isfile(env):
        return env
    cfg = _cfg_get('COSIM', 'IVERILOG')
    if cfg and os.path.isfile(cfg):
        return cfg
    return shutil.which('iverilog')


def iverilog_libdir():
    """Resolve the dir containing libvvp (ngspice's ivlng dlopens it), or None.
    Falls back to <iverilog_prefix>/lib derived from the compiler path."""
    env = os.environ.get('ESIM_IVERILOG_LIB')
    if env and os.path.isdir(env):
        return env
    cfg = _cfg_get('COSIM', 'IVERILOG_LIB')
    if cfg and os.path.isdir(cfg):
        return cfg
    binary = iverilog_binary()
    if binary:
        cand = os.path.join(_prefix_of(binary), 'lib')
        if os.path.isdir(cand):
            return cand
    return None


def _has_libvvp(libdir):
    """True if a libvvp shared object is present in libdir (any OS naming)."""
    if not libdir or not os.path.isdir(libdir):
        return False
    try:
        return any(name.startswith('libvvp') or name == 'vvp.dll'
                   for name in os.listdir(libdir))
    except OSError:
        return False


# --------------------------------------------------------------------------- #
#  Capability gates (used by the UI / runtime to fail clearly, never crash)
# --------------------------------------------------------------------------- #
def has_iverilog():
    """True if iverilog AND its libvvp are available (needed to BUILD a model
    and for ivlng to run it)."""
    return bool(iverilog_binary()) and _has_libvvp(iverilog_libdir())


def has_dcosim(force=False):
    """True if a full d_cosim run is possible here: ngspice carries the ivlng
    adapter AND iverilog/libvvp are present. Cached after first call."""
    global _dcosim_capable
    if _dcosim_capable is None or force:
        cmdir = ngspice_codemodel_dir()
        ngspice_ok = bool(cmdir) and any(
            name.startswith('ivlng') for name in os.listdir(cmdir))
        _dcosim_capable = ngspice_ok and has_iverilog()
    return _dcosim_capable


def loader_path_var():
    """Name of the dynamic-loader search-path env var for this OS (so ngspice's
    ivlng adapter can find libvvp): PATH on Windows, LD_LIBRARY_PATH elsewhere."""
    return 'PATH' if _WIN else 'LD_LIBRARY_PATH'


def missing_reason():
    """Human-readable reason d_cosim is unavailable, for UI messages. Empty
    string when everything is present."""
    if not has_iverilog():
        if not iverilog_binary():
            return ("iverilog not found. d_cosim needs Icarus Verilog built "
                    "with libvvp (--enable-libvvp).")
        return ("libvvp not found next to iverilog. Rebuild Icarus Verilog "
                "with --enable-libvvp.")
    cmdir = ngspice_codemodel_dir()
    if not (cmdir and any(n.startswith('ivlng') for n in os.listdir(cmdir))):
        return ("This ngspice build has no ivlng adapter. d_cosim needs the "
                "eSim/nghdl-simulator ngspice (>= 44, --enable-xspice).")
    return ""
