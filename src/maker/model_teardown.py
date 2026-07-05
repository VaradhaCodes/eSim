# =========================================================================
#             FILE: model_teardown.py
#
#      DESCRIPTION: Pure, dependency-free helpers for removing generated block
#                   models (NgVeri / d_cosim / NGHDL) from the shared ngspice
#                   install. Kept out of NgVeri.py (which imports Qt + hdlparse
#                   via Maker) so the teardown core -- the part that does the
#                   irreversible file surgery -- is unit-testable in isolation
#                   with nothing but stdlib.
#
#            NOTES: stdlib only (os). Do NOT add Qt / config / hdlparse imports
#                   here; that is the whole point of the split.
#
#  ORGANIZATION: eSim Team at FOSSEE, IIT Bombay
# =========================================================================

import os


def _safe_model_subdir(base, name):
    """Resolve ``<base>/<name>`` for deletion, but ONLY when it is provably a
    single-component subdirectory strictly inside ``base``.

    Returns the absolute path, or ``None`` when ``name`` is empty/blank, holds a
    path separator, is ``.``/``..``, or resolves to ``base`` itself or outside
    it. Callers MUST treat ``None`` as "do not delete anything".

    This is the guard that stops a blank model name from collapsing
    ``os.path.join(base, "")`` to ``"base/"`` and ``shutil.rmtree`` wiping the
    whole models directory.
    """
    if not name or not str(name).strip():
        return None
    name = str(name).strip()
    if (os.sep in name or (os.altsep and os.altsep in name)
            or name in ('.', '..')):
        return None
    base_abs = os.path.abspath(base)
    target = os.path.abspath(os.path.join(base_abs, name))
    if target == base_abs:
        return None
    try:
        if os.path.commonpath([base_abs, target]) != base_abs:
            return None
    except ValueError:
        # Different drives (Windows) -> not a subpath.
        return None
    return target


def _strip_modpath_line(path, name):
    """Remove every line equal to ``name`` from the modpath.lst at ``path``
    (idempotent). Returns True if a line was dropped. Safe when the file is
    absent or ``name`` is blank (both -> no change)."""
    name = (name or "").strip()
    if not name:
        return False
    try:
        with open(path) as f:
            lines = f.readlines()
    except OSError:
        return False
    kept = [ln for ln in lines if ln.strip() != name]
    if len(kept) == len(lines):
        return False
    with open(path, 'w') as f:
        f.writelines(kept)
    return True


def _prune_modpath(path, base, marker="ifspec.ifs"):
    """Rewrite the modpath.lst at ``path`` keeping only entries whose build dir
    ``<base>/<name>/<marker>`` still exists, de-duplicated. Returns the dropped
    (ghost / duplicate) names. A single orphaned entry makes cmpp abort the
    WHOLE code-model build, so this runs before every .cm rebuild."""
    try:
        with open(path) as f:
            entries = [ln.strip() for ln in f]
    except OSError:
        return []
    kept, dropped, seen = [], [], set()
    for name in entries:
        if not name:
            continue
        if name in seen:
            dropped.append(name)            # duplicate line
            continue
        if os.path.isfile(os.path.join(base, name, marker)):
            kept.append(name)
            seen.add(name)
        else:
            dropped.append(name)            # ghost: build dir / marker gone
    if dropped:
        with open(path, 'w') as f:
            for name in kept:
                f.write(name + "\n")
    return dropped


def _resolve_backend(xml_loc, name):
    """Resolve which backend owns model ``name`` from the on-disk modelParamXML
    layout -- the single source of truth that survives restarts and backend
    switches::

        NgVeriCosim/<name>.xml -> "cosim"   (Icarus d_cosim)
        Nghdl/<name>.xml       -> "nghdl"   (GHDL/NGHDL)
        otherwise              -> "ngveri"  (legacy Verilator)

    cosim wins over nghdl wins over ngveri if xml files ever coexist (they
    should not -- one name, one backend -- but a deterministic precedence keeps
    teardown from running the wrong dismantler and silently leaving a model)."""
    if not xml_loc:
        return "ngveri"
    if os.path.isfile(os.path.join(xml_loc, 'NgVeriCosim', name + '.xml')):
        return "cosim"
    if os.path.isfile(os.path.join(xml_loc, 'Nghdl', name + '.xml')):
        return "nghdl"
    return "ngveri"


def _nghdl_sym_path(src_home):
    """Absolute path of the shared eSim_Nghdl.kicad_sym, in eSim's generated
    symbol-lib dir (~/.esim/kicad_symbols) -- mirroring where createkicad now
    puts eSim_Ngveri. Lazily migrates a legacy /usr/share (or old Windows)
    copy in on first use so existing users keep their accumulated models.

    kicad_symlib is stdlib-only, so importing it does not break this module's
    "no Qt/config/hdlparse" contract. The dual import handles both layouts:
    packaged (maker.*) and the flat vendored NGHDL tarball."""
    try:
        from .kicad_symlib import generated_symlib_path
    except ImportError:                 # flat vendored layout (NGHDL package)
        from kicad_symlib import generated_symlib_path
    legacy = []
    if os.name == 'nt':
        inst_dir = (src_home or "").replace('\\eSim', '')
        legacy.append(inst_dir + '/KiCad/share/kicad/symbols')
    return generated_symlib_path("eSim_Nghdl", legacy_dirs=legacy)
