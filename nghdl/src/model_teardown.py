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


def _atomic_write_text(path, data):
    """Replace ``path`` with ``data`` atomically (temp file + os.replace).

    Delegates to kicad_symlib._atomic_write so there is ONE implementation of
    this. kicad_symlib is stdlib-only, so importing it does not break this
    module's "no Qt/config/hdlparse" contract, and the dual import handles both
    layouts exactly like _nghdl_sym_path below: packaged (maker.*) and the flat
    vendored NGHDL tarball.

    modpath.lst is tiny but load-bearing: a crash / full disk part-way through
    a rewrite leaves a truncated list, and cmpp then aborts the WHOLE
    code-model build with an error that points nowhere near here."""
    try:
        from .kicad_symlib import _atomic_write
    except ImportError:                 # flat vendored layout (NGHDL package)
        from kicad_symlib import _atomic_write
    _atomic_write(path, data)


def _ensure_modpath(path):
    """Make sure the modpath.lst at ``path`` exists, together with its parent
    directory, creating an empty list when it does not. Returns ``path``.

    The digital-model root is built lazily -- on a fresh tree it appears only
    once the remove dialog has listed models, or once a build has run -- so
    every reader and appender has to be able to conjure it. Without this, the
    first-ever convert on a fresh install died on a bare FileNotFoundError that
    surfaced as a generic "Error in Ngspice code model generation"."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    # 'a' creates the file when absent and is a no-op when it already exists --
    # unlike 'w', which would truncate a list another process just wrote.
    with open(path, 'a'):
        pass
    return path


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
    _atomic_write_text(path, ''.join(kept))
    return True


def _append_modpath_line(path, name):
    """Append ``name`` to the modpath.lst at ``path`` unless it is already
    listed (idempotent). Returns True only when a line was written; ``name``
    blank -> no change.

    Guarantees the new entry starts on its own line. A hand-edited file whose
    last line lacks a trailing newline would otherwise be glued to the new name
    ("oldnamenewname") -- a single ghost entry like that makes cmpp abort the
    WHOLE ghdl.cm build, the exact failure _prune_modpath exists to clean up."""
    name = (name or "").strip()
    if not name:
        return False
    try:
        with open(path) as f:
            content = f.read()
    except OSError:
        content = ""
    if name in (ln.strip() for ln in content.splitlines()):
        return False
    prefix = "" if (not content or content.endswith("\n")) else "\n"
    with open(path, 'a') as f:
        f.write(prefix + name + "\n")
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
        _atomic_write_text(path, ''.join(name + "\n" for name in kept))
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
