# =========================================================================
#             FILE: verilog_library.py
#
#      DESCRIPTION: The user's Verilog designs on disk, under
#                   <workspace>/VerilogLibrary/<module>/.
#
#                   Why this exists: a design authored inside eSim used to have
#                   nowhere to live. The Author editor could not create a file
#                   at all (Save with no path just failed), and the Verify stage
#                   invented a path under a hardcoded ~/eSim-Workspace that
#                   ignored the workspace the user actually picked. So the only
#                   designs that survived a session were the ones written in
#                   some other editor and opened read-only.
#
#                   Now every design eSim holds has a home, named after its top
#                   module, written without the user asking. That home is also
#                   what Convert builds from, so the library is not a second
#                   copy of anything -- it IS the design.
#
#                   Pure stdlib + hdl.ports (no Qt), so the naming and layout
#                   rules can be tested without driving a GUI.
#
#  ORGANIZATION: eSim Team at FOSSEE, IIT Bombay
# =========================================================================
import os
import re
import shutil
import time

from configuration import paths
from .hdl.ports import find_modules, strip_comments, top_module_name

#: Folder under the workspace that holds every design.
LIBRARY_DIRNAME = "VerilogLibrary"

#: Per-design subfolder holding Convert-time snapshots.
HISTORY_DIRNAME = ".history"

_ENDMODULE_RE = re.compile(r'\bendmodule\b')


def library_root():
    """``<workspace>/VerilogLibrary``, from the workspace the user chose.

    Anchored to ``paths.read_workspace`` rather than a hardcoded home, so
    designs land beside the projects they belong to and follow the user when
    they point eSim at a different workspace."""
    _check, workspace = paths.read_workspace()
    return os.path.join(workspace, LIBRARY_DIRNAME)


def top_module(code):
    """Name of the design's top module, or '' when nothing parses."""
    return top_module_name(code)


def is_saveable(code):
    """True when ``code`` is worth writing to disk under its own name.

    Guards the autosave against two things that would otherwise litter the
    library. A design is only saveable when it names a top module *and* closes
    every module it opens -- so a design being typed one character at a time
    ("module n", "module na", "module nan") never earns a folder, and neither
    does a paste that is still half-selected. Until this returns True the text
    is perfectly safe: it lives in the DesignBus, in memory."""
    if not code or not code.strip():
        return False
    clean = strip_comments(code)
    modules = find_modules(clean)
    if not modules:
        return False
    if len(_ENDMODULE_RE.findall(clean)) < len(modules):
        return False
    return bool(top_module(code))


def design_dir(module, root=None):
    """``<library>/<module>`` for a design named ``module``."""
    return os.path.join(root or library_root(), str(module))


def design_path(module, root=None):
    """``<library>/<module>/<module>.v`` -- where the design itself lives.

    One folder per design rather than one flat pile of .v files: a design is
    usually more than one file (its testbench, and any helper modules), and two
    unrelated designs that each define a ``mux`` helper would otherwise
    overwrite each other."""
    return os.path.join(design_dir(module, root), str(module) + ".v")


def sibling_path(module, filename, root=None):
    """A file that belongs to ``module``'s design but is not the design itself
    -- its testbench, or a helper module. Kept beside it so the whole design is
    one folder to copy."""
    return os.path.join(design_dir(module, root), filename)


def save_design(code, root=None):
    """Write ``code`` to its own design file and return the path.

    Returns "" (writing nothing) when the code is not saveable yet, which is
    what makes this callable straight off a "text changed" signal."""
    if not is_saveable(code):
        return ""
    module = top_module(code)
    target = design_path(module, root)
    return write_text(target, code)


def write_text(target, text):
    """Create the parent folder and write ``text``. "" on any OSError -- a
    library write is a convenience, never a reason to interrupt the user."""
    try:
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(target, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
    except OSError:
        return ""
    return target


def snapshot(module, code, root=None, stamp=None):
    """Keep a dated copy of what a Convert was run on, in ``.history/``.

    Deliberately NOT called on every autosave: snapshotting keystrokes buries
    the folder in hundreds of near-identical files. One entry per build is a
    record of the versions that actually became models -- which is the version
    someone writing this up needs to point at."""
    if not module:
        return ""
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    target = os.path.join(design_dir(module, root), HISTORY_DIRNAME,
                          "%s-%s.v" % (module, stamp))
    return write_text(target, code)


def list_designs(root=None):
    """``[(name, path, mtime), ...]`` for every design in the library, most
    recently touched first. Folders with no design file are skipped, so a
    half-removed leftover never shows up as something the user can open."""
    base = root or library_root()
    found = []
    try:
        entries = os.listdir(base)
    except OSError:
        return found
    for name in entries:
        path = design_path(name, base)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        found.append((name, path, mtime))
    found.sort(key=lambda item: item[2], reverse=True)
    return found


def remove_design(name, root=None):
    """Delete one design's whole folder. True when something was removed.

    Refuses a blank or path-bearing name: os.path.join(base, "") is base
    itself, and rmtree on that would take every design the user has."""
    if not name or not str(name).strip():
        return False
    safe = str(name).strip()
    if os.path.basename(safe) != safe or safe in (os.curdir, os.pardir):
        return False
    target = design_dir(safe, root)
    try:
        shutil.rmtree(target)
    except (OSError, FileNotFoundError):
        return False
    return True
