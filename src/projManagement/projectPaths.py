# =========================================================================
#          FILE: projectPaths.py
#
#   DESCRIPTION: Single source of truth for an eSim project's identity.
#
#                eSim historically derived a project's "stem" (the basename
#                shared by <stem>.proj/.cir/.sch/.cir.out/.sub/... ) from the
#                *folder name*. That breaks whenever the folder is renamed or
#                differs from the files inside it (e.g. IP-Library circuits
#                shipped in a folder named "eSim_Project_Files" containing
#                MACProject.*). The real, unique anchor is the project file
#                itself: exactly one ``*.proj`` per project folder (subcircuits
#                use ``*.sub``). These helpers resolve the stem from that anchor
#                instead of the folder name, so the folder may be named anything.
#
#                Dependency-free (os, glob) on purpose: imported from
#                Validation, Appconfig and the kicadtoNgspice tabs without
#                pulling in Qt.
#
#  ORGANIZATION: eSim Team at FOSSEE, IIT Bombay
# =========================================================================

import os
import glob


def find_anchors(directory, ext='proj'):
    """
    Return the sorted list of anchor files (``*.<ext>``) directly inside
    ``directory``. ``ext`` is given without a leading dot ('proj', 'sub').

    @params
        :directory  => the folder to scan
        :ext        => anchor extension without dot (default 'proj')

    @return
        sorted list of absolute/!relative paths (as given by glob)
    """
    if not directory or not os.path.isdir(directory):
        return []
    return sorted(glob.glob(os.path.join(directory, '*.' + ext)))


def stem_from_file(path):
    """
    Return the stem (basename without its final extension) of a concrete
    file path. Use this in code that is already handed the project's ``.cir``
    (or ``.sch``/``.sub``) file -- the stem is right there in the filename and
    needs no folder lookup.

    Note: strips a single extension, so pass a ``.cir`` (not ``.cir.out``).

    @params
        :path   => a file path such as '/ws/weird_folder/MACProject.cir'

    @return
        the stem, e.g. 'MACProject'
    """
    return os.path.splitext(os.path.basename(str(path)))[0]


def resolve_stem(directory, ext='proj'):
    """
    Resolve the canonical stem of a project/subcircuit folder from its anchor
    file, independent of the folder's own name.

    @params
        :directory  => the project (or subcircuit) folder
        :ext        => anchor extension without dot ('proj' or 'sub')

    @return
        (stem, status) where status is one of:
            'ok'        => exactly one anchor; stem taken from it
            'ambiguous' => more than one anchor. stem is the one whose name
                           matches the folder name if present, else None
                           (caller should prompt the user to choose)
            'missing'   => no anchor found; stem falls back to the folder
                           basename so legacy/malformed projects still limp
                           along instead of hard-failing
    """
    anchors = find_anchors(directory, ext)
    folder = os.path.basename(os.path.normpath(str(directory)))

    if len(anchors) == 1:
        return stem_from_file(anchors[0]), 'ok'

    if len(anchors) > 1:
        for anchor in anchors:
            if stem_from_file(anchor) == folder:
                return folder, 'ambiguous'
        return None, 'ambiguous'

    return folder, 'missing'


def main_schematic(proj_dir, stem):
    """
    Resolve the project's main schematic file.

    Prefers the ``schematicFile`` token recorded inside the ``.proj`` (this is
    the authoritative pointer eSim already writes on project creation), then
    falls back to ``<stem>.kicad_sch`` (KiCad 6+) and finally ``<stem>.sch``
    (KiCad 4). Returns an existing path, or the best-guess ``<stem>.kicad_sch``
    path if nothing exists yet (caller decides how to treat a missing file).

    @params
        :proj_dir   => the project folder
        :stem        => the resolved project stem

    @return
        path to the main schematic (may not exist if the project is incomplete)
    """
    # 1. Authoritative pointer inside the .proj, if readable.
    for proj in find_anchors(proj_dir, 'proj'):
        try:
            with open(proj, 'r') as fh:
                for line in fh:
                    words = line.split()
                    if len(words) >= 2 and words[0] == 'schematicFile':
                        candidate = os.path.join(proj_dir, words[1])
                        if os.path.exists(candidate):
                            return candidate
        except (IOError, OSError):
            pass
        break  # only the first/only .proj is the anchor

    # 2 & 3. Convention-based fallbacks.
    kicad6 = os.path.join(proj_dir, str(stem) + '.kicad_sch')
    kicad4 = os.path.join(proj_dir, str(stem) + '.sch')
    if os.path.exists(kicad6):
        return kicad6
    if os.path.exists(kicad4):
        return kicad4
    return kicad6
