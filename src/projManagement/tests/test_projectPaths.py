# =============================================================================
#  test_projectPaths.py -- unit tests for the project-identity helpers that
#  decouple a project's stem from its folder name. Pure functions on temp dirs;
#  no Qt, no kicad-cli.
# =============================================================================
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(os.path.dirname(HERE))          # .../src
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from projManagement.projectPaths import (             # noqa: E402
    find_anchors, resolve_stem, stem_from_file, main_schematic,
)


def _touch(path):
    with open(path, 'w') as fh:
        fh.write('')


def _mkproj(root, folder, stem, proj_body=None, files=('cir', 'sch')):
    """Create <root>/<folder>/<stem>.<ext...> and return the folder path."""
    d = os.path.join(root, folder)
    os.makedirs(d)
    proj = os.path.join(d, stem + '.proj')
    with open(proj, 'w') as fh:
        fh.write(proj_body if proj_body is not None
                 else 'schematicFile ' + stem + '.kicad_sch\n')
    for ext in files:
        _touch(os.path.join(d, stem + '.' + ext))
    return d


# --------------------------------------------------------------------------- #
#  stem_from_file
# --------------------------------------------------------------------------- #
def test_stem_from_file_strips_single_extension():
    assert stem_from_file('/ws/weird_folder/MACProject.cir') == 'MACProject'
    assert stem_from_file('MACProject.sch') == 'MACProject'
    assert stem_from_file('/a/b/handshake.kicad_sch') == 'handshake'


# --------------------------------------------------------------------------- #
#  find_anchors
# --------------------------------------------------------------------------- #
def test_find_anchors_lists_only_matching_ext():
    with tempfile.TemporaryDirectory() as tmp:
        d = _mkproj(tmp, 'eSim_Project_Files', 'MACProject')
        anchors = find_anchors(d, 'proj')
        assert [os.path.basename(a) for a in anchors] == ['MACProject.proj']


def test_find_anchors_empty_and_missing_dir():
    with tempfile.TemporaryDirectory() as tmp:
        empty = os.path.join(tmp, 'empty')
        os.makedirs(empty)
        assert find_anchors(empty, 'proj') == []
    assert find_anchors('/no/such/dir', 'proj') == []
    assert find_anchors(None, 'proj') == []


# --------------------------------------------------------------------------- #
#  resolve_stem -- the core of the fix
# --------------------------------------------------------------------------- #
def test_resolve_stem_folder_name_differs_from_stem():
    # The reported bug: folder 'eSim_Project_Files' holds MACProject.proj.
    with tempfile.TemporaryDirectory() as tmp:
        d = _mkproj(tmp, 'eSim_Project_Files', 'MACProject')
        assert resolve_stem(d, 'proj') == ('MACProject', 'ok')


def test_resolve_stem_legacy_folder_equals_stem():
    with tempfile.TemporaryDirectory() as tmp:
        d = _mkproj(tmp, 'MACProject', 'MACProject')
        assert resolve_stem(d, 'proj') == ('MACProject', 'ok')


def test_resolve_stem_missing_falls_back_to_folder():
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, 'not_a_project')
        os.makedirs(d)
        assert resolve_stem(d, 'proj') == ('not_a_project', 'missing')


def test_resolve_stem_ambiguous_prefers_folder_match():
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, 'alpha')
        os.makedirs(d)
        _touch(os.path.join(d, 'alpha.proj'))
        _touch(os.path.join(d, 'beta.proj'))
        assert resolve_stem(d, 'proj') == ('alpha', 'ambiguous')


def test_resolve_stem_ambiguous_no_match_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, 'weird')
        os.makedirs(d)
        _touch(os.path.join(d, 'one.proj'))
        _touch(os.path.join(d, 'two.proj'))
        assert resolve_stem(d, 'proj') == (None, 'ambiguous')


def test_resolve_stem_subcircuit_uses_sub_anchor():
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, 'odd_sub_folder')
        os.makedirs(d)
        _touch(os.path.join(d, 'my_subckt.sub'))
        assert resolve_stem(d, 'sub') == ('my_subckt', 'ok')


# --------------------------------------------------------------------------- #
#  main_schematic
# --------------------------------------------------------------------------- #
def test_main_schematic_prefers_proj_pointer():
    with tempfile.TemporaryDirectory() as tmp:
        d = _mkproj(tmp, 'folderX', 'MACProject',
                    proj_body='schematicFile MACProject.kicad_sch\n',
                    files=('kicad_sch',))
        got = main_schematic(d, 'MACProject')
        assert got == os.path.join(d, 'MACProject.kicad_sch')


def test_main_schematic_falls_back_to_stem_sch():
    with tempfile.TemporaryDirectory() as tmp:
        # .proj points at a kicad_sch that does not exist; only <stem>.sch does.
        d = _mkproj(tmp, 'folderY', 'demo',
                    proj_body='schematicFile demo.kicad_sch\n',
                    files=('sch',))
        got = main_schematic(d, 'demo')
        assert got == os.path.join(d, 'demo.sch')


def test_main_schematic_missing_returns_best_guess_kicad_sch():
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, 'folderZ')
        os.makedirs(d)
        got = main_schematic(d, 'demo')
        assert got == os.path.join(d, 'demo.kicad_sch')
