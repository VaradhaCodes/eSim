# =============================================================================
#  test_open_validation.py -- regression tests for the reported bug: opening a
#  project whose folder name differs from the files inside it. Validation has no
#  Qt dependency, so it imports cleanly here.
# =============================================================================
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(os.path.dirname(HERE))          # .../src
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from projManagement.Validation import Validation       # noqa: E402


def _touch(path):
    with open(path, 'w') as fh:
        fh.write('')


def _make_project(root, folder, stem):
    """Folder named <folder> containing <stem>.proj/.cir/.cir.out/.sch."""
    d = os.path.join(root, folder)
    os.makedirs(d)
    for ext in ('proj', 'cir', 'cir.out', 'sch'):
        _touch(os.path.join(d, stem + '.' + ext))
    return d


def test_validateOpenproj_true_when_folder_name_differs():
    # Before the fix this returned False because it looked for
    # <folder>.proj. Now any .proj anchor makes it a valid project.
    v = Validation()
    with tempfile.TemporaryDirectory() as tmp:
        d = _make_project(tmp, 'eSim_Project_Files', 'MACProject')
        assert v.validateOpenproj(d) is True


def test_validateOpenproj_false_without_proj():
    v = Validation()
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, 'plain')
        os.makedirs(d)
        _touch(os.path.join(d, 'something.cir'))   # .cir but no .proj
        assert v.validateOpenproj(d) is False


def test_validateCir_resolves_stem_not_folder():
    v = Validation()
    with tempfile.TemporaryDirectory() as tmp:
        d = _make_project(tmp, 'eSim_Project_Files', 'MACProject')
        # No stem passed -> must resolve MACProject from the .proj anchor.
        assert v.validateCir(d) is True
        # Explicit stem also honoured.
        assert v.validateCir(d, 'MACProject') is True
        assert v.validateCir(d, 'WrongName') is False


def test_validateCirOut_resolves_stem_not_folder():
    v = Validation()
    with tempfile.TemporaryDirectory() as tmp:
        d = _make_project(tmp, 'weird_folder', 'MACProject')
        assert v.validateCirOut(d) is True
        assert v.validateCirOut(d, 'MACProject') is True


def test_validateSub_resolves_stem_not_folder():
    v = Validation()
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, 'odd_sub_folder')
        os.makedirs(d)
        # A 2-port subckt whose .sub name differs from the folder.
        with open(os.path.join(d, 'my_subckt.sub'), 'w') as fh:
            fh.write('.subckt my_subckt 1 2\n.ends my_subckt\n')
        assert v.validateSub(d, 2) == "True"
        assert v.validateSub(d, 3) == "PORT"
