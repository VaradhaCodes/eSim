# =============================================================================
#  test_stem_from_input.py -- Seam B regression guard. The kicadtoNgspice tabs
#  (Source/Analysis/Model/Microcontroller/DeviceModel/SubcircuitTab/
#  KicadtoNgspice) are handed the project's .cir and must derive the stem from
#  that filename, NOT from the enclosing folder's name. This reproduces exactly
#  what those tabs compute, without needing a Qt app.
# =============================================================================
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(os.path.dirname(HERE))          # .../src
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from projManagement.projectPaths import stem_from_file   # noqa: E402


def test_previous_values_path_uses_stem_not_folder():
    with tempfile.TemporaryDirectory() as tmp:
        # Folder name deliberately != project stem (IP-Library shape).
        d = os.path.join(tmp, 'eSim_Project_Files')
        os.makedirs(d)
        kicadFile = os.path.join(d, 'MACProject.cir')

        # Exactly what the tabs now do:
        (projpath, filename) = os.path.split(kicadFile)
        project_name = stem_from_file(kicadFile)
        prev = os.path.join(projpath, project_name + '_Previous_Values.xml')

        assert project_name == 'MACProject'
        assert prev == os.path.join(d, 'MACProject_Previous_Values.xml')
        # The old folder-name behaviour would have produced this -- guard it.
        assert not prev.endswith('eSim_Project_Files_Previous_Values.xml')
