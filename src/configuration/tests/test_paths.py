import os

from configuration import paths


def test_install_resources_are_cwd_independent(tmp_path, monkeypatch):
    expected_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    monkeypatch.chdir(tmp_path)
    assert paths.repo_root() == expected_root
    assert paths.image_path("logo.png") == os.path.join(
        expected_root, "images", "logo.png")
    assert paths.library_path("modelParamXML") == os.path.join(
        expected_root, "library", "modelParamXML")


def test_user_paths_are_uniform_and_expand_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert paths.esim_config_dir() == os.path.join(str(tmp_path), ".esim")
    assert paths.nghdl_config_path() == os.path.join(
        str(tmp_path), ".nghdl", "config.ini")


def test_workspace_round_trip_preserves_spaces(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    workspace = os.path.join(str(tmp_path), "Lab Projects")
    paths.write_workspace(2, workspace)
    assert paths.read_workspace() == ("2", workspace)


def test_workspace_read_falls_back_for_truncated_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    os.makedirs(paths.esim_config_dir())
    with open(paths.esim_config_path("workspace.txt"), "w") as fh:
        fh.write("")
    assert paths.read_workspace("/fallback") == ("0", "/fallback")
