"""Tests for the NGHDL (GHDL) model-removal core.

The teardown logic lives in model_teardown as pure, dependency-free functions
(the GUI callers drag in Qt + hdlparse, so they are not importable in a bare
test env). These tests pin the irreversible bits -- modpath line strip, ghost
prune, backend resolution, the blank/`..` rmtree guard -- plus an end-to-end
add->remove->add->remove cycle that composes those helpers with the shared
kicad_symlib symbol writer exactly as the NGHDL app's _remove_nghdl_models
does. The model_teardown module is byte-identical between src/maker and
nghdl/src (drift-guarded below), so this one suite covers both copies.

All pure file ops: no GHDL, ngspice, Qt, or NGHDL install required.
"""
import filecmp
import os
import shutil

import maker.model_teardown as mt
import maker.kicad_symlib as ksym


def _block(name):
    return f'(symbol "{name}" (pin_names (offset 0)) (property "Ref" "U"))'


# ── _strip_modpath_line ─────────────────────────────────────────────────────

def _write_modpath(path, names):
    with open(path, "w") as f:
        for n in names:
            f.write(n + "\n")


def _read_modpath(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def test_strip_removes_exact_line_only(tmp_path):
    mp = str(tmp_path / "modpath.lst")
    _write_modpath(mp, ["and2", "and2_gate", "mux"])
    assert mt._strip_modpath_line(mp, "and2") is True
    assert _read_modpath(mp) == ["and2_gate", "mux"]   # prefix sibling kept


def test_strip_is_idempotent(tmp_path):
    mp = str(tmp_path / "modpath.lst")
    _write_modpath(mp, ["mux"])
    assert mt._strip_modpath_line(mp, "mux") is True
    assert mt._strip_modpath_line(mp, "mux") is False  # already gone
    assert _read_modpath(mp) == []


def test_strip_blank_name_is_noop(tmp_path):
    mp = str(tmp_path / "modpath.lst")
    _write_modpath(mp, ["mux"])
    assert mt._strip_modpath_line(mp, "   ") is False
    assert mt._strip_modpath_line(mp, "") is False
    assert _read_modpath(mp) == ["mux"]


def test_strip_absent_file_is_noop(tmp_path):
    assert mt._strip_modpath_line(str(tmp_path / "nope.lst"), "mux") is False


# ── _prune_modpath ──────────────────────────────────────────────────────────

def _make_model_dir(base, name, with_marker=True):
    d = os.path.join(base, name)
    os.makedirs(d, exist_ok=True)
    if with_marker:
        open(os.path.join(d, "ifspec.ifs"), "w").close()


def test_prune_drops_ghost_and_duplicate(tmp_path):
    base = str(tmp_path)
    mp = os.path.join(base, "modpath.lst")
    _make_model_dir(base, "live")
    _make_model_dir(base, "halfbuilt", with_marker=False)  # ghost: no ifspec
    _write_modpath(mp, ["live", "halfbuilt", "live", "gone"])
    dropped = mt._prune_modpath(mp, base)
    assert sorted(dropped) == ["gone", "halfbuilt", "live"]  # dup live + ghosts
    assert _read_modpath(mp) == ["live"]


def test_prune_noop_when_all_valid(tmp_path):
    base = str(tmp_path)
    mp = os.path.join(base, "modpath.lst")
    _make_model_dir(base, "a")
    _make_model_dir(base, "b")
    _write_modpath(mp, ["a", "b"])
    assert mt._prune_modpath(mp, base) == []
    assert _read_modpath(mp) == ["a", "b"]


# ── _resolve_backend ────────────────────────────────────────────────────────

def _touch_xml(xml_loc, sub, name):
    d = os.path.join(xml_loc, sub)
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, name + ".xml"), "w").close()


def test_resolve_backend_precedence(tmp_path):
    xml = str(tmp_path)
    assert mt._resolve_backend(xml, "ghost") == "ngveri"     # nothing on disk
    _touch_xml(xml, "Nghdl", "vhd")
    assert mt._resolve_backend(xml, "vhd") == "nghdl"
    _touch_xml(xml, "NgVeriCosim", "ic")
    assert mt._resolve_backend(xml, "ic") == "cosim"
    # cosim wins over nghdl if both ever exist for one name
    _touch_xml(xml, "Nghdl", "both")
    _touch_xml(xml, "NgVeriCosim", "both")
    assert mt._resolve_backend(xml, "both") == "cosim"


def test_resolve_backend_empty_loc_is_ngveri():
    assert mt._resolve_backend("", "anything") == "ngveri"


# ── _safe_model_subdir guard (rmtree-all protection) ────────────────────────

def test_safe_model_subdir_rejects_dangerous_names(tmp_path):
    base = str(tmp_path)
    for bad in ("", "   ", ".", "..", "a/b", "a" + os.sep + "b"):
        assert mt._safe_model_subdir(base, bad) is None, bad


def test_safe_model_subdir_accepts_plain_name(tmp_path):
    base = str(tmp_path)
    got = mt._safe_model_subdir(base, "mux")
    assert got == os.path.join(os.path.abspath(base), "mux")


def test_nghdl_sym_path_posix(tmp_path, monkeypatch):
    # HOME redirected so we never create/touch the real ~/.esim.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: p.replace("~", str(tmp_path), 1))
    p = mt._nghdl_sym_path("")
    assert p.endswith("eSim_Nghdl.kicad_sym")
    assert os.path.join(".esim", "kicad_symbols") in p


# ── end-to-end: add -> remove -> add -> remove stays clean ──────────────────

def _add(env, name):
    """Replicate what an NGHDL upload leaves on disk."""
    sym, xml_dir, mp, ghdl, rel = env
    # modpath line (idempotent, like addingModelInModpath)
    existing = _read_modpath(mp) if os.path.exists(mp) else []
    if name not in existing:
        with open(mp, "a") as f:
            f.write(name + "\n")
    # param XML
    os.makedirs(xml_dir, exist_ok=True)
    open(os.path.join(xml_dir, name + ".xml"), "w").close()
    # symbol block
    parts = ksym._read_parts(sym)
    parts[name] = _block(name)
    ksym._write_lib(sym, parts)
    # source + release dirs with the cmpp marker
    _make_model_dir(ghdl, name)
    _make_model_dir(rel, name)


def _remove(env, name):
    """Replicate the NGHDL app's _remove_nghdl_models using the same helpers."""
    sym, xml_dir, mp, ghdl, rel = env
    mt._strip_modpath_line(mp, name)
    parts = ksym._read_parts(sym)
    if parts.pop(name, None) is not None:
        ksym._write_lib(sym, parts)
    try:
        os.remove(os.path.join(xml_dir, name + ".xml"))
    except FileNotFoundError:
        pass
    for base in (ghdl, rel):
        d = mt._safe_model_subdir(base, name)
        if d:
            shutil.rmtree(d, ignore_errors=True)


def _present(env, name):
    sym, xml_dir, mp, ghdl, rel = env
    return {
        "modpath": name in (_read_modpath(mp) if os.path.exists(mp) else []),
        "xml": os.path.exists(os.path.join(xml_dir, name + ".xml")),
        "symbol": name in ksym._read_parts(sym),
        "src": os.path.isdir(os.path.join(ghdl, name)),
        "rel": os.path.isdir(os.path.join(rel, name)),
    }


def test_add_remove_cycle_is_clean(tmp_path):
    env = (
        str(tmp_path / "eSim_Nghdl.kicad_sym"),
        str(tmp_path / "Nghdl"),
        str(tmp_path / "ghdl" / "modpath.lst"),
        str(tmp_path / "ghdl"),
        str(tmp_path / "release"),
    )
    os.makedirs(os.path.dirname(env[2]), exist_ok=True)
    # A second model that must survive every cycle untouched.
    _add(env, "keeper")

    for _ in range(25):
        _add(env, "churn")
        assert all(_present(env, "churn").values())
        _remove(env, "churn")
        assert not any(_present(env, "churn").values())
        # idempotent second remove
        _remove(env, "churn")
        assert not any(_present(env, "churn").values())
        # bystander intact throughout
        assert all(_present(env, "keeper").values())


# ── drift guard: eSim canonical == NGHDL vendored copy ──────────────────────

def test_vendored_teardown_is_byte_identical():
    canonical = mt.__file__
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(canonical)))
    vendored = os.path.join(repo_root, "nghdl", "src", "model_teardown.py")
    assert os.path.exists(vendored), (
        "NGHDL vendored copy missing: " + vendored)
    assert filecmp.cmp(canonical, vendored, shallow=False), (
        "src/maker/model_teardown.py and nghdl/src/model_teardown.py have "
        "drifted; edit one and copy it verbatim to the other.")


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(mt.__file__)))


def test_vendored_remove_dialog_is_byte_identical():
    # The searchable remove-model picker is shared with the NGHDL standalone app
    # by vendoring (it cannot import eSim). Its dual-layout Dialogs import keeps
    # both copies truly identical, so a plain byte compare guards the drift.
    root = _repo_root()
    canonical = os.path.join(root, "src", "maker", "RemoveItemsDialog.py")
    vendored = os.path.join(root, "nghdl", "src", "RemoveItemsDialog.py")
    assert os.path.exists(vendored), (
        "NGHDL vendored copy missing: " + vendored)
    assert filecmp.cmp(canonical, vendored, shallow=False), (
        "src/maker/RemoveItemsDialog.py and nghdl/src/RemoveItemsDialog.py "
        "have drifted; edit one and copy it verbatim to the other.")


def test_vendored_dialogs_is_byte_identical():
    # RemoveItemsDialog needs Dialogs.warning; NGHDL has no configuration pkg,
    # so Dialogs.py (PyQt6-only) is vendored beside it for the flat-import path.
    root = _repo_root()
    canonical = os.path.join(root, "src", "configuration", "Dialogs.py")
    vendored = os.path.join(root, "nghdl", "src", "Dialogs.py")
    assert os.path.exists(vendored), (
        "NGHDL vendored copy missing: " + vendored)
    assert filecmp.cmp(canonical, vendored, shallow=False), (
        "src/configuration/Dialogs.py and nghdl/src/Dialogs.py have drifted; "
        "edit one and copy it verbatim to the other.")
