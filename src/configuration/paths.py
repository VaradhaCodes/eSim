"""Canonical filesystem locations used throughout eSim.

All install resources are anchored to this file, never the process working
directory.  Per-user state lives under ``~/.esim`` on every platform.
"""

import os
import tempfile


def repo_root():
    """Return the eSim checkout/install root."""
    return os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))


def src_root():
    return os.path.join(repo_root(), "src")


def image_path(*parts):
    return os.path.join(repo_root(), "images", *parts)


def library_path(*parts):
    return os.path.join(repo_root(), "library", *parts)


def user_home():
    """Return the current user's home, expanded at call time for testability."""
    return os.path.expanduser("~")


def esim_config_dir():
    return os.path.join(user_home(), ".esim")


def esim_config_path(*parts):
    return os.path.join(esim_config_dir(), *parts)


def nghdl_config_path():
    return os.path.join(user_home(), ".nghdl", "config.ini")


def atomic_write_text(path, text, encoding="utf-8"):
    """Atomically replace ``path`` with ``text`` and return ``path``."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", dir=parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return path


def write_json_atomic(path, data, indent=2):
    """Atomically write ``data`` as JSON to ``path`` (tmp file + os.replace).

    Bare ``open(w) + json.dump`` leaves a truncated/corrupt file if the process
    dies mid-write; every config writer should go through this instead.
    """
    import json
    return atomic_write_text(path, json.dumps(data, indent=indent))


def read_workspace(default_path=None, default_check="0"):
    """Return ``(check, path)`` from workspace.txt with safe defaults."""
    fallback = default_path or os.path.join(user_home(), "eSim-Workspace")
    try:
        with open(esim_config_path("workspace.txt"), encoding="utf-8") as fh:
            check, path = fh.readline().strip().split(" ", 1)
        if not path:
            raise ValueError("workspace path is empty")
        return check, path
    except (OSError, ValueError):
        return str(default_check), fallback


def write_workspace(check, path):
    """Persist workspace selection without exposing a truncated file."""
    return atomic_write_text(
        esim_config_path("workspace.txt"), f"{int(check)} {path}")
