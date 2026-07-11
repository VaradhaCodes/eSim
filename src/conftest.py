"""Repo-wide pytest guardrails.

Every test gets a throwaway user home. Without this, any test that reaches
``paths.user_home()`` (directly or through Appconfig/bootstrap) reads and
WRITES the developer's real ``~/.esim`` — on Windows ``os.path.expanduser``
resolves ``USERPROFILE`` and ignores ``HOME``, so a test that only patched
``HOME`` silently escaped its sandbox. One such escape wrote a pytest tmp
path into the real ``workspace.txt`` and bricked every subsequent eSim
launch for the non-elevated user (GUI-thread hang persisting the project
registry into an admin-owned, since-deleted directory).

Patching both variables here makes home isolation automatic for the whole
suite instead of a per-test convention that will be forgotten again.
"""
import pytest


@pytest.fixture(autouse=True)
def isolated_user_home(tmp_path_factory, monkeypatch):
    home = tmp_path_factory.mktemp("user-home")
    monkeypatch.setenv("HOME", str(home))          # POSIX expanduser
    monkeypatch.setenv("USERPROFILE", str(home))   # Windows expanduser
    # USERPROFILE wins in ntpath.expanduser, but keep the pair consistent so
    # code reading HOMEDRIVE/HOMEPATH directly cannot escape either.
    monkeypatch.delenv("HOMEDRIVE", raising=False)
    monkeypatch.delenv("HOMEPATH", raising=False)
    yield home
