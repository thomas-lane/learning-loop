"""`.env` loading: existing environment wins, empty template values are skipped, secrets stay out
of synced files."""

import os

from learning_loop.core.config import REPO_ROOT
from learning_loop.core.envfile import load_env
from learning_loop.hosts.remote import RSYNC_EXCLUDES


def test_load_env_precedence_and_empty_values(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("NEW_ONE=abc\nALREADY=from-file\nEMPTY=\n# comment\nexport QUOTED=\"x y\"\n")
    monkeypatch.setenv("ALREADY", "from-shell")
    for k in ("NEW_ONE", "EMPTY", "QUOTED"):
        monkeypatch.delenv(k, raising=False)
    assert load_env(f) == f
    assert os.environ["NEW_ONE"] == "abc"
    assert os.environ["ALREADY"] == "from-shell"
    assert "EMPTY" not in os.environ
    assert os.environ["QUOTED"] == "x y"
    for k in ("NEW_ONE", "QUOTED"):
        monkeypatch.delenv(k)


def test_missing_env_file_is_fine(tmp_path):
    assert load_env(tmp_path / "nope.env") is None


def test_template_has_no_values_and_secrets_are_never_synced():
    for line in (REPO_ROOT / ".env.example").read_text().splitlines():
        if line and not line.startswith("#"):
            assert line.endswith("="), f"template must not contain a value: {line}"
    assert ".env" in RSYNC_EXCLUDES
    assert ".env" in (REPO_ROOT / ".gitignore").read_text().split()
