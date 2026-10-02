"""Local session semantics, fingerprint coverage, and the in-container fingerprint path."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys

import pytest
from _env_helpers import count_errors_instance

from evaluation.agents.tools import ToolConfig
from learning_loop.core.interfaces import StateSpec
from learning_loop.core.records import RestoreCapability
from learning_loop.episodes import fingerprint as fp
from learning_loop.episodes.envs.harbor_session import _FINGERPRINT_B64, HarborSession
from learning_loop.episodes.envs.local_session import LocalSession

SPEC = StateSpec(restore=RestoreCapability.DETERMINISTIC_REPLAY, fingerprint_paths=["/app"], fingerprint_exclude=["*/__pycache__"])


@pytest.fixture()
async def session(tmp_path):
    inst = count_errors_instance(tmp_path / "tasks")
    s = LocalSession(tmp_path / "tasks" / "count-errors__easy__s1" / "environment" / "files", ToolConfig("/app", 5, 8000))
    await s.start()
    yield s
    await s.close()


async def test_virtual_workdir_mapping(session):
    te = await session.execute("c1", "bash", {"command": "pwd; ls /app/data; echo /application"}, None)
    assert te.observation.splitlines()[:2] == ["[exit code 0]", "/app"]
    assert "app.log" in te.observation and "/application" in te.observation
    te = await session.execute("c2", "write_file", {"path": "out/x.txt", "content": "hi"}, None)
    assert te.observation == "Wrote 2 characters to /app/out/x.txt"
    assert (session.real_workdir / "out" / "x.txt").read_text() == "hi"
    te = await session.execute("c3", "read_file", {"path": "/app/out/x.txt"}, None)
    assert te.observation == "     1\thi\n"


async def test_command_timeout_kills_process(session):
    te = await session.execute("c", "bash", {"command": "sleep 30; echo late", "timeout_sec": 1}, None)
    assert te.exit_code == 124 and "timed out after 1s" in te.observation and "late" not in te.observation
    assert te.duration_sec < 10


async def test_fingerprint_covers_content_perms_symlinks_not_mtime(session):
    w = session.real_workdir
    base, lines = await session.fingerprint_with_detail(SPEC)
    assert lines[0] == 'C "/app" dir' and any(line.startswith('F "/app/data/app.log"') for line in lines)
    os.utime(w / "data" / "app.log", (1, 1))
    assert (await session.fingerprint(SPEC)) == base  # mtime is not state
    (w / "__pycache__").mkdir()
    (w / "__pycache__" / "x.pyc").write_bytes(b"\0")
    assert (await session.fingerprint(SPEC)) == base  # declared exclude
    assert (await session.fingerprint(StateSpec(fingerprint_paths=["/app"]))) != base  # ...only when declared
    os.chmod(w / "data" / "app.log", 0o600)
    fp_perm = await session.fingerprint(SPEC)
    assert fp_perm != base
    os.chmod(w / "data" / "app.log", 0o644)
    assert (await session.fingerprint(SPEC)) == base
    os.symlink("data/app.log", w / "link")
    fp_link = await session.fingerprint(SPEC)
    os.unlink(w / "link")
    os.symlink("data/worker.log", w / "link")
    assert fp_link not in (base, await session.fingerprint(SPEC))
    os.unlink(w / "link")
    with open(w / "data" / "app.log", "a") as f:
        f.write("x\n")
    assert (await session.fingerprint(SPEC)) != base


def test_shipped_script_matches_in_process(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "a.txt").write_text("hello")
    os.symlink("a.txt", tmp_path / "d" / "l")
    arg = json.dumps({"paths": [str(tmp_path / "d")], "exclude": [], "cwd": str(tmp_path)})
    script = base64.b64decode(_FINGERPRINT_B64)
    out = subprocess.run([sys.executable, "-I", "-B", "-", arg], input=script, capture_output=True, check=True, cwd="/")
    data = json.loads(out.stdout)
    assert (data["sha256"], data["lines"]) == fp.compute([str(tmp_path / "d")], [], str(tmp_path))


class _HostEnv:
    """Minimal stand-in for a Harbor BaseEnvironment: host subprocesses, real paths."""

    def __init__(self):
        self.commands: list[str] = []

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        from learning_loop.episodes.envs.local_session import LocalExecResult

        self.commands.append(command)
        p = subprocess.run(["bash", "-c", command], cwd=cwd, capture_output=True, text=True, timeout=timeout_sec)
        return LocalExecResult(return_code=p.returncode, stdout=p.stdout, stderr=p.stderr)

    async def upload_file(self, source_path, target_path):
        import shutil

        shutil.copyfile(source_path, target_path)


async def test_harbor_session_fingerprint_command_path(tmp_path):
    """HarborSession ships fingerprint.py via base64 + `python3 -I -B -` (read-only)."""
    work = tmp_path / "w"
    work.mkdir()
    (work / "f.txt").write_text("abc")
    env = _HostEnv()
    s = HarborSession(env, ToolConfig(str(work), 10, 8000), restore=RestoreCapability.DETERMINISTIC_REPLAY, measure_cpu=False)
    digest, lines = await s.fingerprint_with_detail(StateSpec(fingerprint_paths=[str(work)]))
    assert (digest, lines) == fp.compute([str(work)], [], str(work))
    assert "python3 -I -B -" in env.commands[-1]
    assert sorted(p.name for p in work.iterdir()) == ["f.txt"]  # nothing written
    te = await s.execute("c", "bash", {"command": "echo hi"}, '{"command": "echo hi"}')
    assert te.observation == "[exit code 0]\nhi" and te.requested_arguments_raw == '{"command": "echo hi"}'
    assert te.cpu_time_sec is None and s.capabilities.measures_cpu is False
