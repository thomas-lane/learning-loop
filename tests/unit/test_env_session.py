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
    assert te.observation == "[/app/out/x.txt: lines 1-1 of 1]\n     1\u2192hi\n[the file does not end with a newline]"
    assert te.stdout == "hi"  # the file's own text, without the line-number prefix


async def test_bash_separates_stderr_and_reports_the_commands_own_exit_code(session):
    te = await session.execute("c", "bash", {"command": "echo out1; echo err1 >&2; echo out2; echo err2 >&2; exit 3"}, None)
    assert te.observation == "[exit code 3]\nout1\nout2\n[stderr]\nerr1\nerr2"
    te = await session.execute("c2", "bash", {"command": "printf x"}, None)
    assert te.observation == "[exit code 0]\nx\n[output does not end with a newline]"
    te = await session.execute("c3", "bash", {"command": "exit 124"}, None)  # the command's own 124 is not a timeout
    assert te.exit_code == 124 and "timed out" not in te.observation


async def test_tool_arguments_of_the_wrong_type_are_observations(session):
    for name, args in [("bash", {"command": 123}), ("bash", {"command": "echo a\x00b"}), ("read_file", {"path": ["x"]}),
                       ("write_file", {"path": "out/", "content": "x"}), ("read_file", {"path": "data/app.log", "start_line": 0})]:
        te = await session.execute("c", name, args, None)
        assert te.observation.startswith("[error]") and not te.executed, (name, args, te.observation)


async def test_read_file_pages_and_explains_what_it_shows(session):
    w = session.real_workdir
    (w / "big.txt").write_text("".join(f"line {i}\n" for i in range(1, 451)))
    te = await session.execute("c", "read_file", {"path": "big.txt"}, None)
    lines = te.observation.splitlines()
    assert lines[0] == "[/app/big.txt: lines 1-200 of 450]" and lines[1] == "     1\u2192line 1" and lines[200] == "   200\u2192line 200"
    assert lines[-1] == "[250 more lines; next start_line=201]"
    te = await session.execute("c", "read_file", {"path": "big.txt", "start_line": 401}, None)
    assert te.observation.splitlines()[0] == "[/app/big.txt: lines 401-450 of 450]" and te.observation.endswith("   450\u2192line 450")
    te = await session.execute("c", "read_file", {"path": "big.txt", "start_line": 500}, None)
    assert te.observation == "[/app/big.txt: 450 lines; start_line 500 is past the end]"
    (w / "empty.txt").write_text("")
    assert (await session.execute("c", "read_file", {"path": "empty.txt"}, None)).observation == "[/app/empty.txt: empty file]"
    assert (await session.execute("c", "read_file", {"path": "nope.txt"}, None)).observation == "[error] /app/nope.txt: no such file"
    assert (await session.execute("c", "read_file", {"path": "data"}, None)).observation == "[error] /app/data: is a directory"
    (w / "wide.txt").write_text("".join("x" * 100 + "\n" for _ in range(200)))  # 20,200 characters > the 8000 limit
    te = await session.execute("c", "read_file", {"path": "wide.txt"}, None)
    assert len(te.observation) <= 8000 and te.observation.splitlines()[-1].startswith("[") and "next start_line=" in te.observation
    # Lines end at "\n" only, so numbers match grep -n and tracebacks even after a form feed.
    (w / "ff.py").write_text("a = 1\n\x0c\nb = 2\n")
    te = await session.execute("c", "read_file", {"path": "ff.py"}, None)
    assert te.observation.splitlines()[-1] == "     3\u2192b = 2" and "lines 1-3 of 3" in te.observation


async def test_write_file_keeps_an_existing_files_mode_and_creates_new_files_644(session):
    w = session.real_workdir
    (w / "run.sh").write_text("#!/bin/sh\necho old\n")
    os.chmod(w / "run.sh", 0o755)
    await session.execute("c", "write_file", {"path": "run.sh", "content": "#!/bin/sh\necho new\n"}, None)
    assert oct(os.stat(w / "run.sh").st_mode & 0o777) == "0o755" and (w / "run.sh").read_text().endswith("new\n")
    await session.execute("c", "write_file", {"path": "new/file.txt", "content": "a"}, None)
    assert oct(os.stat(w / "new" / "file.txt").st_mode & 0o777) == "0o644" and (w / "new" / "file.txt").read_text() == "a"


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
