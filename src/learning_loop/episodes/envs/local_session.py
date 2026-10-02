"""Local fixture environment: a temp directory + host subprocesses.

NOT A SANDBOX. Commands run on the host as the current user. It exists only so
repository fixture tasks and tests can exercise the episode loop, replay and
grading without Docker, driven by scripted policies. `LocalFixtureBackend`
refuses live (model-driven) policies unless explicitly allowed.

Path model: tasks are written against a virtual work directory (default
`/app`, same as the Docker tasks). The session maps it onto a fresh temp
directory: occurrences of the virtual prefix in bash commands and file-tool
paths are rewritten to the real directory before execution, and the real
directory is rewritten back to the virtual prefix in outputs, so observations
do not depend on where the temp directory lives. Bash runs with a fixed,
minimal environment (LC_ALL=C, TZ=UTC, HOME/TMPDIR outside the work dir).

Fingerprints use the same `fingerprint.py` code as the Docker session, with
real paths displayed as virtual ones. Restoration = a fresh copy of the task's
`environment/files/` plus deterministic replay.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shutil
import signal
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.agents.tools import ToolConfig

from ...core.interfaces import EnvCapabilities, StateSpec
from ...core.records import RestoreCapability
from .. import fingerprint as fp
from .base import EnvInfraError, ToolSession, build_inputs_identity


@dataclass
class LocalExecResult:
    return_code: int
    stdout: str | None = None
    stderr: str | None = None


class _LocalExec:
    enforces_timeout = True

    def __init__(self, session: "LocalSession"):
        self.s = session

    async def exec(self, command: str, cwd: str | None = None, env: dict[str, str] | None = None, timeout_sec: int | None = None, user: str | int | None = None) -> LocalExecResult:
        s = self.s
        real_cmd = s.to_real(command)
        real_cwd = s.to_real(cwd) if cwd else str(s.real_workdir)
        if not Path(real_cwd).is_dir():
            return LocalExecResult(return_code=1, stdout="", stderr=f"cwd does not exist: {cwd}")
        try:
            proc = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                real_cmd,
                cwd=real_cwd,
                env={**s.base_env, **(env or {})},
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as e:  # the environment itself failed (not the command)
            raise EnvInfraError(f"could not spawn bash: {type(e).__name__}: {e}") from e
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
            rc = proc.returncode if proc.returncode is not None else -1
        except asyncio.CancelledError:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
            raise
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
            out, err = await proc.communicate()
            rc = 124
        return LocalExecResult(
            return_code=rc,
            stdout=s.to_virtual(out.decode("utf-8", "replace")),
            stderr=s.to_virtual(err.decode("utf-8", "replace")),
        )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        dst = Path(self.s.to_real(target_path))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, dst)


class LocalSession(ToolSession):
    def __init__(self, files_dir: Path | None, tool_cfg: ToolConfig, restore: RestoreCapability = RestoreCapability.DETERMINISTIC_REPLAY, root: Path | None = None):
        self.files_dir = Path(files_dir) if files_dir else None
        self.tool_cfg = tool_cfg
        self.virtual_workdir = tool_cfg.workdir.rstrip("/") or "/"
        self._root_arg = root
        self.root: Path | None = None
        self.real_workdir: Path | None = None
        self.target = _LocalExec(self)
        self.capabilities = EnvCapabilities(restore=restore, fingerprint=True, measures_cpu=False, measures_memory=False, workdir=self.virtual_workdir)
        self._virt_re = re.compile(r"(?<![\w./-])" + re.escape(self.virtual_workdir) + r"(?![\w.-])")
        self.base_env: dict[str, str] = {}

    # -- lifecycle ---------------------------------------------------------- #

    async def start(self) -> "LocalSession":
        base = Path(tempfile.mkdtemp(prefix="ll-local-", dir=self._root_arg))
        self.root = base
        self.real_workdir = base / "work"
        (base / "home").mkdir()
        (base / "tmp").mkdir()
        if self.files_dir and self.files_dir.exists():
            shutil.copytree(self.files_dir, self.real_workdir, symlinks=True)
        else:
            self.real_workdir.mkdir()
        self._real_variants = sorted({str(self.real_workdir), os.path.realpath(self.real_workdir)}, key=len, reverse=True)
        self.base_env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(base / "home"),
            "TMPDIR": str(base / "tmp"),
            "LC_ALL": "C",
            "LANG": "C",
            "TZ": "UTC",
        }
        return self

    async def close(self) -> None:
        if self.root is not None:
            shutil.rmtree(self.root, ignore_errors=True)
            self.root = None

    async def __aenter__(self) -> "LocalSession":
        return await self.start()

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def image_identity(self) -> dict[str, Any] | None:
        """The fixture 'image' is the task's files directory (no container)."""
        if self.files_dir is None:
            return None
        return build_inputs_identity("local_fixture", self.files_dir)

    # -- path mapping ------------------------------------------------------- #

    def to_real(self, text: str) -> str:
        assert self.real_workdir is not None, "session not started"
        return self._virt_re.sub(lambda _m: str(self.real_workdir), text)

    def to_virtual(self, text: str) -> str:
        for v in self._real_variants:
            text = text.replace(v, self.virtual_workdir)
        return text

    def real_path(self, virtual_path: str) -> Path:
        return Path(self.to_real(virtual_path))

    # -- fingerprint -------------------------------------------------------- #

    async def fingerprint_with_detail(self, spec: StateSpec) -> tuple[str, list[str]]:
        assert self.real_workdir is not None
        real = str(self.real_workdir)
        paths = [self.to_real(p) for p in spec.fingerprint_paths]
        return fp.compute(paths, spec.fingerprint_exclude, cwd=real, path_map=[(real, self.virtual_workdir)])
