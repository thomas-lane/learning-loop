"""Start/stop an hf_server process owned by the caller (managed inference mode).

Only the process started here is ever signalled. A port that is already in use
is an error (another service may own it); nothing else is stopped.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class ServerStartError(RuntimeError):
    pass


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def http_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 600.0) -> tuple[int, dict[str, Any]]:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


class ManagedHFServer:
    def __init__(
        self,
        profile: str,
        port: int,
        log_path: Path,
        checkpoint_dir: Path | None = None,
        host: str = "127.0.0.1",
        device: str = "auto",
        allow_cpu_fallback: bool = False,
        extra_args: list[str] | None = None,
    ):
        self.profile, self.port, self.host = profile, port, host
        self.log_path = Path(log_path)
        self.checkpoint_dir = checkpoint_dir
        self.device, self.allow_cpu_fallback = device, allow_cpu_fallback
        self.extra_args = extra_args or []
        self.proc: subprocess.Popen[bytes] | None = None
        self._log = None

    @property
    def api_base(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def start(self, timeout_sec: float = 600.0) -> dict[str, Any]:
        if port_in_use(self.host, self.port):
            raise ServerStartError(f"{self.host}:{self.port} is already in use; refusing to start (and not touching its owner)")
        cmd = [sys.executable, "-m", "learning_loop.serving.hf_server", "--profile", self.profile,
               "--host", self.host, "--port", str(self.port), "--device", self.device, *self.extra_args]
        if self.checkpoint_dir is not None:
            cmd += ["--checkpoint-dir", str(self.checkpoint_dir)]
        if self.allow_cpu_fallback:
            cmd.append("--allow-cpu-fallback")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "ab")
        self.proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT)
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise ServerStartError(f"hf_server exited with {self.proc.returncode}; see {self.log_path}:\n{self.log_tail()}")
            try:
                status, info = http_json(f"http://{self.host}:{self.port}/health", timeout=2.0)
                if status == 200:
                    return info
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                pass
            time.sleep(1.0)
        self.stop()
        raise ServerStartError(f"hf_server not healthy after {timeout_sec}s; see {self.log_path}")

    def log_tail(self, n: int = 40) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-n:])
        except OSError:
            return ""

    def stop(self, timeout_sec: float = 30.0) -> int | None:
        if self.proc is None:
            return None
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=timeout_sec)
        rc = self.proc.returncode
        if self._log is not None:
            self._log.close()
            self._log = None
        return rc

    def __enter__(self) -> "ManagedHFServer":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()
