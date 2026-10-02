"""Tools exposed to the model.

Each Tool is two things:
  1. a JSON schema the model sees (sent as `tools=[...]` in the chat request, and
     rendered into the prompt by the server's chat template), and
  2. an async handler `(target, args, cfg) -> ToolOutput` that acts on the task
     environment through an exec target: a Harbor `BaseEnvironment` (Docker) or
     the local fixture environment. Both provide `exec(command, cwd=,
     timeout_sec=)` and `upload_file(source_path, target_path)`.

`run_tool()` turns a call into a `ToolOutput`: argument errors and
command-level failures (non-zero exits, command timeouts) become `[error] ...`
or `[exit code N]` observations the model can recover from. Failures of the
environment transport itself (Harbor exec/upload exceptions, the exec
backstop timeout, a subprocess that cannot be spawned) raise `EnvInfraError`,
which the episode loop turns into an INFRA stop (never an observation).
The environment session truncates
`raw_output` head+tail with `truncate()`; that truncated text is exactly what
the model sees as the tool result.

To add a tool: write a handler and append a Tool to TOOLS. Changing a schema or
an output format changes the learner's inputs: treat it as a new experiment.
"""

from __future__ import annotations

import shlex
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol


class EnvInfraError(RuntimeError):
    """The environment transport failed (not the command): the episode stops as INFRA."""


@dataclass
class ToolConfig:
    workdir: str
    command_timeout_sec: int  # per-call budget: the model may ask for less, never more
    max_output_chars: int


class ExecTarget(Protocol):
    async def exec(self, command: str, cwd: str | None = None, env: dict[str, str] | None = None, timeout_sec: int | None = None, user: str | int | None = None) -> Any: ...

    async def upload_file(self, source_path: Path | str, target_path: str) -> Any: ...


@dataclass
class ToolOutput:
    raw_output: str  # full text before truncation
    exit_code: int | None = None
    stdout: str | None = None
    stderr: str | None = None
    error: str | None = None  # tool-level error (also visible in raw_output)
    executed: bool = True  # False when nothing ran (unknown tool, missing argument)
    timeout_sec: int | None = None  # effective per-call timeout (bash)
    timed_out: bool | None = None  # the command hit its timeout (bash; None for other tools)


Handler = Callable[[Any, dict[str, Any], ToolConfig], Awaitable[ToolOutput]]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for the arguments
    handler: Handler

    def schema(self) -> dict[str, Any]:
        """OpenAI-style function tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def truncate(text: str, limit: int) -> str:
    """Keep the head and tail of long output so the model still sees both ends."""
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - 2 * half
    return f"{text[:half]}\n\n[... {omitted} characters omitted ...]\n\n{text[-half:]}"


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #


async def _env_call(coro: Awaitable[Any]) -> Any:
    """Await an exec-target call; anything but a command timeout is a transport failure."""
    try:
        return await coro
    except (EnvInfraError, TimeoutError):
        raise
    except Exception as e:
        raise EnvInfraError(f"{type(e).__name__}: {e}") from e


def effective_timeout(args: dict[str, Any], cfg: ToolConfig) -> int:
    """min(requested, budget), at least 1s. Raises ValueError on a non-integer request."""
    budget = int(cfg.command_timeout_sec)
    req = args.get("timeout_sec")
    if not req:  # absent / 0 / "" -> the budget (as before)
        return budget
    try:
        if isinstance(req, bool) or not isinstance(req, (int, float, str)):
            raise TypeError
        value = int(req)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"timeout_sec must be an integer, got {req!r}") from None
    return max(1, min(value, budget))


async def bash(env: Any, args: dict[str, Any], cfg: ToolConfig) -> ToolOutput:
    command = args["command"]
    try:
        timeout = effective_timeout(args, cfg)
    except ValueError as e:
        return ToolOutput(raw_output=f"[error] {e}", error=str(e), executed=False)
    try:
        result = await _env_call(env.exec(command, cwd=cfg.workdir, timeout_sec=timeout))
    except TimeoutError as e:  # a target that enforces the timeout by raising (not exit 124)
        msg = f"command timed out after {timeout}s" + (f" ({e})" if str(e) else "")
        return ToolOutput(raw_output=f"[error] {msg}", error=msg, timeout_sec=timeout, timed_out=True)
    header = f"[exit code {result.return_code}]"
    # Exec targets that enforce the timeout inside the environment (so no
    # process outlives its call) report it as exit code 124, like coreutils.
    timed_out = result.return_code == 124 and bool(getattr(env, "enforces_timeout", False))
    if timed_out:
        header += f" [command timed out after {timeout}s]"
    parts = [header]
    if result.stdout:
        parts.append(result.stdout.rstrip("\n"))
    if result.stderr:
        parts.append(f"[stderr]\n{result.stderr.rstrip()}")
    return ToolOutput(
        raw_output="\n".join(parts), exit_code=result.return_code, stdout=result.stdout, stderr=result.stderr, timeout_sec=timeout, timed_out=timed_out
    )


async def read_file(env: Any, args: dict[str, Any], cfg: ToolConfig) -> ToolOutput:
    path = args["path"]
    start = int(args.get("start_line") or 1)
    n = int(args.get("num_lines") or 200)
    # cat -n style numbering makes follow-up edits easier to target.
    cmd = f"awk 'NR>={start} && NR<{start + n} {{ printf \"%6d\\t%s\\n\", NR, $0 }}' {shlex.quote(path)}"
    result = await _env_call(env.exec(cmd, cwd=cfg.workdir))
    if result.return_code != 0:
        err = (result.stderr or "").strip() or "could not read file"
        return ToolOutput(raw_output=f"[error] {err}", exit_code=result.return_code, stdout=result.stdout, stderr=result.stderr, error=err)
    return ToolOutput(raw_output=result.stdout or "[empty]", exit_code=0, stdout=result.stdout, stderr=result.stderr)


async def write_file(env: Any, args: dict[str, Any], cfg: ToolConfig) -> ToolOutput:
    path = args["path"]
    content = args["content"]
    if not isinstance(content, str):
        raise TypeError("content must be a string")
    if not path.startswith("/"):
        path = f"{cfg.workdir.rstrip('/')}/{path}"
    # Upload via a temp file instead of shell-escaping arbitrary content.
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".tmp") as f:
        f.write(content)
        local = Path(f.name)
    try:
        # Command-level problems (parent is a file, target is a directory) are observations;
        # a failing upload after this check is a transport failure.
        q = shlex.quote(path)
        pre = await _env_call(env.exec(f"mkdir -p {shlex.quote(str(Path(path).parent))} && [ ! -d {q} ]"))
        if pre.return_code != 0:
            err = (pre.stderr or "").strip() or f"cannot write {path}: its directory could not be created or it is a directory"
            return ToolOutput(raw_output=f"[error] {err}", exit_code=pre.return_code, stdout=pre.stdout, stderr=pre.stderr, error=err, executed=False)
        await _env_call(env.upload_file(local, path))
    finally:
        local.unlink(missing_ok=True)
    return ToolOutput(raw_output=f"Wrote {len(content)} characters to {path}")


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

TOOLS: list[Tool] = [
    Tool(
        name="bash",
        description=(
            "Run a shell command in the task's Linux container and return its exit "
            "code, stdout and stderr. Each call is a fresh shell: `cd` and exported "
            "variables do not persist between calls. Non-interactive only."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to run."},
                "timeout_sec": {
                    "type": "integer",
                    "description": "Optional timeout in seconds (default 60).",
                },
            },
            "required": ["command"],
        },
        handler=bash,
    ),
    Tool(
        name="read_file",
        description="Read a text file with line numbers. Use start_line/num_lines to page through large files.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "description": "1-based, default 1."},
                "num_lines": {"type": "integer", "description": "Default 200."},
            },
            "required": ["path"],
        },
        handler=read_file,
    ),
    Tool(
        name="write_file",
        description="Create or overwrite a file with the given content.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
        handler=write_file,
    ),
]

TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def tool_schemas() -> list[dict[str, Any]]:
    return [t.schema() for t in TOOLS]


async def run_tool(env: Any, name: str, args: dict[str, Any], cfg: ToolConfig) -> ToolOutput:
    """Run one tool. Argument and command errors become tool outputs so the model
    can see and recover from them; `EnvInfraError` (transport failure) propagates."""
    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        msg = f"unknown tool {name!r}. Available: {', '.join(TOOLS_BY_NAME)}"
        return ToolOutput(raw_output=f"[error] {msg}", error=msg, executed=False)
    try:
        return await tool.handler(env, args, cfg)
    except EnvInfraError:
        raise
    except KeyError as e:
        msg = f"missing required argument {e}"
        return ToolOutput(raw_output=f"[error] {msg}", error=msg, executed=False)
    except Exception as e:  # bad argument values (environment failures are EnvInfraError)
        msg = f"{type(e).__name__}: {e}"
        return ToolOutput(raw_output=f"[error] {msg}", error=msg, executed=False)
