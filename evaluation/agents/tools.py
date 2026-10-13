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

import json
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


def _string_arg(args: dict[str, Any], name: str) -> str:
    """A required string argument; a wrong type or a NUL byte is the model's error (an
    observation), never a transport failure."""
    value = args[name]
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string, got {type(value).__name__}")
    if "\x00" in value:
        raise ValueError(f"{name} must not contain a NUL byte")
    return value


def _ends(text: str, what: str) -> str:
    """`text` as shown to the model: verbatim, plus a note when it does not end with a newline
    (otherwise invisible, and it matters when a file is graded byte for byte)."""
    return text if text.endswith("\n") else text + f"\n[{what} does not end with a newline]\n"


# The command runs in its own bash with stderr captured to a file; the wrapper then prints this
# marker, the command's exit code and the captured stderr. Harbor's exec merges a command's
# stderr into stdout in no fixed order, so without the wrapper errors and data would interleave
# differently from run to run (and replays compare observations exactly). A missing marker means
# the wrapper itself was killed: the per-call timeout, never the command's own exit code 124.
_MARK = "\x1fLL-EXIT:"
_WRAP = (
    '__ll_err=$(mktemp) || exit 125; bash -c {cmd} 2>"$__ll_err"; __ll_rc=$?; '
    "printf '\\n\\037LL-EXIT:%d\\n' \"$__ll_rc\"; cat \"$__ll_err\"; rm -f \"$__ll_err\"; exit 0"
)


async def bash(env: Any, args: dict[str, Any], cfg: ToolConfig) -> ToolOutput:
    command = _string_arg(args, "command")
    try:
        timeout = effective_timeout(args, cfg)
    except ValueError as e:
        return ToolOutput(raw_output=f"[error] {e}", error=str(e), executed=False)
    try:
        result = await _env_call(env.exec(_WRAP.format(cmd=shlex.quote(command)), cwd=cfg.workdir, timeout_sec=timeout))
    except TimeoutError as e:  # a target that enforces the timeout by raising (not exit 124)
        msg = f"command timed out after {timeout}s" + (f" ({e})" if str(e) else "")
        return ToolOutput(raw_output=f"[error] {msg}", error=msg, timeout_sec=timeout, timed_out=True)
    out = result.stdout or ""
    cut = out.rfind("\n" + _MARK)
    if cut >= 0:  # the command finished: its own exit code and its separate stderr
        stdout = out[:cut]
        rc_line, _, stderr = out[cut + 1 + len(_MARK):].partition("\n")
        rc = int(rc_line) if rc_line.strip().lstrip("-").isdigit() else result.return_code
        stderr += result.stderr or ""
        timed_out = False
        header = f"[exit code {rc}]"
    else:  # killed before the marker: the per-call timeout (exit 124) or the wrapper failed
        stdout, stderr, rc = out, result.stderr or "", result.return_code
        timed_out = rc == 124 and bool(getattr(env, "enforces_timeout", False))
        header = f"[exit code {rc}]" + (f" [command timed out after {timeout}s]" if timed_out else "")
    parts = [header + "\n"]
    if stdout:
        parts.append(_ends(stdout, "output"))
    if stderr:
        parts.append("[stderr]\n" + _ends(stderr, "stderr"))
    return ToolOutput(raw_output="".join(parts).rstrip("\n"), exit_code=rc, stdout=stdout, stderr=stderr, timeout_sec=timeout, timed_out=timed_out)


# Runs with the container's python3 (part of every task profile): one page of a regular file,
# cut at whole lines to fit the output limit, as JSON for the formatter below. Lines end at "\n"
# only, as grep -n, sed and Python's tracebacks count them (str.splitlines would also split at
# form feeds and other separators and shift every later line number).
_READ_PAGE = r"""
import json, os, stat, sys
path, start, count, cap = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
GUTTER = 8  # characters of the line-number prefix the formatter adds
def out(**kw):
    print(json.dumps(kw)); sys.exit(0)
try:
    st = os.stat(path)
except FileNotFoundError:
    out(error="no such file")
except PermissionError:
    out(error="permission denied")
except OSError as e:
    out(error=e.strerror or str(e))
if stat.S_ISDIR(st.st_mode):
    out(error="is a directory")
if not stat.S_ISREG(st.st_mode):
    out(error="not a regular file")
try:
    with open(path, "rb") as f:
        data = f.read()
except PermissionError:
    out(error="permission denied")
text = data.decode("utf-8", "replace")
parts = text.split("\n")
lines = [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])
total = len(lines)
if start > total:
    out(total=total, start=start, lines=[], final_newline=text.endswith("\n"), binary=b"\0" in data)
page, size = [], 0
for line in lines[start - 1:start - 1 + count]:
    if page and size + len(line) + GUTTER > cap:
        break
    page.append(line[:cap] if not page else line)
    size += len(page[-1]) + GUTTER
out(total=total, start=start, lines=page, final_newline=text.endswith("\n"), binary=b"\0" in data)
"""


LINE_ARROW = "\u2192"  # →


async def read_file(env: Any, args: dict[str, Any], cfg: ToolConfig) -> ToolOutput:
    path = _string_arg(args, "path")
    if not path.startswith("/"):
        path = f"{cfg.workdir.rstrip('/')}/{path}"
    try:
        start = 1 if args.get("start_line") is None else int(args["start_line"])
        count = 200 if args.get("num_lines") is None else int(args["num_lines"])
    except (TypeError, ValueError, OverflowError):
        raise ValueError("start_line and num_lines must be integers") from None
    if start < 1 or count < 1:
        raise ValueError("start_line and num_lines must be at least 1")
    cap = max(200, cfg.max_output_chars - 400)  # room for the header and footer lines
    cmd = f"python3 -I -c {shlex.quote(_READ_PAGE)} {shlex.quote(path)} {start} {count} {cap}"
    result = await _env_call(env.exec(cmd, cwd=cfg.workdir, timeout_sec=60))
    try:
        page = json.loads((result.stdout or "").strip().splitlines()[-1])
    except (IndexError, ValueError):
        err = ((result.stdout or "") + (result.stderr or "")).strip()[-300:] or "could not read file"
        return ToolOutput(raw_output=f"[error] {err}", exit_code=result.return_code, stdout=result.stdout, stderr=result.stderr, error=err)
    if "error" in page:
        err = f"{path}: {page['error']}"
        return ToolOutput(raw_output=f"[error] {err}", exit_code=1, error=err)
    total, lines = page["total"], page["lines"]
    if total == 0:
        return ToolOutput(raw_output=f"[{path}: empty file]", exit_code=0, stdout="")
    if not lines:
        return ToolOutput(raw_output=f"[{path}: {total} lines; start_line {start} is past the end]", exit_code=0, stdout="")
    end = start + len(lines) - 1
    text = "".join(lines)
    # Each line after its number and an arrow, as Claude Code's Read tool shows files: line numbers
    # are what tracebacks, grep -n and tests refer to. A tab after the number was mistaken for
    # the file's own separator (and copied into files the model wrote); an arrow is not.
    numbered = "".join(f"{n:>6}{LINE_ARROW}{line}" for n, line in enumerate(lines, start))
    shown = [f"[{path}: lines {start}-{end} of {total}" + (" (binary data shown as text)" if page["binary"] else "") + "]\n",
             numbered if numbered.endswith("\n") else numbered + "\n"]
    if end < total:
        shown.append(f"[{total - end} more lines; next start_line={end + 1}]")
    elif not page["final_newline"]:
        shown.append("[the file does not end with a newline]")
    return ToolOutput(raw_output="".join(shown).rstrip("\n"), exit_code=0, stdout=text)


async def write_file(env: Any, args: dict[str, Any], cfg: ToolConfig) -> ToolOutput:
    path = _string_arg(args, "path")
    content = args["content"]
    if not isinstance(content, str):
        raise ValueError("content must be a string")
    if not path or path.endswith("/"):
        raise ValueError(f"path must name a file, got {path!r}")
    if not path.startswith("/"):
        path = f"{cfg.workdir.rstrip('/')}/{path}"
    # Upload via a temp file instead of shell-escaping arbitrary content.
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".tmp", encoding="utf-8", newline="") as f:
        f.write(content)
        local = Path(f.name)
    try:
        # Command-level problems (parent is a file, target is a directory) are observations;
        # a failing upload after this check is a transport failure. An existing file's mode is
        # kept; the upload itself would leave the host temp file's 0600 and the host's owner.
        q = shlex.quote(path)
        pre = await _env_call(env.exec(
            f"mkdir -p {shlex.quote(str(Path(path).parent))} || exit 1; if [ -d {q} ]; then echo 'is a directory'; exit 1; fi; "
            f"if [ -e {q} ]; then python3 -I -c 'import os, sys; print(format(os.stat(sys.argv[1]).st_mode & 0o7777, \"o\"))' {q}; fi",
            cwd=cfg.workdir, timeout_sec=60))
        if pre.return_code != 0:
            reason = ((pre.stdout or "") + (pre.stderr or "")).strip()[-300:] or "its directory could not be created"
            err = f"cannot write {path}: {reason}"
            return ToolOutput(raw_output=f"[error] {err}", exit_code=pre.return_code, stdout=pre.stdout, stderr=pre.stderr, error=err, executed=False)
        mode = (pre.stdout or "").strip() or "644"
        await _env_call(env.upload_file(local, path))
        post = await _env_call(env.exec(f'chown "$(id -u):$(id -g)" {q} && chmod {shlex.quote(mode)} {q}', cwd=cfg.workdir, timeout_sec=60))
        if post.return_code != 0:
            reason = ((post.stdout or "") + (post.stderr or "")).strip()[-300:]
            raise EnvInfraError(f"could not set owner and mode of {path}: {reason}")
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
            "Run a shell command in the task's Linux container. Returns the command's exit code, its "
            "standard output, and its standard error after a `[stderr]` line. Each call is a fresh bash "
            "shell started in /app: `cd` and exported variables do not persist between calls. "
            "Non-interactive: standard input is empty. Long output keeps its beginning and end."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to run."},
                "timeout_sec": {
                    "type": "integer",
                    "description": "Optional timeout in seconds. It cannot exceed the episode's limit, which is also the default.",
                },
            },
            "required": ["command"],
        },
        handler=bash,
    ),
    Tool(
        name="read_file",
        description=(
            "Read a text file. Each line is shown after its line number and a \u2192 arrow, which are "
            "not part of the file. A header gives the line range and the total number of lines; when "
            "more lines follow, a final note gives the next start_line. A page also ends early to fit "
            "the output limit."
        ),
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
        description=(
            "Create or overwrite a file with exactly the given content (nothing is added: end the "
            "content with a newline if the file should end with one). Missing parent directories are "
            "created. An existing file keeps its permissions; a new file gets mode 644. A relative "
            "path is relative to /app."
        ),
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
