"""Load `.env` from the repository root into the process environment.

Every `loop` command calls `load_env()` first, so credentials can live in a git-ignored `.env`
(start from `.env.example`). Values already set in the environment win over the file, and empty
values are ignored. Values are
never printed or recorded; configs refer to variables by name only (`*_env` keys). Subprocesses
started by `loop` (trainer, local model server, Harbor agent) inherit the loaded environment;
remote hosts never receive it.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

from .config import REPO_ROOT


def env_path() -> Path:
    """`$LOOP_ENV_FILE` if set, else `<repo>/.env`."""
    return Path(os.environ.get("LOOP_ENV_FILE") or REPO_ROOT / ".env")


def load_env(path: Path | None = None) -> Path | None:
    """Load the env file if it exists; return its path. Variables already set in the environment
    win, and empty values (`KEY=` lines left from .env.example) are skipped, so an unfilled
    template never overrides a default."""
    p = path or env_path()
    if not p.is_file():
        return None
    for key, value in dotenv_values(p).items():
        if value and key not in os.environ:
            os.environ[key] = value
    return p
