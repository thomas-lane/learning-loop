"""Training entry point run as its own process (locally or on an SSH training host).

    python -m learning_loop.training.run --request request.json --work-dir DIR [--ref-cache-dir DIR]

`request.json` is a serialized `interfaces.TrainRequest`. On success the only
line on stdout is the absolute path of the published `checkpoint.json`
(library output is redirected to stderr). `DIR/result.json` records the outcome.

Exit codes: 0 published (or already published), 3 no trainable examples
(caller records a no-update cycle), 2 invalid request, 4 the work dir is locked by
another live trainer (nothing was read or written; result.json belongs to that
trainer), 1 other failure.
Re-running with the same work dir resumes an interrupted stage. The process holds an
exclusive, non-blocking flock on DIR/.trainer.lock for its whole lifetime, so two
trainers never share a work dir; the kernel releases it when the process exits or dies.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..interfaces import TrainRequest
from ..storage import atomic_write_json, now_iso, read_json
from .common import NoTrainableExamples, TrainingRequestError
from .render import RenderError

EXIT_WORK_DIR_LOCKED = 4
LOCK_NAME = ".trainer.lock"


class WorkDirLocked(RuntimeError):
    pass


def lock_work_dir(work_dir: Path) -> int:
    """Exclusive non-blocking flock on `work_dir/.trainer.lock`; returns the fd to keep open.
    Raises WorkDirLocked when another process holds it."""
    fd = os.open(work_dir / LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        holder = ""
        with contextlib.suppress(OSError):
            holder = os.pread(fd, 64, 0).decode(errors="replace").strip()
        os.close(fd)
        raise WorkDirLocked(f"{work_dir} is locked by another trainer{f' (pid {holder})' if holder else ''}") from None
    os.ftruncate(fd, 0)
    os.pwrite(fd, f"{os.getpid()}\n".encode(), 0)
    return fd


def get_trainer(name: str, **kwargs: Any) -> Any:
    if name == "fixture":
        from .fixture import FixtureTrainer

        return FixtureTrainer(**{k: v for k, v in kwargs.items() if k == "_test_interrupt_after_step"})
    if name == "trl_dpo":
        from .dpo import TrlDpoTrainer

        return TrlDpoTrainer(**kwargs)
    raise TrainingRequestError(f"unknown trainer {name!r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m learning_loop.training.run", description=__doc__.splitlines()[0])
    ap.add_argument("--request", required=True, type=Path)
    ap.add_argument("--work-dir", required=True, type=Path)
    ap.add_argument("--ref-cache-dir", type=Path, default=None, help="reference log-prob cache (trl_dpo)")
    a = ap.parse_args(argv)

    a.work_dir.mkdir(parents=True, exist_ok=True)
    try:
        lock_fd = lock_work_dir(a.work_dir)
    except WorkDirLocked as e:
        print(f"work dir locked: {e}; refusing to run a second trainer in it", file=sys.stderr)
        return EXIT_WORK_DIR_LOCKED
    try:
        return _run(a)
    finally:
        os.close(lock_fd)


def _run(a: argparse.Namespace) -> int:
    result_path = a.work_dir / "result.json"
    started = now_iso()
    real_stdout = sys.stdout
    try:
        request = TrainRequest.model_validate(read_json(a.request))
        trainer_name = request.training_config.get("trainer", "trl_dpo")
        kwargs: dict[str, Any] = {}
        if a.ref_cache_dir is not None and trainer_name == "trl_dpo":
            kwargs["ref_cache_dir"] = a.ref_cache_dir
        trainer = get_trainer(trainer_name, **kwargs)
        with contextlib.redirect_stdout(sys.stderr):
            record = trainer.train(request, a.work_dir)
        ckpt_json = Path(record.checkpoint.adapter_path or "") / "checkpoint.json"
        atomic_write_json(result_path, {"status": "published", "checkpoint_json": str(ckpt_json),
                                        "checkpoint_id": record.checkpoint.checkpoint_id, "started_at": started, "finished_at": now_iso()})
        print(str(ckpt_json), file=real_stdout, flush=True)
        return 0
    except NoTrainableExamples as e:
        atomic_write_json(result_path, {"status": "no_trainable_examples", "error": str(e), "render": e.render, "started_at": started, "finished_at": now_iso()})
        print(f"no trainable examples: {e}", file=sys.stderr)
        return 3
    except (TrainingRequestError, ValidationError, RenderError) as e:
        atomic_write_json(result_path, {"status": "invalid_request", "error": str(e), "started_at": started, "finished_at": now_iso()})
        traceback.print_exc()
        return 2
    except Exception as e:  # noqa: BLE001 - recorded, then non-zero exit
        atomic_write_json(result_path, {"status": "failed", "error": f"{type(e).__name__}: {e}", "started_at": started, "finished_at": now_iso()})
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
