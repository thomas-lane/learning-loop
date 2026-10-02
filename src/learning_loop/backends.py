"""Episode backends: run a planned episode in a fresh environment, grade it separately.

HarborDockerBackend
    Runs one Harbor `Trial` programmatically with the repo's `ToolAgent`
    (`evaluation.agents.tool_agent:ToolAgent`), passing the `EpisodePlan` as a
    file. Tasks use Harbor's *separate* verifier mode: after the agent phase,
    only the task's declared `artifacts` are copied out of the agent container
    and into a fresh container built from the task's `tests/` directory, where
    the hidden tests run; the hidden tests never enter the agent's container,
    and the verifier directory is emptied before grading (a reward file planted
    by the agent is discarded). The agent's authoritative records
    (`events.jsonl`, `episode.json`) are written to a host directory that is
    not mounted into the container.

LocalFixtureBackend
    Runs fixture tasks (`[metadata.local_fixture]`) in-process with
    `LocalSession` (temp dir + host subprocesses: NOT a sandbox) and grades a
    separate copy of the declared artifacts with the task's grader script.
    Refuses model-driven policies unless `allow_live_policy=True`.

Neither backend retries. Failures are classified (INFRA vs SAFETY vs the
episode's own stop) and returned; bounded retry is the coordinator's job.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from evaluation.agents.tools import ToolConfig

from .episode import EpisodeCore, load_core, run_episode, write_episode_files
from .events import EventLog, load_turns, read_events
from .interfaces import EpisodePlan, EpisodeResult
from .policy import make_policy
from .records import EventKind, RestoreCapability, StopCategory, TaskInstance, Usage
from .storage import atomic_write_json
from .tasks import load_state_spec, read_task_toml

AGENT_IMPORT_PATH = "evaluation.agents.tool_agent:ToolAgent"


def _fresh_record_dir(out_dir: Path) -> Path:
    rec = out_dir / "episode"
    if rec.exists() and any(rec.iterdir()):
        raise FileExistsError(f"{rec} already holds an episode record; preserve it and use a fresh directory")
    rec.mkdir(parents=True, exist_ok=True)
    return rec


def _stub_core(plan: EpisodePlan, rec: Path, reason: str, category: StopCategory, infra_error: str | None) -> EpisodeCore:
    """Core for an episode whose agent never wrote episode.json.

    Counts come from events.jsonl when it exists and parses (turns and malformed
    turns via `load_turns`); otherwise turn counts are unavailable (n_turns=None,
    listed in extra["unavailable_counts"]), never a measured 0."""
    n_req = n_calls = 0
    n_turns: int | None = None
    n_malformed = 0
    n_prefix = 0
    extra: dict[str, Any] = {"episode_record": "missing (agent did not finish writing episode.json)"}
    ev_path = rec / "events.jsonl"
    evs = None
    if ev_path.exists():
        try:
            evs = read_events(ev_path)
        except Exception as e:  # e.g. a torn last line after a crash
            extra["events_error"] = f"{type(e).__name__}: {e}"
    if evs is not None:
        n_req = sum(1 for e in evs if e.kind == EventKind.REQUEST)
        n_calls = sum(1 for e in evs if e.kind == EventKind.TOOL_RESULT and not e.data.get("replay"))
        turns = load_turns(evs)
        n_turns = len(turns)
        n_malformed = sum(1 for t in turns if t.malformed)
        n_prefix = sum(1 for t in turns if t.origin == "replayed")
        extra["counts_source"] = "events.jsonl"
    else:
        extra["unavailable_counts"] = ["n_turns", "n_malformed_turns", "n_requests", "n_tool_calls"]
    return EpisodeCore(
        episode_id=plan.episode_id,
        role=plan.role,
        instance_id=plan.instance_id,
        attempt_index=plan.attempt_index,
        seed=plan.seed,
        checkpoint_id=plan.policy.checkpoint.checkpoint_id if plan.policy.checkpoint else plan.policy.served_model_name,
        stop_reason=reason,
        stop_category=category,
        usage=Usage(source="none") if n_req or evs is None else Usage(input_tokens=0, output_tokens=0, source="none"),
        n_requests=n_req,
        n_tool_calls=n_calls,
        n_malformed_turns=n_malformed,
        n_turns=n_turns,
        n_prefix_turns=n_prefix,
        infra_error=infra_error,
        extra=extra,
    )


# --------------------------------------------------------------------------- #
# Harbor + Docker
# --------------------------------------------------------------------------- #


class HarborDockerBackend:
    name = "harbor_docker"

    def __init__(self, concurrency: int = 1, delete: bool = True, agent_import_path: str = AGENT_IMPORT_PATH, timeout_margin_sec: float = 60.0):
        self._sem = asyncio.Semaphore(max(1, concurrency))
        self.delete = delete  # remove containers + locally built images after each trial
        self.agent_import_path = agent_import_path
        self.timeout_margin_sec = timeout_margin_sec

    def restore_capability(self, instance: TaskInstance) -> RestoreCapability:
        return load_state_spec(Path(instance.task_dir)).restore

    async def run(self, instance: TaskInstance, plan: EpisodePlan, out_dir: Path) -> EpisodeResult:
        async with self._sem:
            return await self._run(instance, plan, Path(out_dir).resolve())

    def trial_config(self, instance: TaskInstance, plan: EpisodePlan, plan_path: Path, rec: Path, trials_dir: Path) -> Any:
        from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig

        name = re.sub(r"[^A-Za-z0-9_.-]", "_", plan.episode_id)[:60]
        return TrialConfig(
            task=TaskConfig(path=Path(instance.task_dir)),
            trial_name=name,
            trials_dir=trials_dir,
            agent=AgentConfig(
                import_path=self.agent_import_path,
                model_name=plan.policy.served_model_name or plan.policy.kind,
                kwargs={"episode_plan_path": str(plan_path), "record_dir": str(rec)},
                # The episode enforces budgets.agent_timeout_sec itself; Harbor's limit is a backstop.
                override_timeout_sec=plan.budgets.agent_timeout_sec + self.timeout_margin_sec,
            ),
            environment=EnvironmentConfig(delete=self.delete),
        )

    async def _run(self, instance: TaskInstance, plan: EpisodePlan, out_dir: Path) -> EpisodeResult:
        from harbor.trial.trial import Trial

        out_dir.mkdir(parents=True, exist_ok=True)
        rec = _fresh_record_dir(out_dir)
        plan_path = out_dir / "agent_plan.json"
        atomic_write_json(plan_path, plan)
        trials_dir = out_dir / "harbor"
        cfg = self.trial_config(instance, plan, plan_path, rec, trials_dir)
        trial_dir = trials_dir / cfg.trial_name
        t0 = time.monotonic()
        result = None
        harbor_error: str | None = None
        try:
            trial = await Trial.create(cfg)
            result = await trial.run()
        except Exception as e:  # trial could not be created/run at all
            harbor_error = f"{type(e).__name__}: {e}"
        wall = time.monotonic() - t0

        reward = None
        exc_type = exc_msg = None
        if result is not None:
            if result.verifier_result is not None and result.verifier_result.rewards is not None:
                reward = {k: float(v) for k, v in result.verifier_result.rewards.items()}
            if result.exception_info is not None:
                exc_type, exc_msg = result.exception_info.exception_type, result.exception_info.exception_message
        elif harbor_error:
            exc_type, exc_msg = "HarborError", harbor_error

        core_path = rec / "episode.json"
        override: tuple[str, StopCategory] | None = None
        infra_error: str | None = None
        if core_path.exists():
            core = load_core(core_path)
        else:
            infra_error = f"{exc_type}: {exc_msg}" if exc_type else "agent wrote no episode.json"
            cat = StopCategory.SAFETY if exc_type == "AgentTimeoutError" else StopCategory.INFRA
            reason = "safety:agent_timeout" if cat == StopCategory.SAFETY else f"infra:{exc_type or 'no_episode_record'}"
            core = _stub_core(plan, rec, reason, cat, infra_error)
        if core_path.exists() and exc_type:
            if exc_type == "AgentTimeoutError":
                if core.stop_reason != "safety:agent_timeout":
                    override = ("safety:agent_timeout", StopCategory.SAFETY)
            else:
                infra_error = f"{exc_type}: {exc_msg}"[:2000]
                override = (f"infra:{exc_type}", StopCategory.INFRA)
        elif core_path.exists() and reward is None and core.stop_category not in (StopCategory.INFRA,):
            infra_error = "verifier produced no reward"
            override = ("infra:no_reward", StopCategory.INFRA)

        extra: dict[str, Any] = {
            "backend": self.name,
            "wall_sec": round(wall, 3),
            "harbor_exception": {"type": exc_type, "message": (exc_msg or "")[:2000]} if exc_type else None,
            "verifier_environment_mode": (result.verifier_environment_mode.value if result is not None and result.verifier_environment_mode else None),
            "task_content_hash": instance.content_hash,
        }
        summary = core.to_summary(
            reward=reward,
            state_spec=plan.state_spec,
            trial_dir=str(trial_dir),
            events_path=str(rec / "events.jsonl"),
            stop_override=override,
            infra_error=infra_error,
            extra=extra,
        )
        atomic_write_json(rec / "summary.json", summary)
        return EpisodeResult(summary=summary, out_dir=rec, replay_ok=core.replay_ok, replay_mismatches=list(core.replay_mismatches))


# --------------------------------------------------------------------------- #
# Local fixture tasks
# --------------------------------------------------------------------------- #


class LocalFixtureBackend:
    name = "local_fixture"

    def __init__(self, allow_live_policy: bool = False, grade_timeout_sec: float = 60.0, work_root: Path | None = None):
        self.allow_live_policy = allow_live_policy
        self.grade_timeout_sec = grade_timeout_sec
        self.work_root = work_root

    @staticmethod
    def local_meta(task_dir: Path) -> dict[str, Any]:
        meta = read_task_toml(task_dir).get("metadata", {}).get("local_fixture")
        if not meta:
            raise ValueError(f"{task_dir} is not a local fixture task (no [metadata.local_fixture])")
        return meta

    def restore_capability(self, instance: TaskInstance) -> RestoreCapability:
        try:
            self.local_meta(Path(instance.task_dir))
        except ValueError:
            return RestoreCapability.NONE
        return load_state_spec(Path(instance.task_dir)).restore

    async def run(self, instance: TaskInstance, plan: EpisodePlan, out_dir: Path) -> EpisodeResult:
        from .envs.local_session import LocalSession

        task_dir = Path(instance.task_dir)
        meta = self.local_meta(task_dir)
        if plan.policy.kind != "scripted" and not self.allow_live_policy:
            raise PermissionError("the local fixture environment is not a sandbox; it runs scripted policies only (allow_live_policy=False)")
        out_dir = Path(out_dir).resolve()
        rec = _fresh_record_dir(out_dir)
        log = EventLog(rec / "events.jsonl", plan.episode_id)
        spec = plan.state_spec
        tool_cfg = ToolConfig(workdir=plan.workdir, command_timeout_sec=plan.budgets.tool_timeout_sec, max_output_chars=plan.budgets.max_output_chars)
        session = LocalSession(task_dir / meta.get("files", "environment/files"), tool_cfg, restore=spec.restore, root=self.work_root)
        policy = make_policy(plan.policy)
        t0 = time.monotonic()

        def flush(messages: list[dict[str, Any]]) -> None:
            atomic_write_json(rec / "messages.json", {"tools": plan.tools, "messages": messages})

        async with session:
            try:
                run = await run_episode(plan, session, policy, log, on_turn=flush)
            finally:
                await policy.aclose()
            write_episode_files(rec, run, plan.tools)
            reward, grading = self._grade(task_dir, meta, session)
        atomic_write_json(rec / "grading.json", grading)
        override = None
        infra_error = None
        if reward is None:
            infra_error = f"grading failed: {grading.get('error')}"
            override = ("infra:grading", StopCategory.INFRA)
        summary = run.core.to_summary(
            reward=reward,
            state_spec=spec,
            events_path=str(rec / "events.jsonl"),
            stop_override=override,
            infra_error=infra_error,
            extra={"backend": self.name, "wall_sec": round(time.monotonic() - t0, 3), "task_content_hash": instance.content_hash, "sandbox": False},
        )
        atomic_write_json(rec / "summary.json", summary)
        return EpisodeResult(summary=summary, out_dir=rec, replay_ok=run.core.replay_ok, replay_mismatches=list(run.core.replay_mismatches))

    def _grade(self, task_dir: Path, meta: dict[str, Any], session: Any) -> tuple[dict[str, float] | None, dict[str, Any]]:
        """Copy declared artifacts (regular files only) + tests/ into a fresh dir and run the grader there."""
        artifacts = read_task_toml(task_dir).get("artifacts", [])
        grade_root = Path(tempfile.mkdtemp(prefix="ll-grade-", dir=self.work_root))
        info: dict[str, Any] = {"artifacts": {}}
        try:
            for a in artifacts:
                src_virtual = a if isinstance(a, str) else a["source"]
                src = session.real_path(src_virtual)
                dst = grade_root / src_virtual.lstrip("/")
                if src.is_symlink() or not src.exists():
                    info["artifacts"][src_virtual] = "missing" if not src.is_symlink() else "skipped:symlink"
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src.is_file():
                    shutil.copyfile(src, dst)
                    info["artifacts"][src_virtual] = "copied"
                else:
                    info["artifacts"][src_virtual] = "skipped:not_a_regular_file"
            shutil.copytree(task_dir / "tests", grade_root / "tests")
            out = grade_root / "reward.json"
            grader = grade_root / meta.get("grader", "tests/grade.py")
            proc = subprocess.run(
                [sys.executable, "-I", "-B", str(grader), "--root", str(grade_root), "--out", str(out)],
                cwd=grade_root,
                capture_output=True,
                text=True,
                timeout=self.grade_timeout_sec,
            )
            info["stdout"] = proc.stdout[-4000:]
            info["stderr"] = proc.stderr[-4000:]
            info["exit_code"] = proc.returncode
            if not out.exists():
                info["error"] = "grader wrote no reward.json"
                return None, info
            reward = json.loads(out.read_text())
            if not isinstance(reward, dict) or not all(isinstance(v, (int, float)) for v in reward.values()):
                info["error"] = f"invalid reward file: {reward!r}"
                return None, info
            info["reward"] = reward
            return {k: float(v) for k, v in reward.items()}, info
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as e:
            info["error"] = f"{type(e).__name__}: {e}"
            return None, info
        finally:
            shutil.rmtree(grade_root, ignore_errors=True)
