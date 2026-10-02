"""Shared helpers for Workstream-A tests (episode loop, environments, backends)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evaluation.agents.tools import tool_schemas
from learning_loop.core.config import REPO_ROOT
from learning_loop.core.interfaces import EpisodePlan, ReplaySpec
from learning_loop.core.records import EpisodeBudgets, EpisodeRole, PolicySpec, SamplingConfig, TaskInstance
from learning_loop.tasks import instances

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "env"
COUNT_POLICY = FIXTURES / "count_errors_policy.yaml"
SYSTEM_PROMPT = (REPO_ROOT / "evaluation" / "agents" / "system_prompt.md").read_text().format(workdir="/app")


def count_errors_instance(dest: Path, instance_id: str = "count-errors/easy/s1") -> TaskInstance:
    splits = instances.load_splits(REPO_ROOT / "evaluation" / "splits" / "fixture.yaml")
    return instances.materialize(splits, dest, ids=[instance_id])[instance_id]


def scripted_spec(path: Path | str = COUNT_POLICY, **kw: Any) -> PolicySpec:
    return PolicySpec(kind="scripted", served_model_name="scripted-fixture", scripted_path=str(path), **kw)


def make_plan(
    instance: TaskInstance,
    episode_id: str,
    policy: PolicySpec | None = None,
    *,
    seed: int | None = 7,
    replay: ReplaySpec | None = None,
    role: EpisodeRole = EpisodeRole.COLLECT,
    **budget_overrides: Any,
) -> EpisodePlan:
    budgets = dict(max_turns=12, max_episode_tokens=None, tool_timeout_sec=20, max_output_chars=8000, agent_timeout_sec=60.0)
    budgets.update(budget_overrides)
    return EpisodePlan(
        episode_id=episode_id,
        role=EpisodeRole.BRANCH if replay is not None else role,
        instance_id=instance.instance_id,
        attempt_index=None if replay is not None else 0,
        seed=seed,
        policy=policy or scripted_spec(),
        budgets=EpisodeBudgets(**budgets),
        system_prompt=SYSTEM_PROMPT,
        instruction=instances.read_instruction(Path(instance.task_dir)),
        tools=tool_schemas(),
        state_spec=instances.load_state_spec(Path(instance.task_dir)),
        replay=replay,
    )


def tool_call_message(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {"role": "assistant", "content": "", "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}]}


__all__ = ["SamplingConfig"]
