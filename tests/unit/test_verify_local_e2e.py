"""End-to-end without Docker or a model: a scripted learner in a local fixture
environment -> scripted edit -> continuation verification through the real
episode runner's deterministic replay -> preference pair -> immutable export.

Uses Workstream A's `run_episode`, `LocalSession` and `ScriptedPolicy` through a
tiny in-test backend (grading reads /app/answer.txt after the episode)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures" / "verify"))
import verify_builders as vb  # noqa: E402

from evaluation.agents.tools import ToolConfig  # noqa: E402
from learning_loop.editor import ScriptedEditor, SourceContext  # noqa: E402
from learning_loop.envs.local_session import LocalSession  # noqa: E402
from learning_loop.episode import run_episode  # noqa: E402
from learning_loop.events import EventLog  # noqa: E402
from learning_loop.interfaces import EpisodePlan, EpisodeResult  # noqa: E402
from learning_loop.policy import ScriptedPolicy  # noqa: E402
from learning_loop.preferences import build_preference, export_dataset, load_dataset  # noqa: E402
from learning_loop.records import EpisodeRole, PolicySpec, RestoreCapability, Split, TaskInstance  # noqa: E402
from learning_loop.verify import ContinuationVerifier  # noqa: E402

LOG = "\n".join(
    [
        '10.0.0.7 - - [01/Jan/2026:00:00:01 +0000] "GET /a HTTP/1.1" 503 12',
        '10.0.0.7 - - [01/Jan/2026:00:00:01 +0000] "GET /b HTTP/1.1" 500 9',
        '192.168.4.20 - - [01/Jan/2026:00:00:01 +0000] "GET /c HTTP/1.1" 502 9',
        '192.168.4.20 - - [01/Jan/2026:00:00:01 +0000] "GET /d HTTP/1.1" 200 9',
        '192.168.4.20 - - [01/Jan/2026:00:00:01 +0000] "GET /e HTTP/1.1" 200 9',
    ]
) + "\n"
COUNT = "awk '$9 ~ /^5/ {print $1}' /app/logs/access.log | sort | uniq -c | sort -rn | head -1"
SCRIPT = {
    "schema": "scripted_policy/v1",
    "label": "fixture learner for workstream C e2e",
    "rules": [
        {"name": "explore", "when": {"turn": 0}, "respond": {"tool_calls": [{"name": "bash", "arguments": {"command": "ls /app/logs"}}]}},
        {"name": "count", "when": {"last_observation_regex": "access\\.log"}, "respond": {"tool_calls": [{"name": "bash", "arguments": {"command": COUNT}}]}},
        {"name": "write", "when": {"last_observation_regex": "^\\s*\\d+ (\\S+)$"}, "respond": {"tool_calls": [{"name": "write_file", "arguments": {"path": "/app/answer.txt", "content": "${1}\n"}}]}},
        {"name": "done-after-write", "when": {"last_observation_regex": "^Wrote "}, "respond": {"content": "Done."}},
        {"name": "done-after-silent-command", "when": {"last_observation_regex": "\\A\\[exit code 0\\]\\Z"}, "respond": {"content": "Done."}},
    ],
}
EDIT = COUNT + " | awk '{print $2}' > /app/answer.txt"


class LocalScriptedBackend:
    name = "test_local"

    def __init__(self, files_dir: Path, work_root: Path):
        self.files_dir, self.work_root = files_dir, work_root

    def restore_capability(self, instance: TaskInstance) -> RestoreCapability:
        return RestoreCapability.DETERMINISTIC_REPLAY

    async def run(self, instance: TaskInstance, plan: EpisodePlan, out_dir: Path) -> EpisodeResult:
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = ToolConfig(workdir="/app", command_timeout_sec=10, max_output_chars=4000)
        async with LocalSession(self.files_dir, cfg, root=self.work_root) as s:
            run = await run_episode(plan, s, ScriptedPolicy(plan.policy), EventLog(out_dir / "events.jsonl", plan.episode_id))
            answer = s.real_path("/app/answer.txt")
            ok = answer.exists() and answer.read_text().strip() == "10.0.0.7"
        reward = {"reward": 1.0 if ok else 0.0} if run.core.stop_category.value in ("model", "budget") else None
        summary = run.core.to_summary(reward=reward, state_spec=plan.state_spec, events_path=str(out_dir / "events.jsonl"))
        return EpisodeResult(summary=summary, out_dir=out_dir, replay_ok=run.core.replay_ok, replay_mismatches=run.core.replay_mismatches)


def _files(root: Path, log: str) -> Path:
    (root / "logs").mkdir(parents=True)
    (root / "logs" / "access.log").write_text(log)
    return root


def _plan(script: Path) -> EpisodePlan:
    return vb.make_plan("src-e2e", vb.state_spec(fingerprint_paths=["/app"])).model_copy(
        update={"policy": PolicySpec(kind="scripted", scripted_path=str(script)), "budgets": vb.make_plan().budgets.model_copy(update={"max_episode_tokens": None})}
    )


@pytest.fixture()
def world(tmp_path):
    script = tmp_path / "learner.yaml"
    script.write_text(yaml.safe_dump(SCRIPT))
    files = _files(tmp_path / "files", LOG)
    backend = LocalScriptedBackend(files, tmp_path / "work")
    (tmp_path / "work").mkdir()
    return tmp_path, script, backend


async def test_edit_verify_prefer_export_end_to_end(world):
    tmp, script, backend = world
    inst = vb.make_instance(tmp)
    src_res = await backend.run(inst, _plan(script), tmp / "source")
    s = src_res.summary
    assert s.role == EpisodeRole.COLLECT and s.success is True and s.stop_reason == "model_finished" and s.n_requests == 4

    src = SourceContext.load(s, inst)
    edits = tmp / "edits.yaml"
    edits.write_text(yaml.safe_dump({"editor_id": "e2e-fixture-editor", "proposals": [{"instance_id": inst.instance_id, "turn_index": 0, "replacement": {"name": "bash", "arguments": {"command": EDIT}}, "justification": "count and write in one step"}]}))
    prop = await ScriptedEditor(edits).propose(s, src.turns, inst, src.plan.instruction, src.plan.tools, system_prompt=src.plan.system_prompt)
    assert prop.status == "proposed", prop.rejection_reasons

    verifier = ContinuationVerifier(backend, {s.episode_id: src}, root_seed=5, out_root=tmp / "ver")
    rec = await verifier.verify(prop)
    assert rec.accepted, rec.reasons
    o = next(b for b in rec.branches if b.branch == "original")
    e = next(b for b in rec.branches if b.branch == "edited")
    assert o.replay_ok and e.replay_ok and o.continuation_seed == e.continuation_seed
    assert o.episode.n_requests == 3 and e.episode.n_requests == 1  # fresh continuations, not the source suffix
    assert o.cost.shared_prefix_tokens == 0 and o.cost.intervention_request_input_tokens == src.turns[0].usage.input_tokens
    assert o.cost.intervention_tokens_source == "fixture_estimate"
    assert rec.operational_usage.total == o.cost.continuation_tokens + e.cost.continuation_tokens

    pair = build_preference(events=src.events, turns=src.turns, proposal=prop, verification=rec, instance=inst, split=Split.TRAIN, tools=src.plan.tools, learner_checkpoint_id=s.checkpoint_id, cycle=0, run_id="e2e")
    ex = pair.example
    assert [m["role"] for m in ex.prompt] == ["system", "user"]
    assert ex.chosen[0]["tool_calls"][0]["id"] == ex.rejected[0]["tool_calls"][0]["id"]
    assert json.loads(ex.chosen[0]["tool_calls"][0]["function"]["arguments"]) == {"command": EDIT}
    m = export_dataset([pair], tmp / "dataset")
    loaded, _ = load_dataset(tmp / "dataset")
    assert m["n_examples"] == 1 and loaded[0].example == ex


async def test_perturbed_environment_fails_closed(world):
    tmp, script, backend = world
    inst = vb.make_instance(tmp)
    src_res = await backend.run(inst, _plan(script), tmp / "source")
    src = SourceContext.load(src_res.summary, inst)
    prop = (await ScriptedEditor(_edits(tmp, inst, turn=1)).propose(src_res.summary, src.turns, inst, src.plan.instruction, src.plan.tools, system_prompt=src.plan.system_prompt))
    assert prop.status == "proposed", prop.rejection_reasons
    # Same file names (identical `ls` observation) but different content: the fingerprint must catch it.
    other = _files(tmp / "files2", LOG.replace("503", "200"))
    perturbed = LocalScriptedBackend(other, tmp / "work")
    rec = await ContinuationVerifier(perturbed, {src.summary.episode_id: src}, root_seed=5, out_root=tmp / "ver").verify(prop)
    assert not rec.accepted
    assert all(not b.replay_ok for b in rec.branches)
    assert any(r.startswith("replay_failed:") for r in rec.reasons)
    assert all(b.episode.n_requests == 0 for b in rec.branches)  # no model call after a mismatch


def _edits(tmp: Path, inst: TaskInstance, turn: int) -> Path:
    p = tmp / f"edits-{turn}.yaml"
    p.write_text(yaml.safe_dump({"proposals": [{"instance_id": inst.instance_id, "turn_index": turn, "replacement": {"name": "bash", "arguments": {"command": EDIT}}}]}))
    return p
