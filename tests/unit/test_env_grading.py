"""No false-positive rewards from no-op or malicious outputs (local fixture backend)."""

from __future__ import annotations

from _env_helpers import FIXTURES, count_errors_instance, make_plan, scripted_spec

from learning_loop.backends import LocalFixtureBackend
from learning_loop.events import load_turns
from learning_loop.storage import read_json


async def test_nop_scores_zero(tmp_path):
    inst = count_errors_instance(tmp_path / "tasks")
    nop = tmp_path / "nop.yaml"
    nop.write_text("schema: scripted_policy/v1\nrules:\n  - {name: nop, respond: {content: done}}\n")
    res = await LocalFixtureBackend().run(inst, make_plan(inst, "nop", scripted_spec(nop)), tmp_path / "o")
    assert res.summary.reward == {"reward": 0.0} and res.summary.success is False
    assert read_json(res.out_dir / "grading.json")["artifacts"] == {"/app/answer.txt": "missing"}


async def test_reference_command_scores_one(tmp_path):
    inst = count_errors_instance(tmp_path / "tasks", "count-errors/medium/s201")
    ref = tmp_path / "ref.yaml"
    ref.write_text(
        "schema: scripted_policy/v1\nrules:\n"
        "  - {name: done, when: {min_turn: 1}, respond: {content: done}}\n"
        "  - {name: solve, respond: {tool_calls: [{name: bash, arguments: {command: \"find /app/data -type f -name '*.log' -exec cat {} + | grep -c ERROR > /app/answer.txt\"}}]}}\n"
    )
    res = await LocalFixtureBackend().run(inst, make_plan(inst, "ref", scripted_spec(ref)), tmp_path / "o")
    assert res.summary.reward == {"reward": 1.0} and res.summary.success


async def test_malicious_outputs_get_no_reward(tmp_path):
    inst = count_errors_instance(tmp_path / "tasks")
    res = await LocalFixtureBackend().run(inst, make_plan(inst, "evil", scripted_spec(FIXTURES / "malicious_policy.yaml")), tmp_path / "o")
    s = res.summary
    assert s.reward == {"reward": 0.0} and s.success is False and s.partial_reward == 0.0
    grading = read_json(res.out_dir / "grading.json")
    assert grading["artifacts"]["/app/answer.txt"] == "skipped:symlink"
    turns = load_turns(res.out_dir / "events.jsonl")
    # hidden tests/solution are not in the learner's environment
    peek = turns[1].tool_executions[0].observation
    assert "No such file or directory" in peek and "EXPECTED" not in peek and "grade" not in peek
    assert s.n_tool_calls == 5
