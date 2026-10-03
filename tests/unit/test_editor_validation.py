"""Editor view boundary, proposal parsing and pre-execution validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures" / "verify"))
import verify_builders as vb  # noqa: E402

from learning_loop.core.interfaces import PolicyDecision  # noqa: E402
from learning_loop.core.records import EditProposal, PolicySpec, ProposedCall, SamplingConfig, Usage  # noqa: E402
from learning_loop.editing.editor import (  # noqa: E402
    LLMEditor,
    ScriptedEditor,
    build_trajectory_view,
    extract_constants,
    hidden_path_references,
    editor_tools,
    make_edited_message,
    proposal_from_tool_calls,
    select_sources,
    validate_proposal,
)
from learning_loop.episodes.events import load_turns  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PROMPT = REPO / "prompts" / "editor" / "v2.md"
GOOD_CMD = vb.PIPELINE + " | awk '{print $2}' > /app/answer.txt"


def source(tmp_path, **kw):
    summary, path = vb.write_source_episode(tmp_path, **kw)
    return summary, load_turns(path)


def prop(turn_index=0, name="bash", args=None, call_id=None, src="src-ep") -> EditProposal:
    return EditProposal(
        proposal_id="p1",
        source_episode_id=src,
        instance_id="log-triage/easy/s0",
        editor_id="ed",
        status="proposed",
        turn_index=turn_index,
        tool_call_id=call_id if call_id is not None else f"call_{turn_index}",
        replacement=ProposedCall(name=name, arguments=args if args is not None else {"command": GOOD_CMD}),
        justification="merge exploration into one pipeline",
    )


def check(p, turns, **kw):
    return validate_proposal(p, turns=turns, tools=vb.TOOLS, instruction=vb.INSTRUCTION, source_episode_id="src-ep", system_prompt=vb.SYSTEM, **kw)


def test_valid_single_call_edit_passes(tmp_path):
    _, turns = source(tmp_path)
    assert check(prop(), turns) == []


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({0: {"extra_calls": 1}}, "multi_tool_call_turn:2"),
        ({0: {"malformed": True}}, "malformed_turn"),
        ({0: {"repaired": True}}, "repaired_turn"),
        ({0: {"reasoning": "I should list the logs first."}}, "nonempty_reasoning"),
        ({0: {"content": "Let me look."}}, "nonempty_assistant_content"),
    ],
)
def test_ineligible_turns_rejected(tmp_path, overrides, expected):
    _, turns = source(tmp_path, overrides=overrides)
    reasons = check(prop(), turns)
    assert expected in reasons


def test_empty_reasoning_string_is_allowed(tmp_path):
    _, turns = source(tmp_path, overrides={0: {"reasoning": "  "}})
    assert check(prop(), turns) == []


def test_final_answer_turn_and_missing_turn(tmp_path):
    _, turns = source(tmp_path)
    assert "no_tool_call_turn" in check(prop(turn_index=4), turns)
    assert check(prop(turn_index=9), turns) == ["turn_not_found:9"]


def test_unknown_tool_and_schema_violation(tmp_path):
    _, turns = source(tmp_path)
    assert "unknown_tool:python" in check(prop(name="python", args={"code": "print(1)"}), turns)
    reasons = check(prop(args={"cmd": "ls"}), turns)
    assert any(r.startswith("schema_violation:") and "command" in r for r in reasons)
    reasons = check(prop(args={"command": "ls", "timeout_sec": "soon"}), turns)
    assert any(r.startswith("schema_violation:") for r in reasons)


def test_identical_replacement_rejected(tmp_path):
    _, turns = source(tmp_path)
    assert "identical_replacement" in check(prop(args={"command": "ls /app/logs"}), turns)


def test_id_and_source_mismatch(tmp_path):
    _, turns = source(tmp_path)
    assert "tool_call_id_mismatch" in check(prop(call_id="call_zzz"), turns)
    assert "source_trajectory_mismatch" in check(prop(src="other"), turns)


@pytest.mark.parametrize(
    "cmd, hidden",
    [
        ("cat /tests/test.sh", "hidden_path:/tests"),
        ("ls /solution", "hidden_path:/solution"),
        ("cat /logs/verifier/reward.txt", "hidden_path:/logs/verifier"),
        ("cd /app && cat ../tests/test_hidden.py", "hidden_path:/tests"),
    ],
)
def test_hidden_paths_rejected(tmp_path, cmd, hidden):
    _, turns = source(tmp_path)
    assert hidden in check(prop(args={"command": cmd}), turns)


def test_workdir_tests_are_not_hidden():
    assert hidden_path_references({"command": "python -m pytest ./tests /app/tests -q"}) == []


def test_ungrounded_constant_rejected_but_grounded_accepted(tmp_path):
    _, turns = source(tmp_path)
    hard = {"command": "echo 10.0.0.7 > /app/answer.txt"}
    # At turn 0 the IP was only observed later (turn 1/2 observations, final answer).
    assert "ungrounded_constant:10.0.0.7" in check(prop(turn_index=0, args=hard), turns)
    # At turn 3 it is in the prefix (turn 2 observation), so it is grounded.
    p3 = prop(turn_index=3, name="bash", args=hard)
    assert not [r for r in check(p3, turns) if r.startswith("ungrounded")]


def test_ungrounded_path_from_later_observation(tmp_path):
    turns_spec = [
        ("bash", {"command": "ls /app/logs"}, "[exit code 0]\n/app/logs/secret-rotated.log"),
        ("bash", {"command": "cat /app/logs/secret-rotated.log"}, "[exit code 0]\nx"),
    ]
    _, turns = source(tmp_path, turns=turns_spec)
    # the path is visible before turn 1 but not before turn 0
    p0 = prop(turn_index=0, args={"command": "cat /app/logs/secret-rotated.log | wc -l"})
    assert "ungrounded_constant:/app/logs/secret-rotated.log" in check(p0, turns)
    p1 = prop(turn_index=1, args={"command": "wc -l /app/logs/secret-rotated.log"})
    assert check(p1, turns) == []


def test_extract_constants_kinds():
    got = extract_constants({"command": "grep \"GET /api\" /var/log/x.log | awk '$9>=500' ; echo 10.1.2.3 7 42"})
    assert "10.1.2.3" in got and "500" in got  # numbers are extracted inside quoted literals too
    assert "$9>=500" in got and "GET /api" in got and "/var/log/x.log" in got
    assert "42" in got and "7" in got  # printed numbers count at any length
    small = extract_constants({"command": "sort -k2 x | head -1 | awk '{print $2}' > /app/answer.txt"})
    assert "1" not in small and "2" not in small  # flags / $N field refs are not output constants


def test_view_contains_only_declared_data(tmp_path):
    summary, turns = source(tmp_path, overrides={1: {"reasoning": "hmm"}})
    summary = summary.model_copy(update={"trial_dir": "/secret/trial", "reward": {"reward": 1.0, "hidden_detail": 3.0}})
    view = build_trajectory_view(turns, vb.INSTRUCTION, vb.TOOLS, outcome=summary, source_trajectory_id="src-ep", system_prompt=vb.SYSTEM)
    assert set(view) == {"view_version", "source_trajectory_id", "instruction", "system_prompt", "tools", "turns", "editable_turns",
                         "later_observations_included", "outcome"}
    blob = json.dumps(view)
    assert "/secret/trial" not in blob and "events.jsonl" not in blob and "hidden_detail" not in blob
    assert set(view["outcome"]) == {"success", "partial_reward", "total_tokens", "n_requests", "n_tool_calls"}
    t0 = view["turns"][0]
    assert t0["observations"] == [{"tool_call_id": "call_0", "observation": vb.DEFAULT_TURNS[0][2]}]
    assert t0["eligible"] is True
    assert view["turns"][1]["eligible"] is False and "nonempty_reasoning" in view["turns"][1]["ineligible_reasons"]
    assert view["turns"][-1]["eligible"] is False  # final answer
    assert view["editable_turns"] == [t["turn_index"] for t in view["turns"] if t["eligible"]] and 1 not in view["editable_turns"]


def test_view_without_later_observations(tmp_path):
    _, turns = source(tmp_path, overrides={0: {"content": "x"}})
    view = build_trajectory_view(turns, vb.INSTRUCTION, vb.TOOLS, include_later_observations=False, source_trajectory_id="src-ep")
    # earliest eligible turn is 1: its own and later observations are hidden
    assert [t["turn_index"] for t in view["turns"]] == [0, 1]
    assert "observations" in view["turns"][0] and "observations" not in view["turns"][1]
    assert "10.0.0.7" not in json.dumps(view) and "outcome" not in view


def test_edited_message_keeps_id_and_content(tmp_path):
    _, turns = source(tmp_path)
    orig = turns[0].assistant_message
    edited = make_edited_message(orig, ProposedCall(name="bash", arguments={"command": "ls"}))
    assert edited["tool_calls"][0]["id"] == "call_0" and edited["content"] == orig["content"]
    assert edited["tool_calls"][0]["function"]["arguments"] == '{"command":"ls"}'
    assert orig["tool_calls"][0]["function"]["arguments"] == '{"command":"ls /app/logs"}'  # original untouched


def _answer(*calls):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"e{i}", "type": "function", "function": {"name": n, "arguments": a if isinstance(a, str) else json.dumps(a)}}
        for i, (n, a) in enumerate(calls)]}


def test_editor_tools_wrap_each_learner_tool():
    tools = editor_tools(vb.TOOLS, [0, 2])
    names = [t["function"]["name"] for t in tools]
    assert names == [f"replace_with_{t['function']['name']}" for t in vb.TOOLS] + ["abstain"]
    bash = next(t for t in tools if t["function"]["name"] == "replace_with_bash")["function"]["parameters"]
    learner = next(t for t in vb.TOOLS if t["function"]["name"] == "bash")["function"]["parameters"]
    assert bash["properties"]["edit_turn"] == {"type": "integer", "enum": [0, 2], "description": bash["properties"]["edit_turn"]["description"]}
    assert set(learner["properties"]) < set(bash["properties"]) and bash["required"][0] == "edit_turn"
    with pytest.raises(ValueError, match="reserve"):
        editor_tools([{"type": "function", "function": {"name": "x", "parameters": {"type": "object", "properties": {"edit_turn": {}}}}}], [0])


def test_proposal_from_tool_calls(tmp_path):
    _, turns = source(tmp_path)
    ids = dict(proposal_id="p", source_episode_id="s", instance_id="i", editor_id="e")
    p = proposal_from_tool_calls(_answer(("replace_with_bash", {"edit_turn": 0, "command": "ls -la", "edit_justification": "j"})), turns, **ids)
    assert (p.status, p.turn_index, p.tool_call_id, p.justification) == ("proposed", 0, "call_0", "j")
    assert p.replacement == ProposedCall(name="bash", arguments={"command": "ls -la"})  # editor-only fields stripped
    assert proposal_from_tool_calls(_answer(("abstain", {"edit_justification": "fine"})), turns, **ids).status == "abstained"
    for msg, reason in [
        ({"role": "assistant", "content": '{"decision": "edit"}'}, "no_tool_call"),
        (_answer(("abstain", {}), ("abstain", {})), "multiple_tool_calls:2"),
        (_answer(("rm_rf", {})), "unknown_editor_tool:rm_rf"),
        (_answer(("replace_with_bash", {"command": "ls"})), "response_schema:edit_turn"),
        (_answer(("replace_with_bash", "not json")), "response_schema:arguments"),
    ]:
        p = proposal_from_tool_calls(msg, turns, **ids)
        assert p.status == "invalid" and p.rejection_reasons[0] == reason, (reason, p.rejection_reasons)
    p = proposal_from_tool_calls({"role": "assistant", "content": "<|tool_call>call:x{"}, turns, parse_errors=["unterminated"], **ids)
    assert p.rejection_reasons == ["no_tool_call", "unparsed_tool_call:unterminated"]


def test_select_sources_only_successful_collection(tmp_path):
    ok, _ = vb.write_source_episode(tmp_path / "a")
    bad, _ = vb.write_source_episode(tmp_path / "b", episode_id="b", success=False)
    assert select_sources([ok, bad]) == [ok]


class FakePolicy:
    def __init__(self, answer: dict | str, infra_error: str | None = None):
        self.spec = PolicySpec(kind="openai", served_model_name="base", sampling=SamplingConfig(temperature=0.0, max_output_tokens=512))
        self.answer = answer if isinstance(answer, dict) else {"role": "assistant", "content": answer}
        self.infra_error = infra_error
        self.calls = []

    async def decide(self, messages, tools, seed):
        self.calls.append((messages, tools, seed))
        return PolicyDecision(
            raw_request={}, raw_response={"id": "r"}, history_message=self.answer,
            finish_reason="stop", usage=Usage(input_tokens=900, output_tokens=60), latency_sec=0.5,
            infra_error=self.infra_error, seed_sent=seed,
        )


def _editor(policy, **kw):
    return LLMEditor(policy, mode="initial_policy", checkpoint_id="base", prompt_path=PROMPT, root_seed=5, **kw)


async def test_llm_editor_valid_proposal_and_identity(tmp_path):
    summary, turns = source(tmp_path)
    inst = vb.make_instance(tmp_path)
    resp = _answer(("replace_with_bash", {"edit_turn": 0, "command": GOOD_CMD, "edit_justification": "one pipeline"}))
    pol = FakePolicy(resp)
    ed = _editor(pol)
    p = await ed.propose(summary, turns, inst, vb.INSTRUCTION, vb.TOOLS, system_prompt=vb.SYSTEM)
    assert p.status == "proposed" and p.rejection_reasons == [] and p.editor_id == ed.editor_id
    assert p.usage.input_tokens == 900
    messages, tools, seed = pol.calls[0]
    assert [t["function"]["name"] for t in tools] == [t["function"]["name"] for t in editor_tools(vb.TOOLS, [0])]
    assert messages[0]["content"] == PROMPT.read_text()
    assert "one pipeline" not in json.dumps(messages)  # nothing editor-generated is fed back
    # identity: stable, and independent of the learner checkpoint that produced the source
    assert _editor(FakePolicy(resp)).editor_id == ed.editor_id
    p2 = tmp_path / "v2.md"
    p2.write_text(PROMPT.read_text() + "\nextra")
    assert LLMEditor(FakePolicy(resp), mode="initial_policy", checkpoint_id="base", prompt_path=p2, root_seed=5).editor_id != ed.editor_id
    assert LLMEditor(FakePolicy(resp), mode="current_learner", checkpoint_id="c1", prompt_path=PROMPT, root_seed=5).editor_id != ed.editor_id
    # proposal id and seed are deterministic
    p_again = await _editor(FakePolicy(resp)).propose(summary, turns, inst, vb.INSTRUCTION, vb.TOOLS, system_prompt=vb.SYSTEM)
    assert p_again.proposal_id == p.proposal_id and pol.calls[0][2] == p_again.raw_response["seed"]


async def test_llm_editor_abstain_garbage_and_infra(tmp_path):
    summary, turns = source(tmp_path)
    inst = vb.make_instance(tmp_path)
    p = await _editor(FakePolicy(_answer(("abstain", {"edit_justification": "fine as is"})))).propose(summary, turns, inst, vb.INSTRUCTION, vb.TOOLS)
    assert p.status == "abstained" and p.justification == "fine as is"
    p = await _editor(FakePolicy("I think turn 0 is fine")).propose(summary, turns, inst, vb.INSTRUCTION, vb.TOOLS)
    assert p.status == "invalid" and p.rejection_reasons == ["no_tool_call"]
    p = await _editor(FakePolicy("", infra_error="connection refused")).propose(summary, turns, inst, vb.INSTRUCTION, vb.TOOLS)
    assert p.status == "invalid" and p.rejection_reasons == ["editor_infra_error:connection refused"]
    hard = _answer(("replace_with_bash", {"edit_turn": 0, "command": "echo 10.0.0.7 > /app/answer.txt", "edit_justification": "x"}))
    p = await _editor(FakePolicy(hard)).propose(summary, turns, inst, vb.INSTRUCTION, vb.TOOLS, system_prompt=vb.SYSTEM)
    assert p.status == "invalid" and "ungrounded_constant:10.0.0.7" in p.rejection_reasons


async def test_scripted_editor(tmp_path):
    summary, turns = source(tmp_path)
    inst = vb.make_instance(tmp_path)
    script = tmp_path / "edits.yaml"
    script.write_text(
        "editor_id: scripted-test\nproposals:\n"
        f"  - instance_id: log-triage/easy/s0\n    turn_index: 0\n    replacement: {{name: bash, arguments: {{command: {json.dumps(GOOD_CMD)}}}}}\n    justification: merge\n"
        "  - instance_id: other/x\n    decision: abstain\n"
    )
    ed = ScriptedEditor(script)
    p = await ed.propose(summary, turns, inst, vb.INSTRUCTION, vb.TOOLS, system_prompt=vb.SYSTEM)
    assert p.status == "proposed" and p.tool_call_id == "call_0" and p.editor_id == "scripted-test"
    other = summary.model_copy(update={"instance_id": "nobody"})
    p = await ed.propose(other, turns, inst.model_copy(update={"instance_id": "nobody"}), vb.INSTRUCTION, vb.TOOLS)
    assert p.status == "abstained"


def test_repo_scripted_fixture_loads():
    ed = ScriptedEditor(REPO / "tests" / "fixtures" / "verify" / "scripted_edits.yaml")
    assert ed.entries and ed.editor_id


async def test_make_editor_and_propose_for(tmp_path):
    from learning_loop.core.config import EditorConfig
    from learning_loop.core.records import CheckpointRef
    from learning_loop.editing.editor import make_editor

    summary, _ = vb.write_source_episode(tmp_path / "item")
    (tmp_path / "item" / "summary.json").write_text(summary.model_copy(update={"events_path": None}).model_dump_json())
    inst = vb.make_instance(tmp_path)
    ckpt = CheckpointRef(checkpoint_id="base:q", model_profile="q", base_model="Q/q", base_revision="0" * 40)
    scripted = make_editor(EditorConfig(mode="scripted", scripted_path=str(REPO / "tests/fixtures/verify/scripted_edits.yaml")), ckpt, None)
    p = await scripted.propose_for(proposal_id="prop-x", source=summary, source_dir=tmp_path / "item", instance=inst, seed=3)
    assert p.proposal_id == "prop-x" and p.status == "proposed", p.rejection_reasons
    resp = _answer(("replace_with_bash", {"edit_turn": 0, "command": GOOD_CMD, "edit_justification": "j"}))
    pol = FakePolicy(resp)
    llm = make_editor(EditorConfig(mode="initial_policy"), ckpt, None, policy=pol)
    p = await llm.propose_for(proposal_id="prop-y", source=summary, source_dir=tmp_path / "item", instance=inst, seed=42)
    assert p.status == "proposed" and pol.calls[0][2] == 42 and p.proposal_id == "prop-y"
    # the system prompt of the view comes from the source plan (learner's own prompt)
    assert json.loads(pol.calls[0][0][1]["content"])["system_prompt"] == vb.SYSTEM
    # fixed editor identity: the same initial checkpoint gives the same id regardless of the learner
    assert make_editor(EditorConfig(mode="initial_policy"), ckpt, None, policy=FakePolicy(resp)).editor_id == llm.editor_id


async def test_trajectory_without_editable_turns_is_not_sent(tmp_path):
    summary, turns = source(tmp_path, overrides={i: {"content": "thinking aloud"} for i in range(len(vb.DEFAULT_TURNS))})
    pol = FakePolicy(_answer(("abstain", {"edit_justification": "x"})))
    p = await _editor(pol).propose(summary, turns, vb.make_instance(tmp_path), vb.INSTRUCTION, vb.TOOLS)
    assert p.status == "abstained" and p.rejection_reasons == ["skipped:no_editable_turns"] and pol.calls == []
