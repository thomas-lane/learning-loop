"""Grounding heuristic (hindsight constants) and budget-argument checks of the editor."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures" / "verify"))
import verify_builders as vb  # noqa: E402

from learning_loop.editor import budget_violations, contains_token, grounding_violations, validate_proposal  # noqa: E402
from learning_loop.events import load_turns  # noqa: E402
from learning_loop.records import EditProposal, ProposedCall  # noqa: E402

COUNT_INSTRUCTION = "Count the lines containing ERROR across all files in /app/logs/. Write the total to /app/answer.txt."


def turns_of(tmp_path, spec):
    _, path = vb.write_source_episode(tmp_path, spec)
    return load_turns(path)


def ground(args, turns, turn_index=0, instruction=COUNT_INSTRUCTION):
    return grounding_violations(args, turns=turns, turn_index=turn_index, instruction=instruction, tools=vb.TOOLS, system_prompt=vb.SYSTEM)


def proposal(args, turn_index=0, name="bash") -> EditProposal:
    return EditProposal(
        proposal_id="p", source_episode_id="src-ep", instance_id="i", editor_id="ed", status="proposed",
        turn_index=turn_index, tool_call_id=f"call_{turn_index}", replacement=ProposedCall(name=name, arguments=args), justification="j",
    )


def validate(p, turns, instruction=COUNT_INSTRUCTION):
    return validate_proposal(p, turns=turns, tools=vb.TOOLS, instruction=instruction, source_episode_id="src-ep", system_prompt=vb.SYSTEM)


COUNT_TURNS = [
    ("bash", {"command": "ls /app/logs"}, "[exit code 0]\na.log\nb.log"),
    ("bash", {"command": "grep -c ERROR /app/logs/a.log /app/logs/b.log"}, "[exit code 0]\n/app/logs/a.log:5\n/app/logs/b.log:12"),
    ("bash", {"command": "echo 17 > /app/answer.txt"}, "[exit code 0]"),
]


def test_two_digit_count_answer_is_rejected(tmp_path):
    """count-errors (smoke family): answers are 4..28, so the old >=3-digit floor let them through."""
    turns = turns_of(tmp_path, COUNT_TURNS)
    assert ground({"command": "echo 17 > /app/answer.txt"}, turns) == ["ungrounded_constant:17"]
    assert "ungrounded_constant:17" in validate(proposal({"command": "echo 17 > /app/answer.txt"}), turns)
    # written through write_file content, or printf / a here-doc: same
    assert ground({"path": "/app/answer.txt", "content": "17\n"}, turns) == ["ungrounded_constant:17"]
    assert ground({"command": "printf '%s\\n' 17 > /app/answer.txt"}, turns) == ["ungrounded_constant:17"]
    assert ground({"command": "cat > /app/answer.txt <<EOF\n17\nEOF"}, turns) == ["ungrounded_constant:17"]
    # the counts 5 / 12 were only observed at turn 1: hardcoding them at turn 0 is hindsight too
    assert ground({"command": "echo $((5+12)) > /app/answer.txt"}, turns) == ["ungrounded_constant:12", "ungrounded_constant:5"]
    assert ground({"command": "echo 12 > /tmp/x"}, turns) == ["ungrounded_constant:12"]


def test_computing_the_count_is_accepted(tmp_path):
    turns = turns_of(tmp_path, COUNT_TURNS)
    good = {"command": "cat /app/logs/*.log | grep -c ERROR > /app/answer.txt"}
    assert ground(good, turns) == []
    assert validate(proposal(good), turns) == []
    # small numbers outside output contexts (flags, awk field refs) are not constants
    pipe = {"command": "grep -c ERROR /app/logs/*.log | awk -F: '{s+=$2} END {print s}' | head -1 > /app/answer.txt"}
    assert ground(pipe, turns) == []


def test_two_digit_answer_grounded_before_the_decision_is_fine(tmp_path):
    turns = turns_of(tmp_path, COUNT_TURNS)
    # at turn 2 both counts were observed; 17 itself never appears before turn 2 except as
    # the original action at turn 2, which is allowed context
    assert ground({"command": "echo 17 > /app/answer.txt && cat /app/answer.txt"}, turns, turn_index=2) == []


def test_hindsight_only_in_a_later_tool_call_is_rejected(tmp_path):
    spec = [
        ("bash", {"command": "ls /app"}, "[exit code 0]\ndata.csv"),
        ("bash", {"command": "cut -d, -f2 /app/data.csv"}, "[exit code 0]\n500\n120"),
        ("bash", {"command": "echo 620 > /app/answer.txt"}, "[exit code 0]"),
    ]
    turns = turns_of(tmp_path, spec)
    # 620 was computed "in the head" and appears only in the later call's arguments
    assert ground({"command": "echo 620 > /app/answer.txt"}, turns) == ["ungrounded_constant:620"]
    # a quoted string written to the answer only in a later call is caught as well
    spec2 = spec[:2] + [("bash", {"command": "echo 'row-total-ok' > /app/answer.txt"}, "[exit code 0]")]
    turns2 = turns_of(tmp_path / "b", spec2)
    assert ground({"command": "echo 'row-total-ok' > /app/answer.txt"}, turns2) == ["ungrounded_constant:row-total-ok"]


def test_learners_own_later_procedure_may_move_earlier(tmp_path):
    """Merging the learner's later pipeline into turn 0 is the intended edit; the awk
    program appears only in a later tool call but is code, not an observed fact."""
    _, path = vb.write_source_episode(tmp_path)
    turns = load_turns(path)
    good = {"command": vb.PIPELINE + " | awk '{print $2}' > /app/answer.txt"}
    assert grounding_violations(good, turns=turns, turn_index=0, instruction=vb.INSTRUCTION, tools=vb.TOOLS, system_prompt=vb.SYSTEM) == []


@pytest.mark.parametrize(
    "prefix_obs, later_obs, constant",
    [
        ("-rw 1 root 1234 app.log", "total 123", "123"),  # 123 is not grounded by 1234
        ("10.0.0.99 ok", "10.0.0.9 bad", "10.0.0.9"),  # IP prefix of another IP
        ("took 0.123s", "value 123", "123"),  # decimal fraction
        ("/app/report.csv.bak", "/app/report.csv", "/app/report.csv"),
    ],
)
def test_substrings_do_not_ground(tmp_path, prefix_obs, later_obs, constant):
    spec = [
        ("bash", {"command": "ls -l /app"}, "[exit code 0]\n" + prefix_obs),
        ("bash", {"command": "cat /app/x"}, "[exit code 0]\n" + later_obs),
    ]
    turns = turns_of(tmp_path, spec)
    assert ground({"command": f"echo {constant} > /app/answer.txt"}, turns, turn_index=1) == [f"ungrounded_constant:{constant}"]


def test_later_substrings_do_not_trigger(tmp_path):
    # a replacement constant that only occurs INSIDE a longer later token is not hindsight
    spec = [
        ("bash", {"command": "ls /app"}, "[exit code 0]\nx"),
        ("bash", {"command": "cat /app/x"}, "[exit code 0]\nid 1234 and 10.0.0.99"),
    ]
    turns = turns_of(tmp_path, spec)
    assert ground({"command": "echo 123 10.0.0.9 > /tmp/y"}, turns) == []


def test_contains_token_boundaries():
    assert contains_token("a 123 b", "123") and contains_token("x=123.", "123") and contains_token("(17)", "17")
    assert not contains_token("1234", "123") and not contains_token("0.123", "123") and not contains_token("123.5", "123")
    assert not contains_token("10.0.0.99", "10.0.0.9") and contains_token("ip 10.0.0.9:80", "10.0.0.9")
    assert contains_token("cat /app/logs/a.log", "/app/logs") and not contains_token("/data/app/logs", "/app/logs")


# --------------------------------------------------------------------------- #
# Budget-like arguments
# --------------------------------------------------------------------------- #


def test_replacement_cannot_change_timeout(tmp_path):
    turns = turns_of(tmp_path, COUNT_TURNS)
    good = {"command": "cat /app/logs/*.log | grep -c ERROR > /app/answer.txt"}
    assert validate(proposal({**good, "timeout_sec": 3600}), turns) == ["budget_argument_changed:timeout_sec"]
    assert validate(proposal({**good, "timeout_sec": 5}), turns) == ["budget_argument_changed:timeout_sec"]  # any change
    assert validate(proposal(good), turns) == []


def test_replacement_must_keep_original_timeout(tmp_path):
    spec = [("bash", {"command": "ls /app/logs", "timeout_sec": 30}, "[exit code 0]\na.log")] + COUNT_TURNS[1:]
    turns = turns_of(tmp_path, spec)
    good = {"command": "cat /app/logs/*.log | grep -c ERROR > /app/answer.txt"}
    assert validate(proposal({**good, "timeout_sec": 30}), turns) == []
    assert "budget_argument_changed:timeout_sec" in validate(proposal(good), turns)  # dropping it changes the budget too
    assert "budget_argument_changed:timeout_sec" in validate(proposal({**good, "timeout_sec": 300}), turns)


def test_budget_violations_helper():
    orig = {"id": "c", "type": "function", "function": {"name": "bash", "arguments": '{"command":"ls","timeout_sec":60}'}}
    assert budget_violations(orig, {"command": "ls -1", "timeout_sec": 60}) == []
    assert budget_violations(orig, {"command": "ls -1", "timeout_sec": 61, "max_output_chars": 10}) == [
        "budget_argument_changed:timeout_sec",
        "budget_argument_changed:max_output_chars",
    ]
    assert budget_violations(None, {"command": "ls", "timeout": 5}) == ["budget_argument_changed:timeout"]


def test_output_contexts():
    from learning_loop.editor import extract_constants

    assert "17" in extract_constants({"command": "cd /app && echo 17 > answer.txt"})
    assert "17" in extract_constants({"command": "python3 -c 'print(17)' > /app/answer.txt"})
    assert "17" in extract_constants({"command": "x=$(echo 17); echo $x > /app/answer.txt"})
    assert "17" in extract_constants({"command": "python3 -c \"open('/app/answer.txt','w').write('17')\""})  # short quoted number
    # awk's print is program text, not a shell output: its field refs and paths are not output constants
    got = extract_constants({"command": "awk '$9 ~ /^5/ {print $1}' /app/logs/access.log | head -1"})
    assert "1" not in got and "5" not in got and "9" not in got


# Mirrors a proposal a live Qwen3-1.7B editor made (and an earlier grounding rule accepted):
# after a full `cat` of the logs, jump to writing the count that only a later `wc -l` printed.
CAT_TURNS = [
    ("bash", {"command": "ls data"}, "[exit code 0]\na.log\nb.log"),
    ("bash", {"command": "cat data/*.log"},
     "[exit code 0]\n2026-09-10 12:00:01 ERROR disk\n2026-09-10 12:00:02 INFO ok\nreq=10 ERROR timeout"),
    ("bash", {"command": "grep -h ERROR data/*.log | wc -l"}, "[exit code 0]\n10"),
    ("write_file", {"path": "/app/answer.txt", "content": "10\n"}, "Wrote 3 characters to /app/answer.txt"),
]


def test_written_answer_not_grounded_by_coincidental_prefix_tokens(tmp_path):
    turns = turns_of(tmp_path, CAT_TURNS)
    # "10" occurs in the prefix only inside longer lines (a date, a request id): not a result
    assert ground({"path": "/app/answer.txt", "content": "10"}, turns, turn_index=2) == ["ungrounded_constant:10"]


def test_written_value_grounded_when_shown_as_a_result_line(tmp_path):
    turns = turns_of(tmp_path, CAT_TURNS)
    # at turn 3 the learner has been shown `10` as the output of wc -l: writing it is grounded
    assert ground({"path": "/app/answer.txt", "content": "10"}, turns, turn_index=3) == []
