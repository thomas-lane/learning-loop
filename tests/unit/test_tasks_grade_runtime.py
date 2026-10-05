"""The shared grader (tests/grade.py in every task): grader kinds, artifact reading,
reward-file hardening, and refusing to grade when the probe or sandbox fails."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from learning_loop.tasks.runtime import grade as g

FORGED = "class A(int):\n    def __eq__(self, o):\n        return True\n\n\ndef add(a, b):\n    return A(0)\n"
CHECKS = {"kind": "checks", "path": "/app/calc.py", "module": "calc", "checks": [
    {"name": "add", "func": "add", "kind": "value", "args": [2, 3], "expected": 5},
    {"name": "add_empty_raises", "func": "add", "kind": "raises", "args": [None, None]},
]}


def reader(files: dict[str, bytes | None]):
    return lambda p: files.get(p)


@pytest.mark.parametrize("text,reward", [(b"42\n", 1.0), (b"  42  ", 1.0), (b"42.0", 0.0), (b"\xff\xfe", 0.0), (None, 0.0)])
def test_exact(text, reward):
    assert g.grade({"kind": "exact", "path": "/app/a", "expected": "42"}, reader({"/app/a": text}))[0] == reward


@pytest.mark.parametrize("text,reward", [(b"3.14", 1.0), (b"3.1405", 1.0), (b"3.2", 0.0), (b"nan", 0.0), (b"pi", 0.0)])
def test_numeric(text, reward):
    key = {"kind": "numeric", "path": "/app/a", "expected": 3.14, "abs_tol": 0.001}
    assert g.grade(key, reader({"/app/a": text}))[0] == reward


@pytest.mark.parametrize("text,reward", [(b'{"b": [1, 2], "a": 1}', 1.0), (b'{"a": 1, "b": [2, 1]}', 0.0), (b'{"a": NaN}', 0.0), (b"{", 0.0)])
def test_json_ignores_key_order_only(text, reward):
    key = {"kind": "json", "path": "/app/a", "expected": {"a": 1, "b": [1, 2]}}
    assert g.grade(key, reader({"/app/a": text}))[0] == reward


def test_checks_in_process_accepts_only_plain_observations():
    good = b"def add(a, b):\n    if a is None:\n        raise ValueError('empty')\n    return a + b\n"
    assert g.grade(CHECKS, reader({"/app/calc.py": good}), trusted=True)[0] == 1.0
    assert g.grade(CHECKS, reader({"/app/calc.py": FORGED.encode()}), trusted=True)[0] == 0.0
    assert g.grade(CHECKS, reader({"/app/calc.py": b"def add(:"}), trusted=True)[0] == 0.0
    assert g.grade(CHECKS, reader({}), trusted=True)[0] == 0.0


def test_judge_rejects_observations_of_the_wrong_shape():
    checks = CHECKS["checks"]
    assert g.judge(checks, {"add": {"value": 5}, "add_empty_raises": {"raised": "ValueError", "value_error": True}}) == {"add": True, "add_empty_raises": True}
    assert g.judge(checks, {"add": {"value": True}}) == {"add": False, "add_empty_raises": False}  # bool is not a number
    assert g.judge(checks, {"add": {"value": 5, "extra": 1}})["add"] is False
    assert g.judge(checks, "not a dict") == {"add": False, "add_empty_raises": False}


def test_checks_outside_the_sandbox_refuse_to_grade():
    if os.geteuid() == 0:
        pytest.skip("runs as root")
    with pytest.raises(g.GraderError, match="must run as root"):
        g.grade(CHECKS, reader({"/app/calc.py": FORGED.encode()}), key_dir="/nonexistent")


def test_unknown_kind():
    with pytest.raises(g.GraderError):
        g.grade({"kind": "vibes"}, reader({}))


def test_read_artifact_refuses_symlinks_directories_and_oversized_files(tmp_path, monkeypatch):
    (tmp_path / "app").mkdir()
    (tmp_path / "secret").write_text("42")
    (tmp_path / "app" / "link").symlink_to(tmp_path / "secret")
    (tmp_path / "app" / "dir").mkdir()
    (tmp_path / "app" / "ok").write_text("42")
    (tmp_path / "app" / "big").write_bytes(b"x" * 11)
    monkeypatch.setattr(g, "MAX_ARTIFACT_BYTES", 10)
    assert g.read_artifact(str(tmp_path), "/app/ok") == b"42"
    for name in ("link", "dir", "big", "missing"):
        assert g.read_artifact(str(tmp_path), f"/app/{name}") is None, name


def test_write_reward_replaces_planted_files_without_following_links(tmp_path):
    out = tmp_path / "verifier"
    out.mkdir()
    victim = tmp_path / "victim"
    victim.write_text("untouched")
    (out / "reward.txt").write_text("1")
    (out / "reward.json").symlink_to(victim)
    g.write_reward(str(out / "reward.json"), 0.25)
    assert json.loads((out / "reward.json").read_text()) == {"reward": 0.25}
    assert not (out / "reward.txt").exists() and victim.read_text() == "untouched"
    assert not (out / "reward.json").is_symlink()


def _task(tmp_path: Path, key: dict, answer: str | None) -> tuple[Path, Path]:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "key.json").write_text(json.dumps(key))
    if answer is not None:
        (tmp_path / "app").mkdir()
        (tmp_path / "app" / "answer.txt").write_text(answer)
    out = tmp_path / "out"
    out.mkdir()
    return tests / "key.json", out / "reward.json"


def test_main_grades_from_root_and_key(tmp_path):
    key, out = _task(tmp_path, {"kind": "exact", "path": "/app/answer.txt", "expected": "7"}, "7\n")
    assert g.main(["--root", str(tmp_path), "--key", str(key), "--out", str(out)]) == 0
    assert json.loads(out.read_text()) == {"reward": 1.0}


def test_main_refuses_to_grade_when_the_probe_fails(tmp_path):
    key, out = _task(tmp_path, {"kind": "exact", "path": "/app/answer.txt", "expected": "7"}, "7\n")
    out.write_text('{"reward": 1.0}')  # planted
    rc = g.main(["--root", str(tmp_path), "--key", str(key), "--out", str(out), "--probe", json.dumps({"tools": ["no-such-tool-xyz"]})])
    assert rc == g.EXIT_PROBE_FAILED and not out.exists()


def test_main_writes_no_reward_on_grader_error(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("runs as root")
    key, out = _task(tmp_path, CHECKS, None)
    assert g.main(["--root", str(tmp_path), "--key", str(key), "--out", str(out)]) == g.EXIT_GRADER_ERROR
    assert not out.exists()


def test_a_checks_key_without_checks_is_an_error():
    with pytest.raises(g.GraderError, match="at least one check"):
        g.grade(CHECKS | {"checks": []}, reader({}), trusted=True)


def test_judge_rejects_every_forged_observation_shape():
    checks = [
        {"name": "v", "func": "mean", "kind": "value", "args": [[1, 2]], "expected": 1.5},
        {"name": "r", "func": "mean", "kind": "raises", "args": [[]]},
        {"name": "m", "func": "median", "kind": "no_mutation", "args": [[3, 1, 2]]},
    ]
    good = {"v": {"value": 1.5}, "r": {"raised": "ValueError", "value_error": True}, "m": {"after": [3, 1, 2]}}
    assert g.judge(checks, good) == {"v": True, "r": True, "m": True}
    forged = [
        {"v": {"value": True}, "r": {"raised": "X", "value_error": 1}, "m": {"after": [1, 2, 3]}},
        {"v": {"value": "1.5"}, "r": {"raised": "X", "value_error": "true"}, "m": {"after": [3, 1, 2], "x": 1}},
        {"v": {"value": 1.5, "ok": True}, "r": {"returned": True}, "m": None},
        {"v": True, "r": True, "m": True},
        {"v": {"value": float("nan")}},
        [],
        "all passed",
    ]
    for obs in forged:
        assert not any(g.judge(checks, obs).values()), obs


def test_observe_reports_plain_values_not_objects():
    class AlwaysEqual(float):
        __hash__ = float.__hash__

        def __eq__(self, other):
            return True

    class NotANumber:
        def __eq__(self, other):
            return True

    lib = {"mean": lambda xs: AlwaysEqual(0), "median": lambda xs: NotANumber()}
    checks = [{"name": "a", "func": "mean", "kind": "value", "args": [[1, 2]]}, {"name": "b", "func": "median", "kind": "value", "args": [[1]]}]
    obs = json.loads(json.dumps(g.observe(lib, checks)))
    assert obs == {"a": {"value": 0.0}, "b": {"value": None}}
    assert g.judge([{**checks[0], "expected": 1.5}, {**checks[1], "expected": 1}], obs) == {"a": False, "b": False}


def test_main_refuses_files_that_differ_from_the_rendered_digest(tmp_path, monkeypatch):
    key, out = _task(tmp_path, {"kind": "exact", "path": "/app/answer.txt", "expected": "7"}, "7\n")
    here = Path(g.__file__).parent
    monkeypatch.setattr(g, "__file__", str(key.parent / "grade.py"))
    for name in ("grade.py", "probe.py"):
        (key.parent / name).write_bytes((here / name).read_bytes())
    monkeypatch.setenv(g.TESTS_DIGEST_ENV, g.tests_digest(str(key.parent)))
    args = ["--root", str(tmp_path), "--key", str(key), "--out", str(out)]
    assert g.main(args) == 0 and json.loads(out.read_text()) == {"reward": 1.0}
    key.write_text(json.dumps({"kind": "exact", "path": "/app/answer.txt", "expected": "8"}))  # a stale/other key
    assert g.main(args) == g.EXIT_STALE_FILES and not out.exists()
