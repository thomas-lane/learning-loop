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
    return g.MemoryView({p: (data, 0o644) for p, data in files.items() if data is not None})


@pytest.mark.parametrize("text,reward", [(b"42\n", 1.0), (b"  42  ", 1.0), (b"42.0", 0.0), (b"\xff\xfe", 0.0), (None, 0.0)])
def test_exact(text, reward):
    assert g.grade({"kind": "exact", "path": "/app/a", "expected": "42"}, reader({"/app/a": text}))[0] == reward


@pytest.mark.parametrize("text,reward", [(b"3.14", 1.0), (b"3.1405", 1.0), (b"3.2", 0.0), (b"nan", 0.0), (b"pi", 0.0)])
def test_numeric(text, reward):
    key = {"kind": "numeric", "path": "/app/a", "expected": 3.14, "abs_tol": 0.001}
    assert g.grade(key, reader({"/app/a": text}))[0] == reward


@pytest.mark.parametrize("text,reward", [(b'{"b": [1, 2], "a": 1}', 1.0), (b'{"a": 1, "b": [2, 1]}', 0.0), (b'{"a": NaN}', 0.0), (b"{", 0.0)])
def test_parsed_json_ignores_key_order_only(text, reward):
    key = {"kind": "parsed", "format": "json", "path": "/app/a", "expected": {"a": 1, "b": [1, 2]}}
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


@pytest.mark.parametrize(
    "fmt,text,expected,reward",
    [
        ("toml", b'[a]\nb = 1\nc = "x"\n', {"a": {"c": "x", "b": 1}}, 1.0),
        ("toml", b"[a\n", {"a": {}}, 0.0),
        ("dotenv", b"# c\nA=1\n\nB = two\n", {"A": "1", "B": "two"}, 1.0),
        ("dotenv", b"A=1\nA=2\n", {"A": "2"}, 0.0),  # duplicate keys fail
        ("dotenv", b"A=1\njunk\n", {"A": "1"}, 0.0),
        ("lines", b" x \n\ny\n", ["x", "y"], 1.0),
        ("lines", b"y\nx\n", ["x", "y"], 0.0),
        ("line-set", b"y\nx\n", ["x", "y"], 1.0),
    ],
)
def test_parsed_formats(fmt, text, expected, reward):
    assert g.grade({"kind": "parsed", "format": fmt, "path": "/app/f", "expected": expected}, reader({"/app/f": text}))[0] == reward


def _sha(b):
    import hashlib

    return hashlib.sha256(b).hexdigest()


def test_tree_counts_correct_entries_against_expected_plus_extras():
    key = {"kind": "tree", "root": "/app/out", "expected": {"a.txt": {"sha256": _sha(b"A")}, "d/b.txt": {"sha256": _sha(b"B"), "mode": "0600"}}}
    good = g.MemoryView({"/app/out/a.txt": (b"A", 0o644), "/app/out/d/b.txt": (b"B", 0o600), "/app/other": (b"x", 0o644)})
    assert g.grade(key, good)[0] == 1.0
    wrong_mode = g.MemoryView({"/app/out/a.txt": (b"A", 0o644), "/app/out/d/b.txt": (b"B", 0o644)})
    assert g.grade(key, wrong_mode)[0] == 0.5
    extra = g.MemoryView({"/app/out/a.txt": (b"A", 0o644), "/app/out/d/b.txt": (b"B", 0o600), "/app/out/c.txt": (b"C", 0o644)})
    assert g.grade(key, extra)[0] == pytest.approx(2 / 3)
    assert g.grade(key, g.MemoryView({}))[0] == 0.0


def test_tree_on_disk_sees_links_and_modes_without_following(tmp_path):
    out = tmp_path / "app" / "out"
    (out / "d").mkdir(parents=True)
    (out / "a.txt").write_bytes(b"A")
    (out / "d" / "b.txt").write_bytes(b"B")
    (out / "d" / "b.txt").chmod(0o600)
    (tmp_path / "secret").write_bytes(b"A")
    (out / "link").symlink_to(tmp_path / "secret")
    tree = g.DiskView(str(tmp_path)).tree("/app/out")
    assert tree["a.txt"] == ("F", _sha(b"A"), 0o644) and tree["d/b.txt"] == ("F", _sha(b"B"), 0o600)
    assert tree["link"] == ("L", str(tmp_path / "secret"))


def test_checks_equal_and_named_exceptions():
    src = b"def f(x):\n    if x < 0:\n        raise KeyError(x)\n    return [x, str(x)]\n"
    key = {"kind": "checks", "path": "/app/m.py", "module": "m", "checks": [
        {"name": "eq", "func": "f", "kind": "equal", "args": [2], "expected": [2, "2"]},
        {"name": "eq_type", "func": "f", "kind": "equal", "args": [1], "expected": [1.0, "1"]},  # 1 != 1.0 as JSON
        {"name": "raises", "func": "f", "kind": "raises", "args": [-1], "exception": "KeyError"},
        {"name": "raises_wrong", "func": "f", "kind": "raises", "args": [-1], "exception": "ValueError"},
    ]}
    reward, log = g.grade(key, reader({"/app/m.py": src}), trusted=True)
    assert reward == 0.5 and "PASS eq" in log and "PASS raises" in log


def test_commands_compare_stdout_exit_and_output_files():
    script = b"import sys\ndata = open(sys.argv[1]).read()\nopen('out.txt', 'w').write(data.upper())\nprint(len(data))\nsys.exit(3 if 'x' in data else 0)\n"
    key = {"kind": "commands", "files": ["/app/tool.py"], "workdir": "/app", "timeout_sec": 10, "checks": [
        {"name": "ok", "argv": ["python3", "tool.py", "in.txt"], "inputs": {"in.txt": "ab"}, "stdout": "2\n", "exit": 0, "outputs": {"out.txt": "AB"}},
        {"name": "exit", "argv": ["python3", "tool.py", "in.txt"], "inputs": {"in.txt": "x"}, "exit": 3},
        {"name": "wrong", "argv": ["python3", "tool.py", "in.txt"], "inputs": {"in.txt": "ab"}, "stdout": "3"},
        {"name": "slow", "argv": ["python3", "-c", "import time; time.sleep(5)"], "exit": 0},
    ]}
    key["timeout_sec"] = 2
    reward, log = g.grade(key, reader({"/app/tool.py": script}), trusted=True)
    assert reward == 0.5, log
    assert any(ln.startswith("FAIL slow") and "timed out" in ln for ln in log)
    assert g.grade(key, reader({}), trusted=True)[0] == 0.0  # no program: every check fails
