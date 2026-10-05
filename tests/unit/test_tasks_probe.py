"""The environment probe: each expectation produces exactly its violation."""

from __future__ import annotations

import os

from learning_loop.tasks.runtime import probe

EXPECT = {"network": "none", "env": {"TZ": "UTC"}, "tools": ["python3"], "hostname": "task", "allowed_processes": ["sh", "sleep"]}


def _observed(**over):
    base = {
        "egress": {"dns": False, "tcp": False},
        "env": {"TZ": "UTC"},
        "tools": {"python3": "/usr/bin/python3"},
        "hostname": "task",
        "processes": [(1, "sh"), (7, "sleep")],
    }
    return base | over


def test_a_matching_environment_has_no_violations():
    assert probe.check(_observed(), EXPECT) == []


def test_each_mismatch_is_reported():
    assert probe.check(_observed(egress={"dns": True, "tcp": False}), EXPECT) == ["egress:dns"]
    assert probe.check(_observed(egress=None), EXPECT) == ["egress:dns", "egress:tcp"]
    assert probe.check(_observed(env={"TZ": None}), EXPECT) == ["env:TZ"]
    assert probe.check(_observed(tools={"python3": None}), EXPECT) == ["tool:python3"]
    assert probe.check(_observed(hostname="abc123"), EXPECT) == ["hostname"]
    assert probe.check(_observed(processes=[(1, "sh"), (9, "nc"), (10, "nc")]), EXPECT) == ["process:nc"]
    assert probe.check(_observed(processes=None), EXPECT) == ["process:no_proc"]


def test_egress_is_only_checked_when_the_profile_has_no_network():
    assert probe.check(_observed(egress=None), {k: v for k, v in EXPECT.items() if k != "network"}) == []


def test_run_observes_the_local_environment(monkeypatch):
    monkeypatch.setenv("LL_PROBE_TEST", "1")
    r = probe.run({"env": {"LL_PROBE_TEST": "1"}, "tools": ["sh"]})
    assert r["ok"] is True and r["observed"]["env"] == {"LL_PROBE_TEST": "1"} and r["observed"]["egress"] is None
    assert probe.run({"env": {"LL_PROBE_TEST": "2"}})["violations"] == ["env:LL_PROBE_TEST"]
    if os.path.isdir("/proc"):
        assert all(pid != os.getpid() for pid, _ in r["observed"]["processes"])


def test_tree_digest_covers_content_and_paths_not_times_or_modes(tmp_path):
    a = tmp_path / "a"
    (a / "data").mkdir(parents=True)
    (a / "data" / "x.txt").write_text("1\n")
    d1 = probe.tree_digest(str(a))
    os.utime(a / "data" / "x.txt", (0, 0))
    os.chmod(a / "data" / "x.txt", 0o600)
    assert probe.tree_digest(str(a)) == d1
    (a / "data" / "x.txt").write_text("2\n")
    assert probe.tree_digest(str(a)) != d1
    (a / "data" / "x.txt").write_text("1\n")
    (a / "data" / "extra").mkdir()
    assert probe.tree_digest(str(a)) != d1


def test_app_content_mismatch_is_a_violation(tmp_path):
    (tmp_path / "f").write_text("x")
    expect = {"app_digest": {"path": str(tmp_path), "sha256": probe.tree_digest(str(tmp_path))}}
    assert probe.run(expect)["violations"] == []
    (tmp_path / "f").write_text("y")
    assert probe.run(expect)["violations"] == ["app_content"]
