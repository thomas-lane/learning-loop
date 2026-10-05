"""Remote training path against a fake SSH host (a local directory): two cycles, so the second
cycle must push the *local* copy of the incoming adapter; pulled checkpoints are verified and
their adapter_path rewritten to the local path; a still-running remote trainer is not relaunched."""

import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from learning_loop.core.config import REPO_ROOT
from learning_loop.core.storage import read_json
from learning_loop.orchestration import coordinator as co

EXP = REPO_ROOT / "experiments" / "fixture-two-cycles.yaml"


class FakeRemote:
    launches: list[list[str]] = []

    def __init__(self, host, dry_run=False):
        self.alias, self.workdir = host.ssh_alias, host.workdir
        self.root = Path(host.workdir)

    def push_repo(self):
        pass

    def push(self, local, rel):
        local, dest = Path(local), self.root / rel
        if local.is_dir():
            shutil.copytree(local, dest / local.name if rel.endswith("/") else dest, dirs_exist_ok=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(local, dest)

    def pull(self, rel, local):
        shutil.copytree(self.root / rel.rstrip("/"), local)

    def start_detached(self, argv, log_rel):
        FakeRemote.launches.append(argv)
        i = argv.index("python")
        cmd = [sys.executable, *argv[i + 1:]]
        (self.root / log_rel).parent.mkdir(parents=True, exist_ok=True)
        p = subprocess.Popen(cmd, cwd=self.root, stdout=open(self.root / log_rel, "w"), stderr=subprocess.STDOUT)
        p.wait()  # finishes quickly (fixture trainer); keeps the test deterministic
        return p.pid

    def pid_alive(self, pid, needle=None):
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def run(self, argv, timeout=60, check=True):
        if argv[0] == "rm":
            for f in argv[2:]:
                (self.root / f).unlink(missing_ok=True)
            return subprocess.CompletedProcess(argv, 0, "", "")
        assert argv[0] == "mkdir"
        for d in argv[2:]:
            (self.root / d).mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(argv, 0, "", "")

    def read_text(self, rel):
        p = self.root / rel
        return p.read_text() if p.exists() else None


def test_two_cycles_with_remote_training_host(tmp_path, monkeypatch):
    remote_root = tmp_path / "gpu-box"
    remote_root.mkdir()
    machine = tmp_path / "m.yaml"
    m = yaml.safe_load((REPO_ROOT / "configs/machines/examples/fixture.yaml").read_text())
    m["training"] = {"host": {"kind": "ssh", "ssh_alias": "gpu-box", "workdir": str(remote_root)}, "device": "cpu", "allow_cpu_fallback": True}
    machine.write_text(yaml.safe_dump(m))
    monkeypatch.setattr(co, "Remote", FakeRemote)
    FakeRemote.launches = []
    ctx = co.create_run(EXP, machine, run_id="rt", runs_dir=tmp_path / "runs")
    asyncio.run(co.run_all(ctx, log=lambda _: None))
    c0, c1 = (read_json(co.cycle_state_path(ctx, c)) for c in (0, 1))
    assert c0["update"] == c1["update"] == "trained"
    for st in (c0, c1):
        path = Path(st["learner_out"]["adapter_path"])
        assert path.is_relative_to(ctx.checkpoints_dir.resolve()) and path.exists()  # local, not the host's
        assert st["checkpoint_record"]["training_config"]["remote_origin"]["host"] == "gpu-box"
    assert c1["checkpoint_record"]["metrics"]["initial_weights_match_incoming"] is True
    assert len(FakeRemote.launches) == 2

    # Resume after a lost connection: the launch record exists and the result is published,
    # so the trainer is not started again.
    (ctx.stage_dir(1, "train") / "checkpoint_path.txt").unlink()
    co.train_checkpoint(ctx, 1, co.CheckpointRef.model_validate(c1["learner_in"]), ctx.stage_dir(1, "dataset"))
    assert len(FakeRemote.launches) == 2


def test_remote_pull_rejects_tampered_adapter(tmp_path, monkeypatch):
    remote_root = tmp_path / "gpu-box"
    remote_root.mkdir()
    machine = tmp_path / "m.yaml"
    m = yaml.safe_load((REPO_ROOT / "configs/machines/examples/fixture.yaml").read_text())
    m["training"] = {"host": {"kind": "ssh", "ssh_alias": "gpu-box", "workdir": str(remote_root)}, "device": "cpu", "allow_cpu_fallback": True}
    machine.write_text(yaml.safe_dump(m))

    class Tamper(FakeRemote):
        def pull(self, rel, local):
            super().pull(rel, local)
            p = Path(local) / "fixture_adapter.json"
            p.chmod(0o644)
            p.write_text(p.read_text().replace("0", "1", 1))

    monkeypatch.setattr(co, "Remote", Tamper)
    ctx = co.create_run(EXP, machine, run_id="rt2", runs_dir=tmp_path / "runs", overrides=["cycles=1"])
    with pytest.raises(RuntimeError, match="sha256"):
        asyncio.run(co.run_all(ctx, log=lambda _: None))


def test_submitted_coordinator_log_is_where_the_dashboard_reads_it(tmp_path, monkeypatch):
    """`loop submit` sends the coordinator's console output to runs/<id>/logs/coordinator.log, so a
    run brought back with `loop fetch` shows its log on the dashboard without --log."""
    from learning_loop.hosts import remote_jobs
    from learning_loop.reporting import dashboard

    host_root = tmp_path / "coordinator-box"
    host_root.mkdir()
    machine = tmp_path / "m.yaml"
    m = yaml.safe_load((REPO_ROOT / "configs/machines/examples/lab-gpu.yaml").read_text())
    m["coordinator"]["workdir"] = str(host_root)
    machine.write_text(yaml.safe_dump(m))

    class Host(FakeRemote):
        def run_state(self, rel):
            return "missing"

        def start_detached(self, argv, log_rel):
            assert argv[:3] == ["uv", "run", "loop"] and (self.root / log_rel).parent.is_dir()
            (self.root / log_rel).write_text("cycle 0: evaluating c000\n")
            return 4242

    monkeypatch.setattr(remote_jobs, "Remote", Host)
    monkeypatch.setattr(remote_jobs, "SUBMISSIONS", tmp_path / "_submissions")
    msg = remote_jobs.submit(str(EXP), str(machine), run_id="sub1")
    assert msg.endswith("log: runs/sub1/logs/coordinator.log")
    assert (host_root / "runs/sub1/logs/coordinator.log").exists() and not (host_root / "runs/sub1/coordinator.log").exists()

    monkeypatch.setattr(remote_jobs, "REPO_ROOT", tmp_path / "laptop")
    remote_jobs.fetch("sub1", str(machine))
    rd = tmp_path / "laptop" / "runs" / "sub1"
    section = dashboard.sec_log({"v": dashboard.RunView(dir=rd, base="/run/sub1/")})
    assert "cycle 0: evaluating c000" in section
