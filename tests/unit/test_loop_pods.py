"""Runpod pod lifecycle against a fake Runpod REST API (real HTTP) and a fake SSH layer, plus the
pod-side watchdog script run for real with a short idle window."""

import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import yaml

from learning_loop import pods, remote
from learning_loop.config import REPO_ROOT, MachineProfile

KEY = "rp_test_secret_key"


class FakeRunpod:
    """GET /pods/{id}, POST /pods/{id}/start|stop with Bearer auth; a started pod gets an address
    after `boot_polls` GETs."""

    def __init__(self, status="EXITED", boot_polls=2, busy_starts=0):
        self.status, self.boot_polls, self.calls, self.gets_since_start = status, boot_polls, [], 0
        self.busy_starts = busy_starts  # start calls refused as "not enough free GPUs" first
        self.busy_creates, self.create_cost = 0, 1.64  # create calls refused for capacity first; price
        self.created: dict[str, dict] = {}  # id -> {"name", "status", "gets", "body"}
        self.others = [{"id": "someoneelse", "name": "my-own-pod", "desiredStatus": "RUNNING"}]
        fake = self

        class H(BaseHTTPRequestHandler):
            def _reply(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(data)

            def _auth(self):
                if self.headers.get("User-Agent", "").startswith("Python-urllib"):
                    self._reply(403, {"title": "Error 1010: Access denied"})  # Cloudflare's browser check
                    return False
                if self.headers.get("Authorization") != f"Bearer {KEY}":
                    self._reply(401, {"error": "unauthorized"})
                    return False
                return True

            def _body(self):
                n = int(self.headers.get("Content-Length") or 0)
                return json.loads(self.rfile.read(n) or b"{}")

            def do_GET(self):  # noqa: N802
                fake.calls.append(("GET", self.path))
                if not self._auth():
                    return
                if self.path == "/v1/pods":
                    pods = [{"id": i, "name": p["name"], "desiredStatus": p["status"]} for i, p in fake.created.items()]
                    self._reply(200, pods + fake.others)
                    return
                pid = self.path.rsplit("/", 1)[-1]
                if pid in fake.created:
                    p = fake.created[pid]
                    p["gets"] += 1
                    up = p["status"] == "RUNNING" and p["gets"] > fake.boot_polls
                    self._reply(200, {"id": pid, "name": p["name"], "desiredStatus": p["status"],
                                      "publicIp": "10.9.8.9" if up else "", "portMappings": {"22": 40200} if up else {}})
                    return
                fake.gets_since_start += 1
                booted = fake.status == "RUNNING" and fake.gets_since_start > fake.boot_polls
                self._reply(200, {"id": "pod1", "desiredStatus": fake.status,
                                  "publicIp": "10.9.8.7" if booted else "", "portMappings": {"22": 40100} if booted else {}})

            def do_DELETE(self):  # noqa: N802
                fake.calls.append(("DELETE", self.path))
                if not self._auth():
                    return
                pid = self.path.rsplit("/", 1)[-1]
                if pid not in fake.created:
                    self._reply(404, {"error": "pod not found"})
                    return
                fake.created[pid]["status"] = "TERMINATED"
                self.send_response(204)
                self.end_headers()

            def do_POST(self):  # noqa: N802
                fake.calls.append(("POST", self.path))
                if not self._auth():
                    return
                if self.path == "/v1/pods":
                    body = self._body()
                    fake.calls.append(("BODY", json.dumps(body)))
                    if fake.busy_creates:
                        fake.busy_creates -= 1
                        self._reply(500, {"error": "create pod: There are no longer any instances available with the requested specifications."})
                        return
                    pid = f"new{len(fake.created) + 1}"
                    fake.created[pid] = {"name": body["name"], "status": "RUNNING", "gets": 0, "body": body}
                    self._reply(201, {"id": pid, "name": body["name"], "desiredStatus": "RUNNING", "costPerHr": fake.create_cost,
                                      "gpu": {"displayName": body["gpuTypeIds"][0]}})
                    return
                if self.path.endswith("/start") and fake.busy_starts:
                    fake.busy_starts -= 1
                    self._reply(500, {"error": "start pod: There are not enough free GPUs on the host machine to start this pod.", "status": 500})
                    return
                if self.path.endswith("/start"):
                    fake.status, fake.gets_since_start = "RUNNING", 0
                elif self.path.endswith("/stop"):
                    fake.status = "EXITED"
                self._reply(200, {})

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class FakeRemote:
    """Records what would run over SSH; reachable once the endpoint is registered."""

    log: list = []

    def __init__(self, host, dry_run=False):
        self.host = host
        self.workdir = host.workdir
        self.alias = f"runpod:{host.pod_ref}"
        self.ep = pods.endpoint(host.pod_ref)  # raises if the lifecycle has not resolved the pod

    def _ssh(self, command, timeout=60, check=True, stdin=None):
        FakeRemote.log.append(("ssh", self.workdir, command, stdin is not None))
        return subprocess.CompletedProcess([], 0, stdout="WARNING: no volume is mounted at /workspace\n", stderr="")

    def run(self, argv, timeout=60, check=True):
        FakeRemote.log.append(("run", self.workdir, argv))
        return subprocess.CompletedProcess(argv, 0, "", "")

    def push_repo(self):
        FakeRemote.log.append(("push_repo", self.workdir))

    def start_detached(self, argv, log_rel):
        FakeRemote.log.append(("detached", self.workdir, argv))
        return 4242


def machine(**runpod_kw) -> MachineProfile:
    m = yaml.safe_load((REPO_ROOT / "configs/machines/examples/runpod.yaml").read_text())
    m["runpod"].update(runpod_kw)
    return MachineProfile.model_validate(m)


def created_machine(tmp_path, **spec_kw) -> MachineProfile:
    m = yaml.safe_load((REPO_ROOT / "configs/machines/examples/runpod-a100.yaml").read_text())
    key = tmp_path / "id_test"
    key.write_text("PRIVATE")
    (tmp_path / "id_test.pub").write_text("ssh-ed25519 AAAATEST user@laptop")
    m["runpod"]["identity_file"] = str(key)
    m["runpod"]["create"]["a100"].update(spec_kw)
    return MachineProfile.model_validate(m)


@pytest.fixture
def api(monkeypatch, tmp_path):
    fake = FakeRunpod()
    monkeypatch.setattr(remote, "Remote", FakeRemote)
    monkeypatch.setattr(pods, "GPU_RETRY_MIN_SEC", 0.01)
    monkeypatch.setattr(pods, "KNOWN_HOSTS_DIR", tmp_path / "kh")
    FakeRemote.log = []
    yield fake
    fake.close()


def lifecycle(api, tmp_path, **kw):
    m = machine(**kw)
    client = pods.RunpodClient(KEY, api.url)
    return pods.PodLifecycle(m, tmp_path / "logs", log=lambda _: None, client=client, poll_sec=0.01)


def test_starts_prepares_watches_and_stops(api, tmp_path):
    with lifecycle(api, tmp_path, heartbeat_sec=10) as lc:
        ep = pods.endpoint("examplepodid")
        assert (ep.ip, ep.port) == ("10.9.8.7", 40100) and lc.started_by_us == {"examplepodid"}
        assert "-p" in ep.ssh_options() and "40100" in ep.ssh_options()
    calls = [c for c in api.calls if c[0] == "POST"]
    assert calls == [("POST", "/v1/pods/examplepodid/start"), ("POST", "/v1/pods/examplepodid/stop")]
    assert api.status == "EXITED"
    kinds = [e[0] for e in FakeRemote.log]
    assert kinds.count("push_repo") == 1
    setup = [e for e in FakeRemote.log if e[0] == "ssh" and e[3]]  # setup script fed on stdin
    assert setup and setup[0][1] == "/" and "WORKDIR=/workspace/learn-from-experience bash -s" in setup[0][2]
    assert ("run", "/workspace/learn-from-experience", ["uv", "sync", "--frozen", "--extra", "train"]) in FakeRemote.log
    watchdog = [e for e in FakeRemote.log if e[0] == "detached"][0][2]
    assert watchdog[:2] == ["python3", "scripts/pod_watchdog.py"] and watchdog[3] == str(30 * 60)
    events = [json.loads(line)["event"] for line in (tmp_path / "logs" / "pod-lifecycle.jsonl").read_text().splitlines()]
    assert events == ["start_requested", "ready", "prepared", "watchdog_started", "stop_requested"]
    with pytest.raises(pods.RunpodError):
        pods.endpoint("examplepodid")  # address forgotten after the command


def test_pod_is_stopped_when_the_command_fails(api, tmp_path):
    with pytest.raises(RuntimeError, match="boom"):
        with lifecycle(api, tmp_path):
            raise RuntimeError("boom")
    assert api.status == "EXITED"


def test_running_pod_is_used_and_stop_when_done_false_leaves_it_running(api, tmp_path):
    api.status, api.boot_polls = "RUNNING", 0
    with lifecycle(api, tmp_path, stop_when_done=False) as lc:
        assert lc.started_by_us == set()
    assert api.status == "RUNNING" and not [c for c in api.calls if c[0] == "POST"]


def test_account_key_never_reaches_the_pod(api, tmp_path):
    with lifecycle(api, tmp_path):
        pass
    assert KEY not in json.dumps(FakeRemote.log)
    assert KEY not in (tmp_path / "logs" / "pod-lifecycle.jsonl").read_text()


def test_missing_api_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with pytest.raises(pods.RunpodError, match="RUNPOD_API_KEY is not set"):
        pods.client_for(machine().runpod)


def test_bad_key_surfaces_http_error_without_the_key(api):
    with pytest.raises(pods.RunpodError, match="HTTP 401") as e:
        pods.RunpodClient("wrong-key", api.url).get("examplepodid")
    assert "wrong-key" not in str(e.value)


def test_start_waits_for_a_free_gpu_then_gives_up(api, tmp_path, monkeypatch):
    monkeypatch.setattr(pods, "GPU_RETRY_MIN_SEC", 0.01)
    api.busy_starts = 2
    with lifecycle(api, tmp_path):
        assert api.status == "RUNNING"
    starts = [c for c in api.calls if c == ("POST", "/v1/pods/examplepodid/start")]
    assert len(starts) == 3 and api.status == "EXITED"
    events = [json.loads(line)["event"] for line in (tmp_path / "logs" / "pod-lifecycle.jsonl").read_text().splitlines()]
    assert events[:2] == ["waiting_for_gpu", "start_requested"]

    api.busy_starts = 10**6
    lc = lifecycle(api, tmp_path / "again")
    lc.cfg = lc.cfg.model_copy(update={"start_timeout_sec": 0})
    with pytest.raises(pods.RunpodError, match="no free GPU"):
        with lc:
            pass
    assert api.status == "EXITED"


def test_terminated_pod_is_refused(api, tmp_path):
    api.status = "TERMINATED"
    with pytest.raises(pods.RunpodError, match="terminated"):
        with lifecycle(api, tmp_path):
            pass


def test_runpod_profile_validation():
    with pytest.raises(ValueError, match="runpod"):
        m = yaml.safe_load((REPO_ROOT / "configs/machines/examples/runpod.yaml").read_text())
        del m["runpod"]
        MachineProfile.model_validate(m)
    with pytest.raises(ValueError, match="coordinator"):
        m = yaml.safe_load((REPO_ROOT / "configs/machines/examples/runpod.yaml").read_text())
        m["coordinator"] = {"kind": "runpod", "pod_id": "abc", "workdir": "/w"}
        MachineProfile.model_validate(m)


def test_watchdog_stops_after_stale_heartbeat(tmp_path):
    hb, logf, marker = tmp_path / "hb", tmp_path / "wd.jsonl", tmp_path / "stopped"
    hb.touch()
    env = {**os.environ, "WATCHDOG_STOP_CMD": f"touch {marker}"}
    p = subprocess.Popen([sys.executable, str(REPO_ROOT / "scripts" / "pod_watchdog.py"), str(hb), "1", "0.2", str(logf)], env=env)
    try:
        # keep the heartbeat fresh for a while: no stop
        for _ in range(8):
            hb.touch()
            time.sleep(0.2)
        assert not marker.exists()
        p.wait(timeout=10)  # then let it go stale
    finally:
        if p.poll() is None:
            p.kill()
    assert marker.exists() and p.returncode == 0
    assert "stop requested via override" in logf.read_text()


def created_lifecycle(api, tmp_path, m):
    return pods.PodLifecycle(m, tmp_path / "logs", log=lambda _: None, client=pods.RunpodClient(KEY, api.url), poll_sec=0.01)


def ledger(tmp_path):
    return [json.loads(line) for line in (tmp_path / "kh" / "created.jsonl").read_text().splitlines()]


def test_created_pod_is_created_prepared_watched_and_terminated(api, tmp_path):
    m = created_machine(tmp_path)
    with created_lifecycle(api, tmp_path, m) as lc:
        (pid,) = lc.created.values()
        ep = pods.endpoint("new-a100")
        assert (ep.ip, ep.port) == ("10.9.8.9", 40200) and api.created[pid]["status"] == "RUNNING"
    assert api.created[pid]["status"] == "TERMINATED"
    body = api.created[pid]["body"]
    assert body["name"].startswith("lfe-a100-") and body["gpuTypeIds"] == ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"]
    assert body["gpuTypePriority"] == "custom" and body["cloudType"] == "SECURE" and body["volumeInGb"] == 0
    assert body["allowedCudaVersions"] == ["13.0"] and body["ports"] == ["22/tcp"]
    assert body["env"] == {"PUBLIC_KEY": "ssh-ed25519 AAAATEST user@laptop"}  # only the public key reaches the pod
    assert KEY not in json.dumps(body) and KEY not in json.dumps(FakeRemote.log)
    watchdog = [e for e in FakeRemote.log if e[0] == "detached"][0][2]
    assert watchdog[-1] == "terminate"
    assert [e["event"] for e in ledger(tmp_path)] == ["created", "terminated"]
    assert pods.ledger_open_pods() == {}
    events = [json.loads(line)["event"] for line in (tmp_path / "logs" / "pod-lifecycle.jsonl").read_text().splitlines()]
    assert events == ["created", "ready", "prepared", "watchdog_started", "terminate_requested"]


def test_created_pod_is_terminated_when_the_command_fails(api, tmp_path):
    with pytest.raises(RuntimeError, match="boom"):
        with created_lifecycle(api, tmp_path, created_machine(tmp_path)):
            raise RuntimeError("boom")
    assert [p["status"] for p in api.created.values()] == ["TERMINATED"]


def test_creation_waits_for_capacity_then_gives_up_without_leaving_pods(api, tmp_path):
    api.busy_creates = 2
    with created_lifecycle(api, tmp_path, created_machine(tmp_path)):
        pass
    assert len([c for c in api.calls if c == ("POST", "/v1/pods")]) == 3
    api.busy_creates = 10**6
    lc = created_lifecycle(api, tmp_path / "again", created_machine(tmp_path))
    lc.cfg = lc.cfg.model_copy(update={"start_timeout_sec": 0})
    with pytest.raises(pods.RunpodError, match="became available"):
        with lc:
            pass
    assert all(p["status"] == "TERMINATED" for p in api.created.values())


def test_pod_above_the_price_limit_is_terminated_at_once(api, tmp_path):
    api.create_cost = 3.99
    with pytest.raises(pods.RunpodError, match="max_cost_per_hr"):
        with created_lifecycle(api, tmp_path, created_machine(tmp_path)):
            pass
    assert [p["status"] for p in api.created.values()] == ["TERMINATED"]
    assert pods.ledger_open_pods() == {}


def test_cleanup_terminates_only_ledger_pods_with_the_prefix(api, tmp_path, monkeypatch):
    m = created_machine(tmp_path)
    api.created = {"leak1": {"name": "lfe-a100-x", "status": "RUNNING", "gets": 0, "body": {}},
                   "renamed": {"name": "my-pod-now", "status": "RUNNING", "gets": 0, "body": {}}}
    pods.ledger_append("created", "leak1")
    pods.ledger_append("created", "renamed")
    pods.ledger_append("created", "gone")
    monkeypatch.setattr(pods, "client_for", lambda cfg: pods.RunpodClient(KEY, api.url))
    dry = pods.cleanup(m, dry_run=True)
    assert any("leak1" in d and "would terminate" in d for d in dry) and api.created["leak1"]["status"] == "RUNNING"
    out = pods.cleanup(m)
    assert api.created["leak1"]["status"] == "TERMINATED" and api.created["renamed"]["status"] == "RUNNING"
    assert api.others[0]["desiredStatus"] == "RUNNING"  # never in the ledger: untouched
    assert any("left alone" in o for o in out) and set(pods.ledger_open_pods()) == {"renamed"}


def test_watchdog_terminate_action(tmp_path):
    hb, logf, marker = tmp_path / "hb", tmp_path / "wd.jsonl", tmp_path / "action"
    hb.touch()
    env = {**os.environ, "WATCHDOG_STOP_CMD": f"echo $WATCHDOG_ACTION > {marker}"}
    p = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "pod_watchdog.py"), str(hb), "0.2", "0.1", str(logf), "terminate"],
                       env=env, timeout=10)
    assert p.returncode == 0 and marker.read_text().strip() == "terminate"
    assert "terminate requested via override terminate" in logf.read_text()


def test_created_pod_profile_validation():
    m = yaml.safe_load((REPO_ROOT / "configs/machines/examples/runpod-a100.yaml").read_text())
    m["training"]["host"]["pod"] = "missing"
    with pytest.raises(ValueError, match="not defined in runpod.create"):
        MachineProfile.model_validate(m)
    with pytest.raises(ValueError, match="exactly one of pod_id"):
        MachineProfile.model_validate({**m, "training": {"host": {"kind": "runpod", "pod_id": "abc", "pod": "a100", "workdir": "/w"}}})
