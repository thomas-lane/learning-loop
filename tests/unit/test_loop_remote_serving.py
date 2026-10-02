"""Remote (SSH) model serving: the adapter is addressed by its repository-relative path on the
server host and pushed there only when missing; the base model needs no push."""

from pathlib import Path

from learning_loop import inference
from learning_loop.config import HostRef, InferenceProfile, load_model_profile
from learning_loop.records import CheckpointRef


class FakeRemote:
    calls: list = []
    existing: set = set()

    def __init__(self, host, dry_run=False):
        self.host = host

    def read_text(self, rel):
        return "{}" if rel in FakeRemote.existing else None

    def run(self, argv, timeout=60, check=True):
        FakeRemote.calls.append(("run", argv))

    def push(self, local, rel):
        FakeRemote.calls.append(("push", str(local), rel))


def _manager(tmp_path, monkeypatch):
    monkeypatch.setattr(inference, "Remote", FakeRemote)
    monkeypatch.setattr(inference, "REPO_ROOT", tmp_path)
    FakeRemote.calls, FakeRemote.existing = [], set()
    prof = InferenceProfile(mode="managed", backend="hf_transformers", port=8765, device="cuda",
                            host=HostRef(kind="ssh", ssh_alias="runpod", workdir="/workspace/lfe"))
    return inference.InferenceManager(prof, load_model_profile("qwen3-0.6b"), tmp_path / "logs")


def _ckpt(tmp_path, adapter: Path | None):
    return CheckpointRef(checkpoint_id="c000-abc", model_profile="qwen3-0.6b", base_model="Qwen/Qwen3-0.6B",
                         base_revision="c1899de289a04d12100db370d81485cdf75e47ca", adapter_path=str(adapter) if adapter else None)


def test_remote_adapter_is_pushed_and_addressed_on_the_host(tmp_path, monkeypatch):
    mgr = _manager(tmp_path, monkeypatch)
    adapter = tmp_path / "runs" / "r1" / "checkpoints" / "c000-abc"
    adapter.mkdir(parents=True)
    argv = mgr._server_argv(_ckpt(tmp_path, adapter))
    assert argv[argv.index("--checkpoint-dir") + 1] == "/workspace/lfe/runs/r1/checkpoints/c000-abc"
    assert ("push", str(adapter.resolve()), "runs/r1/checkpoints/") in FakeRemote.calls


def test_remote_adapter_already_on_host_is_not_pushed(tmp_path, monkeypatch):
    mgr = _manager(tmp_path, monkeypatch)
    adapter = tmp_path / "runs" / "r1" / "checkpoints" / "c000-abc"
    adapter.mkdir(parents=True)
    FakeRemote.existing = {"runs/r1/checkpoints/c000-abc/checkpoint.json"}  # trained on that host
    mgr._server_argv(_ckpt(tmp_path, adapter))
    assert not [c for c in FakeRemote.calls if c[0] == "push"]


def test_remote_base_model_needs_no_push(tmp_path, monkeypatch):
    mgr = _manager(tmp_path, monkeypatch)
    argv = mgr._server_argv(_ckpt(tmp_path, None))
    assert "--checkpoint-dir" not in argv and "--base-checkpoint-id" in argv and FakeRemote.calls == []
