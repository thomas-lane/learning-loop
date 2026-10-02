"""Remote (SSH) model serving: the adapter is addressed by its repository-relative path on the
server host and pushed there only when missing; the base model needs no push."""

from pathlib import Path

from learning_loop.core.config import HostRef, InferenceProfile, load_model_profile
from learning_loop.core.records import CheckpointRef
from learning_loop.serving import lifecycle


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
    monkeypatch.setattr(lifecycle, "Remote", FakeRemote)
    monkeypatch.setattr(lifecycle, "REPO_ROOT", tmp_path)
    FakeRemote.calls, FakeRemote.existing = [], set()
    prof = InferenceProfile(mode="managed", backend="hf_transformers", port=8765, device="cuda",
                            host=HostRef(kind="ssh", ssh_alias="runpod", workdir="/workspace/lfe"))
    return lifecycle.InferenceManager(prof, load_model_profile("qwen3-0.6b"), tmp_path / "logs")


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


def test_base_checkpoints_are_served_through_a_zero_lora_of_the_experiment_shape(tmp_path, monkeypatch):
    import json

    import yaml

    from learning_loop.core.config import REPO_ROOT as ROOT
    from learning_loop.core.config import ExperimentConfig, MachineProfile, load_experiment
    from learning_loop.orchestration.coordinator import serving_record, zero_lora_spec

    exp, _ = load_experiment(ROOT / "experiments" / "pilot.yaml")
    learner = load_model_profile("gemma-4-e4b-it")
    spec = zero_lora_spec(exp, learner)
    assert spec == {"r": 16, "alpha": 32, "target_modules": sorted(learner.lora_target_modules),
                    "exclude_modules": learner.lora_exclude_modules}
    fixture = ExperimentConfig.model_validate({**exp.model_dump(), "training": {**exp.training.model_dump(), "trainer": "fixture"}})
    assert zero_lora_spec(fixture, learner) is None  # no LoRA training, nothing to match

    mgr = _manager(tmp_path, monkeypatch)
    mgr.zero_lora = spec
    argv = mgr._server_argv(_ckpt(tmp_path, None))
    assert json.loads(argv[argv.index("--zero-lora") + 1]) == spec
    adapter = tmp_path / "runs" / "r1" / "checkpoints" / "c000-abc"
    adapter.mkdir(parents=True)
    assert "--zero-lora" not in mgr._server_argv(_ckpt(tmp_path, adapter))  # trained checkpoints serve their own adapter

    machine = MachineProfile.model_validate(yaml.safe_load((ROOT / "configs/machines/examples/runpod-a100.yaml").read_text()))
    rec = serving_record(machine, learner, exp)
    assert rec["base_checkpoints_served_as"] == "zero_lora" and rec["zero_lora"] == spec
