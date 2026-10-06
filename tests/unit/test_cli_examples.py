"""The committed experiment/machine examples validate through the CLI exactly as documented."""

from pathlib import Path

import pytest
import yaml

from learning_loop.cli import main
from learning_loop.core.config import REPO_ROOT

M = REPO_ROOT / "configs" / "machines" / "examples"
E = REPO_ROOT / "experiments"

CASES = [
    (E / "fixture-two-cycles.yaml", M / "fixture.yaml", []),
    (E / "fixture-two-cycles.yaml", M / "fixture-docker.yaml", []),
    (E / "smoke-mac.yaml", M / "mac-local.yaml", []),
    (E / "smoke-mac.yaml", M / "mac-local.yaml", ["--set", "learner.model_profile=qwen3-1.7b"]),
    (E / "pilot.yaml", M / "lab-gpu.yaml", []),
    (E / "frozen-baseline.yaml", M / "lab-gpu.yaml", []),
    (E / "frozen-baseline.yaml", M / "external-llama.yaml", []),
    (E / "smoke-mac.yaml", M / "runpod.yaml", ["--set", "learner.model_profile=qwen3-1.7b"]),
    (E / "pilot.yaml", M / "runpod-a100.yaml", []),
    (E / "pilot.yaml", M / "runpod-a100-vllm.yaml", []),
    (E / "pilot-12b.yaml", M / "runpod-a100-vllm.yaml", []),
    (E / "pilot.yaml", M / "runpod-a100.yaml", ["--set", "learner.model_profile=gemma-4-e2b-it"]),
    *[(p, M / "lab-gpu.yaml", []) for p in sorted((E / "ablations").glob("*.yaml"))],
]


@pytest.mark.parametrize("exp,machine,extra", CASES, ids=lambda v: v.name if isinstance(v, Path) else "")
def test_examples_validate(exp, machine, extra, capsys):
    assert main(["validate", str(exp), "--machines", str(machine), *extra]) == 0
    assert "OK: configuration valid" in capsys.readouterr().out


def test_fixed_dataset_control_requires_a_frozen_export(capsys):
    assert main(["validate", str(E / "fixed-dataset-control.yaml"), "--machines", str(M / "lab-gpu.yaml")]) == 2
    assert "fixed_dataset" in capsys.readouterr().err
    fixture_export = REPO_ROOT / "tests" / "fixtures" / "train" / "fixture_prefs"
    assert main(["validate", str(E / "fixed-dataset-control.yaml"), "--machines", str(M / "lab-gpu.yaml"),
                 "--set", f"training.fixed_dataset={fixture_export}"]) == 2  # fixture data refused by default
    assert main(["validate", str(E / "fixed-dataset-control.yaml"), "--machines", str(M / "lab-gpu.yaml"),
                 "--set", f"training.fixed_dataset={fixture_export}", "--set", "labels.allow_fixture_data='true'"]) == 0


def test_managed_llama_cpp_rejected(tmp_path, capsys):
    # the run can launch only hf_server (hf_transformers or vllm); llama.cpp is served externally
    m = yaml.safe_load((M / "mac-local.yaml").read_text())
    m["inference"]["backend"] = "llama_cpp"
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(m))
    assert main(["validate", str(E / "smoke-mac.yaml"), "--machines", str(tmp_path / "m.yaml")]) == 2
    assert "managed inference serves hf_transformers or vllm" in capsys.readouterr().err


def test_training_device_outside_the_profile_rejected(capsys):
    # gemma declares supported_train_devices [cuda]; the laptop profile trains on mps
    assert main(["validate", str(E / "pilot.yaml"), "--machines", str(M / "mac-local.yaml")]) == 2
    assert "supported_train_devices ['cuda']" in capsys.readouterr().err


def test_external_eval_dry_run_requires_pinned_version(tmp_path, capsys):
    args = ["external-eval", "--checkpoint", "base", "--model-profile", "gemma-4-e4b-it",
            "--machines", str(M / "external-llama.yaml"), "--out", str(tmp_path), "--n-tasks", "3"]
    assert main([*args, "--dataset", "terminal-bench"]) == 2
    assert main([*args, "--dataset", "terminal-bench@2.0"]) == 0
    out = capsys.readouterr().out
    assert "dry run" in out and (next(tmp_path.glob("*/protocol.json"))).exists()
