"""Rendering: the task directory is a deterministic function of the spec, and every
environment invariant the renderer owns holds in what it writes."""

from __future__ import annotations

import json
import os
import shlex
import stat
import tomllib
from pathlib import Path

import pytest
import yaml
from _task_helpers import fixture_family
from harbor.models.task.config import TaskConfig

from learning_loop.core.records import RestoreCapability
from learning_loop.core.storage import sha256_tree
from learning_loop.episodes.backends import LocalFixtureBackend
from learning_loop.tasks.instances import load_state_spec
from learning_loop.tasks.render import FIXED_MTIME, RUNTIME_DIR, render
from learning_loop.tasks.spec import PROFILES, Family

CASES = [("sum_numbers", "easy"), ("sum_numbers", "hard"), ("fix_add", "easy"), ("copy_private", "easy"), ("upper_tool", "easy")]


@pytest.fixture(params=CASES, ids=[f"{m}-{d}" for m, d in CASES])
def rendered(request, tmp_path) -> tuple[Family, Path]:
    module, difficulty = request.param
    fam = fixture_family(module)
    render(fam, difficulty, 3, tmp_path / "t")
    return fam, tmp_path / "t"


def _all_paths(root: Path) -> list[Path]:
    return [root, *sorted(root.rglob("*"))]


@pytest.mark.parametrize("module,difficulty", CASES)
def test_render_is_deterministic_in_bytes_modes_and_mtimes(tmp_path, module, difficulty):
    fam = fixture_family(module)
    render(fam, difficulty, 3, tmp_path / "a")
    render(fam, difficulty, 3, tmp_path / "b")
    render(fam, difficulty, 4, tmp_path / "c")
    assert sha256_tree(tmp_path / "a") == sha256_tree(tmp_path / "b")
    assert sha256_tree(tmp_path / "a") != sha256_tree(tmp_path / "c")
    for p in _all_paths(tmp_path / "a"):
        st = p.lstat()
        assert int(st.st_mtime) == FIXED_MTIME, p
        twin = (tmp_path / "b" / p.relative_to(tmp_path / "a")).lstat()
        assert stat.S_IMODE(st.st_mode) == stat.S_IMODE(twin.st_mode) and st.st_mtime_ns == twin.st_mtime_ns, p


def test_images_are_pinned_with_fixed_env_and_no_run(rendered):
    fam, d = rendered
    profile = PROFILES[fam.profile]
    for dockerfile in (d / "environment" / "Dockerfile", d / "tests" / "Dockerfile"):
        lines = [ln for ln in dockerfile.read_text().splitlines() if ln and not ln.startswith("#")]
        instructions = [ln.split()[0] for ln in lines]
        assert set(instructions) <= {"FROM", "ENV", "WORKDIR", "COPY"}, instructions
        assert [ln for ln in lines if ln.startswith("FROM")] == [f"FROM {profile.base_image}"]
        assert "@sha256:" in profile.base_image
        env = dict(kv.split("=", 1) for kv in shlex.split(next(ln for ln in lines if ln.startswith("ENV"))[4:]))
        assert env == dict(profile.env)


def test_both_containers_have_no_network_and_fixed_hostname(rendered):
    fam, d = rendered
    for compose in (d / "environment" / "docker-compose.yaml", d / "tests" / "docker-compose.yaml"):
        main = yaml.safe_load(compose.read_text())["services"]["main"]
        assert main == {"network_mode": "none", "hostname": PROFILES[fam.profile].hostname}


def test_answer_key_and_graders_stay_out_of_the_agent_context(rendered):
    fam, d = rendered
    env_files = {p.name for p in (d / "environment").rglob("*")}
    assert not env_files & {"key.json", "grade.py", "probe.py", "solve.sh", "test.sh"}
    assert (d / "tests" / "key.json").is_file()
    for name in ("grade.py", "probe.py"):
        assert (d / "tests" / name).read_bytes() == (RUNTIME_DIR / name).read_bytes()


def test_verifier_runs_the_shared_grader_behind_the_profile_probe(rendered):
    fam, d = rendered
    argv = shlex.split((d / "tests" / "test.sh").read_text().splitlines()[-1])
    assert argv[:4] == ["exec", "python3", "-I", "-B"] and argv[4] == "/tests/grade.py"
    assert json.loads(argv[argv.index("--probe") + 1]) == PROFILES[fam.profile].probe_expectation()
    assert (d / "tests" / "test.sh").stat().st_mode & 0o111 and (d / "solution" / "solve.sh").stat().st_mode & 0o111


def test_task_toml_is_valid_harbor_config_with_the_replay_contract(rendered):
    fam, d = rendered
    cfg = TaskConfig.model_validate_toml((d / "task.toml").read_text())
    assert cfg.verifier.environment_mode.value == "separate"
    key = json.loads((d / "tests" / "key.json").read_text())
    expected_artifacts = {"tree": lambda k: [k["root"]], "commands": lambda k: k["files"]}.get(key["kind"], lambda k: [k["path"]])(key)
    assert [str(a) if isinstance(a, str) else a for a in cfg.artifacts] == expected_artifacts
    meta = tomllib.loads((d / "task.toml").read_text())["metadata"]
    assert meta["generator"] == fam.generator_id and meta["generator_seed"] == 3 and meta["network"] == "none"
    params = json.loads(meta["params_json"])
    assert {"oracle_reward", "nop_reward", "draws", "shortcut_rewards"} <= set(params)
    spec = load_state_spec(d)
    assert spec.restore == RestoreCapability.DETERMINISTIC_REPLAY and spec.fingerprint_paths == ["/app"]
    assert ("local_fixture" in meta) == fam.local_fixture


def test_shortcuts_are_rendered_next_to_the_oracle(tmp_path):
    fam = fixture_family("sum_numbers")
    render(fam, "easy", 1, tmp_path / "t")
    assert sorted(p.name for p in (tmp_path / "t" / "solution" / "shortcuts").iterdir()) == ["all-files.sh"]


def test_render_refuses_a_non_empty_directory(tmp_path):
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / "stale").write_text("x")
    with pytest.raises(FileExistsError):
        render(fixture_family("sum_numbers"), "easy", 1, tmp_path / "t")


def test_local_fixture_families_cannot_use_the_checks_grader(tmp_path):
    fam = fixture_family("fix_add")
    bad = Family(**{**fam.__dict__, "local_fixture": True})
    with pytest.raises(ValueError, match="local fixture"):
        render(bad, "easy", 1, tmp_path / "t")


class _Session:
    def __init__(self, root: Path):
        self.root = root

    def real_path(self, virtual: str) -> Path:
        return self.root / virtual.lstrip("/")


def test_local_fixture_backend_grades_with_the_rendered_shared_grader(tmp_path):
    fam = fixture_family("sum_numbers")
    d = tmp_path / "t"
    render(fam, "easy", 1, d)
    expected = json.loads((d / "tests" / "key.json").read_text())["expected"]
    work = tmp_path / "work"
    (work / "app").mkdir(parents=True)
    backend = LocalFixtureBackend()
    meta = backend.local_meta(d)
    (work / "app" / "answer.txt").write_text(expected + "\n")
    reward, info = backend._grade(d, meta, _Session(work))
    assert reward == {"reward": 1.0}, info
    (work / "app" / "answer.txt").write_text("0\n")
    assert backend._grade(d, meta, _Session(work))[0] == {"reward": 0.0}
    os.unlink(work / "app" / "answer.txt")
    assert backend._grade(d, meta, _Session(work))[0] == {"reward": 0.0}


def test_verifier_env_carries_the_digest_of_the_grader_files(rendered):
    from learning_loop.tasks.runtime.grade import TESTS_DIGEST_ENV, tests_digest

    fam, d = rendered
    cfg = TaskConfig.model_validate_toml((d / "task.toml").read_text())
    assert cfg.verifier.env == {TESTS_DIGEST_ENV: tests_digest(str(d / "tests"))}


def test_same_size_files_with_different_content_get_different_mtimes(tmp_path):
    # BuildKit skips re-sending a context file whose path, size and mtime it has seen before
    fam = fixture_family("sum_numbers")
    render(fam, "easy", 1, tmp_path / "a")
    render(fam, "easy", 2, tmp_path / "b")
    ka, kb = (tmp_path / x / "tests" / "key.json" for x in "ab")
    if ka.read_bytes() != kb.read_bytes():
        assert ka.stat().st_mtime_ns != kb.stat().st_mtime_ns
    assert int(ka.stat().st_mtime) == FIXED_MTIME


def test_replay_contract_carries_the_agent_probe_with_the_app_digest(rendered):
    from learning_loop.tasks.runtime.probe import tree_digest

    fam, d = rendered
    probe = load_state_spec(d).env_probe
    assert probe == PROFILES[fam.profile].probe_expectation() | {"app_digest": {"path": "/app", "sha256": tree_digest(str(d / "environment" / "files"))}}


def test_symlinks_are_rendered_as_links_and_counted_in_the_app_digest(tmp_path):
    from learning_loop.tasks.runtime.probe import tree_digest
    from learning_loop.tasks.spec import ExactAnswer, Solution, TaskSpec

    def build(ctx, links):
        return TaskSpec(instruction="x", files={"data/a.txt": "1\n"}, symlinks=links, grader=ExactAnswer("/app/answer.txt", "1"),
                        oracle=Solution("echo 1 > /app/answer.txt\n", lambda f: {"/app/answer.txt": "1\n"}))

    digests = []
    for i, links in enumerate(({"data/cur": "a.txt", "data/old": "../missing.txt"}, {"data/cur": "a.txt", "data/old": "../other.txt"})):
        fam = Family(name="links", version=1, cluster="fixture", skills=(), difficulties={"easy": {}}, build=lambda ctx, links=links: build(ctx, links))
        d = tmp_path / f"t{i}"
        render(fam, "easy", 1, d)
        files = d / "environment" / "files" / "data"
        assert (files / "old").is_symlink() and not (files / "old").exists() and (files / "cur").read_text() == "1\n"
        digests.append(tree_digest(str(d / "environment" / "files")))
    assert digests[0] != digests[1]  # the link target is learner-visible state
