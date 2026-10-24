"""Generation: every instance is checked before it can exist (oracle passes, shortcuts and
doing nothing fail), and the result is a pure function of the seed."""

from __future__ import annotations

import pytest
from _task_helpers import fixture_family

from learning_loop.tasks.generate import MAX_DRAWS, GenerationError, file_bytes, generate_spec
from learning_loop.tasks.spec import ExactAnswer, Family, Reject, Solution, TaskSpec


def _family(build, **kw) -> Family:
    return Family(name="t", version=1, cluster="fixture", skills=(), difficulties={"easy": {}}, build=build, **kw)


def _answer(value: str) -> Solution:
    return Solution(shell="true\n", model=lambda f: {"/app/answer.txt": value + "\n"})


def _spec(ctx, *, truth="42", oracle="42", shortcut=None, files=None, path="/app/answer.txt") -> TaskSpec:
    return TaskSpec(
        instruction="x",
        files=files if files is not None else {"data.txt": str(ctx.rng.randint(0, 10**9))},
        grader=ExactAnswer(path, truth),
        oracle=_answer(oracle),
        shortcuts={"s": _answer(shortcut)} if shortcut is not None else {},
    )


def test_same_seed_same_spec_different_seed_different_spec():
    fam = fixture_family("sum_numbers")
    a, b, c = (generate_spec(fam, "hard", s) for s in (5, 5, 6))
    assert file_bytes(a.spec) == file_bytes(b.spec) and a.spec.grader == b.spec.grader
    assert file_bytes(a.spec) != file_bytes(c.spec)


def test_fixture_families_record_their_baselines():
    s = generate_spec(fixture_family("sum_numbers"), "easy", 1)
    assert s.nop_reward == 0.0 and s.shortcut_rewards == {"all-files": 0.0}
    f = generate_spec(fixture_family("fix_add"), "easy", 1)
    assert f.nop_reward == 0.5 and f.shortcut_rewards == {"forge-eq": 0.5}


def test_an_oracle_model_that_fails_is_a_family_bug():
    with pytest.raises(GenerationError, match="oracle model scores 0.0"):
        generate_spec(_family(lambda ctx: _spec(ctx, oracle="41")), "easy", 1)


def test_a_shortcut_that_succeeds_forces_a_new_draw():
    calls = []

    def build(ctx):
        calls.append(1)
        lucky = ctx.rng.random() < 0.5
        return _spec(ctx, shortcut="42" if lucky else "7")

    g1 = generate_spec(_family(build), "easy", 9)
    n = len(calls)
    g2 = generate_spec(_family(build), "easy", 9)
    assert g1.draws == g2.draws == n and file_bytes(g1.spec) == file_bytes(g2.spec)
    assert g1.shortcut_rewards == {"s": 0.0}


def test_a_shortcut_that_always_succeeds_exhausts_the_draws():
    with pytest.raises(GenerationError, match=f"no valid instance in {MAX_DRAWS} draws.*shortcut 's'"):
        generate_spec(_family(lambda ctx: _spec(ctx, shortcut="42")), "easy", 1)


def test_doing_nothing_must_not_succeed():
    def build(ctx):
        return _spec(ctx, files={"answer.txt": "42\n"})

    with pytest.raises(GenerationError, match="doing nothing"):
        generate_spec(_family(build), "easy", 1)


def test_build_can_reject_a_draw():
    def build(ctx):
        if ctx.rng.random() < 0.7:
            raise Reject("tie")
        return _spec(ctx)

    assert generate_spec(_family(build), "easy", 2).draws >= 1


def test_artifacts_must_live_under_the_workdir():
    with pytest.raises(GenerationError, match="outside /app"):
        generate_spec(_family(lambda ctx: _spec(ctx, path="/tmp/answer.txt")), "easy", 1)


def test_unknown_difficulty():
    with pytest.raises(KeyError, match="unknown difficulty"):
        generate_spec(fixture_family("sum_numbers"), "medium", 1)
