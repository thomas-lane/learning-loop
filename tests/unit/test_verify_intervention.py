"""Acceptance fails when the fixed action at turn k errored, timed out or left no record."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures" / "verify"))
import verify_builders as vb  # noqa: E402

from learning_loop.core.records import BranchCost, BranchResult, EpisodeRole, EpisodeSummary, StopCategory, Usage  # noqa: E402
from learning_loop.editing.verify import evaluate_strict_all_success, intervention_tool_reasons  # noqa: E402
from test_verify_branches import outcomes, setup, verifier  # noqa: E402

OK = {"success": True, "requests": [(900, 10)]}


@pytest.mark.parametrize(
    "record, expected",
    [
        ({"executed": True, "error": "TimeoutError: exec exceeded 60s", "exit_code": None, "timed_out": True}, ["intervention_timeout:edited:r0"]),
        ({"executed": True, "error": None, "exit_code": 124, "timed_out": True}, ["intervention_timeout:edited:r0"]),
        ({"executed": True, "error": "docker cp failed", "exit_code": None, "timed_out": False}, ["intervention_tool_error:edited:r0"]),
        ({"executed": False, "error": "invalid arguments", "exit_code": None, "timed_out": False}, ["intervention_tool_error:edited:r0"]),
        (None, ["missing_intervention_record:edited:r0"]),
    ],
)
async def test_intervention_failures_reject_even_if_continuation_succeeds(tmp_path, record, expected):
    src, prop = setup(tmp_path)
    rec = await verifier(tmp_path, vb.FakeBackend(outcomes(edit={**OK, "intervention_tool": record})), src).verify(prop)
    assert not rec.accepted
    assert rec.reasons == expected  # the cheaper, successful continuation does not hide it


async def test_nonzero_exit_code_alone_is_not_a_failure(tmp_path):
    src, prop = setup(tmp_path)
    rec_ok = {"executed": True, "error": None, "exit_code": 1, "timed_out": False}
    rec = await verifier(tmp_path, vb.FakeBackend(outcomes(edit={**OK, "intervention_tool": rec_ok})), src).verify(prop)
    assert rec.accepted, rec.reasons


async def test_original_branch_failure_also_rejects(tmp_path):
    src, prop = setup(tmp_path)
    bad = {"success": True, "requests": [(1000, 20), (1100, 20), (1200, 10)], "intervention_tool": {"executed": True, "error": "boom", "exit_code": None, "timed_out": False}}
    rec = await verifier(tmp_path, vb.FakeBackend(outcomes(orig=bad)), src).verify(prop)
    assert not rec.accepted and rec.reasons == ["intervention_tool_error:original:r0"]


async def test_original_branch_may_repeat_the_source_tool_error(tmp_path):
    # The learner's call was refused by the tool; the edit fixes it. The original branch re-runs
    # the refused call and gets the same error, which is the outcome being compared, not a defect.
    err = "no such file: /app/logs/acess.log"
    src, prop = setup(tmp_path, overrides={0: {"tool_error": err}})
    same = {"executed": False, "error": err, "exit_code": None, "timed_out": False}
    rec = await verifier(tmp_path, vb.FakeBackend(outcomes(orig={**OK, "requests": [(1000, 20), (1100, 20)], "intervention_tool": same})), src).verify(prop)
    assert rec.accepted, rec.reasons
    other = {**same, "error": "docker cp failed", "executed": True}  # a failure the source did not have
    src2, prop2 = setup(tmp_path / "b", overrides={0: {"tool_error": err}})
    rec2 = await verifier(tmp_path / "b", vb.FakeBackend(outcomes(orig={**OK, "requests": [(1000, 20), (1100, 20)], "intervention_tool": other})), src2).verify(prop2)
    assert not rec2.accepted and rec2.reasons == ["intervention_tool_error:original:r0"]
    # the edited action is still held to "executed without error"
    src3, prop3 = setup(tmp_path / "c", overrides={0: {"tool_error": err}})
    rec3 = await verifier(tmp_path / "c", vb.FakeBackend(outcomes(edit={**OK, "intervention_tool": same})), src3).verify(prop3)
    assert not rec3.accepted and rec3.reasons == ["intervention_tool_error:edited:r0"]


def _branch(label, extra, replay_ok=True):
    ep = EpisodeSummary(
        episode_id=f"b-{label}", role=EpisodeRole.BRANCH, instance_id="i", checkpoint_id="base", stop_reason="model_finished",
        stop_category=StopCategory.MODEL, success=True, partial_reward=1.0, usage=Usage(input_tokens=1, output_tokens=1), n_requests=1, n_tool_calls=1, extra=extra,
    )
    cost = BranchCost(shared_prefix_tokens=10, intervention_request_input_tokens=5, intervention_tokens=5, intervention_tokens_source="t", continuation_tokens=100 if label == "original" else 50)
    return BranchResult(branch=label, repetition=0, continuation_seed=1, episode=ep, replay_ok=replay_ok, cost=cost)


def test_missing_record_fails_closed_and_can_be_waived_explicitly():
    good = {"intervention_tool": {"executed": True, "error": None, "exit_code": 0, "timed_out": False}}
    branches = [_branch("original", good), _branch("edited", {})]
    acc = evaluate_strict_all_success(branches, 1, min_token_saving=1)
    assert not acc.accepted and acc.reasons == ["missing_intervention_record:edited:r0"]
    assert evaluate_strict_all_success(branches, 1, min_token_saving=1, require_intervention_record=False).accepted


def test_replay_failure_does_not_add_missing_record_noise():
    assert intervention_tool_reasons(_branch("edited", {}, replay_ok=False)) == []
