"""Core tasks must not depend on inference waiting time (local fixture version).

The same fixed action sequence is run with and without inserted delays before
every model response; observations, fingerprints and reward must be identical.
The Docker version is tests/integration/test_harbor_tasks.py (marker `docker`).
"""

from __future__ import annotations

import pytest
from _env_helpers import count_errors_instance, make_plan, scripted_spec

from learning_loop.backends import LocalFixtureBackend
from learning_loop.events import read_events
from learning_loop.policy import load_script
from learning_loop.records import EventKind


def _trace(events_path):
    evs = read_events(events_path)
    obs = [e.data["observation"] for e in evs if e.kind == EventKind.TOOL_RESULT]
    fps = [e.data["fingerprint"] for e in evs if e.kind == EventKind.FINGERPRINT]
    acts = [e.data["history_message"] for e in evs if e.kind == EventKind.RESPONSE]
    return obs, fps, acts


@pytest.mark.parametrize("family_id", ["count-errors/easy/s1", "count-errors/medium/s201"])
async def test_delays_do_not_change_state_observations_or_reward(tmp_path, family_id):
    inst = count_errors_instance(tmp_path / "tasks", family_id)
    script = load_script(scripted_spec().scripted_path)
    delayed = tmp_path / "delayed.yaml"
    delayed.write_text(
        "schema: scripted_policy/v1\ndelay_sec: 0.4\n" + "rules:\n" + "".join(
            f"  - {r.model_dump_json(exclude_defaults=True)}\n" for r in script.rules
        )
    )
    b = LocalFixtureBackend()
    fast = await b.run(inst, make_plan(inst, "fast"), tmp_path / "fast")
    slow = await b.run(inst, make_plan(inst, "slow", scripted_spec(delayed)), tmp_path / "slow")
    assert slow.summary.timing.endpoint_sec >= 0.4 * slow.summary.n_requests - 0.05
    assert _trace(fast.out_dir / "events.jsonl") == _trace(slow.out_dir / "events.jsonl")
    assert fast.summary.reward == slow.summary.reward
    assert fast.summary.success == slow.summary.success
