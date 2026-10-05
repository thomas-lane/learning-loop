"""Preference construction boundaries, immutable export, fixed datasets, history selection."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures" / "verify"))
import verify_builders as vb  # noqa: E402

from learning_loop.core.config import DataSelectionConfig  # noqa: E402
from learning_loop.core.records import (  # noqa: E402
    EditProposal,
    PreferenceExample,
    PreferenceProvenance,
    ProposedCall,
    Split,
    VerificationRecord,
)
from learning_loop.editing.editor import SourceContext  # noqa: E402
from learning_loop.editing.preferences import (  # noqa: E402
    PreferenceError,
    PreferencePair,
    bound_history,
    build_pair,
    build_preference,
    check_prompt,
    export_dataset,
    freeze_fixed_dataset,
    load_dataset,
    select_pairs,
    select_training_pairs,
)

REPO = Path(__file__).resolve().parents[2]
EDIT_CMD = vb.PIPELINE + " | awk '{print $2}' > /app/answer.txt"
JUSTIFICATION = "The learner could have computed the answer in one pipeline right away."


def accepted_ver(pid="prop-1", **kw) -> VerificationRecord:
    base = dict(verification_id="ver-1", proposal_id=pid, mode="continuation", acceptance_rule="strict_all_success_v1", accepted=True, reasons=["accepted"], mean_saving=1234.0, evidence_label="one_observed_successful_preference")
    base.update(kw)
    return VerificationRecord(**base)


def fixture(tmp_path, turn_index=2):
    summary, path = vb.write_source_episode(tmp_path / "src")
    inst = vb.make_instance(tmp_path)
    src = SourceContext.load(summary, inst, path)
    prop = EditProposal(proposal_id="prop-1", source_episode_id="src-ep", instance_id=inst.instance_id, editor_id="ed-1", status="proposed", turn_index=turn_index, tool_call_id=f"call_{turn_index}", replacement=ProposedCall(name="bash", arguments={"command": EDIT_CMD}), justification=JUSTIFICATION)
    return src, inst, prop


def build(src, inst, prop, ver=None, split=Split.TRAIN):
    return build_preference(events=src.events, turns=src.turns, proposal=prop, verification=ver or accepted_ver(), instance=inst, split=split, tools=src.plan.tools, learner_checkpoint_id="base", cycle=0, run_id="run-1")


def test_pair_prompt_is_exact_prefix_without_future_or_editor_text(tmp_path):
    src, inst, prop = fixture(tmp_path, turn_index=2)
    pair = build(src, inst, prop)
    ex = pair.example
    req2 = next(e for e in src.events if e.kind.value == "request" and e.turn_index == 2)
    assert ex.prompt == req2.data["messages"]
    assert [m["role"] for m in ex.prompt] == ["system", "user", "assistant", "tool", "assistant", "tool"]
    blob = json.dumps(ex.prompt)
    assert "42 10.0.0.7" not in blob and vb.FINAL_ANSWER not in blob  # later observations / final answer
    assert JUSTIFICATION not in json.dumps(ex.model_dump())  # editor text is provenance-only... and not even there
    assert "strict_all_success" not in blob and "accepted" not in blob  # no verifier text
    # chosen / rejected
    assert ex.rejected == [src.turns[2].assistant_message]
    (chosen,) = ex.chosen
    call = chosen["tool_calls"][0]
    assert call["id"] == "call_2" and chosen["content"] == "" and call["function"]["name"] == "bash"
    assert isinstance(call["function"]["arguments"], str) and json.loads(call["function"]["arguments"]) == {"command": EDIT_CMD}
    assert ex.chosen != ex.rejected and ex.tools == vb.TOOLS
    pv = pair.provenance
    assert pv.verification_mode == "continuation" and pv.split == Split.TRAIN and pv.editor_id == "ed-1" and pv.mean_saving == 1234.0
    assert set(PreferenceExample.model_fields) == {"schema_version", "pair_id", "prompt", "chosen", "rejected", "tools"}


def test_turn_zero_prompt_is_system_and_user_only(tmp_path):
    src, inst, prop = fixture(tmp_path, turn_index=0)
    ex = build(src, inst, prop).example
    assert [m["role"] for m in ex.prompt] == ["system", "user"]


def test_refuses_unaccepted_and_audit_verifications(tmp_path):
    src, inst, prop = fixture(tmp_path)
    with pytest.raises(PreferenceError):
        build(src, inst, prop, ver=accepted_ver(accepted=False, reasons=["tie"]))
    with pytest.raises(PreferenceError):
        build(src, inst, prop, ver=accepted_ver(purpose="audit"))
    with pytest.raises(PreferenceError):
        build(src, inst, prop.model_copy(update={"status": "invalid"}))


def test_refuses_invalid_candidates(tmp_path):
    src, inst, prop = fixture(tmp_path)
    bad = prop.model_copy(update={"replacement": ProposedCall(name="bash", arguments={"cmd": "x"})})
    with pytest.raises(PreferenceError, match="chosen:schema_violation"):
        build(src, inst, bad)
    same = prop.model_copy(update={"replacement": ProposedCall(name="bash", arguments=json.loads(src.turns[2].assistant_message["tool_calls"][0]["function"]["arguments"]))})
    with pytest.raises(PreferenceError, match="chosen_equals_rejected"):
        build(src, inst, same)
    summary, path = vb.write_source_episode(tmp_path / "rep", overrides={2: {"repaired": True}})
    src2 = SourceContext.load(summary, inst, path)
    with pytest.raises(PreferenceError, match="malformed/repaired"):
        build(src2, inst, prop)


def test_check_prompt_detects_leaks():
    prompt = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    assert check_prompt(prompt, 0) == []
    later_tool = prompt + [{"role": "tool", "tool_call_id": "call_5", "content": "x"}]
    assert any("contains_later_tool_calls" in e for e in check_prompt(later_tool, 0, later_call_ids=["call_5"]))
    leaked = [{"role": "system", "content": "s"}, {"role": "user", "content": "u " + JUSTIFICATION}]
    assert "prompt:contains_forbidden_text" in check_prompt(leaked, 0, [JUSTIFICATION])
    assert any("assistant_turns" in e for e in check_prompt(prompt, 1))
    assert any("unexpected_roles" in e for e in check_prompt(prompt + [{"role": "verifier", "content": "ok"}], 0))


def test_build_pair_from_item_dir(tmp_path):
    summary, _ = vb.write_source_episode(tmp_path / "item")
    (tmp_path / "item" / "summary.json").write_text(summary.model_dump_json())
    _, inst, prop = fixture(tmp_path / "x")
    pair = build_pair(proposal=prop, verification=accepted_ver(), source_dir=tmp_path / "item", instance=inst, split=Split.TRAIN, learner_checkpoint_id="base", cycle=3, run_id="r")
    assert pair.provenance.cycle == 3 and pair.example.tools == vb.TOOLS


# --------------------------------------------------------------------------- #
# Synthetic pairs for export / selection tests
# --------------------------------------------------------------------------- #


def mk(i: int, instance="inst-a", family="fam", split=Split.TRAIN, mode="continuation", kind="verified", cycle=0) -> PreferencePair:
    pid = f"pair-{i:03d}"
    msg = lambda c: {"role": "assistant", "content": "", "tool_calls": [{"id": "c0", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": c})}}]}  # noqa: E731
    ex = PreferenceExample(pair_id=pid, prompt=[{"role": "system", "content": "s"}, {"role": "user", "content": f"task {i}"}], chosen=[msg("a")], rejected=[msg(f"b{i}")], tools=vb.TOOLS)
    pv = PreferenceProvenance(pair_id=pid, verification_mode=mode, instance_id=instance, family=family, difficulty="easy", split=split, source_episode_id="s", proposal_id="p", verification_id="v", learner_checkpoint_id="base", editor_id="e", cycle=cycle, run_id="r", mean_saving=10.0, kind=kind)
    return PreferencePair(ex, pv)


def test_export_is_immutable_and_hashed(tmp_path):
    pairs = [mk(1), mk(2, instance="inst-b"), mk(1)]  # duplicate pair_id deduped
    out = tmp_path / "ds"
    m = export_dataset(pairs, out, extra_manifest={"cycle": 0})
    assert m["n_examples"] == 2 and m["pair_ids"] == ["pair-001", "pair-002"] and m["meta"]["cycle"] == 0
    assert m["composition"]["by_family"] == {"fam": 2} and m["verification_modes"] == ["continuation"]
    assert m["dataset_sha256"] == m["preferences_sha256"]
    loaded, man = load_dataset(out)
    assert [p.example for p in loaded] == [mk(1).example, mk(2).example] and man == m
    assert not os.access(out / "preferences.jsonl", os.W_OK)
    assert export_dataset(pairs, out) == m  # identical content: resume returns the sealed export
    with pytest.raises(FileExistsError):
        export_dataset([mk(9)], out)  # different content is refused
    # nested export into a sub-directory of an existing dir works (coordinator layout)
    m2 = export_dataset([mk(3)], tmp_path / "nest" / "current")
    m3 = export_dataset([mk(3)], tmp_path / "nest")
    assert m2["dataset_sha256"] == m3["dataset_sha256"]
    # tampering is detected
    os.chmod(out / "preferences.jsonl", 0o644)
    (out / "preferences.jsonl").write_text("{}\n")
    with pytest.raises(PreferenceError, match="sha256 mismatch"):
        load_dataset(out)


def test_empty_export_is_a_valid_no_update_dataset(tmp_path):
    m = export_dataset([], tmp_path / "empty")
    assert m["n_examples"] == 0 and (tmp_path / "empty" / "preferences.jsonl").read_text() == ""


def test_conflicting_duplicate_pair_ids_refused(tmp_path):
    a, b = mk(1), mk(2)
    clash = PreferencePair(b.example.model_copy(update={"pair_id": "pair-001"}), a.provenance)
    with pytest.raises(PreferenceError, match="conflicting"):
        export_dataset([a, clash], tmp_path / "x")


@pytest.mark.parametrize(
    "pairs, kwargs, match",
    [
        ([mk(1), mk(2, split=Split.DEV)], {}, "held-out instance"),
        ([mk(1), mk(2, split=Split.FINAL)], {}, "held-out instance"),
        ([mk(1, instance="inst-h")], {"heldout_instance_ids": {"inst-h"}}, "is held out"),
        ([mk(1, family="heldout-fam")], {"forbidden_families": {"heldout-fam"}}, "forbidden family"),
        ([mk(1), mk(2, kind="fixture")], {}, "mixed pair kinds"),
        ([mk(1, instance="other")], {"train_instance_ids": {"inst-a"}}, "not in the training panel"),
    ],
)
def test_export_refusals(tmp_path, pairs, kwargs, match):
    with pytest.raises(PreferenceError, match=match):
        export_dataset(pairs, tmp_path / "x", **kwargs)
    assert not (tmp_path / "x" / "manifest.json").exists()


def test_freeze_fixed_dataset_copies_bytes(tmp_path):
    src = tmp_path / "frozen"
    m0 = export_dataset([mk(1), mk(2)], src)
    m1 = freeze_fixed_dataset(src, tmp_path / "c0" / "dataset")
    m2 = freeze_fixed_dataset(src, tmp_path / "c1" / "dataset")
    assert m1["dataset_sha256"] == m2["dataset_sha256"] == m0["dataset_sha256"]
    assert m1["meta"]["condition"] == "fixed_dataset" and m1["n_examples"] == 2
    with pytest.raises(PreferenceError):
        freeze_fixed_dataset(src, tmp_path / "c2", heldout_instance_ids={"inst-a"})


def test_freeze_repo_training_fixture(tmp_path):
    m = freeze_fixed_dataset(REPO / "tests" / "fixtures" / "train" / "fixture_prefs", tmp_path / "ds")
    assert m["kinds"] == ["fixture"] and m["n_examples"] >= 1


# --------------------------------------------------------------------------- #
# Selection and buffer
# --------------------------------------------------------------------------- #


def pool(n_inst: int, per: int, start: int = 0, cycle: int = 0) -> list[PreferencePair]:
    return [mk(start + i * per + j, instance=f"inst-{i}", cycle=cycle) for i in range(n_inst) for j in range(per)]


def ids(pairs):
    return [p.example.pair_id for p in pairs]


def test_selection_is_deterministic_and_order_independent():
    cur, hist = pool(3, 2, 0), pool(4, 5, 100)
    a = select_pairs(cur, hist, selection="current_and_history", history_fraction=0.5, seed=7)
    b = select_pairs(list(reversed(cur)), list(reversed(hist)), selection="current_and_history", history_fraction=0.5, seed=7)
    assert ids(a) == ids(b)
    assert len(a) == 12 and set(ids(cur)) <= set(ids(a))  # all current + an equal number of history pairs
    c = select_pairs(cur, hist, selection="current_and_history", history_fraction=0.5, seed=8)
    assert ids(c) != ids(a)


def test_selection_is_task_balanced():
    hist = [mk(100 + j, instance="big") for j in range(20)] + pool(4, 1, 200)
    sel = select_pairs(pool(1, 5), hist, selection="current_and_history", history_fraction=0.5, seed=1)
    hist_sel = [p for p in sel if p.example.pair_id >= "pair-100"]
    counts = {}
    for p in hist_sel:
        counts[p.provenance.instance_id] = counts.get(p.provenance.instance_id, 0) + 1
    assert len(hist_sel) == 5 and set(counts) == {"big", "inst-0", "inst-1", "inst-2", "inst-3"}


def test_current_only_and_no_update():
    cur, hist = pool(2, 2), pool(2, 2, 100)
    assert ids(select_pairs(cur, hist, selection="current_only", history_fraction=0.5, seed=1)) == sorted(ids(cur))
    assert select_pairs([], hist, selection="current_and_history", history_fraction=0.5, seed=1) == []
    capped = select_pairs(cur, hist, selection="current_and_history", history_fraction=0.5, seed=1, max_examples=2)
    assert len(capped) == 2 and len([p for p in capped if p.example.pair_id < "pair-100"]) == 1


def test_bounded_history_and_config_wrapper():
    hist = pool(5, 10, 100)
    b = bound_history(hist, 12, seed=3)
    assert len(b) == 12 and ids(b) == ids(bound_history(list(reversed(hist)), 12, seed=3))
    assert {p.provenance.instance_id for p in b} == {f"inst-{i}" for i in range(5)}
    with pytest.raises(PreferenceError):
        bound_history([mk(1, split=Split.DEV)], 5, seed=0)
    cfg = DataSelectionConfig(selection="current_and_history", buffer_capacity=4, history_fraction=0.5)
    sel = select_training_pairs(pool(2, 3), hist, cfg, seed=11)
    assert len(sel) == 6 + 4  # history bounded to 4 by capacity
    assert all(p.provenance.cycle == 0 for p in sel)  # original provenance kept
    cfg2 = DataSelectionConfig(selection="current_only")
    assert len(select_training_pairs(pool(2, 3), hist, cfg2, seed=11)) == 6
