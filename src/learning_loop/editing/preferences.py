"""Preference pairs from accepted verifications; immutable dataset export; history buffer.

A pair is exactly::

    prompt   = the exact messages sent to the learner for source request k
               (everything before the intervened assistant turn, nothing after)
    chosen   = [edited assistant turn]   same content (empty) + one tool call with
               the original tool_call_id and JSON-string arguments
    rejected = [original assistant turn] exactly as it entered the history
    tools    = the learner's tool schemas

Editor justifications, verifier results and branch continuations live only in
the companion `PreferenceProvenance` (and the verification records it names);
they are never rendered into a prompt.

Exports are immutable: `preferences.jsonl` and `provenance.jsonl` are written
atomically, then `manifest.json` (hashes, counts, composition) seals the
export; files are made read-only; re-exporting identical content returns the
sealed manifest (safe resume) and different content is refused. Export refuses held-out instances (split != train or listed
held-out ids), forbidden families and mixed fixture/verified kinds.

History: `bound_history` keeps at most `buffer_capacity` earlier pairs by a
seeded, task-balanced selection; `select_pairs` mixes current and history pairs
(`current_only` / `current_and_history` with `history_fraction`). With no
current pairs nothing is selected (a no-update cycle).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, NamedTuple

from ..core.records import (
    EditProposal,
    Event,
    Message,
    PreferenceExample,
    PreferenceProvenance,
    Split,
    TaskInstance,
    ToolSchema,
    TurnRecord,
    VerificationRecord,
)
from ..core.seeds import derive_seed, stable_id
from ..core.storage import atomic_write_text, now_iso, read_json, read_jsonl, sha256_file
from ..episodes.events import first_request_messages
from .editor import SourceContext, make_edited_message, parse_call_arguments, tool_calls_of, validate_call

DATASET_FORMAT = "learning_loop.editing.preferences/v1"
_ALLOWED_PROMPT_ROLES = {"system", "user", "assistant", "tool"}


class PreferencePair(NamedTuple):
    example: PreferenceExample
    provenance: PreferenceProvenance


class PreferenceError(ValueError):
    pass


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def check_single_call(message: Message, tools: list[ToolSchema], what: str) -> list[str]:
    """A candidate completion: one assistant message with exactly one valid tool call."""
    errs = []
    if message.get("role") != "assistant":
        errs.append(f"{what}:not_assistant")
    calls = tool_calls_of(message)
    if len(calls) != 1:
        return errs + [f"{what}:tool_calls={len(calls)}"]
    fn = calls[0].get("function") or {}
    if not isinstance(fn.get("arguments"), str):
        errs.append(f"{what}:arguments_not_json_string")
    args = parse_call_arguments(calls[0])
    if args is None:
        return errs + [f"{what}:arguments_not_object"]
    if not calls[0].get("id"):
        errs.append(f"{what}:missing_tool_call_id")
    errs += [f"{what}:{e}" for e in validate_call(tools, fn.get("name", ""), args)]
    return errs


def check_prompt(
    prompt: list[Message],
    turn_index: int,
    forbidden_texts: Iterable[str] = (),
    later_call_ids: Iterable[str] = (),
) -> list[str]:
    """The prompt holds no messages from turn k onward and no editor/verifier text.

    Structural checks: exactly k assistant messages, not ending on an assistant
    message, and no tool-call id or tool result belonging to turn k or later."""
    errs = []
    if not prompt:
        return ["prompt:empty"]
    roles = [m.get("role") for m in prompt]
    if set(roles) - _ALLOWED_PROMPT_ROLES:
        errs.append(f"prompt:unexpected_roles:{sorted(set(roles) - _ALLOWED_PROMPT_ROLES)}")
    n_assistant = roles.count("assistant")
    if n_assistant != turn_index:
        errs.append(f"prompt:assistant_turns={n_assistant}!={turn_index}")
    if roles[-1] == "assistant":
        errs.append("prompt:ends_with_assistant")
    later = {c for c in later_call_ids if c}
    ids_in_prompt = {m.get("tool_call_id") for m in prompt if m.get("role") == "tool"}
    ids_in_prompt |= {c.get("id") for m in prompt for c in tool_calls_of(m)}
    if ids_in_prompt & later:
        errs.append(f"prompt:contains_later_tool_calls:{sorted(ids_in_prompt & later)}")
    blob = json.dumps(prompt, ensure_ascii=False)
    for t in forbidden_texts:
        if t and len(t.strip()) >= 8 and (t in blob or json.dumps(t, ensure_ascii=False)[1:-1] in blob):
            errs.append("prompt:contains_forbidden_text")
            break
    return errs


def verifier_texts(verification: VerificationRecord) -> list[str]:
    """Verifier-side strings that must never appear in a student prompt."""
    out = [r for r in verification.reasons if r != "accepted"]
    for b in verification.branches:
        out.append(b.episode.episode_id)
        out.extend(b.replay_mismatches)
    return out


def pair_id_for(proposal_id: str, verification_id: str) -> str:
    return stable_id("pair", proposal_id, verification_id)


def build_preference(
    *,
    events: list[Event],
    turns: list[TurnRecord],
    proposal: EditProposal,
    verification: VerificationRecord,
    instance: TaskInstance,
    split: Split | str,
    tools: list[ToolSchema],
    learner_checkpoint_id: str,
    cycle: int,
    run_id: str,
    kind: str = "verified",
) -> PreferencePair:
    """Build one pair from an ACCEPTED acceptance-purpose verification; raises PreferenceError."""
    if not verification.accepted:
        raise PreferenceError("verification not accepted")
    if verification.purpose != "acceptance":
        raise PreferenceError(f"{verification.purpose} verifications never create pairs")
    if verification.proposal_id != proposal.proposal_id or proposal.status != "proposed":
        raise PreferenceError("proposal/verification mismatch or invalid proposal")
    assert proposal.turn_index is not None and proposal.replacement is not None
    k = proposal.turn_index
    turn = next((t for t in turns if t.turn_index == k), None)
    if turn is None:
        raise PreferenceError(f"turn {k} not found")
    if turn.malformed or turn.repaired or turn.origin != "model":
        raise PreferenceError("original turn is malformed/repaired/not a model turn")
    prompt = first_request_messages(events, k)
    if prompt is None:
        raise PreferenceError(f"no request recorded for turn {k}")
    prompt = json.loads(json.dumps(prompt))
    rejected = json.loads(json.dumps(turn.assistant_message))
    chosen = make_edited_message(rejected, proposal.replacement)
    later_ids = [c.get("id") for t in turns if t.turn_index >= k for c in tool_calls_of(t.assistant_message)]
    errs = check_prompt(prompt, k, [proposal.justification or "", *verifier_texts(verification)], later_ids)
    errs += check_single_call(chosen, tools, "chosen") + check_single_call(rejected, tools, "rejected")
    if chosen == rejected:
        errs.append("chosen_equals_rejected")
    if tool_calls_of(chosen)[0].get("id") != tool_calls_of(rejected)[0].get("id"):
        errs.append("tool_call_id_changed")
    if errs:
        raise PreferenceError("; ".join(errs))
    pid = pair_id_for(proposal.proposal_id, verification.verification_id)
    example = PreferenceExample(pair_id=pid, prompt=prompt, chosen=[chosen], rejected=[rejected], tools=json.loads(json.dumps(tools)))
    prov = PreferenceProvenance(
        pair_id=pid,
        verification_mode=verification.mode,
        instance_id=instance.instance_id,
        family=instance.family,
        difficulty=instance.difficulty,
        split=Split(split),
        source_episode_id=proposal.source_episode_id,
        proposal_id=proposal.proposal_id,
        verification_id=verification.verification_id,
        learner_checkpoint_id=learner_checkpoint_id,
        editor_id=proposal.editor_id,
        cycle=cycle,
        run_id=run_id,
        mean_saving=verification.mean_saving,
        kind=kind,  # type: ignore[arg-type]
    )
    return PreferencePair(example, prov)


def build_pair(
    *,
    proposal: EditProposal,
    verification: VerificationRecord,
    source_dir: str | Path,
    instance: TaskInstance,
    split: Split | str,
    learner_checkpoint_id: str,
    cycle: int,
    run_id: str,
    kind: str = "verified",
) -> PreferencePair:
    """Coordinator entry point: build a pair from the source item dir (events.jsonl)."""
    ctx = SourceContext.from_dir(source_dir, instance)
    if ctx.summary.episode_id != proposal.source_episode_id:
        raise PreferenceError("source dir does not hold the proposal's source episode")
    return build_preference(
        events=ctx.events,
        turns=ctx.turns,
        proposal=proposal,
        verification=verification,
        instance=instance,
        split=split,
        tools=ctx.plan.tools,
        learner_checkpoint_id=learner_checkpoint_id,
        cycle=cycle,
        run_id=run_id,
        kind=kind,
    )


# --------------------------------------------------------------------------- #
# Export (immutable dataset directories)
# --------------------------------------------------------------------------- #


def dedupe(pairs: Iterable[PreferencePair]) -> list[PreferencePair]:
    """Drop exact duplicates by pair_id; conflicting content under one id is an error."""
    seen: dict[str, PreferencePair] = {}
    for p in pairs:
        pid = p.example.pair_id
        if p.provenance.pair_id != pid:
            raise PreferenceError(f"provenance/example pair_id mismatch: {pid}")
        if pid in seen:
            if seen[pid].example != p.example:
                raise PreferenceError(f"conflicting content for pair_id {pid}")
            continue
        seen[pid] = p
    return list(seen.values())


def check_exportable(
    pairs: list[PreferencePair],
    *,
    forbidden_families: Iterable[str] = (),
    heldout_instance_ids: Iterable[str] | None = None,
    train_instance_ids: set[str] | None = None,
) -> None:
    """Refuse held-out instances (split != train, or listed held-out ids), forbidden
    families, instances outside the training panel and mixed kinds."""
    forbidden = set(forbidden_families)
    heldout = set(heldout_instance_ids or ())
    errs = []
    for p in pairs:
        pv = p.provenance
        if pv.split != Split.TRAIN:
            errs.append(f"{pv.pair_id}: held-out instance {pv.instance_id} (split={pv.split.value})")
        if pv.instance_id in heldout:
            errs.append(f"{pv.pair_id}: instance {pv.instance_id} is held out")
        if pv.family in forbidden:
            errs.append(f"{pv.pair_id}: forbidden family {pv.family}")
        if train_instance_ids is not None and pv.instance_id not in train_instance_ids:
            errs.append(f"{pv.pair_id}: instance {pv.instance_id} not in the training panel")
    kinds = {p.provenance.kind for p in pairs}
    if len(kinds) > 1:
        errs.append(f"mixed pair kinds {sorted(kinds)}: fixture data never mixes with verified data")
    if errs:
        raise PreferenceError("refusing export: " + "; ".join(errs))


def composition(provs: list[PreferenceProvenance]) -> dict[str, Any]:
    fam_diff: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for p in provs:
        fam_diff[p.family][p.difficulty or "none"] += 1
    return {
        "by_family": dict(Counter(p.family for p in provs)),
        "by_difficulty": dict(Counter(p.difficulty or "none" for p in provs)),
        "by_family_difficulty": {f: dict(d) for f, d in fam_diff.items()},
        "by_cycle": {str(k): v for k, v in Counter(p.cycle for p in provs).items()},
        "by_learner_checkpoint": dict(Counter(p.learner_checkpoint_id for p in provs)),
        "n_instances": len({p.instance_id for p in provs}),
    }


def _manifest(out_dir: Path, provs: list[PreferenceProvenance], pair_ids: list[str], meta: dict[str, Any]) -> dict[str, Any]:
    m = {
        "format": DATASET_FORMAT,
        "created_at": now_iso(),
        "n_examples": len(pair_ids),
        "pair_ids": pair_ids,
        "preferences_sha256": sha256_file(out_dir / "preferences.jsonl"),
        "provenance_sha256": sha256_file(out_dir / "provenance.jsonl"),
        "verification_modes": sorted({p.verification_mode for p in provs}),
        "kinds": sorted({p.kind for p in provs}),
        "composition": composition(provs),
        "meta": meta,
    }
    m["dataset_sha256"] = m["preferences_sha256"]  # what the trainer reads and hashes
    return m


def _existing(out_dir: Path, preferences_text: str) -> dict[str, Any] | None:
    """Write-once semantics: an existing export with identical training content is
    returned as is (safe resume); different content is refused."""
    mpath = out_dir / "manifest.json"
    if not mpath.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        return None
    manifest = read_json(mpath)
    if manifest.get("preferences_sha256") != hashlib.sha256(preferences_text.encode()).hexdigest():
        raise FileExistsError(f"dataset already exported with different content (immutable): {out_dir}")
    return manifest


def _seal(out_dir: Path, manifest: dict[str, Any]) -> None:
    """manifest.json is written last: its presence marks a complete, frozen export."""
    atomic_write_text(out_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    for name in ("preferences.jsonl", "provenance.jsonl", "manifest.json"):
        os.chmod(out_dir / name, 0o444)


def export_dataset(
    pairs: Iterable[PreferencePair],
    out_dir: str | Path,
    *,
    forbidden_families: Iterable[str] = (),
    heldout_instance_ids: Iterable[str] | None = None,
    train_instance_ids: set[str] | None = None,
    meta: dict[str, Any] | None = None,
    extra_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write an immutable dataset (preferences.jsonl, provenance.jsonl, manifest.json)
    into `out_dir` and return the manifest. If a sealed export already exists it
    is returned when its training content is identical, else refused. Files are
    written atomically; the manifest last. Other entries in `out_dir` (e.g. a
    nested `current/` export) are left alone."""
    out_dir = Path(out_dir)
    items = sorted(dedupe(pairs), key=lambda p: p.example.pair_id)
    check_exportable(
        items,
        forbidden_families=forbidden_families,
        heldout_instance_ids=heldout_instance_ids,
        train_instance_ids=train_instance_ids,
    )
    prefs_text = "".join(p.example.model_dump_json() + "\n" for p in items)
    existing = _existing(out_dir, prefs_text)
    if existing is not None:
        return existing
    atomic_write_text(out_dir / "preferences.jsonl", prefs_text)
    atomic_write_text(out_dir / "provenance.jsonl", "".join(p.provenance.model_dump_json() + "\n" for p in items))
    manifest = _manifest(out_dir, [p.provenance for p in items], [p.example.pair_id for p in items], {**(meta or {}), **(extra_manifest or {})})
    _seal(out_dir, manifest)
    return manifest


def load_dataset(dataset_dir: str | Path, verify: bool = True) -> tuple[list[PreferencePair], dict[str, Any] | None]:
    """Pairs of a dataset dir; checks file hashes against the manifest when present."""
    d = Path(dataset_dir)
    manifest = read_json(d / "manifest.json") if (d / "manifest.json").exists() else None
    if verify and manifest is not None:
        for name, key in (("preferences.jsonl", "preferences_sha256"), ("provenance.jsonl", "provenance_sha256")):
            if key in manifest and sha256_file(d / name) != manifest[key]:
                raise PreferenceError(f"{d / name}: sha256 mismatch with manifest (dataset modified?)")
    exs = [PreferenceExample.model_validate(r) for r in read_jsonl(d / "preferences.jsonl")]
    provs = {r["pair_id"]: PreferenceProvenance.model_validate(r) for r in read_jsonl(d / "provenance.jsonl")}
    missing = [e.pair_id for e in exs if e.pair_id not in provs]
    if missing:
        raise PreferenceError(f"{d}: pairs without provenance: {missing}")
    return [PreferencePair(e, provs[e.pair_id]) for e in exs], manifest


def load_pairs(dataset_dir: str | Path) -> list[PreferencePair]:
    return load_dataset(dataset_dir)[0]


def freeze_fixed_dataset(
    src: str | Path,
    out_dir: str | Path,
    *,
    allowed_instance_ids: set[str] | None = None,
    heldout_instance_ids: Iterable[str] | None = None,
    forbidden_families: Iterable[str] = (),
) -> dict[str, Any]:
    """Fixed-dataset control: copy a frozen dataset byte-for-byte into this cycle's
    dataset dir (so every cycle trains on identical content, same dataset hash).
    No fresh preferences are ever added. Same export checks as `export_dataset`."""
    src, out_dir = Path(src), Path(out_dir)
    pairs, src_manifest = load_dataset(src)
    ids = [p.example.pair_id for p in pairs]
    if len(set(ids)) != len(ids):
        raise PreferenceError(f"{src}: duplicate pair_ids")
    check_exportable(pairs, forbidden_families=forbidden_families, heldout_instance_ids=heldout_instance_ids, train_instance_ids=allowed_instance_ids)
    existing = _existing(out_dir, (src / "preferences.jsonl").read_text())
    if existing is not None:
        return existing
    for name in ("preferences.jsonl", "provenance.jsonl"):
        atomic_write_text(out_dir / name, (src / name).read_text())
    meta = {
        "condition": "fixed_dataset",
        "fixed_dataset_source": str(src),
        "source_manifest_sha256": sha256_file(src / "manifest.json") if src_manifest is not None else None,
    }
    manifest = _manifest(out_dir, [p.provenance for p in pairs], ids, meta)
    _seal(out_dir, manifest)
    return manifest


# --------------------------------------------------------------------------- #
# Historical buffer and seeded, task-balanced selection
# --------------------------------------------------------------------------- #


def balanced_sample(pairs: list[PreferencePair], n: int, rng: random.Random) -> list[PreferencePair]:
    """Seeded round-robin over task instances: every instance gets a turn before any
    instance gets a second pair. Independent of input order."""
    if n >= len(pairs):
        return sorted(pairs, key=lambda p: p.example.pair_id)
    groups: dict[str, list[PreferencePair]] = defaultdict(list)
    for p in sorted(pairs, key=lambda p: p.example.pair_id):
        groups[p.provenance.instance_id].append(p)
    order = sorted(groups)
    rng.shuffle(order)
    for k in order:
        rng.shuffle(groups[k])
    out: list[PreferencePair] = []
    depth = 0
    while len(out) < n:
        for k in order:
            if depth < len(groups[k]) and len(out) < n:
                out.append(groups[k][depth])
        depth += 1
    return out


def _sample(pairs: list[PreferencePair], n: int, rng: random.Random, task_balanced: bool) -> list[PreferencePair]:
    if task_balanced:
        return balanced_sample(pairs, n, rng)
    ordered = sorted(pairs, key=lambda p: p.example.pair_id)
    return ordered if n >= len(ordered) else rng.sample(ordered, n)


def _rng(seed: int, *parts: Any) -> random.Random:
    return random.Random(derive_seed(seed, "data_selection", *parts))


def bound_history(history: list[PreferencePair], capacity: int, seed: int, task_balanced: bool = True) -> list[PreferencePair]:
    """The bounded historical buffer: at most `capacity` earlier pairs, chosen by a
    seeded (task-balanced) selection. Held-out pairs are refused. Older pairs keep
    their original provenance (learner, cycle, verification) - they are not
    re-verified under the latest learner."""
    merged = dedupe(history)
    bad = sorted({p.provenance.instance_id for p in merged if p.provenance.split != Split.TRAIN})
    if bad:
        raise PreferenceError(f"held-out pairs cannot enter the buffer: {bad}")
    return _sample(merged, min(capacity, len(merged)), _rng(seed, "buffer"), task_balanced)


def select_pairs(
    current: list[PreferencePair],
    history: list[PreferencePair],
    *,
    selection: str,
    history_fraction: float,
    seed: int,
    max_examples: int | None = None,
    task_balanced: bool = True,
) -> list[PreferencePair]:
    """Choose this cycle's training pairs.

    - No current pairs => [] (a no-update cycle; history alone never triggers training).
    - current_only: all current pairs (capped at max_examples).
    - current_and_history: all current pairs plus history pairs so that history is
      `history_fraction` of the selection (history_fraction=1.0 => all history).
      With max_examples, history gets floor(max*fraction) slots and current the rest.
    Pairs already in `current` are never drawn again from `history`.
    """
    current = dedupe(current)
    if not current:
        return []
    cur_ids = {p.example.pair_id for p in current}
    history = [p for p in dedupe(history) if p.example.pair_id not in cur_ids]
    rng_cur, rng_hist = _rng(seed, "current"), _rng(seed, "history")
    if selection not in ("current_only", "current_and_history"):
        raise ValueError(f"unknown selection {selection!r}")
    if selection == "current_only" or history_fraction == 0.0 or not history:
        n_cur = len(current) if max_examples is None else min(len(current), max_examples)
        return _sample(current, n_cur, rng_cur, task_balanced)
    if max_examples is None:
        n_cur = len(current)
        n_hist = len(history) if history_fraction >= 1.0 else round(n_cur * history_fraction / (1.0 - history_fraction))
    else:
        n_hist = int(max_examples * history_fraction)
        n_cur = min(len(current), max_examples - min(n_hist, len(history)))
    n_hist = min(n_hist, len(history))
    return _sample(current, n_cur, rng_cur, task_balanced) + _sample(history, n_hist, rng_hist, task_balanced)


def select_training_pairs(current: list[PreferencePair], history: list[PreferencePair], cfg: Any, *, seed: int) -> list[PreferencePair]:
    """Config-driven selection (`config.DataSelectionConfig`): bound the history
    to `buffer_capacity`, then `select_pairs`."""
    hist = bound_history(history, cfg.buffer_capacity, seed, cfg.task_balanced) if cfg.selection == "current_and_history" else []
    return select_pairs(
        current,
        hist,
        selection=cfg.selection,
        history_fraction=cfg.history_fraction,
        seed=seed,
        max_examples=cfg.max_examples_per_cycle,
        task_balanced=cfg.task_balanced,
    )
