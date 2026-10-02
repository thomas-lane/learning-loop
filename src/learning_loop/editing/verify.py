"""Branch verification of edit proposals and the acceptance rule.

Continuation verification (default)
-----------------------------------
From the SAME source events two ReplaySpecs are built that differ only in the
fixed intervention message (original vs edited). Each runs as a fresh BRANCH
episode through an `EpisodeBackend`: clean reset, deterministic replay of turns
0..k-1 (no model calls; observations and fingerprints checked, fail closed),
the fixed action at turn k, then a fresh continuation by the frozen current
learner. Both branches of a repetition use the same continuation seed
(`seeds.continuation_seed`; the branch label is not an input) and the same
budgets. Nothing observed downstream of turn k in the source run is reused.

Counterfactual episode token cost per branch (`records.BranchCost`)::

    shared prefix       = input+output of source requests 0..k-1
    intervention input  = input tokens of source request k (shared, counted once)
    intervention tokens = learner-tokenizer length of the rendered fixed turn
    continuation        = provider input+output of the branch's own requests

The source's generated completion at turn k is represented only by its rendered
length (never added again). Verification operational usage (tokens actually
spent on continuations) is reported separately in `operational_usage`.

Acceptance rule `strict_all_success_v1`: valid replay on every branch; the fixed
action at turn k executed without tool error or timeout on every branch (recorded
by the episode loop; a missing record fails closed); complete
success on BOTH branches for EVERY repetition; only model-finished stops (no
budget/safety/infra/replay/model-error); every cost present; mean saving
(original - edited) > 0, >= min_token_saving and >= min_relative_saving of the
original. Anything else is rejected with every failed criterion listed. With a
single repetition an accepted record is labeled "one observed successful
preference", not proof of reliable improvement.

Local verification (`LocalVerifier`)
-----------------------------------
Only for tasks whose StateSpec names a supported `local_equivalence` contract.
Implemented: `same_state_after_action` - from two freshly restored states, the
declared-state fingerprint after the edited action equals the fingerprint after
the original action, both actions execute without tool errors, the source
episode succeeded, and the edited rendered turn is shorter. No continuation is
run, so these records are labeled mode=local, their costs cover only the
fixed turn (mean_cost_* = rendered intervention tokens; continuation None),
and they must not be mixed with continuation-verified pairs.
"""

from __future__ import annotations

import json
import math
import re
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .editor import SourceContext, make_edited_message, parse_call_arguments, tool_calls_of
from ..episodes.events import first_request_messages, load_turns
from ..core.interfaces import (
    EnvironmentSession,
    EpisodeBackend,
    EpisodePlan,
    EpisodeResult,
    ReplaySpec,
    ReplayTurn,
    StateSpec,
)
from ..episodes.episode import source_image_identity
from ..core.records import (
    BranchCost,
    BranchResult,
    EditProposal,
    EpisodeRole,
    EpisodeSummary,
    Event,
    Message,
    RestoreCapability,
    SamplingConfig,
    PolicySpec,
    StopCategory,
    TaskInstance,
    ToolSchema,
    TurnRecord,
    Usage,
    VerificationRecord,
)
from ..core.seeds import continuation_seed, stable_id
from ..core.storage import atomic_write_json, read_json

ACCEPTANCE_RULE = "strict_all_success_v1"
LOCAL_ACCEPTANCE_RULE = "local_same_state_v1"
LOCAL_CONTRACTS = ("same_state_after_action",)

InterventionCounter = Callable[[list[Message], Message, list[ToolSchema]], tuple[int | None, str]]


def fixture_intervention_counter(prompt: list[Message], message: Message, tools: list[ToolSchema]) -> tuple[int, str]:
    """Clearly-labeled estimate for scripted fixtures: ceil(chars/4) of the turn JSON."""
    text = json.dumps({k: message.get(k) for k in ("content", "tool_calls")}, ensure_ascii=False, sort_keys=True)
    return math.ceil(len(text) / 4), "fixture_estimate"


class UnsupportedLocalContract(ValueError):
    pass


def check_local_contract(state_spec: StateSpec) -> str:
    """Validation for verification.mode=local: the task must name a supported contract."""
    c = state_spec.local_equivalence
    if c is None:
        raise UnsupportedLocalContract("task declares no local_equivalence contract; local verification unsupported")
    if c not in LOCAL_CONTRACTS:
        raise UnsupportedLocalContract(f"unsupported local_equivalence contract {c!r} (supported: {', '.join(LOCAL_CONTRACTS)})")
    if not state_spec.fingerprint_paths:
        raise UnsupportedLocalContract(f"{c} needs declared fingerprint_paths")
    return c


# --------------------------------------------------------------------------- #
# Source context and replay specs
# --------------------------------------------------------------------------- #


def build_replay_spec(
    events: list[Event],
    turns: list[TurnRecord],
    turn_index: int,
    label: str,
    intervention_message: Message,
    source_episode_id: str,
) -> ReplaySpec:
    """Replay spec for the decision state before `turn_index`, from source events only.

    Uses the exact messages of source request k as the history prefix and turns
    0..k-1 (executed arguments, shown observations, fingerprints) as prefix turns.
    Nothing from turn k onward is included."""
    by_index = {t.turn_index: t for t in turns}
    if turn_index not in by_index:
        raise ValueError(f"turn {turn_index} not in source")
    if sorted(i for i in by_index if i < turn_index) != list(range(turn_index)):
        raise ValueError(f"source turns before {turn_index} are not contiguous")
    history = first_request_messages(events, turn_index)
    if history is None:
        raise ValueError(f"no model request recorded for turn {turn_index}")
    n_assistant = sum(1 for m in history if m.get("role") == "assistant")
    if n_assistant != turn_index:
        raise ValueError(f"history prefix has {n_assistant} assistant messages, expected {turn_index}")
    prefix_turns = []
    for i in range(turn_index):
        t = by_index[i]
        prefix_turns.append(
            ReplayTurn(
                turn_index=i,
                assistant_message=t.assistant_message,
                executed_arguments=[te.executed_arguments if te.executed else None for te in t.tool_executions],
                expected_observations=[te.observation for te in t.tool_executions],
                expected_fingerprint_before=t.fingerprint_before,
            )
        )
    k = by_index[turn_index]
    return ReplaySpec(
        source_episode_id=source_episode_id,
        history_prefix=json.loads(json.dumps(history)),
        prefix_turns=prefix_turns,
        intervention_turn=turn_index,
        intervention_label=label,
        intervention_message=intervention_message,
        expected_fingerprint_before_intervention=k.fingerprint_before,
        prefix_usage=Usage.sum([by_index[i].usage or Usage(input_tokens=None, output_tokens=None) for i in range(turn_index)]),
        intervention_request_input_tokens=k.usage.input_tokens if k.usage else None,
        expected_image_identity=source_image_identity(list(events)),
    )


def branch_replay_specs(proposal: EditProposal, src: SourceContext) -> dict[str, ReplaySpec]:
    """Original and edited specs from the same source events (differ only in the fixed turn)."""
    if proposal.turn_index is None or proposal.replacement is None:
        raise ValueError("proposal has no turn/replacement")
    k = proposal.turn_index
    turn = next(t for t in src.turns if t.turn_index == k)
    original_msg = turn.assistant_message
    edited_msg = make_edited_message(original_msg, proposal.replacement)
    sid = src.summary.episode_id
    return {
        "original": build_replay_spec(src.events, src.turns, k, "original", original_msg, sid),
        "edited": build_replay_spec(src.events, src.turns, k, "edited", edited_msg, sid),
    }


def branch_plan(
    source_plan: EpisodePlan,
    spec: ReplaySpec,
    *,
    episode_id: str,
    seed: int,
    policy: PolicySpec | None = None,
    sampling: SamplingConfig | None = None,
) -> EpisodePlan:
    """A fresh BRANCH episode plan: same task, prompt, tools, budgets and state spec."""
    pol = (policy or source_plan.policy).model_copy(deep=True)
    if sampling is not None:
        pol.sampling = sampling.model_copy(deep=True)
    return source_plan.model_copy(
        update={
            "episode_id": episode_id,
            "role": EpisodeRole.BRANCH,
            "attempt_index": None,
            "seed": seed,
            "policy": pol,
            "replay": spec,
            "record_fingerprints": True,
        },
        deep=True,
    )


def continuation_usage(result: EpisodeResult) -> Usage:
    """Usage of the branch's own model requests (continuation after turn k)."""
    path = result.summary.events_path
    if path and Path(path).exists():
        turns = load_turns(Path(path))
        model_turns = [t for t in turns if t.origin == "model"]
        return Usage.sum([t.usage or Usage(input_tokens=None, output_tokens=None) for t in model_turns])
    return result.summary.usage


def compute_branch_cost(
    spec: ReplaySpec, continuation: Usage, counter: InterventionCounter, tools: list[ToolSchema]
) -> BranchCost:
    n, source = counter(spec.history_prefix, spec.intervention_message, tools)
    return BranchCost(
        shared_prefix_tokens=spec.prefix_usage.total,
        intervention_request_input_tokens=spec.intervention_request_input_tokens,
        intervention_tokens=n,
        intervention_tokens_source=source,
        continuation_tokens=continuation.total,
    )


# --------------------------------------------------------------------------- #
# Acceptance
# --------------------------------------------------------------------------- #


@dataclass
class Acceptance:
    accepted: bool
    reasons: list[str]
    mean_cost_original: float | None
    mean_cost_edited: float | None
    mean_saving: float | None
    evidence_label: str | None


def intervention_tool_reasons(b: BranchResult, *, require_record: bool = True) -> list[str]:
    """Reasons from the branch's executed fixed action at turn k.

    The episode loop records `episode.extra["intervention_tool"] =
    {"executed", "error", "exit_code", "timed_out"}` after executing the
    intervention. A fixed action that did not execute, raised a tool error or
    timed out is not a valid comparison point even if the continuation later
    succeeds. A missing record fails closed (`require_record`), except when
    replay failed before the intervention was reached (already rejected). A
    nonzero exit code alone is not a failure (`grep -c` with no match exits 1)."""
    tag = f"{b.branch}:r{b.repetition}"
    rec = b.episode.extra.get("intervention_tool")
    if not isinstance(rec, Mapping):
        return [f"missing_intervention_record:{tag}"] if require_record and b.replay_ok else []
    out = []
    if rec.get("timed_out"):
        out.append(f"intervention_timeout:{tag}")
    if rec.get("executed") is not True or (rec.get("error") and not rec.get("timed_out")):
        out.append(f"intervention_tool_error:{tag}")
    return out


def evaluate_strict_all_success(
    branches: list[BranchResult],
    repetitions: int,
    *,
    min_token_saving: float,
    min_relative_saving: float = 0.0,
    require_intervention_record: bool = True,
) -> Acceptance:
    """`strict_all_success_v1`; reasons list every failed criterion.

    Continuation mode requires each branch's intervention execution record
    (`intervention_tool_reasons`); it fails closed when the record is missing."""
    reasons: list[str] = []
    for label in ("original", "edited"):
        reps = sorted(b.repetition for b in branches if b.branch == label)
        if len(reps) != repetitions or len(set(reps)) != repetitions:
            reasons.append(f"missing_branches:{label}:{len(reps)}/{repetitions}")
    seeds: dict[int, set[int]] = {}
    for b in branches:
        seeds.setdefault(b.repetition, set()).add(b.continuation_seed)
    if any(len(s) != 1 for s in seeds.values()):
        reasons.append("unmatched_continuation_seeds")
    for b in sorted(branches, key=lambda b: (b.repetition, b.branch)):
        tag = f"{b.branch}:r{b.repetition}"
        ep = b.episode
        if not b.replay_ok:
            reasons.append(f"replay_failed:{tag}")
        if ep.stop_category != StopCategory.MODEL:
            reasons.append(f"bad_stop:{tag}:{ep.stop_reason}")
        if ep.success is None:
            reasons.append(f"not_graded:{tag}")
        elif not ep.success:
            reasons.append(f"not_success:{tag}")
        if b.cost.total is None:
            reasons.append(f"missing_cost:{tag}")
        reasons += intervention_tool_reasons(b, require_record=require_intervention_record)
    orig = [b.cost.total for b in branches if b.branch == "original"]
    edit = [b.cost.total for b in branches if b.branch == "edited"]
    mean_o = mean_e = saving = None
    if orig and edit and all(c is not None for c in orig + edit):
        mean_o = sum(orig) / len(orig)  # type: ignore[arg-type]
        mean_e = sum(edit) / len(edit)  # type: ignore[arg-type]
        saving = mean_o - mean_e
        if saving == 0:
            reasons.append("tie")
        elif saving < 0:
            reasons.append(f"no_saving:{saving:g}")
        else:
            if saving < min_token_saving:
                reasons.append(f"saving_below_min:{saving:g}<{min_token_saving:g}")
            if mean_o > 0 and saving / mean_o < min_relative_saving:
                reasons.append(f"relative_saving_below_min:{saving / mean_o:.4f}<{min_relative_saving:g}")
    elif not reasons:
        reasons.append("missing_cost")
    accepted = not reasons
    label = None
    if accepted:
        label = "one_observed_successful_preference" if repetitions == 1 else f"all_{repetitions}_continuation_pairs_successful"
    return Acceptance(accepted, reasons or ["accepted"], mean_o, mean_e, saving, label)


# --------------------------------------------------------------------------- #
# Continuation verifier
# --------------------------------------------------------------------------- #

Sources = Mapping[str, SourceContext] | Callable[[str], SourceContext]
PlanFactory = Callable[[TaskInstance, EpisodeRole, str, int, ReplaySpec], EpisodePlan]
_PLAN_INVARIANTS = ("system_prompt", "instruction", "tools", "state_spec", "budgets", "instance_id")


def _preserve(path: Path) -> None:
    if not path.exists():
        return
    n = 1
    while path.with_name(f"{path.name}.interrupted-{n}").exists():
        n += 1
    path.rename(path.with_name(f"{path.name}.interrupted-{n}"))


class _VerifierBase:
    mode: str
    sources: Sources | None
    out_root: Path | None

    def _resolve(
        self,
        proposal: EditProposal,
        purpose: str,
        round_index: int,
        repetitions: int,
        verification_id: str | None,
        source_dir: str | Path | None,
        instance: TaskInstance | None,
        out_dir: str | Path | None,
    ) -> tuple[SourceContext, str, Path]:
        if proposal.status != "proposed":
            raise ValueError(f"only valid proposals are verified (status={proposal.status})")
        if source_dir is not None:
            if instance is None:
                raise ValueError("source_dir needs the task instance")
            src = SourceContext.from_dir(source_dir, instance)
        elif self.sources is not None:
            src = self.sources(proposal.source_episode_id) if callable(self.sources) else self.sources[proposal.source_episode_id]
        else:
            raise ValueError("no source: pass source_dir+instance or construct the verifier with sources")
        if src.summary.episode_id != proposal.source_episode_id:
            raise ValueError("source episode does not match the proposal")
        vid = verification_id or stable_id("verif", proposal.proposal_id, purpose, round_index, self.mode, repetitions)
        if out_dir is not None:
            vdir = Path(out_dir)
        elif self.out_root is not None:
            vdir = self.out_root / purpose / vid
        else:
            raise ValueError("no output location: pass out_dir or construct the verifier with out_root")
        return src, vid, vdir


class ContinuationVerifier(_VerifierBase):
    """Verifies proposals by fresh original/edited continuations (see module doc)."""

    mode = "continuation"

    def __init__(
        self,
        backend: EpisodeBackend,
        sources: Sources | None = None,
        *,
        root_seed: int,
        out_root: str | Path | None = None,
        continuations_per_branch: int = 1,
        min_token_saving: float = 1.0,
        min_relative_saving: float = 0.0,
        intervention_counter: InterventionCounter = fixture_intervention_counter,
        policy: PolicySpec | None = None,
        sampling: SamplingConfig | None = None,
        plan_factory: PlanFactory | None = None,
        acceptance_rule: str = ACCEPTANCE_RULE,
    ):
        if acceptance_rule != ACCEPTANCE_RULE:
            raise ValueError(f"unknown acceptance rule {acceptance_rule!r}")
        if min_token_saving <= 0:
            raise ValueError("min_token_saving must be > 0 (ties are never accepted)")
        if continuations_per_branch < 1:
            raise ValueError("continuations_per_branch must be >= 1")
        self.backend = backend
        self.sources = sources
        self.root_seed = root_seed
        self.out_root = Path(out_root) if out_root is not None else None
        self.repetitions = continuations_per_branch
        self.min_token_saving = min_token_saving
        self.min_relative_saving = min_relative_saving
        self.counter = intervention_counter
        self.policy = policy
        self.sampling = sampling
        self.plan_factory = plan_factory
        self.acceptance_rule = acceptance_rule

    def _precheck(self, proposal: EditProposal, src: SourceContext) -> list[str]:
        reasons = []
        cap = self.backend.restore_capability(src.instance)
        if cap != RestoreCapability.DETERMINISTIC_REPLAY:
            reasons.append(f"restore_unsupported:{cap.value}")
        if src.plan.state_spec.restore != RestoreCapability.DETERMINISTIC_REPLAY:
            reasons.append(f"state_spec_restore:{src.plan.state_spec.restore.value}")
        turn = next((t for t in src.turns if t.turn_index == proposal.turn_index), None)
        if turn is None:
            reasons.append(f"turn_not_found:{proposal.turn_index}")
        elif src.plan.state_spec.fingerprint_paths and turn.fingerprint_before is None:
            reasons.append("missing_source_fingerprint")
        return reasons

    def make_branch_plan(self, src: SourceContext, spec: ReplaySpec, episode_id: str, seed: int) -> EpisodePlan:
        if self.plan_factory is None:
            return branch_plan(src.plan, spec, episode_id=episode_id, seed=seed, policy=self.policy, sampling=self.sampling)
        plan = self.plan_factory(src.instance, EpisodeRole.BRANCH, episode_id, seed, spec)
        plan = plan.model_copy(update={"role": EpisodeRole.BRANCH, "replay": spec, "seed": seed, "attempt_index": None, "record_fingerprints": True})
        drift = [k for k in _PLAN_INVARIANTS if getattr(plan, k) != getattr(src.plan, k)]
        if drift:
            # The restored conversation would not match what the learner saw.
            raise ValueError(f"branch plan differs from the source plan in {drift}")
        return plan

    async def verify(
        self,
        proposal: EditProposal,
        purpose: str = "acceptance",
        *,
        verification_id: str | None = None,
        source_dir: str | Path | None = None,
        instance: TaskInstance | None = None,
        out_dir: str | Path | None = None,
        round_index: int = 0,
    ) -> VerificationRecord:
        reps = self.repetitions
        src, vid, vdir = self._resolve(proposal, purpose, round_index, reps, verification_id, source_dir, instance, out_dir)
        rec_path = vdir / "verification.json"
        if rec_path.exists():  # completed earlier: never redo logical work
            return VerificationRecord.model_validate(read_json(rec_path))
        pre = self._precheck(proposal, src)
        if pre:
            rec = VerificationRecord(
                verification_id=vid, proposal_id=proposal.proposal_id, mode="continuation",
                acceptance_rule=self.acceptance_rule, accepted=False, reasons=pre, purpose=purpose,  # type: ignore[arg-type]
                operational_usage=Usage(input_tokens=0, output_tokens=0, source="none"),
            )
            atomic_write_json(rec_path, rec)
            return rec
        specs = branch_replay_specs(proposal, src)
        branches: list[BranchResult] = []
        spent: list[Usage] = []
        for r in range(reps):
            rep = round_index * reps + r
            seed = continuation_seed(self.root_seed, proposal.proposal_id, rep, purpose)
            for label in ("original", "edited"):
                spec = specs[label]
                eid = stable_id("branch", proposal.proposal_id, purpose, label, rep)
                plan = self.make_branch_plan(src, spec, eid, seed)
                bdir = vdir / f"{label}-r{rep:02d}"
                _preserve(bdir)
                result = await self.backend.run(src.instance, plan, bdir)
                cont = continuation_usage(result)
                spent.append(cont)
                replay_ok = bool(result.replay_ok) and result.summary.stop_category != StopCategory.REPLAY
                branches.append(
                    BranchResult(
                        branch=label,  # type: ignore[arg-type]
                        repetition=rep,
                        continuation_seed=seed,
                        episode=result.summary,
                        replay_ok=replay_ok,
                        replay_mismatches=list(result.replay_mismatches),
                        cost=compute_branch_cost(spec, cont, self.counter, src.plan.tools),
                    )
                )
        acc = evaluate_strict_all_success(branches, reps, min_token_saving=self.min_token_saving, min_relative_saving=self.min_relative_saving)
        rec = VerificationRecord(
            verification_id=vid,
            proposal_id=proposal.proposal_id,
            mode="continuation",
            acceptance_rule=self.acceptance_rule,
            branches=branches,
            accepted=acc.accepted,
            reasons=acc.reasons,
            mean_cost_original=acc.mean_cost_original,
            mean_cost_edited=acc.mean_cost_edited,
            mean_saving=acc.mean_saving,
            operational_usage=Usage.sum(spent),
            purpose=purpose,  # type: ignore[arg-type]
            evidence_label=acc.evidence_label,
        )
        atomic_write_json(rec_path, rec)
        return rec


# --------------------------------------------------------------------------- #
# Local verifier
# --------------------------------------------------------------------------- #

SessionFactory = Callable[[TaskInstance], AbstractAsyncContextManager[EnvironmentSession]]


def apply_normalizers(text: str, normalizers: list[dict[str, str]]) -> tuple[str, list[str]]:
    """Apply only task-declared normalizers; returns (text, reasons of those that matched)."""
    applied = []
    for n in normalizers:
        new = re.sub(n["pattern"], n.get("replacement", ""), text)
        if new != text:
            applied.append(n.get("reason") or n["pattern"])
        text = new
    return text, applied


async def replay_prefix(session: EnvironmentSession, spec: ReplaySpec, state_spec: StateSpec) -> tuple[list[str], list[str]]:
    """Re-execute prefix turns in a fresh session; returns (mismatches, normalizers applied)."""
    mismatches: list[str] = []
    applied: list[str] = []
    for turn in spec.prefix_turns:
        if turn.expected_fingerprint_before is not None:
            fp = await session.fingerprint(state_spec)
            if fp != turn.expected_fingerprint_before:
                mismatches.append(f"fingerprint_before_turn:{turn.turn_index}")
                return mismatches, applied
        calls = tool_calls_of(turn.assistant_message)
        for i, call in enumerate(calls):
            args = turn.executed_arguments[i] if i < len(turn.executed_arguments) else None
            if args is None:
                continue  # not executed in the source either (visible error observation)
            name = (call.get("function") or {}).get("name", "")
            te = await session.execute(call.get("id", ""), name, args, (call.get("function") or {}).get("arguments"))
            got, a1 = apply_normalizers(te.observation, state_spec.observation_normalizers)
            exp, a2 = apply_normalizers(turn.expected_observations[i], state_spec.observation_normalizers)
            applied += a1 + a2
            if got != exp:
                mismatches.append(f"observation:turn{turn.turn_index}:call{i}")
                return mismatches, applied
    if spec.expected_fingerprint_before_intervention is not None:
        fp = await session.fingerprint(state_spec)
        if fp != spec.expected_fingerprint_before_intervention:
            mismatches.append(f"fingerprint_before_turn:{spec.intervention_turn}")
    return mismatches, sorted(set(applied))


class LocalVerifier(_VerifierBase):
    """Cheaper verification for tasks with an explicit local equivalence contract."""

    mode = "local"

    def __init__(
        self,
        session_factory: SessionFactory,
        sources: Sources | None = None,
        *,
        out_root: str | Path | None = None,
        min_token_saving: float = 1.0,
        intervention_counter: InterventionCounter = fixture_intervention_counter,
    ):
        if min_token_saving <= 0:
            raise ValueError("min_token_saving must be > 0")
        self.session_factory = session_factory
        self.sources = sources
        self.out_root = Path(out_root) if out_root is not None else None
        self.min_token_saving = min_token_saving
        self.counter = intervention_counter

    async def _run_branch(self, src: SourceContext, spec: ReplaySpec, label: str, vid: str) -> tuple[BranchResult, dict[str, Any]]:
        state_spec = src.plan.state_spec
        call = tool_calls_of(spec.intervention_message)[0]
        fn = call.get("function") or {}
        args = parse_call_arguments(call) or {}
        detail: dict[str, Any] = {}
        async with self.session_factory(src.instance) as session:
            mismatches, applied = await replay_prefix(session, spec, state_spec)
            detail["normalizers_applied"] = applied
            te = None
            fp_after = None
            if not mismatches:
                te = await session.execute(call.get("id", ""), fn.get("name", ""), args, fn.get("arguments"))
                fp_after = await session.fingerprint(state_spec)
        detail.update(fingerprint_after=fp_after, tool_error=(te.error if te else None), executed=(te.executed if te else False))
        episode = EpisodeSummary(
            episode_id=stable_id("local", vid, label),
            role=EpisodeRole.BRANCH,
            instance_id=src.instance.instance_id,
            checkpoint_id=src.summary.checkpoint_id,
            stop_reason="replay:" + ";".join(mismatches) if mismatches else "local:action_executed",
            stop_category=StopCategory.REPLAY if mismatches else StopCategory.MODEL,
            usage=Usage(input_tokens=0, output_tokens=0, source="none"),  # no model calls in local mode
            n_requests=0,
            n_tool_calls=sum(len(t.expected_observations) for t in spec.prefix_turns) + (1 if te else 0),
            extra={"verification_mode": "local", **detail, "observation": te.observation if te else None},
        )
        n, source = self.counter(spec.history_prefix, spec.intervention_message, src.plan.tools)
        cost = BranchCost(
            shared_prefix_tokens=spec.prefix_usage.total,
            intervention_request_input_tokens=spec.intervention_request_input_tokens,
            intervention_tokens=n,
            intervention_tokens_source=source,
            continuation_tokens=None,  # not observed in local mode
        )
        br = BranchResult(branch=label, repetition=0, continuation_seed=0, episode=episode, replay_ok=not mismatches, replay_mismatches=mismatches, cost=cost)  # type: ignore[arg-type]
        return br, detail

    async def verify(
        self,
        proposal: EditProposal,
        purpose: str = "acceptance",
        *,
        verification_id: str | None = None,
        source_dir: str | Path | None = None,
        instance: TaskInstance | None = None,
        out_dir: str | Path | None = None,
        round_index: int = 0,
    ) -> VerificationRecord:
        src, vid, vdir = self._resolve(proposal, purpose, round_index, 1, verification_id, source_dir, instance, out_dir)
        check_local_contract(src.plan.state_spec)
        rec_path = vdir / "verification.json"
        if rec_path.exists():
            return VerificationRecord.model_validate(read_json(rec_path))
        specs = branch_replay_specs(proposal, src)
        branches, details = [], {}
        for label in ("original", "edited"):
            br, det = await self._run_branch(src, specs[label], label, vid)
            branches.append(br)
            details[label] = det
        reasons: list[str] = []
        if src.summary.success is not True:
            reasons.append("source_not_successful")
        for b in branches:
            if not b.replay_ok:
                reasons.append(f"replay_failed:{b.branch}:r0")
            d = details[b.branch]
            if not d["executed"] or d["tool_error"]:
                reasons.append(f"action_error:{b.branch}")
        fo, fe = details["original"]["fingerprint_after"], details["edited"]["fingerprint_after"]
        if fo is None or fe is None:
            reasons.append("missing_fingerprint_after")
        elif fo != fe:
            reasons.append("state_differs_after_action")
        to = branches[0].cost.intervention_tokens
        tn = branches[1].cost.intervention_tokens
        saving = None
        if to is None or tn is None:
            reasons.append("missing_cost")
        else:
            saving = float(to - tn)
            if saving == 0:
                reasons.append("tie")
            elif saving < 0:
                reasons.append(f"no_saving:{saving:g}")
            elif saving < self.min_token_saving:
                reasons.append(f"saving_below_min:{saving:g}<{self.min_token_saving:g}")
        accepted = not reasons
        rec = VerificationRecord(
            verification_id=vid,
            proposal_id=proposal.proposal_id,
            mode="local",
            acceptance_rule=LOCAL_ACCEPTANCE_RULE,
            branches=branches,
            accepted=accepted,
            reasons=reasons or ["accepted"],
            mean_cost_original=float(to) if to is not None else None,
            mean_cost_edited=float(tn) if tn is not None else None,
            mean_saving=saving,
            operational_usage=Usage(input_tokens=0, output_tokens=0, source="none"),
            purpose=purpose,  # type: ignore[arg-type]
            evidence_label="local_same_state_after_action" if accepted else None,
        )
        atomic_write_json(rec_path, rec)
        return rec


# --------------------------------------------------------------------------- #
# Factory (coordinator entry point)
# --------------------------------------------------------------------------- #


def make_verifier(
    *,
    mode: str,
    backend: EpisodeBackend,
    config: Any,
    learner_profile: Any = None,
    policy_spec: PolicySpec | None = None,
    plan_factory: PlanFactory | None = None,
    root_seed: int,
    scripted: bool = False,
    session_factory: SessionFactory | None = None,
    intervention_counter: InterventionCounter | None = None,
) -> ContinuationVerifier | LocalVerifier:
    """Build the configured verifier from a `config.VerificationConfig`.

    Intervention lengths use the learner tokenizer/template
    (`token_count.get_counter(learner_profile)`) unless the run is scripted, in
    which case the labeled fixture estimate is used."""
    if intervention_counter is None:
        from .token_count import get_counter

        # scripted runs: token_count's labeled fixture estimate; otherwise the learner tokenizer/template
        intervention_counter = get_counter(None if scripted else learner_profile)
    if mode == "continuation":
        return ContinuationVerifier(
            backend,
            root_seed=root_seed,
            continuations_per_branch=config.continuations_per_branch,
            min_token_saving=config.min_token_saving,
            min_relative_saving=config.min_relative_saving,
            intervention_counter=intervention_counter,
            policy=policy_spec,
            sampling=config.sampling,
            plan_factory=plan_factory,
            acceptance_rule=config.acceptance_rule,
        )
    if mode == "local":
        factory = session_factory or getattr(backend, "session_factory", None)
        if factory is None:
            raise UnsupportedLocalContract(f"backend {getattr(backend, 'name', backend)!r} provides no session_factory for local verification")
        return LocalVerifier(factory, min_token_saving=config.min_token_saving, intervention_counter=intervention_counter)
    raise ValueError(f"unknown verification mode {mode!r}")
