"""Typed, versioned records shared by every stage.

These are the contracts between collection, editing/verification, dataset
construction, training and reporting. Each record carries `schema_version`;
bump it (and add a reader shim) when a field changes meaning.

Conventions:
- Missing measurements are `None` ("unavailable"), never 0.
- Provider payloads are kept verbatim (`raw`) next to normalized fields.
- Messages use the OpenAI chat format exactly as supplied to the learner.
  Tool-call `arguments` are the JSON *string* that was placed in the history.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1

Message = dict[str, Any]  # OpenAI chat message as sent to the learner
ToolSchema = dict[str, Any]  # OpenAI function-tool schema


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)
    schema_version: int = SCHEMA_VERSION


# --------------------------------------------------------------------------- #
# Tasks and identities
# --------------------------------------------------------------------------- #


class Split(str, Enum):
    TRAIN = "train"
    DEV = "dev"
    FINAL = "final"
    EXTERNAL = "external"


class RestoreCapability(str, Enum):
    """What an environment can honestly promise about restoring a decision state."""

    NONE = "none"  # no branching from this environment
    DETERMINISTIC_REPLAY = "deterministic_replay"  # clean reset + replay, verified by fingerprints
    APPROXIMATE_REPLAY = "approximate_replay"  # simulator-style; never mixed with exact replay


class TaskInstance(Record):
    """One concrete, content-addressed task instance (a Harbor task dir)."""

    instance_id: str  # stable, human-readable, e.g. "log-triage/easy/s3"
    family: str
    difficulty: str | None = None  # easy | medium | hard; None for a task directory not rendered from a split
    skills: list[str] = Field(default_factory=list)
    generator: str | None = None  # family@vN that rendered the task; None when not rendered from a split
    generator_seed: int | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    task_dir: str  # path of the Harbor task directory
    content_hash: str  # sha256 over the task dir (instruction, env, tests, solution)
    public_content_hash: str | None = None  # hash of learner-visible inputs only (collision checks)
    restore: RestoreCapability = RestoreCapability.NONE


# --------------------------------------------------------------------------- #
# Model / checkpoint identities
# --------------------------------------------------------------------------- #


class CheckpointRef(Record):
    """Immutable learner identity. `adapter_path` is None for the base model."""

    checkpoint_id: str  # e.g. "base" or "<run_id>/c001-<hash8>"
    model_profile: str
    base_model: str  # HF repo id
    base_revision: str  # exact commit sha
    adapter_path: str | None = None
    adapter_sha256: str | None = None  # hash over adapter weight+config files
    parent_checkpoint_id: str | None = None


class SamplingConfig(BaseModel):
    """Decoding settings sent with every request of one role (learner, editor, continuations)."""

    model_config = ConfigDict(extra="forbid")
    temperature: float | None = Field(default=None, description="Sampling temperature; null leaves the server default (OpenAI semantics: 1.0); 0 is greedy.")
    top_p: float | None = Field(default=None, description="Nucleus sampling threshold; null leaves the server default.")
    max_output_tokens: int | None = Field(default=None, description="Per-request generation limit (sent as `max_tokens`), reasoning included.")
    extra_body: dict[str, Any] = Field(default_factory=dict, description="Extra request fields passed through verbatim (e.g. `{top_k: 20}` for llama.cpp/vLLM).")


class PolicySpec(Record):
    """How to reach a policy. Secret-free: keys are env-var *names*."""

    kind: Literal["openai", "scripted"]
    checkpoint: CheckpointRef | None = None
    served_model_name: str | None = None  # the `model` field sent to the endpoint
    api_base: str | None = None
    api_key_env: str | None = None
    send_seed: bool = True
    seed_supported: bool | None = None  # declared by the model profile; None = unknown
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    scripted_path: str | None = None  # fixture scripts only (tests / smoke)
    request_timeout_sec: float | None = None  # per-request timeout (machine profile)
    max_concurrent_requests: int | None = None  # per-endpoint request concurrency (machine profile)


# --------------------------------------------------------------------------- #
# Usage and events
# --------------------------------------------------------------------------- #


class Usage(BaseModel):
    """Token usage for one request (or a sum). Subset fields never add to totals."""

    model_config = ConfigDict(extra="forbid")
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None  # subset of input_tokens
    reasoning_tokens: int | None = None  # subset of output_tokens
    source: Literal["provider", "fixture_estimate", "mixed", "none"] = "provider"

    @property
    def total(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        return self.input_tokens + self.output_tokens

    @staticmethod
    def sum(items: list["Usage"]) -> "Usage":
        """Sum usages; any unavailable component makes that component unavailable."""
        if not items:
            return Usage(input_tokens=0, output_tokens=0, source="none")

        def add(field: str) -> int | None:
            vals = [getattr(u, field) for u in items]
            return None if any(v is None for v in vals) else sum(vals)

        sources = {u.source for u in items}
        return Usage(
            input_tokens=add("input_tokens"),
            output_tokens=add("output_tokens"),
            cached_input_tokens=add("cached_input_tokens"),
            reasoning_tokens=add("reasoning_tokens"),
            source=sources.pop() if len(sources) == 1 else "mixed",
        )


class EventKind(str, Enum):
    EPISODE_START = "episode_start"
    REQUEST = "request"  # full request as sent (messages, tools, sampling, seed)
    RESPONSE = "response"  # raw provider payload + parsed assistant message + usage
    PARSE_ERROR = "parse_error"
    REPAIR = "repair"  # any change between model output and what entered history
    TOOL_CALL = "tool_call"  # requested + actually executed arguments
    TOOL_RESULT = "tool_result"  # stdout/stderr/exit, raw output, truncated observation
    FINGERPRINT = "fingerprint"  # declared task-state fingerprint at a decision point
    ENV_PROBE = "env_probe"  # the environment probe before turn 0 (tasks/runtime/probe.py)
    REPLAY_ACTION = "replay_action"  # a fixed (non-model) action executed during replay
    REPLAY_CHECK = "replay_check"  # comparison of replayed obs/fingerprint to source
    INTERVENTION = "intervention"  # the fixed original/edited action at the branch point
    INFRA_ERROR = "infra_error"
    EPISODE_END = "episode_end"


class Event(Record):
    """One append-only line of `events.jsonl`."""

    seq: int
    kind: EventKind
    ts: str  # ISO-8601 UTC
    episode_id: str
    turn_index: int | None = None  # assistant-turn index (0-based) this belongs to
    data: dict[str, Any] = Field(default_factory=dict)


class ToolExecution(Record):
    call_id: str
    name: str
    requested_arguments_raw: str | None  # exactly as emitted by the model (or fixed action)
    executed_arguments: dict[str, Any] | None  # None if not executed
    executed: bool
    exit_code: int | None = None
    stdout: str | None = None
    stderr: str | None = None
    raw_output: str  # full tool output before truncation
    observation: str  # exactly what the learner was shown
    truncated: bool = False
    duration_sec: float | None = None
    cpu_time_sec: float | None = None  # scope: container cgroup delta, when measurable
    peak_memory_bytes: int | None = None
    error: str | None = None  # tool-level error text (also visible in observation)
    timeout_sec: int | None = None  # effective per-call timeout = min(requested, budget) (bash)
    timed_out: bool | None = None  # the command hit that timeout (bash; None: not applicable)


class TurnRecord(Record):
    """A derived view of one assistant turn, reconstructed from events.jsonl."""

    turn_index: int
    request_seq: int | None  # None for replayed/fixed turns (no model call)
    origin: Literal["model", "replayed", "intervention_original", "intervention_edited"]
    assistant_message: Message  # exactly as appended to the history
    finish_reason: str | None = None
    usage: Usage | None = None  # None for replayed/fixed turns
    malformed: bool = False  # unparseable/invalid tool call(s) in this turn
    repaired: bool = False  # the history message differs from the raw model output
    tool_executions: list[ToolExecution] = Field(default_factory=list)
    fingerprint_before: str | None = None
    latency_sec: float | None = None


# --------------------------------------------------------------------------- #
# Episodes
# --------------------------------------------------------------------------- #


class EpisodeRole(str, Enum):
    EVAL = "eval"  # unassisted policy evaluation
    COLLECT = "collect"  # training-panel experience collection
    BRANCH = "branch"  # editor-assisted continuation (never a policy evaluation)


class StopCategory(str, Enum):
    MODEL = "model"  # model ended the episode (no tool calls)
    BUDGET = "budget"  # experimental budget (turns, episode tokens, output length)
    SAFETY = "safety"  # wall-clock safety limit
    MODEL_ERROR = "model_error"  # endpoint rejected a request (e.g. context overflow)
    INFRA = "infra"  # environment/transport/endpoint failure; retried under policy
    REPLAY = "replay"  # replay mismatch before branching (fail closed)


class EpisodeBudgets(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_turns: int
    max_episode_tokens: int | None = None  # input+output across requests
    tool_timeout_sec: int
    max_output_chars: int
    agent_timeout_sec: float  # safety wall clock


class Timing(BaseModel):
    model_config = ConfigDict(extra="forbid")
    total_sec: float | None = None
    endpoint_sec: float | None = None  # summed request round-trip latency (not "generation time")
    tool_sec: float | None = None
    queue_sec: float | None = None  # only when actually observable


class EpisodeSummary(Record):
    episode_id: str
    role: EpisodeRole
    instance_id: str
    attempt_index: int | None = None  # eval/collect attempt; None for branches
    seed: int | None = None
    checkpoint_id: str
    stop_reason: str
    stop_category: StopCategory
    reward: dict[str, float] | None = None  # verifier output; None if not graded
    partial_reward: float | None = None  # reward["reward"]
    success: bool | None = None  # complete success (partial_reward >= task threshold)
    usage: Usage  # sum over all learner requests in this episode (provider-reported)
    n_requests: int
    n_tool_calls: int
    n_malformed_turns: int = 0
    timing: Timing = Field(default_factory=Timing)
    tool_cpu_sec: float | None = None
    tool_peak_memory_bytes: int | None = None
    infra_error: str | None = None
    trial_dir: str | None = None  # Harbor trial dir, if run through Harbor
    events_path: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Editing and verification
# --------------------------------------------------------------------------- #


class ProposedCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    arguments: dict[str, Any]


class EditProposal(Record):
    proposal_id: str
    source_episode_id: str
    instance_id: str
    editor_id: str  # editor identity (mode + checkpoint + prompt hash + decoding)
    status: Literal["proposed", "abstained", "invalid"]
    turn_index: int | None = None
    tool_call_id: str | None = None
    replacement: ProposedCall | None = None
    justification: str | None = None  # editor-only; never enters a student prompt
    rejection_reasons: list[str] = Field(default_factory=list)  # validation failures
    raw_response: dict[str, Any] | None = None
    usage: Usage | None = None
    duration_sec: float | None = None


class BranchCost(BaseModel):
    """Counterfactual episode token cost for one branch continuation.

    total = shared_prefix + intervention_request_input + intervention_tokens + continuation
    """

    model_config = ConfigDict(extra="forbid")
    shared_prefix_tokens: int | None  # input+output of source requests before the turn
    intervention_request_input_tokens: int | None  # input of the source request at the turn
    intervention_tokens: int | None  # learner-tokenizer length of the rendered fixed turn
    intervention_tokens_source: str  # e.g. "hf_template:<tokenizer sha>" or "fixture_estimate"
    continuation_tokens: int | None  # input+output of continuation requests (provider)

    @property
    def total(self) -> int | None:
        parts = [
            self.shared_prefix_tokens,
            self.intervention_request_input_tokens,
            self.intervention_tokens,
            self.continuation_tokens,
        ]
        return None if any(p is None for p in parts) else sum(parts)


class BranchResult(Record):
    branch: Literal["original", "edited"]
    repetition: int
    continuation_seed: int
    episode: EpisodeSummary
    replay_ok: bool
    replay_mismatches: list[str] = Field(default_factory=list)
    cost: BranchCost


class VerificationRecord(Record):
    verification_id: str
    proposal_id: str
    mode: Literal["continuation"]
    acceptance_rule: str
    branches: list[BranchResult] = Field(default_factory=list)
    accepted: bool
    reasons: list[str] = Field(default_factory=list)  # every failed criterion, or ["accepted"]
    mean_cost_original: float | None = None
    mean_cost_edited: float | None = None
    mean_saving: float | None = None
    operational_usage: Usage | None = None  # learner tokens actually spent verifying
    purpose: Literal["acceptance", "audit", "confirmation"] = "acceptance"
    # Strength of the evidence, e.g. "one_observed_successful_preference" (added by editing/verify.py)
    evidence_label: str | None = None


# --------------------------------------------------------------------------- #
# Preferences and checkpoints
# --------------------------------------------------------------------------- #


class PreferenceExample(Record):
    """Exactly what the trainer sees. Nothing else may be read into the prompt."""

    pair_id: str
    prompt: list[Message]  # messages before the intervened assistant turn
    chosen: list[Message]  # [edited assistant turn]
    rejected: list[Message]  # [original assistant turn]
    tools: list[ToolSchema]


class PreferenceProvenance(Record):
    """Companion record keyed by pair_id; never rendered into model inputs."""

    pair_id: str
    verification_mode: Literal["continuation"]
    instance_id: str
    family: str
    difficulty: str | None
    split: Split
    source_episode_id: str
    proposal_id: str
    verification_id: str
    learner_checkpoint_id: str  # learner that produced the source + continuations
    editor_id: str
    cycle: int
    run_id: str
    mean_saving: float | None
    kind: Literal["verified", "fixture"] = "verified"  # fixture = labeled smoke data


class CheckpointRecord(Record):
    checkpoint: CheckpointRef
    status: Literal["published"] = "published"
    created_at: str
    cycle: int | None
    run_id: str | None
    reference_checkpoint_id: str | None  # DPO reference used to train this one
    dataset_sha256: str | None
    n_train_examples: int | None
    optimizer_steps: int | None
    train_seed: int | None
    trainer: str  # "trl_dpo" | "fixture"
    tokenizer_sha256: str | None = None
    chat_template_sha256: str | None = None
    adapter_config: dict[str, Any] | None = None
    training_config: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    load_check: dict[str, Any] | None = None  # post-publication reload verification
