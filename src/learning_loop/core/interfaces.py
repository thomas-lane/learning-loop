"""The four small interfaces. Implementations live in their own modules.

Environment adapter  -> episodes/envs/ (Harbor Docker; local fixture)
Policy               -> episodes/policy.py (OpenAI-compatible endpoint; scripted fixture)
Editor / verifier    -> editing/editor.py, editing/verify.py
Trainer              -> training/ (TRL DPO; labeled fixture trainer)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .records import (
    CheckpointRecord,
    CheckpointRef,
    EditProposal,
    EpisodeBudgets,
    EpisodeRole,
    EpisodeSummary,
    Message,
    PolicySpec,
    RestoreCapability,
    TaskInstance,
    ToolExecution,
    ToolSchema,
    Usage,
    VerificationRecord,
)

# --------------------------------------------------------------------------- #
# Environment adapter
# --------------------------------------------------------------------------- #


class StateSpec(BaseModel):
    """Task-declared replay contract (task.toml `[metadata.learning_loop]`)."""

    model_config = ConfigDict(extra="forbid")
    restore: RestoreCapability = RestoreCapability.NONE
    fingerprint_paths: list[str] = Field(default_factory=list)  # absolute container paths
    fingerprint_exclude: list[str] = Field(default_factory=list)  # narrow globs, recorded
    observation_normalizers: list[dict[str, str]] = Field(default_factory=list)  # [{pattern, replacement, reason}]
    success_threshold: float = 1.0  # complete success: reward["reward"] >= threshold
    reward_key: str = "reward"
    local_equivalence: str | None = None  # named local-verification contract, if any
    caveats: list[str] = Field(default_factory=list)  # unmodeled aspects, e.g. "network: public (unmodeled)"


@dataclass
class EnvCapabilities:
    restore: RestoreCapability
    fingerprint: bool
    measures_cpu: bool = False
    measures_memory: bool = False
    workdir: str = "/app"


@runtime_checkable
class EnvironmentSession(Protocol):
    """A live task environment for one episode. Created fresh per episode/branch."""

    capabilities: EnvCapabilities

    async def execute(self, call_id: str, name: str, arguments: dict[str, Any], raw_arguments: str | None) -> ToolExecution:
        """Run one tool call; tool errors are returned inside the ToolExecution."""
        ...

    async def fingerprint(self, spec: StateSpec) -> str:
        """Hash of declared task state (content, perms, symlinks, cwd). Read-only."""
        ...


# --------------------------------------------------------------------------- #
# Episode plans (what an episode runner is asked to do)
# --------------------------------------------------------------------------- #


class ReplayTurn(BaseModel):
    """One historical assistant turn to re-execute without calling the model."""

    model_config = ConfigDict(extra="forbid")
    turn_index: int
    assistant_message: Message  # exact history message (tool_calls with JSON-string args)
    executed_arguments: list[dict[str, Any] | None]  # per tool call, as actually executed
    expected_observations: list[str]  # per tool call, exact truncated observation shown
    expected_fingerprint_before: str | None = None


class ReplaySpec(BaseModel):
    """Restore the decision state at `intervention_turn` and apply a fixed action."""

    model_config = ConfigDict(extra="forbid")
    source_episode_id: str
    history_prefix: list[Message]  # exact conversation before the intervention turn
    prefix_turns: list[ReplayTurn]  # turns 0..k-1 to replay in the fresh environment
    intervention_turn: int  # k
    intervention_label: str  # "original" | "edited"
    intervention_message: Message  # fixed assistant message (single tool call)
    expected_fingerprint_before_intervention: str | None = None
    prefix_usage: Usage  # source usage of requests 0..k-1 (budget accounting)
    intervention_request_input_tokens: int | None  # source request k input tokens
    # Source environment image identity (episode_start "image_identity"); None = not recorded.
    expected_image_identity: dict[str, Any] | None = None


class EpisodePlan(BaseModel):
    """Serializable instructions for one episode; passed to the Harbor agent as a file."""

    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    episode_id: str
    role: EpisodeRole
    instance_id: str
    attempt_index: int | None = None
    seed: int | None = None
    policy: PolicySpec
    budgets: EpisodeBudgets
    system_prompt: str  # fully rendered (workdir filled)
    instruction: str
    tools: list[ToolSchema]
    state_spec: StateSpec
    replay: ReplaySpec | None = None
    record_fingerprints: bool = True  # fingerprint before each model turn (for later replay)
    workdir: str = "/app"  # tool working directory (must match the system prompt)


@dataclass
class EpisodeResult:
    summary: EpisodeSummary
    out_dir: Path  # directory holding events.jsonl, messages.json, trajectory.json, episode.json
    replay_ok: bool | None = None  # None when not a replay episode
    replay_mismatches: list[str] = field(default_factory=list)


class EpisodeBackend(Protocol):
    """Runs a planned episode in a fresh environment and grades it separately."""

    name: str

    def restore_capability(self, instance: TaskInstance) -> RestoreCapability: ...

    async def run(self, instance: TaskInstance, plan: EpisodePlan, out_dir: Path) -> EpisodeResult: ...


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #


@dataclass
class PolicyDecision:
    """One model call. `history_message` is what enters the conversation."""

    raw_request: dict[str, Any]
    raw_response: dict[str, Any] | None
    history_message: Message
    finish_reason: str | None
    usage: Usage
    latency_sec: float | None
    parse_errors: list[str] = field(default_factory=list)  # per-call parse failures
    repairs: list[dict[str, Any]] = field(default_factory=list)  # every history edit, visible
    request_error: str | None = None  # endpoint rejected (e.g. 400 context overflow)
    infra_error: str | None = None  # transport/5xx; not the model's fault
    seed_sent: int | None = None
    call_errors: dict[str, str] = field(default_factory=dict)  # tool_call_id -> parse error (call not executed)


class Policy(Protocol):
    spec: PolicySpec

    async def decide(self, messages: list[Message], tools: list[ToolSchema], seed: int | None) -> PolicyDecision: ...


# --------------------------------------------------------------------------- #
# Editor / verifier
# --------------------------------------------------------------------------- #


class Editor(Protocol):
    editor_id: str

    async def propose(self, source: EpisodeSummary, turns: list[Any], instance: TaskInstance, instruction: str, tools: list[ToolSchema]) -> EditProposal: ...


class Verifier(Protocol):
    async def verify(self, proposal: EditProposal, purpose: str = "acceptance") -> VerificationRecord: ...


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #


class TrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    cycle: int
    dataset_dir: str  # immutable preference export (preferences.jsonl + manifest.json)
    incoming: CheckpointRef  # continued adapter AND frozen DPO reference
    model_profile: str
    training_config: dict[str, Any]  # TrainingConfig.model_dump()
    seed: int
    output_root: str  # checkpoints are published under output_root/<checkpoint_id>/
    device: str
    allow_cpu_fallback: bool = False


class Trainer(Protocol):
    name: str

    def train(self, request: TrainRequest, work_dir: Path) -> CheckpointRecord: ...
