"""Strict YAML schemas: experiments (science), machines (deployment), models (identity).

- Unknown keys are rejected everywhere (`extra="forbid"`).
- Minimal inheritance: an experiment may name one `base:` file; the child is a
  deep-merge override (dicts merge, lists/scalars replace). Nothing else.
- Credentials are never values: only env-var names (`*_env`) or SSH aliases.
- `resolve()` produces the fully resolved, secret-free config saved in each run.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .records import SamplingConfig

REPO_ROOT = Path(__file__).resolve().parents[3]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# Model profiles (configs/models/*.yaml)
# --------------------------------------------------------------------------- #


class ServingBackendProfile(Strict):
    """One way of serving this model. `adapter_formats` states what it accepts."""

    backend: Literal["hf_transformers", "llama_cpp", "vllm", "scripted"] = Field(description="Serving software; must equal the key it is listed under in `serving`.")
    artifact: str | None = Field(default=None, description="Served weights when they differ from `base_model` (e.g. a GGUF `repo:file`); null serves `base_model` itself.")
    artifact_revision: str | None = Field(default=None, description="Exact revision of `artifact`.")
    adapter_formats: list[Literal["peft_lora", "gguf_lora"]] = Field(default_factory=list, description="Adapter formats this backend can load. Learning runs require `peft_lora` (what the trainer produces).")
    quantization: str | None = Field(default=None, description="Quantization of the served artifact (e.g. `Q8_0`); recorded in run.json. A quantized artifact is not a trainable source.")
    status: Literal["tested", "untested", "unsupported"] = Field(default="untested", description="Whether this backend+model combination has actually been exercised in this repository. `validate` warns when it is not `tested`.")
    launch_args: list[str] = Field(default_factory=list, description="Extra engine flags for managed launches (vLLM server flags).")
    engine_package: str | None = Field(default=None, description="`vllm` only: pip requirement installed into a separate environment on the serving host (`.engines/`), e.g. `vllm==0.30.0`.")
    notes: str | None = Field(default=None, description="Free text: what was verified and what was not.")


class ModelProfile(Strict):
    """Exact identity of a model and how it is rendered, served and trained."""

    schema_version: int = Field(default=1, description="Schema version of this file.")
    name: str = Field(description="Profile name; must match the file name and is referenced by experiments.")
    base_model: str = Field(description="Hugging Face repo id of the trainable source checkpoint.")
    base_revision: str = Field(description="Exact 40-hex commit of `base_model` (never a branch name).")
    tokenizer: str | None = Field(default=None, description="Tokenizer repo if different from `base_model` (loaded at `base_revision`).")
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict, description="Arguments passed to the chat template everywhere it is rendered (serving, training, token counts), e.g. `enable_thinking: false`.")
    chat_template_sha256: str | None = Field(default=None, description="SHA-256 of the tokenizer's chat template; loading fails if the cached template differs.")
    tool_call_format: Literal["openai_json", "qwen3_xml", "gemma4"] = Field(default="openai_json", description="How the model writes tool calls; the reference HF server parses `qwen3_xml` and `gemma4`.")
    serving: dict[str, ServingBackendProfile] = Field(description="Serving backends keyed by backend name.")
    training_dtype: Literal["float32", "bfloat16", "float16"] = Field(default="float32", description="Weight dtype for training and reference log-probs (logits are always computed in float32).")
    lora_target_modules: list[str] = Field(default_factory=list, description="Default LoRA target module names (overridable by `training.lora.target_modules`).")
    lora_exclude_modules: str | None = Field(default=None, description="Regex (PEFT `exclude_modules`, full match on module paths) for modules never adapted even when their names match, e.g. a multimodal checkpoint's vision/audio towers.")
    supported_train_devices: list[Literal["mps", "cuda", "cpu"]] = Field(default_factory=list, description="Devices training may resolve to; the trainer refuses others (empty = not declared).")
    seed_supported: bool | None = Field(default=None, description="Declared: does the serving path honor request seeds? Recorded; never proves bitwise reproducibility.")
    notes: str | None = Field(default=None, description="Free text: verification status and caveats.")


# --------------------------------------------------------------------------- #
# Machine profiles (configs/machines/{examples,local}/*.yaml)
# --------------------------------------------------------------------------- #


class HostRef(Strict):
    """`local`, an SSH alias from the user's ~/.ssh/config, or a Runpod pod managed by the command."""

    kind: Literal["local", "ssh", "runpod"] = Field(default="local", description="`local`, `ssh` (a fixed host reached by an alias) or `runpod` (a pod the command starts or creates, reaches by its current address, and stops or terminates; see the machine profile's `runpod` block).")
    ssh_alias: str | None = Field(default=None, description="`ssh` only: host alias from ~/.ssh/config (keys, ports and jump hosts stay there).")
    pod_id: str | None = Field(default=None, description="`runpod` only: id of an existing pod, started and stopped by each command (never created or terminated). Exclusive with `pod`.")
    pod: str | None = Field(default=None, description="`runpod` only: name of a spec in `runpod.create`; each command creates a pod from it and terminates it at the end. Exclusive with `pod_id`.")
    workdir: str | None = Field(default=None, description="Absolute path of the repository checkout on that host. Required for `ssh` and `runpod`.")

    @property
    def pod_ref(self) -> str | None:
        """Key of the pod within one command: the pod id, or `new-<spec>` for a created pod."""
        return self.pod_id or (f"new-{self.pod}" if self.pod else None)

    @model_validator(mode="after")
    def _check(self) -> "HostRef":
        if self.kind == "ssh" and not (self.ssh_alias and self.workdir):
            raise ValueError("ssh hosts need ssh_alias and workdir")
        if self.kind == "runpod" and not (bool(self.pod_id) != bool(self.pod) and self.workdir):
            raise ValueError("runpod hosts need workdir and exactly one of pod_id (an existing pod) or pod (a spec in runpod.create)")
        if self.kind != "ssh" and self.ssh_alias:
            raise ValueError("ssh_alias is only valid with kind: ssh")
        if self.kind != "runpod" and (self.pod_id or self.pod):
            raise ValueError("pod_id and pod are only valid with kind: runpod")
        if self.ssh_alias and not re.fullmatch(r"[A-Za-z0-9_.@-]+", self.ssh_alias):
            raise ValueError(f"invalid ssh alias {self.ssh_alias!r}")
        if self.pod_id and not re.fullmatch(r"[a-z0-9]+", self.pod_id):
            raise ValueError(f"invalid pod id {self.pod_id!r}")
        if self.pod and not re.fullmatch(r"[a-z0-9][a-z0-9-]*", self.pod):
            raise ValueError(f"invalid pod spec name {self.pod!r}")
        return self


class PodSpec(Strict):
    """A pod each command creates and terminates (`runpod.create.<name>`)."""

    gpu_types: list[str] = Field(min_length=1, description="Acceptable Runpod GPU type ids in order of preference, e.g. `NVIDIA A100 80GB PCIe`, `NVIDIA A100-SXM4-80GB`.")
    max_cost_per_hr: float = Field(gt=0, description="Upper limit on the pod's hourly price (USD); a created pod that costs more is terminated at once and the command fails.")
    cloud_type: Literal["SECURE", "COMMUNITY"] = Field(default="SECURE", description="Runpod cloud to create the pod in.")
    image: str = Field(default="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404", description="Container image; it must run sshd and install `PUBLIC_KEY` (Runpod's PyTorch images do). The project environment brings its own torch.")
    container_disk_gb: int = Field(default=200, ge=20, description="Ephemeral disk for the environment, model downloads and scratch files (discarded on termination).")
    allowed_cuda_versions: list[str] = Field(default_factory=lambda: ["13.0"], description="Host CUDA (driver) versions to accept; the locked torch wheels need CUDA 13.")


class RunpodConfig(Strict):
    """Runpod pods used by `kind: runpod` hosts: existing pods (start/stop) or pods created per command."""

    api_key_env: str = Field(default="RUNPOD_API_KEY", description="Environment variable holding the Runpod API key on the coordinator (e.g. from .env). Never sent to the pod.")
    api_base: str = Field(default="https://rest.runpod.io/v1", description="Runpod REST API base URL.")
    identity_file: str = Field(default="~/.ssh/id_ed25519", description="Private key for SSH to the pods (its public key must be in your Runpod account settings).")
    ssh_user: str = Field(default="root", description="SSH user on the pods.")
    start_timeout_sec: int = Field(default=900, ge=60, description="How long to wait for a started or created pod to report its address and accept SSH (a stopped pod may wait for its GPU to become free; creation is retried while no GPU of the listed types is available).")
    stop_when_done: bool = Field(default=True, description="Existing pods (`pod_id`): stop every pod this command used when it ends (success, failure or Ctrl-C). Created pods are always terminated.")
    idle_stop_minutes: int = Field(default=30, ge=5, description="Pod-side watchdog: the pod stops itself (a created pod terminates itself) when the coordinator's heartbeat is older than this (covers a crashed or sleeping laptop).")
    heartbeat_sec: int = Field(default=60, ge=10, description="How often the coordinator refreshes the heartbeat on each pod.")
    create: dict[str, PodSpec] = Field(default_factory=dict, description="Pod specs referenced by hosts' `pod:`; each command creates one pod per spec and terminates it however the command ends.")


class InferenceProfile(Strict):
    """How learner/editor requests are served.

    managed: the coordinator starts/stops a server process it owns, per checkpoint.
    external: an already-running endpoint; its served identity must be declared and
              the run refuses to use it for any other checkpoint.
    scripted: deterministic fixture policy (tests/smoke only).
    """

    mode: Literal["managed", "external", "scripted"] = Field(description="`managed` (the run starts/stops its own server per checkpoint), `external` (an existing endpoint serving one declared checkpoint) or `scripted` (fixture policy).")
    backend: Literal["hf_transformers", "llama_cpp", "vllm", "scripted"] = Field(description="Serving backend; must be declared in the model profile's `serving` for managed LoRA runs.")
    host: HostRef = Field(default_factory=HostRef, description="Where a managed server runs.")
    api_base: str | None = Field(default=None, description="OpenAI-compatible base URL. Required for `external`; for a remote managed server, a local tunnel URL. Default for managed: `http://127.0.0.1:<port>/v1`.")
    api_key_env: str | None = Field(default=None, description="Name of the environment variable holding the API key (never the key itself).")
    served_checkpoint_id: str | None = Field(default=None, description="External mode: the checkpoint identity you assert the endpoint serves (e.g. `base:<profile>@<rev12>`); runs needing any other checkpoint are refused.")
    served_model_name: str | None = Field(default=None, description="External mode: the `model` id to request (e.g. a GGUF name); default: the endpoint's only listed model.")
    port: int | None = Field(default=None, description="Managed mode: server port (required).")
    device: Literal["mps", "cuda", "cpu", "auto"] = Field(default="auto", description="Managed mode: accelerator for the server.")
    startup_timeout_sec: int = Field(default=600, description="Managed mode: how long to wait for a server to load and list the checkpoint.")
    request_timeout_sec: int = Field(default=600, description="Per-request client timeout; a timeout is an infrastructure failure.")
    request_concurrency: int = Field(default=1, description="Maximum concurrent requests to this endpoint (enforced per endpoint), independent of Docker concurrency.")

    @model_validator(mode="after")
    def _check(self) -> "InferenceProfile":
        if self.mode == "external" and not (self.api_base and self.served_checkpoint_id):
            raise ValueError("external inference needs api_base and served_checkpoint_id")
        if self.mode == "managed" and self.port is None:
            raise ValueError("managed inference needs a port")
        if self.mode == "scripted" and self.backend != "scripted":
            raise ValueError("scripted mode requires backend: scripted")
        return self


class TrainingHostProfile(Strict):
    """Where and on what device the trainer process runs."""

    host: HostRef = Field(default_factory=HostRef, description="`local` (subprocess) or an SSH host (inputs pushed, trainer started detached, checkpoint pulled back).")
    device: Literal["mps", "cuda", "cpu", "auto"] = Field(default="auto", description="`auto` picks CUDA, then MPS, among the model profile's supported devices.")
    allow_cpu_fallback: bool = Field(default=False, description="Allow CPU training when no accelerator is available (recorded in checkpoint metadata).")


class MachineProfile(Strict):
    """Deployment: where the coordinator, environments, inference and training run."""

    schema_version: int = Field(default=1, description="Schema version of this file.")
    name: str = Field(description="Profile name (recorded in run.json).")
    coordinator: HostRef = Field(default_factory=HostRef, description="Where `loop run` executes; an SSH host is used by `loop submit/fetch/remote-status`.")
    environment_backend: Literal["harbor_docker", "local_fixture"] = Field(default="harbor_docker", description="`harbor_docker` (Harbor trials in Docker) or `local_fixture` (host subprocesses; scripted policies and fixture tasks only; not a sandbox).")
    docker_concurrency: int = Field(default=1, description="Maximum concurrent episodes (containers).")
    inference: InferenceProfile = Field(description="Learner inference (and the editor's, unless `editor_inference` is set).")
    editor_inference: InferenceProfile | None = Field(default=None, description="Separate endpoint for the editor; when unset the editor shares the learner's serving slot (servers are swapped sequentially).")
    training: TrainingHostProfile = Field(default_factory=TrainingHostProfile, description="Training host and device.")
    runpod: RunpodConfig | None = Field(default=None, description="Pod lifecycle settings; required when any host has `kind: runpod`.")
    runs_dir: str = Field(default="runs", description="Directory for run directories (relative to the repository root).")
    hardware_notes: str | None = Field(default=None, description="Free text recorded with the run.")

    def hosts(self) -> list[tuple[str, HostRef]]:
        """(role, host) for every role that has a host."""
        out = [("coordinator", self.coordinator), ("inference", self.inference.host), ("training", self.training.host)]
        if self.editor_inference is not None:
            out.append(("editor_inference", self.editor_inference.host))
        return out

    def pod_ids(self) -> list[str]:
        """Pod refs used by this profile: existing pod ids and `new-<spec>` for created pods."""
        return sorted({h.pod_ref for _, h in self.hosts() if h.kind == "runpod" and h.pod_ref})

    def existing_pod_ids(self) -> list[str]:
        return sorted({h.pod_id for _, h in self.hosts() if h.kind == "runpod" and h.pod_id})

    @model_validator(mode="after")
    def _runpod(self) -> "MachineProfile":
        if self.pod_ids() and self.runpod is None:
            raise ValueError("hosts with kind: runpod need a `runpod:` block (use `runpod: {}` for the defaults)")
        for _, h in self.hosts():
            if h.pod and h.pod not in (self.runpod.create if self.runpod else {}):
                raise ValueError(f"host pod {h.pod!r} is not defined in runpod.create")
        if self.coordinator.kind == "runpod":
            raise ValueError("the coordinator cannot be a Runpod pod (task containers need Docker, which pods lack)")
        return self


# --------------------------------------------------------------------------- #
# Experiment definitions (experiments/*.yaml)
# --------------------------------------------------------------------------- #


class LearnerConfig(Strict):
    """The learner model and where it starts."""

    model_profile: str = Field(description="Model profile name (configs/models/<name>.yaml).")
    initial_checkpoint: str = Field(default="base", description="`base` or a published checkpoint directory to start from.")
    scripted_policy: str | None = Field(default=None, description="FIXTURE only: scripted learner file; requires `inference.mode: scripted` (and vice versa).")


class SeedConfig(Strict):
    """Root seeds. Evaluation seeds depend only on `root`; all loop streams also on `loop_seed`."""

    root: int = Field(description="Root seed. Evaluation seeds depend only on this, instance and attempt, so runs compare pairwise.")
    loop_seed: int = Field(default=0, description="Independent learning-loop seed (one run = one loop seed); feeds collection, editor, continuation, data-selection and training streams.")


class ExposureStage(Strict):
    """One step of a family-exposure schedule."""

    from_cycle: int = Field(description="First cycle this step applies to.")
    families: list[str] = Field(description="Families collected from this cycle on (must appear in the collection panel).")


class TaskSelection(Strict):
    """Which instances supply training experience."""

    splits: str = Field(description="Split file (evaluation/splits/*.yaml) defining instances and panels.")
    collection_panel: str = Field(description="Panel collected from; must be a `train`-split panel.")
    attempts_per_instance: int = Field(ge=1, description="Independently seeded collection attempts per instance per cycle.")
    exposure_schedule: list[ExposureStage] | None = Field(default=None, description="Optional family schedule; the latest step whose `from_cycle` <= cycle applies.")


class EvaluationConfig(Strict):
    """Unassisted evaluation of each cycle's learner."""

    dev_panels: list[str] = Field(description="Panels evaluated during the loop; must be `dev`-split panels.")
    attempts_per_instance: int = Field(ge=1, description="Attempts per instance (same declared seeds for every checkpoint).")
    cadence: Literal["every_cycle", "first_and_last"] = Field(default="every_cycle", description="Evaluate every cycle, or only the initial and final checkpoints.")
    final_panels: list[str] = Field(default_factory=list, description="Final-test panels; validated but only run by `loop evaluate --final`, never by the loop.")


class EpisodeConfig(Strict):
    """Learner episode budgets and prompts."""

    sampling: SamplingConfig = Field(description="Learner decoding for evaluation and collection.")
    max_turns: int = Field(ge=1, description="Experimental budget: model turns per episode (a branch's prefix and intervention count).")
    max_episode_tokens: int | None = Field(default=None, description="Experimental budget: input+output tokens over all requests; an endpoint without usage then stops the episode (`budget:usage_unavailable`).")
    tool_timeout_sec: int = Field(default=60, description="Per-command timeout; the model cannot raise it.")
    agent_timeout_sec: float = Field(default=600.0, description="Safety wall-clock limit per episode (a safety stop, not a budget).")
    max_output_chars: int = Field(default=8000, description="Tool output shown to the learner is truncated head+tail to this many characters (the full output is recorded).")
    system_prompt: str = Field(default="evaluation/agents/system_prompt.md", description="System prompt file; `{workdir}` is filled in.")


class EditorConfig(Strict):
    """The retrospective editor."""

    mode: Literal["initial_policy", "current_learner", "external", "scripted"] = Field(description="`initial_policy` (fixed initial checkpoint; default condition), `current_learner` (changes every cycle), `external` (a fixed separate model) or `scripted` (fixture).")
    model_profile: str | None = Field(default=None, description="`external` only: the editor's model profile.")
    checkpoint: str | None = Field(default=None, description="`external` only: `base` or a checkpoint directory (default `base`).")
    prompt: str = Field(default="prompts/editor/v2.md", description="Editor prompt; its SHA-256 is part of the editor identity.")
    sampling: SamplingConfig = Field(default_factory=lambda: SamplingConfig(temperature=0.0, max_output_tokens=2048), description="Editor decoding.")
    proposals_per_source: int = Field(default=1, ge=1, description="Proposals per successful source; only 1 is supported (more would need fresh-seed confirmation).")
    include_outcome_metrics: bool = Field(default=True, description="Show the editor scalar outcomes of the source episode (success, token totals).")
    include_later_observations: bool = Field(default=True, description="Show the editor observations after each turn (hindsight is still filtered at validation).")
    assistant_text_policy: Literal["reject_nonempty"] = Field(default="reject_nonempty", description="Strict mode: turns with assistant text or reasoning are not editable.")
    scripted_path: str | None = Field(default=None, description="`scripted` only: fixture proposals file.")

    @model_validator(mode="after")
    def _check(self) -> "EditorConfig":
        if self.mode == "external" and not self.model_profile:
            raise ValueError("editor.mode=external requires editor.model_profile")
        if self.mode != "external" and (self.model_profile or self.checkpoint):
            raise ValueError("editor.model_profile/checkpoint are only valid with mode=external")
        if self.mode == "scripted" and not self.scripted_path:
            raise ValueError("editor.mode=scripted requires scripted_path")
        return self


class AuditConfig(Strict):
    """Optional re-verification of accepted edits (reported only; datasets never change)."""

    fraction: float = Field(default=0.0, ge=0.0, le=1.0, description="Seeded random fraction of accepted edits to re-verify with fresh seeds.")
    continuations_per_branch: int = Field(default=2, ge=1, description="Continuations per branch in an audit.")


class VerificationConfig(Strict):
    """Branch comparison and acceptance."""

    mode: Literal["continuation", "local"] = Field(default="continuation", description="`continuation` (fresh learner continuations from both branches). `local` is refused at validation: no backend supports it yet.")
    continuations_per_branch: int = Field(default=1, ge=1, description="Matched continuations per branch; all must succeed on both branches.")
    acceptance_rule: Literal["strict_all_success_v1"] = Field(default="strict_all_success_v1", description="Named acceptance rule (see docs/experiment.md).")
    min_token_saving: float = Field(default=1.0, gt=0, description="Minimum mean counterfactual token saving (original - edited) to accept; ties never pass.")
    min_relative_saving: float = Field(default=0.0, ge=0.0, lt=1.0, description="Minimum saving as a fraction of the original branch cost.")
    sampling: SamplingConfig | None = Field(default=None, description="Continuation decoding; default `episode.sampling`.")
    audit: AuditConfig = Field(default_factory=AuditConfig, description="Optional audits.")


class DataSelectionConfig(Strict):
    """Which accepted pairs a cycle trains on."""

    selection: Literal["current_only", "current_and_history"] = Field(default="current_and_history", description="Train on this cycle's pairs only, or add a sample of earlier cycles' pairs. History alone never triggers training.")
    buffer_capacity: int = Field(default=256, ge=1, description="Maximum historical pairs kept in the pool (seeded, task-balanced).")
    history_fraction: float = Field(default=0.5, ge=0.0, le=1.0, description="Target share of historical pairs in the training set.")
    task_balanced: bool = Field(default=True, description="Sample history round-robin over instances.")
    max_examples_per_cycle: int | None = Field(default=None, description="Cap on the training set size per cycle.")


class DpoConfig(Strict):
    """DPO hyperparameters (TRL). Pilot values are initial choices, not tuned."""

    beta: float = Field(default=0.1, description="DPO beta.")
    loss_type: str = Field(default="sigmoid", description="TRL DPO loss type.")
    learning_rate: float = Field(default=5e-6, description="Learning rate.")
    per_device_batch_size: int = Field(default=1, description="Pairs per device step.")
    gradient_accumulation_steps: int = Field(default=1, description="Micro-batches per optimizer step.")
    max_length: int = Field(default=4096, description="Prompt+completion token limit; longer examples are dropped with a recorded reason, never truncated.")
    warmup_steps: int = Field(default=0, description="Learning-rate warmup steps.")
    max_grad_norm: float = Field(default=1.0, description="Gradient clipping norm.")
    gradient_checkpointing: bool = Field(default=False, description="Recompute activations in the backward pass: much less memory for long prompts, roughly a third slower; same values.")


class LoraConfig(Strict):
    """LoRA shape. Continuing an adapter requires the same r, alpha and target modules."""

    r: int = Field(default=16, description="LoRA rank.")
    alpha: int = Field(default=32, description="LoRA alpha.")
    dropout: float = Field(default=0.05, description="LoRA dropout; TRL disables dropout, so the effective value (recorded) is 0.")
    target_modules: list[str] | None = Field(default=None, description="Module names; default: the model profile's `lora_target_modules`.")


class TrainingConfig(Strict):
    """How each cycle's checkpoint is trained."""

    trainer: Literal["trl_dpo", "fixture"] = Field(default="trl_dpo", description="`trl_dpo` (TRL + PEFT) or `fixture` (labeled pseudo-adapter for orchestration tests).")
    data: DataSelectionConfig = Field(default_factory=DataSelectionConfig, description="Training-set selection.")
    fixed_dataset: str | None = Field(default=None, description="`fixed_dataset` condition only: frozen preference export directory used every cycle.")
    dpo: DpoConfig = Field(default_factory=DpoConfig, description="DPO hyperparameters.")
    lora: LoraConfig = Field(default_factory=LoraConfig, description="LoRA shape.")
    optimizer_steps: int = Field(ge=1, description="Exact optimizer steps per training cycle (the effort matched across conditions).")
    reference: Literal["incoming_checkpoint"] = Field(default="incoming_checkpoint", description="DPO reference: the learner frozen at the start of the cycle, including its adapter.")
    adapter_init: Literal["continue_incoming"] = Field(default="continue_incoming", description="Continue the incoming adapter's weights (a new LoRA only from the base model).")
    optimizer_state: Literal["reset_each_cycle"] = Field(default="reset_each_cycle", description="Fresh optimizer/scheduler each cycle (restored only when resuming an interrupted stage).")


class RuntimeConfig(Strict):
    """Execution policy that affects results bookkeeping."""

    infra_retries: int = Field(default=2, ge=0, description="Retries of an infrastructure failure (fresh environment each time; failed attempts are kept and counted).")
    docker_concurrency: int | None = Field(default=None, description="Lower the machine profile's Docker concurrency for this experiment.")


class ExperimentConfig(Strict):
    """Scientific definition of one run."""

    schema_version: int = Field(default=1, description="Schema version of this file.")
    name: str = Field(description="Lowercase name `[a-z0-9._-]`; prefix of run ids.")
    description: str = Field(default="", description="Free text.")
    condition: Literal["learning", "frozen_baseline", "fixed_dataset"] = Field(default="learning", description="`learning` (the full loop), `frozen_baseline` (evaluate the initial checkpoint; cycles must be 0) or `fixed_dataset` (train on one frozen export; no collection/editing).")
    learner: LearnerConfig = Field(description="Learner model and starting checkpoint.")
    cycles: int = Field(ge=0, description="Training cycles; the run evaluates cycles+1 checkpoints (the last cycle only evaluates).")
    seeds: SeedConfig = Field(description="Root and loop seeds.")
    tasks: TaskSelection = Field(description="Split file and collection panel.")
    evaluation: EvaluationConfig = Field(description="Development and final panels.")
    episode: EpisodeConfig = Field(description="Learner budgets, prompts and decoding.")
    editor: EditorConfig = Field(description="Editor condition.")
    verification: VerificationConfig = Field(default_factory=VerificationConfig, description="Branch verification and acceptance.")
    training: TrainingConfig = Field(description="Trainer, data selection and hyperparameters.")
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig, description="Retries and concurrency limits.")
    labels: dict[str, str] = Field(default_factory=dict, description="Free-form string labels recorded with the run (e.g. `hyperparameters: initial choices`). `allow_fixture_data: 'true'` permits training a real learner on a fixture export.")

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", v):
            raise ValueError("experiment name must be lowercase [a-z0-9._-]")
        return v

    @model_validator(mode="after")
    def _conditions(self) -> "ExperimentConfig":
        if self.condition == "fixed_dataset" and not self.training.fixed_dataset:
            raise ValueError("condition=fixed_dataset requires training.fixed_dataset")
        if self.condition != "fixed_dataset" and self.training.fixed_dataset:
            raise ValueError("training.fixed_dataset is only valid with condition=fixed_dataset")
        if self.verification.mode == "local" and self.verification.continuations_per_branch != 1:
            raise ValueError("local verification has no continuations; leave continuations_per_branch=1")
        if self.editor.proposals_per_source > 1:
            # Several candidates per source would need fresh-seed confirmation and selection of one
            # pair per source before acceptance; that path is not implemented, so it is refused.
            raise ValueError("proposals_per_source > 1 is not supported yet (needs confirmation with fresh seeds)")
        return self


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return data


_SECRET_KEY = re.compile(r"(api_key|token|secret|password)$", re.I)


def assert_no_secrets(obj: Any, where: str = "config") -> None:
    """Credential-looking keys must be env-var references (`*_env`), never values."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and _SECRET_KEY.search(k) and v not in (None, ""):
                raise ValueError(f"{where}: key {k!r} looks like a literal credential; use {k}_env")
            assert_no_secrets(v, f"{where}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            assert_no_secrets(v, f"{where}[{i}]")


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """`--set a.b.c=value` (value parsed as YAML). Recorded in the resolved config."""
    out = copy.deepcopy(raw)
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override {ov!r} must look like key.path=value")
        key, val = ov.split("=", 1)
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"override {key!r}: {part!r} is not a mapping")
        node[parts[-1]] = yaml.safe_load(val)
    return out


def load_experiment(path: str | Path, overrides: list[str] | None = None) -> tuple[ExperimentConfig, dict[str, Any]]:
    """Returns (validated config, merged raw dict). One level of `base:` allowed."""
    path = Path(path)
    raw = _load_yaml(path)
    if "base" in raw:
        base_path = (path.parent / raw.pop("base")).resolve()
        base_raw = _load_yaml(base_path)
        if "base" in base_raw:
            raise ValueError(f"{base_path}: nested `base:` is not supported (one level only)")
        raw = _deep_merge(base_raw, raw)
    if overrides:
        raw = apply_overrides(raw, overrides)
    assert_no_secrets(raw, str(path))
    return ExperimentConfig.model_validate(raw), raw


def load_machine(path: str | Path) -> MachineProfile:
    raw = _load_yaml(Path(path))
    assert_no_secrets(raw, str(path))
    return MachineProfile.model_validate(raw)


def model_profile_path(name: str) -> Path:
    return REPO_ROOT / "configs" / "models" / f"{name}.yaml"


def load_model_profile(name_or_path: str) -> ModelProfile:
    p = Path(name_or_path)
    if p.suffix not in (".yaml", ".yml"):  # bare profile name (names may contain dots, e.g. qwen3-0.6b)
        p = model_profile_path(name_or_path)
    raw = _load_yaml(p)
    assert_no_secrets(raw, str(p))
    prof = ModelProfile.model_validate(raw)
    if not re.fullmatch(r"[0-9a-f]{40}", prof.base_revision):
        raise ValueError(f"{p}: base_revision must be an exact 40-hex commit sha")
    return prof


def repo_path(p: str | Path) -> Path:
    """Resolve a config-relative path against the repository root."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p
