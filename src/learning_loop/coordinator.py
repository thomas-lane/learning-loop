"""Cycle orchestration: collect -> edit -> verify -> dataset -> train -> evaluate.

The coordinator owns run directories, stage manifests, work-item identity,
infrastructure retries, checkpoint lineage and serving swaps. Stage logic lives
in the component modules (backends, editor, verify, preferences, training).

Every stage is resumable: completed work items (stable IDs) are skipped,
interrupted ones keep their artifacts (`*.interrupted-N`) and are re-executed
in a fresh environment. A run directory is locked while a coordinator is
mutating it.
"""

from __future__ import annotations

import asyncio
import json
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from . import provenance, seeds
from .config import (
    ExperimentConfig,
    MachineProfile,
    ModelProfile,
    load_experiment,
    load_machine,
    load_model_profile,
    repo_path,
)
from .inference import InferenceManager
from .interfaces import EpisodeBackend, EpisodePlan, TrainRequest
from .records import (
    CheckpointRecord,
    CheckpointRef,
    EditProposal,
    EpisodeBudgets,
    EpisodeRole,
    EpisodeSummary,
    SamplingConfig,
    Split,
    StopCategory,
    TaskInstance,
    VerificationRecord,
)
from .remote import Remote
from .storage import (
    StageManifest,
    WorkItem,
    JsonlAppender,
    atomic_write_json,
    now_iso,
    preserve_interrupted,
    read_json,
    run_lock,
    write_once_json,
)

StageFn = Callable[[WorkItem, Path], Awaitable[dict[str, Any]]]


class PlanError(ValueError):
    """Configuration is valid YAML but cannot be executed as declared."""


# --------------------------------------------------------------------------- #
# Run context
# --------------------------------------------------------------------------- #


@dataclass
class RunContext:
    run_dir: Path
    run_id: str
    exp: ExperimentConfig
    raw: dict[str, Any]
    machine: MachineProfile
    learner_profile: ModelProfile
    editor_profile: ModelProfile
    instances: dict[str, TaskInstance]
    panels: dict[str, list[str]]  # panel -> instance ids
    panel_split: dict[str, Split]
    instance_split: dict[str, Split]
    held_out_families: list[str] = field(default_factory=list)
    managers: dict[str, InferenceManager] = field(default_factory=dict)
    _backend: EpisodeBackend | None = None

    # -- paths -------------------------------------------------------------- #

    def cycle_dir(self, cycle: int) -> Path:
        return self.run_dir / "cycles" / f"cycle-{cycle:03d}"

    def stage_dir(self, cycle: int, stage: str) -> Path:
        return self.cycle_dir(cycle) / stage

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    # -- components --------------------------------------------------------- #

    @property
    def backend(self) -> EpisodeBackend:
        if self._backend is None:
            self._backend = make_backend(self)
        return self._backend

    def manager(self, role: str) -> InferenceManager:
        if role not in self.managers:
            prof = self.machine.inference if role == "learner" else (self.machine.editor_inference or self.machine.inference)
            model = self.learner_profile if role == "learner" else self.editor_profile
            self.managers[role] = InferenceManager(prof, model, self.run_dir / "logs", role=role)
        return self.managers[role]

    def pods(self, log: Callable[[str], None] | None = None):
        """Start/prepare/watch the Runpod pods this run uses, and stop them when it ends
        (a no-op without `kind: runpod` hosts)."""
        from .pods import PodLifecycle

        return PodLifecycle(self.machine, self.run_dir / "logs", log=log)

    def shared_manager_for_editor(self) -> bool:
        """Editor and learner share one server slot when no separate editor inference is declared."""
        return self.machine.editor_inference is None

    def stop_serving(self) -> None:
        for m in self.managers.values():
            m.stop()
        # append-only across coordinator sessions (resumes never erase earlier lifecycle events)
        log = JsonlAppender(self.run_dir / "logs" / "serving-lifecycle.jsonl")
        for role, m in self.managers.items():
            for ev in m.events:
                log.append({"role": role, **ev})
            m.events.clear()


def make_backend(ctx: RunContext) -> EpisodeBackend:
    from .backends import HarborDockerBackend, LocalFixtureBackend

    if ctx.machine.environment_backend == "local_fixture":
        return LocalFixtureBackend()
    conc = min(ctx.machine.docker_concurrency, ctx.exp.runtime.docker_concurrency or ctx.machine.docker_concurrency)
    return HarborDockerBackend(concurrency=conc)


# --------------------------------------------------------------------------- #
# Validation / planning
# --------------------------------------------------------------------------- #


def load_all(exp_path: str | Path, machine_path: str | Path, overrides: list[str] | None = None) -> tuple[ExperimentConfig, dict[str, Any], MachineProfile, ModelProfile, ModelProfile]:
    exp, raw = load_experiment(exp_path, overrides)
    machine = load_machine(machine_path)
    learner = load_model_profile(exp.learner.model_profile)
    editor = load_model_profile(exp.editor.model_profile) if exp.editor.mode == "external" else learner
    return exp, raw, machine, learner, editor


def check_compatibility(exp: ExperimentConfig, machine: MachineProfile, learner: ModelProfile, editor: ModelProfile) -> list[str]:
    """Raise PlanError for incompatible settings; return non-fatal notes."""
    notes: list[str] = []
    trains = exp.condition in ("learning", "fixed_dataset") and exp.cycles >= 1
    inf = machine.inference
    if (inf.mode == "scripted") != bool(exp.learner.scripted_policy):
        raise PlanError("learner.scripted_policy and scripted inference go together (fixtures only)")
    if exp.condition == "fixed_dataset":
        fd = repo_path(exp.training.fixed_dataset)  # type: ignore[arg-type]
        if not (fd / "preferences.jsonl").exists():
            raise PlanError(f"training.fixed_dataset {fd} has no preferences.jsonl (point it at a frozen export)")
        man = read_json(fd / "manifest.json") if (fd / "manifest.json").exists() else {}
        is_fixture = "fixture" in (man.get("kinds") or []) or man.get("kind") == "fixture"
        if is_fixture and inf.mode != "scripted" and exp.labels.get("allow_fixture_data") != "true":
            raise PlanError(f"{fd} holds FIXTURE preferences; set labels.allow_fixture_data: 'true' to train a real learner on them deliberately")
        if is_fixture:
            notes.append(f"fixed dataset {fd} is labeled fixture data")
    if exp.condition == "frozen_baseline" and exp.cycles != 0:
        raise PlanError("frozen_baseline evaluates the initial checkpoint only; set cycles: 0")
    if exp.verification.mode == "local":
        # verify.LocalVerifier needs an environment session factory; no backend in this build
        # exposes one, so the mode would only fail after collection and editing were paid for.
        raise PlanError("verification.mode=local is not supported by the available environment backends")
    initial_id = (
        base_checkpoint(learner).checkpoint_id
        if exp.learner.initial_checkpoint == "base"
        else CheckpointRecord.model_validate(read_json(Path(exp.learner.initial_checkpoint) / "checkpoint.json")).checkpoint.checkpoint_id
    )
    if inf.mode == "external" and inf.served_checkpoint_id != initial_id:
        raise PlanError(f"external endpoint declares served_checkpoint_id={inf.served_checkpoint_id!r}; this run needs {initial_id!r}")
    ed_inf = machine.editor_inference
    uses_model_editor = exp.condition == "learning" and exp.editor.mode != "scripted"
    if uses_model_editor and ed_inf is not None and ed_inf.mode == "external":
        if exp.editor.mode == "current_learner":
            raise PlanError("editor=current_learner changes every cycle; a fixed external editor endpoint cannot serve it")
        needed = initial_id if exp.editor.mode == "initial_policy" else (
            base_checkpoint(editor).checkpoint_id if (exp.editor.checkpoint or "base") == "base"
            else CheckpointRecord.model_validate(read_json(Path(exp.editor.checkpoint) / "checkpoint.json")).checkpoint.checkpoint_id
        )
        if ed_inf.served_checkpoint_id != needed:
            raise PlanError(f"editor endpoint declares {ed_inf.served_checkpoint_id!r}; the editor is {needed!r}")
    if uses_model_editor and ed_inf is None and inf.mode == "scripted":
        raise PlanError("a model editor needs editor_inference when the learner is scripted")
    if trains and inf.mode == "external":
        raise PlanError("learning runs need managed (or scripted) inference: an external endpoint cannot serve new checkpoints")
    if trains and exp.training.trainer == "trl_dpo" and inf.mode == "managed":
        sb = learner.serving.get(inf.backend)
        if sb is None:
            raise PlanError(f"model profile {learner.name} declares no {inf.backend!r} serving backend")
        if "peft_lora" not in sb.adapter_formats:
            raise PlanError(
                f"{learner.name} via {inf.backend} accepts {sb.adapter_formats or 'no'} adapters, but the trainer "
                "produces peft_lora; OpenAI-compatible HTTP alone does not imply LoRA support"
            )
        if sb.status != "tested":
            notes.append(f"serving backend {inf.backend} for {learner.name} is marked {sb.status}")
    if trains and exp.training.trainer == "trl_dpo":
        sup, dev = learner.supported_train_devices, machine.training.device
        if sup and dev != "auto" and dev not in sup:
            raise PlanError(f"{learner.name} trains only on supported_train_devices {sup}; the machine's training.device is {dev!r}")
    if machine.pod_ids():
        import os

        rp = machine.runpod
        if machine.existing_pod_ids():
            notes.append(f"Runpod pods {machine.existing_pod_ids()} are started if needed and "
                         + ("stopped when the command ends" if rp.stop_when_done else "left running (watchdog stops them when idle)")
                         + f"; pod-side watchdog stops a pod after {rp.idle_stop_minutes} idle minutes")
        for name, spec in rp.create.items():
            if f"new-{name}" in machine.pod_ids():
                notes.append(f"each command creates a {spec.cloud_type.lower()} pod `{name}` ({' / '.join(spec.gpu_types)}, at most "
                             f"${spec.max_cost_per_hr}/hr) and terminates it when the command ends; its watchdog terminates it "
                             f"after {rp.idle_stop_minutes} idle minutes")
        if not os.environ.get(rp.api_key_env):
            notes.append(f"{rp.api_key_env} is not set (add it to .env): runs that use the pods will fail at start")
    if inf.mode == "scripted" or exp.training.trainer == "fixture" or exp.editor.mode == "scripted":
        notes.append("FIXTURE components in use (scripted policy/editor or fixture trainer): engineering check only, not a research result")
    if exp.editor.mode == "current_learner":
        notes.append("editor=current_learner: editor identity changes with every checkpoint (declared condition)")
    if machine.training.device == "cpu" or machine.training.allow_cpu_fallback:
        notes.append("CPU training explicitly allowed; recorded in checkpoint metadata")
    return notes


def resolve_tasks(exp: ExperimentConfig, tasks_dir: Path, extra_panels: list[str] | None = None) -> tuple[dict[str, TaskInstance], dict[str, list[str]], dict[str, Split], dict[str, Split], list[str]]:
    from . import tasks as task_mod

    splits = task_mod.load_splits(repo_path(exp.tasks.splits))
    needed = list(dict.fromkeys([exp.tasks.collection_panel, *exp.evaluation.dev_panels, *exp.evaluation.final_panels, *(extra_panels or [])]))
    for p in needed:
        if p not in splits.panels:
            raise PlanError(f"panel {p!r} not defined in {exp.tasks.splits} (have {sorted(splits.panels)})")
    panel_split = {name: Split(panel.split) for name, panel in splits.panels.items()}
    if panel_split[exp.tasks.collection_panel] != Split.TRAIN:
        raise PlanError(f"collection panel {exp.tasks.collection_panel!r} must be a train-split panel")
    for p in exp.evaluation.dev_panels:
        if panel_split[p] != Split.DEV:
            raise PlanError(f"dev panel {p!r} has split {panel_split[p].value}; per-cycle monitoring must use dev panels")
    for p in exp.evaluation.final_panels:
        if panel_split[p] not in (Split.FINAL, Split.EXTERNAL):
            raise PlanError(f"final panel {p!r} has split {panel_split[p].value}")
    if exp.tasks.exposure_schedule:
        fams = {i.family for i in splits.panels[exp.tasks.collection_panel].instances}
        for st in exp.tasks.exposure_schedule:
            unknown = set(st.families) - fams
            if unknown:
                raise PlanError(f"exposure schedule families {sorted(unknown)} are not in the collection panel")
    instances = task_mod.materialize(splits, tasks_dir, panels=needed)
    task_mod.validate_splits(splits, instances)  # overlap, forbidden families, content collisions
    panels = {p: [i.instance_id for i in splits.panels[p].instances] for p in needed}
    instance_split: dict[str, Split] = {}
    for p in needed:
        for iid in panels[p]:
            instance_split[iid] = panel_split[p]
    return instances, panels, panel_split, instance_split, list(splits.held_out_families)


@dataclass
class Workload:
    rows: list[dict[str, Any]]
    notes: list[str]

    def render(self) -> str:
        lines = ["cycle  stage      episodes/items  note"]
        for r in self.rows:
            lines.append(f"{r['cycle']:>5}  {r['stage']:<10} {r['count']:>14}  {r.get('note', '')}")
        return "\n".join(lines + [f"note: {n}" for n in self.notes])


def plan_workload(ctx: RunContext) -> Workload:
    exp = ctx.exp
    rows: list[dict[str, Any]] = []
    dev_eps = sum(len(ctx.panels[p]) for p in exp.evaluation.dev_panels) * exp.evaluation.attempts_per_instance
    for c in range(exp.cycles + 1):
        if eval_due(exp, c):
            rows.append({"cycle": c, "stage": "eval", "count": dev_eps, "note": f"panels {exp.evaluation.dev_panels}"})
        if c == exp.cycles or exp.condition == "frozen_baseline":
            continue
        if exp.condition == "learning":
            n_collect = len(collection_instances(ctx, c)) * exp.tasks.attempts_per_instance
            rows.append({"cycle": c, "stage": "collect", "count": n_collect})
            rows.append({"cycle": c, "stage": "edit", "count": f"<={n_collect * exp.editor.proposals_per_source}", "note": "successful sources only"})
            reps = 1 if exp.verification.mode == "local" else exp.verification.continuations_per_branch
            rows.append({"cycle": c, "stage": "verify", "count": f"<={n_collect * exp.editor.proposals_per_source * 2 * reps}", "note": f"mode {exp.verification.mode}"})
        rows.append({"cycle": c, "stage": "train", "count": exp.training.optimizer_steps, "note": f"optimizer steps ({exp.training.trainer})"})
    notes = [f"final panels {exp.evaluation.final_panels} run only via `loop evaluate`"] if exp.evaluation.final_panels else []
    return Workload(rows, notes)


def eval_due(exp: ExperimentConfig, cycle: int) -> bool:
    if exp.evaluation.cadence == "every_cycle":
        return True
    return cycle in (0, exp.cycles)


def collection_instances(ctx: RunContext, cycle: int) -> list[str]:
    ids = ctx.panels[ctx.exp.tasks.collection_panel]
    sched = ctx.exp.tasks.exposure_schedule
    if not sched:
        return ids
    active = [s for s in sched if s.from_cycle <= cycle]
    if not active:
        return []
    fams = set(max(active, key=lambda s: s.from_cycle).families)
    return [i for i in ids if ctx.instances[i].family in fams]


# --------------------------------------------------------------------------- #
# Run creation / loading
# --------------------------------------------------------------------------- #


def new_run_id(exp: ExperimentConfig) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{exp.name}-{stamp}-s{exp.seeds.loop_seed}"


def create_run(exp_path: str | Path, machine_path: str | Path, run_id: str | None = None, runs_dir: Path | None = None, overrides: list[str] | None = None) -> RunContext:
    exp, raw, machine, learner, editor = load_all(exp_path, machine_path, overrides)
    notes = check_compatibility(exp, machine, learner, editor)
    run_id = run_id or new_run_id(exp)
    run_dir = (runs_dir or repo_path(machine.runs_dir)) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    instances, panels, panel_split, instance_split, held_out = resolve_tasks(exp, run_dir / "tasks")
    ctx = RunContext(run_dir, run_id, exp, raw, machine, learner, editor, instances, panels, panel_split, instance_split, held_out)
    initial = initial_checkpoint(ctx)
    write_once_json(
        run_dir / "run.json",
        {
            "run_id": run_id,
            "experiment_path": str(exp_path),
            "overrides": overrides or [],
            "experiment": exp.model_dump(mode="json"),
            "machine": machine.model_dump(mode="json"),
            "model_profiles": {"learner": learner.model_dump(mode="json"), "editor": editor.model_dump(mode="json")},
            "serving": serving_record(machine, learner),
            "initial_checkpoint": initial.model_dump(mode="json"),
            "panels": panels,
            "panel_split": {k: v.value for k, v in panel_split.items()},
            "held_out_families": held_out,
            "instances": {k: v.model_dump(mode="json") for k, v in instances.items()},
            "compatibility_notes": notes,
        },
    )
    record_provenance(run_dir, run_id)
    atomic_write_json(run_dir / "seed_schedule.json", seed_schedule(ctx))
    return ctx


def record_provenance(run_dir: Path, run_id: str) -> None:
    """provenance.json is written once at creation; every later invocation (resume, re-run of
    an evaluation) appends its own record so code/package changes between sessions stay visible."""
    rec = provenance.collect({"run_id": run_id, "recorded_at": now_iso()})
    if not (run_dir / "provenance.json").exists():
        write_once_json(run_dir / "provenance.json", rec)
    else:
        JsonlAppender(run_dir / "invocations.jsonl").append(rec)


def serving_record(machine: MachineProfile, learner: ModelProfile) -> dict[str, Any]:
    """Serving backend/artifact/quantization/concurrency, so timing comparisons are interpretable."""
    inf = machine.inference
    sb = learner.serving.get(inf.backend)
    return {
        "mode": inf.mode,
        "backend": inf.backend,
        "artifact": (sb.artifact if sb and sb.artifact else learner.base_model) if inf.mode != "scripted" else None,
        "artifact_revision": (sb.artifact_revision if sb and sb.artifact else learner.base_revision) if inf.mode != "scripted" else None,
        "quantization": sb.quantization if sb else None,
        "backend_status": sb.status if sb else None,
        "chat_template_kwargs": learner.chat_template_kwargs,
        "request_concurrency": inf.request_concurrency,
        "docker_concurrency": machine.docker_concurrency,
        "device": inf.device,
    }


def open_run(run_dir: Path, machine_path: str | Path | None = None) -> RunContext:
    run_dir = Path(run_dir)
    meta = read_json(run_dir / "run.json")
    exp = ExperimentConfig.model_validate(meta["experiment"])
    learner = ModelProfile.model_validate(meta["model_profiles"]["learner"])
    editor = ModelProfile.model_validate(meta["model_profiles"]["editor"])
    if machine_path:
        # A deployment override (e.g. new ports/hosts): validate it and keep a record, since
        # run.json['machine'] then no longer describes the executing deployment alone.
        machine = load_machine(machine_path)
        if meta.get("kind", "learning") == "learning":
            check_compatibility(exp, machine, learner, editor)
        JsonlAppender(run_dir / "machine-overrides.jsonl").append({"at": now_iso(), "path": str(machine_path), "machine": machine.model_dump(mode="json")})
    else:
        machine = MachineProfile.model_validate(meta["machine"])
    instances = {k: TaskInstance.model_validate(v) for k, v in meta["instances"].items()}
    panel_split = {k: Split(v) for k, v in meta["panel_split"].items()}
    instance_split = {iid: panel_split[p] for p, ids in meta["panels"].items() for iid in ids}
    return RunContext(run_dir, meta["run_id"], exp, meta["experiment"], machine, learner, editor, instances, meta["panels"], panel_split, instance_split, meta.get("held_out_families", []))


def seed_schedule(ctx: RunContext) -> dict[str, Any]:
    exp = ctx.exp
    root = exp.seeds.root
    return {
        "root": root,
        "loop_seed": exp.seeds.loop_seed,
        "eval": {
            iid: [seeds.attempt_seed(root, iid, a) for a in range(exp.evaluation.attempts_per_instance)]
            for p in exp.evaluation.dev_panels + exp.evaluation.final_panels
            for iid in ctx.panels[p]
        },
        "streams": "collect/editor/continuation/training seeds derive from (root, loop_seed, stream, ids); see seeds.py",
    }


def loop_root(ctx: RunContext) -> int:
    """Root for loop-specific streams; evaluation seeds deliberately ignore loop_seed."""
    return seeds.derive_seed(ctx.exp.seeds.root, "training", "loop", ctx.exp.seeds.loop_seed)


def initial_checkpoint(ctx: RunContext) -> CheckpointRef:
    spec = ctx.exp.learner.initial_checkpoint
    if spec == "base":
        return base_checkpoint(ctx.learner_profile)
    rec = CheckpointRecord.model_validate(read_json(Path(spec) / "checkpoint.json"))
    if rec.checkpoint.model_profile != ctx.learner_profile.name:
        raise PlanError(f"initial checkpoint {spec} was trained for {rec.checkpoint.model_profile}")
    return rec.checkpoint


def base_checkpoint(profile: ModelProfile) -> CheckpointRef:
    return CheckpointRef(
        checkpoint_id=f"base:{profile.name}@{profile.base_revision[:12]}",
        model_profile=profile.name,
        base_model=profile.base_model,
        base_revision=profile.base_revision,
    )


# --------------------------------------------------------------------------- #
# Generic stage runner with bounded infra retries
# --------------------------------------------------------------------------- #


async def run_stage(
    ctx: RunContext,
    cycle: int | None,
    stage: str,
    items: list[tuple[str, dict[str, Any]]],
    fn: StageFn,
    concurrency: int = 1,
    stage_root: Path | None = None,
) -> StageManifest:
    root = stage_root or (ctx.stage_dir(cycle, stage) if cycle is not None else ctx.run_dir / stage)
    mpath = root / "manifest.json"
    m = StageManifest.load_or_create(mpath, stage, cycle)
    m.ensure_items([i for i, _ in items], {i: meta for i, meta in items})
    if m.status == "done" and not m.pending():
        return m  # nothing to do: never re-stamp a completed stage on resume
    m.status = "running"
    m.started_at = m.started_at or now_iso()
    interval = [now_iso(), ""]
    m.active_intervals.append(interval)
    m.save(mpath)
    sem = asyncio.Semaphore(max(1, concurrency))
    lock = asyncio.Lock()
    retries = ctx.exp.runtime.infra_retries

    async def one(item: WorkItem) -> None:
        async with sem:
            item_dir = root / "items" / item.item_id
            while True:
                if item.status == "running" or (item_dir.exists() and item.status != "done"):
                    preserve_interrupted(item_dir, item)  # prior interrupted/failed attempt stays visible
                async with lock:
                    item.status = "running"
                    item.attempts += 1
                    item.updated_at = now_iso()
                    m.save(mpath)
                try:
                    out = await fn(item, item_dir)
                    infra = out.get("infra_error")
                except Exception as e:  # noqa: BLE001 - classify, record, never swallow silently
                    out, infra = {"exception": f"{type(e).__name__}: {e}"}, None
                    async with lock:
                        item.status = "failed"
                        item.error = out["exception"]
                        item.updated_at = now_iso()
                        m.save(mpath)
                    return
                async with lock:
                    if infra:
                        item.infra_failures += 1
                        item.error = infra
                        if item.infra_failures <= retries:
                            item.status = "pending"
                            m.save(mpath)
                            continue
                        item.status = "infra_failed"
                    else:
                        item.status = "done"
                        item.error = None
                    item.output = str(item_dir.relative_to(ctx.run_dir))
                    item.meta.update({k: v for k, v in out.items() if k in ("success", "stop_reason", "status", "accepted")})
                    item.updated_at = now_iso()
                    m.save(mpath)
                return

    await asyncio.gather(*(one(w) for w in m.pending()))
    failed = [w for w in m.items.values() if w.status == "failed"]
    m.status = "failed" if failed else "done"
    m.finished_at = now_iso()
    interval[1] = m.finished_at
    m.summary = m.counts()
    m.save(mpath)
    if failed:
        raise RuntimeError(f"stage {stage} (cycle {cycle}): {len(failed)} item(s) failed with errors, e.g. {failed[0].item_id}: {failed[0].error}")
    return m


# --------------------------------------------------------------------------- #
# Episodes
# --------------------------------------------------------------------------- #


def tool_schemas() -> list[dict[str, Any]]:
    from evaluation.agents.tools import TOOLS

    return [t.schema() for t in TOOLS]


def episode_budgets(exp: ExperimentConfig) -> EpisodeBudgets:
    e = exp.episode
    return EpisodeBudgets(
        max_turns=e.max_turns,
        max_episode_tokens=e.max_episode_tokens,
        tool_timeout_sec=e.tool_timeout_sec,
        max_output_chars=e.max_output_chars,
        agent_timeout_sec=e.agent_timeout_sec,
    )


def make_plan(ctx: RunContext, inst: TaskInstance, role: EpisodeRole, episode_id: str, seed: int | None, policy_spec, attempt: int | None = None, replay=None) -> EpisodePlan:
    from .tasks import load_state_spec

    state = load_state_spec(Path(inst.task_dir))
    workdir = "/app"
    system_prompt = repo_path(ctx.exp.episode.system_prompt).read_text().format(workdir=workdir)
    instruction = (Path(inst.task_dir) / "instruction.md").read_text()
    return EpisodePlan(
        episode_id=episode_id,
        role=role,
        instance_id=inst.instance_id,
        attempt_index=attempt,
        seed=seed,
        policy=policy_spec,
        budgets=episode_budgets(ctx.exp),
        system_prompt=system_prompt,
        instruction=instruction,
        tools=tool_schemas(),
        state_spec=state,
        replay=replay,
    )


def learner_policy(ctx: RunContext, ckpt: CheckpointRef, sampling: SamplingConfig | None = None):
    if "editor" in ctx.managers and ctx.machine.editor_inference is None:
        ctx.managers["editor"].stop()  # shared slot: the editor server must go before the learner returns
    mgr = ctx.manager("learner")
    mgr.ensure(ckpt)
    return mgr.policy_spec(ckpt, sampling or ctx.exp.episode.sampling, scripted_path=scripted_learner_path(ctx))


def scripted_learner_path(ctx: RunContext) -> str | None:
    p = ctx.exp.learner.scripted_policy
    return str(repo_path(p)) if p else None


async def run_episode_item(ctx: RunContext, inst: TaskInstance, plan: EpisodePlan, item_dir: Path) -> dict[str, Any]:
    item_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(item_dir / "plan.json", plan)
    res = await ctx.backend.run(inst, plan, item_dir)
    atomic_write_json(item_dir / "summary.json", res.summary)
    s = res.summary
    return {
        "success": s.success,
        "stop_reason": s.stop_reason,
        "infra_error": s.infra_error if s.stop_category == StopCategory.INFRA else None,
    }


def load_summary(ctx: RunContext, item: WorkItem) -> EpisodeSummary | None:
    if not item.output:
        return None
    p = ctx.run_dir / item.output / "summary.json"
    return EpisodeSummary.model_validate(read_json(p)) if p.exists() else None


async def stage_eval(ctx: RunContext, cycle: int, ckpt: CheckpointRef, panels: list[str], stage: str = "eval", stage_root: Path | None = None) -> StageManifest:
    exp = ctx.exp
    items: list[tuple[str, dict[str, Any]]] = []
    for p in panels:
        for iid in ctx.panels[p]:
            for a in range(exp.evaluation.attempts_per_instance):
                eid = seeds.stable_id("ep", ctx.run_id, stage, cycle, ckpt.checkpoint_id, p, iid, a)
                items.append((eid, {"panel": p, "instance_id": iid, "attempt": a, "checkpoint_id": ckpt.checkpoint_id}))
    policy = learner_policy(ctx, ckpt) if items else None

    async def fn(item: WorkItem, item_dir: Path) -> dict[str, Any]:
        iid, a = item.meta["instance_id"], item.meta["attempt"]
        seed = seeds.attempt_seed(exp.seeds.root, iid, a)  # paired across checkpoints and loop seeds
        plan = make_plan(ctx, ctx.instances[iid], EpisodeRole.EVAL, item.item_id, seed, policy, attempt=a)
        return await run_episode_item(ctx, ctx.instances[iid], plan, item_dir)

    return await run_stage(ctx, cycle, stage, items, fn, concurrency=ctx.backend_concurrency(), stage_root=stage_root)


async def stage_collect(ctx: RunContext, cycle: int, ckpt: CheckpointRef) -> StageManifest:
    exp = ctx.exp
    items = []
    for iid in collection_instances(ctx, cycle):
        for a in range(exp.tasks.attempts_per_instance):
            eid = seeds.stable_id("ep", ctx.run_id, "collect", cycle, ckpt.checkpoint_id, iid, a)
            items.append((eid, {"instance_id": iid, "attempt": a, "checkpoint_id": ckpt.checkpoint_id}))
    policy = learner_policy(ctx, ckpt) if items else None

    async def fn(item: WorkItem, item_dir: Path) -> dict[str, Any]:
        iid, a = item.meta["instance_id"], item.meta["attempt"]
        seed = seeds.derive_seed(loop_root(ctx), "learner_attempt", "collect", cycle, iid, a)
        plan = make_plan(ctx, ctx.instances[iid], EpisodeRole.COLLECT, item.item_id, seed, policy, attempt=a)
        return await run_episode_item(ctx, ctx.instances[iid], plan, item_dir)

    return await run_stage(ctx, cycle, "collect", items, fn, concurrency=ctx.backend_concurrency())


def _backend_concurrency(self: RunContext) -> int:
    if self.machine.environment_backend == "local_fixture":
        return 1
    return max(1, min(self.machine.docker_concurrency, self.exp.runtime.docker_concurrency or self.machine.docker_concurrency))


RunContext.backend_concurrency = _backend_concurrency  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Editing and verification
# --------------------------------------------------------------------------- #


def editor_checkpoint(ctx: RunContext, learner: CheckpointRef) -> CheckpointRef:
    mode = ctx.exp.editor.mode
    if mode == "current_learner":
        return learner
    if mode == "external":
        spec = ctx.exp.editor.checkpoint or "base"
        if spec == "base":
            return base_checkpoint(ctx.editor_profile)
        return CheckpointRecord.model_validate(read_json(Path(spec) / "checkpoint.json")).checkpoint
    # initial_policy (and scripted): the fixed initial instruction-tuned policy, immutable across cycles
    return CheckpointRef.model_validate(read_json(ctx.run_dir / "run.json")["initial_checkpoint"])


def build_editor(ctx: RunContext, learner: CheckpointRef):
    from .editor import make_editor

    ed_ckpt = editor_checkpoint(ctx, learner)
    cfg = ctx.exp.editor
    if cfg.mode == "scripted":
        return make_editor(cfg, ed_ckpt, policy_spec=None)
    if ctx.machine.editor_inference is None and ed_ckpt.model_profile != ctx.learner_profile.name:
        ctx.manager("learner").stop()  # one serving slot: swap to the editor's model
        mgr = ctx.manager("editor")
    else:
        mgr = ctx.manager("learner" if ctx.shared_manager_for_editor() else "editor")
    mgr.ensure(ed_ckpt)
    spec = mgr.policy_spec(ed_ckpt, cfg.sampling)
    return make_editor(cfg, ed_ckpt, policy_spec=spec)


def source_item_dir(ctx: RunContext, w: WorkItem) -> Path:
    return ctx.run_dir / w.output  # type: ignore[operator]


def ref_path(ctx: RunContext, p: Path) -> str:
    """Store paths relative to the run when inside it (runs stay relocatable)."""
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ctx.run_dir.resolve()))
    except ValueError:
        return str(p)  # imported sources from another run (edit-replay)


def resolve_ref(ctx: RunContext, ref: str) -> Path:
    p = Path(ref)
    return p if p.is_absolute() else ctx.run_dir / p


def successful_sources(ctx: RunContext, collect: StageManifest) -> list[tuple[Path, EpisodeSummary]]:
    """(source item dir, summary) for completed, completely successful collection episodes."""
    out = []
    for w in collect.items.values():
        if w.status != "done":
            continue
        s = load_summary(ctx, w)
        if s is not None and s.success:
            out.append((source_item_dir(ctx, w), s))
    return out


async def stage_edit(ctx: RunContext, cycle: int, learner: CheckpointRef, sources: list[tuple[Path, EpisodeSummary]]) -> StageManifest:
    editor = build_editor(ctx, learner) if sources else None
    items = []
    for src_dir, s in sources:
        for k in range(ctx.exp.editor.proposals_per_source):
            pid = seeds.stable_id("prop", ctx.run_id, cycle, s.episode_id, editor.editor_id, k)
            items.append((pid, {"source_episode_id": s.episode_id, "source_dir": ref_path(ctx, src_dir), "instance_id": s.instance_id, "k": k, "editor_id": editor.editor_id}))

    async def fn(item: WorkItem, item_dir: Path) -> dict[str, Any]:
        src_dir = resolve_ref(ctx, item.meta["source_dir"])
        summary = EpisodeSummary.model_validate(read_json(src_dir / "summary.json"))
        inst = ctx.instances[summary.instance_id]
        proposal = await editor.propose_for(
            proposal_id=item.item_id,
            source=summary,
            source_dir=src_dir,
            instance=inst,
            seed=seeds.derive_seed(loop_root(ctx), "editor_proposal", item.item_id),
        )
        item_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(item_dir / "proposal.json", proposal)
        infra = next((r for r in proposal.rejection_reasons if r.startswith("editor_infra_error")), None)
        return {"status": proposal.status, "infra_error": infra}

    return await run_stage(ctx, cycle, "edit", items, fn, concurrency=1)


def load_proposals(ctx: RunContext, edit: StageManifest) -> list[tuple[EditProposal, Path]]:
    out = []
    for w in edit.items.values():
        if w.status == "done" and w.output:
            p = ctx.run_dir / w.output / "proposal.json"
            out.append((EditProposal.model_validate(read_json(p)), resolve_ref(ctx, w.meta["source_dir"])))
    return out


def make_verifier(ctx: RunContext, learner: CheckpointRef, purpose: str = "acceptance"):
    from .verify import make_verifier as _make

    v = ctx.exp.verification
    if purpose == "audit":
        v = v.model_copy(update={"continuations_per_branch": v.audit.continuations_per_branch})
    sampling = v.sampling or ctx.exp.episode.sampling
    policy = learner_policy(ctx, learner, sampling) if v.mode == "continuation" else None
    return _make(
        mode=v.mode,
        backend=ctx.backend,
        config=v,
        learner_profile=ctx.learner_profile,  # audit/acceptance differ only in repetitions and seed stream
        policy_spec=policy,
        plan_factory=lambda inst, role, eid, seed, replay: make_plan(ctx, inst, role, eid, seed, policy, replay=replay),
        root_seed=loop_root(ctx),
        scripted=ctx.machine.inference.mode == "scripted",
    )


async def stage_verify(ctx: RunContext, cycle: int, learner: CheckpointRef, proposals: list[tuple[EditProposal, Path]], purpose: str = "acceptance", stage: str = "verify") -> StageManifest:
    valid = [(p, src) for p, src in proposals if p.status == "proposed"]
    verifier = make_verifier(ctx, learner, purpose) if valid else None
    items = [
        (seeds.stable_id("ver", p.proposal_id, purpose), {"proposal_id": p.proposal_id, "source_dir": ref_path(ctx, src), "purpose": purpose})
        for p, src in valid
    ]
    by_id = {p.proposal_id: p for p, _ in valid}

    async def fn(item: WorkItem, item_dir: Path) -> dict[str, Any]:
        p = by_id[item.meta["proposal_id"]]
        src_dir = resolve_ref(ctx, item.meta["source_dir"])
        rec: VerificationRecord = await verifier.verify(
            proposal=p,
            verification_id=item.item_id,
            source_dir=src_dir,
            instance=ctx.instances[p.instance_id],
            out_dir=item_dir,
            purpose=purpose,
        )
        atomic_write_json(item_dir / "verification.json", rec)
        infra = next((b.episode.infra_error for b in rec.branches if b.episode.stop_category == StopCategory.INFRA), None)
        return {"accepted": rec.accepted, "infra_error": infra}

    return await run_stage(ctx, cycle, stage, items, fn, concurrency=ctx.backend_concurrency())


def load_verifications(ctx: RunContext, m: StageManifest) -> list[VerificationRecord]:
    out = []
    for w in m.items.values():
        if w.output and (ctx.run_dir / w.output / "verification.json").exists():
            out.append(VerificationRecord.model_validate(read_json(ctx.run_dir / w.output / "verification.json")))
    return out


async def stage_audit(ctx: RunContext, cycle: int, learner: CheckpointRef, accepted: list[VerificationRecord], proposals: list[tuple[EditProposal, Path]]) -> None:
    frac = ctx.exp.verification.audit.fraction
    if frac <= 0 or not accepted:
        return
    rng = random.Random(seeds.derive_seed(loop_root(ctx), "audit", "select", cycle))
    chosen = {v.proposal_id for v in accepted if rng.random() < frac}
    subset = [(p, s) for p, s in proposals if p.proposal_id in chosen]
    if subset:
        # Audits never change the frozen dataset; they are reported separately.
        await stage_verify(ctx, cycle, learner, subset, purpose="audit", stage="audit")


# --------------------------------------------------------------------------- #
# Dataset and training
# --------------------------------------------------------------------------- #


def stage_dataset(ctx: RunContext, cycle: int, learner: CheckpointRef, verifications: list[VerificationRecord], proposals: list[tuple[EditProposal, Path]]) -> dict[str, Any]:
    from . import preferences as prefs

    out_dir = ctx.stage_dir(cycle, "dataset")
    if (out_dir / "manifest.json").exists():
        return read_json(out_dir / "manifest.json")  # immutable once exported
    if ctx.exp.condition == "fixed_dataset":
        src = repo_path(ctx.exp.training.fixed_dataset)  # type: ignore[arg-type]
        heldout = {iid for iid, sp in ctx.instance_split.items() if sp != Split.TRAIN}
        return prefs.freeze_fixed_dataset(src, out_dir, heldout_instance_ids=heldout, forbidden_families=set(ctx.held_out_families))
    by_pid = {p.proposal_id: (p, src) for p, src in proposals}
    current = []
    for v in verifications:
        if not v.accepted or v.purpose != "acceptance":
            continue
        p, src = by_pid[v.proposal_id]
        inst = ctx.instances[p.instance_id]
        current.append(
            prefs.build_pair(
                proposal=p,
                verification=v,
                source_dir=src,
                instance=inst,
                split=ctx.instance_split[p.instance_id],
                learner_checkpoint_id=learner.checkpoint_id,
                cycle=cycle,
                run_id=ctx.run_id,
                kind=pair_kind(ctx),
            )
        )
    history = []
    for c in range(cycle):
        cur = ctx.stage_dir(c, "dataset") / "current"
        if cur.exists():
            history.extend(prefs.load_pairs(cur))
    selected = prefs.select_training_pairs(current, history, ctx.exp.training.data, seed=seeds.derive_seed(loop_root(ctx), "data_selection", cycle))
    heldout = {iid for iid, sp in ctx.instance_split.items() if sp != Split.TRAIN}
    forbidden = set(ctx.held_out_families)
    prefs.export_dataset(current, out_dir / "current", heldout_instance_ids=heldout, forbidden_families=forbidden)
    return prefs.export_dataset(selected, out_dir, heldout_instance_ids=heldout, forbidden_families=forbidden, extra_manifest={"cycle": cycle, "n_current": len(current), "n_history_pool": len(history)})


class NoUpdate(Exception):
    """The trainer found no trainable examples (e.g. all dropped as oversize)."""

    def __init__(self, message: str, render: dict[str, Any] | None = None):
        super().__init__(message)
        self.render = render


def training_data_summary(ds: dict[str, Any], render: dict[str, Any] | None, max_length: int | None) -> dict[str, Any]:
    """What the trainer actually used of the exported dataset (recorded in cycle.json, reported).
    Dropped pairs change the training data, so they are counted and listed, never silent."""
    out: dict[str, Any] = {"n_exported": ds.get("n_examples", 0), "max_length": max_length}
    if render is None:
        return {**out, "rendered": False}
    return {**out, "rendered": True, "n_trained": render.get("n_kept"), "n_dropped": render.get("n_dropped"),
            "dropped_by_reason": render.get("dropped_by_reason") or {}, "dropped": render.get("dropped") or [],
            "kept_tokens": render.get("kept_tokens")}


def pair_kind(ctx: RunContext) -> str:
    """Pairs produced with a scripted learner or scripted editor are fixture data, never 'verified'."""
    scripted = ctx.machine.inference.mode == "scripted" or ctx.exp.editor.mode == "scripted"
    return "fixture" if scripted else "verified"


def train_checkpoint(ctx: RunContext, cycle: int, incoming: CheckpointRef, dataset_dir: Path) -> CheckpointRecord:
    """Run the trainer in a separate process (memory isolation; same entry point remotely).

    Re-running resumes: the trainer restores its own stage checkpoint from the work dir, and
    an already-published checkpoint for identical inputs is returned as-is.
    """
    work = ctx.stage_dir(cycle, "train")
    done = work / "checkpoint_path.txt"
    if done.exists():
        return CheckpointRecord.model_validate(read_json(Path(done.read_text().strip()) / "checkpoint.json"))
    req = TrainRequest(
        run_id=ctx.run_id,
        cycle=cycle,
        dataset_dir=str(dataset_dir),
        incoming=incoming,
        model_profile=ctx.learner_profile.name,
        training_config=ctx.exp.training.model_dump(mode="json"),
        seed=seeds.derive_seed(loop_root(ctx), "training", cycle),
        output_root=str(ctx.checkpoints_dir),
        device=ctx.machine.training.device,
        allow_cpu_fallback=ctx.machine.training.allow_cpu_fallback,
    )
    work.mkdir(parents=True, exist_ok=True)
    write_once_json(work / "request.json", req)
    if ctx.machine.training.host.kind == "local":
        argv = [
            sys.executable, "-m", "learning_loop.training.run",
            "--request", str(work / "request.json"),
            "--work-dir", str(work / "work"),
            "--ref-cache-dir", str(ctx.run_dir / "cache" / "ref_logps"),
        ]
        with open(work / "train.log", "a") as log:
            proc = subprocess.run(argv, stdout=subprocess.PIPE, stderr=log, text=True)
        if proc.returncode == 3:
            res = read_json(work / "work" / "result.json") if (work / "work" / "result.json").exists() else {}
            raise NoUpdate(res.get("error") or "no trainable examples", render=res.get("render"))
        if proc.returncode == 4:
            raise RuntimeError(f"training work dir {work / 'work'} is held by another trainer process; not starting a second one")
        if proc.returncode != 0:
            raise RuntimeError(f"training failed (exit {proc.returncode}); see {work / 'train.log'}")
        ckpt_json = Path(proc.stdout.strip().splitlines()[-1])
    else:
        ckpt_json = train_remote(ctx, work, req)
    rec = CheckpointRecord.model_validate(read_json(ckpt_json))
    if rec.reference_checkpoint_id != incoming.checkpoint_id:
        raise RuntimeError(f"trainer used reference {rec.reference_checkpoint_id}, expected incoming {incoming.checkpoint_id}")
    if rec.checkpoint.parent_checkpoint_id != incoming.checkpoint_id:
        raise RuntimeError("checkpoint lineage does not continue the incoming learner")
    done.write_text(str(ckpt_json.parent))
    return rec


def train_remote(ctx: RunContext, work: Path, req: TrainRequest, poll_sec: int = 30) -> Path:
    """Push inputs, start the trainer detached in its own process group, poll its result file,
    then pull the checkpoint into a staging dir, verify the adapter hash, publish it locally and
    rewrite adapter_path to the local path. The remote training log is copied to
    `train/remote-train.log`.

    Reconciliation: the launch record (PID) is kept in the stage dir. On resume, a still-running
    remote trainer is waited for, never relaunched (the PID check also matches the command line,
    so a reused PID after a pod restart does not count); the trainer refuses a held work-dir lock.
    If the trainer is gone without a published result (e.g. the pod was stopped or lost), the
    stage is relaunched: it resumes from the trainer's own checkpoint when the remote work dir
    survived (a pod volume), otherwise it restarts from step 0. Each relaunch is recorded in
    `train/relaunches.jsonl`.
    """
    import time

    from .training.common import adapter_sha256

    remote = Remote(ctx.machine.training.host)
    rel = f"runs/{ctx.run_id}"
    stage_rel = f"{rel}/remote-train/c{req.cycle:03d}"
    launch = work / "remote_launch.json"
    result_rel = f"{stage_rel}/work/result.json"

    def remote_result() -> dict[str, Any] | None:
        txt = remote.read_text(result_rel)
        try:
            return json.loads(txt) if txt else None
        except json.JSONDecodeError:
            return None

    def alive() -> bool:
        return launch.exists() and remote.pid_alive(read_json(launch)["pid"], needle=stage_rel)

    def fetch_log() -> None:
        try:
            remote.pull(f"{stage_rel}/train.log", work / "remote-train.log")
        except Exception:  # noqa: BLE001 - diagnostics only
            pass

    res = remote_result()
    if not alive() and not (res and res.get("status") == "published"):
        if launch.exists():
            JsonlAppender(work / "relaunches.jsonl").append({
                "at": now_iso(), "previous": read_json(launch), "previous_result": res,
                "resumes_from_remote_state": remote.read_text(f"{stage_rel}/work/request.json") is not None,
            })
        remote.run(["rm", "-f", result_rel])  # never mistake an earlier attempt's result for this one
        remote.push_repo()
        remote.run(["mkdir", "-p", stage_rel, f"{rel}/checkpoints"])
        remote.push(Path(req.dataset_dir), f"{stage_rel}/")
        incoming = req.incoming
        if incoming.adapter_path:
            local_in = ctx.checkpoints_dir / incoming.checkpoint_id  # always push from the local copy
            if remote.read_text(f"{rel}/checkpoints/{incoming.checkpoint_id}/checkpoint.json") is None:
                remote.push(local_in, f"{rel}/checkpoints/")  # published dirs are immutable: never overwrite
            incoming = incoming.model_copy(update={"adapter_path": f"{remote.workdir}/{rel}/checkpoints/{incoming.checkpoint_id}"})
        remote_req = req.model_copy(
            update={
                "dataset_dir": f"{remote.workdir}/{stage_rel}/{Path(req.dataset_dir).name}",
                "output_root": f"{remote.workdir}/{rel}/checkpoints",
                "incoming": incoming,
            }
        )
        atomic_write_json(work / "remote_request.json", remote_req)
        remote.push(work / "remote_request.json", f"{stage_rel}/request.json")
        pid = remote.start_detached(
            ["uv", "run", "--extra", "train", "python", "-m", "learning_loop.training.run", "--request", f"{stage_rel}/request.json", "--work-dir", f"{stage_rel}/work"],
            f"{stage_rel}/train.log",
        )
        atomic_write_json(launch, {"pid": pid, "host": remote.alias, "started_at": now_iso()})
    while True:
        res = remote_result()
        if res and res.get("status") in ("published", "no_trainable_examples", "invalid_request", "failed"):
            break
        if not alive():
            res = remote_result()
            if res and res.get("status"):
                break
            fetch_log()
            raise RuntimeError(f"remote trainer exited without a result; see {work / 'remote-train.log'}")
        time.sleep(poll_sec)
    fetch_log()
    if res["status"] == "no_trainable_examples":
        raise NoUpdate(res.get("error", "no trainable examples (remote)"), render=res.get("render"))
    if res["status"] != "published":
        raise RuntimeError(f"remote training {res['status']}: {res.get('error')}")
    cid = res["checkpoint_id"]
    final = ctx.checkpoints_dir / cid
    if not final.exists():
        staging = ctx.checkpoints_dir / f".{cid}.pulling"
        if staging.exists():
            shutil.rmtree(staging)
        remote.pull(f"{rel}/checkpoints/{cid}/", staging)
        rec = CheckpointRecord.model_validate(read_json(staging / "checkpoint.json"))
        files = ("fixture_adapter.json",) if rec.trainer == "fixture" else ("adapter_model.safetensors", "adapter_config.json")
        if not rec.checkpoint.adapter_sha256 or adapter_sha256(staging, files) != rec.checkpoint.adapter_sha256:
            raise RuntimeError(f"pulled adapter {cid} does not match its recorded sha256")
        rec.checkpoint.adapter_path = str(final.resolve())  # local path; remote origin kept below
        rec.training_config = {**rec.training_config, "remote_origin": {"host": remote.alias, "path": f"{remote.workdir}/{rel}/checkpoints/{cid}"}}
        (staging / "checkpoint.json").chmod(0o644)
        atomic_write_json(staging / "checkpoint.json", rec)
        (staging / "checkpoint.json").chmod(0o444)
        staging.rename(final)
    return final / "checkpoint.json"


# --------------------------------------------------------------------------- #
# Cycle loop
# --------------------------------------------------------------------------- #


def cycle_state_path(ctx: RunContext, cycle: int) -> Path:
    return ctx.cycle_dir(cycle) / "cycle.json"


def learner_for_cycle(ctx: RunContext, cycle: int) -> CheckpointRef:
    if cycle == 0:
        return CheckpointRef.model_validate(read_json(ctx.run_dir / "run.json")["initial_checkpoint"])
    prev = read_json(cycle_state_path(ctx, cycle - 1))
    if prev.get("status") != "done":
        raise RuntimeError(f"cycle {cycle - 1} is not complete")
    return CheckpointRef.model_validate(prev["learner_out"])


async def run_cycle(ctx: RunContext, cycle: int, log: Callable[[str], None]) -> None:
    exp = ctx.exp
    spath = cycle_state_path(ctx, cycle)
    state = read_json(spath) if spath.exists() else {}
    if state.get("status") == "done":
        return
    learner = learner_for_cycle(ctx, cycle)
    state.update({"cycle": cycle, "learner_in": learner.model_dump(mode="json"), "status": "running", "started_at": state.get("started_at") or now_iso()})
    atomic_write_json(spath, state)

    if eval_due(exp, cycle):
        log(f"cycle {cycle}: evaluating {learner.checkpoint_id} on {exp.evaluation.dev_panels}")
        await stage_eval(ctx, cycle, learner, exp.evaluation.dev_panels)

    last = cycle == exp.cycles or exp.condition == "frozen_baseline"
    if last:
        state.update({"status": "done", "learner_out": learner.model_dump(mode="json"), "update": "final_evaluation_only", "finished_at": now_iso()})
        atomic_write_json(spath, state)
        return

    verifications: list[VerificationRecord] = []
    proposals: list[tuple[EditProposal, Path]] = []
    if exp.condition == "learning":
        log(f"cycle {cycle}: collecting with {learner.checkpoint_id}")
        collect = await stage_collect(ctx, cycle, learner)
        sources = successful_sources(ctx, collect)
        log(f"cycle {cycle}: {len(sources)} successful source trajectories -> editor")
        edit = await stage_edit(ctx, cycle, learner, sources)
        proposals = load_proposals(ctx, edit)
        log(f"cycle {cycle}: verifying {sum(p.status == 'proposed' for p, _ in proposals)} valid proposals")
        ver = await stage_verify(ctx, cycle, learner, proposals)
        verifications = load_verifications(ctx, ver)
        await stage_audit(ctx, cycle, learner, [v for v in verifications if v.accepted], proposals)

    ctx.stop_serving()  # free the accelerator before training
    ds = stage_dataset(ctx, cycle, learner, verifications, proposals)
    n = ds.get("n_examples", 0)
    rec: CheckpointRecord | None = None
    max_len = exp.training.dpo.max_length if exp.training.trainer == "trl_dpo" else None
    training_data: dict[str, Any] | None = None
    if n > 0:
        log(f"cycle {cycle}: training on {n} pairs ({exp.training.optimizer_steps} optimizer steps, reference = {learner.checkpoint_id})")
        try:
            rec = train_checkpoint(ctx, cycle, learner, ctx.stage_dir(cycle, "dataset"))
            training_data = training_data_summary(ds, (rec.metrics or {}).get("render"), max_len)
        except NoUpdate as e:
            training_data = training_data_summary(ds, e.render, max_len)
            log(f"cycle {cycle}: trainer found no trainable examples ({e}) -> no-update cycle")
        if training_data and training_data.get("n_dropped"):
            log(f"cycle {cycle}: WARNING {training_data['n_dropped']} of {n} pairs dropped before training: {training_data['dropped_by_reason']}")
    if rec is None:
        if n == 0:
            log(f"cycle {cycle}: no acceptable pairs -> no-update cycle (learner unchanged)")
        state.update({"status": "done", "update": "no_update", "learner_out": learner.model_dump(mode="json"), "reference": None, "dataset": ds,
                      "training_data": training_data, "finished_at": now_iso()})
        atomic_write_json(spath, state)
        return
    state.update(
        {
            "status": "done",
            "update": "trained",
            "learner_out": rec.checkpoint.model_dump(mode="json"),
            "reference": learner.checkpoint_id,
            "dataset": ds,
            "training_data": training_data,
            "checkpoint_record": rec.model_dump(mode="json"),
            "finished_at": now_iso(),
        }
    )
    atomic_write_json(spath, state)


def _existing_manifest(ctx: RunContext, cycle: int, stage: str) -> StageManifest | None:
    p = ctx.stage_dir(cycle, stage) / "manifest.json"
    return StageManifest.model_validate(read_json(p)) if p.exists() else None


def _require(m: StageManifest | None, stage: str, cycle: int) -> StageManifest:
    if m is None or m.status != "done":
        raise RuntimeError(f"stage {stage} of cycle {cycle} has not completed; run it first")
    return m


async def run_single_stage(ctx: RunContext, cycle: int, stage: str) -> None:
    """Run one stage of one cycle using the outputs of earlier stages on disk.

    Stage results land in the same manifests the full loop uses, so a later
    `loop resume` skips whatever was completed here.
    """
    learner = learner_for_cycle(ctx, cycle)
    if stage == "eval":
        await stage_eval(ctx, cycle, learner, ctx.exp.evaluation.dev_panels)
    elif stage == "collect":
        await stage_collect(ctx, cycle, learner)
    elif stage == "edit":
        collect = _require(_existing_manifest(ctx, cycle, "collect"), "collect", cycle)
        await stage_edit(ctx, cycle, learner, successful_sources(ctx, collect))
    elif stage == "verify":
        proposals = load_proposals(ctx, _require(_existing_manifest(ctx, cycle, "edit"), "edit", cycle))
        ver = await stage_verify(ctx, cycle, learner, proposals)
        await stage_audit(ctx, cycle, learner, [v for v in load_verifications(ctx, ver) if v.accepted], proposals)
    elif stage == "dataset":
        edit, ver = _existing_manifest(ctx, cycle, "edit"), _existing_manifest(ctx, cycle, "verify")
        proposals = load_proposals(ctx, edit) if edit else []
        verifications = load_verifications(ctx, _require(ver, "verify", cycle)) if ctx.exp.condition == "learning" else []
        print(json.dumps(stage_dataset(ctx, cycle, learner, verifications, proposals), indent=2, default=str))
    elif stage == "train":
        ds_dir = ctx.stage_dir(cycle, "dataset")
        if not (ds_dir / "manifest.json").exists():
            raise RuntimeError(f"no exported dataset for cycle {cycle}; run the dataset stage first")
        ctx.stop_serving()
        try:
            rec = train_checkpoint(ctx, cycle, learner, ds_dir)
            print(f"published {rec.checkpoint.checkpoint_id} -> {rec.checkpoint.adapter_path}")
        except NoUpdate as e:
            print(f"no update: {e}")
    else:
        raise ValueError(f"unknown stage {stage}")


# --------------------------------------------------------------------------- #
# Standalone evaluation and editor-comparison runs
# --------------------------------------------------------------------------- #


def evaluate_checkpoint(exp_path: str | Path, machine_path: str | Path, checkpoint: str, panels: list[str], final: bool = False, run_id: str | None = None, overrides: list[str] | None = None, runs_dir: Path | None = None) -> Path:
    """Evaluate any saved checkpoint without retraining. Output is its own run dir that
    the learning loop never reads (final/external results cannot select adapters)."""
    exp, raw, machine, learner, editor = load_all(exp_path, machine_path, overrides)
    if machine.inference.mode == "scripted" and not exp.learner.scripted_policy:
        raise PlanError("scripted inference requires learner.scripted_policy")
    ckpt = base_checkpoint(learner) if checkpoint == "base" else CheckpointRecord.model_validate(read_json(Path(checkpoint) / "checkpoint.json")).checkpoint
    if ckpt.model_profile != learner.name:
        raise PlanError(f"checkpoint {ckpt.checkpoint_id} belongs to {ckpt.model_profile}, experiment learner is {learner.name}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = run_id or f"{exp.name}-eval-{stamp}"
    run_dir = (runs_dir or repo_path(machine.runs_dir)) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    instances, all_panels, panel_split, instance_split, held_out = resolve_tasks(exp, run_dir / "tasks", extra_panels=panels)
    for p in panels:
        if panel_split[p] in (Split.FINAL, Split.EXTERNAL) and not final:
            raise PlanError(f"panel {p!r} is a {panel_split[p].value} panel: pass --final once method/selection decisions are frozen")
        if panel_split[p] == Split.TRAIN:
            raise PlanError(f"panel {p!r} is a training panel; evaluate on dev/final panels")
    ctx = RunContext(run_dir, run_id, exp, raw, machine, learner, editor, instances, all_panels, panel_split, instance_split, held_out)
    write_once_json(
        run_dir / "run.json",
        {
            "run_id": run_id,
            "kind": "evaluation",
            "final": final,
            "experiment_path": str(exp_path),
            "experiment": exp.model_dump(mode="json"),
            "machine": machine.model_dump(mode="json"),
            "model_profiles": {"learner": learner.model_dump(mode="json"), "editor": editor.model_dump(mode="json")},
            "initial_checkpoint": ckpt.model_dump(mode="json"),
            "evaluated_checkpoint": ckpt.model_dump(mode="json"),
            "evaluated_panels": panels,
            "panels": all_panels,
            "panel_split": {k: v.value for k, v in panel_split.items()},
            "held_out_families": held_out,
            "instances": {k: v.model_dump(mode="json") for k, v in instances.items()},
        },
    )
    record_provenance(run_dir, run_id)
    asyncio.run(_run_evaluation(ctx, ckpt, panels))
    return run_dir


async def _run_evaluation(ctx: RunContext, ckpt: CheckpointRef, panels: list[str]) -> None:
    with run_lock(ctx.run_dir):
        with ctx.pods():
            try:
                await stage_eval(ctx, 0, ckpt, panels)
            finally:
                ctx.stop_serving()
        write_report(ctx)


def edit_replay(source_run: Path, cycle: int, exp_path: str | Path, machine_path: str | Path, run_id: str | None = None, runs_dir: Path | None = None, overrides: list[str] | None = None) -> Path:
    """Controlled editor comparison: send the SAME saved source trajectories (one cycle of
    another run) through this experiment's editor + verifier. Uses the source run's task
    instances and the learner that produced those trajectories; no collection, no training."""
    src = open_run(Path(source_run))
    exp, raw, machine, learner, editor = load_all(exp_path, machine_path, overrides)
    if exp.condition != "learning":
        raise PlanError("edit-replay needs condition: learning (editing + verification)")
    if learner.name != src.learner_profile.name:
        raise PlanError("edit-replay must use the source run's learner profile")
    check_compatibility(exp.model_copy(update={"cycles": 0, "condition": "frozen_baseline"}), machine, learner, editor)
    _require(_existing_manifest(src, cycle, "collect"), "collect", cycle)
    src_learner = CheckpointRef.model_validate(read_json(cycle_state_path(src, cycle))["learner_in"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = run_id or f"{exp.name}-editreplay-{stamp}"
    run_dir = (runs_dir or repo_path(machine.runs_dir)) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(run_dir, run_id, exp, raw, machine, learner, editor, src.instances, src.panels, src.panel_split, src.instance_split, src.held_out_families)
    write_once_json(
        run_dir / "run.json",
        {
            "run_id": run_id,
            "kind": "edit_replay",
            "imported_sources": {"run": str(Path(source_run).resolve()), "cycle": cycle, "learner": src_learner.model_dump(mode="json")},
            "experiment_path": str(exp_path),
            "overrides": overrides or [],
            "experiment": exp.model_dump(mode="json"),
            "machine": machine.model_dump(mode="json"),
            "model_profiles": {"learner": learner.model_dump(mode="json"), "editor": editor.model_dump(mode="json")},
            "initial_checkpoint": read_json(src.run_dir / "run.json")["initial_checkpoint"],
            "panels": src.panels,
            "panel_split": {k: v.value for k, v in src.panel_split.items()},
            "held_out_families": src.held_out_families,
            "instances": {k: v.model_dump(mode="json") for k, v in src.instances.items()},
        },
    )
    record_provenance(run_dir, run_id)
    asyncio.run(_run_edit_replay(ctx, src, cycle, src_learner))
    return run_dir


async def _run_edit_replay(ctx: RunContext, src: RunContext, cycle: int, src_learner: CheckpointRef) -> None:
    collect = _require(_existing_manifest(src, cycle, "collect"), "collect", cycle)
    sources = successful_sources(src, collect)
    with run_lock(ctx.run_dir):
        with ctx.pods():
            try:
                edit = await stage_edit(ctx, 0, src_learner, sources)
                proposals = load_proposals(ctx, edit)
                ver = await stage_verify(ctx, 0, src_learner, proposals)
                ctx.stop_serving()
                stage_dataset(ctx, 0, src_learner, load_verifications(ctx, ver), proposals)
            finally:
                ctx.stop_serving()
        write_report(ctx)


def resume_run(run_dir: Path, machine_path: str | Path | None = None) -> None:
    """Resume any run kind from its manifests: learning loops, standalone evaluations and
    editor replays each re-enter their own stages (never the full loop for the latter two)."""
    run_dir = Path(run_dir)
    meta = read_json(run_dir / "run.json")
    kind = meta.get("kind", "learning")
    ctx = open_run(run_dir, machine_path)
    record_provenance(run_dir, ctx.run_id)
    if kind == "learning":
        asyncio.run(run_all(ctx))
    elif kind == "evaluation":
        ckpt = CheckpointRef.model_validate(meta["evaluated_checkpoint"])
        asyncio.run(_run_evaluation(ctx, ckpt, meta["evaluated_panels"]))
    elif kind == "edit_replay":
        imp = meta["imported_sources"]
        src = open_run(Path(imp["run"]))
        asyncio.run(_run_edit_replay(ctx, src, imp["cycle"], CheckpointRef.model_validate(imp["learner"])))
    else:
        raise PlanError(f"unknown run kind {kind!r}")


async def run_all(ctx: RunContext, log: Callable[[str], None] | None = None) -> None:
    log = log or (lambda m: print(m, flush=True))
    kind = read_json(ctx.run_dir / "run.json").get("kind", "learning")
    if kind != "learning":
        raise PlanError(f"{ctx.run_dir.name} is a {kind} run; use `loop resume`, which re-enters its own stages")
    with run_lock(ctx.run_dir):
        with ctx.pods(log):
            try:
                for c in range(ctx.exp.cycles + 1):
                    await run_cycle(ctx, c, log)
            finally:
                ctx.stop_serving()  # before the pods stop, so server logs can be copied back
        write_report(ctx)


def write_report(ctx: RunContext) -> Path:
    from .report import write_run_report

    return write_run_report(ctx.run_dir)


def run_status(run_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"run": run_dir.name, "cycles": {}}
    for cdir in sorted((run_dir / "cycles").glob("cycle-*")):
        entry: dict[str, Any] = {}
        if (cdir / "cycle.json").exists():
            st = read_json(cdir / "cycle.json")
            entry["status"] = st.get("status")
            entry["update"] = st.get("update")
            entry["learner_in"] = st.get("learner_in", {}).get("checkpoint_id")
            entry["learner_out"] = (st.get("learner_out") or {}).get("checkpoint_id")
            td = st.get("training_data")
            if td:
                entry["training_data"] = {k: td.get(k) for k in ("n_exported", "n_trained", "n_dropped", "dropped_by_reason") if k in td}
        for m in sorted(cdir.glob("*/manifest.json")):
            data = read_json(m)
            if "stage" not in data or "items" not in data:
                continue  # e.g. the dataset export manifest
            sm = StageManifest.model_validate(data)
            entry[sm.stage] = {
                "status": sm.status,
                **sm.counts(),
                "attempts": sum(w.attempts for w in sm.items.values()),
                "infra_failures": sum(w.infra_failures for w in sm.items.values()),
            }
        out["cycles"][cdir.name] = entry
    return out




