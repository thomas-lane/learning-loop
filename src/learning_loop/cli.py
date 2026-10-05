"""`loop`: command-line entry point of the learning loop.

Every command is documented by its own `--help` and in docs/cli.md (generated from this
parser with `loop docs-gen`; browse all docs with `loop docs`). Exit codes: 0 success, 1 failure, 2 invalid usage/config/plan.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

from .core.config import REPO_ROOT, load_machine, repo_path


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_validate(a: argparse.Namespace) -> int:
    from .orchestration.coordinator import RunContext, check_compatibility, load_all, plan_workload, resolve_tasks
    import tempfile

    exp, raw, machine, learner, editor = load_all(a.experiment, a.machines, a.set)
    notes = check_compatibility(exp, machine, learner, editor)
    with tempfile.TemporaryDirectory() as td:
        instances, panels, panel_split, instance_split, held_out = resolve_tasks(exp, Path(td))
        ctx = RunContext(Path(td), "validate", exp, raw, machine, learner, editor, instances, panels, panel_split, instance_split, held_out)
        wl = plan_workload(ctx)
        wl.notes = notes + wl.notes
        print(f"experiment {exp.name} (condition={exp.condition}, cycles={exp.cycles}) on machine {machine.name}")
        print(f"learner {learner.name}: {learner.base_model}@{learner.base_revision[:12]}  editor mode: {exp.editor.mode}")
        for p, ids in panels.items():
            print(f"panel {p:<24} split={panel_split[p].value:<6} instances={len(ids)}")
        print(wl.render())
    print("OK: configuration valid; nothing was started.")
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    from .orchestration.coordinator import create_run, run_all

    ctx = create_run(a.experiment, a.machines, run_id=a.run_id, overrides=a.set)
    print(f"run dir: {ctx.run_dir}")
    asyncio.run(run_all(ctx))
    print(f"done: {ctx.run_dir}")
    return 0


def cmd_resume(a: argparse.Namespace) -> int:
    from .orchestration.coordinator import resume_run

    resume_run(Path(a.run_dir), a.machines)
    return 0


def cmd_status(a: argparse.Namespace) -> int:
    from .orchestration.coordinator import run_status

    _print(run_status(Path(a.run_dir)))
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    from .reporting.report import write_run_report

    print(write_run_report(Path(a.run_dir)))
    return 0


def cmd_dashboard(a: argparse.Namespace) -> int:
    from .reporting.dashboard import Dashboard, serve

    if a.log and not a.run_dir:
        print("error: --log needs RUN_DIR (it replaces that run's coordinator log)", file=sys.stderr)
        return 2
    run_dir = Path(a.run_dir) if a.run_dir else None
    if run_dir is not None and not (run_dir / "run.json").is_file():
        print(f"error: {run_dir} is not a run directory (no run.json)", file=sys.stderr)
        return 2
    dash = Dashboard(runs_dir=repo_path(a.runs_dir), run_dir=run_dir, log=Path(a.log) if a.log else None)
    try:
        return serve(dash, a.host, a.port, open_browser=a.open)
    except OSError as e:
        print(f"error: cannot listen on {a.host}:{a.port} ({e.strerror or e}); try --port 0 or another port", file=sys.stderr)
        return 2


def cmd_compare(a: argparse.Namespace) -> int:
    from .reporting.report import compare_conditions, compare_runs

    out_dir = Path(a.out) if a.out else None
    if a.run_b is None:  # `loop compare RUN_A RUN_B`
        if len(a.run_a) != 2:
            raise SystemExit("usage: loop compare RUN_A RUN_B   or   loop compare A1 A2 ... --vs B1 B2 ...")
        a.run_a, a.run_b = a.run_a[:1], a.run_a[1:]
    if len(a.run_a) == 1 and len(a.run_b) == 1:
        print(compare_runs(Path(a.run_a[0]), Path(a.run_b[0]), out_dir=out_dir, panel=a.panel))
    else:
        # several independent loop seeds per condition (or one baseline run against several
        # seeds): per-seed paired effects (--vs minus positional), then spread
        res = compare_conditions([Path(p) for p in a.run_a], [Path(p) for p in a.run_b], out_dir=out_dir, panel=a.panel,
                                 label_a="baseline", label_b="experiment")
        print(res["markdown"])
    return 0


def cmd_stage(a: argparse.Namespace) -> int:
    from .orchestration import coordinator as co

    ctx = co.open_run(Path(a.run_dir), a.machines)

    async def go() -> None:
        from .core.storage import run_lock

        with run_lock(ctx.run_dir), ctx.pods():
            co.record_provenance(ctx.run_dir, ctx.run_id)
            try:
                await co.run_single_stage(ctx, a.cycle, a.stage)
            finally:
                ctx.stop_serving()

    asyncio.run(go())
    return 0


def cmd_evaluate(a: argparse.Namespace) -> int:
    from .orchestration import coordinator as co

    run_dir = co.evaluate_checkpoint(a.experiment, a.machines, a.checkpoint, a.panels, final=a.final, run_id=a.run_id, overrides=a.set)
    print(f"evaluation run: {run_dir}")
    return 0


def cmd_edit_replay(a: argparse.Namespace) -> int:
    from .orchestration import coordinator as co

    run_dir = co.edit_replay(Path(a.source_run), a.cycle, a.experiment, a.machines, run_id=a.run_id, overrides=a.set)
    print(f"editor-comparison run: {run_dir}")
    return 0


def cmd_external_eval(a: argparse.Namespace) -> int:
    from .orchestration.external_eval import build_external_job

    cfg_path, argv = build_external_job(a.dataset, a.checkpoint, a.machines, a.out, n_tasks=a.n_tasks, task_names=a.task, model_profile=a.model_profile)
    print(f"wrote Harbor job config: {cfg_path}")
    print("command:", " ".join(argv))
    if not a.execute:
        print("dry run: pass --execute to launch (downloads the pinned dataset; can be expensive)")
        return 0
    return subprocess.call(argv, cwd=REPO_ROOT)


def cmd_smoke(a: argparse.Namespace) -> int:
    from .orchestration.smoke import run_smoke

    return run_smoke(a.level, machines=a.machines, keep=a.keep)


def cmd_preflight(a: argparse.Namespace) -> int:
    from .hosts.preflight import main as preflight_main

    return preflight_main(["--profile", a.model_profile, *([] if a.trainable else ["--no-train"])])


def cmd_render_tasks(a: argparse.Namespace) -> int:
    from .tasks.instances import load_splits, materialize, validate_splits

    splits = load_splits(a.splits)
    instances = materialize(splits, repo_path(a.out), panels=a.panel or None, ids=a.id or None)
    for note in validate_splits(splits, instances):
        print(f"note: {note}", file=sys.stderr)
    for iid, inst in instances.items():
        print(f"{iid}\t{inst.task_dir}")
    return 0


def cmd_submit(a: argparse.Namespace) -> int:
    from .hosts.remote_jobs import submit

    print(submit(a.experiment, a.machines, run_id=a.run_id))
    return 0


def cmd_sync_hosts(a: argparse.Namespace) -> int:
    from .hosts.remote_jobs import sync_hosts

    for line in sync_hosts(a.machines, dry_run=a.dry_run):
        print(line)
    return 0


def cmd_pod(a: argparse.Namespace) -> int:
    from .hosts.pods import cleanup, client_for, pod_status

    m = load_machine(a.machines)
    if not m.pod_ids():
        print("error: this machine profile has no kind: runpod hosts", file=sys.stderr)
        return 2
    if a.action == "cleanup":
        for line in cleanup(m, dry_run=a.dry_run) or ["no created pods are recorded as alive"]:
            print(line)
        return 0
    if a.action in ("start", "stop"):
        if not m.existing_pod_ids():
            print("error: start/stop apply to existing pods (pod_id); created pods live only for one command", file=sys.stderr)
            return 2
        client = client_for(m.runpod)
        for pod in m.existing_pod_ids():
            (client.start if a.action == "start" else client.stop)(pod)
            print(f"{pod}: {a.action} requested")
    for row in pod_status(m):
        print(f"{row['pod_id']}: {row['status']}  ssh: {row['address'] or '-'}  gpu: {row['gpu'] or '-'}")
    return 0


def cmd_fetch(a: argparse.Namespace) -> int:
    from .hosts.remote_jobs import fetch

    print(fetch(a.run_id, a.machines))
    return 0


def cmd_remote_status(a: argparse.Namespace) -> int:
    from .hosts.remote_jobs import remote_status

    _print(remote_status(a.run_id, a.machines))
    return 0


SET_HELP = "override an experiment value, e.g. --set cycles=1 (YAML-parsed; repeatable; recorded in run.json)"

# Exit codes shared by every command (documented in docs/cli.md).
EXIT_CODES = {
    0: "success",
    1: "unexpected error (traceback printed), a failed smoke/preflight check, or a failed stage item",
    2: "invalid usage, configuration or plan (message printed as `error: ...`); nothing was started",
}


def _cmd(sub, name: str, fn, help: str, description: str) -> argparse.ArgumentParser:
    s = sub.add_parser(name, help=help, description=f"{help}.\n\n{description}", formatter_class=argparse.RawDescriptionHelpFormatter)
    s.set_defaults(fn=fn)
    return s


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="loop", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    # --- experiments -------------------------------------------------------- #
    s = _cmd(sub, "validate", cmd_validate, "Resolve configs and print the planned workload",
             "Loads the experiment (with `base:` and --set), the machine profile and the model profiles, checks\n"
             "compatibility (serving/adapter formats, external endpoint identities, panels and splits, that the\n"
             "initial checkpoint was trained for the learner's model profile), materializes\n"
             "task instances in a temporary directory, and prints panels and the per-cycle workload.\n"
             "Side effects: none (no run directory, no model, no Docker).")
    s.add_argument("experiment", help="experiment YAML (experiments/*.yaml)")
    s.add_argument("--machines", required=True, help="machine profile YAML (configs/machines/...)")
    s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help=SET_HELP)

    s = _cmd(sub, "run", cmd_run, "Create a run and execute all cycles",
             "Creates runs/<run-id>/ (run.json, provenance.json, tasks/), then runs every cycle: eval, collect,\n"
             "edit, verify, dataset, train; the last cycle only evaluates. Writes reports/ at the end.\n"
             "Side effects: Docker task containers; starts/stops model servers it owns (managed inference);\n"
             "training subprocesses or SSH jobs. Holds the run lock for its whole duration.")
    s.add_argument("experiment", help="experiment YAML")
    s.add_argument("--machines", required=True, help="machine profile YAML")
    s.add_argument("--run-id", help="run directory name (default: <experiment>-<UTC-stamp>-s<loop_seed>)")
    s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help=SET_HELP)

    s = _cmd(sub, "resume", cmd_resume, "Resume a run from its manifests",
             "Re-enters the stages of the run's kind (learning, evaluation or edit_replay). Completed work items\n"
             "are skipped; interrupted ones keep their directory as *.interrupted-N and run again in a fresh\n"
             "environment; infra_failed items stay terminal. Appends this invocation to invocations.jsonl.\n"
             "Side effects: as for `run`.")
    s.add_argument("run_dir", help="runs/<run-id>")
    s.add_argument("--machines", help="use a different machine profile (e.g. new ports/hosts); validated and "
                   "recorded in machine-overrides.jsonl")

    s = _cmd(sub, "stage", cmd_stage, "Run one stage of one cycle inside an existing run",
             "Runs eval, collect, edit, verify, dataset or train for one cycle using the earlier stages' outputs\n"
             "on disk and the same manifests as the full loop, so a later `resume` skips what was done here.\n"
             "Does not update cycle.json. Holds the run lock. Appends this invocation to invocations.jsonl.")
    s.add_argument("run_dir", help="runs/<run-id>")
    s.add_argument("--cycle", type=int, required=True, help="cycle number (0-based)")
    s.add_argument("--stage", required=True, choices=["eval", "collect", "edit", "verify", "dataset", "train"], help="stage to run")
    s.add_argument("--machines", help="use a different machine profile (validated and recorded)")

    s = _cmd(sub, "evaluate", cmd_evaluate, "Evaluate a saved checkpoint on declared panels",
             "Creates a separate run of kind `evaluation` (cycles/cycle-000/eval/ only) using the experiment's\n"
             "episode settings, splits and evaluation seed schedule, so results pair with learning runs.\n"
             "The learning loop never reads evaluation runs. Final-test panels require --final.\n"
             "Side effects: Docker; serving the checkpoint (managed inference).")
    s.add_argument("--experiment", required=True, help="supplies episode settings, splits and seeds")
    s.add_argument("--machines", required=True, help="machine profile YAML")
    s.add_argument("--checkpoint", required=True, help="'base' or a published checkpoint directory (runs/<id>/checkpoints/<ckpt>)")
    s.add_argument("--panels", nargs="+", required=True, help="panel names from the experiment's split file")
    s.add_argument("--final", action="store_true", help="allow final-test/external panels (use only after method decisions are frozen)")
    s.add_argument("--run-id", help="run directory name (default: <experiment>-eval-<UTC-stamp>)")
    s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help=SET_HELP)

    s = _cmd(sub, "edit-replay", cmd_edit_replay, "Edit and verify saved source trajectories with another editor",
             "Creates a run of kind `edit_replay` that reads one cycle's successful collection episodes from a\n"
             "source run (in place, read-only) and runs this experiment's editor, verification and dataset export\n"
             "on them with the learner that produced them. No collection, no training. Used for controlled editor\n"
             "comparisons.")
    s.add_argument("source_run", help="runs/<source-run-id>")
    s.add_argument("--cycle", type=int, default=0, help="source cycle whose collection episodes are edited (default 0)")
    s.add_argument("--experiment", required=True, help="experiment supplying the editor/verification settings")
    s.add_argument("--machines", required=True, help="machine profile YAML")
    s.add_argument("--run-id", help="run directory name (default: <experiment>-editreplay-<UTC-stamp>)")
    s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help=SET_HELP)

    # --- inspection ----------------------------------------------------------- #
    s = _cmd(sub, "status", cmd_status, "Print per-cycle stage status",
             "Reads cycle.json and stage manifests: status, item counts by state, attempts and infra failures,\n"
             "learner in/out per cycle. Side effects: none.")
    s.add_argument("run_dir", help="runs/<run-id>")

    s = _cmd(sub, "report", cmd_report, "Regenerate the run's CSV and markdown reports",
             "Rewrites runs/<run-id>/reports/ from manifests, summaries, proposals, verifications, dataset\n"
             "manifests and cycle.json. Safe to run at any time.")
    s.add_argument("run_dir", help="runs/<run-id>")

    s = _cmd(sub, "compare", cmd_compare, "Paired comparison of runs on shared panels",
             "Two runs (`loop compare A B`): episodes paired by panel/instance/attempt/seed; full success\n"
             "transition counts; token deltas only where both succeed, with coverage; tokens withheld when model\n"
             "identities or usage sources differ. Several runs per side (`loop compare A1 A2 --vs B1 B2`): runs\n"
             "matched by loop seed, per-seed paired effects first, then spread across seeds; a single run on either\n"
             "side (e.g. one frozen baseline: `loop compare BASE --vs L1 L2 L3`) is compared with every run of the\n"
             "other. Every effect is the --vs (experiment) side minus the positional (baseline) side. Prints\n"
             "markdown; never writes into the compared runs.")
    s.add_argument("run_a", nargs="+", help="baseline run(s); with exactly two runs and no --vs, the second is the experiment")
    s.add_argument("--vs", dest="run_b", nargs="+", required=False, help="experiment run(s)")
    s.add_argument("--panel", help="restrict to one panel")
    s.add_argument("--out", help="also write the comparison files to this directory")

    s = _cmd(sub, "dashboard", cmd_dashboard, "Watch runs live in a local web dashboard",
             "Serves a read-only HTML view of RUN_DIR, or without it an index of every run under --runs-dir (newest\n"
             "first: kind, status, current cycle and stage, and whether the coordinator PID recorded in the run's\n"
             ".lock is alive; the lock is never taken). A run page shows the current cycle and stage, the\n"
             "unassisted evaluation across cycles (charts, per-cycle table, deltas against cycle 0 on matched\n"
             "items, paired comparison, family/difficulty breakdowns), the cycle timeline, recent episodes with\n"
             "transcript pages, proposals and verifications, pod/serving events and the coordinator log.\n"
             "Numbers use the definitions of `loop report`. Pages reload every 10 s. Never writes, locks or\n"
             "modifies a run, so it is safe next to a running `loop run`. Runs in the foreground until Ctrl-C.")
    s.add_argument("run_dir", nargs="?", help="runs/<run-id> to show (default: an index of the runs under --runs-dir)")
    s.add_argument("--runs-dir", default="runs", help="directory whose runs the index lists (relative to the repository root)")
    s.add_argument("--log", help="file to tail as the coordinator log, e.g. saved `loop run` console output (default: the "
                   "run's logs/coordinator.log, else its coordinator.log, where older `loop submit` runs wrote it); "
                   "needs RUN_DIR")
    s.add_argument("--port", type=int, default=8090, help="port to listen on (0 picks a free port)")
    s.add_argument("--host", default="127.0.0.1", help="address to bind; anything but loopback exposes run outputs and "
                   "transcripts to the network")
    s.add_argument("--open", action="store_true", help="open the dashboard in the default browser")

    # --- other evaluations ---------------------------------------------------- #
    s = _cmd(sub, "external-eval", cmd_external_eval, "Evaluate a checkpoint on a version-pinned Harbor dataset",
             "Writes <out>/<dataset>-<version>-<ckpt>/job.yaml and protocol.json (agent protocol: this repository's\n"
             "ToolAgent; subset; checkpoint identity) and prints the Harbor command. Dry run by default; --execute\n"
             "runs it (downloads the dataset; can be expensive). With managed inference, start the server for the\n"
             "checkpoint yourself first. Results are never read by the learning loop.")
    s.add_argument("--dataset", required=True, help="name@version, e.g. terminal-bench@2.0 (unpinned versions are refused)")
    s.add_argument("--checkpoint", required=True, help="'base' or a published checkpoint directory")
    s.add_argument("--model-profile", help="required with --checkpoint base (resolves the exact base identity)")
    s.add_argument("--machines", required=True, help="machine profile YAML (its inference endpoint is used)")
    s.add_argument("--out", default="runs/external", help="output root (default runs/external)")
    s.add_argument("--n-tasks", type=int, help="evaluate only the first N tasks of the dataset")
    s.add_argument("--task", action="append", help="explicit task subset (repeatable)")
    s.add_argument("--execute", action="store_true", help="launch Harbor instead of printing the command")

    s = _cmd(sub, "render-tasks", cmd_render_tasks, "Render a split's task instances into Harbor task directories",
             "Renders every instance of the split file (or only the given panels and ids) into OUT/<id>, one\n"
             "Harbor task directory each, exactly as a run materializes them, and validates the split. An\n"
             "existing instance directory is reused when its family version is current and refused otherwise.\n"
             "Point `harbor run -p OUT` at the result to run tasks outside the loop.")
    s.add_argument("splits", help="split file, e.g. evaluation/splits/pilot.yaml")
    s.add_argument("--out", default="evaluation/rendered", help="output directory (default evaluation/rendered)")
    s.add_argument("--panel", action="append", help="render only this panel's instances (repeatable)")
    s.add_argument("--id", action="append", help="render only this instance id (repeatable)")

    # --- checks ------------------------------------------------------------- #
    s = _cmd(sub, "smoke", cmd_smoke, "Run a bounded smoke test",
             "fixture: scripted learner/editor + fixture trainer, local fixture backend, 2 cycles (seconds).\n"
             "fixture-docker: the same on Harbor Docker containers (minutes).\n"
             "train: real LoRA DPO on labeled fixture pairs (Qwen3-0.6B), publish + reload check, serve the\n"
             "adapter, one real Docker task episode (success not required).\n"
             "live: experiments/smoke-mac.yaml with the real small learner/editor.\n"
             "Outputs go to runs/_smoke/; fixture runs are deleted on success unless --keep. Engineering checks\n"
             "only, never evidence of learning.")
    s.add_argument("level", choices=["fixture", "fixture-docker", "train", "live"], help="which smoke test")
    s.add_argument("--machines", help="machine profile (default: examples/fixture*.yaml or examples/mac-local.yaml)")
    s.add_argument("--keep", action="store_true", help="keep fixture smoke run directories")

    s = _cmd(sub, "preflight", cmd_preflight, "Check the local environment for the smoke test",
             "Checks host, Docker daemon and architecture, free disk, memory, that the model profile's pinned\n"
             "model is cached with the pinned chat template, and the accelerator. Exits 1 if a check fails.\n"
             "Does not download models.")
    s.add_argument("--model-profile", default="qwen3-0.6b", help="model profile to check (default qwen3-0.6b)")
    s.add_argument("--trainable", action="store_true", help="also load a tiny model and run one LoRA training step on the device")

    s = _cmd(sub, "docs-gen", cmd_docs_gen, "Regenerate or check the generated documentation",
             "Renders the reference parts of docs/cli.md from this parser and of docs/configuration.md from\n"
             "the config schemas (only between the generated-region markers). --check exits 1 if either file\n"
             "is out of date (the test suite runs the same check).")
    s.add_argument("--check", action="store_true", help="do not write; fail if the files differ from the generated text")

    s = _cmd(sub, "docs", cmd_docs, "Browse the documentation in a local web server",
             "Serves the repository's Markdown as HTML (starting at docs/index.md) with navigation, per-page\n"
             "contents and working links between documents; files are re-read on every request, so edits\n"
             "show on reload. Other repository text files are shown as source; run outputs, virtualenvs,\n"
             "private machine profiles and hidden directories are never served. Mermaid diagrams load\n"
             "mermaid.js from a CDN. Runs in the foreground until Ctrl-C.")
    s.add_argument("--port", type=int, default=8000, help="port to listen on (0 picks a free port)")
    s.add_argument("--host", default="127.0.0.1", help="address to bind; anything but loopback exposes the documents to the network")
    s.add_argument("--open", action="store_true", help="open the docs in the default browser")

    # --- remote ----------------------------------------------------------------- #
    s = _cmd(sub, "submit", cmd_submit, "Start a run on the SSH coordinator in the machine profile",
             "Validates locally, asks the host whether the run id exists or is locked (refuses either), rsyncs\n"
             "the code checkout (not runs/, artifacts/ or private profiles), copies the machine profile into the\n"
             "run, and starts `loop run` detached. Records the remote PID in runs/_submissions/<run-id>.json.")
    s.add_argument("experiment", help="experiment YAML (path inside the repository)")
    s.add_argument("--machines", required=True, help="machine profile whose coordinator is an SSH host")
    s.add_argument("--run-id", help="run id (default: <experiment>-<UTC-stamp>-s<loop_seed>)")

    s = _cmd(sub, "sync-hosts", cmd_sync_hosts, "Prepare the SSH hosts of a machine profile",
             "For every SSH host named in the machine profile (coordinator, inference, editor inference,\n"
             "training): creates the workdir, rsyncs the code checkout (not runs/, artifacts/ or private\n"
             "profiles) and runs `uv sync --frozen` there, with the `train` extra on GPU roles. Run it once per\n"
             "new or restarted host (e.g. a Runpod pod); runs also push code before serving and training.\n"
             "Requires uv and rsync on the host (see docs/runpod.md).")
    s.add_argument("--machines", required=True, help="machine profile YAML")
    s.add_argument("--dry-run", action="store_true", help="only list which hosts would be prepared and how; contact nothing")

    s = _cmd(sub, "pod", cmd_pod, "Show, start, stop or clean up the Runpod pods of a machine profile",
             "status (read-only): state and current SSH address of the profile's existing pods and of every pod\n"
             "`loop` created that its ledger (artifacts/runpod/created.jsonl) still lists as alive.\n"
             "start / stop: start or stop the existing pods (`pod_id`) by hand (runs do this automatically).\n"
             "cleanup: terminate created pods that are still alive (e.g. after the laptop died mid-command); only\n"
             "pods in the ledger whose name starts with `lfe-` are touched.\n"
             "Needs the API key variable named in the profile (e.g. RUNPOD_API_KEY in .env).")
    s.add_argument("action", choices=["status", "start", "stop", "cleanup"], help="what to do")
    s.add_argument("--machines", required=True, help="machine profile with kind: runpod hosts")
    s.add_argument("--dry-run", action="store_true", help="cleanup only: list what would be terminated")

    s = _cmd(sub, "fetch", cmd_fetch, "Copy a remote run directory back",
             "rsyncs runs/<run-id>/ from the SSH coordinator into the local runs/.")
    s.add_argument("run_id", help="run id")
    s.add_argument("--machines", required=True, help="machine profile whose coordinator is an SSH host")

    s = _cmd(sub, "remote-status", cmd_remote_status, "Show whether a remote run is still running",
             "Reports the remote run lock (locked | free | missing), whether the submitted PID is alive, and the\n"
             "remote `loop status`. Use it after a lost SSH session before resuming anything.")
    s.add_argument("run_id", help="run id")
    s.add_argument("--machines", required=True, help="machine profile whose coordinator is an SSH host")
    return p


def cmd_docs_gen(a: argparse.Namespace) -> int:
    from .docs_tools.docgen import check_docs, write_docs

    if a.check:
        stale = check_docs()
        for path in stale:
            print(f"out of date: {path} (run `uv run loop docs-gen`)", file=sys.stderr)
        return 1 if stale else 0
    for path in write_docs():
        print(f"wrote {path}")
    return 0


def cmd_docs(a: argparse.Namespace) -> int:
    from .docs_tools.docserver import serve

    try:
        return serve(a.host, a.port, open_browser=a.open)
    except OSError as e:
        print(f"error: cannot listen on {a.host}:{a.port} ({e.strerror or e}); try --port 0 or another port", file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    from pydantic import ValidationError

    from .core.storage import RunLockedError
    from .hosts.pods import RunpodError
    from .orchestration.coordinator import PlanError

    from .core.envfile import load_env

    load_env()  # repo-root .env (git-ignored); existing environment variables win
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except (PlanError, ValidationError, FileNotFoundError, FileExistsError, ValueError, RunpodError, RunLockedError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main", "build_parser", "load_machine", "repo_path"]
