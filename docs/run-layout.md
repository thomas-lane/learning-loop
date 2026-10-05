# Run directory layout

Each run writes to one git-ignored directory, `runs/<run-id>/` (`runs/` is the machine profile's
`runs_dir`). Paths inside are relative where possible, so a run can be copied between machines
(`loop fetch`). How the files are produced: [architecture.md](architecture.md); terms:
[glossary.md](glossary.md).

| Label | Meaning |
|---|---|
| **write-once** | rewriting with different content is refused, so the inputs a run started with cannot change under it |
| **atomic** | replaced whole (temp file + rename), so a crash never leaves a half-written file for a resume to read |
| **append-only** | JSONL that only grows, one fsynced line per record; a torn last line from a crash is ignored |
| **read-only** | files are mode `0444` and never overwritten, so data that later records point to cannot change |

## Learning run

Run id: `<experiment>-<UTC stamp>-s<loop_seed>`, or `--run-id`.

```text
runs/<run-id>/
├── .lock                     run lock (see below)
├── run.json                  write-once: resolved configs and inputs (see below)
├── provenance.json           write-once: code version, packages, uv.lock hash, hardware, argv
├── invocations.jsonl         append-only: provenance of each `loop resume` and `loop stage`, or repeated command with this `--run-id`
├── machine-overrides.jsonl   append-only: profiles passed to `loop resume/stage --machines`
├── seed_schedule.json        evaluation seeds per instance and attempt
├── tasks/
│   ├── instances.json        task instance records, including content hashes
│   └── <instance-dir>/       task directory: instance id with `/` -> `__` (e.g. `count-errors__easy__s1`)
├── cycles/cycle-NNN/
│   ├── cycle.json            atomic: what the cycle did (see below)
│   ├── eval/                 evaluation episodes on the dev panels (every cycle, or first and last)
│   ├── collect/              training-panel episodes (the editor's sources)    ┐
│   ├── edit/                 edit proposals                                    │ `learning`
│   ├── verify/               branch comparisons                                │ condition
│   ├── audit/                re-verification of a sample of accepted edits     │ only
│   │                         (when `verification.audit.fraction` > 0)          ┘
│   ├── dataset/              preference pairs
│   └── train/                training stage files
├── preflight/<start time>/   Harbor trial directories of each Docker-host preflight
├── checkpoints/cNNN-<hash12>/   read-only published checkpoints
├── cache/ref_logps/          DPO reference log-probs, reused on resume (trl_dpo; remote training keeps them in its work dir)
├── logs/
│   ├── coordinator.log       console output of a coordinator started by `loop submit`
│   ├── <role>-server-<checkpoint>-<unix time>.log   model server output, including vLLM's (copied back from remote hosts)
│   ├── serving-lifecycle.jsonl  append-only: model server start/ready/stop events
│   ├── preflight.jsonl       append-only: one Docker-host preflight record per coordinator start that runs episodes (passed, failed or skipped, with each family's trial)
│   └── pod-lifecycle.jsonl   append-only: Runpod pod events
└── reports/                  CSV and markdown reports (see below)
```

- `.lock`: every command that runs stages holds an exclusive `flock` on it, so two coordinators
  never write one run. The file stays after the command ends and names the last holder (`pid`,
  `host`, `since`); a leftover file does not block the next command.
- `run.json`: experiment (with `--set` overrides), machine and model profiles, serving setup
  (including whether base checkpoints are served through a [zero LoRA](glossary.md)), initial
  checkpoint, panels, splits, held-out families and instances.
- `cycle.json`: `status`, `learner_in`, `learner_out`, `reference` (the DPO reference), `update`
  (`trained`, `no_update`, `final_evaluation_only`), `dataset` (the dataset manifest),
  `training_data` (exported, trained and dropped pairs with reasons) and `checkpoint_record`.
- The last cycle (`cycle-<cycles>`) has only `eval/` and `cycle.json`. A
  [no-update cycle](glossary.md) has `dataset/`, and `train/` only when pairs were exported but the
  trainer dropped all of them.

## Stage directories

`eval/`, `collect/`, `edit/`, `verify/` and `audit/` share one shape:

```text
<stage>/
├── manifest.json                  atomic: stage status and per-item status
└── items/
    ├── <item-id>/                 output of the successful (or latest) execution
    └── <item-id>.interrupted-N/   earlier interrupted or infra-failed execution, kept as-is
```

Each manifest item has a `status` (`pending`, `running`, `done`, `failed`, `infra_failed`),
`attempts`, `infra_failures`, `output`, `interrupted_dirs` and `meta`. A resume never reruns
`done`, `failed` or `infra_failed` items, so counts that later stages used stay fixed
([reopening failed items](operations.md#reopening-failed-items)). Earlier executions are renamed
rather than deleted, so their token spend still counts in the report. `active_intervals` records
each execution's start and end, so durations exclude time spent interrupted.

### Episode item (`eval/`, `collect/`)

```text
items/ep-<hash>/
├── plan.json              inputs: policy, seed, budgets, prompts, tools, task state spec
├── summary.json           outcome: success, reward, stop reason, token usage, timing (the copy the loop reads)
├── agent_plan.json        plan handed to the Harbor agent (Docker)
├── episode/               authoritative record, on the host and never mounted into the container
│   ├── events.jsonl       append-only complete event log
│   ├── episode.json       episode as written by the agent loop
│   ├── summary.json       the same summary, written by the backend
│   ├── messages.json      final message history and tool schemas (a view, not the record)
│   ├── trajectory.json    Harbor ATIF view for `harbor view` (Docker)
│   └── grading.json       local grader output (local fixture backend)
└── harbor/<trial>/        Harbor's trial directory (Docker)
    ├── result.json, config.json, trial.log, lock.json
    ├── agent/             copies of episode files; the container can write here, so the loop never reads it
    ├── artifacts/         declared artifacts copied to the verifier
    └── verifier/          reward.json or reward.txt, test-stdout.txt
```

### Edit item (`edit/`)

`items/prop-<hash>/proposal.json`: `status` (`proposed`, `abstained`, `invalid`), `turn_index`,
`replacement` (the proposed call), `rejection_reasons`, `raw_response`, `usage`, and the editor's
`justification` (never used for training). From a model editor, `raw_response` holds the request
messages, seed and model response, plus `answer_recovery: bare_tool_call` when the answer call
was accepted although written without the model format's call markers. A source with no editable
turn is recorded as `abstained` with the reason `skipped:no_editable_turns` and no `raw_response`
or `usage`, because no request was made ([editing](experiment.md#editing)).

### Verify and audit items (`verify/`, `audit/`)

```text
items/ver-<hash>/
├── verification.json      per-branch results and costs, accepted, reasons,
│                          evidence label, purpose (acceptance or audit)
├── original-r00/          branch episode: `episode/`, plus `agent_plan.json` and `harbor/` on Docker
├── edited-r00/
└── ...-rNN/               one pair per continuation repetition
```

A proposal rejected before any branch runs has only `verification.json`.

## Dataset

```text
dataset/                   read-only files; re-exporting different content is refused
├── manifest.json          written last: n_examples, dataset_sha256, composition, kinds (`verified` | `fixture`)
├── preferences.jsonl      training rows only: {pair_id, prompt, chosen, rejected, tools}
├── provenance.jsonl       evidence per pair_id: instance, source episode, proposal, verification, saving
└── current/               same three files, this cycle's accepted pairs only
```

The trainer reads `dataset/` (this cycle's pairs plus sampled history from earlier cycles'
`current/`; see [experiment.md](experiment.md)). In a `fixed_dataset` run, `preferences.jsonl`
and `provenance.jsonl` are byte copies of the frozen export, the manifest records its source, and
there is no `current/`.

## Training stage and checkpoints

```text
train/
├── request.json           write-once: dataset, incoming checkpoint, training config, seed
├── train.log              trainer stderr (local training)
├── checkpoint_path.txt    published checkpoint's directory, written after the lineage check;
│                          marks the stage done
├── remote_request.json    request with remote paths      ┐
├── remote_launch.json     remote trainer PID and host    │ remote training only
├── remote-train.log       trainer log, copied back       │
├── relaunches.jsonl       append-only: relaunches after  │
│                          the trainer was lost           ┘
└── work/                  trainer working directory (resume state)
    ├── .trainer.lock      one trainer per work directory
    ├── request.json, result.json   result status: published | no_trainable_examples | invalid_request | failed
    ├── render_report.json kept and dropped examples, with reasons (trl_dpo)
    ├── train_logs.jsonl   per-step metrics (trl_dpo)
    ├── trainer/checkpoint-N/   optimizer, scheduler and RNG state (trl_dpo)
    └── fixture_state.json resume state (fixture trainer)

checkpoints/cNNN-<hash12>/   read-only, published by atomic rename
├── adapter_model.safetensors, adapter_config.json, README.md   LoRA adapter (trl_dpo)
├── fixture_adapter.json   labeled stand-in adapter (fixture trainer)
└── checkpoint.json        lineage, adapter hash, dataset hash, config, steps, seeds, reload check
```

## Reports

`reports/` is written when `loop run`, `resume`, `evaluate` or `edit-replay` finishes, and by
`loop report`. It is regenerated from the run and can be deleted at any time.

| File | Content |
|---|---|
| `report.md` | readable summary of everything below, plus lineage and a first-vs-last comparison |
| `episodes.csv` | one row per eval or collect episode (not branches) |
| `summary_by_*.csv` | grouped by checkpoint/cycle/panel, family, difficulty, skill, role/cycle |
| `stage_effort.csv` | usage, active time, optimizer steps, retries per stage |
| `proposals.csv`, `verifications.csv`, `branch_costs.csv` | editing and verification detail |
| `training_data.csv` | per cycle: exported, trained and dropped pairs, `max_length`, token lengths |

`loop compare` prints or writes to `--out`; it never writes into either run.

## Other run kinds

`run.json` `kind` decides which stages `loop resume` re-enters.

| Kind | Created by | Default run id | Contents |
|---|---|---|---|
| `learning` (no `kind` field) | `loop run` | see above | the layout above |
| `evaluation` | `loop evaluate` | `<experiment>-eval-<stamp>` | `cycles/cycle-000/eval/` only (no `cycle.json`); `run.json` names the checkpoint and panels |
| `edit_replay` | `loop edit-replay` | `<experiment>-editreplay-<stamp>` | `cycles/cycle-000/{edit,verify,dataset}/`; `imported_sources` in `run.json` names the source run, read in place |

## Other files

On the machine running `loop` (paths relative to the repository root, independent of `runs_dir`):

| Path | Content |
|---|---|
| `runs/_smoke/` | `loop smoke` output: `fixture-<stamp>/` (deleted on success unless `--keep`), `train-<stamp>/`, `live-<stamp>/` |
| `runs/_submissions/<run-id>.json` | `loop submit` record: host, work directory, remote PID |
| `runs/_pods/pod-lifecycle.jsonl` | pod events from `loop sync-hosts` (no run directory) |
| `runs/external/<dataset>-<version>-<checkpoint>/` | `loop external-eval` (default `--out`): `job.yaml`, `protocol.json`, Harbor `jobs/` |
| `artifacts/runpod/<pod>.known_hosts` | pod SSH host key, reset each time a command reaches the pod |
| `artifacts/runpod/created.jsonl` | append-only ledger of created pods; the only pods `loop` may terminate |

On a remote training or inference host (same run id; copies or scratch, so losing them loses no
results):

| Path | Content |
|---|---|
| `runs/<run-id>/remote-train/cNNN/` | request, dataset copy, work directory, `train.log` |
| `runs/<run-id>/checkpoints/` | incoming checkpoints pushed there and new ones published there |
| `runs/_servers/` | model server logs; with vLLM, also `zero-lora-<hash>/`, the [zero LoRA](glossary.md) served for base checkpoints |
| `.engines/<package>/` | vLLM only: the engine's own environment (vLLM pins its own torch), installed on first use; `vllm==0.30.0` becomes `vllm__0.30.0` |
| `runs/_pod/` | Runpod only: heartbeat, watchdog log and PID |

When the machine running `loop` serves with vLLM itself, it has `.engines/<package>/` and
`runs/_servers/zero-lora-<hash>/` too.

On a coordinator host used through `loop submit`, `runs/<run-id>/` is the run directory itself,
plus `submitted-machine.yaml`, and the coordinator's console output goes to `logs/coordinator.log`,
which `loop dashboard` tails; `loop fetch` copies it back.

## Finding things

```bash
uv run loop status runs/<id>
jq . runs/<id>/cycles/cycle-000/cycle.json
jq '.items[] | {item_id, status, meta}' runs/<id>/cycles/cycle-000/collect/manifest.json
jq '{accepted, reasons, evidence_label}' runs/<id>/cycles/cycle-000/verify/items/*/verification.json
```
