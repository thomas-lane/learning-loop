# Operations and troubleshooting

How to run the system in practice: the order of steps for a real experiment, the laptop smoke
path, and what to do when something goes wrong. Commands are documented in [cli.md](cli.md),
configuration in [configuration.md](configuration.md), run contents in [run-layout.md](run-layout.md).

## Running a real experiment

1. **Deployment profile.** For a Runpod GPU pod, follow [runpod.md](runpod.md) first. Copy the
   closest example from `configs/machines/examples/` to `configs/machines/local/<name>.yaml`
   (git-ignored), and put credentials in `.env` (from `.env.example`). Fixed SSH hosts are aliases
   from `~/.ssh/config`; for a model server on such a host, keep an SSH tunnel open to the
   declared `api_base` (Runpod pods get their tunnel from the run).
2. **Check the hosts.** `uv run loop preflight --trainable --model-profile <profile>` on the machine
   that will train, and the fast plus Docker test suites on the coordinator. On a new GPU host run
   `uv run pytest -m train tests/train` once (validated so far on Apple MPS and one RTX 3090).
3. **Validate every configuration you intend to run**, including controls:
   `uv run loop validate <experiment> --machines <profile>`. Read the printed notes (untested
   backends, fixture components).
4. **Calibrate on the initial checkpoint.** Run the frozen baseline
   (`experiments/frozen-baseline.yaml`, adjusted to your learner) to see which dev instances are
   easy, intermittent, or never solved under the declared budget. Tasks with zero observed
   successes are reported as such, not as unsolvable. Any tuning uses dev panels only.
5. **Run the conditions with several loop seeds** (same `seeds.root`, so evaluation is paired):
   ```bash
   for s in 0 1 2; do
     uv run loop run experiments/pilot.yaml --machines M --set seeds.loop_seed=$s --run-id pilot-s$s
   done
   uv run loop run experiments/frozen-baseline.yaml --machines M --run-id baseline
   uv run loop run experiments/fixed-dataset-control.yaml --machines M --run-id fixed-s0 \
       --set training.fixed_dataset=runs/pilot-s0/cycles/cycle-000/dataset
   ```
   The fixed-dataset control uses the same optimizer steps per cycle; its missing
   collection/editing cost is reported, not assumed equal.
6. **Monitor each cycle** (see [Watching a run](#watching-a-run)). `loop dashboard`, `loop status`
   and `loop report`: check proposal rejection reasons, acceptance reasons, infra retries, and the
   success transition counts (lost successes are listed explicitly, never hidden by cheaper
   surviving episodes).
7. **Compare.** `uv run loop compare runs/baseline --vs runs/pilot-s0 runs/pilot-s1 runs/pilot-s2`.
   Token deltas are withheld when model identities or usage sources differ.
8. **Freeze, then evaluate final panels.** Write down the method and checkpoint-selection
   decisions first, then run `loop evaluate ... --final` on the final panels (same-family and
   held-out family). Optional external benchmarks: `loop external-eval` (a dry run until
   `--execute`). Neither feeds back into any run.
9. **Keep the evidence.** `loop fetch` remote runs; archive whole run directories (they are
   self-contained and relocatable).

Editor comparisons reuse saved trajectories: `loop edit-replay runs/pilot-s0 --cycle 0
--experiment experiments/ablations/editor-external.yaml --machines M`.

## Watching a run

- **Live page.** `uv run loop dashboard runs/<id> --open` serves a read-only page at
  `http://localhost:8090/` that reloads every 10 s (paused by its checkbox or while a section is
  expanded): current cycle and stage with ETAs, the unassisted evaluation across cycles (charts,
  per-cycle table, deltas against cycle 0 on matched instance/attempt/seed items, paired success
  transitions), the cycle timeline, recent episodes with full transcripts and grader output,
  proposals and verifications, pod and serving events, and estimated pod spend. Without a run
  directory, `uv run loop dashboard` lists every run under `runs/` with its status and whether its
  coordinator process is alive. It only reads files (never the run lock), so it is safe next to a
  running `loop run`, and works the same on a run copied back with `loop fetch`.
- **Coordinator log.** `loop run` prints its progress to the console only. To see it on the page,
  save it (`uv run loop run ... 2>&1 | tee pilot.log`) and start the dashboard with
  `--log pilot.log`; without `--log` the page shows `runs/<id>/logs/coordinator.log` if that file
  exists.
- **Reading the numbers.** Success, token means and stop categories use the definitions of
  `loop report` (ungraded episodes count as not successful; episodes without usage are left out of
  token means). A running cycle's numbers cover only its finished episodes, so charts leave it out
  until it finishes. Each cycle has few evaluation episodes: the page carries no intervals, and
  `loop report` / `loop compare` give the instance-level ones.
- **Text views.** `uv run loop status runs/<id>` prints per-cycle stage counts, attempts and infra
  failures as JSON; `uv run loop report runs/<id>` rewrites `runs/<id>/reports/` (safe at any time).

| Symptom | Cause | Action |
|---|---|---|
| `error: cannot listen on 127.0.0.1:8090 (...)` | another process (e.g. an earlier dashboard) uses the port | `--port 0` picks a free port, or choose another |
| `error: ... is not a run directory (no run.json)` | the path is not `runs/<id>` (or the run was never created) | pass the run directory; omit it to list all runs |
| `error: --log needs RUN_DIR` | `--log` names one run's console log | add the run directory |
| status "coordinator not running" while cycles are unfinished | the PID in the run's `.lock` has exited (crash, Ctrl-C, reboot) | `loop resume runs/<id>` |
| coordinator "unknown" | the lock holder ran on another host (SSH coordinator) or no coordinator ever held the lock | `loop remote-status <run-id> --machines M` for remote runs |
| a section shows a traceback | a file it reads has an unexpected shape | the rest of the page still renders; the files themselves are never changed |

## Laptop smoke path

In increasing cost: `loop smoke fixture` (seconds), `loop smoke fixture-docker` (minutes),
`loop smoke train` (about two minutes of MPS training plus one Docker episode), `loop smoke live`
(the real small learner; 10-20 minutes). Stop there: do not leave longer training or load tests
running on a fanless laptop. The small Qwen models have so far solved none of the smoke task's
attempts, so live cycles end as no-updates; `README.md` shows how to exercise the real editor on
saved successful trajectories instead.

## Troubleshooting

### Configuration and startup

| Symptom | Cause | Action |
|---|---|---|
| `error: ...` and exit 2 before anything starts | a schema, cross-field or plan check failed | read the message; the rules are in [configuration.md](configuration.md#cross-field-rules) |
| `error: external endpoint declares served_checkpoint_id=...; this run needs ...` | the endpoint serves a different checkpoint than the run needs | point the run at the right endpoint, or set `served_checkpoint_id` to what it really serves |
| `error: ... declares no 'hf_transformers' serving backend` / `accepts ... adapters, but the trainer produces peft_lora` | the chosen serving backend cannot load trained adapters for this model | use a backend whose model-profile entry lists `peft_lora` |
| preflight `docker ... daemon not reachable` | Docker is not running | start Docker Desktop / the daemon |
| preflight `model_access ... not in the local HF cache` | the pinned revision was never downloaded | `uv run python -c "from huggingface_hub import snapshot_download as s; s('<base_model>', revision='<base_revision>')"` |
| `... is locked by another coordinator (pid ...)` | another `loop` process is working on this run | wait or stop that process; the lock disappears with its process, so it cannot go stale |
| `refusing to overwrite immutable file with different content: .../run.json` | reusing a run id with a different configuration | pick a new `--run-id`, or `loop resume` the existing run unchanged |

### Serving

| Symptom | Cause | Action |
|---|---|---|
| `learner server exited with N while loading <ckpt>; log: ...` | the server crashed: port already in use, out of memory, bad adapter | read the log named in the message; change `inference.port` in your local profile if the port is taken |
| `... not ready after Ns` | slow first load or a hung server | raise `inference.startup_timeout_sec`; check the log |
| `remote server group ... did not stop; not starting another` | a remote server ignored TERM and KILL | log into the host and stop that process group yourself before resuming |
| many `infra:` stops with connection errors or timeouts | the endpoint went away or requests exceed `request_timeout_sec` | check the server log and the tunnel; failed attempts are retried `runtime.infra_retries` times and kept |
| `model_error:...context_length_exceeded` | the conversation outgrew the server's context | lower `episode.max_output_chars` or `max_turns`, or serve with a longer context |
| HF warning about unauthenticated requests | a model load checked the hub | harmless; the pinned revision is still used |

### Episodes and stages

| Symptom | Cause | Action |
|---|---|---|
| frequent `budget:output_truncated` | replies hit `max_output_tokens` before a complete tool call | raise `episode.sampling.max_output_tokens` |
| `model_error:unparsed_tool_call` | the model wrote a tool call the server could not parse (e.g. invalid JSON escapes) | a model behavior, counted as a malformed turn; nothing to fix in the harness |
| `budget:usage_unavailable` | `max_episode_tokens` is set but the endpoint returned no usage | use an endpoint that reports usage, or remove the token budget |
| a stage item is `infra_failed` | it failed on infrastructure more than `runtime.infra_retries` times | see "Reopening failed items" below |
| a stage item is `failed` and the command exits 1 | an unexpected error (traceback on the console, message in the manifest item's `error`) | failed items are never retried and keep their stage failing on every resume: fix the cause, reopen the item (below), then `loop resume` |
| every cycle is a no-update | no completely successful collection episodes, or no valid/accepted proposals | this is a valid result; check the report's proposal and verification reasons before changing anything; never substitute fixture data |

### Editing and verification

| Symptom (reason in the report) | Meaning | Action |
|---|---|---|
| `nonempty_reasoning`, `nonempty_assistant_content` | the learner writes reasoning/text with its tool calls; strict mode cannot edit such turns | serve the learner with reasoning disabled (`chat_template_kwargs`) |
| `identical_replacement`, `unparseable_response`, `response_schema` | weak editor output | try a stronger editor condition (`editor.mode: external`) via `edit-replay` |
| `ungrounded_constant:<value>` | the replacement contains a value first seen after the decision (possible hindsight) | expected to reject answer hardcoding; inspect `proposals.csv` if it rejects legitimate edits |
| `replay_failed:<branch>:rN` with `replay:observation:...` | a replayed command printed something different (timestamps, randomness) | make the task deterministic, or declare a narrow, justified normalizer in the task |
| `replay:fingerprint:...` | the restored state differs from the source | the task has unmodeled state; fix the task |
| `replay:image_mismatch` | the task image changed since the source episode (Dockerfile or base digest) | re-collect with the current image; old sources cannot be branched |
| `not_success:edited:rN` | the edited branch failed where the original succeeded | the edit was not safe; rejected by design |
| `tie`, `saving_below_min:...` | no real saving | rejected by design |
| `intervention_timeout:...`, `intervention_tool_error:...` | the fixed action itself failed or timed out | rejected by design |

### Training

| Symptom | Cause | Action |
|---|---|---|
| "trainer found no trainable examples" (trainer exit 3) | every pair was dropped, e.g. longer than `dpo.max_length` | see `train/work/render_report.json`; raise `training.dpo.max_length` for future runs; the cycle is recorded as a no-update |
| `cycle N: WARNING k of n pairs dropped before training: {...}` | some exported pairs were not trained on (e.g. longer than `dpo.max_length`) | the drops are in `cycle.json` (`training_data`), `loop status` and the report's "Training data actually used"; they change the training data, so treat them as an experimental variable (raise `max_length` and use a larger GPU rather than accept them) |
| `training work dir ... is held by another trainer process` (exit 4) | a previous trainer on the same stage is still running | wait for it or stop it; never run two |
| `trainer used reference X, expected incoming Y` / lineage error | the published checkpoint does not continue the incoming learner | do not bypass: find why the wrong checkpoint was used |
| `error: <model> trains only on supported_train_devices [...]; the machine's training.device is ...` | the machine profile trains on a device the model profile does not declare (e.g. Gemma on `mps`) | use a machine profile with a supported training device |
| training refuses the device | the resolved device is not in the model profile's `supported_train_devices`, or CPU without `allow_cpu_fallback` | train on a supported device, or allow CPU explicitly (recorded) |
| `CUDA out of memory` in `train/remote-train.log` or `train.log` | a long prompt plus the model does not fit | set `training.dpo.gradient_checkpointing: true`, lower `training.dpo.max_length` (longer pairs are dropped and recorded), or use a larger GPU |
| `pulled adapter ... does not match its recorded sha256` | a remote checkpoint copy is incomplete or altered | re-run the train stage (`loop resume`); the staging copy is discarded |

### Remote runs

| Symptom | Cause | Action |
|---|---|---|
| `ssh <alias>: timed out ... (remote state unknown; reconcile before retrying)` | the connection dropped; the remote process may still be running | `loop remote-status <run-id> --machines M`; resume only when the lock is `free` and the PID is gone |
| `run ... is still locked on <alias>` on `loop submit` | that run id is running remotely | wait, or choose a new run id |
| `run ... already exists on <alias>` | a previous submission created it | resume it on the host: `ssh <alias>`, then `uv run loop resume runs/<run-id>` |

### Runpod pods

| Symptom | Cause | Action |
|---|---|---|
| `RUNPOD_API_KEY is not set: add it to .env` | no account key on this machine | `cp .env.example .env` and fill in `RUNPOD_API_KEY` |
| `Runpod API ...: HTTP 401` / `HTTP 403` | wrong or revoked key, or a key without access to the pod | create a key in the Runpod console (Settings -> API Keys) and update `.env` |
| `pod ...: no free GPU on its host yet; retrying until the start timeout` | a stopped pod restarts only on the machine it was created on, and others are using that machine's GPUs | wait; the start is retried for `runpod.start_timeout_sec` |
| `pod ... cannot start: its host machine has had no free GPU for Ns` | the GPU did not free up in time | retry later, raise `runpod.start_timeout_sec`, or deploy a new pod in the console and put its id in the machine profile |
| `pod ... is terminated; create a new pod and update the machine profile` | the pod was terminated (`loop` never terminates pods) | deploy a new pod in the console and update `pod_id` |
| `pod ... at IP:PORT does not accept SSH` | your public key is not in the Runpod account settings (a pod reads it at start), or the pod exposes no TCP port 22 | add the key under Settings -> SSH Public Keys, or fix the template's ports (see [runpod.md](runpod.md)) |
| a pod is still running after a command ended | the command was killed with SIGKILL, or the laptop lost power | the pod's watchdog stops (created pods: terminates) it after `runpod.idle_stop_minutes`; `loop pod stop --machines M` stops an existing pod now, `loop pod cleanup --machines M` terminates leftover created pods |
| `no ... GPU became available in ... cloud within Ns` | no machine has a free GPU of the spec's types | retry later, add GPU types to `runpod.create.<spec>.gpu_types`, or try `cloud_type: COMMUNITY` |
| `created pod ... costs $X/hr, above ...max_cost_per_hr=...; terminated` | the current price of the GPU type is above the limit | raise `max_cost_per_hr` or change the GPU types |

### Reopening failed items

`failed` and `infra_failed` are terminal on purpose: re-running them silently on resume would
change denominators after later stages had used them. There is no command to reopen an item.
Once the cause is fixed (or was transient):

- **Evaluation items** can be reopened at any time: with no coordinator running, set the item's
  `status` to `pending` in `cycles/cycle-NNN/eval/manifest.json`, then run
  `loop stage <run> --cycle NNN --stage eval` and `loop report <run>`.
- **Collect, edit or verify items** can only be reopened the same way while their cycle has not
  exported its dataset yet (`cycles/cycle-NNN/dataset/manifest.json` absent); then `loop resume`.
  Afterwards the dataset is frozen and a reopened item would not change it.

The manual edit is not recorded anywhere else, so note it with the run. Earlier attempts stay in
`*.interrupted-N` directories either way.
