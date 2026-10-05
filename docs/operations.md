# Operations and troubleshooting

How to run a real experiment, watch it, and recover when something goes wrong. Commands are in
[cli.md](cli.md), configuration in [configuration.md](configuration.md), run contents in
[run-layout.md](run-layout.md).

## Running a real experiment

1. **Machine profile.** Copy the closest example from `configs/machines/examples/` to
   `configs/machines/local/<name>.yaml` (git-ignored); put credentials in `.env` (from
   `.env.example`). For a Runpod pod, follow [runpod.md](runpod.md). For a fixed SSH host, use an
   alias from `~/.ssh/config`; if the learner server runs there, keep your own SSH tunnel open to
   `inference.api_base` (as in `configs/machines/examples/lab-gpu.yaml`). Pods need none: the run
   opens their tunnel, since a pod's address changes at every start.
2. **Check the hosts.** Run `uv run loop preflight --trainable --model-profile <model>` (a name
   from `configs/models/`, without `.yaml`) on the training machine, and `uv run pytest` and
   `uv run pytest -m docker tests/integration` on the coordinator. On a new GPU host, run
   `uv run pytest -m train tests/train` once: it trains and serves a small LoRA adapter in about
   a minute, catching driver and library problems early.
3. **Validate every configuration**, controls included: `uv run loop validate <experiment> --machines <profile>`.
   Read the notes it prints (untested backends, fixture components).
4. **Calibrate.** Run the frozen baseline (`experiments/frozen-baseline.yaml`, adjusted to your
   learner) to see which dev instances are easy, intermittent or never solved. Tune on dev panels
   only, so the final panels stay unseen until step 8.
5. **Run each condition with several loop seeds**, keeping `seeds.root` fixed so evaluations pair up:
   ```bash
   for s in 0 1 2; do
     uv run loop run experiments/pilot.yaml --machines M --set seeds.loop_seed=$s --run-id pilot-s$s
   done
   uv run loop run experiments/frozen-baseline.yaml --machines M --run-id baseline
   uv run loop run experiments/fixed-dataset-control.yaml --machines M --run-id fixed-s0 \
       --set training.fixed_dataset=runs/pilot-s0/cycles/cycle-000/dataset
   ```
6. **Watch each cycle** ([below](#watching-a-run)): proposal rejection reasons, acceptance
   reasons, infra retries and success transitions (lost successes are listed explicitly).
7. **Compare:** `uv run loop compare runs/baseline --vs runs/pilot-s0 runs/pilot-s1 runs/pilot-s2`.
   Effects are reported as the `--vs` side minus the positional side, here pilot minus baseline,
   so a better pilot shows a positive success change and a negative token delta. The output names
   the runs on each side.
8. **Freeze decisions, then evaluate final panels.** Write down the method and checkpoint choice
   first, then run `loop evaluate ... --final`. Optionally run `loop external-eval` (a dry run until
   `--execute`). Neither feeds back into a run.
9. **Keep the evidence.** `loop fetch` remote runs and archive whole run directories (they are
   self-contained).

To compare editors on saved trajectories: `loop edit-replay runs/pilot-s0 --cycle 0
--experiment experiments/ablations/editor-external.yaml --machines M`.

## Watching a run

`uv run loop dashboard runs/<id> --open` serves a read-only page at `http://localhost:8090/`
that reloads every 10 s: progress and ETAs, evaluation across cycles, recent episodes with
transcripts, proposals, verifications, and pod spend. It never takes the run lock, so it is safe
next to a running `loop run`. Without a run directory it lists all runs. To show the console log,
save it (`uv run loop run ... 2>&1 | tee pilot.log`) and add `--log pilot.log`.

Numbers follow `loop report`'s definitions (ungraded episodes count as failures); a cycle still
running is left out of the charts. For confidence intervals use `loop report` and `loop compare`.
`loop status runs/<id>` prints stage counts as JSON.

| Symptom | Cause | Action |
|---|---|---|
| `error: cannot listen on 127.0.0.1:8090 (...)` | the port is in use | `--port 0`, or another port |
| `error: ... is not a run directory (no run.json)` | the path is not `runs/<id>` | pass a run directory, or none to list all runs |
| `error: --log needs RUN_DIR ...` | `--log` belongs to one run | add the run directory |
| run status `coordinator not running` while cycles are unfinished | the coordinator exited (crash, Ctrl-C, reboot) | `loop resume runs/<id>` |
| run status `unknown (no lock holder on this host)` | the coordinator runs on another host, or never started | remote runs: `loop remote-status <run-id> --machines M` |
| a section shows a traceback | a file it reads has an unexpected shape | the rest of the page still works; no file is changed |

## Laptop smoke tests

Smoke levels and durations are in
[README.md → Smoke tests](../README.md#smoke-tests-bounded-laptop-safe). On a laptop, stop at
`loop smoke live`; do not leave longer training or load tests running unattended.

## Troubleshooting

Messages are quoted as the code prints them; `...`, `N` and `<...>` mark variable parts. Terms
such as *no-update*, *infra* and *fixture* are in [glossary.md](glossary.md).

### Configuration and startup

| Symptom | Cause | Action |
|---|---|---|
| `error: ...` and exit 2 before anything starts | a schema, cross-field or plan check failed | read the message; rules are in [configuration.md](configuration.md#cross-field-rules) |
| `error: external endpoint declares served_checkpoint_id=...; this run needs ...` | the endpoint serves a different checkpoint than the command needs: the initial checkpoint (`validate`, `run`), `--checkpoint` (`evaluate`) or the source cycle's learner (`edit-replay`) | use the right endpoint, or correct `served_checkpoint_id` |
| `error: editor endpoint declares ...; the editor is ...` | the editor's external endpoint serves a different checkpoint than the editor's ([configuration.md](configuration.md#cross-field-rules)) | use the right endpoint, or correct `editor_inference.served_checkpoint_id` |
| `error: configs/models/<m>.yaml marks serving backend '<b>' unsupported; the <learner or editor> cannot be served with it` | the model profile records that this backend does not work for this model | choose another `inference.backend` (or `editor_inference.backend`) that the profile lists |
| `managed inference serves hf_transformers or vllm (learning_loop.serving.hf_server); run <b> yourself and declare it as mode: external` | a managed endpoint names a backend the repository's server cannot run | start that server yourself and use `mode: external` with `api_base` and `served_checkpoint_id`, or use `hf_transformers` or `vllm` |
| `error: <path>: name '<n>' must match the file name '<stem>' (experiments refer to profiles by file name)` | a model profile was copied or renamed without updating `name` | make `name` equal the file name |
| `serving.<k>.backend is '<b>'; it must equal its key '<k>'` | a model profile's `serving` entry is filed under another backend's key | rename the key or fix `backend` |
| `error: model profile ... declares no '...' serving backend` / `... accepts ... adapters, but the trainer produces peft_lora ...` | the serving backend cannot load trained adapters | use a backend whose model-profile entry lists `peft_lora` |
| preflight `docker daemon not reachable` | Docker is not running | start Docker |
| preflight `... not in the local HF cache` | the pinned revision was never downloaded | `uv run python -c "from huggingface_hub import snapshot_download as s; s('<base_model>', revision='<base_revision>')"` |
| `error: ... is locked by another coordinator (...)` and exit 2 | another `loop` process is using this run; the refused command changed nothing, not even `invocations.jsonl` | wait or stop it (the lock ends with its process) |
| `refusing to overwrite immutable file with different content: .../run.json` | a run id was reused with a different configuration | use a new `--run-id`, or resume the run unchanged |

### Serving

| Symptom | Cause | Action |
|---|---|---|
| `... server exited with N while loading <ckpt>; log: ...` / `... server on <kind> host exited while loading <ckpt>; log: ...` | port in use, out of memory, or a bad adapter | read the log; change `inference.port` if the port is taken |
| `... server for <ckpt> not ready after Ns; log: ...` | slow first load (the host downloads the model, or `uv run` builds the environment) or a hung server | check the log; on a fixed SSH host, run `loop sync-hosts` first so the environment is built in advance; raise `inference.startup_timeout_sec` |
| `vLLM exited with N while starting` in the server log | the vLLM engine failed to start, for example too little free GPU memory (`--gpu-memory-utilization`), a missing build tool, or an unsupported model and LoRA combination | read vLLM's own error in the server log in `runs/<id>/logs/` (a remote host's log is copied there from its `runs/_servers/` when the server stops); the engine environment is in `.engines/` on the host, named after `engine_package` with characters other than letters, digits, `.`, `_` and `-` replaced by `_` (`vllm==0.30.0` → `.engines/vllm__0.30.0/`) |
| `infra:` stops with `vLLM returned HTTP ...` | the vLLM engine rejected or failed a request | read the server log; check that the model profile's `serving.vllm.launch_args` suit its `engine_package` version |
| `remote server group ... did not stop; not starting another` | a remote server ignored TERM and KILL | stop that process group on the host, then resume |
| many `infra:` stops (connection errors, timeouts) | the endpoint or tunnel went away, or `inference.request_timeout_sec` is too short | check the server log and tunnel; attempts are retried `runtime.infra_retries` times |
| `model_error:...context_length_exceeded` | the conversation outgrew the context | lower `episode.max_output_chars` or `episode.max_turns`, or serve a longer context |
| Hugging Face warning about unauthenticated requests | a model load contacted the hub | harmless |

### Episodes and stages

| Symptom | Cause | Action |
|---|---|---|
| frequent `budget:output_truncated` | replies hit `max_output_tokens` mid tool call | raise `episode.sampling.max_output_tokens` |
| `model_error:unparsed_tool_call` | the reply's only tool calls were unparseable | none: model behavior, counted as a malformed turn |
| `budget:usage_unavailable` | `max_episode_tokens` is set but the endpoint reports no usage | use an endpoint that reports usage, or drop the token budget |
| a stage item is `infra_failed` | it exceeded `runtime.infra_retries` | [reopen it](#reopening-failed-items) once fixed |
| a stage item is `failed` and the command exits 1 | an unexpected error (traceback on the console, message in the item's `error`) | fix the cause, [reopen the item](#reopening-failed-items), `loop resume` |
| every cycle is a no-update | no fully successful collection episodes, or no accepted proposals | a valid result: check the report's proposal and verification reasons; never substitute fixture data |

### Editing and verification

Common rejection reasons, as shown in the report, `proposals.csv` and `verifications.csv`
(`<branch>` is `original` or `edited`, `rN` the repetition). The full acceptance rule is in
[experiment.md](experiment.md#verification-and-acceptance).

| Reason | Meaning | Action |
|---|---|---|
| `nonempty_reasoning`, `nonempty_assistant_content` | the learner writes text or reasoning with its tool calls | disable reasoning in the model profile (`chat_template_kwargs: {enable_thinking: false}`) |
| `skipped:no_editable_turns` (status `abstained`) | no turn of the source is editable, so no editor request was made | none if expected; which turns are editable is defined in [experiment.md](experiment.md#editing) |
| `no_tool_call`, `multiple_tool_calls:N`, `unknown_editor_tool:...`, `unparsed_tool_call:...`, `response_schema:...`, `identical_replacement` | the editor did not answer with exactly one valid answer-tool call (`unparsed_tool_call` gives the parse error of a malformed call) | try `editor.mode: external` via `edit-replay` |
| `editor_output_truncated` (with the reasons above) | the editor's reply hit `editor.sampling.max_output_tokens` before a complete answer call | raise `editor.sampling.max_output_tokens` |
| `editor_infra_error:...` | the editor endpoint failed (connection error, timeout, HTTP 408, 429 or 5xx); the edit item is retried `runtime.infra_retries` times, then becomes `infra_failed`, and the report counts it apart from invalid proposals | check the editor server log and tunnel |
| `editor_request_error:...` | the editor endpoint rejected the request with another 4xx, e.g. `context_length_exceeded` | read the message; for context errors, serve the editor with a longer context |
| `ungrounded_constant:<value>` | the replacement uses a value seen only later (possible hindsight) | check `proposals.csv` if legitimate edits are rejected |
| `replay_failed:<branch>:rN` with `replay:observation:...` | a replayed command printed something different | make the task deterministic, or declare a narrow normalizer in the task |
| `replay:fingerprint:...` | the restored state differs from the source | fix the task's undeclared state |
| `replay:image_mismatch` | the task image changed since the source episode | collect again with the current image |
| `not_success:<branch>:rN`, `tie`, `no_saving:...`, `saving_below_min:...`, `relative_saving_below_min:...` | a branch failed, or the edit saved too few tokens | none: rejected by design |
| `intervention_timeout:...`, `intervention_tool_error:...` | the fixed action itself timed out or raised a tool error | none: rejected by design |
| `bad_stop:<branch>:rN:<stop reason>`, `not_graded:...`, `missing_cost:...` | a branch hit a budget, infra or safety stop, went ungraded, or had no usage | look up the stop reason above |

### Training

| Symptom | Cause | Action |
|---|---|---|
| `cycle N: trainer found no trainable examples (...) -> no-update cycle` | every pair was dropped (e.g. longer than `training.dpo.max_length`) | recorded as a no-update; see `training_data` in `cycles/cycle-NNN/cycle.json`; raise `training.dpo.max_length` |
| `cycle N: WARNING k of n pairs dropped before training: {...}` | some pairs were not trained on | listed in `cycle.json`, `loop status` and the report; raise `training.dpo.max_length` (larger GPU) rather than accept them |
| `training work dir ... is held by another trainer process; ...` (exit 4) | a trainer for this stage is still running | wait or stop it |
| `trainer used reference X, expected incoming Y` / `checkpoint lineage does not continue the incoming learner` | the wrong checkpoint was published | find the cause; do not bypass |
| `error: <model> trains only on supported_train_devices [...]; the machine's training.device is ...` | e.g. Gemma on `mps` | use a supported training device |
| `device ... unavailable (...) and allow_cpu_fallback is false` / `device ... is not in the model profile's supported_train_devices [...]` | the requested accelerator is absent, or the device used (possibly the CPU fallback) is not in the model profile's list | use a supported device; set the machine's `training.allow_cpu_fallback` only for a model that lists `cpu` |
| `CUDA out of memory` in `cycles/cycle-NNN/train/train.log` (local) or `remote-train.log` (remote) | the pair plus model do not fit | `training.dpo.gradient_checkpointing: true`, lower `training.dpo.max_length`, or a larger GPU |
| `pulled adapter ... does not match its recorded sha256` | an incomplete remote copy | `loop resume` (re-runs the train stage) |

### Remote runs

| Symptom | Cause | Action |
|---|---|---|
| `ssh <alias>: timed out after Ns (remote state unknown; reconcile before retrying)` | the connection dropped; the remote process may still run | `loop remote-status <run-id> --machines M`; resume only when the lock is `free` and the process is gone |
| `run ... is still locked on <alias>: a coordinator is running; not resubmitting` | that run is running remotely | wait, or use a new run id |
| `run ... already exists on <alias>; ...` | an earlier submission created it | `ssh <alias>`, then `uv run loop resume runs/<run-id>` |

### Runpod pods

Setup is in [runpod.md](runpod.md). An *existing pod* is one you made in the Runpod console and
name by `pod_id`; `loop` only starts and stops it. A *created pod* is made by `loop` from a
`runpod.create` spec and terminated when the command ends.

| Symptom | Cause | Action |
|---|---|---|
| `RUNPOD_API_KEY is not set: add it to .env ...` | no API key on this machine | add `RUNPOD_API_KEY` to `.env` |
| `Runpod API ...: HTTP 401` / `HTTP 403` | wrong, revoked or unauthorized key | create a key (console → Settings → API Keys) |
| `pod ...: no free GPU on its host yet; retrying until the start timeout` | existing pod: its machine's GPUs are taken | wait |
| `pod ... cannot start: its host machine has had no free GPU for Ns ...` | existing pod: still no GPU at the timeout | retry later, raise `runpod.start_timeout_sec`, or deploy a new pod and update `pod_id` |
| `pod ... is terminated; create a new pod and update the machine profile` | the existing pod was terminated (`loop` never terminates existing pods) | deploy a new pod; update `pod_id` |
| `new-<spec>: no ... available yet; retrying until the start timeout` | created pod: no free GPU of the listed types | wait |
| `no ... GPU became available in ... cloud within Ns; ...` | created pod: still none at the timeout | retry later, add `gpu_types`, or try `cloud_type: COMMUNITY` |
| `created pod ... costs $X/hr, above runpod.create.<spec>.max_cost_per_hr=...; terminated` | the price is above your limit (pod already terminated) | raise `max_cost_per_hr` or change `gpu_types` |
| `pod ... did not report a public IP and SSH port within Ns (...); its GPU may be taken` | no TCP port 22 exposed, or no free GPU | check the template's ports |
| `pod ... at IP:PORT does not accept SSH: ...` / `Permission denied (publickey)` | existing pod started with no account SSH key (`authorized_keys` contains `null`), or wrong `identity_file` | save the key (Settings → SSH Public Keys), stop and start the pod; check `runpod.identity_file` |
| `Your SSH client doesn't support PTY` | the proxied `ssh.runpod.io` address (fixed-SSH-host mode) | use the public IP and port mapped to 22 (`echo $RUNPOD_PUBLIC_IP $RUNPOD_TCP_PORT_22` on the pod) |
| `WARNING: could not stop pod ...` | the final stop request failed | `loop pod stop --machines M` or the console |
| `WARNING: could not terminate pod ...` | the final terminate request failed | `loop pod cleanup --machines M` or the console |
| a pod still runs after a command ended | SIGKILL or power loss | the watchdog stops/terminates it after `runpod.idle_stop_minutes`; now: `loop pod stop` (existing) or `loop pod cleanup` (created) |
| CUDA errors about an old driver, or "no kernel image" | the driver lacks CUDA 13 | use a CUDA 13.x pod ([runpod.md](runpod.md#choosing-the-pod)) |
| disk full on `/workspace` | models, environment and uv cache exceed the volume | enlarge the volume, or clear `~/.cache/uv` |

### Reopening failed items

`failed` and `infra_failed` items are never retried automatically, so counts later stages used
stay fixed. To reopen one after fixing the cause, first make sure no coordinator is running (it
rewrites the manifest and would overwrite your edit), then:

- **Evaluation item** (any time): set its `status` to `pending` in
  `cycles/cycle-NNN/eval/manifest.json`, run `loop stage <run> --cycle N --stage eval`, then
  `loop report <run>`. `loop resume` would skip it once the cycle is finished.
- **Collect, edit or verify item** (only while `cycles/cycle-NNN/dataset/manifest.json` does not
  exist): set its `status` to `pending` in that stage's `manifest.json`, then `loop resume <run>`.
  After the dataset is exported, the item may run again but cannot change training, because
  the exported dataset is never rewritten.

Note the manual edit with the run; nothing else records it. Earlier attempts stay in
`*.interrupted-N` directories.
