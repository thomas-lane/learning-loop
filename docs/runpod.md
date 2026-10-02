# Runpod deployment

How to use a Runpod GPU pod with this project. There are three modes:

- **A pod per command (`kind: runpod`, `pod: <spec>`, recommended).** Every `loop` command that
  needs a GPU creates a pod from a spec in the machine profile (GPU types, price limit, image),
  prepares it, uses it, and terminates it. Nothing is kept on the pod, so any machine with a
  free GPU of the listed types will do ([below](#a-pod-per-command)).
- **An existing pod (`kind: runpod`, `pod_id: ...`).** You create the pod once in the Runpod
  console; every command starts it, prepares it, uses it, and stops it again. A stopped pod can
  only restart on the machine it was created on, so starts can be refused for hours when that
  machine is busy.
- **Fixed SSH host (`kind: ssh`).** You start, stop and reach the machine yourself; the project
  only uses it over SSH. This works for any Linux GPU machine, Runpod or not
  ([below](#fixed-ssh-host-mode)).

In the Runpod modes a watchdog on the pod stops (or terminates) it if your laptop goes away. All
results stay on your laptop in `runs/<run-id>/`.

**Status:**
- **Fixed-SSH-host mode** was validated on one pod: SSH setup, the setup script,
  `loop sync-hosts`, the real training tests on CUDA, and live runs from a laptop coordinator
  ([what was validated](#what-was-validated)).
- **A pod per command** is tested against a fake Runpod API and fake SSH; its live use is
  recorded under [what was validated](#what-was-validated).
- **Existing-pod mode** is tested against a fake Runpod API and fake SSH. Live so far: the API key,
  `loop pod status`, and a start refused by Runpod because the pod's machine had no free GPU
  (retried for the start timeout, then stopped with a clear error). A full live start is pending.
- **Gemma on a pod**: E4B served and trained on an A100 in a probe; no Gemma learning run yet
  ([below](#gemma-4-on-a-pod)).

Runpod's interface changes over time, so check their docs where a step depends on it.

## How a pod fits in

```text
laptop (coordinator)                                  Runpod pod (GPU host)
  uv run loop run ...                                   /workspace/learn-from-experience
  Runpod REST API: start / status / stop  ──────────>   (pod started, address looked up)
  Docker task + verifier containers   ── ssh / rsync ─>   model server (managed, per checkpoint)
  runs/<run-id>/ (all results, logs)  <─ rsync ───────    trainer (LoRA DPO), checkpoints
  http://127.0.0.1:8765  ═══ ssh -L tunnel (by the run) ═> 127.0.0.1:8765 (server, loopback only)
  heartbeat every minute  ───────────────────────────>   watchdog: stops the pod when it goes stale
```

- **The coordinator stays off the pod.** Tasks run in Docker containers, and a standard Runpod
  pod is itself a container without a Docker daemon (as far as we know at the time of writing).
  Your laptop (or another Linux machine with Docker) runs the coordinator and the task
  containers; the pod only serves models and trains.
- **The pod's jobs.** The run starts and stops the model server on the pod for each checkpoint,
  runs training there, and pulls each published checkpoint back. Requests reach the server
  through an SSH tunnel that the run opens. Never expose the server's port through Runpod's public
  HTTP proxy, which has no authentication.

## Choosing the pod

**Template.** The *Runpod PyTorch* template (any recent version) is a good base, but not for its
PyTorch. This project installs its own pinned environment with `uv` (torch 2.14, CUDA 13
libraries bundled in the wheels) into the project's own `.venv`, and never uses the template's
torch. What you need from the template:

- Linux on x86_64 with `apt` (Ubuntu-based);
- an SSH server that installs your public key;
- SSH exposed as a TCP port (*full* SSH, port 22);
- a volume mounted at `/workspace`.

The PyTorch templates provide these. Any other Ubuntu-based image with `sshd` works too.

**GPU driver: needs CUDA 13.** The locked torch wheels use CUDA 13. The pod's NVIDIA driver
must support CUDA 13.x (driver branch R580 or newer). If Runpod's deploy page offers a CUDA
version filter, select 13.x. The setup script prints the driver's supported CUDA version and
warns if it is below 13.

**GPU size (estimates).**

| Learner | Suggested GPU memory | Notes |
|---|---|---|
| Qwen3-0.6B, Qwen3-1.7B | 24 GB (e.g. RTX 3090/4090, A5000, L4) | validated on an RTX 3090; serving and training run one after the other |
| Gemma-4-E4B (`pilot.yaml`) | 80 GB (A100/H100) for `max_length: 32768` | measured on an A100: serving 15 GB; training peaks at 24 GB for a 7.3k-token pair and 54 GB for a 28.9k-token pair (about 1.3 GB per 1k tokens on top of the weights) |
| Gemma-4-E2B | 24–48 GB depending on `max_length` | bf16 weights are ~10 GB; untested |

**Volume (existing pods).** Attach a volume at `/workspace`. Stopping a pod resets everything outside the volume;
without one, every start re-installs the environment (about 10 GB) and re-downloads the models.
Allow at least 50 GB for the Qwen models and 100 GB or more for Gemma. The setup script warns
when no volume is mounted.

**Pod type.** Use a GPU Pod, not Serverless. Secure or Community Cloud both work. A stopped
pod restarts only on the same machine; while that machine's GPUs are taken by others, Runpod
refuses the start. `loop` retries for `runpod.start_timeout_sec` and then fails with a clear
message (retry later, or deploy a new pod and update `pod_id`).

**SSH key.** Save your public key in the Runpod account settings *before* creating the pod.
- **How it reaches the pod:** it's installed into `/root/.ssh/authorized_keys` on every pod start
  (via the `PUBLIC_KEY` environment variable).
- **Automatic mode needs this:** the key is re-installed on each start, so a fix made inside the
  pod is lost when the run restarts it.
- **The proxy login proves nothing here.** `ssh.runpod.io` can work even when no key is saved.
  On a pod started without an account key, `PUBLIC_KEY` is the literal string `null` and full SSH
  is refused.

## A pod per command

Copy the example and adjust the spec:

```bash
cp configs/machines/examples/runpod-a100.yaml configs/machines/local/runpod-a100.yaml
```

```yaml
runpod:
  create:
    a100:
      gpu_types: [NVIDIA A100 80GB PCIe, NVIDIA A100-SXM4-80GB]   # Runpod GPU type ids, preference order
      cloud_type: SECURE
      max_cost_per_hr: 2.5            # a pod priced above this is terminated at once
      container_disk_gb: 200          # environment + model downloads; discarded on termination
```

Hosts then refer to the spec with `host: {kind: runpod, pod: a100, workdir: ...}`. You need the
API key in `.env` and an SSH key pair (`identity_file`, default `~/.ssh/id_ed25519`); its public
half is passed to the pod as `PUBLIC_KEY`, so nothing has to be saved in the Runpod account.

Each command that uses the GPU:

1. creates a pod named `lfe-<spec>-<UTC stamp>` with the first listed GPU type that is free
   (creation is retried for `start_timeout_sec` while none is), on a host whose driver supports
   CUDA 13 (`allowed_cuda_versions`), and records it in `artifacts/runpod/created.jsonl`;
2. terminates it immediately if its hourly price is above `max_cost_per_hr`;
3. prepares it like any pod (setup script, code, environment) and downloads the model on the
   first server start (allow for this in `inference.startup_timeout_sec`);
4. terminates it when the command ends, whether it succeeds, fails or is interrupted.

A resumed run gets a new pod; an interrupted training stage restarts from step 0. If the laptop
dies, the pod's watchdog terminates it after `idle_stop_minutes` (if the pod-scoped key may not
terminate, it stops the pod instead). To remove anything left over:

```bash
uv run loop pod status  --machines configs/machines/local/runpod-a100.yaml   # created pods still alive
uv run loop pod cleanup --machines configs/machines/local/runpod-a100.yaml --dry-run
uv run loop pod cleanup --machines configs/machines/local/runpod-a100.yaml
```

`cleanup` terminates only pods that are in the ledger and still named `lfe-*`; it never touches
other pods in your account.

## Setup (existing pod)

1. **Save your SSH public key** in Runpod (Settings → SSH Public Keys).
2. **Create the pod** as above (volume at `/workspace`, TCP port 22 exposed). Note its pod id,
   e.g. `abc123def4567x`. You can stop it straight away: the runs start it.
3. **Add your Runpod API key to `.env`** (Runpod console → Settings → API Keys). It stays on this
   machine; the pod's watchdog uses the pod's own scoped key.

   ```bash
   cp .env.example .env && chmod 600 .env    # then fill in RUNPOD_API_KEY=...
   ```

4. **Create your machine profile** and set `pod_id` (and `identity_file` if your key is not
   `~/.ssh/id_ed25519`):

   ```bash
   cp configs/machines/examples/runpod.yaml configs/machines/local/runpod.yaml
   ```

5. **Check access**, read-only: `uv run loop pod status --machines configs/machines/local/runpod.yaml`
6. **Prepare the pod once** (starts it, installs tools, syncs code, builds the environment, stops
   it):

   ```bash
   uv run loop sync-hosts --machines configs/machines/local/runpod.yaml
   ```

7. **Optionally verify the GPU path** on a new kind of GPU or image: `loop pod start`, then use
   the address `loop pod status` prints, then `loop pod stop`.

   ```bash
   ssh -p <port> root@<ip> 'cd /workspace/learn-from-experience && uv run --extra train pytest -m train tests/train -q'
   ```

## Running

```bash
uv run loop validate experiments/smoke-mac.yaml --machines configs/machines/local/runpod.yaml \
    --set learner.model_profile=qwen3-1.7b
caffeinate -i uv run loop run experiments/smoke-mac.yaml --machines configs/machines/local/runpod.yaml \
    --set learner.model_profile=qwen3-1.7b --run-id pod-smoke
```

`smoke-mac.yaml` is just the smallest live experiment; nothing in it is Mac-specific.
`caffeinate` keeps a Mac awake. Each command that uses the pod does the following:

1. Starts the pod if it is stopped, waits until it reports an address and accepts SSH, and
   prepares it (setup script, code, environment; quick when nothing changed).
2. Starts the watchdog on the pod and sends it a heartbeat every `heartbeat_sec`.
3. Runs as usual: model servers start and stop on the pod behind an SSH tunnel the run opens and
   closes, and training runs there, with each checkpoint pulled back and hash-checked. Server and
   training logs are copied into the run directory.
4. Stops the pod when the command ends, whether it succeeds, fails or is interrupted with Ctrl-C
   (`stop_when_done: true`).

**If the laptop sleeps, crashes or loses the network,** the heartbeat stops and the watchdog
stops the pod after `idle_stop_minutes`. Afterwards, `loop resume runs/<run-id>` restarts it and
continues from the run's manifests:
- completed work is kept;
- an interrupted model server is simply started again;
- an interrupted **training stage** resumes from the trainer's own checkpoint if the pod's
  `/workspace` survived (a volume), otherwise it restarts from step 0. Every relaunch is recorded
  in `cycles/cycle-NNN/train/relaunches.jsonl`.

Manual control is always available:

```bash
uv run loop pod status --machines configs/machines/local/runpod.yaml
uv run loop pod start  --machines configs/machines/local/runpod.yaml
uv run loop pod stop   --machines configs/machines/local/runpod.yaml
```

## Logs and records

| Where | What |
|---|---|
| `runs/<run-id>/logs/pod-lifecycle.jsonl` | created (pod id, GPU, price) or start, ready (address), prepared (with warnings), watchdog started, stop or terminate for every command |
| `artifacts/runpod/created.jsonl` | ledger of every pod `loop` created and terminated |
| `runs/<run-id>/logs/<role>-server-<ckpt>-<ts>.log` | model server logs, copied back from the pod when each server stops |
| `runs/<run-id>/cycles/cycle-NNN/train/remote-train.log` | the trainer's log, copied back after each training stage |
| `runs/_pods/pod-lifecycle.jsonl` | lifecycle events of `loop sync-hosts` (no run directory) |
| on the pod: `runs/_pod/watchdog.jsonl` | the watchdog's own log, including any stop it requested |
| `artifacts/runpod/<pod>.known_hosts` | the pod's SSH host key, trusted on first use and reset on each command (pods get new host keys on every start) |

## Fixed SSH host mode

For a machine you manage yourself (a Runpod pod you start and stop by hand, or any Linux GPU
box), use `kind: ssh` hosts with an alias from `~/.ssh/config`:

1. Add the alias (`HostName`, `Port`, `User root`, `IdentityFile`); for a Runpod pod use the
   *full* SSH address, the public IP and the port mapped to 22.
2. Prepare the host: `ssh <alias> 'bash -s' < scripts/setup_gpu_host.sh`.
3. In the machine profile, set `host: {kind: ssh, ssh_alias: <alias>, workdir: ...}` and
   `api_base: http://127.0.0.1:8765/v1`. Then run `uv run loop sync-hosts --machines <profile>`.
4. During runs keep the tunnel open yourself: `ssh -N -L 8765:127.0.0.1:8765 <alias>`.

The run still syncs code before serving and training and copies logs back. It never starts,
stops or watches the machine: stop it yourself when you are done.

## Gemma 4 on a pod

`experiments/pilot.yaml` uses Gemma-4-E4B (`configs/models/gemma-4-e4b-it.yaml`);
`configs/models/gemma-4-e2b-it.yaml` is a smaller fallback (`--set learner.model_profile=gemma-4-e2b-it`).
What is in place, and what has not run yet:

- The reference server (`backend: hf_transformers`) serves both with the `gemma4` tool-call
  parser. The parser round-trips the pinned chat templates (tokenizer-only tests), and E4B's base
  model completed a three-turn tool-calling exchange on an A100 (below); the pilot run served its
  cycle-0 adapter in cycle 1.
- LoRA is restricted to the language model (`lora_exclude_modules` skips the vision/audio towers).
- Training memory: the trainer computes logits only over the completion tokens, and
  `pilot.yaml` enables `training.dpo.gradient_checkpointing` with `max_length: 32768`. On an A100,
  E4B LoRA DPO ran on 7.3k- and 28.9k-token pairs (peaks 24 GB and 54 GB, see the GPU size table).
- Both checkpoints are public (not gated) at the pinned revisions; the pod downloads them on the
  first server start, so allow a longer `inference.startup_timeout_sec` on a pod without a volume.
- vLLM (`inference.backend: vllm`) runs as `hf_server`'s generation engine (below).

Treat a first Gemma run as integration work: `loop evaluate` of the base checkpoint, then a
one-cycle run, before the full pilot.

## Faster serving with vLLM

`hf_server` generates with transformers one token at a time from Python: about 10–15 tokens/s
for Gemma-4-E4B on an A100, which makes model time about 98% of an episode. With
`inference.backend: vllm` (`configs/machines/examples/runpod-a100-vllm.yaml`), `hf_server` still
renders every prompt with `training/render.py`, parses tool calls and counts tokens with the
training tokenizer, but a vLLM child process does the generation from token ids:

- vLLM is installed on first use into `.engines/vllm==<version>/` on the pod (its own torch; a few
  minutes on a new pod), from the model profile's `serving.vllm.engine_package`;
- sampling is fully explicit (`--generation-config vllm`), prefix caching is off, and base
  checkpoints are served through the zero LoRA like trained ones;
- `request_concurrency` sets how many requests vLLM batches; `docker_concurrency` must be at
  least as large for batching to help.

Measured on an A100 with Gemma-4-E4B and the zero LoRA (768-token generations):

| Engine | Server start | Tokens/s |
|---|---|---|
| `hf_server` (transformers) | ~1 min | 10.2 |
| vLLM, profile launch args (full decode CUDA graphs, no compile) | 88 s | 76.2 unbatched; 505 total with 8 concurrent requests |
| vLLM default (`-O2`, torch.compile) | 210 s | 81.3 |

Before relying on vLLM for a model profile, run the equivalence gate on a GPU host (it needs a
trained adapter of the model, e.g. a published checkpoint copied to the host):

```bash
uv run --extra train python -m learning_loop.serving.equivalence --profile gemma-4-e4b-it \
    --adapter runs/<run>/checkpoints/<ckpt> --out equivalence.json
```

It compares token log-probabilities and greedy decoding against transformers + PEFT for the base
model, the trained adapter and a strong random adapter (see `serving/equivalence.py`).

## Costs

- A running pod bills by the hour. In both Runpod modes it runs only while a command uses it,
  plus at most `idle_stop_minutes` if the coordinator disappears.
- A pod per command costs nothing between commands, but each command spends a few minutes
  creating the pod, building the environment and downloading the model.
- A stopped existing pod still bills its volume as storage. Terminate existing pods you no longer
  need in the Runpod console; `loop` terminates only pods it created.
- Setting `stop_when_done: false` keeps a pod running between commands (faster successive
  commands). The watchdog still stops it once no command has sent a heartbeat for
  `idle_stop_minutes`.

## Pod-specific troubleshooting

General problems are covered in [operations.md](operations.md#troubleshooting).

| Symptom | Cause | Action |
|---|---|---|
| `RUNPOD_API_KEY is not set` | no key in `.env` or the environment | add it to `.env` (see `.env.example`) |
| `Runpod API ... HTTP 401` | wrong or revoked API key | create a new key in the Runpod console |
| `no free GPU on its host yet; retrying until the start timeout` / `cannot start: its host machine has had no free GPU` | the pod can only restart on its original machine, whose GPUs are in use (Runpod: "not enough free GPUs on the host machine") | wait for the retry; otherwise retry later, raise `runpod.start_timeout_sec`, or deploy a new pod and update `pod_id` |
| `pod ... did not report a public IP and SSH port within ...s` | the pod started but exposes no TCP port 22 | check the template's ports in the console |
| `new-<spec>: no ... available yet; retrying until the start timeout` / `no ... GPU became available` | no machine has a free GPU of the listed types in that cloud | wait, add GPU types to the spec, or try `cloud_type: COMMUNITY` |
| `created pod ... costs $X/hr, above ...max_cost_per_hr` | the GPU type's current price is above your limit | the pod was terminated; raise the limit or change the GPU types |
| `WARNING: could not terminate pod ...` | the terminate request failed at the end of a command | `loop pod cleanup --machines M`, or terminate it in the console; its watchdog terminates it after `idle_stop_minutes` regardless |
| `pod ... does not accept SSH` / `Permission denied (publickey)` | no SSH key saved in the account when the pod started (`authorized_keys` contains `null`), or the wrong `identity_file` | save the key in Runpod settings, then stop and start the pod |
| `Your SSH client doesn't support PTY` | a non-interactive command through the proxied `ssh.runpod.io` address (fixed-SSH-host mode) | use the full SSH address; inside the pod, `echo $RUNPOD_PUBLIC_IP $RUNPOD_TCP_PORT_22` prints it |
| `WARNING: could not stop pod ...` | the stop request failed at the end of a command | stop it with `loop pod stop` or in the console; its watchdog stops it after `idle_stop_minutes` regardless |
| `... not ready after 900s` on the first server start | the model download or environment build took longer | run `loop sync-hosts` first; raise `inference.startup_timeout_sec` |
| CUDA errors about the driver being too old, or "no kernel image" | the pod's driver does not support CUDA 13 | use a pod with a CUDA 13.x driver |
| disk full on `/workspace` | models + environment + uv cache exceed the volume | enlarge the volume, or clear `~/.cache/uv` on the pod |

## What was validated

**One pod:** Runpod PyTorch template, RTX 3090 24 GB, Ubuntu 24.04.3, driver 580.159.03 (CUDA 13.0),
no volume attached, laptop coordinator (macOS, Docker Desktop). Fixed-SSH-host mode:

| Step | Result |
|---|---|
| Full SSH | worked after fixing `authorized_keys` (it contained `null`: no key in the account when the pod started); the proxied address refused non-interactive commands |
| `scripts/setup_gpu_host.sh` | ran twice; the template already had `rsync`, `uv` 0.9.0 and `python3` 3.12; caches linked onto `/workspace`; driver check reported CUDA 13.0 |
| `loop sync-hosts` | pushed the code and built the locked environment in about 40 s (torch 2.14.0+cu130 on the system Python 3.12) |
| `pytest -m train tests/train` on the pod | 7 passed (real LoRA DPO on CUDA, continued adapter, incoming-adapter reference, resume, reload, adapter serving) |
| preflight (`--no-docker --allow-network`) | all OK; its memory check reports the host's RAM, not the pod's allotment |
| Live learning run (Qwen3-1.7B) | ran end to end; the server was started, swapped and stopped on the pod; the learner solved 0/6 collection attempts, so the cycle was a recorded no-update |
| Remote training, 2 cycles (fixed-dataset control on labeled fixture pairs, Qwen3-0.6B) | trained on CUDA each cycle; checkpoints pulled, hash-checked and relabeled with local paths; cycle 1 continued the pod-trained adapter with it as the DPO reference; each adapter was served on the pod from its pod path; nothing left running afterwards |

**Automatic mode, same pod (later session):** `loop pod status` and the API key worked once the
client sent its own User-Agent (Runpod's Cloudflare front end rejects Python's default with
error 1010). Two start attempts were refused with "not enough free GPUs on the host machine";
the second was retried for 900 s, then the command failed with the no-free-GPU message and
requested a stop. The pod stayed `EXITED`.

**A pod per command, Gemma 4 E4B probe (A100-SXM4-80GB, Secure Cloud, $1.59/hr):** a scratch
script driven through the normal pod lifecycle (not a `loop` command).

| Step | Result |
|---|---|
| Create | the pod was created (the PCIe type was not available, so the SXM type was used), recorded in the ledger, reachable over SSH 24 s later with the public key passed at creation; prepared in about 30 s |
| Serve | base E4B loaded in 62 s (15 GB on the GPU); a three-turn tool-calling exchange parsed cleanly (`grep -c`, `write_file`, then a final answer); about 15.6 generated tokens/s, single sequence |
| Train | LoRA DPO, 2 steps each on fixture pairs of 7.3k and 28.9k tokens with `max_length: 32768` and gradient checkpointing: peaks 23.8 GB and 53.6 GB; LoRA on the 7 language-model projections only (vision/audio excluded); first-step reference log-ratios 0.0; reload check exact |
| Terminate | terminated at the end; `loop pod cleanup --dry-run` found nothing |
| Model loading straight to the GPU (`device_map`, CUDA only) | the 7.3k-token stage took 128 s instead of 450 s: first load 57 s including the download, reload check 16 s, reference log-probs 10 s, training 24 s |
| `pytest -m train tests/train` on the A100 (three more created pods) | 7 passed twice with deterministic kernels; with the default (efficient) attention kernels, a resumed stage differed from an uninterrupted one by 2-9e-5 between identical runs, which is CUDA kernel noise, not resume logic |

**vLLM engine, Gemma-4-E4B (A100, created pods; scratch scripts through the normal pod
lifecycle):** vLLM 0.30.0 installed into `.engines/` on each new pod (it needs `ninja` on the
engine environment's PATH, which `ensure_engine` provides). The speeds in the table above were
measured on 768-token generations. The equivalence gate passed with the profile's launch
arguments and the pilot's cycle-0 adapter: identical prompt ids; mean |delta log-prob| per token
0.007 (zero LoRA) and 0.0055 (trained adapter); a strong random adapter's effect within 4.9% per
sequence (per-token correlation 0.995); zero LoRA exactly equal to no LoRA inside vLLM; greedy
decoding identical to transformers on all three prompts. A learning run with the vLLM engine has
not run yet.

**Not covered yet:**
- the existing-pod mode on a real pod (start, the watchdog's self-stop, stop at the end);
- the watchdog's self-termination of a created pod with the pod-scoped key;
- a Gemma learning run; serving a trained Gemma adapter; vLLM on a pod;
- a pod with a volume across a restart;
- `loop submit` (a remote coordinator).
