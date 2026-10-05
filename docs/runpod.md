# Runpod deployment

How to serve and train on a Runpod GPU pod while your laptop runs the experiment. There are three
ways to use a GPU machine:

| Mode | Machine profile | What `loop` does with the pod |
|---|---|---|
| [A pod per command](#a-pod-per-command) (recommended) | `kind: runpod`, `pod: <spec>` | creates a pod from your spec for each command, uses it, terminates it |
| [An existing pod](#setup-existing-pod) | `kind: runpod`, `pod_id: ...` | starts your pod for each command, uses it, stops it |
| [Fixed SSH host](#fixed-ssh-host-mode) | `kind: ssh` | only connects over SSH; you start and stop the machine |

Which modes have run on real pods is in the [README status table](../README.md#status).

> **Pods cost money while they run.** In both Runpod modes, every command stops (existing pod,
> unless `stop_when_done: false`) or terminates (created pod) the pod when it ends, whether it
> succeeds, fails or is interrupted with Ctrl-C. If your laptop sleeps or crashes, a watchdog on
> the pod stops or terminates it after `runpod.idle_stop_minutes`. `loop` only terminates pods it
> created itself. In fixed-SSH-host mode, stopping the machine is your job. After a crash, check
> with `loop pod status`.

All results stay on your laptop in `runs/<run-id>/`. Runpod's console changes over time, so check
their docs where a step depends on it.

## How a pod fits in

```text
laptop (coordinator)                                  Runpod pod (GPU host)
  uv run loop run ...                                   /workspace/learn-from-experience
  Runpod REST API: create / start / stop  ─────────>    (pod started, address looked up)
  Docker task + verifier containers   ── ssh / rsync ─>   model server (one per checkpoint)
  runs/<run-id>/ (all results, logs)  <─ rsync ───────    trainer (LoRA DPO), checkpoints
  http://127.0.0.1:8765  ═══ ssh -L tunnel (by the run) ═> 127.0.0.1:8765 (server, loopback only)
  heartbeat every minute  ───────────────────────────>   watchdog: stops the pod when it goes stale
```

- **The coordinator runs on your laptop** (or another Linux machine with Docker), because the
  tasks run in Docker containers and a standard Runpod pod has no Docker daemon. The pod only
  serves models and trains.
- **The server is reached only through the run's SSH tunnel.** Never expose its port through
  Runpod's public HTTP proxy, which has no authentication.

The full process and data flow are in [architecture.md](architecture.md).

## Choosing the pod

**Image.** The project installs its own pinned environment with `uv` (torch 2.14, CUDA 13
libraries in the wheels) and never uses the image's PyTorch. The image needs:

- Linux on x86_64 with `apt` (Ubuntu-based);
- an SSH server that installs your public key from `PUBLIC_KEY`;
- SSH exposed as a TCP port (*full* SSH, port 22).

Runpod's *PyTorch* templates provide all of this; so does any Ubuntu-based image with `sshd`.

**GPU driver: CUDA 13.** The pod's NVIDIA driver must support CUDA 13.x (driver branch R580 or
newer). If Runpod's deploy page has a CUDA version filter, pick 13.x. The setup script prints the
driver's CUDA version and warns if it is below 13. Created pods get this automatically
(`allowed_cuda_versions`).

**GPU memory (estimates).**

| Learner | Suggested GPU memory | Notes |
|---|---|---|
| Qwen3-0.6B, Qwen3-1.7B | 24 GB (e.g. RTX 3090/4090, A5000, L4) | serving and training take turns, so only one model is on the GPU at a time |
| Gemma-4-E4B (`pilot.yaml`) | 80 GB (A100/H100) for `max_length: 32768` | serving uses 15 GB; training with gradient checkpointing peaks at 24 GB for a 7.3k-token pair and 54 GB for a 28.9k-token pair (about 1.3 GB per 1k tokens on top of the weights) |
| Gemma-4-E2B | 24–48 GB depending on `max_length` | bf16 weights are about 10 GB |

**Disk.**
- Created pods use `container_disk_gb` from the spec, discarded on termination.
- Existing pods should have a volume at `/workspace`: stopping a pod wipes everything outside it,
  so without one every start reinstalls the environment (about 10 GB) and downloads the models
  again. Allow at least 50 GB for the Qwen models and 100 GB or more for Gemma. The setup script
  warns when no volume is mounted.

**Pod type.** A GPU Pod, not Serverless. Secure and Community Cloud both work.

## A pod per command

1. Copy the example profile and adjust its spec:

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

   Hosts refer to the spec by name: `host: {kind: runpod, pod: a100, workdir: ...}`.
2. Put `RUNPOD_API_KEY` in `.env` (from `.env.example`).
3. Make sure you have an SSH key pair at `runpod.identity_file` (default `~/.ssh/id_ed25519`).
   Its public half is passed to the pod at creation, so nothing needs saving in the Runpod account.

Each command that needs the GPU then:

1. Creates a pod named `lfe-<spec>-<UTC stamp>` with the first listed GPU type that is free, on a
   host whose driver supports CUDA 13, and records it in the ledger `artifacts/runpod/created.jsonl`.
   While no GPU is free, creation is retried for `start_timeout_sec`.
2. Terminates it at once if its hourly price is above `max_cost_per_hr`.
3. Prepares it (setup script, code, environment). The model is downloaded on the first server
   start, so allow for that in `inference.startup_timeout_sec`.
4. Runs the command (see [Running](#running)).
5. Terminates the pod when the command ends.

A resumed run gets a new pod, so an interrupted training stage restarts from step 0. If the laptop
dies, the watchdog terminates the pod after `idle_stop_minutes`; if the pod's own API key is not
allowed to terminate, it stops the pod instead. To find and remove leftovers:

```bash
uv run loop pod status  --machines configs/machines/local/runpod-a100.yaml   # created pods still alive
uv run loop pod cleanup --machines configs/machines/local/runpod-a100.yaml --dry-run
uv run loop pod cleanup --machines configs/machines/local/runpod-a100.yaml
```

`cleanup` terminates only pods that are in the ledger and still named `lfe-*`; it never touches
other pods in your account.

## Setup (existing pod)

1. **Save your SSH public key** in Runpod (Settings → SSH Public Keys) *before* creating the pod.
   Runpod writes it to `/root/.ssh/authorized_keys` on every start (via `PUBLIC_KEY`), so a key
   added by hand inside the pod is lost at the next start. If no key was saved, `PUBLIC_KEY` is the
   literal string `null` and full SSH is refused, even though the proxied `ssh.runpod.io` login may
   still work.
2. **Create the pod** with a volume at `/workspace` and TCP port 22 exposed (see
   [Choosing the pod](#choosing-the-pod)). Note its id, e.g. `abc123def4567x`. You can stop it
   straight away; runs start it.
3. **Add your Runpod API key to `.env`** (Runpod console → Settings → API Keys). The key stays on
   this machine; the watchdog uses the pod's own scoped key.

   ```bash
   cp .env.example .env && chmod 600 .env    # then fill in RUNPOD_API_KEY=...
   ```

4. **Create your machine profile** and set `pod_id` (and `identity_file` if your key is not
   `~/.ssh/id_ed25519`):

   ```bash
   cp configs/machines/examples/runpod.yaml configs/machines/local/runpod.yaml
   ```

5. **Check access** (read-only): `uv run loop pod status --machines configs/machines/local/runpod.yaml`
6. **Prepare the pod once.** This starts it, installs tools, syncs the code, builds the environment
   and stops it again:

   ```bash
   uv run loop sync-hosts --machines configs/machines/local/runpod.yaml
   ```

7. **Optional, for a new kind of GPU or image: run the training tests on the pod.** Start it with
   `loop pod start`, use the address `loop pod status` prints, then `loop pod stop`:

   ```bash
   ssh -p <port> root@<ip> 'cd /workspace/learn-from-experience && uv run --extra train pytest -m train tests/train -q'
   ```

A stopped pod can restart only on the machine it was created on. While others use that machine's
GPUs, Runpod refuses the start, sometimes for hours; `loop` retries for `runpod.start_timeout_sec`
and then fails. This is why a pod per command is recommended.

## Running

```bash
uv run loop validate experiments/smoke-mac.yaml --machines configs/machines/local/runpod.yaml \
    --set learner.model_profile=qwen3-1.7b
caffeinate -i uv run loop run experiments/smoke-mac.yaml --machines configs/machines/local/runpod.yaml \
    --set learner.model_profile=qwen3-1.7b --run-id pod-smoke
```

`smoke-mac.yaml` is simply the smallest live experiment; nothing in it is Mac-specific.
`caffeinate -i` keeps a Mac awake. Each command that uses the pod:

1. Starts the pod (or creates one), waits until it has an address and accepts SSH, and prepares
   it (setup script, code, environment; quick when nothing changed).
2. Starts the watchdog on the pod and sends it a heartbeat every `heartbeat_sec`.
3. Runs as usual. Model servers start and stop on the pod behind the run's SSH tunnel; training
   runs there, and each checkpoint is pulled back and hash-checked. Server and training logs are
   copied into the run directory.
4. Stops (or terminates) the pod when the command ends.

**If the laptop sleeps, crashes or loses the network,** the heartbeat stops and the watchdog
stops the pod (a created pod: terminates it) after `idle_stop_minutes`. `loop resume runs/<run-id>`
then starts the pod again (or creates a new one) and continues:

- completed work is kept;
- an interrupted model server is started again;
- an interrupted training stage resumes from the trainer's own checkpoint if the pod's
  `/workspace` survived (a volume); otherwise it restarts from step 0. Every relaunch is recorded
  in `cycles/cycle-NNN/train/relaunches.jsonl`.

Manual control of an existing pod:

```bash
uv run loop pod status --machines configs/machines/local/runpod.yaml
uv run loop pod start  --machines configs/machines/local/runpod.yaml
uv run loop pod stop   --machines configs/machines/local/runpod.yaml
```

**Logs.** Each command's pod events (created or started, ready, prepared, watchdog, stopped or
terminated) go to `runs/<run-id>/logs/pod-lifecycle.jsonl`; for `loop sync-hosts`, which has no
run, to `runs/_pods/pod-lifecycle.jsonl`. Server and trainer logs are copied into the run directory.
[run-layout.md](run-layout.md) lists every file, including the pod-side watchdog log and the SSH
host keys under `artifacts/runpod/`.

## Fixed SSH host mode

For a machine you manage yourself (a Runpod pod you start and stop by hand, or any Linux GPU
machine):

1. Add an alias to `~/.ssh/config` (`HostName`, `Port`, `User root`, `IdentityFile`). For a Runpod
   pod, use the *full* SSH address: the public IP and the port mapped to 22.
2. Prepare the host: `ssh <alias> 'bash -s' < scripts/setup_gpu_host.sh`.
3. In the machine profile, set `host: {kind: ssh, ssh_alias: <alias>, workdir: ...}` and
   `api_base: http://127.0.0.1:8765/v1`. Then run `uv run loop sync-hosts --machines <profile>`.
4. During runs, keep the tunnel open yourself: `ssh -N -L 8765:127.0.0.1:8765 <alias>`.

The run still syncs code before serving and training and copies logs back, but it never starts,
stops or watches the machine. Stop it yourself when you are done.

## Gemma 4 on a pod

`experiments/pilot.yaml` uses Gemma-4-E4B (`configs/models/gemma-4-e4b-it.yaml`). The smaller
Gemma-4-E2B is a fallback: `--set learner.model_profile=gemma-4-e2b-it`.

- **Serving.** `hf_server` serves both with the `gemma4` tool-call parser, using either the
  transformers engine (`inference.backend: hf_transformers`) or the much faster vLLM engine
  ([below](#faster-serving-with-vllm), with measured speeds).
- **Training.** LoRA applies to the language model only (`lora_exclude_modules` skips the vision
  and audio towers). The trainer computes logits only over completion tokens, and `pilot.yaml`
  enables `training.dpo.gradient_checkpointing` with `max_length: 32768`; memory figures are in
  [the GPU table](#choosing-the-pod).
- **Download.** Both checkpoints are public (not gated) at their pinned revisions. The pod
  downloads them on the first server start, so allow a longer `inference.startup_timeout_sec`
  when the pod has no volume.

Before the full pilot, run `loop evaluate` on the base checkpoint and then a one-cycle run. The
evaluation shows whether base E4B solves enough training tasks to yield preference pairs, and the
one-cycle run shows the time and cost of a cycle, both before you pay for every cycle.

## Faster serving with vLLM

The transformers engine generates one token at a time from Python, so model time dominates an
episode. With `inference.backend: vllm` (example: `configs/machines/examples/runpod-a100-vllm.yaml`),
`hf_server` hands generation to a vLLM child process but still renders every prompt with
`training/render.py`, parses tool calls and counts tokens with the training tokenizer. What the
learner sees and what is counted therefore stay the same as with the transformers engine.

- On first use the pod installs the model profile's `serving.vllm.engine_package` into a
  separate environment, because vLLM pins its own torch. It lives in `.engines/` in the pod's
  repository copy, in a directory named after the package with characters other than letters,
  digits, `.`, `_` and `-` replaced by `_` (`vllm==0.30.0` → `.engines/vllm__0.30.0/`). The
  install takes a few minutes on a new pod and counts against `inference.startup_timeout_sec`.
- Every sampling parameter is sent explicitly with each request (`--generation-config vllm`), so
  the model repository's own sampling defaults never apply. Prefix caching is off, because a
  cached prefix is computed differently from a recomputed one and would change outputs.
- In experiments that train LoRA adapters, base checkpoints are served through a
  [zero LoRA](glossary.md) (as with the transformers engine), so every checkpoint pays the same
  adapter overhead.
- `inference.request_concurrency` sets how many requests vLLM batches together.
  Set `docker_concurrency` at least as large; otherwise too few episodes run at once to fill a
  batch.

Gemma-4-E4B on an A100 with the zero LoRA, 768-token generations:

| Engine | Server start | Tokens/s |
|---|---|---|
| transformers (`hf_transformers`) | about 1 min | 10.2 |
| vLLM with the profile's `launch_args` (full decode CUDA graphs, no torch.compile) | 88 s | 76.2 for one request; 505 in total with 8 concurrent requests |
| vLLM defaults (`-O2`, torch.compile) | 210 s | 81.3 |

vLLM uses its own kernels (for Gemma, bf16 throughout, including the LoRA arithmetic), so its
outputs differ slightly from the transformers engine. Before relying on it for a model profile,
check that it matches, on a GPU host in the repository directory, with a trained adapter of that
model on that host (without `--adapter` the check fails):

```bash
uv run --extra train python -m learning_loop.serving.equivalence --profile gemma-4-e4b-it \
    --adapter runs/<run>/checkpoints/<ckpt> --out equivalence.json
```

It exits with status 1 unless vLLM echoes back exactly the prompt token ids it was sent, its
per-token log-probabilities match transformers + PEFT within set bounds for the base model
(through a zero LoRA) and the trained adapter, and a strong random adapter changes both engines'
outputs in the same way. Greedy-decoding agreement is only reported. The bounds and the report
fields are in `serving/equivalence.py`.

## Costs

- A pod per command costs nothing between commands, but each command spends a few minutes
  before its real work: on an A100 pod, about 25 s until SSH works, about 30 s to prepare the
  pod, and about a minute to download and load Gemma-4-E4B on the first server start.
- A stopped existing pod still bills its volume as storage. Terminate existing pods you no longer
  need in the Runpod console.
- For existing pods, `stop_when_done: false` keeps the pod running between commands (faster
  successive commands). The watchdog still stops it once no command has sent a heartbeat for
  `idle_stop_minutes`. Created pods are always terminated.

## Troubleshooting

Pod error messages and fixes are in
[operations.md → Runpod pods](operations.md#runpod-pods); general problems are in
[operations.md](operations.md#troubleshooting).
