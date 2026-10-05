# Learning Loop

Can an LLM agent learn from its own experience by updating its own weights? This repository runs
one repeatable loop. Each pass is a **cycle**:

1. **Collect**: the learner model, fixed for the cycle, makes several seeded attempts at
   command-line tasks in Harbor/Docker containers. Every event, outcome and token count is logged.
2. **Improve and verify**: a fixed **editor** model proposes one cheaper tool call in a successful
   attempt. The task is rebuilt up to that point in fresh containers, the original and the edited
   call each run, and the learner continues from both. If both sides succeed and the edit saves
   tokens, it becomes a *preference pair*.
3. **Train**: LoRA DPO (TRL + PEFT) trains the learner's adapter further on those pairs, against the
   learner as it was at the start of the cycle. The new checkpoint becomes the next learner.

The question: does this reduce the tokens the agent uses while keeping its success rate, including
on tasks it never trained on? Method and metrics: [docs/experiment.md](docs/experiment.md). Terms:
[glossary](docs/glossary.md). No claim is made that the method is novel or that it works.

## Status

| Component | Status |
|---|---|
| Task families (`log-triage`, `count-errors`, `csv-revenue`, `fix-stats`) as specs, generation checks, renderer (`network_mode: none`, fixed env and file times), shared grader, environment probe, splits | working; every family checked in Docker (oracle, no-op and shortcuts score their predicted rewards, oracle replay) |
| Episode loop, replay, branching, acceptance, preference export | working on the local fixture backend and Docker |
| Orchestration, resume, infra retries, controls, reports, `loop dashboard` | working on two-cycle fixture runs; dashboard also used on a live Runpod run |
| LoRA DPO (Qwen3-0.6B), publish and reload, serving through the reference HF server | working on MPS and CUDA |
| Live small-model loop (`experiments/smoke-mac.yaml`) | runs end to end; Qwen3-0.6B and 1.7B solved none of their 6 attempts per run, so no live cycle has trained yet |
| Live editor on saved trajectories (`loop edit-replay`) | runs; all small-Qwen proposals so far failed validation |
| Runpod / SSH GPU host, laptop as coordinator | fixed SSH host: live runs and remote training on an RTX 3090 pod; pod per command: used by `loop run` for the Gemma-4-E4B pilot runs on A100 pods; existing pod and pod watchdog: fake API only |
| Gemma 4 | E4B: two pilot learning runs on A100 pods. With the transformers engine, cycle 0 trained an adapter (on one accepted pair) that was served in cycle 1, where the run stopped. With the vLLM engine, three cycles completed without an accepted edit, so nothing trained. Tool-call parser tested; E2B **untested** |
| vLLM generation engine behind `hf_server` | matches transformers + PEFT for E4B on an A100 (`serving/equivalence.py`: base, trained and random adapters); used for the vLLM pilot run, which never served a trained adapter |
| `loop submit` (remote coordinator); `loop external-eval --execute` | **untested** (external-eval: only job generation is tested) |

## Setup

Requires macOS or Linux, [uv](https://docs.astral.sh/uv/), and a running Docker daemon.

```bash
cp .env.example .env     # credentials (git-ignored; loaded automatically by `loop`)
uv sync                  # runner/coordinator (Harbor 0.23.0, no torch)
uv sync --extra train    # + torch/peft/trl for training and the reference HF server
uv run pytest            # fast default tests (no Docker, no model)
uv run loop preflight --trainable   # Docker, disk, memory, model cache, MPS, one LoRA step
```

Configs name credential variables (`api_key_env`), never values. Put machine profiles for your own
hosts in `configs/machines/local/` (git-ignored), starting from `configs/machines/examples/`.

## Running

`M` is a machine profile path. Every command and flag is in [docs/cli.md](docs/cli.md).

```bash
# Check configs and print the planned workload (starts nothing)
uv run loop validate experiments/pilot.yaml --machines configs/machines/examples/lab-gpu.yaml

# Run, resume, watch, inspect
uv run loop run experiments/smoke-mac.yaml --machines configs/machines/examples/mac-local.yaml
uv run loop resume runs/<run-id>
uv run loop dashboard runs/<run-id> --open       # http://localhost:8090/; omit the run for all runs
uv run loop status runs/<run-id>
uv run loop report runs/<run-id>                 # -> runs/<run-id>/reports/

# Paired comparison of runs; several loop seeds per side with --vs (effects: --vs side minus first side)
uv run loop compare runs/<baseline-id> runs/<experiment-id>
uv run loop compare runs/<base-s0> runs/<base-s1> --vs runs/<exp-s0> runs/<exp-s1>
uv run loop compare runs/<baseline-id> --vs runs/<exp-s0> runs/<exp-s1>   # one frozen baseline

# One stage of one cycle: eval|collect|edit|verify|dataset|train
uv run loop stage runs/<run-id> --cycle 0 --stage collect

# Evaluate a checkpoint in a separate run (final-test panels need --final)
uv run loop evaluate --experiment experiments/pilot.yaml --machines M \
    --checkpoint runs/<run-id>/checkpoints/<ckpt> --panels final-same-family --final

# Controls and editor comparisons
uv run loop run experiments/frozen-baseline.yaml --machines M
uv run loop run experiments/fixed-dataset-control.yaml --machines M \
    --set training.fixed_dataset=runs/<pilot-run>/cycles/cycle-000/dataset
uv run loop edit-replay runs/<run-id> --cycle 0 --experiment experiments/ablations/editor-external.yaml --machines M

# External benchmark (writes a pinned Harbor job; --execute runs it)
uv run loop external-eval --dataset terminal-bench@<version> --checkpoint <ckpt-dir> --machines M --n-tasks 10

# Runpod: each command creates a pod and terminates it (docs/runpod.md)
caffeinate -i uv run loop run experiments/pilot.yaml --machines configs/machines/local/runpod-a100.yaml
uv run loop pod cleanup --machines configs/machines/local/runpod-a100.yaml   # after a crash

# Linux coordinator over SSH
uv run loop submit experiments/pilot.yaml --machines configs/machines/local/lab.yaml
uv run loop remote-status <run-id> --machines configs/machines/local/lab.yaml
uv run loop fetch <run-id> --machines configs/machines/local/lab.yaml
```

`--set key.path=value` overrides an experiment value (recorded in `run.json`). To run the agent on
the tasks outside the loop, use `evaluation/run.sh` ([evaluation/README.md](evaluation/README.md)).

## Smoke tests (bounded, laptop-safe)

| Command | What it exercises | Time |
|---|---|---|
| `uv run loop smoke fixture` | scripted learner and editor, fixture trainer, local backend, 2 cycles | seconds |
| `uv run loop smoke fixture-docker` | the same in real Harbor containers | ~5 min |
| `uv run loop smoke train` | real LoRA DPO (Qwen3-0.6B, MPS, 2 steps) on **labeled fixture** pairs, reload check, adapter serving, one Docker episode | ~2 min |
| `uv run loop smoke live` | the real small learner and editor on one short task | ~10-20 min |

A passing smoke test is an engineering check, not evidence that learning, Gemma or a GPU host
works.

The editor needs successful trajectories to edit, and the small live learners may produce none.
To try the real editor anyway, run it over the successful scripted-learner episodes of a kept
Docker fixture run:

```bash
uv run loop smoke fixture-docker --keep
uv run loop edit-replay runs/_smoke/fixture-<stamp> --cycle 0 --experiment experiments/fixture-two-cycles.yaml \
    --machines configs/machines/examples/fixture-docker-live-editor.yaml \
    --set editor.mode=initial_policy --set editor.scripted_path=null
```

To run the live smoke experiment with the 1.7B learner, add
`--set learner.model_profile=qwen3-1.7b` to `loop run experiments/smoke-mac.yaml`.

## Documentation

The [documentation index](docs/index.md) lists every document. `uv run loop docs --open` serves
them rendered at http://localhost:8000/.

What each top-level folder holds is in
[architecture.md → Repository layout](docs/architecture.md#repository-layout).
