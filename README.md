# Learn from Experience

Exploring how an LLM agent can learn from its own experience by updating its own weights. The
repository implements one concrete, repeatable loop:

1. **Collect**: a frozen learner checkpoint attempts Harbor/Docker CLI tasks (several seeded
   attempts per instance), recording lossless event logs, success, partial reward and resource use.
2. **Improve and verify**: a fixed editor model proposes one more efficient tool action in a
   successful trajectory. The original decision state is restored in fresh containers, the original
   and the edited action are executed in separate environments, and the same frozen learner
   continues from both. Edits that keep complete success and save tokens become preference pairs.
3. **Train**: LoRA DPO (TRL + PEFT) continues the learner's adapter, with the learner frozen at the
   start of the cycle as the reference. The new checkpoint is published immutably, evaluated
   independently, and becomes the next learner.

The research question: does learning from verified efficiency edits reduce inference-token use
while preserving success, including on held-out instances and families? The method, protocol and
metrics are in [docs/experiment.md](docs/experiment.md). Nothing here claims the method is novel or
that gains are guaranteed.

## Status

| Component | Status |
|---|---|
| Harbor tasks (`log-triage`, `fix-stats`), generators (+ `csv-revenue`, `count-errors`), splits, separate-verifier grading | working; oracle/nop and false-positive checks run in Docker |
| Episode loop, event logs, deterministic replay, branching, acceptance, preference export | working; tested on the local fixture backend and on Docker |
| Cycle orchestration, resume, infra retries, lineage, frozen/fixed-dataset controls, reports | working; two-cycle fixture runs on the local backend and on Docker |
| Live run dashboard (`loop dashboard`) | working; tested on fixture runs and used read-only on a live Runpod pilot run |
| LoRA DPO (Qwen3-0.6B, MPS) with incoming-adapter reference, publish/reload, HF serving | working on this Mac (real 2-step training smoke) |
| Live small-model loop (`experiments/smoke-mac.yaml`) | runs end to end; Qwen3-0.6B and 1.7B solved 0/6 smoke attempts, so cycles were recorded no-updates (see "Smoke tests") |
| Live editor on real saved trajectories (`loop edit-replay`) | runs; small Qwen editors produced only proposals rejected by validation so far |
| Runpod / SSH GPU host for serving and training (laptop coordinator) | fixed-SSH-host mode validated on one RTX 3090 pod (CUDA training tests, live runs, remote training + adapter serving); pod-per-command mode (create, use, terminate) and existing-pod start/stop tested against a fake Runpod API; live use recorded in [docs/runpod.md](docs/runpod.md#what-was-validated) |
| Gemma-4-E4B: serving (`hf_server`, transformers or vLLM engine) and LoRA DPO training on an A100 | exercised in probes and a partial pilot run (see [docs/runpod.md](docs/runpod.md)); vLLM checked against transformers + PEFT with `serving/equivalence.py` |
| Gemma-4-E2B; llama.cpp serving; SSH coordinator (`loop submit`) | implemented or declared, **untested** |

## Setup

Requires macOS or Linux, [uv](https://docs.astral.sh/uv/), and a running Docker daemon.

```bash
cp .env.example .env     # credentials (git-ignored; loaded automatically by `loop`)
uv sync                  # runner/coordinator (Harbor 0.23.0, no torch)
uv sync --extra train    # + torch/peft/trl for training and the reference HF server
uv run pytest            # fast default tests (no Docker, no model)
uv run loop preflight --trainable   # Docker, disk, memory, model cache, MPS, one LoRA step
```

Dependencies are pinned in `pyproject.toml` and locked in `uv.lock`. Machine profiles with private
hosts go in `configs/machines/local/` (git-ignored); start from `configs/machines/examples/`.
Credentials live in `.env` (template `.env.example`, loaded by every `loop` command); configs only
name the variables (`api_key_env`), and SSH uses keys/aliases from `~/.ssh/`.

## Running

```bash
# Validate: resolve configs/splits/profiles and print the planned workload (starts nothing)
uv run loop validate experiments/pilot.yaml --machines configs/machines/examples/lab-gpu.yaml

# Run all cycles, resume after an interruption, inspect
uv run loop run experiments/smoke-mac.yaml --machines configs/machines/examples/mac-local.yaml
uv run loop resume runs/<run-id>
uv run loop dashboard runs/<run-id> --open       # live read-only page at http://localhost:8090/ (no RUN_DIR: all runs)
uv run loop status runs/<run-id>
uv run loop report runs/<run-id>                 # -> runs/<run-id>/reports/report.md + CSVs

# Compare two runs on shared panels (paired by instance/attempt seed); several loop seeds per side
uv run loop compare runs/<baseline-id> runs/<experiment-id>
uv run loop compare runs/<base-s0> runs/<base-s1> runs/<base-s2> --vs runs/<exp-s0> runs/<exp-s1> runs/<exp-s2>

# Individual stages inside a run (same manifests as the full loop)
uv run loop stage runs/<run-id> --cycle 0 --stage collect   # eval|collect|edit|verify|dataset|train

# Evaluate any saved checkpoint (final-test panels need --final; results never feed training)
uv run loop evaluate --experiment experiments/pilot.yaml --machines M \
    --checkpoint runs/<run-id>/checkpoints/<ckpt> --panels final-same-family --final

# Controls and editor comparisons
uv run loop run experiments/frozen-baseline.yaml --machines M
uv run loop run experiments/fixed-dataset-control.yaml --machines M \
    --set training.fixed_dataset=runs/<pilot-run>/cycles/cycle-000/dataset
uv run loop edit-replay runs/<run-id> --cycle 0 --experiment experiments/ablations/editor-external.yaml --machines M

# External benchmark against a saved checkpoint (writes a pinned Harbor job; --execute to run)
uv run loop external-eval --dataset terminal-bench@<version> --checkpoint <ckpt-dir> --machines M --n-tasks 10

# Runpod GPU for serving + training: a pod created and terminated by each command (see docs/runpod.md)
caffeinate -i uv run loop run experiments/pilot.yaml --machines configs/machines/local/runpod-a100.yaml
uv run loop pod cleanup --machines configs/machines/local/runpod-a100.yaml   # leftovers after a crash

# Linux coordinator over SSH (Docker runs there); fetch results back
uv run loop submit experiments/pilot.yaml --machines configs/machines/local/lab.yaml
uv run loop remote-status <run-id> --machines configs/machines/local/lab.yaml
uv run loop fetch <run-id> --machines configs/machines/local/lab.yaml
```

`--set key.path=value` overrides an experiment value; overrides are recorded in `run.json`.
Every command, flag and exit code is listed in [docs/cli.md](docs/cli.md) (`uv run loop <command> --help`
shows the same text; `uv run loop docs` serves all documentation in the browser).
The original Harbor workflow still works: `evaluation/run.sh` runs the learner agent on the tasks
against any OpenAI-compatible server (see [evaluation/README.md](evaluation/README.md)).

## Smoke tests (bounded, laptop-safe)

| Command | What it exercises | Typical time |
|---|---|---|
| `uv run loop smoke fixture` | scripted learner/editor + fixture trainer, local fixture backend, 2 cycles | seconds |
| `uv run loop smoke fixture-docker` | the same against real Harbor containers | ~5 min |
| `uv run loop smoke train` | real LoRA DPO (Qwen3-0.6B, MPS, 2 steps) on **labeled fixture** pairs, publish + reload check, serve the adapter, one real Docker task episode | ~2 min |
| `uv run loop smoke live` | the real small learner/editor loop on one short task | ~10-20 min |

Fixture components are labeled everywhere they appear. A passing smoke test is an engineering
check: it is not evidence of learning efficacy, Gemma compatibility or GPU readiness. Zero
accepted edits in the live smoke is a valid outcome, recorded as a no-update cycle.

On this laptop the small learners did not solve the smoke task (they count with `grep -i`, or
claim to have written `/app/answer.txt` without calling a tool), so the live loop has so far
ended in no-update cycles. To exercise the real editor on real trajectories anyway, run it over
saved successful trajectories (verification then continues with that run's scripted learner):

```bash
uv run loop smoke fixture-docker --keep
uv run loop edit-replay runs/_smoke/fixture-<stamp> --cycle 0 --experiment experiments/fixture-two-cycles.yaml \
    --machines configs/machines/examples/fixture-docker-live-editor.yaml \
    --set editor.mode=initial_policy --set editor.scripted_path=null
```

`--set learner.model_profile=qwen3-1.7b` swaps in the 1.7B learner (same template, pinned).

## Documentation

Start at the [documentation index](docs/index.md), which lists every document by what you want
to do (architecture, method, CLI and configuration references, operations and troubleshooting,
run layout, glossary, tasks, developer guide). Browse it rendered, with working links and
diagrams, at http://localhost:8000/:

```bash
uv run loop docs --open
```

Top-level folders: `experiments/` (experiment YAML), `configs/` (model and machine profiles),
`evaluation/` (Harbor side), `src/learning_loop/` (the loop), `prompts/`, `tests/`, `docs/`, and
the git-ignored outputs `runs/` and `artifacts/`.
