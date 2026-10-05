# Glossary

## Tasks and data

| Term | Meaning |
|---|---|
| **Harbor** | The external framework that runs a task: it starts the task's container, lets an agent act in it, then grades the declared artifacts in a separate verifier container ([evaluation/README.md](../evaluation/README.md)). |
| **Family** | A kind of task whose instances are drawn from a seed, defined by one module in `evaluation/families/` (`FAMILY = Family(...)` with `build(ctx) -> TaskSpec`), e.g. `log-triage`. Families are grouped into clusters of similar skills (`Family.cluster`). |
| **Difficulty** | One of a family's parameter sets: `easy`, `medium` or `hard`. |
| **Task instance** | One concrete Harbor task, id `family/difficulty/sSEED`: the directory the renderer writes for that family, difficulty and seed. |
| **Task spec** | What a family's `build` returns (`TaskSpec` in `src/learning_loop/tasks/spec.py`): instruction, learner-visible files, grader with its hidden answer key, oracle and shortcuts. The renderer turns it into the task directory. |
| **Oracle / shortcut** | The reference solution of a task / a declared wrong method that must fail on every instance. Each is a `Solution` with a `shell` form that runs in the container and a `model` that predicts, in Python, the artifacts the shell form writes. |
| **Profile** | The container environment of a family (`PROFILES` in `spec.py`): pinned base image, environment variables, hostname, required tools, resources and allowed processes. |
| **Environment probe** | `src/learning_loop/tasks/runtime/probe.py`: checks that a running container matches its profile (no network, environment variables, hostname, tools, processes) and, in the agent container, that `/app` holds exactly the rendered files. The verifier writes no reward when it fails; an episode stops as `infra:env_probe:<checks>` before turn 0. |
| **Calibration seed** | A seed in 900000-999999 (`CALIBRATION_SEEDS` in `tasks/instances.py`). Calibration runs a model on instances from these seeds to tune a family's difficulty; split validation rejects them, so no trained or evaluated instance was looked at while tuning. |
| **Docker-host preflight** | The check at the start of every run (`orchestration/preflight.py`): the oracle of one instance per family runs on the run's Docker host and must score its predicted reward. Not the same as `loop preflight`, which checks a machine's Python, model cache and accelerator for the smoke test. |
| **Skill** | A family tag (e.g. `gzip`) used only to group reports. |
| **Split** | An instance's single purpose, set in its split file: `train` (training data), `dev` (evaluated every cycle), or `final`/`external` (evaluated only with `loop evaluate --final`, once method choices are frozen). |
| **Panel** | A named list of same-split instances in a split file (`evaluation/splits/*.yaml`). Experiments name panels (`tasks.collection_panel`, `evaluation.dev_panels`, `evaluation.final_panels`), not splits. |
| **Held-out family** | A family in a split file's `held_out_families`: it may have no train-split instance, so it measures transfer to families never trained on. |
| **Exposure schedule** | `tasks.exposure_schedule`: which families collection uses from which cycle on, for studying retention of earlier families. |
| **Artifact** | A file listed in `task.toml` `artifacts` (e.g. `/app/answer.txt`); the only thing copied from the learner's container to the verifier container. |
| **Complete success** | `reward >= success_threshold` (in `task.toml`; 1.0 for all shipped tasks). Only this counts as success, makes an episode editable, or passes acceptance. |
| **Partial reward** | The verifier's `reward` value from 0 to 1, e.g. 0.33 when a third of fix-stats' hidden checks pass. Reported next to success; a value below the threshold is not success. |
| **Oracle / nop** | Harbor agents that run the reference solution (must score 1.0) / do nothing (the baseline score). |

## Episodes and runs

| Term | Meaning |
|---|---|
| **Learner** | The model being trained: a base model plus, once trained, a LoRA adapter. |
| **Episode** | One learner attempt at one instance in a fresh environment, then grading (id `ep-...`). Verification branches are episodes too (*role* `branch`). |
| **Trajectory** | The recorded turns, tool calls and observations of one episode. |
| **Harbor trial** | Harbor's unit of execution; with `environment_backend: harbor_docker` each episode runs as one trial. |
| **Attempt** | The n-th seeded episode on one instance in a stage (`attempt_index`); evaluation seeds are the same for every checkpoint. |
| **Role** | Why an episode ran: `eval` (unassisted evaluation), `collect` (training experience) or `branch` (verification; never counted as evaluation). |
| **Turn** | One request to the learner and its reply, plus running the reply's tool calls; the tool outputs go into the next request (`turn_index`, from 0). |
| **Stop reason / category** | Why an episode ended, e.g. `budget:max_turns` (full list in [evaluation/README.md](../evaluation/README.md)). Categories: `model` (replied without a tool call), `budget` (turn, token or output limits), `safety` (wall clock), `model_error` (request rejected, or no tool call parsed), `infra` (environment or endpoint failure; retried), `replay` (restored state differs from the source). |
| **Usage source** | Where an episode's token counts come from (`usage.source`): `provider` (reported by the model server), `fixture_estimate` (a fixture's character-based estimate), `none` (no usage measured) or `mixed` (several sources summed). |
| **Run** | One directory under `runs/` made by one command; `run.json` `kind` is `learning` (`loop run`), `evaluation` (`loop evaluate`) or `edit_replay` (`loop edit-replay`). |
| **Smoke run** | A short engineering check made by `loop smoke <level>` under `runs/_smoke/` ([README](../README.md)); it never supports research claims. |
| **Cycle** | One pass of eval → collect → edit → verify → dataset → train with a fixed learner; `cycles: N` trains N times and evaluates N+1 checkpoints, and the last cycle only evaluates. |
| **No-update cycle** | A cycle with no pair to train on (none accepted, or none fits `training.dpo.max_length`); the learner carries over unchanged. The last, evaluation-only cycle is not one. |
| **Stage** | One step of a cycle with its own directory under `cycles/cycle-NNN/`: `eval`, `collect`, `edit`, `verify`, `audit` (optional), `dataset`, `train`. |
| **Manifest** | The `manifest.json` of an `eval`, `collect`, `edit`, `verify` or `audit` stage: its work items and their status. Resuming skips finished items. |
| **Work item** | One unit of stage work with a stable id: an episode (`ep-`), proposal (`prop-`) or verification (`ver-`). Each rerun after an interruption or infra failure is a new *execution*; earlier ones are kept as `<id>.interrupted-N`. |
| **Condition** | One way of running the experiment that results compare: the `condition` setting (`learning`, or a **control**: `frozen_baseline`, never trained, or `fixed_dataset`, training only on one frozen export) or a variant such as another `editor.mode` ([experiment.md](experiment.md#conditions-and-controls)). Runs differing only in loop seed are repeats of one condition. |
| **Loop seed** | `seeds.loop_seed`: picks one independent repeat of a learning run (its collection, editing and training draws); evaluation seeds do not depend on it. |

## Editing and verification

| Term | Meaning |
|---|---|
| **Source (trajectory)** | A collected episode with complete success, given to the editor. |
| **Editor** | The model that reviews a source and answers with one tool call: `replace_with_<tool>` proposes a replacement for one turn's tool call, `abstain` declines ([details](experiment.md#editing)). By default the fixed initial checkpoint (`editor.mode: initial_policy`). |
| **Proposal** | The editor's output for one source: `proposed`, `abstained` (the editor called `abstain`, or the source had no eligible turn and was not sent: `skipped:no_editable_turns`) or `invalid` (unusable response or failed the code checks, with rejection reasons). |
| **Eligible (editable) turn** | A turn the editor may replace: generated by the learner (not replayed), with exactly one tool call that parsed cleanly, entered the history unchanged (no recorded repair) and ran, and no assistant text or reasoning ([why](experiment.md#editing)). |
| **Hindsight constant** | A value in a replacement (IP address, number of 3+ digits, quoted string, file path, or any number it writes or prints) that appears in the episode after the edited turn but not before it. The grounding check rejects it as `ungrounded_constant:<value>` ([details](experiment.md#editing)). |
| **Intervention** | The fixed action at the edited turn of a branch: the original call (`original` branch) or the replacement (`edited` branch). |
| **Replay** | Rebuilding the state before the edited turn in a fresh environment by re-running the source's earlier tool calls without a model; any observation or fingerprint mismatch stops the branch (it *fails closed*). |
| **Normalizer** | A regex replacement declared in `task.toml` (`observation_normalizers`) that removes run-to-run noise (e.g. `ls -l` times) before replay compares observations; replay applies no others. |
| **Fingerprint** | A hash of the files under the task's `fingerprint_paths` (content, type, permissions, owner, symlink target) and the working directory; modification times are not included. |
| **Branch** | One side of a verification: replay, intervention, then a fresh learner continuation. Each proposal gets an `original` and an `edited` branch. |
| **Continuation** | The learner's own turns after the intervention. Both branches of a repetition use the same seed; `verification.continuations_per_branch` sets the repetitions. |
| **Counterfactual episode token cost** | A branch's cost as if it were a whole episode: the source's tokens before the edited turn, the edited turn's request input, the fixed turn's rendered length, and the continuation's usage ([details](experiment.md#verification-and-acceptance)). |
| **Saving** | Mean original-branch cost minus mean edited-branch cost over repetitions. |
| **Acceptance rule** | The criteria for a preference pair; only `strict_all_success_v1` ([experiment.md](experiment.md#verification-and-acceptance)). |
| **Evidence label** | How much evidence backs an accepted edit: `one_observed_successful_preference` with one continuation per branch, `all_<N>_continuation_pairs_successful` with N. |
| **Audit** | Re-verifying a random share of accepted edits with fresh seeds; reported only, never changes a dataset. |

## Training and models

| Term | Meaning |
|---|---|
| **Preference pair** | One training example `{prompt, chosen, rejected, tools}`: the conversation before the edited turn, the edited turn, the original turn and the learner's tool schemas. |
| **Provenance record** | The companion to a preference pair holding its origin and verification evidence; never training input. |
| **Export / dataset** | A read-only directory of pairs (`preferences.jsonl`, `provenance.jsonl`, `manifest.json`): `dataset/current/` is the cycle's new pairs, `dataset/` the training set. |
| **History buffer** | The pool of earlier cycles' pairs (at most `training.data.buffer_capacity`) that `selection: current_and_history` samples into each training set; its pairs are not re-verified. |
| **Dropped pair** | An exported pair the trainer skipped (e.g. longer than `training.dpo.max_length`), counted with its reason. |
| **Fixture** | A labeled stand-in for testing the plumbing without a model: scripted policy or editor, fixture trainer, hand-written pairs. Its pairs are labeled `fixture`, never `verified`. |
| **Base model / revision** | `base_model` and `base_revision` in the model profile: the trainable Hugging Face model at an exact commit. |
| **Serving artifact** | Weights a serving backend loads instead of the base model (`artifact`, e.g. a Q8_0 GGUF); not trainable and not bit-identical to the base model. |
| **LoRA adapter** | Small trained matrices added to chosen weight matrices of the frozen base model (each becomes W + B·A, scaled), stored in PEFT format. Training changes only the adapter. |
| **Checkpoint** | A read-only learner identity: `base:<profile>@<rev12>`, or base plus adapter `cNNN-<hash12>` under `runs/<id>/checkpoints/`. |
| **Incoming checkpoint** | The learner *L_c* at the start of cycle *c*; training continues its adapter, and it is the DPO reference. |
| **Reference** | The fixed model DPO measures change against ([experiment.md](experiment.md#preference-data-and-training)): always the incoming checkpoint including its adapter. |
| **Reload check** | Before publishing, the adapter is reloaded from its files onto a fresh base model and must reproduce the trained weights and log-probabilities within tolerance. |
| **Zero LoRA** | An adapter of the experiment's LoRA shape whose B matrices are all zero, so it adds nothing to the weights. With `training.trainer: trl_dpo`, base checkpoints are served through it, so they pay the same adapter overhead as trained ones. |
| **Lineage** | Parent and reference checkpoint ids, in `checkpoint.json` and `cycle.json`. |

## Machines and configuration

| Term | Meaning |
|---|---|
| **Coordinator** | The `loop` process driving a run (`loop run`, `loop resume`, ...); it owns the run directory, Docker work and model-server swaps. |
| **Environment / serving backend** | Where episodes run (`environment_backend`: `harbor_docker` or `local_fixture`) / the software serving the model (`inference.backend`: `hf_transformers`, `vllm`, `llama_cpp` or `scripted`). Managed serving uses `hf_transformers` or `vllm`; both run this repository's `hf_server`, which renders prompts, parses tool calls and counts tokens; with `vllm` a vLLM child process generates the tokens. `llama_cpp` names a llama.cpp server you run yourself, used only as an external endpoint. |
| **Managed / external / scripted inference** | `inference.mode`: the run starts its own model server / uses a running endpoint serving one declared checkpoint / uses a fixture policy. |
| **Pod lifecycle** | For each command using a `kind: runpod` host: start the existing pod (`pod_id`) or create one (`pod:`, a spec in `runpod.create`), keep its heartbeat fresh, and at the end stop the existing pod (unless `stop_when_done: false`) or terminate the created one. |
| **Created pod** | A pod `loop` created, named `lfe-<spec>-<stamp>` and recorded in the *ledger* `artifacts/runpod/created.jsonl`; the only kind of pod `loop` terminates. |
| **Watchdog** | A process on the pod that stops it (terminates a created pod) when the coordinator's heartbeat is older than `idle_stop_minutes`. |
| **Model profile / machine profile / experiment** | YAML for model identity (`configs/models/`) / deployment (`configs/machines/`) / scientific choices (`experiments/`) ([configuration.md](configuration.md)). |
