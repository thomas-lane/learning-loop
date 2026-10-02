# Glossary

Terms as this repository uses them. Several are easy to confuse; the "not to be confused with"
notes are the point of this page.

## Tasks and data

| Term | Meaning |
|---|---|
| **Family** | A kind of task with one generator, e.g. `log-triage`, `fix-stats`, `csv-revenue`, `count-errors`. |
| **Task instance** | One concrete Harbor task directory, id `family/difficulty/sSEED` (or `family/static` for the hand-written tasks). Content-hashed. Not to be confused with an *attempt* at it. |
| **Difficulty** | `easy` / `medium` / `hard`: a generator parameter set within a family. |
| **Skill** | A tag on a family (e.g. `gzip`, `debugging`), used only to group reports. |
| **Split** | Which purpose an instance serves: `train`, `dev`, `final` or `external`. Every instance has exactly one. |
| **Panel** | A named list of instances of one split, declared in a split file (`evaluation/splits/*.yaml`), e.g. `train-easy-only`, `dev-medium-hard`. Experiments refer to panels, not splits. |
| **Held-out family** | A family that may never appear in a train panel; evaluated only on final panels. |
| **Artifact** | A file the task declares in `task.toml` (e.g. `/app/answer.txt`); the only thing copied from the agent container to the verifier container. |
| **Complete success** | `reward >= success_threshold` (1.0 for all shipped tasks). The gate for editing, acceptance and success rates. |
| **Partial reward** | The verifier's `reward` value, e.g. 0.33 when fix-stats' no-op passes 3 of 9 checks. Reported separately; never counted as success. |
| **Oracle / nop** | Harbor's reference-solution agent and do-nothing agent; they check that a task is solvable and what a no-op scores. |

## Episodes and runs

| Term | Meaning |
|---|---|
| **Episode** | One learner interaction with one fresh task environment, from the first request to a stop, plus grading. Has an `ep-...` id. |
| **Harbor trial** | Harbor's unit of execution (environment + agent + verifier) that runs one episode on the Docker backend. Its directory sits inside the episode's item directory. |
| **Attempt** | The n-th independently seeded episode of one instance within a stage (`attempt_index`). Evaluation attempt seeds are identical across checkpoints. |
| **Role** | Why an episode ran: `eval` (unassisted evaluation), `collect` (training experience), `branch` (verification continuation; never counted as an evaluation). |
| **Turn** | One assistant message in an episode (0-based `turn_index`), usually with tool calls. |
| **Stop reason / category** | Why an episode ended, e.g. `budget:max_turns` in category `budget`. Categories: `model`, `budget` (experimental limits), `safety` (wall clock), `model_error`, `infra`, `replay`. |
| **Run** | One directory under `runs/` created by one command: a learning run, an evaluation run or an edit-replay run (`run.json` `kind`). One learning run has one loop seed. |
| **Cycle** | One pass of eval -> collect -> edit -> verify -> dataset -> train with a frozen learner. `cycles: N` runs N training stages (some may be no-updates) and evaluates the learner N+1 times. |
| **Stage** | One step of a cycle (`eval`, `collect`, `edit`, `verify`, `audit`, `dataset`, `train`), with its own directory and manifest. |
| **Work item** | One unit of stage work with a stable id (`ep-`, `prop-`, `ver-`), tracked in the stage manifest. An item can have several *executions* after interruptions or infra retries. |
| **Condition** | The experimental arm: `learning`, `frozen_baseline`, `fixed_dataset`. Editor and verification variants are configuration, not conditions. |
| **Loop seed** | `seeds.loop_seed`: the independent replicate of a learning run. Method comparisons use several loop seeds; evaluation seeds do not depend on it. |

## Editing and verification

| Term | Meaning |
|---|---|
| **Source (trajectory)** | A completely successful collection episode given to the editor. |
| **Editor** | The component (and the model behind it) that proposes one replacement tool call for one turn of a source. Default: the fixed initial checkpoint (`initial_policy`). |
| **Proposal** | The editor's output: `proposed`, `abstained` or `invalid` (with rejection reasons). |
| **Eligible turn** | A model turn with exactly one well-formed, unrepaired tool call and no assistant text or reasoning. |
| **Grounding** | The heuristic that rejects hindsight constants in a replacement (`ungrounded_constant:<value>`). A filter, not a proof. |
| **Intervention** | The fixed action at the branch point: the original call (`original` branch) or the replacement (`edited` branch). |
| **Replay** | Rebuilding the decision state in a fresh environment by re-executing the source's earlier actions without calling any model, checking observations and fingerprints. Fails closed. |
| **Fingerprint** | A hash of the task's declared state (file content, types, permissions, owners, symlinks, working directory). |
| **Branch** | One side of a verification: replay -> intervention -> fresh continuation of the learner. |
| **Continuation** | The learner's own turns after the intervention in a branch. Original and edited branches use the same continuation seed. |
| **Counterfactual episode token cost** | Branch cost = source prefix tokens + request-*k* input + learner-tokenizer length of the intervention turn + continuation tokens. |
| **Saving** | Mean original cost minus mean edited cost across repetitions. |
| **Acceptance rule** | Named criteria for turning a verification into a preference pair; currently only `strict_all_success_v1`. |
| **Evidence label** | Strength of an accepted record, e.g. `one_observed_successful_preference` with one continuation per branch. |
| **Audit** | Re-verification of a random share of accepted edits with fresh seeds; reported, never changes datasets. |

## Training and models

| Term | Meaning |
|---|---|
| **Preference pair** | `{prompt, chosen, rejected, tools}`: the history before turn *k*, the edited turn, the original turn. Its evidence lives in the companion provenance record. |
| **Export / dataset** | An immutable directory of pairs (`preferences.jsonl`, `provenance.jsonl`, `manifest.json`). `dataset/current/` holds one cycle's new pairs; `dataset/` the selected training set. |
| **History buffer** | Earlier cycles' pairs sampled into the training set (`current_and_history`). Old pairs keep their original provenance; they are not re-verified. |
| **Fixture** | A labeled stand-in used for engineering checks: scripted policy/editor, the fixture trainer, hand-written preference data. Pairs from scripted runs are labeled `fixture`, never `verified`. |
| **No-update cycle** | A cycle with no trainable pairs: the learner carries over unchanged. |
| **Base model / revision** | The trainable source checkpoint (HF repo) at an exact commit, from the model profile. |
| **Serving artifact** | What a backend actually serves, e.g. a Q8_0 GGUF of the base model. Not a trainable source and not bit-identical to it. |
| **Adapter** | LoRA weights (PEFT) on top of the base model. |
| **Checkpoint** | An immutable learner identity: base (`base:<profile>@<rev12>`) or base + adapter (`cNNN-<hash12>`, published under `runs/<id>/checkpoints/`). |
| **Incoming checkpoint** | The learner frozen at the start of a cycle. Its adapter is continued, and it is the DPO reference. |
| **Reference** | The frozen policy DPO compares against; here always the incoming checkpoint *including* its adapter (precomputed log-probs). |
| **Lineage** | Parent and reference ids recorded in each `checkpoint.json` and `cycle.json`. |

## Machines

| Term | Meaning |
|---|---|
| **Coordinator** | The process running `loop run`: it owns the run directory, Docker work and serving swaps. |
| **Pod lifecycle** | For `kind: runpod` hosts: each command starts an existing pod (`pod_id`) or creates one from a spec (`pod:`), prepares it, keeps a heartbeat, and stops (existing) or terminates (created) it at the end. |
| **Created pod** | A pod `loop` created from a `runpod.create` spec for one command, named `lfe-<spec>-<stamp>` and recorded in the ledger `artifacts/runpod/created.jsonl`; the only kind of pod `loop` terminates. |
| **Watchdog** | A process on a pod that stops the pod (a created pod: terminates it) when the coordinator's heartbeat is older than `idle_stop_minutes`, so a sleeping or crashed laptop cannot leave it running. |
| **Dropped pair** | An exported preference pair the trainer did not train on (e.g. longer than `dpo.max_length`); counted with its reason in `cycle.json`, `loop status` and the report. |
| **Managed / external / scripted inference** | The run starts and stops its own model server / uses an existing endpoint for one declared checkpoint / uses a fixture policy. |
| **Model profile / machine profile / experiment** | Model identity / deployment / scientific choices (see [configuration.md](configuration.md)). |
