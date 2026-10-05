# Method

Settings named below (in `code` font) are set in `experiments/*.yaml`; model identities are in
`configs/models/*.yaml`. Terms are defined in [glossary.md](glossary.md).

## Research question

A *learner* model solves agentic tasks with tools. After it succeeds, an *editor* looks back at
the episode and proposes one cheaper action at one turn. The edit is kept only if re-running the
episode with it still succeeds and costs fewer tokens. Kept edits become DPO preference pairs
(edited turn preferred over the original), and the learner is trained on them.

The question: **does training on these verified edits make the learner use fewer tokens per
episode without losing task success?** This is measured on the training instances, on new
instances of the same task families, and on families never trained on, after each *cycle*
(collect, edit, verify, train, evaluate).

## How results are measured

**Cost metric.** The cost of an episode is input tokens + generated tokens, summed over every
request the learner makes (the resent conversation counts every time). It is a count under the
learner's tokenizer, not a hardware-independent compute measure. Success and partial reward are
always reported next to it.

**Paired comparisons.** Every checkpoint is evaluated on the same tasks with the same seeds
(seeds depend only on `seeds.root`, the instance and the attempt number). Two checkpoints or two
runs can therefore be compared episode by episode on identical inputs. Reports show:

- how many pairs both succeed, gained success, lost success, or both fail;
- the token difference, computed only on pairs where both succeed, along with how many pairs
  that covers. A model that got cheaper by failing more cannot look better this way.

Token differences are not reported between runs that count tokens differently (different model
identity, base revision or usage source); they are marked "incomparable token units".

**What counts as an independent sample.** Attempts at the same task instance are correlated, so
they are not treated as independent data points. Reports average the attempts of each instance
first, then compute the mean and a 95% bootstrap interval *over task instances*. With fewer than
5 instances no interval is shown.

**Comparing conditions across loop seeds.** A *condition* is one way of running the experiment,
e.g. the full learning loop vs. a frozen baseline, or one editor vs. another (see
[Conditions and controls](#conditions-and-controls)). One run of the loop is a single random
draw: which episodes are collected, which edits are proposed and how training goes all depend on
`seeds.loop_seed`. A difference between one run of A and one run of B could be luck, so to claim
that A beats B, run each with several loop seeds (independent repeats of the whole experiment)
and pass one run directory per seed:

```bash
loop compare runs/A-seed1 runs/A-seed2 runs/A-seed3 --vs runs/B-seed1 runs/B-seed2 runs/B-seed3
```

Runs are matched by loop seed (by position if seeds are missing or differ). A single B run, such
as a frozen baseline, is compared against every A run. The A-vs-B effect is computed within each
matched pair, as above, then averaged over seeds. A 95% interval across seeds needs at least 3
seeds. Every effect is reported as B minus A (a "lost" pair lost success under B), so with a
frozen baseline after `--vs` the effects read as baseline minus learning.

Smoke runs and runs with *fixtures* (scripted stand-ins for the learner, editor or trainer, used
to test the plumbing without a model) never support statistical claims.

## One cycle

Cycle *c* starts with learner checkpoint *L_c*, which stays fixed for the whole cycle.

1. **Evaluate.** Run *L_c* without help on the development panels (`evaluation.dev_panels`).
2. **Collect.** Run *L_c* on the training panel (`tasks.collection_panel`), optionally only on
   the families an exposure schedule allows in this cycle. Each cycle uses fresh seeds.
3. **Edit.** For each fully successful collected episode, the editor proposes at most one edit
   or abstains (see [Editing](#editing)).
4. **Verify** each valid proposal by re-running both versions (see
   [Verification and acceptance](#verification-and-acceptance)). Accepted edits become
   preference pairs.
5. **Build the dataset.** By default (`training.data.selection: current_and_history`) the
   training set is this cycle's accepted pairs plus pairs from earlier cycles, so the learner
   keeps seeing what it learned before. Earlier pairs are drawn from a pool capped at
   `training.data.buffer_capacity`, sampled evenly across task instances, and make up about
   `training.data.history_fraction` of the set. They are not re-verified against the new learner.
   `current_only` uses this cycle's pairs alone. Either way, a cycle with no new pairs does not
   train. The training set is written once to the cycle's `dataset/` directory as read-only
   files with a content hash, so every checkpoint traces back to the exact data it was trained
   on.
6. **Train** a LoRA adapter with DPO for exactly `training.optimizer_steps` steps. Training
   continues from *L_c*'s adapter, the DPO reference model is *L_c* including its adapter, and
   the optimizer and scheduler restart every cycle. The new checkpoint is published read-only
   and reload-checked before it becomes *L_{c+1}*.

If no pair is accepted (or none fits training's length limit), nothing is trained: the cycle is
recorded as a *no-update* cycle and *L_{c+1} = L_c*.

A run with `cycles: N` trains N times and evaluates N+1 checkpoints; the last cycle only
evaluates. All of this uses development panels.

**Final test set.** Final-test panels (`evaluation.final_panels`) are the held-out test set. The
loop never runs them. Once you have stopped changing the method and its settings, evaluate the
chosen checkpoint with `loop evaluate --final`, so those choices cannot have been tuned to the
test set. The result is a separate run that the loop never reads.

**External benchmarks.** `loop external-eval` writes a Harbor job for a version-pinned benchmark
(e.g. Terminal-Bench) and, with `--execute`, runs it on a checkpoint. It uses this repository's
agent (the same tools and system prompt the learner was trained with), because what training
teaches is specific to that interface. Scores are therefore not comparable with the benchmark's
published leaderboard, which uses each benchmark's own agent. Results go to `runs/external/`,
which the loop never reads.

## Conditions and controls

| Setting | What it tests |
|---|---|
| `condition: learning` | The full loop described above. |
| `condition: frozen_baseline` | The initial checkpoint on the same panels and seeds, with no training (`cycles: 0`). |
| `condition: fixed_dataset` | Training alone: the same optimizer-step budget per cycle on one frozen preference export, with no collection, editing or verification. Those stages' costs are reported as zero, not assumed equal. |
| `editor.mode` | `initial_policy` (default: the initial checkpoint, identical every cycle), `current_learner` (changes every cycle) or `external` (a separate fixed model). |
| `loop edit-replay` | Editor comparisons: re-edit and re-verify the same saved trajectories with a different editor configuration and the same learner. |

Difficulty and family studies reuse this machinery: train on easy instances and evaluate on
medium/hard panels, or follow a family-exposure schedule and check per-family panels for
retention. Gains from mixed-difficulty training are not read as easy-to-hard transfer.

## Episodes

The learner acts only through `bash`, `read_file` and `write_file` in a task container; hidden
tests run in a separate verifier container ([evaluation/README.md](../evaluation/README.md)).
Each episode has two kinds of limits, recorded as different stop categories:

- **Budgets**, part of the experiment: `episode.max_turns` and `episode.max_episode_tokens`. An
  endpoint that reports no usage under a token budget stops the episode
  (`budget:usage_unavailable`). Per-command timeouts are capped at `episode.tool_timeout_sec`.
- **Safety limit:** the wall-clock timeout `episode.agent_timeout_sec`.

An episode ends normally when the learner replies without a tool call; then it is graded. A
reply whose only tool calls the server could not parse also has no tool call, so it stops as an
error instead (`model_error:unparsed_tool_call`).

An *infrastructure failure* is an error outside the learner's control: the container or a Harbor
command fails, or the model endpoint is unreachable, times out or returns a server error. The
episode is retried in a fresh environment up to `runtime.infra_retries` times; earlier attempts
are kept for inspection. An episode that still fails counts as unsuccessful in success rates
rather than being left out.

To verify an edit, the loop recreates the state just before the edited turn: a fresh container
re-executes the earlier tool calls (no model calls), and every output and a fingerprint of the
task's files must match the original episode, or the branch stops before the edit is tried. The
replay contract, stop reasons and known gaps are in [evaluation/README.md](../evaluation/README.md).

## Editing

The editor (`src/learning_loop/editing/editor.py`, prompt `prompts/editor/v2.md`) sees only what
the learner saw: the instruction, system prompt, tool schemas, the learner's turns and their tool
outputs (including later ones), which turns are editable, and optionally scalar outcomes such as
success and token totals. A trajectory with no editable turn is not sent to the editor; it is
recorded as abstained with the reason `skipped:no_editable_turns`.

The editor answers by calling exactly one tool. For each learner tool it is offered a
`replace_with_<tool>` tool, which takes that tool's own parameters (the replacement call) plus
`edit_turn`, limited to the editable turns, and `edit_justification`, kept for audit and never in
training data. Calling `abstain` declines. Because the answer is a tool call rather than written
JSON, the replacement is written in the model's native tool-call format and parsed by its
server's tool-call parser, as the learner's own calls are, and code or file contents in it need
no hand-written escaping. If the reply is exactly one answer call written without the model
format's call markers, it is accepted as that call and recorded (`answer_recovery:
bare_tool_call` in the proposal's raw response). Learner turns are never recovered this way: whether the learner follows its tool-call
format is part of what is measured. The editor cannot approve its own edit or touch grading,
budgets or the environment.

A *turn* is one request to the learner and its reply. The model generates until it stops to
wait for tool results (or ends without calling a tool). The tool calls in that reply run, their
outputs are appended to the conversation, and the next request starts the next turn. A reply can
contain several tool calls, or none.

Proposals are checked by code before anything runs. An edit is rejected unless:

- the original turn has exactly one tool call, which parsed cleanly and ran, and no assistant text
  or reasoning (`editor.assistant_text_policy: reject_nonempty`, see below);
- the replacement is a different, schema-valid call to an existing tool;
- no argument mentions a grader location: `/tests`, `/solution`, `/oracle`, `/logs/verifier`,
  `/logs/agent` or `reward.txt`/`reward.json` (a pattern match on every string argument). These
  files are not in the learner's container, but training on such edits would teach the learner
  to look for them;
- every budget-like argument (`timeout_sec`, `timeout`, `max_output_chars`, `max_tokens`,
  `max_turns`) has exactly the original call's value, and none is added. In practice this is the
  `bash` tool's optional `timeout_sec`, which the learner may set below
  `episode.tool_timeout_sec`. The rule keeps savings from coming from changed limits, so an edit
  that only raises a too-short `timeout_sec` is rejected, even below the cap;
- it contains no **hindsight constants** (below).

**Hindsight constants.** The editor sees how the episode ended, so it could cheat by writing the
final answer straight into an early turn, for example `echo 17 > /app/answer.txt` where `17`
only appeared in a later output. The check extracts constants from the replacement: IP
addresses, numbers of 3+ digits, quoted strings, file paths, and *any* number the replacement
writes or prints. A constant is rejected if it appears after the edited turn but not before it.
For written or printed numbers and IPs, an earlier tool output counts only if the value appeared
there as a whole line (like the output of `wc -l`), because small numbers turn up by chance
inside log lines. It is a heuristic: it can reject an innocent small number and misses hindsight
that is paraphrased or computed at runtime. Rejected proposals are kept with their reasons.

**Why turns with reasoning are not editable.** The chosen and rejected sides of a training
example are whole assistant turns, reasoning included. Keeping the original reasoning in front
of a new call would train the learner to make a call its own reasoning did not lead to; writing
new reasoning would train it on the editor's thoughts. So turns with reasoning are skipped.
Models in thinking mode reason on every turn, so the committed model profiles turn thinking off
(`chat_template_kwargs: {enable_thinking: false}` in `configs/models/*.yaml`). The model then
generates no reasoning at all (it is not generated and hidden); it answers with tool calls
directly.

`editor.proposals_per_source` must be 1: picking the best of several verified edits from one
episode would favor edits whose continuations happened to go well.

## Verification and acceptance

Both versions of the edited turn are run from the same restored state:

```text
original action -> execute -> fresh continuation by L_c
edited action   -> execute -> fresh continuation by L_c
```

Both branches use the same learner, the same remaining budget and the same continuation seed.
The seed is derived from `seeds.root`, the proposal id and the repetition number, never from
which branch it is, so the two branches always get the same one.

**Why re-run the original with a new seed instead of reusing its recorded ending or its seed?**
Only episodes that succeeded are edited, so the recorded ending is a draw already known to have
gone well. Reusing it, or replaying the original with its own seed (which would reproduce that
draw on a deterministic server), would favor the original. Running both branches with one new
seed compares them on equal terms.

**Branch cost**, in tokens:

```text
cost = prefix          input + output of the original requests before the edited turn
     + request input   input of the edited turn's request (same for both branches)
     + action length   the fixed turn rendered with the learner's tokenizer and chat template
     + continuation    usage reported for the rest of the branch
```

Both fixed turns count by rendered length, so the edited turn is not free and the original is not
counted twice. Tokens spent on verification itself are tracked separately.

**Acceptance rule `strict_all_success_v1`.** An edit is accepted only if all of these hold:

- replay is valid on every branch;
- the fixed action ran without a tool error or timeout on every branch;
- both branches fully succeed in every repetition (`verification.continuations_per_branch`);
- no branch stopped for infrastructure, budget, safety, replay or model-error reasons;
- every cost was measured;
- the mean saving is at least `verification.min_token_saving` tokens and at least
  `verification.min_relative_saving` of the original branch's cost.

Ties and missing values reject, and every rejection reason is recorded. With one continuation per
branch, an accepted edit is one observed success, not proof the edit reliably helps.

**Audits.** `verification.audit.fraction` re-verifies a seeded random sample of accepted edits
with fresh seeds. Audit results are reported but never change a frozen dataset.

## Preference data and training

Each training example is `{prompt, chosen, rejected, tools}`: the conversation before the edited
turn, the edited turn, the original turn, and the learner's tool schemas. The tool schemas are
needed because the chat template writes them into the prompt, and training must see the exact
prompt the learner sees. They are a separate field rather than part of `prompt` because chat
templates take them as a separate input (`apply_chat_template(messages, tools=...)`), just as
the model's API takes tools separately from messages. Evidence
(continuations, editor justification, costs) lives in a separate provenance record keyed by
`pair_id`. An export is refused if it contains instances outside the training split, held-out
families, mixed verification modes, or fixture pairs mixed with verified ones.

Rendering uses the learner's pinned tokenizer and chat template. The rendered prompt must be an
exact token prefix of both full renders. Only the completion tokens (through end-of-turn) carry
loss, and chosen and rejected must differ there. Examples longer than `training.dpo.max_length`
are dropped with a recorded reason, never truncated.

**The DPO reference.** For each example, let log π(turn) be the summed log-probability of the
turn's tokens given the prompt. DPO minimizes

```text
loss = -log sigmoid( beta * [ (log π_train(chosen)   - log π_ref(chosen))
                            - (log π_train(rejected) - log π_ref(rejected)) ] )
```

which raises the chosen turn's probability relative to the rejected one, measured against a
frozen *reference* model π_ref, so training cannot drift far from it (`training.dpo.beta` sets
how far). Here the reference is the learner that entered the cycle, *L_c* including its LoRA
adapter. TRL's default for LoRA training would instead use the base model with the adapter
switched off, which equals *L_c* only in cycle 0, so the code does not rely on it.

Backpropagation only goes through the π_train terms, which are recomputed on every step. The two
π_ref terms are fixed numbers per example (one for chosen, one for rejected), so they are
computed once before training and cached rather than recomputed every step. Training starts from
the same weights as the reference, so before the first update both must give the same numbers
(log-ratio within 0.02); otherwise the wrong reference was loaded and training stops with an
error.

## What is recorded

**Per episode:** success, partial reward, stop reason, token usage (cached and reasoning tokens
are subsets of input/output, never added), requests, tool calls and timings. Values that cannot
be measured are null, never 0.

**Per stage:** token usage, durations and optimizer steps for collection, editing, verification
and training. Money is shown only when a rate and its source are configured.

**Editing:** proposal yield, rejection reasons, dataset composition and branch cost differences.

Reports group results by checkpoint/cycle, panel, family, difficulty and skill, and include the
paired comparisons described in [How results are measured](#how-results-are-measured).

## Reproducibility

Each run saves what is needed to reproduce it: resolved configuration, code and package
versions, hardware, task hashes, seeds and checkpoint lineage
([run-layout.md](run-layout.md) lists the files).

Results are not bitwise reproducible across hardware, batching or serving versions, nor across
repeated CUDA training runs (on an A100, identical inputs gave adapters differing by up to about
1e-4 after two steps, from nondeterministic GPU kernels). A resumed training stage matches an
uninterrupted one only up to that noise.

Runs record serving backend, template, quantization, hardware and request concurrency in their
serving record. The backend and concurrency are part of a run's conditions: compare runs only
when both match. The vLLM engine leaves what the learner sees and what is counted unchanged,
because `hf_server` still renders, parses and counts, but it generates with its own GPU kernels and,
for bf16 models (the Gemma profiles), computes the LoRA in bf16 where PEFT uses float32, so its
outputs differ slightly from the transformers engine. With request
concurrency above 1, an output can also depend on which requests share a batch, which adds noise
to seed-matched comparisons. Timings likewise only compare under the same setup. With `training.trainer: trl_dpo`, untrained checkpoints (cycle
0, frozen baseline, an initial-policy editor) are served through an all-zero LoRA adapter: outputs
match the base model exactly and every checkpoint pays the same adapter overhead
(`base_checkpoints_served_as` in the run's serving record).
