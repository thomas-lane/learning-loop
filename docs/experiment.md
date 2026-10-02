# Method, protocol and key contracts

This describes what the code currently does. Parameter values live in `experiments/*.yaml`,
model identities in `configs/models/*.yaml`; they are not repeated here.

## Question and unit of analysis

Does preference training on **verified efficiency edits** reduce the tokens a learner spends per
episode while preserving complete task success, on the training instances, on new instances of
the same families, and on entirely held-out families? Learning is measured across complete
cycles (collect -> edit/verify -> train -> evaluate), not optimizer steps.

The primary episode metric is **input + generated tokens summed over all learner requests**,
counting the repeated prompt input of every request. It is a token count under one tokenizer, not
a hardware-independent compute estimate; runs with different tokenizers are not compared as if
tokens were the same unit. Success and partial reward are always reported next to cost.

The statistical unit is the **task instance** (and, for method comparisons, the independent
learning-loop seed), never individual correlated attempts. Intervals are only printed with at
least five instances (and, across loop seeds, at least three seeds; `loop compare A1 A2 A3 --vs
B1 B2 B3` computes per-seed paired effects first); smoke/fixture runs never support statistical
claims. Token deltas between runs whose model identities or usage sources differ are withheld as
incomparable units.

## Cycle protocol

For cycle *c* with learner checkpoint *L_c* (frozen for the whole cycle):

1. **Evaluate** *L_c* unassisted on the declared development panels. Evaluation seeds depend only
   on `seeds.root`, instance and attempt, so every checkpoint (and every loop seed, and baseline
   runs) is evaluated on the same declared schedule and can be compared pairwise.
2. **Collect** attempts of *L_c* on the training panel (optionally restricted by a family-exposure
   schedule). Collection seeds are separate streams per cycle.
3. **Edit.** For each completely successful collection episode, the editor makes at most one
   proposal (it may abstain). The default editor is the fixed initial instruction-tuned policy,
   identical in every cycle; `current_learner` and `external` editors are declared conditions.
4. **Verify** each valid proposal (below). Accepted comparisons become preference pairs.
5. **Dataset.** Current accepted pairs plus a bounded, seeded, task-balanced sample of earlier
   cycles' pairs (`current_only` or `current_and_history`). Earlier pairs keep their original
   provenance; they are not re-verified under the new learner. History alone never triggers
   training. The export is immutable and hashed.
6. **Train** a LoRA with DPO: the incoming adapter's weights are continued, the reference is *L_c*
   including its adapter, and the optimizer/scheduler are reset each cycle (one cycle = one
   training stage). Resuming an interrupted stage restores its optimizer/scheduler/RNG state; on CUDA
   the result matches an uninterrupted stage only up to run-to-run kernel noise (efficient
   attention accumulates gradients in a nondeterministic order). The
   checkpoint is published atomically to a unique read-only directory and reload-checked; it is
   served (which also verifies loading) before any use as *L_{c+1}*.
7. With no acceptable pairs (or no trainable examples), the cycle is a recorded **no-update**
   cycle and *L_{c+1} = L_c*. Fixture data is never substituted.

Final-test panels are evaluated only with `loop evaluate --final` into separate run directories
that the loop never reads, after method and selection decisions are frozen. External benchmarks
(`loop external-eval`) use a version-pinned Harbor dataset, the repository's own agent protocol
(declared as such), and are likewise never fed back.

### Controls

- **Frozen baseline** (`condition: frozen_baseline`): the initial checkpoint on the same panels
  and evaluation seeds.
- **Fixed-dataset DPO** (`condition: fixed_dataset`): the same optimizer-step budget per cycle on
  one frozen preference export; no collection, editing or verification ever happens, so its extra
  collection/editing costs are zero by construction and reported as such rather than assumed equal.
- **Editor comparisons** (`loop edit-replay`): the same saved source trajectories and learner,
  edited and verified by a different editor configuration.
- **Difficulty and family studies**: an easy-only training panel with medium/hard development
  panels, and family-exposure schedules with per-family development panels (retention). Gains from
  mixed-difficulty training are not interpreted as easy-to-hard transfer.

## Environments and replay

Tasks are Harbor task directories (see `evaluation/README.md`). The learner acts only through
`bash`, `read_file` and `write_file` on the task container; hidden tests and reference solutions
exist only in a separate verifier container that receives declared artifacts after the episode.

A decision state is restored by **clean reset + deterministic replay**: a fresh container of the
same pinned image and generated inputs, re-execution of the earlier turns' executed arguments
without calling any model, and checks that every replayed observation (after task-declared
normalizers only) and the declared state fingerprint match the source. Mismatches fail closed.
The exact historical conversation is restored separately from the event log. Restoration is a
declared capability (`none`, `deterministic_replay`, `approximate_replay`); approximate replay is
never mixed with exact replay, and Harbor conversation loading is not used as filesystem
restoration. Fingerprints cover declared paths' content, types, permissions, owners, symlink
targets and the working directory, not processes, network or clock. Base images are pinned by
digest and each episode records an image identity (build context + pinned base); a branch whose
image identity differs from its source fails closed. Harbor 0.23.0 rejects `no-network` on the
Docker Desktop VM used here, so task containers have public network access; replayable tasks do
not use it, and this is recorded as a caveat on their state contract rather than claimed as
isolation.

Token and turn limits are experimental budgets; the wall-clock timeout is a safety control; they
have distinct stop categories. If the token budget is set but the endpoint omits usage, the episode
stops as `budget:usage_unavailable` instead of running unbudgeted. Per-command tool timeouts are
clamped to the declared budget. A tool-call block the server cannot parse is recorded as a
malformed turn (`model_error:unparsed_tool_call` when no valid call remains), never as a finish. Infrastructure failures get a bounded, predefined retry in a fresh
environment; prior attempts are preserved and failures remain in denominators.

## Editing

The editor is a repository component (`src/learning_loop/editing/editor.py`, prompt
`prompts/editor/v1.md`) with a pinned prompt hash, decoding settings and a proposal budget. It
sees only: the instruction, the learner's system prompt and tool schemas, the learner's turns as
the learner saw them (including later observations), and optional scalar outcome metrics. It
returns structured JSON naming the source trajectory, turn, tool-call id, replacement call and a
justification (kept out of every student prompt). It cannot approve its own edit, change grading
or budgets, or touch the source environment.

Validation happens before execution. The initial, strict condition accepts only:

- a model turn with exactly one well-formed, unrepaired tool call and empty assistant text and
  reasoning (`assistant_text_policy: reject_nonempty`; reasoning is never invented or rewritten);
- a replacement that is a different, schema-valid call to an existing tool;
- no references to grader locations (`/tests`, `/solution`, `/logs/verifier`, reward files);
- no changed budget-like arguments (`timeout_sec` and similar must equal the original call's);
- no **ungrounded constants**: IPs, quoted literals, paths, numbers of three or more digits, and
  any number the replacement writes or prints (e.g. `echo 17 > /app/answer.txt`, `write_file`
  content), that appear after the decision (later observations, later assistant text or later
  tool-call arguments) but not, as whole tokens, in the instruction/system prompt/prefix, are
  rejected. For numbers and IPs the replacement writes or prints, earlier observations count only
  when the value was shown as a whole result line (e.g. a `wc -l` output), because such values
  occur coincidentally inside log lines; this closed a case where a live editor hardcoded a count
  after a `cat` of the logs. This is a heuristic filter, not a proof of absence of hindsight leakage: it can also
  reject a small written number that was not hindsight, and it does not inspect small numbers
  outside write/print contexts. Rejected proposals are retained for audit.

Consequence: models that emit reasoning text on every turn yield no eligible turns under this
condition (serve them with reasoning disabled, as the committed profiles do). One proposal per
source is the only supported setting; several candidates per source would need fresh-seed
confirmation and selection, which is not implemented, so configs asking for it are refused.

## Verification and acceptance

From the same restored pre-decision state:

```text
original fixed action -> execute -> fresh continuation of L_c
edited fixed action   -> execute -> fresh continuation of L_c
```

Both branches use the same frozen learner, matched continuation seeds (the branch label is not a
seed input), and equal remaining budgets (the prefix's usage and turns count against both).
Downstream source observations are never reused, and the source's already-known suffix is never
the comparison baseline.

**Counterfactual episode token cost** of a branch = shared prefix (input+output of the source
requests before turn *k*) + the source request-*k* input (shared, counted once) + the learner
tokenizer/template length of the rendered fixed assistant turn + provider usage of the
continuation. The original turn's generated completion is represented only by its rendered
length, never counted twice, and the inserted edited turn is not free. Tokens the verifier
actually spent (continuations) are a separate operational total.

`strict_all_success_v1` accepts only if: replay is valid on every branch; the fixed intervention
executed without a tool error or timeout on every branch (its outcome is recorded per branch; a
missing record rejects); both branches reach complete success in every scheduled repetition; no
infrastructure, budget, safety or replay stops; every cost is measured; and the mean saving is
positive, at least `min_token_saving`, and at least `min_relative_saving`. Ties and missing values reject. With one continuation per branch
the evidence label is "one observed successful preference", not proof of reliable improvement.
Every rejection reason is recorded.

`local` verification is implemented as a component (`editing.verify.LocalVerifier`) for tasks declaring a
supported equivalence contract (`same_state_after_action`: identical declared-state fingerprint
after the edited and the original action from the same restored state, with a shorter rendered
turn), and local-only pairs are labeled and never mixed with continuation-verified pairs. No
environment backend in this build provides the session factory it needs, so experiments asking
for `mode: local` fail validation rather than running. Optional audits re-verify a
seeded random fraction of accepted edits with fresh seeds; audit results are reported and never
change frozen datasets.

## Preference data and training

Each example is exactly `{prompt, chosen, rejected, tools}`: the historical messages before turn
*k*, the edited assistant turn, the original assistant turn (same tool-call id), and the learner's
tool schemas. Evidence (continuations, editor justification, costs) lives in a companion
provenance record keyed by `pair_id`. Exports reject non-train instances, held-out families,
mixed verification modes and mixed fixture/verified kinds.

Rendering uses the learner's pinned tokenizer and chat template (with the profile's template
arguments). The prompt render must be an exact token prefix of both full renders; only completion
tokens (through the end-of-turn token) carry loss; chosen and rejected must differ in the
completion; oversize examples are dropped with a recorded reason, never truncated.

Reference log-probabilities are precomputed with the incoming learner **including its adapter**
(library defaults, which use the adapter-disabled base model, are not relied on), cached under a
key covering the reference checkpoint and adapter hash, example, token ids, tokenizer, template and
dtype. Every log-probability (reference, DPO loss, trained and reloaded) runs the full sequence
through the model but the output head only over the completion window, in float32; this equals
the full-logits computation and keeps long prompts within memory. A first-step check requires
near-zero policy/reference log-ratios. Checkpoint records hold
lineage, base revision, adapter config, tokenizer/template hashes, dataset hash, training config,
optimizer steps, seeds, metrics and the reload check.

## Measurements and reports

Per episode: complete success, partial reward, stop reason/category, provider usage (input,
output; cached and reasoning tokens as subsets only), requests, tool calls, endpoint round-trip
time (not labeled generation time), tool time, total time, container-cgroup tool CPU time where
measurable; unavailable values are null. Per stage: collection, editor, replay/verification and
training usage, durations and optimizer steps; money only with an explicit configured rate and its
provenance. Also proposal yield, rejection reasons, dataset composition and branch cost
differences.

Reports group by checkpoint/cycle, panel, family, difficulty and skill, and include fixed-panel
paired comparisons: full success transition counts (both succeed / gained / lost / both fail) and
token deltas restricted to pairs where both succeed, with their coverage, so lost successes cannot
hide behind cheaper surviving episodes.

## Reproducibility and provenance

Each run saves the resolved secret-free configuration (including `--set` overrides), code
identity (git revision and dirty fingerprint, or source-tree hashes outside git), package versions,
lockfile hash, OS/hardware, task instance hashes, the seed schedule, checkpoint lineage and stage
manifests. Requested seeds and declared endpoint seed support are recorded; bitwise
reproducibility across hardware, batching or serving versions is not claimed, nor across repeated
CUDA training runs (identical inputs gave adapters differing by up to about 1e-4 after two steps on
an A100). The local smoke test
uses serial inference; research runs record serving backend, template, quantization, hardware and
concurrency so that timing comparisons are interpretable.
