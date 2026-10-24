# first-e4b: pre-registered analysis

This file fixes the analysis of the first all-E4B experiment (`first-e4b.yaml` and
`first-e4b-frozen-baseline.yaml`) before any of its runs start. Results are read only as
described here; anything else is reported as exploratory.

## Question

Does training Gemma-4-E4B (thinking on) on verified, cheaper edits of its own successful episodes
make it solve tasks with fewer tokens without losing success, on new instances of the trained
families and on held-out families?

## Runs

| Run id | Command | What it measures |
|---|---|---|
| `first-e4b-frozen-baseline-1` | `loop run experiments/first-e4b-frozen-baseline.yaml` | The base checkpoint on `dev`. |
| `first-e4b-base-final-1` | `loop evaluate --final --checkpoint base` with `first-e4b.yaml`, panels `final-same-family final-held-out-family final-held-out-cluster` | The base checkpoint on the final panels. |
| `first-e4b-1` | `loop run experiments/first-e4b.yaml` | The learning loop: 3 cycles, `dev` before cycle 0 and after each cycle. |
| `first-e4b-1-final` | `loop evaluate --final --checkpoint <last checkpoint of first-e4b-1>`, same experiment and panels | The trained checkpoint on the final panels. |

All four use `seeds.root` 20261015, so every episode pairs with one of the other checkpoint on
the same instance, attempt and seed. The final panels are evaluated once per checkpoint, after
the learning run has finished, and nothing about the method changes after they are seen.

## Primary comparison

`loop compare runs/first-e4b-base-final-1 runs/first-e4b-1-final --panel final-same-family`,
effects read as trained minus base:

1. **Token change**: the instance-level mean difference in episode tokens on pairs where both
   checkpoints succeed, with its 95% bootstrap interval over instances and the number of pairs
   it covers.
2. **Success change**: the instance-level mean difference in success, with its 95% bootstrap
   interval over instances, and the counts of pairs that both succeed, gained, lost and both fail.

The result is **positive** when the token-change interval lies entirely below 0 and the
success-change interval's lower bound is above -0.10 (success falls by less than 10 percentage
points). It is **negative** when the token-change interval lies entirely above 0 or the
success-change interval lies entirely below 0. Anything else is **inconclusive**.

## Secondary comparisons

- The same two metrics and rule on `final-held-out-family` (near transfer) and
  `final-held-out-cluster` (far transfer), each read separately.
- `loop compare runs/first-e4b-frozen-baseline-1 runs/first-e4b-1 --panel dev`: the dev metrics of
  each cycle's checkpoint against the base, as a trajectory over cycles. Dev results guide nothing
  in this run; they are reported only.
- Loop yield per cycle, from `first-e4b-1`'s report: sources, proposals, abstentions, validation
  failures by reason, verifications, accepted edits, and pairs trained and dropped with reasons.

Per-family and per-difficulty breakdowns are exploratory.

## Limits and exclusions

- There is one loop seed, so the result is one draw of the loop. It supports a claim about this
  run, not about the method across seeds, and no interval across seeds is computed.
- Episodes that fail for infrastructure reasons are retried within `runtime.infra_retries` and
  stay visible in the counts; no other episodes are excluded.
- If no cycle trains a checkpoint, `first-e4b-1-final` is not run and the result is reported as
  "no update", with the loop yield explaining why.
