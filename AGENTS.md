# Developer guide

## Commands

```bash
uv sync --extra train                        # full environment (runner + training)
uv run pytest                                # fast suite: no Docker, no model downloads
uv run pytest -m docker tests/integration    # Harbor/Docker: oracle/nop, replay, timing, grading
uv run pytest -m train tests/train           # real LoRA DPO on Qwen3-0.6B (MPS/CUDA; ~1 min)
uv run loop smoke fixture                    # 2-cycle orchestration with fixtures (seconds)
uv run loop validate <exp.yaml> --machines <machine.yaml>
uv run loop docs-gen                         # regenerate docs/cli.md + docs/configuration.md
uv run loop docs --open                      # browse all docs at http://localhost:8000/
```

Run the Docker/train markers after touching tasks, the agent, replay, rendering or training.
Never leave long training or load tests running unattended on the laptop, never stop services
this repo did not start (e.g. a llama.cpp server on :9931), and never publish models/data or
push from automation. Paid compute is limited to Runpod pods declared in a machine profile
(`hosts/pods.py`): existing pods (`pod_id`) may only be started and stopped; pods from a
`runpod.create` spec may be created, within the spec's GPU types and `max_cost_per_hr`, and must
be terminated by the same command. Every command that starts or creates a pod must stop or
terminate it (or leave its idle watchdog running) however the command ends, and code only ever
terminates pods recorded in the created-pod ledger.

Credentials live in the git-ignored `.env` (template: `.env.example`), loaded automatically by
`loop`; configs refer to them by variable name only, and they are never synced to remote hosts.

## Where things live

Architecture, the module map and data flow are in `docs/architecture.md`; the contents of a run
directory are in `docs/run-layout.md`; terms are defined in `docs/glossary.md`. Quick index
(paths under `src/learning_loop/`, except `evaluation/`, which is at the repository root):

| Concern | Module |
|---|---|
| Records / interfaces / config schemas | `core/records.py`, `core/interfaces.py`, `core/config.py` |
| Seeds and stable IDs | `core/seeds.py` |
| Atomic files, run lock, stage manifests | `core/storage.py` |
| Provenance, `.env` loading | `core/provenance.py`, `core/envfile.py` |
| Episode loop, replay, stop reasons | `episodes/episode.py`, `episodes/events.py` |
| Policies (OpenAI-compatible, scripted fixture) | `episodes/policy.py` |
| Environments, fingerprints, backends | `episodes/envs/`, `episodes/fingerprint.py`, `episodes/backends.py` |
| Tasks, generators, splits | `tasks/instances.py`, `evaluation/generators/`, `evaluation/splits/` |
| Editor, verification, preferences | `editing/editor.py`, `editing/verify.py`, `editing/preferences.py` |
| Training, rendering, token counts | `training/`, `editing/token_count.py` |
| Serving, vLLM engine, lifecycle | `serving/` (`hf_server.py`, `vllm_engine.py`, `equivalence.py`, `tool_parse.py`, `managed.py`, `lifecycle.py`) |
| Orchestration, CLI | `orchestration/coordinator.py`, `orchestration/smoke.py`, `orchestration/external_eval.py`, `cli.py` |
| Remote hosts, pods, preflight | `hosts/remote.py`, `hosts/remote_jobs.py`, `hosts/pods.py`, `hosts/preflight.py` |
| Metrics, reports, live run dashboard (`loop dashboard`) | `reporting/metrics.py`, `reporting/report.py`, `reporting/dashboard.py` |
| Docs generation, docs viewer (`loop docs`) | `docs_tools/docgen.py`, `docs_tools/docserver.py` |

## Tests

```text
tests/unit/          default suite (`uv run pytest`): no Docker, no model downloads, seconds
  test_core_*        contracts: seeds, config validation, storage, usage arithmetic
  test_env_*         episode loop, policies, sessions, replay, timing, grading, hardening
  test_tasks_*       generators and split validation
  test_editor_*      editor view, answer tools, validation, grounding
  test_verify_*      branch specs, costs, acceptance, local end-to-end
  test_prefs_*       preference construction, exports, buffer
  test_train_*       rendering/masking, fixture trainer, profiles, precision, server parsing, vLLM engine
                     against a fake vLLM process (no weights)
  test_report_*      metrics, reports, comparisons, the live dashboard (numbers match reports, escaping, path safety, read-only)
  test_loop_*        orchestration on fixtures: lineage, controls, resume, retries, remote training
  test_cli_examples  every committed experiment/machine example validates through the CLI
  test_loop_remote_* remote training and serving against a fake SSH host
  test_loop_pods     Runpod lifecycle: existing and created pods (fake REST API over HTTP), price limit,
                     cleanup, watchdog script, key handling
  test_core_envfile  .env loading precedence, template has no values, secrets never synced
  test_docs_generated generated docs are current; every config field and CLI argument is described
  test_docs_server   doc viewer rendering, path safety, and that every doc is in the index and navigation
tests/integration/   `-m docker`: real Harbor containers (oracle/nop, replay, timing, forged grading)
tests/train/         `-m train`: real LoRA DPO on Qwen3-0.6B + serving the adapter
tests/fixtures/      scripted policies (env/), scripted edits (loop/, verify/), fixture preferences (train/)
```

Name new test files with the area prefix above. A behavior change comes with a test that would
have failed before it.

## Integrity invariants (keep tests for each)

- **Seeds/IDs** come from `core/seeds.py` (SHA-256 over explicit parts). Evaluation seeds depend only on
  `seeds.root`, instance and attempt, so checkpoints and loop seeds are compared on the same schedule.
  Original/edited continuations share a seed; the branch label is never a seed input.
- **Configs are strict**: unknown keys fail; credentials are env-var names; `run.json` is write-once.
- **No leakage into training**: preference exports accept only train-split instances outside
  held-out families; prompts contain only messages before the intervened turn; editor
  justifications and verification evidence live in companion records, never in `prompt/chosen/rejected`.
- **Hidden grading stays hidden**: tasks use Harbor's separate verifier with explicit artifact
  transfer; graders that execute agent code (fix-stats) run it as an unprivileged child that
  cannot read `/tests` or write `/logs/verifier`, and the parent judges its raw outputs. The host
  agent never writes through the container-writable `/logs/agent` mount except via a
  symlink-refusing copy. The editor sees only the instruction, the learner's system prompt and tool
  schemas, the learner's turns with their tool outputs, and scalar outcomes.
- **Replay fails closed**: fresh environment + re-executed prefix; observation and fingerprint
  mismatches stop the branch before the intervention. Normalizers are task-declared and recorded.
- **Acceptance** (`strict_all_success_v1`): valid replay, the fixed action ran without tool error or
  timeout, complete success on both branches for every repetition, every branch ended by the model
  (stop category `model`), all costs present, a positive mean saving that meets both thresholds.
- **Cost accounting**: counterfactual episode cost = shared prefix + intervention request input +
  learner-tokenizer length of the fixed turn + continuation usage; verification spend is separate.
  Missing measurements are null, never 0; cached/reasoning tokens are subsets, never added.
- **Training**: continue the incoming adapter; the DPO reference is the incoming learner including
  its adapter (precomputed log-probs, cache keyed by reference/example/tokenizer/template); exact
  optimizer-step budget; oversize examples dropped, never truncated; checkpoints published
  atomically to unique read-only dirs and reload-checked.
- **Orchestration**: completed work items are never redone; interrupted/infra-failed attempts are
  preserved (`*.interrupted-N`) and stay visible in counts; `infra_failed` is terminal once its
  retry budget is spent; one coordinator per run (flock) and one trainer per work dir (flock, exit
  4); a lost SSH session is reconciled from the remote PID/result before anything is relaunched;
  final/external evaluations are separate runs the loop never reads, and `loop resume` re-enters
  each run kind's own stages.
- **Fixtures are labeled**: scripted policies/editors, the fixture trainer and fixture preference
  data never silently stand in for real components.
- **Pods are never left running**: the pod lifecycle stops existing pods and terminates created
  pods on success, failure and Ctrl-C; a created pod above its price limit is terminated at once;
  a pod-side watchdog stops (or terminates) a pod whose coordinator heartbeat goes stale; only
  ledger pods named `lfe-*` are ever terminated; the account API key never reaches a pod (the
  watchdog uses the pod-scoped key, created pods get only the SSH public key).
- **Training-data drops are visible**: pairs the trainer drops (e.g. longer than
  `dpo.max_length`) are counted with reasons and pair ids in `cycle.json`, `loop status` and the
  report, never silently.

## Conventions

- Match surrounding style; typed pydantic records with `schema_version`; JSON/JSONL/CSV outputs.
- Experiment YAML holds scientific choices, machine YAML deployment, model YAML identities.
  Keep parameter values in configs, not prose.

## Keeping documentation current

Documentation is part of every change, not a follow-up. When you change behavior, structure,
commands or outputs, update the affected documents **in the same change, without being asked**,
and re-read the edited sections against the code before finishing. Each topic has one owner, so
update that file instead of repeating the information elsewhere:

| If you change... | Update |
|---|---|
| setup, smoke levels, component status (working / untested) | `README.md` |
| a `loop` command or argument | its `help`/`description` in `cli.py`, then `uv run loop docs-gen` (regenerates `docs/cli.md`); conventions and workflows are hand-written in the same file |
| a config field or schema | its `Field(description=...)` in `core/config.py`, then `uv run loop docs-gen` (regenerates `docs/configuration.md`); cross-field rules are hand-written there |
| operating procedures, error messages, recovery steps | `docs/operations.md` |
| the meaning of a term, or a new term | `docs/glossary.md` |
| a new document, or what a document covers | `docs/index.md` and the viewer's `NAV` in `src/learning_loop/docs_tools/docserver.py` |
| components, module responsibilities, process/machine roles, data flow, isolation boundaries, identities, extension points, repository folders | `docs/architecture.md` |
| files or directories a run writes, their names, formats or mutability, run kinds | `docs/run-layout.md` |
| the method: cycle protocol, controls, editing rules, acceptance, cost accounting, training semantics, metrics | `docs/experiment.md` |
| tasks, generators, splits, the agent, grading, the replay contract | `evaluation/README.md` |
| development commands, test layout, integrity invariants, this policy | `AGENTS.md` |

Rules:

- Describe what the code does now (see [Writing documentation](#writing-documentation)). Remove
  statements that became false. Track whether a component has been run or tested only in the
  `README.md` status table, marking anything implemented but not run as untested there.
- Examples must work: every documented `loop` command and config must run as shown. When you add
  or change an experiment/machine example, keep `tests/unit/test_cli_examples.py` covering it.
- Never edit between the `BEGIN GENERATED` / `END GENERATED` markers by hand. Every new CLI
  argument and config field needs a description (enforced by `tests/unit/test_docs_generated.py`,
  which also fails when the generated docs are stale).
- When you change an error message, a stop reason or a rejection reason, update the matching row
  in `docs/operations.md`.
- Keep parameter values in configs, not prose, and keep generated-run facts (provenance, metrics)
  in run outputs, not docs.
- Never present fixture or smoke outputs as research results in any document.

### Writing documentation

Write for a competent engineer or researcher who is new to this repository and to its methods.
They read to understand what the system does and why, then to run it. Each rule below follows
from serving that reader. Before writing, check every claim against the code.

1. **Document the system, not its making.** State what the code does and why. Leave out how the
   system or the document came to be: history, what was tried or tested, plans, and commentary
   on the document itself. If an option or feature cannot be used, change or remove the code (or
   ask the user) instead of documenting that it does not work.
2. **State what happens, not what does not.** Describe behavior positively. Mention that
   something does not happen only when a reasonable reader would otherwise expect it, and then
   say why.
3. **Give every rule its reason.** When the system imposes a constraint or behaves in a way that
   could look arbitrary, explain in a sentence what problem that solves. A reason is never the
   absence of some other feature.
4. **Answer the obvious "why not?"** When the reader will think of a simpler or more natural
   approach, say briefly why it is not used. Address only alternatives a reader would actually
   consider.
5. **Say how each guarantee is enforced.** When the text claims something is checked, prevented
   or guaranteed, describe the mechanism precisely enough that the reader can trust the claim.
   When a list is the rule, give the complete list.
6. **Define terms where they are used.** Any term with a meaning specific to this project or
   field gets a plain definition at first use or a link to `docs/glossary.md`. A definition must
   let the reader tell what is and is not an instance of the term. Definitions agree everywhere;
   the glossary is the reference.
7. **Name concrete things concretely.** Refer to anything the reader may need to find by what it
   is and where it lives, not by a label they have to decode.
8. **Explain borrowed techniques as far as the text depends on them.** When a section only makes
   sense if the reader knows how an external method works, give the minimal explanation needed.
9. **Shorten by removing, not by compressing.** Cut repetition, asides, and content another
   document owns (link to it instead). Never cut the explanation a reader needs: a terse sentence
   the reader cannot interpret is a defect, not brevity. Use one idea per sentence, and lists or
   tables for parallel items.
10. **Review the draft as the reader before finishing.** Fix only the places where that reader
    would be stuck, with the smallest change that resolves it, whether that adds, rewords or
    deletes text. The fixes go into the document; the review's own notes do not.
