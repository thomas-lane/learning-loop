# evaluation/: Harbor tasks, generators, splits and the learner agent

Harbor **0.23.0** tasks, the tool-calling learner agent, deterministic task generators and the
explicit split/panel files used by the learning loop.

```
evaluation/
├── run.sh                      # wrapper: PYTHONPATH + `harbor run -c configs/local-llama.yaml`
├── configs/local-llama.yaml    # Harbor job config: agent, model, URL, tasks, attempts
├── agents/
│   ├── tool_agent.py           # ToolAgent(BaseAgent): Harbor adapter around learning_loop.episodes.episode
│   ├── tools.py                # tool schemas + handlers (bash, read_file, write_file)
│   └── system_prompt.md        # the system prompt, verbatim ({workdir} is filled in)
├── tasks/                      # hand-written tasks: log-triage, fix-stats
├── generators/                 # versioned generators: log-triage, fix-stats, csv-revenue, count-errors
├── splits/                     # hand-authored panels: pilot.yaml, smoke.yaml, fixture.yaml
├── scripts/gen_log_triage_data.py   # regenerates tasks/log-triage/environment/logs
└── jobs/                       # Harbor outputs (gitignored)
```

## Quick start

```bash
# Reference/no-op checks (no model): oracle must score 1.0; nop is the baseline
uv run harbor run -p evaluation/tasks -a oracle -o evaluation/jobs
uv run harbor run -p evaluation/tasks -a nop -o evaluation/jobs      # log-triage 0.0, fix-stats 0.33

# The agent against an OpenAI-compatible server (llama.cpp needs --jinja for tool calls)
evaluation/run.sh                                          # all tasks, config defaults
evaluation/run.sh -p evaluation/tasks/log-triage -k 5      # one task, 5 attempts
evaluation/run.sh -m my-model --ak api_base=http://gpu-box:8000/v1

uv run harbor view evaluation/jobs                         # browse trajectories
```

Docker must be running. CLI flags override the config file. The learning loop does not use
`run.sh`: it runs single trials programmatically (`learning_loop.episodes.backends.HarborDockerBackend`).

## How a trial runs

1. Harbor builds `tasks/<t>/environment/Dockerfile` and starts the agent container.
2. `ToolAgent.run()` executes **on the host** and acts on the container only through Harbor's
   `exec` / `upload_file`. `localhost` in `api_base` is the host.
3. After the agent stops, Harbor copies only the task's declared `artifacts` (e.g.
   `/app/answer.txt`) out of the agent container, stops it, and builds a **separate verifier
   container** from `tests/Dockerfile` (build context `tests/`). The artifacts are uploaded to
   the same paths, `/logs/verifier/` is emptied, and `tests/test.sh` runs there. It must write
   `/logs/verifier/reward.txt` (one number) or `reward.json` (`{"reward": x, ...}`).

So the hidden tests and the reference solution never enter the agent's container, and a reward
file planted by the agent is discarded. (With the previous shared mode, a scripted trial that
wrote a wrong answer and planted `/logs/verifier/reward.json` scored 1.0 on log-triage: the agent
container has `/logs/verifier` mounted and `reward.json` takes precedence over the `reward.txt`
that `test.sh` writes. `tests/integration/test_harbor_tasks.py` now checks this scores 0.) The agent container still has network access (see gotchas), and `/logs/agent` is
mounted (writable) in it, which is why the loop's authoritative records live in a host-only
directory: the agent writes its per-turn views there and copies them into `agent/` only at the
end, with a helper that never follows a symlink the container may have planted.

A grader that *executes* an artifact must not run it where the reward is written. fix-stats
(hand-written and generated) runs `/app/stats.py` in a child `python3 -I -B` process as
`nobody`, after making `/tests` 0700; the child gets only the check inputs and prints raw
observations (numbers, exception types, the list after the call) as one marker line. The
grading process never imports the artifact, accepts only plain JSON numbers / literal booleans
for known checks (anything else, or more than one observation line, fails), kills every
leftover `nobody` process, removes planted reward files and only then writes `reward.json`.
An artifact that overwrites the reward from an `atexit` hook or a forked daemon, or returns
`__eq__`-always-true objects, scores 0 (`test_forged_fix_stats_artifact_does_not_score`).

## The agent (`agents/tool_agent.py`)

**What the model sees, every request, and nothing else:** the system prompt
(`system_prompt.md`), the task's `instruction.md` verbatim as the user message, the tool
schemas from `tools.py` as `tools=[...]` (the server's chat template renders them), and every
earlier assistant message and tool result. Assistant `reasoning_content` is sent back verbatim
when the server returned it; the chat template decides whether it becomes tokens (Gemma 4's keeps
it after the last user message; Qwen3's strips it). Tool outputs are truncated head+tail to
`max_output_chars`.

**Tools:** `bash` (fresh `bash -c` per call, cwd `/app`, 60 s default timeout; inside Docker the
command runs under coreutils `timeout` so a timed-out command cannot linger), `read_file`
(numbered lines, paged) and `write_file` (uploads the content; no shell escaping). Unknown tools,
missing arguments and bad argument values come back as `[error] ...` observations with
`executed=false`. A requested `timeout_sec` is clamped to the per-call budget
(`min(requested, budgets.tool_timeout_sec)`); the effective value and whether the command timed
out are recorded in the tool result (`timeout_sec`, `timed_out`). Command-level failures
(non-zero exit, exit 124 on timeout) are observations; failures of the environment transport
(a Harbor exec/upload exception, Harbor's backstop exec timeout) raise `EnvInfraError` and stop
the episode as `infra:...`, also during replay, without showing the learner anything.

**Loop and stop reasons** (implemented in `learning_loop/episodes/episode.py`): request -> response ->
execute each tool call in order -> append observations -> repeat.

| stop reason | category | meaning |
|---|---|---|
| `model_finished` | model | reply without tool calls |
| `budget:max_turns`, `budget:max_episode_tokens` | budget | experimental budgets |
| `budget:usage_unavailable` | budget | `max_episode_tokens` is set but a response reported no usage |
| `budget:output_truncated` | budget | `finish_reason=length` and no valid tool call |
| `safety:agent_timeout`, `safety:cancelled` | safety | wall-clock limits |
| `model_error:<msg>` | model_error | endpoint rejected the request (4xx, e.g. context overflow) |
| `infra:<msg>` | infra | connection errors, timeouts, 5xx, environment failures |
| `replay:<mismatch>` | replay | restored state differs from the source (branch episodes), incl. `replay:image_mismatch` |

**Malformed tool calls.** Arguments that are not a JSON object are a `parse_error`, the call is
not executed, and the model sees `[error] could not parse arguments as JSON: ...` (or, after
`finish_reason=length`, a "cut off by max_tokens" message). Because llama.cpp re-parses the
arguments of *past* tool calls when rendering the chat template (invalid JSON there made every
later request fail with a 500), such arguments are replaced by `"{}"` in the history. That
replacement is recorded as an explicit `repair` event with the original text; the turn is marked
malformed/repaired and cannot be used as a clean training example. A tool-call block the server
could not parse (reported as `unparsed_tool_call`) is a `parse_error` too and marks the turn
malformed even when other calls in the same turn were valid; a turn with only such blocks stops
as `model_error:unparsed_tool_call`.

**Two modes.** CLI mode (the options in `configs/local-llama.yaml`: `api_base`, `api_key`,
`max_turns`, `temperature`, `max_tokens`, `workdir`, `command_timeout_sec`, `max_output_chars`,
`system_prompt_path`, `extra_body`) sends no request seed and relies on Harbor's
`[agent].timeout_sec`. Plan mode (`episode_plan_path=<EpisodePlan JSON>`, plus `record_dir`) is
used by the loop: the plan fixes the policy (endpoint or scripted fixture), seeds, budgets,
prompts, tools, the replay prefix and the task's state contract. List options with
`PYTHONPATH=.:src uv run harbor agent schema evaluation.agents.tool_agent:ToolAgent`.

**Outputs.**

| file | where | content |
|---|---|---|
| `events.jsonl` | record dir (host-only; default `<trial>/learning_loop/`), copied to `agent/` at the end | append-only lossless record: each request as sent, raw provider response, parse errors, repairs, tool calls with requested vs executed arguments, raw tool output and the exact truncated observation, fingerprints, replay checks |
| `episode.json` | same | stop reason/category, usage (provider-reported; missing = null), counts, timing, tool CPU (container cgroup delta) |
| `trajectory.json` | record dir (every turn), copied to `agent/` at the end | ATIF view for `harbor view` / `harbor analyze` |
| `messages.json` | same | the final request's `{tools, messages}`. Convenient, but **not** a complete record |
| `result.json`, `verifier/` | trial dir | Harbor's result (reward, timings, exceptions) and verifier output |

Harbor's `agent_result.n_input_tokens` / `n_output_tokens` are filled from provider usage, or
left null when the endpoint does not report it.

## Task contract

A task is a Harbor task directory:

```
task.toml         Harbor config; top-level `artifacts = [...]`; [verifier] environment_mode = "separate"
instruction.md    the learner-visible instruction
environment/      Dockerfile + learner-visible files (the only build context of the agent image)
tests/            Dockerfile + test.sh (+ hidden helpers): the separate verifier image
solution/solve.sh reference solution (the `oracle` agent)
```

Rules that keep grading honest:

- Put everything the grader needs from the agent into `artifacts`; nothing else is transferred.
- Keep reward keys uniform across tasks (`reward` only): Harbor averages per key and counts a
  missing key as 0.
- Preserve partial credit where it is meaningful (fix-stats), and keep complete success
  (`reward >= success_threshold`) separate from partial reward.
- Make shortcuts fail: design data so plausible wrong methods give a different answer, and read
  successful trajectories too (a first log-triage version let `cat a b | zcat c.gz`, which
  ignores stdin, pass by accident).
- Never run agent-written code in the grading process: run it isolated (see fix-stats above).
- Pin base images by digest (`FROM python:3.12-slim@sha256:...`): the environment's image
  identity is a hash of its build context, recorded in `episode_start` / `extra.image_identity`
  (with the local image id when reachable, informational only because rebuilds get new ids), and
  a branch whose identity differs from its source stops as `replay:image_mismatch`. (log-triage
  still `apt-get install`s python3 unpinned on top of the pinned base; the identity does not
  cover that.)
- Check `oracle` = 1.0 and `nop` = the documented baseline after every change
  (`uv run pytest -m docker tests/integration`).

### Replay contract: `[metadata.learning_loop]`

```toml
[metadata.learning_loop]
restore = "deterministic_replay"      # or "none" (no branching from this task)
fingerprint_paths = ["/app"]          # declared task state
fingerprint_exclude = ["*/__pycache__"]   # narrow fnmatch globs on absolute paths, with a reason
success_threshold = 1.0
reward_key = "reward"
observation_normalizers = [{ pattern = "...", replacement = "<mtime>", reason = "..." }]
```

Branch episodes restore a decision state by a **fresh environment + deterministic replay**, not
by Harbor's trajectory loading (which restores a conversation, not files). Before branching at
assistant turn *k*:

1. The conversation starts from the exact historical request messages of turn *k*.
2. Each earlier turn's *executed* arguments are re-run in order, without calling the model.
3. Every replayed observation must equal the source observation after applying **only** the
   task's declared normalizers (recorded in the `replay_check` event).
4. The fingerprint of the declared state must equal the source fingerprint before each replayed
   turn (when recorded) and before turn *k*.
5. Any mismatch stops the episode as `replay:<mismatch>` before the intervention is executed or
   the model is called (fail closed).

The fingerprint (`src/learning_loop/episodes/fingerprint.py`, shipped into the container and run with
`python3 -I -B` from `/`) covers, for each declared path recursively: file content hashes, file
type, permissions, owner, symlink targets (not followed), missing paths, and the tool working
directory. It does **not** cover modification times, processes, network or clock state. Tasks
declaring deterministic replay must therefore:

- ship `python3` in the agent image;
- not depend on network access, background processes, persistent shells, wall-clock time or
  randomness outside the declared state;
- be insensitive to inference waiting time (tested with inserted delays, locally and in Docker).

Normalizers must be narrow and justified; the shipped tasks declare one, for `ls -l`
modification times of files written during the episode.

## Generators and splits

`generators/` holds versioned, byte-deterministic generators keyed by family
(`evaluation.generators.generate(family, difficulty, seed, out_dir)`); each writes a complete
task directory in the layout above and self-checks its shortcut resistance:

| family | difficulties vary | traps checked at generation |
|---|---|---|
| `log-triage` | 2/3/4 log files, gzip, archive subdirectory | every proper file subset gives a different top IP; counting >=400 fails; counting any `5xx` field fails (medium/hard) |
| `fix-stats` | 4-6 functions, 1/2/4 injected bugs (several variants each), fewer visible tests on hard | reference scores 1.0; every bug fails a hidden check; hidden inputs differ from visible tests; `nop_reward` recorded in params |
| `csv-revenue` | 1-3 CSV files, quoted commas, a reordered header | ignoring the status filter, summing quantities, counting rows, file subsets, naive comma splitting and fixed column positions all give a different region |
| `count-errors` | 2-4 `.log` files, nested directory on hard | lowercase `error` lines, `ERROR` in a `.txt` decoy |

`csv-revenue` is the held-out family in `splits/pilot.yaml`. `count-errors` also runs on the
local fixture backend (`[metadata.local_fixture]`: `environment/files/` becomes the work
directory, `tests/grade.py` grades a copy of the artifacts). That backend runs host
subprocesses: it is not a sandbox and accepts scripted policies only.

Split files list every instance once (family, difficulty, generator seed or static task dir,
split) and group them into panels of one split (`train`, `dev`, `final`). `learning_loop.tasks.instances`
materializes instances (content-hashed `TaskInstance` records, plus a hash of learner-visible
content) and rejects: duplicate ids, panels mixing splits, held-out families in train panels,
the same generator coordinates under two ids, and identical learner-visible content in different
splits (disjoint seeds alone are not treated as proof of disjoint content). `pilot.yaml`
includes an easy-only training panel with medium/hard dev panels, and per-family dev panels for
family-exposure schedules.

To add a family: write `generators/<family>.py` (FAMILY, VERSION, SKILLS, DIFFICULTIES,
`generate()`), register it in `generators/__init__.py`, check `oracle`/`nop` in Docker, then
reference instances from a split file. Bump VERSION whenever output for an existing seed changes;
`RNG_VERSION` versions the random stream separately (v2 only pinned the base images, so every
instance kept its v1 content).

## Gotchas

- In the smoke runs, gemma-4-E4B showed exactly the tool misuse worth training away: writing
  `read_file`'s line-number prefixes back into the file, piping into a command that ignores
  stdin, and re-running an identical failing command 5 times.
- `max_tokens` covers reasoning *plus* the tool call (e.g. a whole file for `write_file`), so the
  CLI default is 8192.
- `network_mode = "no-network"` is still rejected by Harbor 0.23.0's Docker backend on this Mac
  (re-tested 2026-09-29 with a fix-stats oracle trial): `ValueError: network_mode='no-network' is
  not supported by EnvironmentType.DOCKER environment. Environment providers must enforce the
  requested network policy or reject the task.` Harbor enables `disable_internet` only when its
  egress-control kernel probe (nftables fib rules) succeeds, and it fails on this Docker Desktop
  VM. So tasks use the default (public) network; hidden grading material is protected by the
  separate verifier, not by network isolation, and `load_state_spec` adds the caveat
  `network: public (unmodeled)` to the state spec of every deterministic-replay task without
  `no-network` (it travels with each plan into `episode_start`). A replayed prefix that uses the
  network can therefore change state outside the fingerprint.
- `-i <task>` only works together with `-p`. To run one task, use `-p evaluation/tasks/<task>`.
- Custom agents are imported by module path: `PYTHONPATH` must include the repo root and `src/`
  (`run.sh` does this and prefers the project's `.venv/bin/harbor`).
- After editing a task's `environment/` or `tests/`, pass `--force-build` to rebuild images.
- Harbor's Terminus-2 agent uses no native tool calling (JSON keystrokes into tmux, context
  summarization); this repository keeps its own agent so the tool protocol is fixed and recorded.
