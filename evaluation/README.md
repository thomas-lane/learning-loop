# evaluation/: Harbor tasks, generators, splits and the learner agent

Everything that runs inside Harbor (the framework that builds a task's container,
runs an agent in it and grades the result): the tasks, the tool-calling learner agent, the task
generators, and the split files that assign task instances to training or evaluation. Terms
such as panel, replay, fingerprint and normalizer are defined in the
[glossary](../docs/glossary.md).

```
evaluation/
├── run.sh                      # sets PYTHONPATH (repo root + src/), runs `harbor run -c configs/local-llama.yaml`
├── configs/local-llama.yaml    # Harbor job config: agent, model, URL, tasks, attempts
├── agents/
│   ├── tool_agent.py           # ToolAgent: Harbor adapter around learning_loop.episodes.episode
│   ├── tools.py                # tool schemas and handlers (bash, read_file, write_file)
│   └── system_prompt.md        # the system prompt, verbatim ({workdir} is filled in)
├── tasks/                      # hand-written tasks: log-triage, fix-stats
├── generators/                 # versioned generators: log-triage, fix-stats, csv-revenue, count-errors
├── splits/                     # pilot.yaml (experiments/pilot.yaml), smoke.yaml (Docker smoke), fixture.yaml (local fixture backend)
├── scripts/gen_log_triage_data.py   # regenerates tasks/log-triage/environment/logs
└── jobs/                       # Harbor outputs (gitignored)
```

## Quick start

Docker must be running. CLI flags override the config file.

```bash
# No model: oracle (reference solution) must score 1.0; nop (does nothing) is the baseline
uv run harbor run -p evaluation/tasks -a oracle -o evaluation/jobs
uv run harbor run -p evaluation/tasks -a nop -o evaluation/jobs      # log-triage 0.0, fix-stats 0.33

# The agent against an OpenAI-compatible server
evaluation/run.sh                                          # all tasks, config defaults
evaluation/run.sh -p evaluation/tasks/log-triage -k 5      # one task, 5 attempts
evaluation/run.sh -m my-model --ak api_base=http://gpu-box:8000/v1

uv run harbor view evaluation/jobs                         # browse trajectories
```

The learning loop starts trials from Python instead, through `HarborDockerBackend` in
`src/learning_loop/episodes/backends.py`, which passes each episode's plan to the agent (plan
mode, below).

## Task families

Every family has a generator; log-triage and fix-stats also have a hand-written instance in
`tasks/`. fix-stats scores the fraction of hidden checks passed (partial credit); the others
score 1 if `/app/answer.txt` is right, else 0.

| family | the agent must | easy / medium / hard | wrong methods that fail |
|---|---|---|---|
| `log-triage` | name the client IP with the most HTTP 5xx responses in `/app/logs/` | 2 / 3 / 4 log files, some gzipped; hard adds `archive/` | skipping any file; counting all >= 400; (medium/hard) matching any ` 5xx ` field |
| `fix-stats` | fix the bugs in `/app/stats.py` | 4 / 5 / 6 functions, 1 / 2 / 4 bugs; visible tests cover all / half / a quarter of the bugs | hard-coding visible test values (hidden checks use other inputs) |
| `csv-revenue` | name the region with the highest completed-order revenue in `/app/data/*.csv` | 1 / 2 / 3 files; medium adds quoted commas; hard reorders one file's columns | ignoring status, summing quantities, counting rows; (medium/hard) skipping a file, naive `,` splitting; (hard) fixed column positions |
| `count-errors` | count lines containing `ERROR` in the `.log` files under `/app/data/` | 2 / 3 / 4 files; hard adds a subdirectory | case-insensitive grep, counting the `.txt` decoy, (hard) non-recursive glob |

The log-triage and csv-revenue generators compute each wrong method's answer for every instance
and draw new data until all of them fail. The fix-stats generator draws hidden checks until the
reference scores 1.0 and every bug fails at least one, and records the no-op score as
`nop_reward`. count-errors' traps hold by construction: every `.log` file has a lowercase
`error` line, and the `.txt` decoy and the hard subdirectory file both contain `ERROR` lines.

## How a trial runs

1. Harbor builds `environment/Dockerfile` and starts the agent container.
2. `ToolAgent.run()` runs **on the host** (so `localhost` in `api_base` is the host) and acts on
   the container only through Harbor's `exec` and `upload_file`.
3. When the agent stops, Harbor copies only the task's declared **artifacts** (e.g.
   `/app/answer.txt`) out of the agent container.
4. A **separate verifier container**, built from `tests/Dockerfile` (context `tests/`), receives
   the artifacts at the same paths, empties `/logs/verifier/` and runs `tests/test.sh`, which
   writes `/logs/verifier/reward.txt` (one number) or `reward.json` (`{"reward": x, ...}`).

### Hidden grading

- Hidden tests and the reference solution never enter the agent's container, and a reward file
  the agent plants there is discarded (checked in `tests/integration/test_harbor_tasks.py`).
- `/logs/agent` is writable from the container, so a planted symlink there could redirect the
  host's writes. The agent therefore keeps its records in a host-only directory and copies them
  into `agent/` at the end with `copy_into_untrusted_dir()`, which writes a temp file opened
  with `O_NOFOLLOW` and renames it over the target.
- **The agent container has public network access**: Harbor 0.23.0 rejects
  `network_mode = "no-network"` on Docker Desktop for Mac (its nftables egress probe fails).
  Grading is protected by the separate verifier, not by network isolation.

Agent-written code never runs in the process that writes the reward. The fix-stats grader
(`tests/test_hidden.py`):

- makes `/tests` mode 0700 and runs `/app/stats.py` in a child `python3 -I -B` process as
  `nobody`, which therefore cannot read the expected values;
- has the child print raw observations as JSON on one marker line, and accepts only plain JSON
  numbers and literal booleans, so an object whose `__eq__` is always true fails its check (a
  second observation line fails all checks);
- kills every `nobody` process and deletes any existing reward file before writing
  `reward.json`, so a reward from an `atexit` hook or forked daemon never counts
  (`test_forged_fix_stats_artifact_does_not_score`).

## The agent (`agents/tool_agent.py`)

**The model sees only:** the system prompt, the task's `instruction.md` as the user message, the
tool schemas from `tools.py` (as `tools=[...]`, rendered by the server's chat template), and every
earlier assistant message and tool result. Tool output over `max_output_chars` keeps its head and
tail. Returned `reasoning_content` is sent back verbatim; the chat template decides whether it
becomes tokens (Gemma 4's and Qwen3's keep it for every turn after the last user message).

**Tools:** `bash` (fresh `bash -c` per call in `/app`), `read_file` (numbered lines, 200 per
page) and `write_file` (uploads the content). A requested `timeout_sec` is capped at the budget
(`episode.tool_timeout_sec` in plan mode, `command_timeout_sec` in CLI mode, both 60 s by
default); the result records the effective `timeout_sec` and `timed_out`. In Docker each command
runs under coreutils `timeout` when the image has it, so a timed-out command does not keep
running and change the state that replay compares. Failing commands, unknown tools and bad
arguments come back as `[error] ...` observations. An environment failure (Harbor exec/upload
exception or backstop timeout) raises `EnvInfraError` and stops the episode as `infra:...`,
also during replay.

**Stop reasons** (`learning_loop/episodes/episode.py`; [what to do about
them](../docs/operations.md#episodes-and-stages)):

| stop reason | category | meaning |
|---|---|---|
| `model_finished` | model | reply without tool calls |
| `budget:max_turns`, `budget:max_episode_tokens` | budget | experimental budgets |
| `budget:usage_unavailable` | budget | `max_episode_tokens` is set but a response reported no usage |
| `budget:output_truncated` | budget | `finish_reason=length` and no valid tool call |
| `safety:agent_timeout` | safety | the plan's wall-clock limit (`agent_timeout_sec`) |
| `safety:cancelled` | safety | cancelled from outside, e.g. by Harbor's `[agent].timeout_sec` in CLI mode |
| `model_error:<msg>` | model_error | the endpoint rejected the request (4xx other than 408/429, e.g. context overflow) |
| `infra:<msg>` | infra | connection errors, timeouts, 408/429, 5xx, environment failures |
| `replay:<mismatch>` | replay | branch episodes only: the restored state differs from the source ([replay contract](#replay-contract-metadatalearning_loop)) |

**Malformed tool calls** (arguments that are not a JSON object, or a block the server reports as
`unparsed_tool_call`) are not executed; the model gets an `[error] ...` observation and the turn
is marked malformed. The editor rejects a malformed turn (`malformed_turn`), so it never becomes
the rejected side of a preference pair. Invalid arguments are replaced by `"{}"` in the history,
because llama.cpp re-parses past arguments to render the chat template and would fail every later
request; the original is kept in a `repair` event. A turn with only unparsed blocks stops as
`model_error:unparsed_tool_call`.

**Modes.** *CLI mode* takes the options in `configs/local-llama.yaml` (`max_tokens` covers
reasoning plus the tool call, hence 8192), sends no request seed and relies on Harbor's
`[agent].timeout_sec`. *Plan mode* (`episode_plan_path=<EpisodePlan JSON>`, plus `record_dir`)
is what the loop uses: the plan (`EpisodePlan` in `src/learning_loop/core/interfaces.py`) fixes
policy, seeds, budgets, prompts, tools, replay prefix and the task's state contract. List
options with
`PYTHONPATH=.:src uv run harbor agent schema evaluation.agents.tool_agent:ToolAgent`.

**Outputs**, written to the host-only `record_dir` (default `<trial>/learning_loop/`) and copied
into the trial's `agent/` at the end:

| file | content |
|---|---|
| `events.jsonl` | append-only, lossless record: requests as sent, raw responses, parse errors, repairs, requested vs executed tool arguments, raw and truncated tool output, fingerprints, replay checks |
| `episode.json` | stop reason and category, provider-reported usage (missing = null), counts, timing, tool CPU time |
| `trajectory.json` | ATIF view for `harbor view` / `harbor analyze` |
| `messages.json` | the final message history and tool schemas (a view; `events.jsonl` is the record) |

Harbor's `result.json` and `verifier/` sit in the trial dir; its token counts come from provider
usage, or are null.

## Task contract

```
task.toml         Harbor config; top-level `artifacts = [...]`; [verifier] environment_mode = "separate"
instruction.md    the instruction the learner sees
environment/      Dockerfile + learner-visible files (the agent image's only build context)
tests/            Dockerfile + test.sh (+ hidden helpers): the separate verifier image
solution/solve.sh reference solution (the `oracle` agent)
```

Rules:

- Everything the grader needs goes in `artifacts`; nothing else is transferred.
- Write only the `reward` key: Harbor averages each key across trials and counts a key missing
  from a trial as 0.
- Keep complete success (`reward >= success_threshold`) separate from partial credit: only
  complete success counts as success or can be edited.
- Make shortcuts fail, and read successful trajectories for answers that pass by luck. For
  example, `cat a b | zcat c.gz` reads only `c.gz` (zcat ignores stdin), which is why log-triage
  generators make every proper subset of the files give a different answer.
- Never run agent-written code in the grading process (see fix-stats above).
- Pin base images by digest. Each episode records an **image identity** (a hash of the build
  context) in `episode_start`, and a branch whose identity differs from its source stops as
  `replay:image_mismatch`. An unpinned tag could change the image without changing that hash.
  Packages installed at build time are outside the hash too (log-triage `apt-get install`s
  python3 unpinned).
- After every change, check `oracle` = 1.0 and `nop` = the documented baseline
  (`uv run pytest -m docker tests/integration`).

### Replay contract: `[metadata.learning_loop]`

```toml
[metadata.learning_loop]
restore = "deterministic_replay"      # or "none" (no branching from this task)
fingerprint_paths = ["/app"]          # the task's declared state
fingerprint_exclude = ["*/__pycache__"]   # narrow fnmatch globs on absolute paths; explain each in a comment
success_threshold = 1.0
reward_key = "reward"
observation_normalizers = [{ pattern = "...", replacement = "<mtime>", reason = "..." }]
```

A branch at assistant turn *k* rebuilds that state in a fresh environment by deterministic
replay (Harbor's trajectory loading restores only the conversation, not the files):

1. The conversation starts from the exact historical request messages of turn *k*.
2. Earlier turns' *executed* arguments are re-run in order, without calling the model.
3. Each replayed observation must equal the source's after applying **only** the task's declared
   normalizers (recorded in a `replay_check` event).
4. The fingerprint of the declared state must match the source's before each replayed turn (when
   recorded) and before turn *k*.
5. Any mismatch stops the episode as `replay:<mismatch>` before the intervention or model call,
   e.g. `replay:observation:turn=2:call=0` or `replay:fingerprint:before_intervention`
   (all names: `_Episode.replay()` in `episode.py`).

The fingerprint (`src/learning_loop/episodes/fingerprint.py`, run in the container with
`python3 -I -B`) covers content hashes, file types, permissions, owners, symlink targets, missing
paths and the working directory. It leaves out modification times, which depend on wall-clock
time and would make every replay differ, and processes, network and clock. A replayable task
must therefore ship `python3` in its image, must not depend on network, background processes,
persistent shells, wall-clock time or undeclared randomness, and must not be affected by
inference delays. Because containers have network access, `load_state_spec` adds the caveat
`network: public (unmodeled)` to every such task without `no-network`. Each normalizer needs
exactly `pattern`, `replacement` and `reason`; the shipped tasks declare one, for `ls -l`
modification times.

## Generators and splits

`generate(family, difficulty, seed, out_dir)` in `evaluation.generators` writes a complete task
directory, byte-identical for the same inputs. `count-errors` also runs on the local fixture
backend (`[metadata.local_fixture]`), which runs commands as host subprocesses with no sandbox.
`LocalFixtureBackend` therefore refuses any policy other than a scripted one.

A **split file** declares every instance once (family, difficulty, generator seed or static task
dir, and split `train`, `dev`, `final` or `external`) and groups instances into **panels** of one
split. `learning_loop.tasks.instances` materializes them as content-hashed `TaskInstance` records
and rejects duplicate ids, panels mixing splits, held-out families in train panels or the train
split, the same generator coordinates under two ids, and identical learner-visible content
(`instruction.md` plus `environment/`) in different splits. The content check exists because
different seeds can still produce the same task. In `pilot.yaml`, `csv-revenue` is held out and
appears only in the `final-held-out-family` panel.

To add a family: write `generators/<family>.py` (`FAMILY`, `VERSION`, `RNG_VERSION`, `SKILLS`,
`DIFFICULTIES`, `generate()`), register it in `generators/__init__.py`, check `oracle`/`nop` in
Docker, and reference instances from a split file. Bump `VERSION` whenever output for an existing
seed changes: materialization rejects a task directory whose recorded `generator`
(`family@vN`) is not current, so stale directories are never reused. `RNG_VERSION` versions the
random stream separately, so changes that leave the random draws alone (such as pinning base
images) keep the same generated data and answers.
