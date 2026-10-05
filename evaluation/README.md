# evaluation/: task families, splits and the learner agent

Everything that runs inside Harbor (the framework that builds a task's container, runs an agent
in it and grades the result): the task families, the tool-calling learner agent and the split
files that assign task instances to training or evaluation. Terms such as panel, replay,
fingerprint and normalizer are defined in the [glossary](../docs/glossary.md).

```
evaluation/
├── families/                   # one module per task family: FAMILY = Family(..., build=build)
├── splits/                     # pilot.yaml (experiments/pilot.yaml), smoke.yaml (Docker smoke), fixture.yaml (local fixture backend)
├── agents/
│   ├── tool_agent.py           # ToolAgent: Harbor adapter around learning_loop.episodes.episode
│   ├── tools.py                # tool schemas and handlers (bash, read_file, write_file)
│   └── system_prompt.md        # the system prompt, verbatim ({workdir} is filled in)
├── run.sh                      # sets PYTHONPATH (repo root + src/), runs `harbor run -c configs/local-llama.yaml`
├── configs/local-llama.yaml    # Harbor job config: agent, model, URL, tasks (evaluation/rendered), attempts
├── rendered/                   # task directories from `loop render-tasks` (gitignored)
└── jobs/                       # Harbor outputs (gitignored)
```

The framework that turns a family into a task directory lives in `src/learning_loop/tasks/`:
`spec.py` (what a family declares), `generate.py` (the per-instance checks), `render.py` (the
only writer of task directories) and `runtime/` (the grader and environment probe that run inside
the containers).

## Quick start

Docker must be running. Task directories are rendered from the families, so render a split (or
some panels of it) first; a run materializes its instances the same way.

```bash
uv run loop render-tasks evaluation/splits/pilot.yaml --panel dev      # -> evaluation/rendered/<id>/

# No model: oracle (reference solution) must succeed; nop (does nothing) scores the no-op baseline
uv run harbor run -p evaluation/rendered -a oracle -o evaluation/jobs
uv run harbor run -p evaluation/rendered -a nop -o evaluation/jobs

# The agent against an OpenAI-compatible server
evaluation/run.sh                                                   # every rendered task, config defaults
evaluation/run.sh -p evaluation/rendered/log-triage__easy__s101 -k 5   # one task, 5 attempts
evaluation/run.sh -m my-model --ak api_base=http://gpu-box:8000/v1

uv run harbor view evaluation/jobs                                  # browse trajectories
```

The learning loop starts trials from Python instead, through `HarborDockerBackend` in
`src/learning_loop/episodes/backends.py`, which passes each episode's plan to the agent (plan
mode, below).

## Task families

A **family** is a kind of task whose instances are drawn from a seed. Each instance is the
directory `render(family, difficulty, seed)` writes. fix-stats scores the fraction of hidden
checks passed (partial credit); the others score 1 if `/app/answer.txt` is right, else 0. The
**shortcuts** are the known wrong methods, each declared as a runnable solution that must fail on
every instance (see [Checks](#checks-every-instance-and-every-family)).

| family | cluster | the agent must | easy / medium / hard | shortcuts (must fail) |
|---|---|---|---|---|
| `log-triage` | text-analytics | name the client IP with the most HTTP 5xx responses in `/app/logs/` | 2 / 3 / 4 log files, some gzipped; hard adds `archive/` | `plain-only` (skip `.gz`), `all-4xx`; medium/hard `any-5xx-field`; hard `top-level-only` |
| `count-errors` | text-analytics | count lines containing `ERROR` in the `.log` files under `/app/data/` | 2 / 3 / 4 files; hard adds a subdirectory | `case-insensitive`, `all-files` (counts the `.txt` decoy); hard `non-recursive` |
| `csv-revenue` | tabular-data | name the region with the highest completed-order revenue in `/app/data/*.csv` | 1 / 2 / 3 files; medium adds quoted commas; hard reorders one file's columns | `no-status-filter`, `quantity-sum`, `row-count`; medium/hard `naive-split`; hard `fixed-columns` |
| `fix-stats` | fix-code | fix the bugs in `/app/stats.py` | 4 / 5 / 6 functions, 1 / 2 / 4 bugs; visible tests cover all / half / a quarter of the bugs | medium/hard `visible-bugs-only` (fixes only the bugs the visible tests show) |

Beyond the declared shortcuts, the log-triage and csv-revenue families redraw until every proper
subset of the files gives a different answer, so skipping any file fails. That also catches
`cat a b | zcat c.gz`, which reads only `c.gz` because zcat ignores stdin. fix-stats redraws
until its hidden checks catch every injected bug. The checks use inputs other than the visible
tests', so hard-coding the visible expectations fails. count-errors also runs on the local
fixture backend (`[metadata.local_fixture]`), which runs commands as host subprocesses with no
sandbox; `LocalFixtureBackend` therefore refuses any policy other than a scripted one.

## How a family defines a task

A family module defines `FAMILY = Family(name, version, cluster, skills, difficulties, build, ...)`
(`src/learning_loop/tasks/spec.py`). `build(ctx)` gets a `GenContext` (`rng`, `difficulty`,
`seed` and the difficulty's `params`) and returns a `TaskSpec`, the whole task in one place:

| field | what it is |
|---|---|
| `instruction` | the learner-visible instruction (`instruction.md`) |
| `files` | learner-visible files, keyed by path relative to `/app` |
| `grader` | a grader kind with its hidden answer key: `ExactAnswer`, `NumericAnswer`, `JsonAnswer` or `Checks` |
| `oracle` | the reference `Solution` |
| `shortcuts` | named wrong `Solution`s that must fail |
| `params` | values recorded in `task.toml` (`params_json`), outside the agent's container |

A `Solution` has two forms. `shell` is a script that runs in the container: the oracle becomes
Harbor's `solution/solve.sh`, and the shortcuts go to `solution/shortcuts/`. `model` is a Python
function that predicts, from the task's files, the artifacts that script writes. The model lets
every instance be checked in milliseconds without Docker; the family's Docker test checks that
each model is right about its script. csv-revenue writes both forms from one Python source, so
they cannot drift apart.

`ctx.rng` is the only randomness a family may use. `tasks/family_lint.py` rejects family modules
that import `random`, `secrets`, `uuid` or `time`, or call `os.urandom`, `.now()`, `.today()` or
`.utcnow()`, and a unit test runs it on every family. This keeps a rendered instance a pure
function of (family, `rng_version`, difficulty, seed), byte for byte on any host. Bump a family's
`version` whenever the output for an existing seed changes: materialization rejects a task
directory rendered by another version, so stale directories are never reused. `rng_version`
versions the random stream separately, for changes that leave the draws alone.

## Checks: every instance and every family

**Every instance, at generation** (`tasks/generate.py`). The shared grader that runs in the
verifier also grades, in-process, the artifacts each `Solution.model` predicts:

- the oracle must reach the success threshold, otherwise the family is wrong and generation fails
  (`GenerationError`);
- every shortcut, and doing nothing (the initial files as they are), must stay below it.

A draw that fails the second check, or for which `build` or a model raises `Reject`, is replaced
by the next draw from the same random stream (up to 100). Families raise `Reject`, for example,
when the answer is tied. The predicted rewards are recorded in `params_json` (`oracle_reward`,
`nop_reward`, `shortcut_rewards`). An instance therefore cannot exist unless it passed these
checks, on every seed and at no Docker cost.

**Every family, in Docker** (`tests/integration/test_tasks_families_docker.py`, marker `docker`).
Generation trusts the models, and this test checks that trust. For one seed of every difficulty,
the oracle, doing nothing and every shortcut run as real scripts in Docker. Each must score
exactly its predicted reward, with the verifier's probe and digest check passing. Then an episode
that writes and runs the oracle is branched at the oracle's run. The branch re-executes the
earlier turn in a fresh container, and every fingerprint and the oracle's output must match.
Run it after changing a family.

## Environment guarantees

The renderer (`tasks/render.py`) is the only code that writes task directories, so these hold by
construction for every task. Its unit tests check each one on the rendered files.

| guarantee | how | why |
|---|---|---|
| No network in either container | `environment/docker-compose.yaml` and `tests/docker-compose.yaml` set `network_mode: "none"`; Harbor appends a task's compose file after its own, for the agent container (context `environment/`) and for the separate verifier (context `tests/`) | outputs, replays and grades cannot depend on anything outside the task. Harbor's own `network_mode = "no-network"` needs nftables support in the Docker host's kernel, which Docker Desktop for Mac lacks; the compose setting works on every Docker host |
| Pinned images, nothing installed | each Dockerfile is `FROM` the profile's digest-pinned base, `ENV`, `WORKDIR`, `COPY`; no `RUN` | a build cannot change with package mirrors, and the build-context hash (which contains the pinned `FROM`) identifies the image exactly, which is what replay compares (`image_identity`) |
| Fixed process environment | the profile's `ENV`: `TZ=UTC`, `LANG`/`LC_ALL=C.UTF-8`, `PYTHONHASHSEED=0`, `PYTHONDONTWRITEBYTECODE=1`, `HOME=/root`; compose sets the hostname `task` | time zone, sort collation and Python set/dict-of-str order would otherwise make the same command print different output in a replay; bytecode caches embed source mtimes and would change the fingerprinted state |
| Fixed file times | every rendered file and directory gets the mtime 2026-01-01T00:00:00Z plus a sub-second part derived from its content; Docker's `COPY` keeps it | `ls -l` and `stat` show the same times on every host and re-render. The sub-second part differs between files with different content because BuildKit sends a build context incrementally and skips a file whose path, size and mtime match a copy it already holds, even from another task's build; with one constant mtime, two same-sized answer keys were built from the stale copy |
| The answer stays hidden | the answer key (`tests/key.json`), grader and probe are only in the verifier's build context; the agent's context is `environment/` | the learner cannot read the answer |
| The verifier grades its own files | `task.toml`'s `[verifier] env` carries `LL_TESTS_SHA256`, a digest of `grade.py`, `probe.py` and `key.json`; `grade.py` recomputes it and refuses to grade on a mismatch | `task.toml` reaches the container without passing through the image build, so even a stale build (above) cannot grade with another task's key |
| The verifier checks its environment | `tests/test.sh` runs `grade.py --probe '<profile expectation>'`; `runtime/probe.py` checks that DNS and a TCP connection to a public address fail, the `ENV` values, the hostname, the required tools and that only Harbor's `sh -c "sleep infinity"` is running | a grade is only written from a container that matches the profile |
| Fixed resource limits | the profile's `cpus` and `memory_mb` in `[environment]` | every container gets the same CPU and memory limits, whatever the host has |

A **profile** (`PROFILES` in `spec.py`) is one container environment: base image, `ENV`,
hostname, required tools, resources and allowed processes. Today there is one, `python@1`
(`python:3.12-slim`). It has bash, coreutils (including `timeout`), gzip, awk, grep, sed, find,
tar, diff and Python 3.12 with `sqlite3`, `csv`, `json` and `tomllib`. It has no git, sqlite3
CLI, make, patch, jq or curl.

## How a trial runs

1. Harbor builds `environment/Dockerfile` and starts the agent container (no network).
2. `ToolAgent.run()` runs **on the host** (so `localhost` in `api_base` is the host) and acts on
   the container only through Harbor's `exec` and `upload_file`.
3. When the agent stops, Harbor copies only the task's declared **artifacts** (e.g.
   `/app/answer.txt`) out of the agent container.
4. A **separate verifier container**, built from `tests/Dockerfile` (context `tests/`, no network),
   receives the artifacts at the same paths, empties `/logs/verifier/` and runs `tests/test.sh`.
   It checks the digest, runs the probe and grades with `tests/grade.py`, which writes
   `/logs/verifier/reward.json` (`{"reward": x}`). When the digest or probe check fails, it
   writes no reward, and the episode stops as `infra:RewardFileNotFoundError` (see
   [operations](../docs/operations.md#episodes-and-stages)).

### Hidden grading

- Hidden answer keys and the reference solution never enter the agent's container, and a reward
  file the agent plants there is discarded (checked in `tests/integration/test_harbor_tasks.py`).
- `/logs/agent` is writable from the container, so a planted symlink there could redirect the
  host's writes. The agent therefore keeps its records in a host-only directory and copies them
  into `agent/` at the end with `copy_into_untrusted_dir()`, which writes a temp file opened
  with `O_NOFOLLOW` and renames it over the target.
- Graders read each artifact with `O_NOFOLLOW` and accept only a regular file up to 10 MB. A
  symlink, directory or missing file counts as no answer, not as an error.
- Before writing, the grader deletes any existing `reward.json`/`reward.txt` and creates the new
  file with `O_EXCL | O_NOFOLLOW`. It writes only the `reward` key: Harbor averages each key
  across trials and counts a key missing from a trial as 0.

Agent-written code never runs in the process that writes the reward. Only the `checks` grader
(fix-stats) executes agent code (`src/learning_loop/tasks/runtime/grade.py`):

- it makes the key's directory mode 0700 and runs a copy of the artifact in a child
  `python3 -I -B` process as `nobody`, which therefore cannot read the expected values;
- the child gets only the check inputs and prints raw observations as JSON on one marker line;
  the grader accepts only plain JSON numbers and literal booleans for known checks, so an object
  whose `__eq__` always returns true fails its check, and a second observation line fails all
  checks;
- it kills every `nobody` process before writing, so a reward from an `atexit` hook or forked
  daemon never counts (`test_forged_fix_stats_artifact_does_not_score`).

Outside a root-owned container (e.g. on the local fixture backend) the `checks` grader refuses to
run and writes no reward, which is why the renderer rejects local-fixture families that use it.

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

What the renderer writes, for reference when reading a task directory:

```
task.toml          Harbor config; top-level `artifacts = [...]`; [verifier] environment_mode = "separate", env = {LL_TESTS_SHA256}
instruction.md     the instruction the learner sees
environment/       Dockerfile, docker-compose.yaml, files/ (becomes /app): the agent image's only build context
tests/             Dockerfile, docker-compose.yaml, test.sh, grade.py, probe.py, key.json: the separate verifier image
solution/solve.sh  the oracle; solution/shortcuts/<name>.sh: the declared wrong solutions
```

Rules a family must follow, beyond what the renderer enforces:

- Keep complete success (`reward >= success_threshold`) separate from partial credit: only
  complete success counts as success or can be edited.
- Make shortcuts fail, and declare each known wrong method as a shortcut so it is checked on
  every instance. Read successful trajectories for answers that pass by luck and add them.
- Draw everything from `ctx.rng`, and derive nothing from the clock or the host.

### Replay contract: `[metadata.learning_loop]`

```toml
[metadata.learning_loop]
restore = "deterministic_replay"      # or "none" (no branching from this task)
fingerprint_paths = ["/app"]          # the task's declared state
fingerprint_exclude = []              # narrow fnmatch globs on absolute paths
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
paths and the working directory. It leaves out modification times, because files written during
an episode get the wall-clock time and would make every replay differ. It also leaves out
processes, network and clock. A replayable task must therefore ship `python3` in its image. It
must not depend on network, background processes, persistent shells, wall-clock time or
undeclared randomness, and must not be affected by inference delays; the environment guarantees
above provide this for rendered tasks. A replayable task whose containers are not
network-isolated (no rendered `network_mode: none` overlay for both, and no Harbor
`no-network`) gets the caveat `network: public (unmodeled)` from `load_state_spec`. Each
normalizer needs exactly `pattern`, `replacement` and `reason`. Rendered tasks declare one, for
the `ls -l` times of files written during the episode.

## Splits

A **split file** declares every instance once (id, family, difficulty, seed and split `train`,
`dev`, `final` or `external`) and groups instances into **panels** of one split.
`learning_loop.tasks.instances` renders them as content-hashed `TaskInstance` records. It rejects
duplicate ids, panels mixing splits, held-out families in train panels or the train split, the
same (family, difficulty, seed) under two ids, unknown families or difficulties, and identical
learner-visible content (`instruction.md` plus `environment/`) in different splits. The content
check exists because different seeds can still produce the same task. In `pilot.yaml`,
`csv-revenue` is held out and appears only in the `final-held-out-family` panel.

## Adding a family

1. Write `families/<family>.py` with `build(ctx) -> TaskSpec` and `FAMILY = Family(...)`, and add
   the module to `FAMILIES` in `families/__init__.py`. Declare every wrong method you know of as
   a shortcut, with a shell form an agent would plausibly write and a model that predicts its
   output exactly.
2. `uv run pytest tests/unit` checks determinism, the randomness lint and that generation
   succeeds; add a test in `tests/unit/test_tasks_families.py` that re-derives the family's traps
   from the rendered files.
3. `uv run pytest -m docker tests/integration/test_tasks_families_docker.py -k <family>` checks
   every model against Docker and the oracle replay.
4. Reference instances from a split file.
