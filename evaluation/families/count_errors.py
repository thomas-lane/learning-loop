"""count-errors: count the lines containing `ERROR` in the `.log` files under /app/data.

The smallest family; it also runs on the local fixture backend, so the orchestration tests
and the smoke split use it. The instruction is the same at every difficulty: count the lines
that contain the uppercase string `ERROR` anywhere (also inside a longer word, once per
line) in every `.log` file under /app/data, including subdirectories.

Data per difficulty (`logs` lists the `.log` files):

- easy: two logs in /app/data and `notes.txt`;
- medium: three logs and `notes.txt`; the logs also have lines that contain `ERROR` only
  inside a longer word (`ERROR_RATE`, `NOERROR`, `ERRORS`) and ERROR lines that contain
  `ERROR` twice;
- hard: as medium, with one more log in a subdirectory (`data/old/`).

Traps, each declared as a shortcut where the data contains it:

- every `.log` file has a lowercase `error` line, so case-insensitive matching overcounts
  (`case-insensitive`, every level);
- `notes.txt` has `ERROR` lines that do not count, so matching every file overcounts
  (`all-files`, every level);
- medium/hard: the in-word lines make whole-word matching (`whole-word`, `grep -w`) and
  matching the level field (`level-field`) undercount, and the twice-`ERROR` lines make
  counting occurrences instead of lines (`occurrences`, `grep -o`) overcount;
- hard: a non-recursive glob misses the subdirectory's log and undercounts
  (`non-recursive`).

Every trap holds by construction (each count it changes is at least 1), so `build` never
redraws.
"""

import re

from learning_loop.tasks.spec import ExactAnswer, Family, Solution, TaskSpec

LEVELS = ["INFO", "INFO", "INFO", "DEBUG", "WARN"]
MSGS = ["request served", "cache miss", "retrying connection", "job finished", "user login", "queue drained"]
# lines whose only `ERROR` is inside a longer word; the level (second field) is never ERROR
IN_WORD = [
    "WARN alert: ERROR_RATE above threshold for {msg}",
    "DEBUG resolver: NOERROR reply while handling {msg}",
    "INFO health: ERRORS=0 since last check ({msg})",
    "WARN monitor: ERROR_BUDGET burning fast during {msg}",
]

INSTRUCTION = """Count the lines that contain the string `ERROR` (case-sensitive) in every `.log` file under `/app/data/`, including any subdirectories.

Write just the number (nothing else) to `/app/answer.txt`.
"""

ORACLE = "find /app/data -type f -name '*.log' -exec cat {} + | grep -c ERROR > /app/answer.txt\n"
CASE_INSENSITIVE = "find /app/data -type f -name '*.log' -exec cat {} + | grep -ci error > /app/answer.txt\n"
ALL_FILES = "find /app/data -type f -exec cat {} + | grep -c ERROR > /app/answer.txt\n"
NON_RECURSIVE = "cat /app/data/*.log | grep -c ERROR > /app/answer.txt\n"
WHOLE_WORD = "find /app/data -type f -name '*.log' -exec cat {} + | grep -cw ERROR > /app/answer.txt\n"
LEVEL_FIELD = "find /app/data -type f -name '*.log' -exec cat {} + | awk '$2 == \"ERROR\"' | wc -l > /app/answer.txt\n"
OCCURRENCES = "find /app/data -type f -name '*.log' -exec cat {} + | grep -o ERROR | wc -l > /app/answer.txt\n"


def _log(rng, n_lines, n_errors, n_lower, n_in_word=0, n_double=0):
    kinds = ["E"] * n_errors + ["e"] * n_lower + ["w"] * n_in_word + ["D"] * n_double
    kinds += ["n"] * (n_lines - len(kinds))
    rng.shuffle(kinds)
    lines = []
    for k in kinds:
        ts = f"2026-09-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}Z"
        if k == "E":
            lines.append(f"{ts} ERROR {rng.choice(MSGS)} (code {rng.randint(100, 999)})")
        elif k == "D":
            lines.append(f"{ts} ERROR {rng.choice(MSGS)}: upstream replied ERROR {rng.randint(500, 504)}")
        elif k == "e":
            lines.append(f"{ts} INFO recovered from transient error in {rng.choice(MSGS)}")
        elif k == "w":
            lines.append(f"{ts} " + rng.choice(IN_WORD).format(msg=rng.choice(MSGS)))
        else:
            lines.append(f"{ts} {rng.choice(LEVELS)} {rng.choice(MSGS)}")
    return "\n".join(lines) + "\n"


def _count(files, take_file, count_line):
    """count_line(line) -> how many the method counts for that line (0 or 1, or occurrences)."""
    n = sum(count_line(line) for rel, data in files.items() if take_file(rel) for line in data.decode().splitlines())
    return {"/app/answer.txt": f"{n}\n"}


def _is_log(rel):
    return rel.startswith("data/") and rel.endswith(".log")


def _has(line):
    return int("ERROR" in line)


def build(ctx):
    rng, p = ctx.rng, ctx.params
    files = {}
    for i, rel in enumerate(p["logs"]):
        # in medium/hard data the first log always gets one in-word and one twice-ERROR line,
        # so those traps never depend on the draw
        lo = 1 if i == 0 else 0
        n_in_word = rng.randint(lo, 3) if p["in_word"] else 0
        n_double = rng.randint(lo, 2) if p["in_word"] else 0
        files[rel] = _log(rng, rng.randint(*p["lines"]), rng.randint(2, 7), rng.randint(1, 4), n_in_word, n_double)
    files["data/notes.txt"] = _log(rng, 8, rng.randint(2, 4), 0)
    oracle = Solution(ORACLE, lambda f: _count(f, _is_log, _has))
    shortcuts = {
        "case-insensitive": Solution(CASE_INSENSITIVE, lambda f: _count(f, _is_log, lambda line: int("error" in line.lower()))),
        "all-files": Solution(ALL_FILES, lambda f: _count(f, lambda rel: rel.startswith("data/"), _has)),
    }
    if p["in_word"]:
        shortcuts["whole-word"] = Solution(WHOLE_WORD, lambda f: _count(f, _is_log, lambda line: int(re.search(r"\bERROR\b", line) is not None)))
        shortcuts["level-field"] = Solution(LEVEL_FIELD, lambda f: _count(f, _is_log, lambda line: int(line.split()[1:2] == ["ERROR"])))
        shortcuts["occurrences"] = Solution(OCCURRENCES, lambda f: _count(f, _is_log, lambda line: line.count("ERROR")))
    if any(rel.count("/") > 1 for rel in p["logs"]):
        shortcuts["non-recursive"] = Solution(NON_RECURSIVE, lambda f: _count(f, lambda rel: _is_log(rel) and rel.count("/") == 1, _has))
    expected = oracle.model({k: v.encode() for k, v in files.items()})["/app/answer.txt"].strip()
    return TaskSpec(
        instruction=INSTRUCTION,
        files=files,
        grader=ExactAnswer("/app/answer.txt", expected),
        oracle=oracle,
        shortcuts=shortcuts,
        params={"logs": list(p["logs"]), "expected": int(expected)},
    )


FAMILY = Family(
    name="count-errors",
    version=5,  # v4: one instruction for every level; medium/hard in-word and twice-ERROR lines
    cluster="text-analytics",
    category="shell",
    skills=("shell", "grep", "counting"),
    difficulties={
        "easy": {"logs": ["data/app.log", "data/worker.log"], "lines": [12, 30], "in_word": False},
        "medium": {"logs": ["data/app.log", "data/worker.log", "data/db.log"], "lines": [20, 40], "in_word": True},
        "hard": {"logs": ["data/app.log", "data/worker.log", "data/db.log", "data/old/app.1.log"], "lines": [20, 40], "in_word": True},
    },
    build=build,
    agent_timeout_sec=120.0,
    local_fixture=True,
)
