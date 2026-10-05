"""count-errors: count the lines containing `ERROR` in the `.log` files under /app/data.

The smallest family; it also runs on the local fixture backend, so the orchestration tests
and the smoke split use it. Its traps hold by construction: every `.log` file has a
lowercase `error` line (case-insensitive matching overcounts), `notes.txt` has `ERROR`
lines that do not count (matching every file overcounts), and on hard a log sits in a
subdirectory (a non-recursive glob undercounts).
"""

from learning_loop.tasks.spec import ExactAnswer, Family, Solution, TaskSpec

LEVELS = ["INFO", "INFO", "INFO", "DEBUG", "WARN"]
MSGS = ["request served", "cache miss", "retrying connection", "job finished", "user login", "queue drained"]

INSTRUCTION = """Count the lines that contain the exact (uppercase) string `ERROR` in all `.log` files under `/app/data/`{extra}.

Write just the number (nothing else) to `/app/answer.txt`.
"""

ORACLE = "find /app/data -type f -name '*.log' -exec cat {} + | grep -c ERROR > /app/answer.txt\n"
CASE_INSENSITIVE = "find /app/data -type f -name '*.log' -exec cat {} + | grep -ci error > /app/answer.txt\n"
ALL_FILES = "find /app/data -type f -exec cat {} + | grep -c ERROR > /app/answer.txt\n"
NON_RECURSIVE = "cat /app/data/*.log | grep -c ERROR > /app/answer.txt\n"


def _log(rng, n_lines, n_errors, n_lower):
    kinds = ["E"] * n_errors + ["e"] * n_lower + ["n"] * (n_lines - n_errors - n_lower)
    rng.shuffle(kinds)
    lines = []
    for k in kinds:
        ts = f"2026-09-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}Z"
        if k == "E":
            lines.append(f"{ts} ERROR {rng.choice(MSGS)} (code {rng.randint(100, 999)})")
        elif k == "e":
            lines.append(f"{ts} INFO recovered from transient error in {rng.choice(MSGS)}")
        else:
            lines.append(f"{ts} {rng.choice(LEVELS)} {rng.choice(MSGS)}")
    return "\n".join(lines) + "\n"


def _count(files, take_file, take_line):
    n = sum(1 for rel, data in files.items() if take_file(rel) for line in data.decode().splitlines() if take_line(line))
    return {"/app/answer.txt": f"{n}\n"}


def _is_log(rel):
    return rel.startswith("data/") and rel.endswith(".log")


def build(ctx):
    rng = ctx.rng
    files = {rel: _log(rng, rng.randint(12, 30), rng.randint(2, 7), rng.randint(1, 4)) for rel in ctx.params["logs"]}
    files["data/notes.txt"] = _log(rng, 8, rng.randint(2, 4), 0)
    truth = _count({k: v.encode() for k, v in files.items()}, _is_log, lambda line: "ERROR" in line)
    shortcuts = {
        "case-insensitive": Solution(CASE_INSENSITIVE, lambda f: _count(f, _is_log, lambda line: "error" in line.lower())),
        "all-files": Solution(ALL_FILES, lambda f: _count(f, lambda rel: rel.startswith("data/"), lambda line: "ERROR" in line)),
    }
    if ctx.params["subdir"]:
        shortcuts["non-recursive"] = Solution(NON_RECURSIVE, lambda f: _count(f, lambda rel: _is_log(rel) and rel.count("/") == 1, lambda line: "ERROR" in line))
    expected = truth["/app/answer.txt"].strip()
    return TaskSpec(
        instruction=INSTRUCTION.format(extra=" (including subdirectories)" if ctx.params["subdir"] else ""),
        files=files,
        grader=ExactAnswer("/app/answer.txt", expected),
        oracle=Solution(ORACLE, lambda f: _count(f, _is_log, lambda line: "ERROR" in line)),
        shortcuts=shortcuts,
        params={"logs": list(ctx.params["logs"]), "expected": int(expected)},
    )


FAMILY = Family(
    name="count-errors",
    version=3,
    cluster="text-analytics",
    category="shell",
    skills=("shell", "grep", "counting"),
    difficulties={
        "easy": {"logs": ["data/app.log", "data/worker.log"], "subdir": False},
        "medium": {"logs": ["data/app.log", "data/worker.log", "data/db.log"], "subdir": False},
        "hard": {"logs": ["data/app.log", "data/worker.log", "data/db.log", "data/old/app.1.log"], "subdir": True},
    },
    build=build,
    agent_timeout_sec=120.0,
    local_fixture=True,
)
