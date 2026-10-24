"""FIXTURE family for the task-framework tests (not an evaluation task).

Fix /app/upper.py so it prints its input file upper-cased. Exercises the commands grader
(agent code runs as `nobody` in the verifier, on hidden inputs) and a shortcut that
hard-codes the visible example.
"""

from learning_loop.tasks.spec import Commands, Family, Solution, TaskSpec

BUGGY = "import sys\n\nprint(open(sys.argv[1]).read().lower(), end='')\n"
FIXED = "import sys\n\nprint(open(sys.argv[1]).read().upper(), end='')\n"
INSTRUCTION = "`python3 /app/upper.py FILE` should print FILE's contents in upper case, but it prints lower case. Example: `/app/example.txt`. Fix `upper.py`.\n"


def _write(src):
    return f"cat > /app/upper.py <<'PY'\n{src}PY\n"


def build(ctx):
    words = ["alpha", "Bravo", "charlie", "Delta", "echo", "foxtrot"]
    example = " ".join(ctx.rng.sample(words, 3)) + "\n"
    hardcoded = f"print({example.upper()!r}, end='')\n"
    checks = []
    for i in range(3):
        text = " ".join(ctx.rng.sample(words, 4)) + "\n"
        checks.append({"name": f"case{i}", "argv": ["python3", "upper.py", "in.txt"], "inputs": {"in.txt": text}, "stdout": text.upper(), "exit": 0})
    return TaskSpec(
        instruction=INSTRUCTION,
        files={"upper.py": BUGGY, "example.txt": example},
        grader=Commands(("/app/upper.py",), tuple(checks)),
        oracle=Solution(_write(FIXED), lambda f: {"/app/upper.py": FIXED}),
        shortcuts={"hardcode-example": Solution(_write(hardcoded), lambda f, s=hardcoded: {"/app/upper.py": s})},
    )


FAMILY = Family(name="upper-tool", version=1, cluster="fixture", skills=("python",), difficulties={"easy": {}}, build=build, category="fixture", agent_timeout_sec=60.0)
