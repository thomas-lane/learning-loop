"""FIXTURE family for the task-framework tests (not an evaluation task).

Fix one injected bug in /app/calc.py. Exercises the checks grader (agent code runs
sandboxed as `nobody` in the verifier) and a forged artifact whose return value claims
equality with everything.
"""

from learning_loop.tasks.spec import Checks, Family, Solution, TaskSpec

CORRECT = '''def add(a, b):
    """Return a + b."""
    return a + b


def scale(xs, k):
    """Return a new list with every element of xs multiplied by k. xs is not modified."""
    return [x * k for x in xs]
'''

BUGS = {
    "add": ("return a + b", "return a - b"),
    "scale": ("return [x * k for x in xs]", "for i in range(len(xs)):\n        xs[i] *= k\n    return xs"),
}

FORGED = '''class Anything(int):
    def __eq__(self, other):
        return True


def add(a, b):
    return Anything(0)


def scale(xs, k):
    return [x * k for x in xs]
'''

INSTRUCTION = "`/app/calc.py` has a bug. Fix it so both functions behave as their docstrings say.\n"


def _write(path, source):
    return f"cat > {path} <<'PY'\n{source}PY\n"


def build(ctx):
    bug = ctx.rng.choice(sorted(BUGS))
    old, new = BUGS[bug]
    a, b = ctx.rng.randint(2, 50), ctx.rng.randint(2, 50)
    checks = (
        {"name": "add", "func": "add", "kind": "value", "args": [a, b], "expected": a + b},
        {"name": "scale_no_mutation", "func": "scale", "kind": "no_mutation", "args": [[ctx.rng.randint(1, 9) for _ in range(4)], ctx.rng.randint(2, 5)]},
    )
    return TaskSpec(
        instruction=INSTRUCTION,
        files={"calc.py": CORRECT.replace(old, new)},
        grader=Checks("/app/calc.py", "calc", checks),
        oracle=Solution(shell=_write("/app/calc.py", CORRECT), model=lambda f: {"/app/calc.py": CORRECT}),
        shortcuts={"forge-eq": Solution(shell=_write("/app/calc.py", FORGED), model=lambda f: {"/app/calc.py": FORGED})},
        params={"bug": bug},
    )


FAMILY = Family(
    name="fix-add",
    version=1,
    cluster="fixture",
    skills=("python",),
    difficulties={"easy": {}},
    build=build,
    category="fixture",
    agent_timeout_sec=60.0,
)
