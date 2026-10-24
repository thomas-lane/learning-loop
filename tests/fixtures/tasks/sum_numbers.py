"""FIXTURE family for the task-framework tests (not an evaluation task).

Sum the numbers in the `.txt` files under /app/data; a `.csv` decoy must be ignored.
Exercises the exact-answer grader, a shortcut (summing every file) and the local fixture
backend.
"""

from learning_loop.tasks.spec import ExactAnswer, Family, Solution, TaskSpec

INSTRUCTION = (
    "Add up the numbers (one per line) in all `.txt` files under `/app/data/`.\n\n"
    "Write just the sum to `/app/answer.txt`.\n"
)


def _sum(files, suffixes):
    return sum(int(x) for rel, data in files.items() if rel.startswith("data/") and rel.endswith(suffixes) for x in data.decode().split())


def build(ctx):
    files = {}
    for i in range(ctx.params["files"]):
        files[f"data/part{i}.txt"] = "".join(f"{ctx.rng.randint(1, 99)}\n" for _ in range(ctx.rng.randint(3, 8)))
    files["data/decoy.csv"] = "".join(f"{ctx.rng.randint(0, 9)}\n" for _ in range(4))
    truth = _sum({k: v.encode() for k, v in files.items()}, (".txt",))
    return TaskSpec(
        instruction=INSTRUCTION,
        files=files,
        grader=ExactAnswer("/app/answer.txt", str(truth)),
        oracle=Solution(
            shell="cat /app/data/*.txt | awk '{ s += $1 } END { print s }' > /app/answer.txt\n",
            model=lambda f: {"/app/answer.txt": f"{_sum(f, ('.txt',))}\n"},
        ),
        shortcuts={
            "all-files": Solution(
                shell="cat /app/data/* | awk '{ s += $1 } END { print s }' > /app/answer.txt\n",
                model=lambda f: {"/app/answer.txt": f"{_sum(f, ('.txt', '.csv'))}\n"},
            ),
        },
        params={"files": ctx.params["files"]},
    )


FAMILY = Family(
    name="sum-numbers",
    version=1,
    cluster="fixture",
    skills=("shell",),
    difficulties={"easy": {"files": 2}, "hard": {"files": 4}},
    build=build,
    category="fixture",
    agent_timeout_sec=60.0,
    local_fixture=True,
)
