"""FIXTURE family for the task-framework tests (not an evaluation task).

Copy every file in /app/in to /app/out with mode 0600. Exercises the tree grader with modes
(and so that modes survive Harbor's artifact transfer) and a shortcut that skips the chmod.
"""

from learning_loop.tasks.spec import Family, FileState, FileTree, Solution, TaskSpec, tree_entry

INSTRUCTION = "Copy every file in `/app/in/` into `/app/out/` (create it), and make each copy readable and writable by its owner only (mode 0600).\n"


def _copies(files, mode):
    return {f"/app/out/{rel[3:]}": FileState(data, mode) for rel, data in files.items() if rel.startswith("in/")}


def build(ctx):
    files = {f"in/f{i}.txt": f"{ctx.rng.randint(0, 10**6)}\n" for i in range(ctx.params["n"])}
    return TaskSpec(
        instruction=INSTRUCTION,
        files=files,
        grader=FileTree("/app/out", {rel[3:]: tree_entry(text, 0o600) for rel, text in files.items()}),
        oracle=Solution("mkdir -p /app/out && cp /app/in/* /app/out/ && chmod 600 /app/out/*\n", lambda f: _copies(f, 0o600)),
        shortcuts={"no-chmod": Solution("mkdir -p /app/out && cp /app/in/* /app/out/\n", lambda f: _copies(f, 0o644))},
    )


FAMILY = Family(name="copy-private", version=1, cluster="fixture", skills=("shell",), difficulties={"easy": {"n": 3}}, build=build, category="fixture", agent_timeout_sec=60.0)
