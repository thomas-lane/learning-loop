"""csv-revenue: which region has the highest revenue (quantity x unit_price) over completed
orders, across all CSV files in /app/data?

Construction: target T gets a_f large completed orders in file f; for each file g a decoy
D_g gets a_f + 1 in every other file (it wins any subset missing g and loses overall since
a_g > n - 1). A refund decoy wins if the status filter is ignored; a volume decoy wins on
quantity or row count but not on revenue. Medium/hard quote product names that contain
commas, so splitting lines on "," misreads them; hard reorders one file's columns, so
reusing the first file's column positions misreads it. Most of the target's product names
contain a comma, so a naive split loses most of the target's revenue. `build` redraws until the answer is
unique and every proper subset of the files gives a different one.

Every solution, right or wrong, is one Python source (`SOLVER`) run with a mode: the shell
form runs it on /app/data, the model runs the same source on the generated files, so the
two forms cannot drift apart.
"""

import csv
import io
import itertools

from learning_loop.tasks.spec import ExactAnswer, Family, Reject, Solution, TaskSpec

REGIONS = ["north", "south", "east", "west", "central", "coastal", "highland", "valley"]
PRODUCTS = ["widget", "gadget", "sprocket", "gizmo", "doohickey", "flange", "bracket"]
COLUMNS = ["order_id", "region", "product", "quantity", "unit_price", "status"]
STATUSES = ["completed", "refunded", "cancelled", "pending"]
# With quoted commas, most of the target's product names contain one, so splitting lines on
# "," misreads most of the target's rows and loses its revenue (other regions: 35%).
COMMA_RATE = 0.35
TARGET_COMMA_RATE = 0.8

INSTRUCTION = """Sales exports from several stores are in `/app/data/` as CSV files (each with a header row).

Revenue for an order is `quantity * unit_price`. Considering only orders whose `status` is `completed`, find the `region` with the highest total revenue across **all** CSV files in `/app/data/`.

Write just that region name (nothing else) to `/app/answer.txt`.
"""

SOLVER = '''
import csv
import io
from collections import defaultdict


def _rows(texts, mode):
    if mode in ("naive-split", "fixed-columns"):
        first = next(csv.reader(io.StringIO(texts[0][1])))
        for _, text in texts:
            cols = first if mode == "fixed-columns" else next(csv.reader(io.StringIO(text)))
            for ln in text.splitlines()[1:]:
                parts = ln.split(",")
                yield {c: parts[i] if i < len(parts) else "" for i, c in enumerate(cols)}
    else:
        for _, text in texts:
            yield from csv.DictReader(io.StringIO(text))


def answer(texts, mode):
    """texts: [(file name, text)] sorted by name. Returns the region, or None if it fails."""
    totals = defaultdict(float)
    try:
        for r in _rows(texts, mode):
            if mode != "no-status-filter" and r["status"] != "completed":
                continue
            if mode == "quantity-sum":
                totals[r["region"]] += int(r["quantity"])
            elif mode == "row-count":
                totals[r["region"]] += 1
            else:
                totals[r["region"]] += int(r["quantity"]) * float(r["unit_price"])
    except (ValueError, KeyError):
        return None
    if not totals:
        return None
    return sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
'''

_SHELL = """python3 - <<'PY'
{solver}
import glob
import os

texts = [(os.path.basename(p), open(p, newline="").read()) for p in sorted(glob.glob("/app/data/*.csv"))]
region = answer(texts, {mode!r})
if region is not None:
    with open("/app/answer.txt", "w") as f:
        f.write(region + "\\n")
PY
"""

_NS: dict = {}
exec(SOLVER, _NS)  # noqa: S102 - our own source; the model form of every solution


def _texts(files):
    return [(rel.split("/")[-1], data.decode()) for rel, data in sorted(files.items()) if rel.startswith("data/") and rel.endswith(".csv")]


def _solution(mode):
    def model(files):
        region = _NS["answer"](_texts(files), mode)
        return {} if region is None else {"/app/answer.txt": region + "\n"}

    return Solution(_SHELL.format(solver=SOLVER, mode=mode), model)


def _totals(texts, only=None):
    """Exact revenue per region (cents) over completed orders, to test uniqueness without float error."""
    t = {}
    for name, text in texts:
        if only is not None and name not in only:
            continue
        for r in csv.DictReader(io.StringIO(text)):
            if r["status"] == "completed":
                t[r["region"]] = t.get(r["region"], 0) + int(r["quantity"]) * round(float(r["unit_price"]) * 100)
    return t


def _unique_top(totals):
    ranked = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[0][0] if ranked and (len(ranked) == 1 or ranked[0][1] != ranked[1][1]) else None


def _render(rows, columns):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({c: r[c] for c in columns})
    return buf.getvalue()


def _order(rng, oid, region, qty, price, status, comma_rate):
    prod = rng.choice(PRODUCTS)
    if rng.random() < comma_rate:
        prod = f"{prod}, {rng.choice(['large', 'small', 'blue', 'refurbished'])}"
    return {"order_id": str(oid), "region": region, "product": prod, "quantity": str(qty), "unit_price": f"{price:.2f}", "status": status}


def _build_files(rng, p):
    n = p["n_files"]
    regions = rng.sample(REGIONS, n + 3)
    target, decoys, refund_decoy, volume_decoy = regions[0], regions[1 : n + 1] if n > 1 else [], regions[n + 1], regions[n + 2]
    names = [f"orders_{i + 1}.csv" for i in range(n)]
    a = {f: rng.randint(n + 4, n + 7) for f in names}

    def big():
        return 10, rng.randint(8000, 8999) / 100  # revenue 800-900

    oid = rng.randint(1000, 5000)
    files = {}
    for i, name in enumerate(names):
        plan = [(target, *big(), "completed") for _ in range(a[name])]
        for g, d in zip(names, decoys):
            if g != name:
                plan += [(d, *big(), "completed") for _ in range(a[name] + 1)]
        plan += [(refund_decoy, *big(), rng.choice(["refunded", "cancelled"])) for _ in range(sum(a.values()) // n + 4)]
        plan += [(volume_decoy, 12, rng.randint(10, 50) / 100, "completed") for _ in range(25)]
        for _ in range(rng.randint(25, 40)):  # small background orders, any status
            plan.append((rng.choice(regions), rng.randint(1, 4), rng.randint(100, 999) / 100, rng.choices(STATUSES, weights=[70, 12, 10, 8])[0]))
        rng.shuffle(plan)
        rows = []
        for region, qty, price, status in plan:
            oid += 1
            rate = 0.0 if not p["quoted_commas"] else TARGET_COMMA_RATE if region == target else COMMA_RATE
            rows.append(_order(rng, oid, region, qty, price, status, rate))
        cols = ["order_id", "status", "unit_price", "quantity", "product", "region"] if p["reorder"] and i == n - 1 else list(COLUMNS)
        files[name] = _render(rows, cols)
    return files, target


def build(ctx):
    p = ctx.params
    files, target = _build_files(ctx.rng, p)
    texts = sorted(files.items())
    if _unique_top(_totals(texts)) != target:
        raise Reject("the target is not the unique top region")
    names = sorted(files)
    for k in range(1, len(names)):
        for subset in itertools.combinations(names, k):
            if _unique_top(_totals(texts, set(subset))) == target:
                raise Reject(f"reading only {subset} gives the right answer")
    shortcuts = {m: _solution(m) for m in ("no-status-filter", "quantity-sum", "row-count")}
    if p["quoted_commas"]:
        shortcuts["naive-split"] = _solution("naive-split")
    if p["reorder"]:
        shortcuts["fixed-columns"] = _solution("fixed-columns")
    return TaskSpec(
        instruction=INSTRUCTION,
        files={f"data/{n}": t for n, t in files.items()},
        grader=ExactAnswer("/app/answer.txt", target),
        oracle=_solution("oracle"),
        shortcuts=shortcuts,
        params={"n_files": p["n_files"], "quoted_commas": p["quoted_commas"], "reorder": p["reorder"]},
    )


FAMILY = Family(
    name="csv-revenue",
    version=3,
    cluster="tabular-data",
    category="data",
    skills=("csv", "quoting", "aggregation", "python"),
    difficulties={
        "easy": {"n_files": 1, "quoted_commas": False, "reorder": False},
        "medium": {"n_files": 2, "quoted_commas": True, "reorder": False},
        "hard": {"n_files": 3, "quoted_commas": True, "reorder": True},
    },
    build=build,
)
