"""csv-revenue family (intended as an entirely held-out family).

"Across all CSV files in /app/data, which region has the highest revenue
(quantity x unit_price) over *completed* orders?"

Shortcut-resistance (checked for every instance, see `check`):
- ignoring the status filter gives a different region;
- summing quantities (or counting rows) instead of revenue gives a different region;
- medium/hard: every proper subset of the files gives a different region, and
  splitting lines naively on "," (quoted product names contain commas) gives a
  different region or fails;
- hard: one file has a different column order, so assuming the first file's
  column positions gives a different region.
"""

from __future__ import annotations

import csv
import io
import itertools
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from .common import PYTHON_BASE, rng_for, task_toml, write

FAMILY = "csv-revenue"
VERSION = 2  # v2: base images pinned by digest
RNG_VERSION = 1  # random stream; unchanged since v1, so instances keep their content
SKILLS = ["csv", "quoting", "aggregation", "python"]
DIFFICULTIES = {
    "easy": {"n_files": 1, "quoted_commas": False, "reorder": False},
    "medium": {"n_files": 2, "quoted_commas": True, "reorder": False},
    "hard": {"n_files": 3, "quoted_commas": True, "reorder": True},
}
REGIONS = ["north", "south", "east", "west", "central", "coastal", "highland", "valley"]
PRODUCTS = ["widget", "gadget", "sprocket", "gizmo", "doohickey", "flange", "bracket"]
COLUMNS = ["order_id", "region", "product", "quantity", "unit_price", "status"]
STATUSES = ["completed", "refunded", "cancelled", "pending"]

Rows = list[dict[str, str]]


def _render(rows: Rows, columns: list[str]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({c: r[c] for c in columns})
    return buf.getvalue()


def _top(totals: dict[str, float]) -> str | None:
    if not totals:
        return None
    ranked = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(ranked) > 1 and abs(ranked[0][1] - ranked[1][1]) < 1e-9:
        return "<tie>"
    return ranked[0][0]


def _agg(rows: Rows, value: Callable[[dict[str, str]], float], only_completed: bool = True) -> str | None:
    t: dict[str, float] = defaultdict(float)
    for r in rows:
        if only_completed and r["status"] != "completed":
            continue
        t[r["region"]] += value(r)
    return _top(t)


def _revenue(r: dict[str, str]) -> float:
    return int(r["quantity"]) * float(r["unit_price"])


def _naive(text: str, header_cols: list[str]) -> Rows:
    """Split on ',' ignoring quotes, using the given column positions."""
    out = []
    for ln in text.splitlines()[1:]:
        parts = ln.split(",")
        out.append({c: parts[i] if i < len(parts) else "" for i, c in enumerate(header_cols)})
    return out


def check(files: dict[str, str]) -> dict[str, Any]:
    parsed = {n: list(csv.DictReader(io.StringIO(t))) for n, t in files.items()}
    all_rows = [r for rs in parsed.values() for r in rs]
    names = sorted(files)
    first_cols = next(csv.reader(io.StringIO(files[names[0]])))
    res: dict[str, Any] = {
        "truth": _agg(all_rows, _revenue),
        "no_status_filter": _agg(all_rows, _revenue, only_completed=False),
        "quantity_sum": _agg(all_rows, lambda r: int(r["quantity"])),
        "row_count": _agg(all_rows, lambda r: 1),
        "subsets": {"+".join(c): _agg([r for n in c for r in parsed[n]], _revenue) for k in range(1, len(names)) for c in itertools.combinations(names, k)},
    }
    for label, cols_for in (("naive_split", lambda n: next(csv.reader(io.StringIO(files[n])))), ("fixed_columns", lambda n: first_cols)):
        try:
            rows = [r for n in names for r in _naive(files[n], cols_for(n))]
            res[label] = _agg(rows, _revenue)
        except (ValueError, KeyError):
            res[label] = "<error>"
    return res


def _order(rng: random.Random, oid: int, region: str, qty: int, price: float, status: str, spec: dict[str, Any]) -> dict[str, str]:
    prod = rng.choice(PRODUCTS)
    if spec["quoted_commas"] and rng.random() < 0.35:
        prod = f"{prod}, {rng.choice(['large', 'small', 'blue', 'refurbished'])}"
    return {"order_id": str(oid), "region": region, "product": prod, "quantity": str(qty), "unit_price": f"{price:.2f}", "status": status}


def _build(rng: random.Random, spec: dict[str, Any]) -> dict[str, str]:
    """Target T gets a_f large completed orders in file f; for each file g a decoy
    D_g gets a_f + 1 in every other file (wins any subset missing g, loses overall
    since a_g > n - 1). A refund decoy wins if the status filter is ignored; a
    volume decoy wins on quantity/row count but not on revenue."""
    n = spec["n_files"]
    regions = rng.sample(REGIONS, n + 3)
    target, decoys, refund_decoy, volume_decoy = regions[0], regions[1 : n + 1] if n > 1 else [], regions[n + 1], regions[n + 2]
    names = [f"orders_{i + 1}.csv" for i in range(n)]
    a = {f: rng.randint(n + 4, n + 7) for f in names}
    big = lambda: (10, rng.randint(8000, 8999) / 100)  # noqa: E731 - revenue 800-900
    oid = rng.randint(1000, 5000)
    files: dict[str, str] = {}
    for i, name in enumerate(names):
        plan: list[tuple[str, int, float, str]] = []
        plan += [(target, *big(), "completed") for _ in range(a[name])]
        for g, d in zip(names, decoys):
            if g != name:
                plan += [(d, *big(), "completed") for _ in range(a[name] + 1)]
        plan += [(refund_decoy, *big(), rng.choice(["refunded", "cancelled"])) for _ in range(sum(a.values()) // n + 4)]
        plan += [(volume_decoy, 12, rng.randint(10, 50) / 100, "completed") for _ in range(25)]
        for _ in range(rng.randint(25, 40)):  # small background orders, any status
            plan.append((rng.choice(regions), rng.randint(1, 4), rng.randint(100, 999) / 100, rng.choices(STATUSES, weights=[70, 12, 10, 8])[0]))
        rng.shuffle(plan)
        rows: Rows = []
        for region, qty, price, status in plan:
            oid += 1
            rows.append(_order(rng, oid, region, qty, price, status, spec))
        cols = list(COLUMNS)
        if spec["reorder"] and i == n - 1:
            cols = ["order_id", "status", "unit_price", "quantity", "product", "region"]
        files[name] = _render(rows, cols)
    return files


def _valid(chk: dict[str, Any], spec: dict[str, Any]) -> bool:
    t = chk["truth"]
    if t in (None, "<tie>"):
        return False
    if t in (chk["no_status_filter"], chk["quantity_sum"], chk["row_count"]):
        return False
    if any(v == t for v in chk["subsets"].values()):
        return False
    if spec["quoted_commas"] and chk["naive_split"] == t:
        return False
    if spec["reorder"] and chk["fixed_columns"] == t:
        return False
    return True


INSTRUCTION = """Sales exports from several stores are in `/app/data/` as CSV files (each with a header row).

Revenue for an order is `quantity * unit_price`. Considering only orders whose `status` is `completed`, find the `region` with the highest total revenue across **all** CSV files in `/app/data/`.

Write just that region name (nothing else) to `/app/answer.txt`.
"""

TEST_SH = """#!/bin/bash
# Hidden grader (separate verifier container; /app/answer.txt is the only transferred artifact).
EXPECTED="{expected}"
ACTUAL="$(tr -d '[:space:]' < /app/answer.txt 2>/dev/null)"
echo "expected: $EXPECTED"
echo "actual:   ${{ACTUAL:-<missing /app/answer.txt>}}"
if [ "$ACTUAL" = "$EXPECTED" ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
"""

SOLVE_SH = """#!/bin/bash
python3 - <<'PY'
import csv, glob
from collections import defaultdict
tot = defaultdict(float)
for path in sorted(glob.glob("/app/data/*.csv")):
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["status"] == "completed":
                tot[r["region"]] += int(r["quantity"]) * float(r["unit_price"])
open("/app/answer.txt", "w").write(max(tot, key=tot.get) + "\\n")
PY
"""


def generate(out_dir: Path, difficulty: str, seed: int) -> dict[str, Any]:
    spec = DIFFICULTIES[difficulty]
    rng = rng_for(FAMILY, RNG_VERSION, difficulty, seed)
    for _ in range(200):
        files = _build(rng, spec)
        chk = check(files)
        if _valid(chk, spec):
            break
    else:  # pragma: no cover
        raise RuntimeError(f"{FAMILY}: could not build a shortcut-resistant instance for seed {seed}")
    out_dir = Path(out_dir)
    for name, text in files.items():
        write(out_dir / "environment" / "data" / name, text)
    write(out_dir / "instruction.md", INSTRUCTION)
    write(out_dir / "environment" / "Dockerfile", "FROM " + PYTHON_BASE + "\n\nWORKDIR /app\nCOPY data/ /app/data/\n")
    write(out_dir / "tests" / "test.sh", TEST_SH.format(expected=chk["truth"]), executable=True)
    write(out_dir / "tests" / "Dockerfile", "FROM " + PYTHON_BASE + "\nCOPY test.sh /tests/test.sh\n")
    write(out_dir / "solution" / "solve.sh", SOLVE_SH, executable=True)
    params = dict(spec)
    write(
        out_dir / "task.toml",
        task_toml(
            family=FAMILY,
            generator=f"{FAMILY}@v{VERSION}",
            generator_seed=seed,
            difficulty=difficulty,
            category="data",
            tags=["csv", "trap:status-filter"] + (["trap:quoted-commas"] if spec["quoted_commas"] else []) + (["trap:column-order"] if spec["reorder"] else []),
            skills=SKILLS,
            artifacts=["/app/answer.txt"],
            params=params,
            comment=f"Generated by evaluation/generators ({FAMILY}@v{VERSION}, {difficulty}, seed {seed}). Do not edit by hand.",
        ),
    )
    return params
