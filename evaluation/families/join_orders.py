"""join-orders: total the quantity each customer ordered, joining the order CSVs to the
customer master by a customer ID that the order system formats inconsistently.

Construction: `customers.csv` holds canonical IDs (`AB-00417`); the monthly files in
`orders/` refer to customers with IDs that differ in letter case and surrounding whitespace,
and on medium/hard also in the number of leading zeros. The answer is a JSON object over
every customer in customers.csv (a left join: customers without orders get 0, orders of
unknown customers are left out).

Traps, all planted in every instance: for each normalization in play (trim, case, leading
zeros) at least one order differs from its customer's ID *only* in that respect, so a join
that skips any one normalization loses that order; at least two customers have no orders
(an inner join drops them); at least two orders belong to IDs absent from customers.csv (an
outer join adds them). Medium/hard spread the orders over several files (reading one file
loses the rest); hard writes one file with its columns in another order (reusing the first
file's column positions breaks) and states the matching rule without an example.

Every solution, right or wrong, is one Python source (`SOLVER`) run with a mode: the shell
form runs it on /app/data, the model runs the same source on the generated files, so the two
forms cannot drift apart.
"""

import csv
import io
import json

from learning_loop.tasks.spec import Family, ParsedAnswer, Solution, TaskSpec

PREFIXES = ["AB", "KL", "MX", "RT", "ZN", "DE", "QP", "HV", "JC", "FW"]  # never "XY", used in the instruction's example
NAMES = ["Ada", "Bruno", "Chen", "Dara", "Elif", "Femi", "Gita", "Hugo", "Ines", "Jonas", "Kofi", "Lena", "Mei", "Nils", "Omar", "Pia", "Quinn", "Rosa", "Sven", "Tara", "Umar", "Vera", "Wen", "Yara"]
COUNTRIES = ["DE", "FR", "NL", "SE", "PL", "ES", "IT"]
ORDER_COLUMNS = ["order_id", "placed", "customer_id", "quantity"]
REORDERED = ["customer_id", "quantity", "order_id", "placed"]
MONTHS = ["2026-07", "2026-08", "2026-09"]

INSTRUCTION = """Customer master data is in `/app/data/customers.csv` (columns `customer_id`, `name`, `country`). Orders are in `/app/data/orders/`, one CSV file per month, each with a header row and the columns `order_id`, `placed`, `customer_id` and `quantity`.

The order system does not store customer IDs as carefully as the master data. {rule}

For **every** customer in `customers.csv`, compute the total `quantity` they ordered across all order files. Customers without any orders must appear with `0`. Orders whose customer is not in `customers.csv` are left out.

Write the result to `/app/answer.json` as a single JSON object that maps each `customer_id`, written exactly as in `customers.csv`, to that total as an integer, for example `{{"XY-00417": 12, "XY-00009": 0}}`.
"""

RULES = {
    "easy": "An ID in an order can differ from the one in `customers.csv` in letter case and in surrounding whitespace, so ` xy-00417` and `Xy-00417 ` are both customer `XY-00417`.",
    "medium": "An ID in an order can differ from the one in `customers.csv` in letter case, in surrounding whitespace and in the number of leading zeros in its number, so ` xy-417`, `XY-000417` and `xY-0417 ` are all customer `XY-00417`.",
    "hard": "Treat two customer IDs as the same customer when they are equal after trimming surrounding whitespace, ignoring letter case and ignoring leading zeros in the numeric part after the dash.",
}

SOLVER = '''
import csv
import io

# Which normalizations each mode applies: S = trim whitespace, C = ignore case, Z = ignore leading zeros.
FLAGS = {"exact-join": "", "no-trim": "CZ", "no-case": "SZ", "no-zeros": "SC"}


def canon(raw, flags):
    s = raw
    if "S" in flags:
        s = s.strip()
    if "C" in flags:
        s = s.upper()
    if "Z" in flags:
        prefix, sep, num = s.partition("-")
        if sep and num.isdigit():
            s = prefix + "-" + str(int(num))
    return s


def _rows(texts, mode):
    first = next(csv.reader(io.StringIO(texts[0][1]))) if texts else []
    for _, text in texts:
        if mode == "fixed-columns":
            cols = first
            for parts in list(csv.reader(io.StringIO(text)))[1:]:
                yield dict(zip(cols, parts))
        else:
            yield from csv.DictReader(io.StringIO(text))


def answer(customers_text, order_texts, mode):
    """order_texts: [(file name, text)] sorted by name. Returns {customer_id: total}, or None if it fails."""
    flags = FLAGS.get(mode, "SCZ")
    if mode == "first-file":
        order_texts = order_texts[:1]
    master = {}
    for r in csv.DictReader(io.StringIO(customers_text)):
        master[canon(r["customer_id"], flags)] = r["customer_id"]
    totals = {} if mode == "inner-join" else {cid: 0 for cid in master.values()}
    try:
        for r in _rows(order_texts, mode):
            key = canon(r["customer_id"], flags)
            qty = int(r["quantity"])
            if key in master:
                totals[master[key]] = totals.get(master[key], 0) + qty
            elif mode == "outer-join":
                totals[key] = totals.get(key, 0) + qty
    except (ValueError, KeyError):
        return None
    return totals
'''

_SHELL = """python3 - <<'PY'
{solver}
import glob
import json
import os

customers = open("/app/data/customers.csv", newline="").read()
orders = [(os.path.basename(p), open(p, newline="").read()) for p in sorted(glob.glob("/app/data/orders/*.csv"))]
result = answer(customers, orders, {mode!r})
if result is not None:
    with open("/app/answer.json", "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\\n")
PY
"""

_NS: dict = {}
exec(SOLVER, _NS)  # noqa: S102 - our own source; the model form of every solution


def _solution(mode):
    def model(files):
        orders = [(rel.split("/")[-1], data.decode()) for rel, data in sorted(files.items()) if rel.startswith("data/orders/") and rel.endswith(".csv")]
        result = _NS["answer"](files["data/customers.csv"].decode(), orders, mode)
        return {} if result is None else {"/app/answer.json": json.dumps(result, indent=2, sort_keys=True) + "\n"}

    return Solution(_SHELL.format(solver=SOLVER, mode=mode), model)


def _variant(rng, prefix, num, need):
    """An order's spelling of customer (prefix, num) that differs from the canonical ID in exactly the respects in `need`."""
    p = prefix
    if "C" in need:
        p = rng.choice([prefix.lower(), prefix[0] + prefix[1].lower(), prefix[0].lower() + prefix[1]])
    n = f"{num:05d}"
    if "Z" in need:
        n = rng.choice([str(num), f"{num:04d}", "0" + n, "00" + n])
        if n == f"{num:05d}":
            n = str(num)  # num < 10000, so this always differs
    s = f"{p}-{n}"
    if "S" in need:
        s = rng.choice([" " + s, s + " ", "  " + s, " " + s + " "])
    return s


def _render(rows, columns):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({c: r[c] for c in columns})
    return buf.getvalue()


def build(ctx):
    rng, p = ctx.rng, ctx.params
    flags = p["flags"]
    nums = rng.sample(range(1, 10000), p["n_customers"] + 3)
    ids = []
    for num in nums[: p["n_customers"]]:
        ids.append((rng.choice(PREFIXES), num))
    orphan_ids = [(rng.choice(PREFIXES), num) for num in nums[p["n_customers"] :]]
    idle = set(rng.sample(range(len(ids)), rng.randint(2, 3)))
    active = [i for i in range(len(ids)) if i not in idle]

    # (customer index or None for an orphan, set of normalizations its spelling needs)
    plan = [(rng.choice(active), {f}) for f in flags]  # one order per normalization that only it fixes
    for i in active:
        for _ in range(rng.randint(1, 3)):
            k = rng.randint(0, len(flags))
            plan.append((i, set(rng.sample(flags, k))))
    for j in range(len(orphan_ids)):
        plan.append((("orphan", j), set(rng.sample(flags, rng.randint(0, len(flags))))))
    rng.shuffle(plan)

    files = {}
    names = rng.sample(NAMES, len(ids))
    files["data/customers.csv"] = _render(
        [{"customer_id": f"{pre}-{num:05d}", "name": names[k], "country": rng.choice(COUNTRIES)} for k, (pre, num) in enumerate(ids)],
        ["customer_id", "name", "country"],
    )
    months = MONTHS[-p["n_files"] :]
    by_month: dict[str, list] = {m: [] for m in months}
    for k, (who, need) in enumerate(plan):
        month = months[k % len(months)] if k < len(months) else rng.choice(months)
        pre, num = orphan_ids[who[1]] if isinstance(who, tuple) else ids[who]
        by_month[month].append({"placed": f"{month}-{rng.randint(1, 28):02d}", "customer_id": _variant(rng, pre, num, need), "quantity": str(rng.randint(1, 9))})
    oid = rng.randint(10000, 60000)
    for i, m in enumerate(months):
        rows = sorted(by_month[m], key=lambda r: r["placed"])  # stable: same-day orders keep their shuffled order
        for r in rows:
            oid += rng.randint(1, 7)
            r["order_id"] = f"O-{oid}"
        cols = REORDERED if p["reorder"] and i == len(months) - 1 else ORDER_COLUMNS
        files[f"data/orders/{m}.csv"] = _render(rows, cols)

    expected = _NS["answer"](files["data/customers.csv"], [(f"{m}.csv", files[f"data/orders/{m}.csv"]) for m in months], "oracle")
    shortcuts = {m: _solution(m) for m in ("exact-join", "no-trim", "no-case", "inner-join", "outer-join")}
    if "Z" in flags:
        shortcuts["no-zeros"] = _solution("no-zeros")
    if p["n_files"] > 1:
        shortcuts["first-file"] = _solution("first-file")
    if p["reorder"]:
        shortcuts["fixed-columns"] = _solution("fixed-columns")
    return TaskSpec(
        instruction=INSTRUCTION.format(rule=RULES[ctx.difficulty]),
        files=files,
        grader=ParsedAnswer("/app/answer.json", "json", expected),
        oracle=_solution("oracle"),
        shortcuts=shortcuts,
        params={"n_customers": p["n_customers"], "n_files": p["n_files"], "flags": p["flags"], "reorder": p["reorder"]},
    )


FAMILY = Family(
    name="join-orders",
    version=1,
    cluster="tabular-data",
    category="data",
    skills=("csv", "join", "normalization", "python"),
    difficulties={
        "easy": {"n_customers": 10, "n_files": 1, "flags": "SC", "reorder": False},
        "medium": {"n_customers": 14, "n_files": 2, "flags": "SCZ", "reorder": False},
        "hard": {"n_customers": 18, "n_files": 3, "flags": "SCZ", "reorder": True},
    },
    build=build,
)
