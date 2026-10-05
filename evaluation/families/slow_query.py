"""slow-query: which query, among those that cannot use an index, spends the most total time
in the query log? (the full-scan query behind the statement timeouts)

Construction: `/app/db/queries.sql` lists single-table queries `SELECT ... FROM <table>
WHERE <column> = ? ...`, `/app/db/table_stats.json` gives each table's row count, columns
and indexes, and the query log (`/app/db/query_log.csv`, plus a gzipped rotated log on hard)
has one row per execution: `ts,query_id,duration_ms,rows_scanned`. A query can use an index
when an index on its own table has its WHERE column first; otherwise it scans the whole
table (its rows_scanned is the table's row count). The target T is a full-scan query on a
big table. Planted decoys, each beating T under one wrong method:

- F, an indexed lookup with by far the most executions (most frequent query);
- H, an indexed query on a low-selectivity column with more total time than T (ignoring
  whether the query can use an index);
- S, a rare full scan with the single slowest execution (slowest occurrence instead of the
  total);
- medium/hard: T's column is indexed on another table and is the second column of a
  composite index on T's table, so treating "the column has an index somewhere" or "the
  column is in an index" as usable excludes T;
- hard: full-scan decoys D_a and D_b whose time is all in one of the two log files, each
  beating T within that file, so reading only one file fails.

Every solution is one Python source (`SOLVER`) run with a mode, as in csv-revenue.
"""

import gzip
import json
from datetime import datetime, timedelta

from learning_loop.tasks.spec import ExactAnswer, Family, gzip_bytes, Reject, Solution, TaskSpec

TABLES = {
    "orders": ["id", "customer_id", "status", "created_at", "total_cents", "region", "coupon_code"],
    "customers": ["id", "email", "country", "created_at", "plan", "referrer"],
    "events": ["id", "user_id", "kind", "created_at", "session_id", "payload"],
    "invoices": ["id", "customer_id", "order_id", "status", "due_date", "amount_cents"],
    "sessions": ["id", "user_id", "token", "ip_address", "created_at", "user_agent"],
    "products": ["id", "sku", "category", "price_cents", "supplier_id", "name"],
    "shipments": ["id", "order_id", "carrier", "tracking_code", "status", "shipped_at"],
    "payments": ["id", "invoice_id", "customer_id", "method", "status", "processor_ref"],
    "audit_log": ["id", "user_id", "action", "created_at", "object_id", "ip_address"],
    "users": ["id", "email", "team_id", "role", "created_at", "last_login"],
}
TAILS = ["", "", " ORDER BY id DESC LIMIT 50", " LIMIT 100", " ORDER BY id LIMIT 20"]

INSTRUCTION = """Since this morning our API has been hitting database statement timeouts. We think one query that cannot use an index is scanning whole tables and hogging the database.

- `/app/db/queries.sql`: every query the application runs, each preceded by a `-- query: <id>` line.
- `/app/db/table_stats.json`: each table's row count, columns and indexes (an index's `columns` are in index order).
- {logs}: one CSV row per query execution during the incident (`ts,query_id,duration_ms,rows_scanned`).

A query can use an index only if an index **on the query's own table** has the query's `WHERE` column as its **first** column; otherwise it scans the whole table. Among the queries that cannot use an index, find the one with the largest **total** time, i.e. the sum of `duration_ms` over all of its executions in {which}.

Write just that query's id (as written in `queries.sql`, nothing else) to `/app/answer.txt`.
"""

SOLVER = r'''
import csv
import io
import re
from collections import defaultdict

QUERY = re.compile(r"--\s*query:\s*(\S+)\s*\n\s*SELECT .*? FROM (\w+) WHERE (\w+) = \?", re.S)


def parse_queries(sql):
    """{query id: (table, where column)}."""
    return {qid: (table, col) for qid, table, col in QUERY.findall(sql)}


def uses_index(stats, table, col, mode):
    if mode == "any-table":
        return any(col in ix["columns"] for t in stats.values() for ix in t["indexes"])
    if mode == "any-position":
        return any(col in ix["columns"] for ix in stats[table]["indexes"])
    return any(ix["columns"][0] == col for ix in stats[table]["indexes"])


def answer(sql, stats, logs, mode):
    """logs: [(file name, csv text)] sorted by name. Returns the query id, or None."""
    queries = parse_queries(sql)
    if mode == "plain-only":
        logs = [(n, t) for n, t in logs if not n.endswith(".gz")]
    total, count, slowest = defaultdict(int), defaultdict(int), {}
    for _, text in logs:
        for r in csv.DictReader(io.StringIO(text)):
            q, d = r["query_id"], int(r["duration_ms"])
            total[q] += d
            count[q] += 1
            slowest[q] = max(slowest.get(q, 0), d)
    if mode == "most-frequent":
        score = count
    elif mode == "slowest-single":
        score = slowest
    else:
        score = total
    if mode in ("most-frequent", "slowest-single", "ignore-index"):
        candidates = list(score)
    else:
        candidates = [q for q in score if q in queries and not uses_index(stats, *queries[q], mode)]
    if not candidates:
        return None
    return sorted(candidates, key=lambda q: (-score[q], q))[0]
'''

_SHELL = """python3 - <<'PY'
{solver}
import glob
import gzip
import json
import os

logs = []
for p in sorted(glob.glob("/app/db/query_log*")):
    data = open(p, "rb").read()
    logs.append((os.path.basename(p), (gzip.decompress(data) if p.endswith(".gz") else data).decode()))
qid = answer(open("/app/db/queries.sql").read(), json.load(open("/app/db/table_stats.json")), logs, {mode!r})
if qid is not None:
    with open("/app/answer.txt", "w") as f:
        f.write(qid + "\\n")
PY
"""

_NS: dict = {}
exec(SOLVER, _NS)  # noqa: S102 - our own source; the model form of every solution


def _logs(files):
    out = []
    for rel, data in sorted(files.items()):
        if rel.startswith("db/query_log"):
            out.append((rel.split("/")[-1], (gzip.decompress(data) if rel.endswith(".gz") else data).decode()))
    return out


def _solution(mode):
    def model(files):
        qid = _NS["answer"](files["db/queries.sql"].decode(), json.loads(files["db/table_stats.json"]), _logs(files), mode)
        return {} if qid is None else {"/app/answer.txt": qid + "\n"}

    return Solution(_SHELL.format(solver=SOLVER, mode=mode), model)


class _Plan:
    """Queries, table stats and executions as they are drawn."""

    def __init__(self, rng, n_files):
        self.rng = rng
        self.queries = []  # (table, column, select columns, tail)
        self.runs = [[] for _ in range(n_files)]  # per log file: [(query index, duration, rows)]

    def add_query(self, table, col):
        cols = [c for c in TABLES[table] if c != col]
        sel = ", ".join(self.rng.sample(cols, self.rng.randint(1, 3)))
        self.queries.append((table, col, sel, self.rng.choice(TAILS)))
        return len(self.queries) - 1

    def run(self, q, f, n, dur, rows):
        for _ in range(n):
            self.runs[f].append((q, dur(), rows()))


def build(ctx):
    rng, p = ctx.rng, ctx.params
    n_files = p["files"]
    tables = rng.sample(sorted(TABLES), p["tables"])
    big, small = tables[:3], tables[3:]
    rows = {t: rng.randint(2_000_000, 9_000_000) if t in big else rng.randint(3_000, 60_000) for t in tables}
    # Indexes: the primary key, plus single-column indexes on some columns.
    indexes = {t: [["id"]] for t in tables}
    plan = _Plan(rng, n_files)
    used = set()  # (table, column) pairs a query already filters on

    def pick(table, indexed, avoid=(), only=None):
        cols = [c for c in TABLES[table][1:] if (table, c) not in used and c not in avoid and (only is None or c in only)]
        if not indexed:  # a full scan must not filter on a column that leads one of the table's indexes
            cols = [c for c in cols if not any(ix[0] == c for ix in indexes[table])]
        if not cols:
            raise Reject("no free column")
        col = rng.choice(cols)
        used.add((table, col))
        if indexed:
            indexes[table].append([col])
        return col

    def split(n):
        """n executions spread over the log files (each gets at least one)."""
        cuts = sorted(rng.sample(range(1, n), n_files - 1)) if n_files > 1 else []
        return [b - a for a, b in zip([0] + cuts, cuts + [n])]

    def full_ms(table):
        return lambda: int(rows[table] / rng.uniform(800, 1200))

    # T: a full scan of a big table.
    t_table = big[0]
    shared = {c for c in TABLES[t_table] if sum(c in cols for cols in TABLES.values()) > 1}
    t_col = pick(t_table, False, only=shared - {"id"} if p["index_traps"] else None)
    t = plan.add_query(t_table, t_col)
    t_n = rng.randint(28, 40)
    if n_files == 1:
        t_split = [t_n]
    else:  # about half in each file, so a decoy can win one file but not both
        first = t_n // 2 + rng.randint(-3, 3)
        t_split = [first, t_n - first]
    for f, n in enumerate(t_split):
        plan.run(t, f, n, full_ms(t_table), lambda: rows[t_table])
    if p["index_traps"]:
        # T's column is the second column of a composite index on its own table, and is
        # indexed (first) on another table that has the column.
        other_col = rng.choice([c for c in TABLES[t_table][1:] if c != t_col])
        indexes[t_table].append([other_col, t_col])
        hosts = [x for x in tables if x != t_table and t_col in TABLES[x]]
        if not hosts:
            extra = [x for x in sorted(TABLES) if x not in tables and t_col in TABLES[x]]
            if not extra:
                raise Reject(f"no other table has column {t_col}")
            tables.append(rng.choice(extra))
            rows[tables[-1]] = rng.randint(3_000, 60_000)
            indexes[tables[-1]] = [["id"]]
            hosts = [tables[-1]]
        host = rng.choice(hosts)
        indexes[host].append([t_col])
        used.add((host, t_col))
        q = plan.add_query(host, t_col)
        for f, n in enumerate(split(rng.randint(40, 80))):
            plan.run(q, f, n, lambda: rng.randint(1, 9), lambda: rng.randint(1, 30))
    t_total_est = t_n * rows[t_table] // 1000

    # H: an indexed but low-selectivity query with more total time than T.
    h_table = rng.choice(big[1:])
    h = plan.add_query(h_table, pick(h_table, True))
    h_n = rng.randint(90, 140)
    h_dur = int(t_total_est * rng.uniform(1.4, 1.8) / h_n)
    for f, n in enumerate(split(h_n)):
        plan.run(h, f, n, lambda: rng.randint(int(h_dur * 0.8), int(h_dur * 1.2)), lambda: rng.randint(40_000, 90_000))
    # F: a cheap indexed lookup run far more often than anything else.
    f_table = rng.choice(tables)
    fq = plan.add_query(f_table, pick(f_table, True))
    for f, n in enumerate(split(rng.randint(260, 340))):
        plan.run(fq, f, n, lambda: rng.randint(1, 6), lambda: rng.randint(1, 3))
    # S: a rare full scan of a big table, with the single slowest execution.
    s_table = rng.choice(big)
    s = plan.add_query(s_table, pick(s_table, False, avoid=(t_col,)))
    plan.runs[rng.randrange(n_files)].append((s, int(t_total_est * rng.uniform(0.45, 0.6)), rows[s_table]))
    plan.runs[rng.randrange(n_files)].append((s, rng.randint(1500, 4000), rows[s_table]))
    if n_files > 1:
        # D_a, D_b: full scans whose executions are all in one file, each beating T there.
        for f in range(n_files):
            d_table = rng.choice(big)
            d = plan.add_query(d_table, pick(d_table, False, avoid=(t_col,)))
            t_here = sum(dur for q, dur, _ in plan.runs[f] if q == t)
            per = int(rows[d_table] / 1000)
            n = t_here // per + rng.randint(1, 3)
            plan.run(d, f, n, lambda: per + rng.randint(-100, 100), lambda: rows[d_table])
    # Background: other indexed lookups and small full scans.
    for _ in range(p["background"]):
        table = rng.choice(tables)
        indexed = rng.random() < 0.6 or table in big
        q = plan.add_query(table, pick(table, indexed))
        for f, n in enumerate(split(rng.randint(8, 60))):
            if indexed:
                plan.run(q, f, n, lambda: rng.randint(1, 25), lambda: rng.randint(1, 200))
            else:
                plan.run(q, f, n, lambda: max(1, int(rows[table] / 1000)) + rng.randint(0, 9), lambda: rows[table])

    # Ids are assigned in a shuffled order, so T's id carries no hint.
    order = list(range(len(plan.queries)))
    rng.shuffle(order)
    ids = {qi: f"q{i + 1:02d}" for i, qi in enumerate(order)}
    sql = []
    for qi in sorted(order, key=lambda qi: ids[qi]):
        table, col, sel, tail = plan.queries[qi]
        sql.append(f"-- query: {ids[qi]}\nSELECT {sel} FROM {table} WHERE {col} = ?{tail};\n")
    stats = {}
    for table in sorted(tables):
        ix = []
        for cols in indexes[table]:
            name = f"{table}_pkey" if cols == ["id"] else f"idx_{table}_{'_'.join(cols)}"
            ix.append({"name": name, "columns": cols})
        stats[table] = {"rows": rows[table], "columns": TABLES[table], "indexes": ix}
    names = ["query_log.csv"] + [f"query_log.{i}.csv.gz" for i in range(1, n_files)]
    files = {"db/queries.sql": "\n".join(sql), "db/table_stats.json": json.dumps(stats, indent=2) + "\n"}
    start = datetime(2026, 9, 1) + timedelta(days=rng.randint(0, 25), hours=8)
    for f, name in enumerate(names):
        runs = plan.runs[f]
        rng.shuffle(runs)
        span = start + timedelta(hours=n_files - 1 - f)  # the rotated (older) file covers the earlier hour
        lines = ["ts,query_id,duration_ms,rows_scanned"]
        stamps = sorted(rng.randint(0, 3599_000) for _ in runs)
        for (q, dur, nrows), ms in zip(runs, stamps):
            ts = span + timedelta(milliseconds=ms)
            lines.append(f"{ts.strftime('%Y-%m-%dT%H:%M:%S.')}{ts.microsecond // 1000:03d}Z,{ids[q]},{dur},{nrows}")
        text = "\n".join(lines) + "\n"
        files[f"db/{name}"] = gzip_bytes(text.encode()) if name.endswith(".gz") else text

    # The answer must be a unique maximum, and each decoy must win under its wrong method.
    target = ids[t]
    fbytes = {k: v.encode() if isinstance(v, str) else v for k, v in files.items()}
    totals = {}
    for _, text in _logs(fbytes):
        for line in text.splitlines()[1:]:
            _, q, d, _ = line.split(",")
            totals[q] = totals.get(q, 0) + int(d)
    queries = _NS["parse_queries"](files["db/queries.sql"])
    full = sorted((v, q) for q, v in totals.items() if not _NS["uses_index"](stats, *queries[q], "oracle"))
    if full[-1][1] != target or (len(full) > 1 and full[-2][0] == full[-1][0]):
        raise Reject("the target is not the unique top full scan")
    logs_desc = "`/app/db/query_log.csv`" if n_files == 1 else "`/app/db/query_log.csv` and the older, rotated `/app/db/query_log.1.csv.gz`"
    which = "the log" if n_files == 1 else "both logs"
    shortcuts = {m: _solution(m) for m in ("most-frequent", "slowest-single", "ignore-index")}
    if p["index_traps"]:
        shortcuts["any-table"] = _solution("any-table")
        shortcuts["any-position"] = _solution("any-position")
    if n_files > 1:
        shortcuts["plain-only"] = _solution("plain-only")
    return TaskSpec(
        instruction=INSTRUCTION.format(logs=logs_desc, which=which),
        files=files,
        grader=ExactAnswer("/app/answer.txt", target),
        oracle=_solution("oracle"),
        shortcuts=shortcuts,
        params={"queries": len(plan.queries), "tables": len(tables), "files": n_files, "index_traps": p["index_traps"]},
    )


FAMILY = Family(
    name="slow-query",
    version=1,
    cluster="diagnosis",
    category="data",
    skills=("sql", "indexes", "logs", "aggregation", "diagnosis"),
    difficulties={
        "easy": {"tables": 5, "files": 1, "index_traps": False, "background": 3},
        "medium": {"tables": 6, "files": 1, "index_traps": True, "background": 5},
        "hard": {"tables": 8, "files": 2, "index_traps": True, "background": 7},
    },
    build=build,
)
