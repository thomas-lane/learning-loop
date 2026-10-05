"""latency-p95: which endpoint has the highest p95 latency (nearest-rank) across access logs
in two formats?

The logs under /app/logs are nginx access logs (`access.log*`, request time in seconds in the
last field) and JSON-lines application logs (`app.jsonl*`, `duration_ms` in milliseconds).
An endpoint is the URL path without its query string.

Construction, with H a base latency in ms (every role is a different endpoint):

- target T: n_T in {40, 60} requests, so the nearest-rank p95 is the j-th largest value,
  j = n_T/20 + 1; its top j values lie in [H, H+60] (p95 = H) and sit only in nginx files,
  at least one in each; its other requests are fast;
- tail decoy L: n_L in {40, 60}, its j-th largest is H - d (second place overall), but its
  j-1 values above that are 2.5-3H. Because n_L is a multiple of 20, the 0-based index
  `int(0.95 * n)` lands one rank higher than nearest-rank, and linear interpolation
  (numpy's default) or `statistics.quantiles` (exclusive) mixes in that next value, so L
  wins under each of those percentile methods;
- mean decoy M: every request 0.6-0.9H, mostly in JSON files (wins on the mean, and when
  seconds are mixed with milliseconds unconverted, and when only JSON files are read, since
  T's JSON requests are all fast);
- max decoy X: a single 8-10H outlier (wins on the max; on medium/hard its URL carries a
  query string no other request has, so grouping by URL with the query string makes that
  one-request group the winner);
- text decoy F: exactly two nginx requests at 1.15-1.4H and many fast JSON ones (wins when
  only the nginx files are read);
- per-file decoys D_g: one request at 1.1-1.35H in every file except g and enough fast
  requests in g that the fast ones hold its p95; reading any proper subset of the files
  that misses g leaves D_g with only slow requests, so it beats T (whose values are all
  <= H+60). That covers skipping compressed or archived files.

So every declared shortcut and every proper subset of the files fails by construction;
`build` still verifies uniqueness and the subsets. Every solution, right or wrong, is one
Python source (`SOLVER`) run with a mode, so the shell and model forms cannot drift apart.
"""

import gzip
import itertools
import json
from datetime import datetime, timedelta

from learning_loop.tasks.spec import ExactAnswer, Family, gzip_bytes, Reject, Solution, TaskSpec

ENDPOINTS = [
    "/api/orders", "/api/users", "/api/search", "/api/cart", "/api/checkout", "/api/login",
    "/api/products", "/api/reports", "/api/invoices", "/api/inventory", "/api/recommendations",
    "/api/payments", "/api/shipping", "/api/reviews", "/api/notifications", "/api/profile",
]
QUERIES = ["page=2", "page=3", "limit=50", "sort=desc", "q=shoes", "q=lamp", "id=4821", "id=77", "lang=en"]
OUTLIER_QUERY = "export=full"  # only on the max decoy's outlier
QUERY_RATE = 0.3
METHODS = ["GET", "GET", "GET", "POST"]

INSTRUCTION = """Our API feels slow and we need to know which endpoint has the worst tail latency.

The request logs are in `/app/logs/`{extra}. They come in two formats:

- nginx access logs (files named `access.log*`): the **last** field of each line is the request time in **seconds**;
- JSON-lines application logs (files named `app.jsonl*`): `duration_ms` is the request time in **milliseconds**.

An endpoint is a URL path without its query string (a request for `/api/items?page=2` is a request to the endpoint `/api/items`). For each endpoint, take the latencies of all of its requests in all of the logs and compute its p95 with the **nearest-rank** method: sort the latencies in ascending order; the p95 is the value at 1-based position ceil(0.95 * n), where n is the endpoint's number of requests.

Write the path of the endpoint with the highest p95 latency (just the path, for example `/api/items`, nothing else) to `/app/answer.txt`.
"""

SOLVER = '''
import json
import statistics
from collections import defaultdict


def wanted(name, mode):
    base = name.split("/")[-1]
    if mode == "text-only":
        return base.startswith("access.log")
    if mode == "json-only":
        return base.startswith("app.jsonl")
    if mode == "plain-only":
        return not base.endswith(".gz")
    if mode == "top-level-only":
        return "/" not in name
    return True


def requests(texts, mode):
    """(url, latency) for every request line."""
    for name, text in texts:
        for line in text.splitlines():
            if not line.strip():
                continue
            if line.lstrip().startswith("{"):
                rec = json.loads(line)
                yield rec["url"], float(rec["duration_ms"])
            else:
                fields = line.split()
                seconds = float(fields[-1])
                yield fields[6], seconds if mode == "no-unit-conversion" else seconds * 1000


def score(values, mode):
    v = sorted(values)
    n = len(v)
    if mode == "mean":
        return sum(v) / n
    if mode == "max":
        return v[-1]
    if mode == "floor-index":
        return v[int(0.95 * n)]
    if mode == "linear-interpolation":
        pos = 0.95 * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        return v[lo] + (pos - lo) * (v[hi] - v[lo])
    if mode == "quantiles-exclusive":
        return statistics.quantiles(v, n=100)[94]
    return v[-(-95 * n // 100) - 1]  # nearest rank: 1-based position ceil(0.95 * n)


def answer(texts, mode):
    """texts: [(path relative to /app/logs, text)] sorted by path. The endpoint, or None."""
    groups = defaultdict(list)
    for url, latency in requests([t for t in texts if wanted(t[0], mode)], mode):
        groups[url if mode == "with-query" else url.split("?")[0]].append(latency)
    if not groups:
        return None
    return sorted(((-score(v, mode), k) for k, v in groups.items()))[0][1]
'''

_SHELL = """python3 - <<'PY'
{solver}
import glob
import gzip
import os

texts = []
for p in sorted(p for p in glob.glob("/app/logs/**/*", recursive=True) if os.path.isfile(p)):
    data = open(p, "rb").read()
    if p.endswith(".gz"):
        data = gzip.decompress(data)
    texts.append((os.path.relpath(p, "/app/logs"), data.decode()))
best = answer(texts, {mode!r})
if best is not None:
    with open("/app/answer.txt", "w") as f:
        f.write(best + "\\n")
PY
"""

_NS: dict = {}
exec(SOLVER, _NS)  # noqa: S102 - our own source; the model form of every solution


def _texts(files):
    return [
        (rel[len("logs/"):], (gzip.decompress(data) if rel.endswith(".gz") else data).decode())
        for rel, data in sorted(files.items())
        if rel.startswith("logs/")
    ]


def _solution(mode):
    def model(files):
        best = _NS["answer"](_texts(files), mode)
        return {} if best is None else {"/app/answer.txt": best + "\n"}

    return Solution(_SHELL.format(solver=SOLVER, mode=mode), model)


def _is_text(name):
    return name.split("/")[-1].startswith("access.log")


def _p95(values):
    """Exact nearest-rank p95 of integer milliseconds."""
    v = sorted(values)
    return v[-(-95 * len(v) // 100) - 1]


def _unique_top(recs, names):
    groups = {}
    for ep, ms, f, _ in recs:
        if f in names:
            groups.setdefault(ep, []).append(ms)
    ranked = sorted(((_p95(v), ep) for ep, v in groups.items()), reverse=True)
    if not ranked:
        return None
    return ranked[0][1] if len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= 5 else None


def _plan(rng, names, n_background):
    """[(endpoint, ms, file, query or None)] and the target endpoint."""
    text_files = [n for n in names if _is_text(n)]
    json_files = [n for n in names if not _is_text(n)]
    n = len(names)
    h = rng.randint(700, 1400)
    eps = rng.sample(ENDPOINTS, 5 + n + n_background)
    target, tail, mean, peak, textd = eps[:5]
    per_file, background = eps[5 : 5 + n], eps[5 + n :]
    recs = []

    def ms(lo, hi):
        return rng.randint(int(lo * h), int(hi * h))

    def add(ep, value, files, query=None):
        recs.append((ep, value, rng.choice(files) if isinstance(files, list) else files, query))

    n_t = rng.choice([40, 60])
    top = [h] + [h + rng.randint(1, 60) for _ in range(n_t // 20)]
    rng.shuffle(top)
    for i, v in enumerate(top):
        add(target, v, text_files[i] if i < len(text_files) else text_files)
    for _ in range(n_t - len(top)):
        add(target, ms(0.05, 0.3), names)

    n_l = rng.choice([40, 60])
    for _ in range(n_l // 20):
        add(tail, ms(2.5, 3.0), names)
    add(tail, h - rng.randint(8, 25), names)
    for _ in range(n_l - n_l // 20 - 1):
        add(tail, ms(0.05, 0.3), names)

    n_m = rng.randint(30, 50)
    for i in range(n_m):
        add(mean, ms(0.6, 0.9), json_files if i < 0.6 * n_m else names)

    n_x = rng.randint(40, 55)
    add(peak, ms(8.0, 10.0), names, OUTLIER_QUERY)
    for _ in range(n_x - 1):
        add(peak, ms(0.1, 0.3), names)

    for _ in range(2):
        add(textd, ms(1.15, 1.4), text_files)
    for _ in range(rng.randint(38, 55)):
        add(textd, ms(0.05, 0.3), json_files)

    for g, ep in zip(names, per_file):
        others = [f for f in names if f != g]
        for f in others:
            add(ep, ms(1.1, 1.35), f)
        for _ in range(20 * len(others) - len(others) + rng.randint(0, 8)):
            add(ep, ms(0.05, 0.3), g)

    for ep in background:
        for _ in range(rng.randint(20, 35)):
            add(ep, ms(0.05, 0.45), names)
    return recs, target


def _ip(rng):
    return f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"


def _render(rng, recs, names, with_queries):
    out = {}
    start = datetime(2026, 9, 1) + timedelta(days=rng.randint(0, 20))
    for age, name in enumerate(names):  # a rotated file (.1, archive/) is a day older
        rows = [r for r in recs if r[2] == name]
        rng.shuffle(rows)
        ts = start - timedelta(days=1 if (".1" in name or "/" in name) else 0)
        lines = []
        for ep, value, _, query in rows:
            ts += timedelta(seconds=rng.randint(1, 60))
            if query is None and with_queries and rng.random() < QUERY_RATE:
                query = rng.choice(QUERIES)
            url = ep if query is None or not with_queries else f"{ep}?{query}"
            method = rng.choice(METHODS)
            status = rng.choices([200, 201, 304, 404], weights=[85, 5, 5, 5])[0]
            if _is_text(name):
                stamp = ts.strftime("%d/%b/%Y:%H:%M:%S +0000")
                lines.append(f'{_ip(rng)} - - [{stamp}] "{method} {url} HTTP/1.1" {status} {rng.randint(120, 40_000)} "-" "curl/8.5.0" {value / 1000:.3f}')
            else:
                rec = {"ts": ts.strftime("%Y-%m-%dT%H:%M:%SZ"), "level": "info", "method": method, "url": url, "status": status, "duration_ms": value}
                lines.append(json.dumps(rec))
        text = "\n".join(lines) + "\n"
        out[f"logs/{name}"] = gzip_bytes(text.encode()) if name.endswith(".gz") else text
    return out


def build(ctx):
    p = ctx.params
    names = p["files"]
    recs, target = _plan(ctx.rng, names, p["background"])
    if _unique_top(recs, set(names)) != target:
        raise Reject("the target is not the unique top endpoint")
    for k in range(1, len(names)):
        for subset in itertools.combinations(names, k):
            if _unique_top(recs, set(subset)) == target:
                raise Reject(f"reading only {subset} gives the right answer")
    modes = ["mean", "max", "floor-index", "linear-interpolation", "quantiles-exclusive", "text-only", "json-only", "no-unit-conversion"]
    if p["queries"]:
        modes.append("with-query")
    if any(n.endswith(".gz") for n in names):
        modes.append("plain-only")
    if any("/" in n for n in names):
        modes.append("top-level-only")
    hard = any("/" in n for n in names)
    return TaskSpec(
        instruction=INSTRUCTION.format(extra=" (including rotated, gzip-compressed and archived logs in subdirectories)" if hard else ""),
        files=_render(ctx.rng, recs, names, p["queries"]),
        grader=ExactAnswer("/app/answer.txt", target),
        oracle=_solution("oracle"),
        shortcuts={m: _solution(m) for m in modes},
        params={"files": list(names), "queries": p["queries"]},
    )


FAMILY = Family(
    name="latency-p95",
    version=1,
    cluster="text-analytics",
    category="data",
    skills=("logs", "json", "percentiles", "units", "aggregation", "python"),
    difficulties={
        "easy": {"files": ["access.log", "app.jsonl"], "queries": False, "background": 3},
        "medium": {"files": ["access.log", "access.log.1", "app.jsonl"], "queries": True, "background": 3},
        "hard": {"files": ["access.log", "access.log.1.gz", "app.jsonl", "archive/app.jsonl.1.gz"], "queries": True, "background": 3},
    },
    build=build,
)
