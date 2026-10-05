"""log-triage: which client IP caused the most HTTP 5xx responses across all access logs?

Construction: target T has c_f 5xx in each file f; for every file g a decoy D_g appears in
all files except g with c_f + delta 5xx each, so any subset missing g is won by D_g, while
over all files T wins because c_g > (n-1)*delta. A 404-heavy decoy wins if every status
>= 400 counts; on medium/hard a decoy whose response *sizes* are 500-599 wins if any
` 5xx ` field counts. `build` redraws until the answer is a unique maximum and every proper
subset of the files gives a different one (that covers skipping rotated, compressed or
archived files, and `cat a b | zcat c.gz`, which reads only c.gz). The declared shortcuts
are those wrong methods as runnable shell pipelines.
"""

import gzip
import itertools
import re
from collections import Counter
from datetime import datetime, timedelta

from learning_loop.tasks.spec import ExactAnswer, Family, gzip_bytes, Reject, Solution, TaskSpec

PATHS = ["/", "/api/users", "/api/orders", "/login", "/static/app.js", "/api/search?q=x"]

INSTRUCTION = """Our web server has been throwing server errors and we need to know which client is triggering the most of them.

The nginx access logs are in `/app/logs/`. Find the client IP address responsible for the most HTTP **5xx** responses across **all** of the logs there{extra}.

Write just that IP address (nothing else) to `/app/answer.txt`.
"""

_TOP = " | sort | uniq -c | sort -rn | head -1 | awk '{ print $2 }' > /app/answer.txt\n"
_ALL = "find /app/logs -type f | sort | while read -r f; do zcat -f \"$f\"; done"
ORACLE = _ALL + " | awk '$9 >= 500 && $9 < 600 { print $1 }'" + _TOP
PLAIN_ONLY = "find /app/logs -type f ! -name '*.gz' | sort | xargs cat | awk '$9 >= 500 && $9 < 600 { print $1 }'" + _TOP
ALL_4XX = _ALL + " | awk '$9 >= 400 { print $1 }'" + _TOP
TOP_LEVEL_ONLY = "find /app/logs -maxdepth 1 -type f | sort | while read -r f; do zcat -f \"$f\"; done | awk '$9 >= 500 && $9 < 600 { print $1 }'" + _TOP
ANY_5XX_FIELD = _ALL + " | grep -E ' 5[0-9]{2} ' | awk '{ print $1 }'" + _TOP


def _ips(rng, n):
    seen, out = set(), []
    while len(out) < n:
        ip = f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
        if ip not in seen:
            seen.add(ip)
            out.append(ip)
    return out


def _line(rng, ts, ip, status, size=None):
    path = rng.choice(PATHS)
    if size is None:
        size = rng.randint(200, 50_000)
        while 500 <= size <= 599:
            size = rng.randint(200, 50_000)  # sizes in 500-599 are reserved for the size decoy
    stamp = ts.strftime("%d/%b/%Y:%H:%M:%S +0000")
    return f'{ip} - - [{stamp}] "GET {path} HTTP/1.1" {status} {size} "-" "curl/8.5.0"'


def _texts(files):
    """Log name (relative to /app/logs) -> text, for the files a solution reads."""
    out = {}
    for rel, data in files.items():
        if rel.startswith("logs/"):
            out[rel[len("logs/"):]] = (gzip.decompress(data) if rel.endswith(".gz") else data).decode()
    return out


def _counts(lines, take):
    return Counter(ln.split()[0] for ln in lines if take(ln))


def _top(lines, take):
    """The IP `sort | uniq -c | sort -rn | head -1` prints: the highest count, and among equal
    counts the byte-wise largest line, i.e. the largest IP string (the profile sets LC_ALL=C.UTF-8)."""
    c = _counts(lines, take)
    if not c:
        raise Reject("no matching lines")
    return max(c.items(), key=lambda kv: (kv[1], kv[0].encode()))[0]


def _unique_top(lines, take):
    """The top IP when it is a unique maximum, else None."""
    best = _counts(lines, take).most_common(2)
    return best[0][0] if best and (len(best) == 1 or best[0][1] != best[1][1]) else None


def _status(ln):
    return int(ln.split()[8])


def _is_5xx(ln):
    return 500 <= _status(ln) < 600


def _answer(texts, names, take):
    return {"/app/answer.txt": _top([ln for n in sorted(names) for ln in texts[n].splitlines()], take) + "\n"}


def _solution(shell, select, take):
    return Solution(shell, lambda f: _answer(_texts(f), [n for n in _texts(f) if select(n)], take))


def _build_logs(rng, names, size_trap):
    n = len(names)
    delta = rng.randint(1, 2)
    ips = _ips(rng, n + 3)
    target, decoys, decoy404, decoy_size = ips[0], ips[1 : n + 1], ips[n + 1], ips[n + 2]
    c = {f: (n - 1) * delta + rng.randint(3, 9) for f in names}
    total_t = sum(c.values())
    start = datetime(2026, 9, 1) + timedelta(days=rng.randint(0, 20))
    out = {}
    for day, name in enumerate(reversed(names)):  # oldest file first, so timestamps increase toward access.log
        recs = [(target, rng.choice([500, 502, 503, 504]), None) for _ in range(c[name])]
        for g, d in zip(names, decoys):
            if g != name:
                recs += [(d, rng.choice([500, 502, 503, 504]), None) for _ in range(c[name] + delta)]
        share = total_t // n + 6
        recs += [(decoy404, 404, None) for _ in range(share)] + [(decoy404, 500, None)]
        if size_trap:
            recs += [(decoy_size, 200, rng.randint(500, 599)) for _ in range(share)]
        for _ in range(rng.randint(450, 650)):
            recs.append((f"192.168.{rng.randint(0, 3)}.{rng.randint(1, 254)}", rng.choices([200, 301, 404, 500], weights=[85, 5, 8, 2])[0], None))
        rng.shuffle(recs)
        ts = start + timedelta(days=day)
        lines = []
        for ip, status, size in recs:
            ts += timedelta(seconds=rng.randint(1, 90))
            lines.append(_line(rng, ts, ip, status, size))
        out[name] = "\n".join(lines) + "\n"
    return out, target


def build(ctx):
    names, size_trap = ctx.params["files"], ctx.params["size_trap"]
    texts, target = _build_logs(ctx.rng, names, size_trap)
    if _unique_top([ln for t in texts.values() for ln in t.splitlines()], _is_5xx) != target:
        raise Reject("the target is not the unique top IP")
    for k in range(1, len(names)):
        for subset in itertools.combinations(names, k):
            if _unique_top([ln for n in subset for ln in texts[n].splitlines()], _is_5xx) == target:
                raise Reject(f"reading only {subset} gives the right answer")
    files = {f"logs/{n}": gzip_bytes(t.encode()) if n.endswith(".gz") else t for n, t in texts.items()}
    shortcuts = {
        "plain-only": _solution(PLAIN_ONLY, lambda n: not n.endswith(".gz"), _is_5xx),
        "all-4xx": _solution(ALL_4XX, lambda n: True, lambda ln: _status(ln) >= 400),
    }
    if any("/" in n for n in names):
        shortcuts["top-level-only"] = _solution(TOP_LEVEL_ONLY, lambda n: "/" not in n, _is_5xx)
    if size_trap:
        shortcuts["any-5xx-field"] = _solution(ANY_5XX_FIELD, lambda n: True, lambda ln: re.search(r" 5[0-9]{2} ", ln) is not None)
    hard = any("/" in n for n in names)
    return TaskSpec(
        instruction=INSTRUCTION.format(extra=" (including rotated, compressed and archived files in subdirectories)" if hard else ""),
        files=files,
        grader=ExactAnswer("/app/answer.txt", target),
        oracle=_solution(ORACLE, lambda n: True, _is_5xx),
        shortcuts=shortcuts,
        params={"files": list(names), "size_trap": size_trap},
    )


FAMILY = Family(
    name="log-triage",
    version=4,  # v4: portable gzip bytes (stored blocks)
    cluster="text-analytics",
    category="shell",
    skills=("shell", "logs", "gzip", "aggregation"),
    difficulties={
        "easy": {"files": ["access.log", "access.log.1.gz"], "size_trap": False},
        "medium": {"files": ["access.log", "access.log.1", "access.log.2.gz"], "size_trap": True},
        "hard": {"files": ["access.log", "access.log.1", "access.log.2.gz", "archive/access.log.3.gz"], "size_trap": True},
    },
    build=build,
)
