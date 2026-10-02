"""log-triage family: "which client IP caused the most HTTP 5xx across all logs?"

Shortcut-resistance (checked for every generated instance, see `check`):
- every proper subset of the log files yields a different top IP (so skipping
  rotated/compressed/archived files, or `cat a b | zcat c.gz`, fails);
- counting all >=400 statuses yields a different IP (a 404-heavy decoy);
- medium/hard: counting any ` 5xx ` field (e.g. `grep ' 5[0-9][0-9] '`) yields a
  different IP (a decoy whose response *sizes* are 500-599);
- the true top IP is a unique maximum.

Construction: target T has c_f 5xx in each file f; for every file g a decoy
D_g appears in all files except g with c_f + delta 5xx each, so any subset
missing g is won by D_g, while over all files T wins when c_g > (n-1)*delta.
"""

from __future__ import annotations

import itertools
import random
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .common import UBUNTU_BASE, rng_for, task_toml, write, write_gz

FAMILY = "log-triage"
VERSION = 2  # v2: base images pinned by digest
RNG_VERSION = 1  # random stream; unchanged since v1, so instances keep their content
SKILLS = ["shell", "logs", "gzip", "aggregation"]
DIFFICULTIES = {
    "easy": {"files": ["access.log", "access.log.1.gz"], "size_trap": False},
    "medium": {"files": ["access.log", "access.log.1", "access.log.2.gz"], "size_trap": True},
    "hard": {"files": ["access.log", "access.log.1", "access.log.2.gz", "archive/access.log.3.gz"], "size_trap": True},
}
PATHS = ["/", "/api/users", "/api/orders", "/login", "/static/app.js", "/api/search?q=x"]


def _ips(rng: random.Random, n: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    while len(out) < n:
        ip = f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
        if ip not in seen:
            seen.add(ip)
            out.append(ip)
    return out


def _line(rng: random.Random, ts: datetime, ip: str, status: int, size: int | None = None) -> str:
    path = rng.choice(PATHS)
    if size is None:
        size = rng.randint(200, 50_000)
        while 500 <= size <= 599:
            size = rng.randint(200, 50_000)  # sizes in 500-599 are reserved for the size decoy
    stamp = ts.strftime("%d/%b/%Y:%H:%M:%S +0000")
    return f'{ip} - - [{stamp}] "GET {path} HTTP/1.1" {status} {size} "-" "curl/8.5.0"'


def _parse(text: str) -> list[tuple[str, int, int]]:
    out = []
    for ln in text.splitlines():
        f = ln.split()
        out.append((f[0], int(f[8]), int(f[9])))
    return out


def _top(rows: list[tuple[str, int, int]], pred) -> str | None:
    c = Counter(ip for ip, st, sz in rows if pred(st, sz))
    if not c:
        return None
    best = c.most_common()
    if len(best) > 1 and best[0][1] == best[1][1]:
        return "<tie>"
    return best[0][0]


def check(files: dict[str, str]) -> dict[str, Any]:
    """Answers under the correct reading and under known shortcuts."""
    rows = {name: _parse(text) for name, text in files.items()}
    all_rows = [r for rs in rows.values() for r in rs]
    five = lambda st, sz: 500 <= st < 600  # noqa: E731
    truth = _top(all_rows, five)
    subsets = {}
    names = sorted(files)
    for k in range(1, len(names)):
        for combo in itertools.combinations(names, k):
            subsets["+".join(combo)] = _top([r for n in combo for r in rows[n]], five)
    return {
        "truth": truth,
        "subsets": subsets,
        "ge400": _top(all_rows, lambda st, sz: st >= 400),
        "any_5xx_field": _top(all_rows, lambda st, sz: 500 <= st < 600 or 500 <= sz < 600),
    }


def _build(rng: random.Random, spec: dict[str, Any]) -> tuple[dict[str, str], str]:
    files = spec["files"]
    n = len(files)
    delta = rng.randint(1, 2)
    ips = _ips(rng, n + 3)
    target, decoys, decoy404, decoy_size = ips[0], ips[1 : n + 1], ips[n + 1], ips[n + 2]
    c = {f: (n - 1) * delta + rng.randint(3, 9) for f in files}
    total_t = sum(c.values())
    start = datetime(2026, 9, 1) + timedelta(days=rng.randint(0, 20))
    out: dict[str, str] = {}
    # Oldest file first so timestamps increase toward access.log.
    for day, name in enumerate(reversed(files)):
        recs: list[tuple[str, int, int | None]] = []
        recs += [(target, rng.choice([500, 502, 503, 504]), None) for _ in range(c[name])]
        for g, d in zip(files, decoys):
            if g != name:
                recs += [(d, rng.choice([500, 502, 503, 504]), None) for _ in range(c[name] + delta)]
        share = total_t // n + 6
        recs += [(decoy404, 404, None) for _ in range(share)] + [(decoy404, 500, None)]
        if spec["size_trap"]:
            recs += [(decoy_size, 200, rng.randint(500, 599)) for _ in range(share)]
        for _ in range(rng.randint(450, 650)):
            ip = f"192.168.{rng.randint(0, 3)}.{rng.randint(1, 254)}"
            recs.append((ip, rng.choices([200, 301, 404, 500], weights=[85, 5, 8, 2])[0], None))
        rng.shuffle(recs)
        ts = start + timedelta(days=day)
        lines = []
        for ip, status, size in recs:
            ts += timedelta(seconds=rng.randint(1, 90))
            lines.append(_line(rng, ts, ip, status, size))
        out[name] = "\n".join(lines) + "\n"
    return out, target


def _valid(chk: dict[str, Any], target: str, size_trap: bool) -> bool:
    if chk["truth"] != target:
        return False
    if any(v == target for v in chk["subsets"].values()):
        return False
    if chk["ge400"] == target:
        return False
    if size_trap and chk["any_5xx_field"] == target:
        return False
    return True


INSTRUCTION = """Our web server has been throwing server errors and we need to know which client is triggering the most of them.

The nginx access logs are in `/app/logs/`. Find the client IP address responsible for the most HTTP **5xx** responses across **all** of the logs there{extra}.

Write just that IP address (nothing else) to `/app/answer.txt`.
"""

TEST_SH = """#!/bin/bash
# Hidden grader (runs in a separate verifier container; /app/answer.txt is the only transferred artifact).
EXPECTED="{expected}"
ACTUAL="$(tr -d '[:space:]' < /app/answer.txt 2>/dev/null)"
echo "expected: $EXPECTED"
echo "actual:   ${{ACTUAL:-<missing /app/answer.txt>}}"
if [ "$ACTUAL" = "$EXPECTED" ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
"""

SOLVE_SH = """#!/bin/bash
# Reference solution: every file under /app/logs (plain or gzip), 5xx statuses only.
find /app/logs -type f | sort | while read -r f; do zcat -f "$f"; done \\
  | awk '$9 >= 500 && $9 < 600 { print $1 }' \\
  | sort | uniq -c | sort -rn | head -1 | awk '{ print $2 }' > /app/answer.txt
"""

ENV_DOCKERFILE = f"""FROM {UBUNTU_BASE}
# A plain shell with coreutils/gzip/awk and python3 - nothing task-specific.
RUN apt-get update && apt-get install -y --no-install-recommends python3 \\
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY logs/ /app/logs/
"""

TESTS_DOCKERFILE = f"""FROM {UBUNTU_BASE}
COPY test.sh /tests/test.sh
"""


def generate(out_dir: Path, difficulty: str, seed: int) -> dict[str, Any]:
    spec = DIFFICULTIES[difficulty]
    rng = rng_for(FAMILY, RNG_VERSION, difficulty, seed)
    for _ in range(50):
        files, target = _build(rng, spec)
        chk = check(files)
        if _valid(chk, target, spec["size_trap"]):
            break
    else:  # pragma: no cover - construction guarantees validity in practice
        raise RuntimeError(f"{FAMILY}: could not build a shortcut-resistant instance for seed {seed}")
    out_dir = Path(out_dir)
    for name, text in files.items():
        p = out_dir / "environment" / "logs" / name
        if name.endswith(".gz"):
            write_gz(p, text)
        else:
            write(p, text)
    extra = " (including rotated, compressed and archived files in subdirectories)" if difficulty == "hard" else ""
    write(out_dir / "instruction.md", INSTRUCTION.format(extra=extra))
    write(out_dir / "environment" / "Dockerfile", ENV_DOCKERFILE)
    write(out_dir / "tests" / "test.sh", TEST_SH.format(expected=target), executable=True)
    write(out_dir / "tests" / "Dockerfile", TESTS_DOCKERFILE)
    write(out_dir / "solution" / "solve.sh", SOLVE_SH, executable=True)
    params = {"files": spec["files"], "size_trap": spec["size_trap"]}
    write(
        out_dir / "task.toml",
        task_toml(
            family=FAMILY,
            generator=f"{FAMILY}@v{VERSION}",
            generator_seed=seed,
            difficulty=difficulty,
            category="shell",
            tags=["logs", "gzip", "trap:rotated-logs"] + (["trap:size-field"] if spec["size_trap"] else []),
            skills=SKILLS,
            artifacts=["/app/answer.txt"],
            params=params,
            comment=f"Generated by evaluation/generators ({FAMILY}@v{VERSION}, {difficulty}, seed {seed}). Do not edit by hand.",
        ),
    )
    return params
