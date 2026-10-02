"""Generate the (committed) log files for tasks/log-triage.

The data is built so that *every* proper subset of the files gives a wrong
answer; only reading all three (including the gzipped one) yields 10.0.0.99:

    access.log only            -> 10.0.0.17      access.log + .1   -> 10.0.0.42
    access.log.1 only          -> 10.0.0.42      access.log + .2gz -> 10.0.0.23
    access.log.2.gz only       -> 10.0.0.23      .1 + .2.gz        -> 10.0.0.23
    all three                  -> 10.0.0.99  <- correct

This matters: a first version let `cat a b | zcat c.gz` (which ignores stdin,
so only reads c.gz) pass by accident - a false-positive reward.

There's also a decoy (10.0.0.8) with many 404s, to catch anyone counting
all >=400 statuses instead of just 5xx.

Re-run with:  python evaluation/scripts/gen_log_triage_data.py
"""

import gzip
import random
from datetime import datetime, timedelta
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "tasks/log-triage/environment/logs"

# 5xx counts per (file, ip). Everything else gets background traffic.
FIVE_XX = {
    "access.log": {"10.0.0.17": 30, "10.0.0.42": 10, "10.0.0.99": 20, "10.0.0.8": 1},
    "access.log.1": {"10.0.0.42": 32, "10.0.0.99": 20, "10.0.0.23": 10, "10.0.0.8": 1},
    "access.log.2.gz": {"10.0.0.99": 20, "10.0.0.23": 45, "10.0.0.8": 1},
}
PATHS = ["/", "/api/users", "/api/orders", "/login", "/static/app.js", "/api/search?q=x"]


def line(rng: random.Random, ts: datetime, ip: str, status: int) -> str:
    path = rng.choice(PATHS)
    size = rng.randint(200, 50_000)
    stamp = ts.strftime("%d/%b/%Y:%H:%M:%S +0000")
    return f'{ip} - - [{stamp}] "GET {path} HTTP/1.1" {status} {size} "-" "curl/8.5.0"'


def main() -> None:
    rng = random.Random(1234)
    OUT.mkdir(parents=True, exist_ok=True)
    # Oldest file first so timestamps increase toward access.log.
    start = datetime(2026, 9, 20)
    for day, name in enumerate(["access.log.2.gz", "access.log.1", "access.log"]):
        records: list[tuple[str, int]] = []
        for ip, n in FIVE_XX[name].items():
            records += [(ip, rng.choice([500, 502, 503, 504])) for _ in range(n)]
        records += [("10.0.0.8", 404) for _ in range(40)]  # the 4xx decoy
        for _ in range(600):  # background: lots of IPs, mostly 2xx, few 5xx
            ip = f"192.168.{rng.randint(0, 3)}.{rng.randint(1, 254)}"
            status = rng.choices([200, 301, 404, 500], weights=[85, 5, 8, 2])[0]
            records.append((ip, status))
        rng.shuffle(records)

        ts = start + timedelta(days=day)
        lines = []
        for ip, status in records:
            ts += timedelta(seconds=rng.randint(1, 90))
            lines.append(line(rng, ts, ip, status))
        text = "\n".join(lines) + "\n"
        if name.endswith(".gz"):
            # mtime=0 keeps the .gz byte-for-byte reproducible.
            with open(OUT / name, "wb") as f, gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as gz:
                gz.write(text.encode())
        else:
            (OUT / name).write_text(text)
    print(f"wrote logs to {OUT}")


if __name__ == "__main__":
    main()
