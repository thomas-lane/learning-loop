"""error-burst: in which UTC minute did the services under /app/logs write the most ERROR lines?

Each service logs in its own time zone. Every timestamp carries its UTC offset
(`2026-09-01T14:03:27+02:00`, or `Z`), except on hard, where one file has offset-less local
timestamps and declares its offset in its first line (`# timezone: UTC-04:00 ...`). The
level is the second field of each line. The answer is the minute as `YYYY-MM-DD HH:MM`.

Construction (minutes are UTC; o_f is file f's offset):

- target minute M_T with c_f ERROR lines in every file f, c_f > (n-1)*delta;
- per-file decoys: for every file g a minute M_g with c_f + delta ERROR lines in every file
  except g. Over all files M_g has T - c_g + (n-1)*delta < T lines, but over any proper subset
  of the files missing g it beats the target, so skipping any file (a compressed one, or the
  one whose timestamps do not parse) fails. On medium/hard the half-hour offset (+05:30) does
  the same for `hours-only-offset`: reading only the hours of the offset moves that file's
  target lines 30 minutes away, and the decoy for that file wins; on hard, reading the
  header file's local times as UTC moves its lines 4 hours away (`naive-as-utc`);
- a local-time decoy: c_f + delta - 1 ERROR lines in each file f at the UTC minute L - o_f,
  so they share the local clock reading L; dropping the offsets piles them into one minute
  (`ignore-offsets`) while in UTC each pile is smaller than the target;
- a sign decoy: c_f + delta - 1 ERROR lines in file f at UTC minute F - 2*o_f, which all land
  on F when the offset is added instead of subtracted (`offset-sign-flipped`);
- medium/hard: at one minute, more WARN/INFO lines whose message contains `ERROR` than the
  target has ERROR lines, so counting every line that contains `ERROR` fails (`grep-error`).

Background ERROR lines never share a planted minute. `build` checks the unique answer and
every proper subset of the files; generation checks every declared shortcut. Every
solution, right or wrong, is one Python source (`SOLVER`) run with a mode, so the shell
and model forms cannot drift apart.
"""

import gzip
import itertools
from collections import Counter
from datetime import datetime, timedelta

from learning_loop.tasks.spec import ExactAnswer, Family, gzip_bytes, Reject, Solution, TaskSpec

SERVICES = {
    "payments-eu.log": ("payments", 120, "line"),
    "orders-us.log": ("orders", -240, "line"),
    "search-in.log": ("search", 330, "line"),
    "orders-us-legacy.log": ("orders", -240, "header"),
    "gateway.log.1.gz": ("gateway", 0, "line"),
}
ERRORS = ["upstream timed out", "card authorization failed", "connection reset by peer", "database deadlock detected", "invalid response from cache", "quota exceeded"]
OTHERS = [("INFO", "request served"), ("INFO", "cache refreshed"), ("DEBUG", "pool stats ok"), ("WARN", "slow query"), ("INFO", "retried after transient error"), ("INFO", "job finished")]
LOOKALIKES = ["WARN alerting: ERROR rate above threshold", "INFO error-report: forwarded ERROR summary", "WARN monitor: ERROR budget burning fast"]
FIRST, LAST = 60, 1380  # planted UTC minutes of the day

INSTRUCTION = """Several of our services wrote their logs to `/app/logs/`{extra}, and we need to know when the worst burst of errors happened.

The servers run in different time zones. {tz} The level of a log line is its second field.

Count the lines whose level is `ERROR` in **all** the logs, grouped by the minute in which they were written in **UTC**, and find the UTC minute with the most ERROR lines.

Write that minute as `YYYY-MM-DD HH:MM` in UTC (for example `2026-01-31 07:05`), nothing else, to `/app/answer.txt`.
"""
TZ_LINE = "Every timestamp carries its UTC offset (for example `2026-01-31T09:05:00+02:00` is 07:05 UTC, and `Z` means UTC)."
TZ_HEADER = "Timestamps carry their UTC offset (`Z` means UTC), except in a file whose first line declares the offset of all of its timestamps."

SOLVER = '''
from collections import Counter
from datetime import datetime, timedelta


def wanted(name, mode):
    if mode == "plain-only":
        return not name.endswith(".gz")
    return True


def offset_minutes(text, mode):
    """'+05:30' -> 330, '-04:00' -> -240, 'Z' -> 0."""
    if text == "Z":
        return 0
    sign = -1 if text[0] == "-" else 1
    hours, minutes = int(text[1:3]), int(text[4:6])
    return sign * (hours * 60 + (0 if mode == "hours-only-offset" else minutes))


def error_minutes(texts, mode):
    for name, text in texts:
        if not wanted(name, mode):
            continue
        header = None
        for line in text.splitlines():
            if line.startswith("#"):
                if "timezone: UTC" in line:
                    header = line.split("timezone: UTC")[1].split()[0]
                continue
            fields = line.split()
            if mode == "grep-error":
                if "ERROR" not in line:
                    continue
            elif len(fields) < 2 or fields[1] != "ERROR":
                continue
            stamp = fields[0]
            if mode == "ignore-offsets":
                yield stamp[:16].replace("T", " ")
                continue
            local = datetime.strptime(stamp[:19], "%Y-%m-%dT%H:%M:%S")
            if stamp[19:]:
                offset = offset_minutes(stamp[19:], mode)
            elif header is not None and mode != "naive-as-utc":
                offset = offset_minutes(header, mode)
            else:
                offset = 0
            sign = 1 if mode == "offset-sign-flipped" else -1
            yield (local + timedelta(minutes=sign * offset)).strftime("%Y-%m-%d %H:%M")


def answer(texts, mode):
    """texts: [(path relative to /app/logs, text)] sorted by path. The minute, or None."""
    counts = Counter(error_minutes(texts, mode))
    if not counts:
        return None
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
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


def _offset_text(minutes):
    if minutes == 0:
        return "Z"
    sign = "-" if minutes < 0 else "+"
    return f"{sign}{abs(minutes) // 60:02d}:{abs(minutes) % 60:02d}"


def _plan(rng, names, lookalikes):
    """{file: [(utc minute of the day, kind)]} with kind 'E' (ERROR) or 'L' (look-alike), and
    the target minute and the planted minutes."""
    n = len(names)
    off = {f: SERVICES[f][1] for f in names}
    delta = rng.randint(1, 2)
    c = {f: (n - 1) * delta + rng.randint(2, 5) for f in names}
    total = sum(c.values())
    lo_l, hi_l = FIRST + max(off.values()), LAST + min(off.values())
    lo_f, hi_f = FIRST + 2 * max(off.values()), LAST + 2 * min(off.values())
    for _ in range(1000):  # redraw the minutes until no two planted minutes are within 2 minutes
        target = rng.randint(FIRST, LAST)
        per_file = {g: rng.randint(FIRST, LAST) for g in names}
        local = rng.randint(lo_l, hi_l)
        flip = rng.randint(lo_f, hi_f)
        look = rng.randint(FIRST, LAST)
        used = sorted([target, *per_file.values(), look, *(local - off[f] for f in names), *(flip - 2 * off[f] for f in names)])
        if all(b - a >= 3 for a, b in zip(used, used[1:])):
            break
    else:
        raise Reject("planted minutes too close")
    planted = {f: [] for f in names}
    for f in names:
        planted[f] += [(target, "E")] * c[f]
        planted[f] += [(local - off[f], "E")] * (c[f] + delta - 1)
        planted[f] += [(flip - 2 * off[f], "E")] * (c[f] + delta - 1)
        for g in names:
            if g != f:
                planted[f] += [(per_file[g], "E")] * (c[f] + delta)
    if lookalikes:
        for _ in range(total + 1):
            planted[rng.choice(names)].append((look, "L"))
    return planted, target, set(used)


def _counts(planted, names):
    return Counter(m for f in names for m, kind in planted[f] if kind == "E")


def _unique_top(counts):
    best = counts.most_common(2)
    return best[0][0] if best and (len(best) == 1 or best[0][1] != best[1][1]) else None


def _render(rng, planted, used, names, day):
    files = {}
    for f in names:
        service, off, style = SERVICES[f]
        rows = [(m * 60 + rng.randint(0, 59), kind) for m, kind in planted[f]]
        for _ in range(rng.randint(15, 30)):  # background ERROR lines, never in a planted minute
            m = rng.randint(0, 1439)
            if all(abs(m - u) > 1 for u in used):
                rows.append((m * 60 + rng.randint(0, 59), "E"))
        for _ in range(rng.randint(90, 150)):
            rows.append((rng.randint(0, 86399), "O"))
        rows.sort()
        lines = [f"# timezone: UTC{_offset_text(off)} (timestamps in this file are local time, without an offset)"] if style == "header" else []
        for sec, kind in rows:
            local = day + timedelta(seconds=sec, minutes=off)
            stamp = local.strftime("%Y-%m-%dT%H:%M:%S") + ("" if style == "header" else _offset_text(off))
            if kind == "E":
                lines.append(f"{stamp} ERROR {service}: {rng.choice(ERRORS)}")
            elif kind == "L":
                lines.append(f"{stamp} {rng.choice(LOOKALIKES)} in {service}")
            else:
                level, msg = rng.choice(OTHERS)
                lines.append(f"{stamp} {level} {service}: {msg}")
        text = "\n".join(lines) + "\n"
        files[f"logs/{f}"] = gzip_bytes(text.encode()) if f.endswith(".gz") else text
    return files


def build(ctx):
    p = ctx.params
    names = p["files"]
    rng = ctx.rng
    planted, target, used = _plan(rng, names, p["lookalikes"])
    if _unique_top(_counts(planted, names)) != target:
        raise Reject("the target is not the unique top minute")
    for k in range(1, len(names)):
        for subset in itertools.combinations(names, k):
            if _unique_top(_counts(planted, subset)) == target:
                raise Reject(f"reading only {subset} gives the right answer")
    day = datetime(2026, 9, 1) + timedelta(days=rng.randint(0, 25))
    expected = (day + timedelta(minutes=target)).strftime("%Y-%m-%d %H:%M")
    modes = ["ignore-offsets", "offset-sign-flipped"]
    if p["lookalikes"]:
        modes.append("grep-error")
    if any(SERVICES[f][1] % 60 for f in names):
        modes.append("hours-only-offset")
    if any(SERVICES[f][2] == "header" for f in names):
        modes.append("naive-as-utc")
    if any(f.endswith(".gz") for f in names):
        modes.append("plain-only")
    header = any(SERVICES[f][2] == "header" for f in names)
    return TaskSpec(
        instruction=INSTRUCTION.format(
            extra=" (including rotated, gzip-compressed logs)" if any(f.endswith(".gz") for f in names) else "",
            tz=TZ_HEADER if header else TZ_LINE,
        ),
        files=_render(rng, planted, used, names, day),
        grader=ExactAnswer("/app/answer.txt", expected),
        oracle=_solution("oracle"),
        shortcuts={m: _solution(m) for m in modes},
        params={"files": list(names), "lookalikes": p["lookalikes"]},
    )


FAMILY = Family(
    name="error-burst",
    version=1,
    cluster="text-analytics",
    category="data",
    skills=("logs", "timezones", "timestamps", "aggregation", "python"),
    difficulties={
        "easy": {"files": ["payments-eu.log", "orders-us.log"], "lookalikes": False},
        "medium": {"files": ["payments-eu.log", "orders-us.log", "search-in.log"], "lookalikes": True},
        "hard": {"files": ["payments-eu.log", "orders-us-legacy.log", "search-in.log", "gateway.log.1.gz"], "lookalikes": True},
    },
    build=build,
)
