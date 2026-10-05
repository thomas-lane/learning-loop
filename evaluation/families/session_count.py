"""session-count: how many user sessions do the event logs under /app/events contain?

A session is a run of one user's events, in time order across all files, in which each
event follows the user's previous event by at most 30 minutes; a gap of more than 1800 s
starts a new session and a gap of exactly 1800 s does not. The instruction states this rule
exactly; the answer is the total number of sessions.

Construction: each user gets 1-4 sessions of 1-6 events, with gaps of at most 25 minutes
inside a session and at least 40 minutes between sessions, each session on one device
(web, mobile or api; one log file per device). Planted on every instance:

- a gap of exactly 1800 s inside a session, which `>=` turns into a new session
  (`non-strict`); and a gap of exactly 1801 s between two sessions;
- a session longer than 30 minutes, which measuring from the session's first event splits
  (`session-start-window`);
- a session that moves from one device to another (from one file to another), which
  counting each file separately splits (`per-file`);
- a user whose sessions alternate mobile, web, mobile, so reading the files one after the
  other without sorting puts the web session after a later event and merges it (`unsorted`);
- users whose sessions overlap in time, so one gap rule over everyone's events counts far
  fewer sessions (`global-gap`); counting events or distinct users is far off.

Medium and hard add an api log, and the mobile log is in the order the app uploaded its
batches (each session's events together, batches by upload time), not in event time. Hard
splits the web log at noon into `web.log.1.gz` and `web.log` and moves the previous
evening's events into `archive/events.log.1.gz`, with one session only there and one
crossing midnight into today's files; skipping compressed or archived files loses that
session. `build` checks every declared shortcut against the truth through generation.

Every solution, right or wrong, is one Python source (`SOLVER`) run with a mode, so the
shell and model forms cannot drift apart.
"""

import gzip
from datetime import datetime, timedelta

from learning_loop.tasks.spec import Family, gzip_bytes, NumericAnswer, Reject, Solution, TaskSpec

DAY = 86400
GAP = 1800
PAGES = ["/", "/products", "/products/12", "/cart", "/checkout", "/search?q=desk", "/account", "/help"]
SCREENS = ["home", "product", "cart", "checkout", "search", "settings", "orders"]
API_CALLS = ["/v1/orders", "/v1/cart", "/v1/products", "/v1/profile", "/v1/payments"]

INSTRUCTION = """We want to know how many user sessions our product had. Activity events from our {sources} are logged in `/app/events/`{extra}. Each line is one event: a UTC timestamp (ISO 8601), then `user=<id>`, then other fields.{order}

Sessions are defined per user. Take each user's events from **all** the files, ordered by time. The user's first event starts a session. Each later event belongs to the same session if it comes at most 30 minutes after that user's previous event; if the gap since the user's previous event is **more than** 30 minutes (strictly more than 1800 seconds), the event starts a new session. A gap of exactly 30 minutes does not start a new session.{independent}

Count the sessions of all users together and write just that number (nothing else) to `/app/answer.txt`.
"""

SOLVER = '''
from datetime import datetime


def wanted(name, mode):
    if mode == "plain-only":
        return not name.endswith(".gz")
    if mode == "top-level-only":
        return "/" not in name
    return True


def events(texts, mode):
    """(file, seconds, user) for every event line, in file order."""
    out = []
    for name, text in texts:
        if not wanted(name, mode):
            continue
        for line in text.splitlines():
            fields = line.split()
            if not fields:
                continue
            ts = datetime.fromisoformat(fields[0].replace("Z", "+00:00")).timestamp()
            user = next(f[len("user="):] for f in fields if f.startswith("user="))
            out.append((name, ts, user))
    return out


def count(evs, mode):
    """Sessions in (seconds, user) events taken in the given order."""
    n, last, start = 0, {}, {}
    for ts, user in evs:
        if user not in last:
            new = True
        elif mode == "non-strict":
            new = ts - last[user] >= 1800
        elif mode == "session-start-window":
            new = ts - start[user] > 1800
        else:
            new = ts - last[user] > 1800
        if new:
            n += 1
            start[user] = ts
        last[user] = ts
    return n


def answer(texts, mode):
    """texts: [(path relative to /app/events, text)] sorted by path. The number of sessions."""
    evs = events(texts, mode)
    if mode == "count-events":
        return len(evs)
    if mode == "distinct-users":
        return len({u for _, _, u in evs})
    if mode == "global-gap":
        ts = sorted(t for _, t, _ in evs)
        return (1 + sum(1 for a, b in zip(ts, ts[1:]) if b - a > 1800)) if ts else 0
    if mode == "per-file":
        return sum(count(sorted((t, u) for n, t, u in evs if n == name), mode) for name, _ in texts if wanted(name, mode))
    if mode == "unsorted":
        return count([(t, u) for _, t, u in evs], mode)
    return count(sorted((t, u) for _, t, u in evs), mode)
'''

_SHELL = """python3 - <<'PY'
{solver}
import glob
import gzip
import os

texts = []
for p in sorted(p for p in glob.glob("/app/events/**/*", recursive=True) if os.path.isfile(p)):
    data = open(p, "rb").read()
    if p.endswith(".gz"):
        data = gzip.decompress(data)
    texts.append((os.path.relpath(p, "/app/events"), data.decode()))
with open("/app/answer.txt", "w") as f:
    f.write(str(answer(texts, {mode!r})) + "\\n")
PY
"""

_NS: dict = {}
exec(SOLVER, _NS)  # noqa: S102 - our own source; the model form of every solution


def _texts(files):
    return [
        (rel[len("events/"):], (gzip.decompress(data) if rel.endswith(".gz") else data).decode())
        for rel, data in sorted(files.items())
        if rel.startswith("events/")
    ]


def _solution(mode):
    return Solution(_SHELL.format(solver=SOLVER, mode=mode), lambda files: {"/app/answer.txt": f"{_NS['answer'](_texts(files), mode)}\n"})


def _session(rng, n_events, device, lo=20, hi=1500):
    """[(gap before the event, device)]; the first gap is filled in by the caller."""
    return [(rng.randint(lo, hi), device) for _ in range(n_events)]


def _timeline(rng, p):
    """{user: [session]} where a session is [(gap before, device)] and the first gap of a
    session is the gap since the user's previous session (or the start offset)."""
    devices = p["devices"]
    users = []
    while len(users) < p["users"]:
        u = f"u{rng.randint(1000, 9999)}"
        if u not in users:
            users.append(u)
    plan = {}
    for u in users:
        plan[u] = [_session(rng, rng.randint(1, 6), rng.choice(devices)) for _ in range(rng.randint(1, 4))]
    special = rng.sample(users, 6)
    # interleaved: mobile, web, mobile sessions (mobile.log sorts before web.log)
    plan[special[0]] = [_session(rng, rng.randint(2, 4), d) for d in ("mobile", "web", "mobile")]
    # exactly 30 minutes inside a session
    s = _session(rng, rng.randint(3, 5), rng.choice(devices))
    s[rng.randint(1, len(s) - 1)] = (GAP, s[0][1])
    plan[special[1]].insert(rng.randint(0, len(plan[special[1]])), s)
    # a session longer than 30 minutes
    plan[special[2]].insert(0, _session(rng, 5, rng.choice(devices), 1000, 1700))
    # a session that moves to another device (two blocks, so no file's events bridge a gap)
    a, b = rng.sample(devices, 2)
    plan[special[3]].append(_session(rng, rng.randint(2, 3), a) + _session(rng, rng.randint(2, 3), b))
    # a second exact boundary: 1801 s between two sessions
    plan[special[4]] = plan[special[4]][:1] + [_session(rng, rng.randint(1, 3), rng.choice(devices))]
    events = {}  # user -> [(seconds since day start, device)]
    for u, sessions in plan.items():
        t = rng.randint(6 * 3600, 12 * 3600)
        out = []
        for i, sess in enumerate(sessions):
            if i > 0:
                t += GAP + 1 if (u == special[4] and i == 1) else rng.randint(2400, 4 * 3600)
            for j, (gap, device) in enumerate(sess):
                if j > 0:
                    t += gap
                out.append((t, device))
        events[u] = out
    if p["archive"]:
        # one session only on the previous evening, one crossing midnight
        u = special[5]
        first = events[u][0][0]
        evening = [(-DAY + rng.randint(20 * 3600, 21 * 3600), "web")]
        for gap, _ in _session(rng, rng.randint(1, 3), "web")[1:]:
            evening.append((evening[-1][0] + gap, "web"))
        cross = [(-rng.randint(1200, 1700), "mobile")]
        while cross[-1][0] < 600:
            cross.append((cross[-1][0] + rng.randint(300, 900), "mobile"))
        if first - cross[-1][0] <= 2400:
            raise Reject("the midnight session runs into the user's first session")
        events[u] = evening + cross + events[u]
    return events


def _file(name_device, t, archive):
    if archive and t < 0:
        return "archive/events.log.1.gz"
    if archive and name_device == "web" and t < 12 * 3600:
        return "web.log.1.gz"
    return f"{name_device}.log"


def _line(rng, ts, user, device):
    stamp = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    if device == "web":
        return f"{stamp} user={user} event=page_view path={rng.choice(PAGES)}"
    if device == "mobile":
        return f"{stamp} user={user} event=screen_view screen={rng.choice(SCREENS)} app={rng.choice(['ios', 'android'])}"
    return f"{stamp} user={user} event=api_call endpoint={rng.choice(API_CALLS)} status={rng.choice([200, 200, 200, 201, 404])}"


def _render(rng, events, p):
    day = datetime(2026, 9, 1) + timedelta(days=rng.randint(0, 25))
    rows = {}  # file -> [(sort key, line)]
    for u, evs in events.items():
        batch_key = 0
        prev = None  # (seconds, file) of the user's previous event
        for t, device in evs:
            name = _file(device, t, p["archive"])
            if name == "mobile.log" and p["mobile_upload_order"]:
                # one upload batch per run of consecutive mobile.log events, uploaded 1 min-6 h later
                if prev is None or prev[1] != name or t - prev[0] > GAP:
                    batch_key = t + rng.randint(60, 6 * 3600)
                key = (batch_key, t)
            else:
                key = (t, 0)
            rows.setdefault(name, []).append((key, _line(rng, day + timedelta(seconds=t), u, device)))
            prev = (t, name)
    files = {}
    for name in sorted(rows):
        text = "".join(line + "\n" for _, line in sorted(rows[name]))
        files[f"events/{name}"] = gzip_bytes(text.encode()) if name.endswith(".gz") else text
    return files


def _truth(events):
    n = 0
    for evs in events.values():
        ts = sorted(t for t, _ in evs)
        n += 1 + sum(1 for a, b in zip(ts, ts[1:]) if b - a > GAP)
    return n


def build(ctx):
    p = ctx.params
    events = _timeline(ctx.rng, p)
    files = _render(ctx.rng, events, p)
    truth = _truth(events)
    if _NS["answer"](_texts({k: v.encode() if isinstance(v, str) else v for k, v in files.items()}), "oracle") != truth:
        raise AssertionError("the oracle disagrees with the construction")
    modes = ["count-events", "distinct-users", "global-gap", "non-strict", "session-start-window", "per-file", "unsorted"]
    if p["archive"]:
        modes += ["plain-only", "top-level-only"]
    sources = {2: "web site and mobile app", 3: "web site, mobile app and public API"}[len(p["devices"])]
    return TaskSpec(
        instruction=INSTRUCTION.format(
            sources=sources,
            extra=" (including rotated, gzip-compressed and archived files in subdirectories)" if p["archive"] else "",
            order="" if p["mobile_upload_order"] else " Each file is in time order, but the files overlap in time.",
            independent="" if p["mobile_upload_order"] else " Events of different users never affect each other's sessions.",
        ),
        files=files,
        grader=NumericAnswer("/app/answer.txt", float(truth)),
        oracle=_solution("oracle"),
        shortcuts={m: _solution(m) for m in modes},
        params={"devices": list(p["devices"]), "users": p["users"], "archive": p["archive"], "expected": truth},
    )


FAMILY = Family(
    name="session-count",
    version=1,
    cluster="text-analytics",
    category="data",
    skills=("logs", "sessionization", "timestamps", "sorting", "python"),
    difficulties={
        "easy": {"devices": ["web", "mobile"], "users": 9, "mobile_upload_order": False, "archive": False},
        "medium": {"devices": ["web", "mobile", "api"], "users": 11, "mobile_upload_order": True, "archive": False},
        "hard": {"devices": ["web", "mobile", "api"], "users": 13, "mobile_upload_order": True, "archive": True},
    },
    build=build,
)
