"""cron-translate: translate job schedule descriptions into crontab lines.

Task: `/app/jobs.txt` lists jobs (name, command, an English schedule; hard adds a status, and
paused jobs are left out). The agent writes `/app/crontab`, one line per active job:
`minute hour day-of-month month day-of-week command`. The file is parsed (never run) with the
grader's `crontab` format, which compares each job by the times it runs: any equivalent cron
syntax (lists or ranges, `*/N` or an explicit list, names, 7 for Sunday, `@daily`) is correct.

Construction: each job is a structured schedule rendered into English. Every instance has an
"every N minutes" job, a job that runs on Sunday, a day-of-month job and a job on other
weekdays. Medium/hard write some times on a 12-hour clock and always include an afternoon
time (hard also `12:xx AM`, which is hour 0); hard phrases some intervals as "every quarter
hour"/"every half hour", adds month lists, business-hours windows and one paused job.

Traps (declared shortcuts, each writing the lines that method produces, all failing by
construction):
- `sunday-is-1`: weekdays numbered 1-7 from Sunday (every weekday off by one);
- `dom-dow-swap`: the day of the month written in the day-of-week field;
- `every-n-as-value`: "every N minutes" written as minute N (and "every N hours" as hour N);
- medium/hard `ignore-pm`: 12-hour times copied without converting (2:30 PM as hour 2);
- hard `include-paused`: paused jobs included.

An agent writes this file directly, so every solution's shell form is a heredoc with its
lines and the model returns the same text.
"""

from learning_loop.tasks.runtime.grade import parse_crontab
from learning_loop.tasks.spec import Family, ParsedAnswer, Reject, Solution, TaskSpec

DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
JOBS = [
    ("db-backup", "/opt/jobs/backup.sh --full"),
    ("rotate-logs", "/usr/sbin/logrotate /etc/logrotate.conf"),
    ("sync-inventory", "/opt/app/bin/sync --source erp"),
    ("send-digest", "/opt/app/bin/mailer digest --weekly"),
    ("purge-cache", "/opt/app/bin/cache purge --expired"),
    ("health-ping", "/opt/monitor/ping.sh https://status.internal"),
    ("rebuild-index", "/opt/search/reindex --all"),
    ("export-metrics", "/opt/app/bin/metrics export /var/metrics"),
    ("renew-certs", "/opt/tls/renew.sh --quiet"),
    ("cleanup-tmp", "/usr/bin/find /tmp/app -mtime +7 -delete"),
    ("billing-run", "/opt/billing/run.sh --cycle monthly"),
    ("vacuum-db", "/opt/jobs/vacuum.sh analyze"),
    ("fetch-rates", "/opt/app/bin/rates fetch --provider ecb"),
    ("archive-orders", "/opt/app/bin/archive orders --older-than 90d"),
]

INSTRUCTION = """The scheduled jobs for our app server are described in `/app/jobs.txt`. Write the crontab for them to `/app/crontab`: one line per {which} job, in standard five-field cron syntax followed by the job's command exactly as given:

```
minute hour day-of-month month day-of-week command
```

The file is checked by when each job runs, so any equivalent standard cron syntax is fine (lists, ranges, steps, names such as `MON`, `@daily`-style shortcuts). Day-of-week counts from 0 = Sunday (7 also means Sunday), and hours are 0-23. A job that runs at a time of day sets both its minute and hour; "every N hours, on the hour" means minute 0. The order of the lines does not matter.
"""


def _time(rng, fmt12, hour=None):
    h = rng.randint(0, 23) if hour is None else hour
    m = rng.choice([0, 5, 10, 15, 20, 30, 40, 45, 50])
    return h, m, fmt12


def _say_time(t):
    h, m, fmt12 = t
    if not fmt12:
        return f"{h:02d}:{m:02d}"
    return f"{h % 12 or 12}:{m:02d} {'AM' if h < 12 else 'PM'}"


def _ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _and(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


# A job's schedule: {"min": ("step", n) | ("val", m), "hour": ("any",) | ("step", n) | ("val", h, fmt12) | ("range", a, b),
#                    "dom": ("any",) | ("vals", [d...]), "mon": ("any",) | ("vals", [m...]), "dow": ("any",) | ("vals", [d...]) | ("range", a, b)}
ANY = ("any",)


def _at(t):
    return {"min": ("val", t[1]), "hour": ("val", t[0], t[2])}


def _kind(rng, kind, fmt12, hard):
    """(description, schedule) for one kind of job."""
    t = _time(rng, fmt12)
    if kind == "every_min":
        n = rng.choice([5, 10, 15, 20, 30])
        phrase = {15: "every quarter hour", 30: "every half hour"}.get(n) if hard and rng.random() < 0.6 else None
        return phrase or f"every {n} minutes", {"min": ("step", n), "hour": ANY, "dom": ANY, "mon": ANY, "dow": ANY}
    if kind == "every_hours":
        n = rng.choice([2, 3, 4, 6, 8, 12])
        return f"every {n} hours, on the hour", {"min": ("val", 0), "hour": ("step", n), "dom": ANY, "mon": ANY, "dow": ANY}
    if kind == "daily":
        return f"every day at {_say_time(t)}", {**_at(t), "dom": ANY, "mon": ANY, "dow": ANY}
    if kind == "sunday":
        return f"every Sunday at {_say_time(t)}", {**_at(t), "dom": ANY, "mon": ANY, "dow": ("vals", [0])}
    if kind == "weekend":
        return f"on Saturdays and Sundays at {_say_time(t)}", {**_at(t), "dom": ANY, "mon": ANY, "dow": ("vals", [0, 6])}
    if kind == "weekly":
        d = rng.randint(1, 6)
        return f"every {DAYS[d]} at {_say_time(t)}", {**_at(t), "dom": ANY, "mon": ANY, "dow": ("vals", [d])}
    if kind == "weekdays":
        return f"Monday through Friday at {_say_time(t)}", {**_at(t), "dom": ANY, "mon": ANY, "dow": ("range", 1, 5)}
    if kind == "dow_list":
        days = sorted(rng.sample(range(1, 7), rng.randint(2, 3)))
        return f"on {_and([DAYS[d] + 's' for d in days])} at {_say_time(t)}", {**_at(t), "dom": ANY, "mon": ANY, "dow": ("vals", days)}
    if kind == "monthly":
        d = rng.randint(2, 28)
        return f"on the {_ordinal(d)} of every month at {_say_time(t)}", {**_at(t), "dom": ("vals", [d]), "mon": ANY, "dow": ANY}
    if kind == "dom_list":
        days = sorted(rng.sample(range(1, 29), 2))
        return f"on the {_ordinal(days[0])} and the {_ordinal(days[1])} of every month at {_say_time(t)}", {**_at(t), "dom": ("vals", days), "mon": ANY, "dow": ANY}
    if kind == "month_list":
        start = rng.randint(1, 3)
        months = [start, start + 3, start + 6, start + 9]
        return f"at {_say_time(t)} on the 1st of {_and([MONTHS[m - 1] for m in months])}", {**_at(t), "dom": ("vals", [1]), "mon": ("vals", months), "dow": ANY}
    if kind == "business":
        n = rng.choice([10, 15, 20, 30])
        a, b = rng.choice([(8, 17), (9, 17), (9, 18), (7, 19)])
        return (
            f"every {n} minutes from {a:02d}:00 through {b:02d}:59, Monday through Friday",
            {"min": ("step", n), "hour": ("range", a, b), "dom": ANY, "mon": ANY, "dow": ("range", 1, 5)},
        )
    raise ValueError(kind)


def _field(spec, bug, what):
    kind = spec[0]
    if kind == "any":
        return "*"
    if kind == "step":
        return str(spec[1]) if bug == "every-n-as-value" else f"*/{spec[1]}"
    if kind == "val":
        if what == "hour" and bug == "ignore-pm" and spec[2]:
            return str(spec[1] % 12 or 12)
        return str(spec[1])
    if what == "dow":
        if bug == "sunday-is-1":
            spec = (kind, spec[1] + 1, spec[2] + 1) if kind == "range" else (kind, [d + 1 for d in spec[1]])
    if kind == "range":
        return f"{spec[1]}-{spec[2]}"
    return ",".join(str(v) for v in sorted(spec[1]))


def _line(job, bug=None):
    s = job["sched"]
    fields = [_field(s["min"], bug, "min"), _field(s["hour"], bug, "hour"), _field(s["dom"], bug, "dom"), _field(s["mon"], bug, "mon"), _field(s["dow"], bug, "dow")]
    if bug == "dom-dow-swap":
        fields[2], fields[4] = fields[4], fields[2]
    return " ".join(fields) + " " + job["command"]


def _crontab(jobs, bug=None):
    return "".join(_line(j, bug) + "\n" for j in jobs if not j["paused"] or bug == "include-paused")


def _heredoc(text):
    return f"cat > /app/crontab <<'EOF'\n{text}EOF\n"


def _fixed(text):
    return Solution(_heredoc(text), lambda files: {"/app/crontab": text})


def build(ctx):
    rng, p = ctx.rng, ctx.params
    hard = p["paused"]
    kinds = list(p["required"]) + rng.sample(p["extra"], p["n_jobs"] - len(p["required"]))
    rng.shuffle(kinds)
    names = rng.sample(JOBS, len(kinds) + (1 if hard else 0))
    fmt12 = [p["clock12"] and rng.random() < 0.5 for _ in kinds]
    jobs = []
    for i, (kind, (name, cmd)) in enumerate(zip(kinds, names)):
        desc, sched = _kind(rng, kind, fmt12[i], hard)
        jobs.append({"name": name, "command": cmd, "desc": desc, "sched": sched, "paused": False})
    if p["clock12"]:
        # an afternoon time on a 12-hour clock (hour 13-23), and on hard a 12:xx AM time (hour 0)
        timed = [j for j in jobs if j["sched"]["hour"][0] == "val"]
        if len(timed) < 2:
            raise Reject("too few jobs with a time of day")
        pm, am = rng.sample(timed, 2)
        _retime(rng, pm, rng.randint(13, 23))
        if hard:
            _retime(rng, am, 0)
    if hard:
        name, cmd = names[-1]
        desc, sched = _kind(rng, rng.choice(["daily", "weekly", "monthly"]), False, hard)
        jobs.insert(rng.randint(0, len(jobs)), {"name": name, "command": cmd, "desc": desc, "sched": sched, "paused": True})
    lines = [_line(j) for j in jobs if not j["paused"]]
    if len(set(lines)) != len(lines):
        raise Reject("two jobs share a crontab line")
    blocks = []
    for j in jobs:
        block = [f"job: {j['name']}", f"command: {j['command']}", f"schedule: {j['desc']}"]
        if hard:
            block.append(f"status: {'paused' if j['paused'] else 'active'}")
        blocks.append("\n".join(block))
    text = "# Scheduled jobs for app-01. All times are server time (UTC).\n\n" + "\n\n".join(blocks) + "\n"
    expected = parse_crontab("".join(ln + "\n" for ln in lines))
    bugs = ["sunday-is-1", "dom-dow-swap", "every-n-as-value"] + (["ignore-pm"] if p["clock12"] else []) + (["include-paused"] if hard else [])
    return TaskSpec(
        instruction=INSTRUCTION.format(which="active (not paused)" if hard else "listed"),
        files={"jobs.txt": text},
        grader=ParsedAnswer("/app/crontab", "crontab", expected),
        oracle=_fixed(_crontab(jobs)),
        shortcuts={b: _fixed(_crontab(jobs, b)) for b in bugs},
        params={"n_jobs": len(jobs), "clock12": p["clock12"], "paused": hard},
    )


def _retime(rng, job, hour):
    """Give a timed job a new hour on the 12-hour clock and re-render its description."""
    old = (job["sched"]["hour"][1], job["sched"]["min"][1], job["sched"]["hour"][2])
    new = (hour, old[1], True)
    job["sched"] = {**job["sched"], "hour": ("val", hour, True)}
    job["desc"] = job["desc"].replace(_say_time(old), _say_time(new))


FAMILY = Family(
    name="cron-translate",
    version=1,
    cluster="config-repair",
    category="config",
    skills=("cron", "translation", "canonical-form"),
    difficulties={
        "easy": {"n_jobs": 4, "required": ["every_min", "sunday", "monthly", "weekly"], "extra": [], "clock12": False, "paused": False},
        "medium": {
            "n_jobs": 6,
            "required": ["every_min", "weekend", "monthly", "weekdays", "dow_list"],
            "extra": ["daily", "every_hours", "dom_list"],
            "clock12": True,
            "paused": False,
        },
        "hard": {
            "n_jobs": 8,
            "required": ["every_min", "sunday", "dom_list", "dow_list", "business", "month_list"],
            "extra": ["daily", "every_hours", "weekdays", "weekend", "monthly"],
            "clock12": True,
            "paused": True,
        },
    },
    build=build,
)
