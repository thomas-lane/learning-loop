"""json-aggregate: per-site count and mean temperature of the valid sensor readings in
several JSON Lines files.

Construction: `/app/data/readings/` holds one `.jsonl` file per day (hard: one of them
gzipped and one in `archive/`). A line is either a single reading (`temp_c` at the top level)
or a batch (`readings`: a list of readings with their own `temp_c`). A reading whose
`temp_c` is null or missing failed and is skipped. Medium/hard mix in a newer record schema
that puts the site in `location.site` instead of `site`. The answer is a JSON object
{site: {"count": valid readings, "mean_temp_c": mean rounded to 2 decimals}}.

Traps, planted in every instance: every file has failed readings (null and missing), single
readings and batches with valid values, so treating a failed reading as 0
(`null-as-zero`), ignoring the nested `readings` lists (`flat-only`), ignoring single
readings (`batch-only`) or reading only the first file (`first-file`) changes some count.
Medium/hard: every file has newer-schema records (`top-level-site` drops them). Hard: skipping
the gzipped file (`plain-only`) or the subdirectory (`non-recursive`) loses readings. `build`
rejects a draw whose exact mean lies halfway between two 2-decimal values.

Every solution is one Python source (`SOLVER`) run with a mode, as in csv_revenue: the shell
form reads /app/data/readings, the model runs the same source on the generated files.
"""

import gzip
import json
from datetime import datetime, timedelta
from fractions import Fraction

from learning_loop.tasks.spec import Family, gzip_bytes, ParsedAnswer, Reject, Solution, TaskSpec

SITES = ["north-hall", "south-hall", "lab-a", "lab-b", "cold-store", "roof", "basement"]

INSTRUCTION = """Temperature sensors upload their data as JSON Lines files (one JSON object per line) to `/app/data/readings/`.{files_note}

Each line is a record from one sensor at one site. A record is either a **single reading**, with `temp_c` at the top level, or a **batch**, whose `readings` list holds several readings, each with its own `temp_c`.{site_note} A reading whose `temp_c` is `null` or missing failed: skip it entirely (it is not a reading of 0).

For each site, count the valid readings across all the files and compute their mean `temp_c`, rounded to 2 decimal places. Write the result to `/app/answer.json` as a JSON object keyed by site name, for example:

```json
{{"some-site": {{"count": 12, "mean_temp_c": 21.43}}}}
```
"""

FILES_NOTE = {
    "easy": "",
    "medium": "",
    "hard": " Use every file under that directory, including subdirectories; files ending in `.gz` are gzip-compressed.",
}
SITE_NOTE = {
    "easy": " The site is in the record's `site` field.",
    "medium": " The site is in the record's `site` field; records from newer firmware have no `site` field and put it in `location.site` instead.",
    "hard": " The site is in the record's `site` field or, in records from newer firmware, in `location.site`.",
}

SOLVER = '''
import json


def _site(rec, mode):
    if "site" in rec:
        return rec["site"]
    if mode == "top-level-site":
        return None
    return (rec.get("location") or {}).get("site")


def _selected(listing, mode):
    """listing: [(path relative to the readings directory, text)] sorted by path."""
    if mode == "first-file":
        return [x for x in listing if "/" not in x[0] and x[0].endswith(".jsonl")][:1]
    if mode == "plain-only":
        return [x for x in listing if not x[0].endswith(".gz")]
    if mode == "non-recursive":
        return [x for x in listing if "/" not in x[0]]
    return listing


def answer(listing, mode):
    values = {}
    for _, text in _selected(listing, mode):
        for line in text.splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            site = _site(rec, mode)
            if site is None:
                continue
            if "readings" in rec:
                readings = [] if mode == "flat-only" else rec["readings"]
            else:
                readings = [] if mode == "batch-only" else [rec]
            for r in readings:
                t = r.get("temp_c")
                if t is None:
                    if mode != "null-as-zero":
                        continue
                    t = 0
                values.setdefault(site, []).append(t)
    return {s: {"count": len(v), "mean_temp_c": round(sum(v) / len(v), 2)} for s, v in sorted(values.items())}
'''

_SHELL = """python3 - <<'PY'
{solver}
import gzip
import os

root = "/app/data/readings"
listing = []
for d, _, names in os.walk(root):
    for n in names:
        p = os.path.join(d, n)
        data = open(p, "rb").read()
        listing.append((os.path.relpath(p, root), (gzip.decompress(data) if n.endswith(".gz") else data).decode()))
listing.sort()
with open("/app/answer.json", "w") as f:
    json.dump(answer(listing, {mode!r}), f, indent=2, sort_keys=True)
    f.write("\\n")
PY
"""

_NS: dict = {}
exec(SOLVER, _NS)  # noqa: S102 - our own source; the model form of every solution

PREFIX = "data/readings/"


def _listing(files):
    out = []
    for rel, data in files.items():
        if rel.startswith(PREFIX):
            out.append((rel[len(PREFIX) :], (gzip.decompress(data) if rel.endswith(".gz") else data).decode()))
    return sorted(out)


def _solution(mode):
    def model(files):
        return {"/app/answer.json": json.dumps(_NS["answer"](_listing(files), mode), indent=2, sort_keys=True) + "\n"}

    return Solution(_SHELL.format(solver=SOLVER, mode=mode), model)


def _expected(listing):
    """The answer computed exactly (tenths of a degree as integers); Reject on a rounding tie."""
    values: dict[str, list[int]] = {}
    for _, text in listing:
        for line in text.splitlines():
            rec = json.loads(line)
            site = rec["site"] if "site" in rec else rec["location"]["site"]
            for r in rec["readings"] if "readings" in rec else [rec]:
                if r.get("temp_c") is not None:
                    values.setdefault(site, []).append(round(r["temp_c"] * 10))
    out = {}
    for site, v in sorted(values.items()):
        hundredths = Fraction(sum(v) * 10, len(v))
        if hundredths.denominator == 2:
            raise Reject(f"the mean for {site} is a rounding tie")
        out[site] = {"count": len(v), "mean_temp_c": round(hundredths) / 100}
    return out


def _temp(rng, base):
    return round(base + rng.randint(-25, 25) / 10, 1)


def _record(rng, device, site, base, ts, kind, newer, failed):
    """One JSON line. kind: "single" or "batch"; failed: "null", "missing" or None for a valid single reading."""
    rec: dict = {"device": device}
    if newer:
        rec["location"] = {"site": site, "rack": rng.randint(1, 9)}
        rec["fw"] = "2.1"
    else:
        rec["site"] = site
    if kind == "single":
        rec["ts"] = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        if failed is None:
            rec["temp_c"] = _temp(rng, base)
        elif failed == "null":
            rec["temp_c"] = None
            rec["error"] = "sensor_timeout"
    else:
        rec["batch"] = rng.randint(100, 999)
        kinds = list(failed or []) + ["ok"] * rng.randint(2, 5)
        rng.shuffle(kinds)
        readings = []
        for i, k in enumerate(kinds):
            r: dict = {"ts": (ts + timedelta(seconds=30 * i)).strftime("%Y-%m-%dT%H:%M:%SZ")}
            if k == "ok":
                r["temp_c"] = _temp(rng, base)
            elif k == "null":
                r["temp_c"] = None
            readings.append(r)
        rec["readings"] = readings
    return rec


def build(ctx):
    rng, p = ctx.rng, ctx.params
    sites = rng.sample(SITES, p["n_sites"])
    base = {s: rng.randint(160, 260) / 10 for s in sites}
    devices = {f"dev-{i:02d}": sites[i % len(sites)] for i in range(1, 2 * len(sites) + 1)}
    names = list(p["files"])
    files = {}
    for name in names:
        ts = datetime.strptime(name.split("/")[-1][:10], "%Y-%m-%d") + timedelta(minutes=rng.randint(0, 180))
        # planted: valid single and batch records, failed readings of both kinds, newer-schema records
        plan = [("single", None), ("batch", None), ("single", "null"), ("single", "missing"), ("batch", ["null"]), ("batch", ["missing"])]
        for _ in range(rng.randint(10, 18)):
            kind = rng.choice(["single", "single", "batch"])
            if kind == "single":
                plan.append(("single", rng.choices([None, "null", "missing"], weights=[80, 10, 10])[0]))
            else:
                plan.append(("batch", rng.choices([[], ["null"], ["missing"], ["null", "missing"]], weights=[70, 12, 12, 6])[0]))
        rng.shuffle(plan)
        newer_flags = [p["newer"] and rng.random() < 0.35 for _ in plan]
        if p["newer"]:
            newer_flags[rng.randrange(len(plan))] = True
            for i, (kind, failed) in enumerate(plan):  # at least one newer record with a valid reading
                if failed in (None, []):
                    newer_flags[i] = True
                    break
        lines = []
        for (kind, failed), newer in zip(plan, newer_flags):
            device = rng.choice(sorted(devices))
            ts += timedelta(minutes=rng.randint(3, 40))
            lines.append(json.dumps(_record(rng, device, devices[device], base[devices[device]], ts, kind, newer, failed or None)))
        text = "\n".join(lines) + "\n"
        files[PREFIX + name] = gzip_bytes(text.encode()) if name.endswith(".gz") else text
    listing = _listing({k: v.encode() if isinstance(v, str) else v for k, v in files.items()})
    expected = _expected(listing)
    if set(expected) != set(sites):
        raise Reject("a site has no valid readings")
    modes = ["null-as-zero", "flat-only", "batch-only", "first-file"]
    if p["newer"]:
        modes.append("top-level-site")
    if any(n.endswith(".gz") for n in names):
        modes.append("plain-only")
    if any("/" in n for n in names):
        modes.append("non-recursive")
    return TaskSpec(
        instruction=INSTRUCTION.format(files_note=FILES_NOTE[ctx.difficulty], site_note=SITE_NOTE[ctx.difficulty]),
        files=files,
        grader=ParsedAnswer("/app/answer.json", "json", expected),
        oracle=_solution("oracle"),
        shortcuts={m: _solution(m) for m in modes},
        params={"files": names, "n_sites": p["n_sites"], "newer": p["newer"]},
    )


FAMILY = Family(
    name="json-aggregate",
    version=1,
    cluster="tabular-data",
    category="data",
    skills=("json", "nested-data", "missing-values", "aggregation", "python"),
    difficulties={
        "easy": {"files": ["2026-08-30.jsonl", "2026-08-31.jsonl"], "n_sites": 3, "newer": False},
        "medium": {"files": ["2026-08-29.jsonl", "2026-08-30.jsonl", "2026-08-31.jsonl"], "n_sites": 4, "newer": True},
        "hard": {"files": ["archive/2026-08-27.jsonl", "2026-08-28.jsonl.gz", "2026-08-29.jsonl", "2026-08-30.jsonl", "2026-08-31.jsonl"], "n_sites": 5, "newer": True},
    },
    build=build,
)
