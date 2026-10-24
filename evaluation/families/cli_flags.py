"""cli-flags: add command-line options to an argparse script, as specified.

/app/report.py summarizes a tab-separated event log (`python3 report.py LOGFILE` prints
`<category>: <count>`, most events first, ties alphabetical). The instruction, the same at
every difficulty, asks for `--limit N` (an integer of at least 1), `--format text|json|csv`,
`--min-count N` (inclusive), `--only`/`--exclude` (repeatable, mutually exclusive), states
that the filters apply before `--limit`, defines the empty JSON (`[]`) and CSV (the header
line) reports, says which CSV fields are quoted and that the others are not, requires exit
status 2 with nothing on stdout for usage errors and exit status 1 when the log cannot be
read. The oracle implements all of it at every difficulty.

Grading: `Commands` runs the agent's script with hidden argv lists on a hidden log and
compares stdout and the exit status. Difficulty decides which options the hidden checks
exercise and what the log holds:

- easy: five categories, one with a double quote (JSON escaping); checks `--limit`,
  `--format text|json` and `--only` given twice, and the usage errors for `--limit` and
  `--format`;
- medium: six categories, one with a double quote and one with a comma (CSV quoting); adds
  `--format csv` and `--min-count`, with one check's `--min-count` equal to a count;
- hard: seven categories; adds `--only`/`--exclude` combined with `--min-count` and
  `--limit`, both together as a usage error, and an unreadable LOGFILE (exit 1).

Some counts tie at every difficulty; the top count is unique by construction (the planted
tie copies a count below the largest).

Traps (declared shortcuts, each the oracle script with one plausible mistake, each caught by a
planted check of its difficulty): `hardcode-examples` (prints the instruction's example
output), `limit-unchecked` (`type=int`, so `--limit 0` and `--limit -1` are accepted),
`json-hand-built` (string formatting, no escaping), `only-not-repeatable` (`--only` without
`action="append"` keeps the last value, a string); medium adds `csv-no-quoting` and
`min-count-exclusive` (`>` instead of `>=`); hard adds `no-mutex`, `limit-before-filter` and
`argparse-filetype` (`argparse.FileType` makes an unreadable log exit 2 instead of 1).
"""

import csv
import io
import json
from datetime import datetime, timedelta

from learning_loop.tasks.spec import Commands, Family, Solution, TaskSpec

PLAIN = ["auth", "billing", "search", "upload", "cache", "mailer", "export", "scheduler", "webhooks", "reports"]
QUOTED = ['search "beta"', 'flags "new-ui"', 'import "legacy"']
COMMA = ["payments, EU", "payments, US", "db, replica"]
MESSAGES = ["request served", "retrying", "job finished", "timeout after 30s", "user signed in", "queue drained", "cache miss"]

ORIGINAL = '''"""Summarize an event log: how many events each category has.

Usage: python3 report.py LOGFILE

Each line of LOGFILE is "<timestamp>\\t<category>\\t<message>" (tab-separated). The report has
one line per category, "<category>: <count>", most events first; categories with equal
counts are in alphabetical order.
"""

import argparse
from collections import Counter


def load(path):
    counts = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\\n").split("\\t")
            if len(parts) >= 2:
                counts[parts[1]] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description="Count events per category.")
    parser.add_argument("logfile")
    args = parser.parse_args()
    counts = load(args.logfile)
    rows = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    for name, count in rows:
        print(f"{name}: {count}")


if __name__ == "__main__":
    main()
'''

POSITIVE = '''

def positive_int(text):
    """argparse type: an integer of at least 1."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}")
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1: {text!r}")
    return value
'''


def _oracle():
    """The reference script: the original plus every option the instruction asks for."""
    src = ORIGINAL.replace("Usage: python3 report.py LOGFILE\n", "Usage: python3 report.py LOGFILE [--limit N] [--format FORMAT] [--min-count N] [--only CATEGORY ... | --exclude CATEGORY ...]\n")
    src = src.replace("import argparse\n", "import argparse\nimport csv\nimport json\nimport sys\n")
    src = src.replace("    return counts\n", "    return counts\n" + POSITIVE)
    opts = (
        '    parser.add_argument("--limit", type=positive_int, help="print at most N categories")\n'
        '    parser.add_argument("--format", choices=["text", "json", "csv"], default="text")\n'
        '    parser.add_argument("--min-count", type=positive_int, default=1, help="only categories with at least N events")\n'
        "    group = parser.add_mutually_exclusive_group()\n"
        '    group.add_argument("--only", action="append", metavar="CATEGORY", help="report only these categories")\n'
        '    group.add_argument("--exclude", action="append", metavar="CATEGORY", help="leave out these categories")\n'
    )
    src = src.replace('    parser.add_argument("logfile")\n', '    parser.add_argument("logfile")\n' + opts)
    body = (
        "    try:\n        counts = load(args.logfile)\n    except OSError as e:\n"
        '        print(f"error: cannot read {args.logfile}: {e.strerror}", file=sys.stderr)\n        sys.exit(1)\n'
        "    rows = [\n        (name, count)\n        for name, count in counts.items()\n"
        "        if count >= args.min_count\n        and (args.only is None or name in args.only)\n        and (args.exclude is None or name not in args.exclude)\n"
        "    ]\n    rows.sort(key=lambda kv: (-kv[1], kv[0]))\n"
        "    if args.limit is not None:\n        rows = rows[: args.limit]\n"
        '    if args.format == "json":\n        print(json.dumps([{"category": name, "count": count} for name, count in rows]))\n'
        '    elif args.format == "csv":\n        out = csv.writer(sys.stdout, lineterminator="\\n")\n        out.writerow(["category", "count"])\n        out.writerows(rows)\n'
        '    else:\n        for name, count in rows:\n            print(f"{name}: {count}")\n'
    )
    old_body = '    counts = load(args.logfile)\n    rows = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))\n    for name, count in rows:\n        print(f"{name}: {count}")\n'
    return _swap(src, (old_body, body))


def _swap(src, *pairs):
    for old, new in pairs:
        if old not in src:
            raise AssertionError(f"not in the oracle: {old!r}")
        src = src.replace(old, new)
    return src


def _shortcuts(level, oracle, example_output):
    """The declared wrong scripts whose mistake a check of this difficulty catches."""
    out = {
        "hardcode-examples": f"import sys\n\nsys.stdout.write({example_output!r})\n",
        "limit-unchecked": _swap(oracle, ('parser.add_argument("--limit", type=positive_int', 'parser.add_argument("--limit", type=int')),
        "json-hand-built": _swap(
            oracle,
            (
                '        print(json.dumps([{"category": name, "count": count} for name, count in rows]))\n',
                '        items = [\'{"category": "%s", "count": %d}\' % (name, count) for name, count in rows]\n        print("[" + ", ".join(items) + "]")\n',
            ),
        ),
        "only-not-repeatable": _swap(
            oracle,
            ('group.add_argument("--only", action="append", ', 'group.add_argument("--only", '),
        ),
    }
    if level != "easy":
        out["csv-no-quoting"] = _swap(
            oracle,
            ('        out = csv.writer(sys.stdout, lineterminator="\\n")\n        out.writerow(["category", "count"])\n        out.writerows(rows)\n',
             '        print("category,count")\n        for name, count in rows:\n            print(f"{name},{count}")\n'),
        )
        out["min-count-exclusive"] = _swap(oracle, ("if count >= args.min_count", "if count > args.min_count"))
    if level == "hard":
        out["no-mutex"] = _swap(
            oracle,
            ("    group = parser.add_mutually_exclusive_group()\n", ""),
            ('    group.add_argument("--only"', '    parser.add_argument("--only"'),
            ('    group.add_argument("--exclude"', '    parser.add_argument("--exclude"'),
        )
        out["limit-before-filter"] = _swap(
            oracle,
            ("    rows = [\n        (name, count)\n        for name, count in counts.items()\n",
             "    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))\n    if args.limit is not None:\n        ranked = ranked[: args.limit]\n    rows = [\n        (name, count)\n        for name, count in ranked\n"),
            ("    if args.limit is not None:\n        rows = rows[: args.limit]\n", ""),
        )
        out["argparse-filetype"] = _swap(
            oracle,
            ('    parser.add_argument("logfile")\n', '    parser.add_argument("logfile", type=argparse.FileType("r", encoding="utf-8"))\n'),
            ('    with open(path, encoding="utf-8") as f:\n        for line in f:\n', "    with path as f:\n        for line in f:\n"),
        )
    return out


def _log(rng, counts):
    events = [c for c, n in counts.items() for _ in range(n)]
    rng.shuffle(events)
    ts = datetime(2026, 9, 1) + timedelta(days=rng.randint(0, 25), seconds=rng.randint(0, 40000))
    lines = []
    for c in events:
        ts += timedelta(seconds=rng.randint(1, 600))
        lines.append(f"{ts.strftime('%Y-%m-%dT%H:%M:%SZ')}\t{c}\t{rng.choice(MESSAGES)}")
    return "\n".join(lines) + "\n"


def _counts(rng, level):
    """Category -> count: a unique top, one tie below it, a planted quote (and comma) category."""
    n = {"easy": 5, "medium": 6, "hard": 7}[level]
    cats = rng.sample(PLAIN, n - (1 if level == "easy" else 2)) + [rng.choice(QUOTED)] + ([rng.choice(COMMA)] if level != "easy" else [])
    values = rng.sample(range(2, 14), n - 1)
    values.append(rng.choice(sorted(values)[:-1]))  # one tie, below the unique top
    rng.shuffle(cats)
    return dict(zip(cats, values))


def _ranked(counts):
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def _render(rows, fmt):
    if fmt == "json":
        return json.dumps([{"category": n, "count": c} for n, c in rows]) + "\n"
    if fmt == "csv":
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["category", "count"])
        w.writerows(rows)
        return buf.getvalue()
    return "".join(f"{n}: {c}\n" for n, c in rows)


def _expected(counts, limit=None, fmt="text", min_count=1, only=None, exclude=None):
    rows = [(n, c) for n, c in _ranked(counts) if c >= min_count and (only is None or n in only) and (exclude is None or n not in exclude)]
    return _render(rows[:limit] if limit is not None else rows, fmt)


def _checks(rng, level, counts):
    ranked = _ranked(counts)
    log = {"events.log": _log(rng, counts)}
    out = []

    def ok(name, args, **kw):
        out.append({"name": name, "argv": ["python3", "report.py", "events.log", *args], "inputs": log, "stdout": _expected(counts, **kw), "exit": 0})

    def usage(name, args):
        out.append({"name": name, "argv": ["python3", "report.py", "events.log", *args], "inputs": log, "stdout": "", "exit": 2})

    k = rng.randint(2, len(ranked) - 2)
    ok("default", [])
    ok("limit", ["--limit", str(k)], limit=k)
    ok("limit_above_count", ["--limit", str(len(ranked) + 3)], limit=len(ranked) + 3)
    ok("json", ["--format", "json"], fmt="json")
    ok("json_limit", ["--limit", "1", "--format", "json"], limit=1, fmt="json")
    usage("limit_zero", ["--limit", "0"])
    usage("limit_negative", ["--limit", "-1"])
    usage("limit_not_integer", ["--limit", "two"])
    usage("unknown_format", ["--format", "xml"])
    a = rng.choice([n for n, _ in ranked])
    b = rng.choice([n for n, c in ranked if c != counts[a]])  # distinct counts, so the larger one passes --min-count
    ok("only_repeated", ["--only", a, "--only", b], only=[a, b])
    if level != "easy":
        ok("csv", ["--format", "csv"], fmt="csv")
        ok("csv_limit", ["--format", "csv", "--limit", str(k)], fmt="csv", limit=k)
        boundary = ranked[len(ranked) // 2][1]
        ok("min_count_boundary", ["--min-count", str(boundary)], min_count=boundary)
        ok("min_count_json", ["--min-count", str(boundary), "--format", "json"], min_count=boundary, fmt="json")
        usage("min_count_zero", ["--min-count", "0"])
    if level == "hard":
        ok("exclude_top_then_limit", ["--exclude", ranked[0][0], "--limit", "2"], exclude=[ranked[0][0]], limit=2)
        ok("only_min_count", ["--only", a, "--only", b, "--min-count", str(min(counts[a], counts[b]) + 1), "--format", "csv"], only=[a, b], min_count=min(counts[a], counts[b]) + 1, fmt="csv")
        usage("only_and_exclude", ["--only", a, "--exclude", b])
        out.append({"name": "missing_log", "argv": ["python3", "report.py", "missing.log"], "stdout": "", "exit": 1})
    return tuple(out)


INSTRUCTION = """`/app/report.py` summarizes an event log: `python3 report.py LOGFILE` prints one line per category, `<category>: <count>`, most events first and categories with equal counts in alphabetical order. Add these command-line options to it; with none of them, it must behave exactly as it does now.

- `--limit N`: print at most N categories (the first N of the report). N must be a whole number of at least 1.
- `--min-count N`: only categories with at least N events. N must be a whole number of at least 1.
- `--only CATEGORY` and `--exclude CATEGORY`: report only the named categories, or all categories except the named ones. Each option can be given several times to name several categories, but using both options in one command is a usage error.
- `--format FORMAT`: `text`, `json` or `csv`, where `text` is the current output and the default.
  - `json`: the whole report on one line: a JSON array of objects `{{"category": <name>, "count": <count>}}` in report order, written exactly as Python's `json.dumps` writes it with its default settings. An empty report is `[]`.
  - `csv`: a header line `category,count`, then one line per category in report order. A field that contains a comma or a double quote is enclosed in double quotes, and each double quote inside it is doubled; every other field is written as it is, without quotes. An empty report is just the header line.

Options can be combined: the filters (`--min-count`, `--only`, `--exclude`) choose the categories first, and `--limit` then keeps the first N of what remains. A usage error, such as an invalid value, an unknown format or conflicting options, must print a message to standard error, nothing to standard output, and exit with status 2. If LOGFILE cannot be read, print an error message to standard error, nothing to standard output, and exit with status 1.

`/app/sample.log` is a small example log. For instance, `python3 report.py sample.log --limit 2 --format json` prints:

```
{example}```

Use only the Python standard library.
"""


def _write(source):
    return f"cat > /app/report.py <<'PY'\n{source}PY\n"


def build(ctx):
    rng, level = ctx.rng, ctx.difficulty
    counts = _counts(rng, level)
    sample_counts = _counts(rng, level)
    sample = _log(rng, sample_counts)
    example = _expected(sample_counts, limit=2, fmt="json")
    oracle = _oracle()
    instruction = INSTRUCTION.format(example=example)
    shortcuts = _shortcuts(level, oracle, example)
    return TaskSpec(
        instruction=instruction,
        files={"report.py": ORIGINAL, "sample.log": sample},
        grader=Commands(("/app/report.py",), _checks(rng, level, counts)),
        oracle=Solution(_write(oracle), lambda f, s=oracle: {"/app/report.py": s}),
        shortcuts={k: Solution(_write(v), lambda f, s=v: {"/app/report.py": s}) for k, v in shortcuts.items()},
        params={"categories": sorted(counts), "counts": [counts[c] for c in sorted(counts)]},
    )


FAMILY = Family(
    name="cli-flags",
    version=4,
    cluster="write-code",
    category="coding",
    skills=("python", "argparse", "cli", "output-formats"),
    difficulties={"easy": {}, "medium": {}, "hard": {}},
    build=build,
)
