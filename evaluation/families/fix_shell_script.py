"""fix-shell-script: make a bash log-summary script correct under `set -euo pipefail`
(partial credit).

`/app/logsum.sh LOG_DIR OUT_FILE [PATTERN]` counts the lines containing PATTERN (a fixed
string, default ERROR) in each `*.log` file directly inside LOG_DIR, writes `<name>: <count>`
lines to OUT_FILE and prints `total: <sum>`; a missing LOG_DIR is an error (exit 2). The
script is built from five sites, each with a catalog of classic strict-mode mistakes:

- `pattern`: `pattern=$3` dies with "unbound variable" when PATTERN is omitted;
- `check_dir`: the error is reported but not propagated (`[ -d ... ] || echo ...`, exit 0),
  or `[ ! -d $log_dir ]` is unquoted, so a missing directory whose name has spaces makes
  `[` fail and the script carries on;
- `count`: `grep -c` without `|| true` exits 1 for a file with no match, which kills the
  script under `set -e`; or PATTERN is read as a regular expression;
- `loop`: `for f in $log_dir/*.log` (a directory with spaces splits) or `for f in $(ls ...)`
  (file names with spaces split);
- `guard`: without `[ -f "$f" ] || continue`, a directory with no .log file feeds the literal
  `*.log` to grep.

Easy has one bug, medium and hard two, one of them *hidden* (no visible case fails because
of it; see below). Each bug is drawn uniformly from the sites whose catalog holds a kind the
level allows (`visible_kinds` for the visible bugs, `hidden_kinds` for the hidden one), then
uniformly from those kinds:

- easy: one of the word-splitting bugs (`unquoted_test`, `unquoted_dir`, `ls_words`), which a
  rewrite with quoted expansions fixes on the way;
- medium: those and the missing guard, for both bugs;
- hard: the hidden bug as at medium; the visible one from the whole catalog, so the bugs
  whose failure gives the least to go on (the silent `set -e` exit of a no-match `grep -c`,
  a regex count, an unbound variable, an error that does not stop the script) appear only
  at hard, and only where a failing case points at them.

Hard also moves the counting function into `/app/lib/count.sh`, which the script sources. The
instruction is the same at every difficulty: fix the script "and any file it sources"; some
visible cases fail, and they may not cover every bug.

The grader (`Commands`) runs the script on hidden inputs, one behavior (check kind) per
check, and compares stdout, the exit status and OUT_FILE; one more check requires the line
`set -euo pipefail` to stay. Besides one kind per catalogued bug, four kinds no bug touches
(files the script must skip, a case-sensitive PATTERN, a line with two matches, a PATTERN
with spaces) cover the rest of the header. `/app/test_logsum.py` holds visible cases as a
repository's regression tests would, so an agent that rewrites the loop sees what it breaks:
one for every kind no hidden bug breaks (`EXPOSED_BY`, even when another bug masks it),
except that as many kinds no bug touches as the hidden bugs break (at least two, at every
level) get no case either. A behavior without a visible case is therefore at least as likely
to work as to hide a bug; `build` redraws when too few untouched kinds are left. All / half /
half of the bugs (one at every level) are *visible*: some case fails because of each of
them, both with that bug alone and with every bug in place (`build` redraws when another bug
masks one: `for f in $(ls ...)` never enters the loop without a .log file, which hides a
missing guard). Every other visible case passes on the script as given, and every visible
case passes once only the hidden bugs are left. The hidden bug is what mostly sets medium
apart from easy (a bug no failing case shows is much harder to find).
Hidden checks never reuse a visible case's arguments. In the regex-looking PATTERN check,
each log's decoy lines (which match PATTERN only as a regular expression) differ in number
from its matching lines, so the regex count always differs from the fixed-string count.

Traps, each declared as a shortcut that must fail:
- `visible-bugs-only` (when some bug is hidden): fix only the bugs the visible cases expose;
- `wrong-fix-<site>`, for every bug except the regex one: every bug fixed correctly except
  this one, which gets a plausible wrong fix (`${3:-}` as the default, a bare `exit` after
  the error message, `! [ -d $log_dir ]` still unquoted, `|| echo 0` after `grep -c`, which
  then prints two lines, quoting the whole glob or the whole `$(ls ...)`, an unquoted
  `[ -f $f ]` guard);
- `find-pipeline`: everything fixed, but the loop rewritten as `find ... | sort | while
  read`, which also descends into subdirectories and loses the total in the pipeline's
  subshell;
- `drop-errexit` (when the `count` bug is the no-match exit, at hard): everything else
  fixed, and the no-match exit silenced by removing `-e` from the `set` line.

Generation runs the scripts with the host's bash; the Docker check runs bash 5.2 in the
container. The script and every variant use only behavior that is the same in bash 3.2 and
5.x (no arrays, `mapfile`, `${var,,}` or `[[ -v ]]`; no `wc`/`uniq -c` output, whose padding
differs between BSD and GNU). Expected outputs are computed in Python from the generated
files, and `build` checks that the correct script passes every hidden check.
"""

import json

from learning_loop.tasks.runtime import grade as grade_runtime
from learning_loop.tasks.spec import Commands, Family, Reject, Solution, TaskSpec

SCRIPT = "/app/logsum.sh"
LIB = "/app/lib/count.sh"

HEADER = """#!/usr/bin/env bash
# logsum.sh LOG_DIR OUT_FILE [PATTERN]
#
# For every regular file directly inside LOG_DIR whose name ends in ".log", count the lines
# that contain PATTERN (a fixed string, not a regular expression; case-sensitive; default
# ERROR). File and directory names may contain spaces.
#
# - OUT_FILE gets one line per log file, "<file name>: <count>", sorted by file name (byte
#   order). It is created, empty, when LOG_DIR has no .log file.
# - stdout gets one line: "total: <sum of the counts>".
# - If LOG_DIR is not a directory, print an error to stderr and exit with status 2 without
#   creating OUT_FILE.
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: logsum.sh LOG_DIR OUT_FILE [PATTERN]" >&2
    exit 2
fi
log_dir=$1
out_file=$2
"""

COUNT_FUNCTION = """count_matches() {
    # Print how many lines of file $1 contain the fixed string $2.
@@count@@}
"""

SOURCE_LIB = """# count_matches FILE STRING
. "$(dirname "$0")/lib/count.sh"
"""

LIB_HEADER = """# Sourced by logsum.sh.

"""

LOOP = """
total=0
: > "$out_file"
@@loop@@@@guard@@    n=$(count_matches "$f" "$pattern")
    printf '%s: %s\\n' "${f##*/}" "$n" >> "$out_file"
    total=$((total + n))
done
echo "total: $total"
"""

FIND_LOOP = """
total=0
: > "$out_file"
find "$log_dir" -type f -name '*.log' | sort | while IFS= read -r f; do
    n=$(count_matches "$f" "$pattern")
    printf '%s: %s\\n' "${f##*/}" "$n" >> "$out_file"
    total=$((total + n))
done
echo "total: $total"
"""

SITES = ("pattern", "check_dir", "count", "loop", "guard")

CORRECT = {
    "pattern": "pattern=${3:-ERROR}\n",
    "check_dir": 'if [ ! -d "$log_dir" ]; then\n    echo "logsum: not a directory: $log_dir" >&2\n    exit 2\nfi\n',
    "count": '    grep -c -F -- "$2" "$1" || true\n',
    "loop": 'for f in "$log_dir"/*.log; do\n',
    "guard": '    [ -f "$f" ] || continue  # no match: the pattern stays literal\n',
}

# site -> bug -> (buggy text, plausible wrong fix or None). Every bug but `regex_pattern` has a
# wrong fix, so each site's bug is drawn uniformly from its catalog entries.
BUGS = {
    "pattern": {
        "unset_default": ("pattern=$3\n", "pattern=${3:-}\n"),
    },
    "check_dir": {
        "error_not_fatal": ('[ -d "$log_dir" ] || echo "logsum: not a directory: $log_dir" >&2\n', '[ -d "$log_dir" ] || { echo "logsum: not a directory: $log_dir" >&2; exit; }\n'),
        "unquoted_test": (
            'if [ ! -d $log_dir ]; then\n    echo "logsum: not a directory: $log_dir" >&2\n    exit 2\nfi\n',
            'if ! [ -d $log_dir ]; then\n    echo "logsum: not a directory: $log_dir" >&2\n    exit 2\nfi\n',
        ),
    },
    "count": {
        "no_match_exits": ('    grep -c -F -- "$2" "$1"\n', '    grep -c -F -- "$2" "$1" || echo 0\n'),
        "regex_pattern": ('    grep -c -- "$2" "$1" || true\n', None),
    },
    "loop": {
        "unquoted_dir": ("for f in $log_dir/*.log; do\n", 'for f in "$log_dir/*.log"; do\n'),
        "ls_words": ('for f in $(ls "$log_dir"/*.log); do\n', 'for f in "$(ls "$log_dir"/*.log)"; do\n'),
    },
    "guard": {
        "no_empty_guard": ("", '    [ -f $f ] || continue\n'),
    },
}

# the check that exposes each bug (every check uses the plain form of the other features)
EXPOSED_BY = {
    "unset_default": ["default_pattern"],
    "error_not_fatal": ["missing_directory"],
    "unquoted_test": ["missing_directory"],
    "no_match_exits": ["file_without_matches"],
    "regex_pattern": ["fixed_string_pattern"],
    "unquoted_dir": ["spaces_in_directory"],
    "ls_words": ["spaces_in_file_names", "spaces_in_directory"],
    "no_empty_guard": ["no_log_files"],
}

WORDS = ["api", "web", "db", "auth", "cache", "worker", "queue", "billing", "search", "mail"]
MESSAGES = ["request handled", "connection reset", "retrying job", "cache miss", "slow query", "user login", "disk usage high", "token expired"]
LEVELS_OTHER = ["INFO", "WARN", "DEBUG"]


def _script(chosen, *, split, loop=LOOP, strict="set -euo pipefail"):
    """{absolute path: text} for the script with `chosen` {site: text} in place of the correct sites."""
    site = {s: chosen.get(s, CORRECT[s]) for s in SITES}
    body = loop.replace("@@loop@@", site["loop"]).replace("@@guard@@", site["guard"])
    count = COUNT_FUNCTION.replace("@@count@@", site["count"])
    head = HEADER.replace("set -euo pipefail", strict) + site["pattern"] + "\n" + site["check_dir"] + "\n"
    if split:
        return {SCRIPT: head + SOURCE_LIB + body, LIB: LIB_HEADER + count}
    return {SCRIPT: head + count + body}


def _shell(files, then=""):
    return "".join(f"mkdir -p {path.rsplit('/', 1)[0]}\ncat > {path} <<'SH'\n{text}SH\n" for path, text in sorted(files.items())) + then


def _solution(files, then=""):
    return Solution(_shell(files, then), lambda f, s=dict(files): dict(s))


# --------------------------------------------------------------------------- #
# Inputs and expected outputs
# --------------------------------------------------------------------------- #


def _log(rng, pattern, hits, decoys=()):
    """Log text with exactly `hits` lines containing `pattern`, plus non-matching lines and
    the given decoy lines (which must not contain `pattern`)."""
    stamp = rng.randint(0, 20000)
    lines = []
    for i in range(hits):
        lines.append(f"ERROR {rng.choice(WORDS)}: {rng.choice(MESSAGES)} {pattern} #{rng.randint(1, 999)}" if pattern != "ERROR" else f"ERROR {rng.choice(WORDS)}: {rng.choice(MESSAGES)}")
    for _ in range(rng.randint(3, 8)):
        lines.append(f"{rng.choice(LEVELS_OTHER)} {rng.choice(WORDS)}: {rng.choice(MESSAGES)} error={rng.randint(0, 9)}")
    lines += list(decoys)
    rng.shuffle(lines)
    out = [f"2026-09-{(stamp // 1440) % 28 + 1:02d}T{(stamp // 60) % 24:02d}:{stamp % 60:02d}:{s:02d} {ln}" for s, ln in enumerate(lines)]
    assert sum(pattern in ln for ln in out) == hits
    return "\n".join(out) + "\n"


def _expected(inputs, log_dir, pattern):
    """(OUT_FILE text, stdout) the correct script produces."""
    prefix = log_dir + "/"
    names = sorted(rel[len(prefix) :] for rel in inputs if rel.startswith(prefix) and "/" not in rel[len(prefix) :] and rel.endswith(".log"))
    counts = [(n, sum(pattern in ln for ln in inputs[prefix + n].splitlines())) for n in names]
    return "".join(f"{n}: {c}\n" for n, c in counts), f"total: {sum(c for _, c in counts)}\n"


def _case(name, log_dir, inputs, pattern=None):
    argv = ["bash", "logsum.sh", log_dir, "out.txt"] + ([pattern] if pattern is not None else [])
    out, stdout = _expected(inputs, log_dir, pattern or "ERROR")
    return {"name": name, "argv": argv, "inputs": inputs, "stdout": stdout, "exit": 0, "outputs": {"out.txt": out}}


def _names(rng, k, spaced=False):
    words = rng.sample(WORDS, k)
    if spaced:
        return [f"{w} {rng.choice(['primary', 'replica', 'east', 'night'])}.log" if i < 2 else f"{w}.log" for i, w in enumerate(words)]
    return [f"{w}.log" for w in words]


def _cases(rng):
    """One check per feature. Each check uses the plain form of every other feature (explicit
    PATTERN, plain names, every log with a match, existing directory), so a bug fails only the
    checks of its own feature."""
    tag = rng.randint(100, 999)

    def logs(log_dir, names, pattern="ERROR", zero=None):
        return {f"{log_dir}/{n}": _log(rng, pattern, 0 if n == zero else rng.randint(1, 6)) for n in names}

    d = f"logs-{tag}"
    default = logs(d, _names(rng, 3))
    default[f"{d}/notes.txt"] = _log(rng, "ERROR", 2)  # not a .log file
    default[f"{d}/{rng.choice(WORDS)}.log.1"] = _log(rng, "ERROR", 3)  # rotated: not *.log
    default[f"{d}/archive/old.log"] = _log(rng, "ERROR", 4)  # not directly inside LOG_DIR

    word, other = rng.sample(["disk", "db", "queue", "cache", "token"], 2)
    pattern = rng.choice([f"{word}.{other}", f"{word}[1]", f"{word}*{other}"])
    decoy = pattern.replace(".", "-").replace("[1]", "1").replace("*", "")  # matches the pattern only as a regex
    d2 = f"var-{tag}"
    custom = {}
    for name in _names(rng, 2):
        hits = rng.randint(1, 4)
        # the regex count is the decoys (plus the hits when they match the regex too), so a
        # decoy count other than the hit count makes it differ from the fixed-string count
        decoys = rng.choice([k for k in range(2, 5) if k != hits])
        custom[f"{d2}/{name}"] = _log(rng, pattern, hits, [f"WARN {decoy} seen"] * decoys)

    d3 = f"logs-{tag + 1}"
    names3 = _names(rng, 3)
    zero = logs(d3, names3, zero=rng.choice(names3))

    d4 = f"srv-{tag}"
    spaced_names = logs(d4, _names(rng, 3, spaced=True))

    d5 = f"nightly logs {tag}"
    spaced_dir = logs(d5, _names(rng, 2))

    d6 = f"empty-{tag}"
    no_logs = {f"{d6}/readme.txt": "ERROR: nothing here\n", f"{d6}/app.log.1": _log(rng, "ERROR", 2)}

    # behaviors no catalogued bug breaks: files the script must skip, a case-sensitive
    # PATTERN, lines (not occurrences) counted, a PATTERN with spaces
    d7 = f"app-{tag}"
    others = logs(d7, _names(rng, 2))
    others[f"{d7}/{rng.choice(WORDS)}.log.{rng.randint(1, 3)}"] = _log(rng, "ERROR", rng.randint(1, 4))
    others[f"{d7}/{rng.choice(['notes', 'todo'])}.txt"] = _log(rng, "ERROR", rng.randint(1, 3))
    others[f"{d7}/{rng.choice(['old', 'archive'])}/{rng.choice(WORDS)}.log"] = _log(rng, "ERROR", rng.randint(1, 4))

    word = rng.choice(["Timeout", "Retry", "Quota"])
    d8 = f"jobs-{tag}"
    cased = {f"{d8}/{n}": _log(rng, word, rng.randint(1, 4), [f"INFO {word.lower()} seen", f"WARN {word.upper()} seen"]) for n in _names(rng, 2)}

    d9 = f"api-{tag}"
    doubled = {}
    for name in _names(rng, 2):
        lines = _log(rng, "ERROR", rng.randint(2, 5)).splitlines()
        hit = [i for i, ln in enumerate(lines) if "ERROR" in ln]
        for i in rng.sample(hit, rng.randint(1, len(hit))):
            lines[i] += " (ERROR repeated)"  # still one line
        doubled[f"{d9}/{name}"] = "\n".join(lines) + "\n"

    phrase = f"{rng.choice(['disk', 'queue', 'cache'])} {rng.choice(['full', 'stalled', 'evicted'])}"
    d10 = f"ops-{tag}"
    spaced_pattern = {f"{d10}/{n}": _log(rng, phrase, rng.randint(1, 4), [f"WARN {phrase.replace(' ', '-')} seen"]) for n in _names(rng, 2)}

    return [
        _case("default_pattern", d, default),
        _case("fixed_string_pattern", d2, custom, pattern),
        _case("file_without_matches", d3, zero, "ERROR"),
        _case("spaces_in_file_names", d4, spaced_names, "ERROR"),
        _case("spaces_in_directory", d5, spaced_dir, "ERROR"),
        _case("no_log_files", d6, no_logs, "ERROR"),
        {"name": "missing_directory", "argv": ["bash", "logsum.sh", f"missing logs {tag}", "out.txt", "ERROR"], "inputs": {}, "stdout": "", "exit": 2},
        _case("skips_other_files", d7, others, "ERROR"),
        _case("case_sensitive_pattern", d8, cased, word),
        _case("lines_not_occurrences", d9, doubled, "ERROR"),
        _case("pattern_with_spaces", d10, spaced_pattern, phrase),
    ]


OMITTED_AT_LEAST = 2  # behaviors no bug touches that get no visible case, at every level

STRICT_CHECK = {"name": "strict_mode_kept", "argv": ["grep", "-q", "-x", "set -euo pipefail", "logsum.sh"], "inputs": {}, "exit": 0}


def _results(files, checks, artifacts):
    """Check name -> passed, running the scripts with the trusted grader (host bash)."""
    view = grade_runtime.MemoryView({path: (text.encode(), 0o755) for path, text in files.items()})
    _, log = grade_runtime.grade(Commands(artifacts, tuple(checks)).key(), view, trusted=True)
    return {ln.split()[1].rstrip(":"): ln.startswith("PASS ") for ln in log if ln.startswith(("PASS ", "FAIL "))}


def _passes(files, checks, artifacts):
    return all(_results(files, checks, artifacts).values())


# --------------------------------------------------------------------------- #
# Visible cases, shortcuts, the task
# --------------------------------------------------------------------------- #

TEST_RUNNER = '''"""Run with: python3 test_logsum.py

Each case runs `bash logsum.sh ARGS...` in an empty temporary directory that holds only the
case's files, then compares stdout, the exit status and OUT_FILE with the expected ones."""

import json
import os
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = json.loads(r"""
@@cases@@
""")


def run(case):
    with tempfile.TemporaryDirectory() as tmp:
        for rel, text in case["files"].items():
            path = os.path.join(tmp, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(text)
        p = subprocess.run(["bash", os.path.join(HERE, "logsum.sh")] + case["args"], cwd=tmp, capture_output=True, text=True)
        problems = []
        if p.returncode != case["exit"]:
            problems.append(f"exit status {p.returncode}, expected {case['exit']}")
        if p.stdout.rstrip("\\n") != case["stdout"].rstrip("\\n"):
            problems.append(f"stdout {p.stdout!r}, expected {case['stdout']!r}")
        for rel, want in case.get("out_files", {}).items():
            path = os.path.join(tmp, rel)
            got = open(path).read() if os.path.isfile(path) else None
            if got is None or got.rstrip("\\n") != want.rstrip("\\n"):
                problems.append(f"{rel} is {got!r}, expected {want!r}")
        if problems and p.stderr:
            problems.append(f"stderr: {p.stderr.strip()!r}")
        return problems


if __name__ == "__main__":
    failed = 0
    for case in CASES:
        problems = run(case)
        print(("PASS " if not problems else "FAIL ") + case["name"] + ("" if not problems else ": " + "; ".join(problems)))
        failed += bool(problems)
    raise SystemExit(1 if failed else 0)
'''


def _test_file(visible):
    cases = [
        {"name": c["name"], "args": c["argv"][2:], "files": c["inputs"], "stdout": c["stdout"], "exit": c["exit"], **({"out_files": c["outputs"]} if "outputs" in c else {})}
        for c in visible
    ]
    return TEST_RUNNER.replace("@@cases@@", json.dumps(cases, indent=1))


INSTRUCTION = """`/app/logsum.sh` summarizes a directory of log files; its header comment describes exactly what it must do (usage: `bash logsum.sh LOG_DIR OUT_FILE [PATTERN]`). It runs under `set -euo pipefail` but has bugs: it breaks on some inputs and gives wrong results on others. Some of the cases in `/app/test_logsum.py` fail, and those cases may not cover every bug.

Fix the script, and any file it sources, so it behaves exactly as its header comment describes for any input. Keep the line `set -euo pipefail` at the top of `logsum.sh` and keep the script a bash script (it is run as `bash logsum.sh ...`, not necessarily from `/app`). Use only bash and standard command-line tools. You may add cases to `/app/test_logsum.py` (run it with `python3 test_logsum.py`), but do not change the existing ones.
"""


def build(ctx):
    rng, p = ctx.rng, ctx.params
    split = p["split_lib"]
    artifacts = (SCRIPT, LIB) if split else (SCRIPT,)
    n_visible = max(1, round(p["n_bugs"] * p["visible_bug_tests"]))

    def draw(kinds, taken, k):
        """{site: bug} for k sites outside `taken`, each with a bug drawn from `kinds`."""
        sites = [s for s in SITES if s not in taken and set(kinds) & set(BUGS[s])]
        return {s: rng.choice(sorted(set(kinds) & set(BUGS[s]))) for s in rng.sample(sites, k)}

    hidden = draw(p["hidden_kinds"], (), p["n_bugs"] - n_visible)
    shown = draw(p["visible_kinds"], hidden, n_visible)
    bugs = dict(sorted({**hidden, **shown}.items()))
    visible_bugs = sorted(shown)
    bug_text = {s: BUGS[s][b][0] for s, b in bugs.items()}
    hidden_only = {s: t for s, t in bug_text.items() if s not in visible_bugs}
    fixed, buggy, partial = _script({}, split=split), _script(bug_text, split=split), _script(hidden_only, split=split)

    # Visible cases: one per behavior (check kind) that no hidden bug breaks (even one another
    # bug masks), except some untouched ones left out below, so they pass once only the hidden
    # bugs are left, as a repository's regression tests would. A case no visible bug is meant
    # to break must pass on the script as given;
    # every visible bug must fail a case on its own and with every bug in place (`for f in
    # $(ls ...)` never enters the loop without a .log file, which would hide a missing guard).
    cases = _cases(rng)
    ok_fixed, ok_partial, ok_buggy = (_results(files, cases, artifacts) for files in (fixed, partial, buggy))
    if not all(ok_fixed.values()):
        raise Reject(f"the correct script fails cases {[n for n, ok in ok_fixed.items() if not ok]}")
    unseen = {name for s in hidden_only for name in EXPOSED_BY[bugs[s]]}
    if any(not ok_partial[c["name"]] for c in cases if c["name"] not in unseen):
        raise Reject("a case fails on a hidden bug it is not meant to show")
    # Leave out as many behaviors no bug touches as the hidden bugs break (at least two), so
    # a behavior without a visible case is as likely to work as to hide a bug.
    untied = sorted({c["name"] for c in cases} - {name for b in bugs.values() for name in EXPOSED_BY[b]})
    if len(untied) < max(OMITTED_AT_LEAST, len(unseen)):
        raise Reject("too few untouched behaviors to leave out")
    omitted = set(rng.sample(untied, max(OMITTED_AT_LEAST, len(unseen))))
    visible = [c for c in cases if c["name"] not in unseen | omitted]
    meant = {name for s in visible_bugs for name in EXPOSED_BY[bugs[s]]}
    if any(not ok_buggy[c["name"]] for c in visible if c["name"] not in meant):
        raise Reject("a visible case fails on a bug it is not meant to show")
    for s in visible_bugs:
        alone = _results(_script({s: bug_text[s]}, split=split), visible, artifacts)
        if not any(not alone[c["name"]] and not ok_buggy[c["name"]] for c in visible):
            raise Reject(f"no visible case shows the {s} bug")  # e.g. masked by another bug
    visible = [dict(c, name=f"visible_{c['name']}") for c in visible]

    seen = [c["argv"] for c in visible]
    for _ in range(100):
        checks = _cases(rng) + [STRICT_CHECK]
        if not any(c["argv"] in seen for c in checks if c is not STRICT_CHECK):
            break  # hidden arguments differ from every visible case's, so replaying those cannot pass
    else:
        raise Reject("hidden checks keep repeating a visible case")
    if failed := [n for n, ok in _results(fixed, checks, artifacts).items() if not ok]:
        raise Reject(f"the correct script fails hidden checks {failed}")

    shortcuts = {"find-pipeline": _solution(_script({}, split=split, loop=FIND_LOOP))}
    if hidden_only:
        shortcuts["visible-bugs-only"] = _solution(partial)
    for s, b in bugs.items():
        if BUGS[s][b][1] is not None:
            shortcuts[f"wrong-fix-{s}"] = _solution(_script({s: BUGS[s][b][1]}, split=split))
    if bugs.get("count") == "no_match_exits":
        shortcuts["drop-errexit"] = _solution(_script({"count": bug_text["count"]}, split=split, strict="set -uo pipefail"))

    files = {path[len("/app/") :]: text for path, text in buggy.items()}
    files["test_logsum.py"] = _test_file(visible)
    return TaskSpec(
        instruction=INSTRUCTION,
        files=files,
        modes={"logsum.sh": 0o755},
        grader=Commands(artifacts, tuple(checks)),
        oracle=_solution(fixed, "cd /app && python3 test_logsum.py\n"),
        shortcuts=shortcuts,
        params={"bugs": bugs, "visible_bug_tests": visible_bugs, "split_lib": split, "n_checks": len(checks)},
    )


# the bug kinds each level draws its visible and hidden bugs from (see the module docstring)
QUOTING = ("unquoted_test", "unquoted_dir", "ls_words")
ALL_BUGS = tuple(sorted(b for kinds in BUGS.values() for b in kinds))

FAMILY = Family(
    name="fix-shell-script",
    version=5,
    cluster="fix-code",
    category="debugging",
    skills=("shell", "bash", "debugging", "quoting"),
    difficulties={
        "easy": {"n_bugs": 1, "visible_bug_tests": 1.0, "split_lib": False, "visible_kinds": QUOTING, "hidden_kinds": ()},
        "medium": {"n_bugs": 2, "visible_bug_tests": 0.5, "split_lib": False, "visible_kinds": QUOTING + ("no_empty_guard",), "hidden_kinds": QUOTING + ("no_empty_guard",)},
        "hard": {"n_bugs": 2, "visible_bug_tests": 0.5, "split_lib": True, "visible_kinds": ALL_BUGS, "hidden_kinds": QUOTING + ("no_empty_guard",)},
    },
    build=build,
)
