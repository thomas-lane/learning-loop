"""organize-files: move support tickets into queue folders chosen by their header fields.

Each ticket in /app/tickets is a file with a header of `<field>:<value>` lines up to the first
empty line, then a body; the value is the rest of the line after the first colon, without the
spaces around it. The queue is `urgent` when the header's Priority is urgent (in any letter
case), otherwise the header's Team in lowercase, and `unassigned` without a Team line. The
grader is a FileTree over /app/tickets: every ticket at `<queue>/<name>` and nothing else, so
copying (originals remain) and partial sorts score below 1. The instruction states every rule
and note at every difficulty, including the exact folder names (all lowercase), so an agent
never needs two folders that differ only in letter case.

Data per difficulty: easy has 10 tickets for 3 teams, with Team and Priority lines written
`Team: billing` (one space, lowercase value), including tickets without a Team line and bodies
that quote forwarded messages; medium has 12 tickets for 4 teams and adds messy values: zero
to two spaces before and after a Team or Priority value (`Team:accounts`), and letter case
varying (Priority per ticket; Team per team, one spelling each, at least one not lowercase,
so folders named by the raw value never differ only in case); hard has the same kinds of
tickets as medium, 20 of them for 6 teams.

Traps and the shortcut that checks each:

- file names carry the team the ticket was opened with, and some tickets were reassigned
  since, so sorting by name (`by-name`) fails;
- some tickets are urgent and have a Team, so routing by Team alone (`ignore-urgent`) fails;
- copying instead of moving (`copy-not-move`) leaves the originals;
- some tickets lack a Team line, so leaving them in place (`skip-missing-team`) fails; some
  bodies quote a forwarded message whose own `Team:`/`Priority:` lines start a line: one
  ticket without a header Team forwards a Team line and one without a header Priority
  forwards `Priority: urgent`, so a `grep -m1 '^Team:'` over the whole file
  (`grep-whole-file`) routes them wrongly;
- medium/hard: a non-urgent ticket of a team spelled not in lowercase, so using the raw value
  as the folder name (`case-sensitive`) fails.

Each shortcut is declared only on the levels whose data contains its trap. Every trap holds by
construction, so the first draw is always used.
"""

from learning_loop.tasks.spec import Family, FileTree, Solution, TaskSpec, tree_entry

TEAMS = ["billing", "shipping", "accounts", "returns", "technical", "sales", "security"]
PRIORITIES = ["low", "normal", "high"]
FIRST = ["dana", "omar", "lena", "raj", "mia", "tom", "ines", "kofi", "yuki", "sam"]
LAST = ["ross", "haddad", "berg", "patel", "novak", "lane", "costa", "mensah", "sato", "reid"]
SUBJECTS = [
    "Charged twice for one order",
    "Parcel marked delivered but not received",
    "Cannot reset my password",
    "Wrong item in the box",
    "Invoice shows the wrong VAT rate",
    "App crashes when I open settings",
    "Question about a bulk discount",
    "Suspicious login from a new device",
    "Need to change the delivery address",
    "Refund still pending after two weeks",
]
BODIES = [
    "Hello, could you please look into this as soon as possible?",
    "I have attached the order number above. Thanks in advance.",
    "This is the second time I am writing about this issue.",
    "Please let me know if you need any more details from me.",
    "We rely on this for our daily work, so a quick answer would help.",
]

INSTRUCTION = """Support tickets are in `/app/tickets/`, one file per ticket. Each file starts with a header of one field per line, written `<field>:<value>`: the field name (such as `Team`), a colon, then the value. The value is the rest of the line after the first colon, without the spaces around it: `Team: billing`, `Team:billing` and `Team:  billing ` all have the value `billing`. The header ends at the first empty line, and the message body follows.

Sort the tickets into queue folders: move every ticket file, keeping its file name and contents, to `/app/tickets/<queue>/<file name>`, where `<queue>` is chosen from the ticket's header:

1. If its `Priority` is `urgent`, the queue is `urgent`. A header without a `Priority` line is not urgent.
2. Otherwise, if it has a `Team` line, the queue is the `Team` value in lowercase (`Team: Billing` goes to `billing`).
3. Otherwise the queue is `unassigned`.

Compare values ignoring letter case (`Priority: URGENT` is urgent). Queue folder names are always lowercase: `urgent`, `unassigned` or a team name in lowercase.

Only the header counts.

Create the queue folders as needed. When you are done, `/app/tickets/` must contain only the queue folders, with every ticket in exactly one of them.
"""

# Every solution that needs header parsing is this one source, run in the container by the
# shell form and in-process by the model, so the two cannot drift apart.
SOLVER = '''
def header(text):
    """{field name: value} of the header (up to the first empty line), first occurrence wins."""
    fields = {}
    for line in text.split("\\n"):
        if line.strip() == "":
            break
        name, sep, value = line.partition(":")
        if sep:
            fields.setdefault(name.strip(), value)
    return fields


def queue(text, mode):
    """The queue folder a ticket goes to under `mode`, or None to leave it where it is."""
    h = header(text)
    fold = (lambda v: v.strip()) if mode == "case-sensitive" else (lambda v: v.strip().lower())
    priority = fold(h["Priority"]) if "Priority" in h else ""
    team = fold(h["Team"]) if "Team" in h else ""
    if priority == "urgent" and mode != "ignore-urgent":
        return "urgent"
    if team:
        return team
    return None if mode == "skip-missing-team" else "unassigned"


def plan(tickets, mode):
    """tickets: {file name: text}. Returns {file name: queue or None}."""
    return {name: queue(text, mode) for name, text in sorted(tickets.items())}
'''

_PY_SHELL = """python3 - <<'PY'
{solver}
import os
import shutil

root = "/app/tickets"
tickets = {{}}
for name in sorted(os.listdir(root)):
    path = os.path.join(root, name)
    if os.path.isfile(path) and not os.path.islink(path):
        with open(path, encoding="utf-8", newline="") as f:
            tickets[name] = f.read()
for name, q in plan(tickets, {mode!r}).items():
    if q is None:
        continue
    os.makedirs(os.path.join(root, q), exist_ok=True)
    {op}(os.path.join(root, name), os.path.join(root, q, name))
PY
"""

BY_NAME = """cd /app/tickets
for f in *.txt; do
  q=${f%%-*}
  mkdir -p "$q" && mv "$f" "$q/"
done
"""

GREP_WHOLE_FILE = """cd /app/tickets
for f in *.txt; do
  team=$(grep -m1 '^Team:' "$f" | cut -d: -f2 | tr -d ' ' | tr 'A-Z' 'a-z')
  pri=$(grep -m1 '^Priority:' "$f" | cut -d: -f2 | tr -d ' ' | tr 'A-Z' 'a-z')
  if [ "$pri" = urgent ]; then q=urgent; elif [ -n "$team" ]; then q=$team; else q=unassigned; fi
  mkdir -p "$q" && mv "$f" "$q/"
done
"""

_NS: dict = {}
exec(SOLVER, _NS)  # noqa: S102 - our own source; the model form of the parsing solutions


def _tickets(files):
    return {rel.split("/", 1)[1]: data.decode() for rel, data in files.items() if rel.startswith("tickets/") and rel.count("/") == 1}


def _moves(files, routes, copy=False):
    tickets = _tickets(files)
    out = {}
    for name, q in routes(tickets).items():
        if q is None:
            continue
        out[f"/app/tickets/{q}/{name}"] = tickets[name]
        if not copy:
            out[f"/app/tickets/{name}"] = None
    return out


def _py_solution(mode, copy=False):
    shell = _PY_SHELL.format(solver=SOLVER, mode="oracle" if copy else mode, op="shutil.copy2" if copy else "shutil.move")
    return Solution(shell, lambda f: _moves(f, lambda t: _NS["plan"](t, "oracle" if copy else mode), copy=copy))


def _grep_first(text, field):
    """`grep -m1 '^Field:' | cut -d: -f2 | tr -d ' ' | tr 'A-Z' 'a-z'` over the whole file."""
    for line in text.split("\n"):
        if line.startswith(field + ":"):
            return line.split(":")[1].replace(" ", "").lower()
    return ""


def _grep_routes(tickets):
    out = {}
    for name, text in tickets.items():
        team, pri = _grep_first(text, "Team"), _grep_first(text, "Priority")
        out[name] = "urgent" if pri == "urgent" else team if team else "unassigned"
    return out


def _padded(rng, value, messy):
    return rng.choice(["", " ", "  "]) + value + rng.choice(["", " ", "  "]) if messy else " " + value


def _styled(value, style):
    return {"lower": value, "capitalize": value.capitalize(), "upper": value.upper()}[style]


def _forwarded(rng, team, priority):
    lines = ["", "Forwarding the earlier request below.", "", "---------- Forwarded message ----------", f"Ticket: T-{rng.randint(10000, 99999)}"]
    if team:
        lines.append(f"Team: {team}")
    if priority:
        lines.append(f"Priority: {priority}")
    lines += [f"Subject: {rng.choice(SUBJECTS)}", "", rng.choice(BODIES)]
    return lines


def _ticket(rng, tid, team, priority, forwarded, messy, spelling):
    who = f"{rng.choice(FIRST)}.{rng.choice(LAST)}@example.com"
    head = [f"Ticket: T-{tid}", f"From: {who}", f"Opened: 2026-09-{rng.randint(1, 28):02d} {rng.randint(7, 19):02d}:{rng.randint(0, 59):02d}", f"Subject: {rng.choice(SUBJECTS)}"]
    tail = []
    if team is not None:
        tail.append(f"Team:{_padded(rng, spelling[team], messy)}")
    if priority is not None:
        style = rng.choice(["lower", "capitalize", "upper"]) if messy else "lower"
        tail.append(f"Priority:{_padded(rng, _styled(priority, style), messy)}")
    rng.shuffle(tail)
    body = rng.sample(BODIES, 2)
    return "\n".join(head + tail + [""] + body + (forwarded or [])) + "\n"


def build(ctx):
    rng, p = ctx.rng, ctx.params
    teams = rng.sample(TEAMS, p["teams"])
    n = p["tickets"]
    # With messy values each team has one spelling, at least one of them not lowercase. Folders named by
    # the raw value then never differ only in letter case, which a case-insensitive host file
    # system would merge when the artifacts are copied out.
    styles = {t: rng.choice(["lower", "capitalize", "upper"]) if p["messy"] else "lower" for t in teams}
    if p["messy"]:
        styles[teams[0]] = rng.choice(["capitalize", "upper"])
    spelling = {t: _styled(t, styles[t]) for t in teams}
    # (name team, header team or None, priority or None, forwarded block or None)
    kinds = ["reassigned"] * 3 + ["urgent"] * 2
    if p["unassigned"]:
        kinds += ["no-team", "no-team-forwarded-team"]
    if p["forwarded"]:
        kinds += ["no-priority-forwarded-urgent", "forwarded-noise"]
    if p["messy"]:
        kinds += ["shouted-team"]
    kinds += ["plain"] * (n - len(kinds))
    rng.shuffle(kinds)
    ids = rng.sample(range(1000, 10000), n)
    files = {}
    routes = {}
    for kind, tid in zip(kinds, ids):
        name_team = teams[0] if kind == "shouted-team" else rng.choice(teams)  # shouted: a non-lowercase Team, not urgent
        team, priority, fwd = name_team, rng.choice(PRIORITIES), None
        if kind == "reassigned":
            team = rng.choice([t for t in teams if t != name_team])
        elif kind == "urgent":
            priority = "urgent"
        elif kind == "no-team":
            team = None
        elif kind == "no-team-forwarded-team":
            team, fwd = None, _forwarded(rng, rng.choice(teams), rng.choice(PRIORITIES))
        elif kind == "no-priority-forwarded-urgent":
            priority, fwd = None, _forwarded(rng, rng.choice(teams), "urgent")
        elif kind == "forwarded-noise":
            fwd = _forwarded(rng, rng.choice([t for t in teams if t != team]), rng.choice(["urgent", "low"]))
        text = _ticket(rng, tid, team, priority, fwd, p["messy"], spelling)
        name = f"{name_team}-{tid}.txt"
        files[f"tickets/{name}"] = text
        routes[name] = _NS["queue"](text, "oracle")
    expected = {f"{q}/{name}": tree_entry(files[f"tickets/{name}"]) for name, q in routes.items()}
    shortcuts = {
        "by-name": Solution(BY_NAME, lambda f: _moves(f, lambda t: {name: name.split("-")[0] for name in t})),
        "copy-not-move": _py_solution("oracle", copy=True),
        "ignore-urgent": _py_solution("ignore-urgent"),
    }
    if p["unassigned"]:
        shortcuts["skip-missing-team"] = _py_solution("skip-missing-team")
    if p["forwarded"]:
        shortcuts["grep-whole-file"] = Solution(GREP_WHOLE_FILE, lambda f: _moves(f, _grep_routes))
    if p["messy"]:
        shortcuts["case-sensitive"] = _py_solution("case-sensitive")
    return TaskSpec(
        instruction=INSTRUCTION,
        files=files,
        grader=FileTree("/app/tickets", expected),
        oracle=_py_solution("oracle"),
        shortcuts=shortcuts,
        params={"tickets": n, "teams": teams, "kinds": sorted(set(kinds))},
    )


FAMILY = Family(
    name="organize-files",
    version=4,
    cluster="filesystem",
    category="shell",
    skills=("shell", "files", "parsing", "mv"),
    difficulties={
        "easy": {"tickets": 10, "teams": 3, "unassigned": True, "forwarded": True, "messy": False},
        "medium": {"tickets": 12, "teams": 4, "unassigned": True, "forwarded": True, "messy": True},
        "hard": {"tickets": 20, "teams": 6, "unassigned": True, "forwarded": True, "messy": True},
    },
    build=build,
)
