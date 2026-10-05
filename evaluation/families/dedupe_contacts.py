"""dedupe-contacts: count the distinct people in a contact list in which the same person
appears several times with differently formatted emails and phone numbers.

Two rows are the same person when their normalized emails or normalized phone numbers are
equal, transitively; empty values match nothing and names are not used. Easy normalizes
emails by trimming and lowercasing and phones to their digits. Medium/hard also drop `+tags`
from every email's local part, drop dots from the local part of `gmail.com` addresses only,
and drop a leading US country code `1` from 11-digit numbers. Hard adds a second file (a CRM
export with its own column names) and states the rules without examples.

Construction: people with unique normalized emails and phones, one to three rows each, every
person's rows connected through their own email or phone. Planted in every instance, each
breaking one wrong method by construction:

- a person linked only by an email that differs in case or whitespace (breaks raw emails:
  `no-normalization`, `phone-only-normalized`, `phone-key-only`, `pair-key`);
- a person linked only by a phone in two formats (breaks raw phones and `email-key-only`);
- two people with an empty phone and two with an empty email (`empty-match` merges them);
- two different people with the same name (`by-name` merges them);
- medium/hard: a `+tag` person (`no-plus-tags`), a gmail dots person (`no-gmail-dots`), two
  different people at another domain whose addresses differ only by a dot (`over-normalize`
  merges them), a `+1` country-code person (`no-country-code`), and a person whose third row
  bridges two earlier, unconnected rows (`greedy`, which marks a row as a duplicate only if
  it matches an earlier row, counts that person twice);
- hard: people whose rows are all in the second file (`first-file`).

Every solution is one Python source (`SOLVER`) run with a mode, as in csv_revenue: the shell
form reads /app/data, the model runs the same source on the generated files.
"""

import csv
import io

from learning_loop.tasks.spec import Family, NumericAnswer, Reject, Solution, TaskSpec

FIRST = ["ann", "ben", "carla", "dev", "eva", "farid", "greta", "hiro", "isla", "jon", "kemal", "lucia", "milan", "nora", "oskar", "priya", "rafael", "sara", "tomas", "uma", "viktor", "wanda", "yusuf", "zoe"]
LAST = ["lee", "ng", "okafor", "silva", "berg", "kaya", "moreau", "tanaka", "haddad", "lund", "rossi", "novak", "ivanova", "dube"]
DOMAINS = ["acme.io", "example.org", "mailbox.net", "fastmail.com", "northwind.co", "gmail.com", "gmail.com"]
TAGS = ["news", "shop", "work", "2025", "promo"]
CRM_COLUMNS = ["Contact ID", "Full Name", "Company", "E-mail", "Phone Number"]

INSTRUCTION = """{files}, merged from several address books, so the same person often appears more than once. Count the distinct people.

Two rows belong to the same person when their normalized emails are equal **or** their normalized phone numbers are equal. This is transitive: if row A matches row B and row B matches row C, all three are one person. An empty email or phone matches nothing. Names are not reliable (different people can share a name), so do not use them.

{rules}

Write just the number of distinct people (nothing else) to `/app/answer.txt`.
"""

FILES = {
    1: "`/app/data/contacts.csv` lists contacts (columns `name`, `email`, `phone`)",
    2: "`/app/data/contacts.csv` lists contacts (columns `name`, `email`, `phone`)",
    3: "Contacts are in `/app/data/contacts.csv` (columns `name`, `email`, `phone`) and in `/app/data/crm_export.csv`, an export from the CRM with its own column names. Together they list contacts",
}
RULES = {
    1: "Normalize an email by trimming surrounding whitespace and lowercasing it. Normalize a phone number by keeping only its digits (`(555) 201-3344` becomes `5552013344`).",
    2: (
        "Normalize an email by trimming surrounding whitespace and lowercasing it, then removing a `+tag` from the part before the `@` (`ann+news@example.org` becomes `ann@example.org`). "
        "For `gmail.com` addresses only, also remove every dot before the `@` (`a.nn@gmail.com` becomes `ann@gmail.com`); for every other domain, dots are significant.\n\n"
        "Normalize a phone number by keeping only its digits; if that leaves 11 digits starting with `1`, drop the leading `1` (the US country code), so `+1 (555) 201-3344` and `555.201.3344` are the same number."
    ),
    3: (
        "Normalize an email by trimming surrounding whitespace, lowercasing it and removing any `+tag` from the part before the `@`; for `gmail.com` addresses only, also remove the dots before the `@` (other domains keep them). "
        "Normalize a phone number to its digits, dropping the leading US country code `1` from 11-digit numbers."
    ),
}

SOLVER = '''
import csv
import io


def norm_email(raw, level, mode):
    if mode in ("no-normalization", "phone-only-normalized"):
        return raw
    e = raw.strip().lower()
    if level >= 2 and "@" in e:
        local, _, domain = e.rpartition("@")
        if mode != "no-plus-tags":
            local = local.split("+")[0]
        if (domain == "gmail.com" or mode == "over-normalize") and mode != "no-gmail-dots":
            local = local.replace(".", "")
        e = local + "@" + domain
    return e


def norm_phone(raw, level, mode):
    if mode in ("no-normalization", "email-only-normalized"):
        return raw
    d = "".join(ch for ch in raw if ch.isdigit())
    if level >= 2 and mode != "no-country-code" and len(d) == 11 and d.startswith("1"):
        d = d[1:]
    return d


def _field(row, names):
    for n in names:
        if n in row:
            return row[n] or ""
    return ""


def rows_of(texts, mode):
    """texts: [(file name, text)] sorted by name."""
    if mode == "first-file":
        texts = texts[:1]
    out = []
    for _, text in texts:
        for r in csv.DictReader(io.StringIO(text)):
            out.append((_field(r, ("name", "Full Name")), _field(r, ("email", "E-mail")), _field(r, ("phone", "Phone Number"))))
    return out


def answer(texts, level, mode):
    rows = rows_of(texts, mode)
    keyed = []
    for name, email, phone in rows:
        e, p = norm_email(email, level, mode), norm_phone(phone, level, mode)
        keys = []
        if mode != "phone-key-only" and (e.strip() or mode == "empty-match"):
            keys.append(("email", e))
        if mode != "email-key-only" and (p.strip() or mode == "empty-match"):
            keys.append(("phone", p))
        keyed.append((name, e, p, keys))
    if mode == "by-name":
        return len({name.strip().lower() for name, _, _, _ in keyed})
    if mode == "pair-key":
        return len({(e, p) for _, e, p, _ in keyed})
    if mode == "greedy":
        seen, count = set(), 0
        for _, _, _, keys in keyed:
            if not any(k in seen for k in keys):
                count += 1
            seen.update(keys)
        return count
    parent = list(range(len(keyed)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first = {}
    for i, (_, _, _, keys) in enumerate(keyed):
        for k in keys:
            if k in first:
                parent[find(i)] = find(first[k])
            else:
                first[k] = i
    return len({find(i) for i in range(len(keyed))})
'''

_SHELL = """python3 - <<'PY'
{solver}
import glob
import os

texts = [(os.path.basename(p), open(p, newline="").read()) for p in sorted(glob.glob("/app/data/*.csv"))]
with open("/app/answer.txt", "w") as f:
    f.write("%d\\n" % answer(texts, {level!r}, {mode!r}))
PY
"""

_NS: dict = {}
exec(SOLVER, _NS)  # noqa: S102 - our own source; the model form of every solution


def _texts(files):
    return [(rel.split("/")[-1], data.decode()) for rel, data in sorted(files.items()) if rel.startswith("data/") and rel.endswith(".csv")]


def _solution(mode, level):
    return Solution(_SHELL.format(solver=SOLVER, level=level, mode=mode), lambda files: {"/app/answer.txt": "%d\n" % _NS["answer"](_texts(files), level, mode)})


# ----------------------------------------------------------------------------- generation

PHONE_STYLES = ["({a}) {e}-{n}", "{a}-{e}-{n}", "{a}.{e}.{n}", "{a}{e}{n}", "{a} {e} {n}"]
CC_STYLES = ["+1 {a} {e} {n}", "+1-{a}-{e}-{n}", "1 ({a}) {e}-{n}", "+1 ({a}) {e}-{n}"]


def _phone(rng, digits, styles, avoid=()):
    a, e, n = digits[:3], digits[3:6], digits[6:]
    options = [s.format(a=a, e=e, n=n) for s in styles]
    options = [o for o in options if o not in avoid] or options
    return rng.choice(options)


def _cased(rng, email):
    """A spelling of `email` that differs from it only in letter case or surrounding whitespace."""
    local, _, domain = email.partition("@")
    variants = [local.capitalize() + "@" + domain, email.upper(), local + "@" + domain.upper(), " " + email, email + " ", local.title() + "@" + domain.capitalize()]
    return rng.choice([v for v in variants if v != email])


class _People:
    """People with unique normalized emails and phones. Emails are derived from the person's
    name; uniqueness is kept even with every dot removed, so only planted pairs collide."""

    def __init__(self, rng, level):
        self.rng, self.level = rng, level
        self.people = []  # each: {"name", "rows": [(email, phone)]}
        self.emails, self.phones = set(), set()

    def person(self):
        return self.rng.choice(FIRST), self.rng.choice(LAST)

    def email(self, who, domain=None):
        rng = self.rng
        first, last = who
        styles = [f"{first}.{last}", f"{first}{last}", f"{first[0]}{last}", f"{first}.{last[0]}"]
        while True:
            dom = domain or rng.choice(DOMAINS)
            loc = rng.choice(styles)
            if (loc.replace(".", ""), dom) in self.emails:
                loc = f"{first[0]}{last}{rng.randint(1, 99)}"
            if (loc.replace(".", ""), dom) not in self.emails:
                self.emails.add((loc.replace(".", ""), dom))
                return f"{loc}@{dom}"

    def dotted(self, domain):
        """A fresh `first.last` at `domain` whose dotless form is unused too."""
        while True:
            first, last = self.person()
            if (first + last, domain) not in self.emails:
                self.emails.add((first + last, domain))
                return first, last

    def phone(self):
        while True:
            d = f"{self.rng.randint(201, 989)}{self.rng.randint(200, 999)}{self.rng.randint(0, 9999):04d}"
            if d not in self.phones:
                self.phones.add(d)
                return d

    def fmt(self, digits, avoid=()):
        styles = PHONE_STYLES + (CC_STYLES if self.level >= 2 and self.rng.random() < 0.25 else [])
        return _phone(self.rng, digits, styles, avoid)

    def add(self, who, rows):
        self.people.append({"name": f"{who[0].capitalize()} {who[1].capitalize()}", "rows": rows})
        return self.people[-1]

    def background(self):
        rng = self.rng
        who = self.person()
        e, p = self.email(who), self.phone()
        rows = [(e, self.fmt(p))]
        for _ in range(rng.choice([0, 0, 1, 1, 2])):
            how = rng.choice(["email", "phone", "both"])
            if how == "phone":
                em = self.email(who) if rng.random() < 0.7 else ""
            else:
                em = _cased(rng, e) if rng.random() < 0.5 else e
            if how == "email":
                ph = self.fmt(self.phone()) if rng.random() < 0.6 else ""
            else:
                ph = self.fmt(p)
            rows.append((em, ph))
        self.add(who, rows)


def _people(rng, level, n_background):
    P = _People(rng, level)
    # linked only by an email differing in case/whitespace; the second phone is another number
    who = P.person()
    e = P.email(who)
    P.add(who, [(e, P.fmt(P.phone())), (_cased(rng, e), P.fmt(P.phone()))])
    # linked only by a phone written in two formats; the emails differ
    who = P.person()
    d = P.phone()
    first = P.fmt(d)
    P.add(who, [(P.email(who), first), (P.email(who), _phone(rng, d, PHONE_STYLES, avoid=(first,)))])
    # empty phones and empty emails on different people
    for _ in range(2):
        who = P.person()
        P.add(who, [(P.email(who), "")])
        P.add(P.person(), [("", P.fmt(P.phone()))])
    # two different people with one name
    who = P.person()
    P.add(who, [(P.email(who), P.fmt(P.phone()))])
    P.add(who, [(P.email(who), P.fmt(P.phone()))])
    bridge = None
    if level >= 2:
        who = P.person()
        e = P.email(who, domain=rng.choice(["acme.io", "example.org", "mailbox.net"]))
        local, _, domain = e.partition("@")
        P.add(who, [(f"{local}+{rng.choice(TAGS)}@{domain}", P.fmt(P.phone())), (e, "")])
        who = P.dotted("gmail.com")  # one person, with and without the dot
        P.add(who, [(f"{who[0]}.{who[1]}@gmail.com", P.fmt(P.phone())), (f"{who[0]}{who[1]}@gmail.com", rng.choice(["", P.fmt(P.phone())]))])
        dom = rng.choice(["acme.io", "northwind.co"])
        who = P.dotted(dom)  # two people (with one name), with and without the dot
        P.add(who, [(f"{who[0]}.{who[1]}@{dom}", P.fmt(P.phone()))])
        P.add(who, [(f"{who[0]}{who[1]}@{dom}", P.fmt(P.phone()))])
        who = P.person()
        d = P.phone()
        P.add(who, [(P.email(who), _phone(rng, d, CC_STYLES)), (P.email(who), _phone(rng, d, PHONE_STYLES))])
        who = P.person()
        e1, d2 = P.email(who), P.phone()
        bridge = P.add(who, [(e1, P.fmt(P.phone())), (P.email(who), P.fmt(d2)), (_cased(rng, e1), P.fmt(d2))])
    for _ in range(n_background):
        P.background()
    return P.people, bridge


def _render(rows, columns, crm):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(columns)
    for i, (name, email, phone) in enumerate(rows):
        if crm:
            w.writerow([f"CRM-{i + 1:04d}", name, "", email, phone])
        else:
            w.writerow([name, email, phone])
    return buf.getvalue()


def build(ctx):
    rng, p = ctx.rng, ctx.params
    level = p["level"]
    people, bridge = _people(rng, level, p["n_background"])
    rows = [(i, j, person["name"], e, ph) for i, person in enumerate(people) for j, (e, ph) in enumerate(person["rows"])]
    rng.shuffle(rows)
    if bridge is not None:  # the bridging row comes after the two rows it connects
        b = people.index(bridge)
        pos = sorted(k for k, r in enumerate(rows) if r[0] == b)
        mine = sorted((r for r in rows if r[0] == b), key=lambda r: r[1])
        for k, r in zip(pos, mine):
            rows[k] = r
    if level == 3:
        split = len(rows) * 11 // 20
        parts = {"contacts.csv": rows[:split], "crm_export.csv": rows[split:]}
        if not {r[0] for r in parts["crm_export.csv"]} - {r[0] for r in parts["contacts.csv"]}:
            raise Reject("no person appears only in the CRM export")
    else:
        parts = {"contacts.csv": rows}
    files = {}
    for name, part in parts.items():
        crm = name == "crm_export.csv"
        files[f"data/{name}"] = _render([(r[2], r[3], r[4]) for r in part], CRM_COLUMNS if crm else ["name", "email", "phone"], crm)
    texts = sorted((n, t) for n, t in ((k.split("/")[-1], v) for k, v in files.items()))
    count = _NS["answer"](texts, level, "oracle")
    if count != len(people):
        raise Reject("the rows do not form exactly one group per person")
    modes = ["no-normalization", "email-only-normalized", "phone-only-normalized", "email-key-only", "phone-key-only", "pair-key", "empty-match", "by-name"]
    if level >= 2:
        modes += ["over-normalize", "no-gmail-dots", "no-plus-tags", "no-country-code", "greedy"]
    if level == 3:
        modes.append("first-file")
    return TaskSpec(
        instruction=INSTRUCTION.format(files=FILES[level], rules=RULES[level]),
        files=files,
        grader=NumericAnswer("/app/answer.txt", float(count)),
        oracle=_solution("oracle", level),
        shortcuts={m: _solution(m, level) for m in modes},
        params={"level": level, "people": count, "rows": len(rows)},
    )


FAMILY = Family(
    name="dedupe-contacts",
    version=1,
    cluster="tabular-data",
    category="data",
    skills=("csv", "normalization", "deduplication", "union-find", "python"),
    difficulties={
        "easy": {"level": 1, "n_background": 8},
        "medium": {"level": 2, "n_background": 10},
        "hard": {"level": 3, "n_background": 14},
    },
    build=build,
)
