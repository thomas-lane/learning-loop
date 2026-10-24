"""The tabular-data families join-orders, sqlite-query, json-aggregate and dedupe-contacts:
deterministic renders, and their traps re-derived from the rendered files, independently of
each family's own solution models; csv-revenue's share of instances whose quoted commas
decide the answer; and, for the whole cluster, one instruction at every difficulty."""

from __future__ import annotations

import csv
import gzip
import importlib.util
import io
import itertools
import json
import re
import sqlite3
from collections import Counter
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path

import pytest

from learning_loop.core.config import REPO_ROOT
from learning_loop.tasks.generate import file_bytes, generate_spec, grade_in_process
from learning_loop.tasks.render import render

MODULES = ("join_orders", "sqlite_query", "json_aggregate", "dedupe_contacts")


def _load(module: str):
    path = REPO_ROOT / "evaluation" / "families" / f"{module}.py"
    spec = importlib.util.spec_from_file_location(f"family_{module}", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MODS = {m: _load(m) for m in (*MODULES, "csv_revenue")}
FAMILIES = {m: MODS[m].FAMILY for m in MODULES}
SEEDS = [1, 2, 3]


def _cases(module):
    return [(d, s) for d in FAMILIES[module].difficulties for s in SEEDS]


def _expected(d: Path):
    return json.loads((d / "tests" / "key.json").read_text())["expected"]


def _files(d: Path, sub: str) -> dict[str, bytes]:
    root = d / "environment" / "files" / sub
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


# ----------------------------------------------------------------------------- join-orders

ID_RE = re.compile(r"^([A-Za-z]+)-(\d+)$")


def _parts(raw: str) -> tuple[str, int]:
    m = ID_RE.match(raw.strip())
    assert m, raw
    return m.group(1).upper(), int(m.group(2))


@pytest.mark.parametrize("difficulty,seed", _cases("join_orders"))
def test_join_orders_traps(tmp_path, difficulty, seed):
    params = render(FAMILIES["join_orders"], difficulty, seed, tmp_path)
    files = {k: v.decode() for k, v in _files(tmp_path, "data").items()}
    expected = _expected(tmp_path)
    customers = [r["customer_id"] for r in csv.DictReader(io.StringIO(files["customers.csv"]))]
    by_parts = {_parts(c): c for c in customers}
    order_files = sorted(k for k in files if k.startswith("orders/"))
    orders = {k: list(csv.DictReader(io.StringIO(files[k]))) for k in order_files}

    def totals(names, key):
        t = {c: 0 for c in customers}
        for n in names:
            for r in orders[n]:
                c = key(r["customer_id"])
                if c in t:
                    t[c] += int(r["quantity"])
        return t

    right = lambda raw: by_parts.get(_parts(raw))  # noqa: E731
    assert totals(order_files, right) == expected
    assert totals(order_files, lambda raw: raw) != expected  # exact-string join
    for k in range(1, len(order_files)):
        for combo in itertools.combinations(order_files, k):
            assert totals(combo, right) != expected, combo
    assert any(v == 0 for v in expected.values())  # an inner join drops these
    orphans = [r for rows in orders.values() for r in rows if right(r["customer_id"]) is None]
    assert len(orphans) >= 2  # an outer join adds these

    # an order whose ID differs from its customer's in exactly one respect, for each normalization in play
    only = Counter()
    for rows in orders.values():
        for r in rows:
            raw, canon = r["customer_id"], right(r["customer_id"])
            if canon is None:
                continue
            spaced, cased = raw != raw.strip(), raw.strip() != raw.strip().upper()
            zeros = ID_RE.match(raw.strip()).group(2) != canon.split("-")[1]
            if [spaced, cased, zeros].count(True) == 1:
                only["S" if spaced else "C" if cased else "Z"] += 1
    assert set(only) == set(params["flags"])
    listed, other = "order_id,placed,customer_id,quantity", "customer_id,quantity,order_id,placed"
    assert [files[n].splitlines()[0] for n in order_files] == [listed if lay == "L" else other for lay in params["layouts"]]


def test_join_orders_column_orders_per_level():
    """Every level has a file whose columns are not in the order the instruction lists them, so
    indexing columns by that order misreads it; medium and hard also have files whose orders
    differ, so reusing the first file's column positions misreads one of them."""
    fam = FAMILIES["join_orders"]
    assert {d: p["layouts"] for d, p in fam.difficulties.items()} == {"easy": "R", "medium": "LR", "hard": "LLR"}
    for difficulty in fam.difficulties:
        spec = generate_spec(fam, difficulty, 1).spec
        assert "fixed-columns" in spec.shortcuts or "listed-columns" in spec.shortcuts
        files = file_bytes(spec)
        for name in spec.shortcuts:
            if name.endswith("-columns"):
                assert grade_in_process(spec, files, spec.shortcuts[name].model(files)) == 0.0, (difficulty, name)


# ----------------------------------------------------------------------------- sqlite-query


@pytest.mark.parametrize("difficulty,seed", _cases("sqlite_query"))
def test_sqlite_query_traps(tmp_path, difficulty, seed):
    params = render(FAMILIES["sqlite_query"], difficulty, seed, tmp_path)
    db = tmp_path / "environment" / "files" / "data" / "shop.db"
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    level, region, month = params["level"], params["region"], f"2026-{params['month']:02d}"
    customers = {r["id"]: dict(r) for r in con.execute("SELECT * FROM customers")}
    orders = [dict(r) for r in con.execute("SELECT * FROM orders")]
    addresses = [dict(r) for r in con.execute("SELECT * FROM addresses")]
    con.close()
    primary = {a["customer_id"]: a["region"] for a in addresses if a["is_primary"]}
    assert len(primary) == len(customers) == len({a["customer_id"] for a in addresses}) == sum(a["is_primary"] for a in addresses)
    signup = {c["id"]: c["region"] for c in customers.values()}  # customers.region, before it is replaced below
    for c in customers.values():
        c["region"] = primary[c["id"]]
    if level < 3:  # one address per customer, and every order ships from its customer's region
        assert len(addresses) == len(customers)
        assert all(o["region"] == customers[o["customer_id"]]["region"] for o in orders)
    deleted = lambda c: bool(c.get("deleted_at"))  # noqa: E731
    assert all("T" in o["placed_at"] for o in orders if o["placed_at"])

    def in_month(o, col="placed_at"):
        return (o[col] or "").startswith(month)

    def qualifies(o):
        c = customers[o["customer_id"]]
        return in_month(o) and not o["deleted_at"] and not deleted(c) and c["region"] == region

    net = sum(o["amount_cents"] - (o["refunded_cents"] or 0) for o in orders if qualifies(o))
    assert abs(_expected(tmp_path) - net / 100) < 1e-9
    good = [o for o in orders if qualifies(o)]
    assert any(o["refunded_cents"] is None for o in good) and any((o["refunded_cents"] or 0) > 0 for o in good)
    assert any(in_month(o) and o["deleted_at"] and customers[o["customer_id"]]["region"] == region and not deleted(customers[o["customer_id"]]) for o in orders)
    assert any(in_month(o) and not o["deleted_at"] and deleted(customers[o["customer_id"]]) and customers[o["customer_id"]]["region"] == region for o in orders)
    last_day = f"{month}-{(datetime(2026, params['month'] + 1, 1) - timedelta(days=1)).day:02d}"
    assert any(o["placed_at"] > f"{last_day} 23:59:59" for o in good)  # left out by BETWEEN ... '<last day> 23:59:59'
    assert any(signup[o["customer_id"]] != region for o in good)  # a customer of the region who signed up elsewhere
    assert any(in_month(o) and not o["deleted_at"] and not deleted(customers[o["customer_id"]]) and customers[o["customer_id"]]["region"] != region and signup[o["customer_id"]] == region for o in orders)
    if level >= 2:
        assert any(not in_month(o, "created_at") for o in good)  # placed in the month, cart opened before it
        assert any(o["placed_at"] is None and in_month(o, "created_at") and customers[o["customer_id"]]["region"] == region for o in orders)
    if level == 3:
        regions_of = Counter((a["customer_id"], a["region"]) for a in addresses)
        assert any(regions_of[(o["customer_id"], region)] > 1 for o in good)  # joining every address double-counts
        assert any(
            in_month(o) and not o["deleted_at"] and customers[o["customer_id"]]["region"] != region and regions_of[(o["customer_id"], region)] and not deleted(customers[o["customer_id"]])
            for o in orders
        )  # a secondary address in the region
        assert any(o["region"] != region for o in good)  # orders.region is the warehouse, not the customer


def _schema(db: bytes) -> dict[str, list[str]]:
    con = sqlite3.connect(":memory:")
    con.deserialize(db)
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
    out = {t: [r[1] for r in con.execute(f"PRAGMA table_info({t})")] for t in tables}
    con.close()
    return out


def test_sqlite_schema_is_the_same_at_every_difficulty():
    """The instruction's rules name `addresses`, `created_at` and `customers.deleted_at` at every
    difficulty, so every difficulty's database has them; only the data differs."""
    fam = FAMILIES["sqlite_query"]
    schemas = {d: _schema(generate_spec(fam, d, 1).spec.files["data/shop.db"]) for d in fam.difficulties}
    assert len({json.dumps(v) for v in schemas.values()}) == 1, schemas
    assert schemas["easy"]["addresses"] == ["id", "customer_id", "region", "city", "is_primary"]
    assert {"created_at", "placed_at", "deleted_at", "region"} <= set(schemas["easy"]["orders"])
    assert {"deleted_at", "region"} <= set(schemas["easy"]["customers"])


def test_sqlite_database_bytes_do_not_carry_the_host_sqlite_version(tmp_path):
    render(FAMILIES["sqlite_query"], "hard", 1, tmp_path)
    data = (tmp_path / "environment" / "files" / "data" / "shop.db").read_bytes()
    assert data[:16] == b"SQLite format 3\x00"
    assert int.from_bytes(data[96:100], "big") == 3046001


# ----------------------------------------------------------------------------- json-aggregate


def _site(rec):
    return rec.get("site") or rec["location"]["site"]


@pytest.mark.parametrize("difficulty,seed", _cases("json_aggregate"))
def test_json_aggregate_traps(tmp_path, difficulty, seed):
    params = render(FAMILIES["json_aggregate"], difficulty, seed, tmp_path)
    raw = _files(tmp_path, "data/readings")
    assert sorted(raw) == sorted(params["files"])
    records = {n: [json.loads(ln) for ln in (gzip.decompress(b) if n.endswith(".gz") else b).decode().splitlines()] for n, b in raw.items()}

    def readings(rec):
        return rec["readings"] if "readings" in rec else [rec]

    def summary(names):
        vals: dict[str, list[Fraction]] = {}
        for n in names:
            for rec in records[n]:
                for r in readings(rec):
                    if r.get("temp_c") is not None:
                        vals.setdefault(_site(rec), []).append(Fraction(str(r["temp_c"])))
        return {s: {"count": len(v), "mean_temp_c": float(round(sum(v) / len(v), 2))} for s, v in vals.items()}

    expected = _expected(tmp_path)
    assert summary(raw) == expected
    for k in range(1, len(raw)):
        for combo in itertools.combinations(raw, k):
            assert summary(combo) != expected, combo
    for n, recs in records.items():  # every file: failed readings both ways, valid singles and valid batches
        rs = [r for rec in recs for r in readings(rec)]
        assert any("temp_c" in r and r["temp_c"] is None for r in rs) and any("temp_c" not in r for r in rs)
        assert any("readings" not in rec and rec.get("temp_c") is not None for rec in recs)
        assert any("readings" in rec and any(r.get("temp_c") is not None for r in rec["readings"]) for rec in recs)
        assert any("site" not in rec for rec in recs) == params["newer"]
        if params["newer"]:
            assert any("site" not in rec and any(r.get("temp_c") is not None for r in readings(rec)) for rec in recs)
    if difficulty == "hard":
        assert any(n.endswith(".gz") for n in raw) and any("/" in n for n in raw)


# ----------------------------------------------------------------------------- dedupe-contacts


def _email(raw: str, tags: bool = True, dots: bool = True, all_domains: bool = False) -> str:
    e = raw.strip().lower()
    if e:
        m = re.fullmatch(r"([^@+]*)((?:\+[^@]*)?)@(.*)", e)
        assert m, raw
        local, tag, domain = m.groups()
        if (domain == "gmail.com" and dots) or all_domains:
            local = local.replace(".", "")
        e = f"{local}{'' if tags else tag}@{domain}"
    return e


def _phone(raw: str, country_code: bool = True) -> str:
    d = re.sub(r"\D", "", raw)
    return d[1:] if country_code and len(d) == 11 and d[0] == "1" else d


def _components(rows, keys) -> int:
    """Connected components of the graph whose nodes are rows and keys, with an edge from each row to each of its keys."""
    adj: dict = {}
    for i, row in enumerate(rows):
        for k in keys(row):
            adj.setdefault(("row", i), set()).add(k)
            adj.setdefault(k, set()).add(("row", i))
    seen: set = set()
    n = 0
    for i in range(len(rows)):
        if ("row", i) in seen:
            continue
        n += 1
        stack = [("row", i)]
        while stack:
            node = stack.pop()
            if node not in seen:
                seen.add(node)
                stack.extend(adj.get(node, ()))
    return n


@pytest.mark.parametrize("difficulty,seed", _cases("dedupe_contacts"))
def test_dedupe_contacts_traps(tmp_path, difficulty, seed):
    params = render(FAMILIES["dedupe_contacts"], difficulty, seed, tmp_path)
    level, variants = params["level"], set(params["variants"])
    files = {k: v.decode() for k, v in _files(tmp_path, "data").items()}
    rows = []
    for name in sorted(files):
        for r in csv.DictReader(io.StringIO(files[name])):
            rows.append((r.get("name", r.get("Full Name")), r.get("email", r.get("E-mail")), r.get("phone", r.get("Phone Number"))))
    assert sorted(files) == (["contacts.csv", "crm_export.csv"] if level == 3 else ["contacts.csv"])

    def count(rows=rows, email=_email, phone=_phone, use=("e", "p"), empty=False):
        def keys(row):
            out = []
            if "e" in use and (row[1].strip() or empty):
                out.append(("e", email(row[1])))
            if "p" in use and (row[2].strip() or empty):
                out.append(("p", phone(row[2])))
            return out

        return _components(rows, keys)

    expected = _expected(tmp_path)
    assert count() == expected == params["people"]  # one group per person built (the instruction's full rules)
    only_needed = lambda e: _email(e, tags="plus-tag" in variants, dots="gmail-dots" in variants)  # noqa: E731
    assert count(email=only_needed, phone=lambda p: _phone(p, "country-code" in variants)) == expected  # the rules a variant needs
    assert count(email=lambda e: e, phone=lambda p: p) > expected  # no normalization
    assert count(phone=lambda p: p) > expected  # emails normalized, phones raw
    assert count(email=lambda e: e) > expected  # phones normalized, emails raw
    assert count(use=("e",)) > expected and count(use=("p",)) > expected  # one key only
    assert len({(_email(e), _phone(p)) for _, e, p in rows}) > expected  # identical (email, phone) pairs
    assert count(empty=True) < expected  # empty values matched to each other
    assert len({n.strip().lower() for n, _, _ in rows}) < expected  # by name
    if "plus-tag" in variants:
        assert count(email=lambda e: _email(e, tags=False)) > expected  # +tag kept
    if "country-code" in variants:
        assert count(phone=lambda p: _phone(p, country_code=False)) > expected  # country code kept
    if "gmail-dots" in variants:
        assert count(email=lambda e: _email(e, all_domains=True)) < expected  # dots dropped for every domain
        assert count(email=lambda e: _email(e, dots=False)) > expected  # gmail dots kept
    if "bridge" in variants:
        seen, greedy = set(), 0  # a row is new unless it matches an earlier row
        for _, e, p in rows:
            ks = {k for k in (("e", _email(e)) if e.strip() else None, ("p", _phone(p)) if p.strip() else None) if k}
            greedy += not (ks & seen)
            seen |= ks
        assert greedy > expected
    if level == 3:
        assert count(rows=rows[: len(files["contacts.csv"].splitlines()) - 1]) < expected  # contacts.csv only


def test_dedupe_contacts_variants_per_level():
    """Easy already has `+tag` emails and `+1` phones; medium and hard add the gmail-dots and
    bridging people."""
    fam = FAMILIES["dedupe_contacts"]
    assert {d: p["variants"] for d, p in fam.difficulties.items()} == {
        "easy": ["plus-tag", "country-code"],
        "medium": ["plus-tag", "country-code", "gmail-dots", "bridge"],
        "hard": ["plus-tag", "country-code", "gmail-dots", "bridge"],
    }
    spec = generate_spec(fam, "easy", 1).spec
    assert {"no-plus-tags", "no-country-code"} <= set(spec.shortcuts) and not {"no-gmail-dots", "over-normalize", "greedy"} & set(spec.shortcuts)


def test_dedupe_contacts_answer_varies_across_seeds():
    """A policy cannot memorize one count per difficulty: the number of people is drawn."""
    fam = FAMILIES["dedupe_contacts"]
    for difficulty in fam.difficulties:
        answers = {generate_spec(fam, difficulty, seed).spec.grader.expected for seed in range(30)}
        assert len(answers) >= 4, (difficulty, answers)


# ----------------------------------------------------------------------------- csv-revenue


def _split_top(texts: list[str]) -> str | None:
    """The top region when every line is split on "," and fields are taken by the header's positions."""
    t: Counter = Counter()
    for text in texts:
        lines = text.splitlines()
        cols = lines[0].split(",")
        for ln in lines[1:]:
            r = dict(zip(cols, ln.split(",")))
            if r["status"] == "completed":
                t[r["region"]] += int(r["quantity"]) * round(float(r["unit_price"]) * 100)
    best = t.most_common(2)
    return best[0][0] if best and (len(best) == 1 or best[0][1] != best[1][1]) else None


def test_csv_revenue_share_of_instances_whose_commas_decide():
    """Every instance quotes product names with commas; in about the level's share of instances
    splitting lines on "," gives a wrong region, and in the others the right one."""
    fam = MODS["csv_revenue"].FAMILY
    for difficulty, p in fam.difficulties.items():
        decided = []
        for seed in range(200):
            spec = generate_spec(fam, difficulty, seed).spec
            texts = [t.decode() for t in file_bytes(spec).values()]
            assert any('"' in t for t in texts)
            decides = spec.params["commas_decide"]
            assert (_split_top(texts) != spec.grader.expected) == decides == ("naive-split" in spec.shortcuts), (difficulty, seed)
            decided.append(decides)
        assert abs(sum(decided) / len(decided) - p["deciding_commas"]) <= 0.1, difficulty
    assert [p["deciding_commas"] for p in fam.difficulties.values()] == [0.35, 0.6, 0.75]


# ----------------------------------------------------------------------------- json-aggregate: numbers by value


def test_json_aggregate_accepts_an_integral_mean_written_as_an_integer():
    """"rounded to 2 decimal places" allows `23` for a mean of 23.0; the grader compares by value."""
    fam = FAMILIES["json_aggregate"]
    for seed in range(1, 400):
        spec = generate_spec(fam, "easy", seed).spec
        expected = spec.grader.expected
        if any(v["mean_temp_c"] == int(v["mean_temp_c"]) for v in expected.values()):
            break
    else:
        pytest.fail("no seed with an integral mean")
    assert spec.grader.key()["numbers"] == "value"
    as_ints = {s: {"count": v["count"], "mean_temp_c": int(v["mean_temp_c"]) if v["mean_temp_c"] == int(v["mean_temp_c"]) else v["mean_temp_c"]} for s, v in expected.items()}
    text = json.dumps(as_ints)
    assert re.search(r'"mean_temp_c": -?\d+[},]', text), text
    files = file_bytes(spec)
    assert grade_in_process(spec, files, {"/app/answer.json": text}) == 1.0
    wrong = {s: dict(v, mean_temp_c=v["mean_temp_c"] + 0.01) for s, v in expected.items()}
    assert grade_in_process(spec, files, {"/app/answer.json": json.dumps(wrong)}) == 0.0


# ----------------------------------------------------------------------------- one instruction per family


def _instruction_for(module: str, params: dict) -> str:
    """The family's one instruction template, filled in with the instance's own values."""
    mod = MODS[module]
    if module == "sqlite_query":
        return mod.INSTRUCTION.format(month=mod.MONTHS[params["month"] - 3], region=params["region"])
    return mod.INSTRUCTION


@pytest.mark.parametrize("module", (*MODULES, "csv_revenue"))
def test_instruction_is_one_template_at_every_difficulty(module):
    """Difficulty comes from the data: every level gets the same instruction text, which states
    every rule that can apply at any level."""
    fam = MODS[module].FAMILY
    for difficulty in fam.difficulties:
        for seed in (1, 2, 3):
            spec = generate_spec(fam, difficulty, seed).spec
            assert spec.instruction == _instruction_for(module, spec.params), (difficulty, seed)


def test_instructions_state_the_rules_of_the_hardest_data():
    """The rules that only the harder levels' data exercises are stated at every level."""
    must = {
        "join_orders": ["leading zeros", "ignoring letter case", "header row naming its columns"],
        "json_aggregate": ["including subdirectories", "`.gz`", "`location.site`"],
        "sqlite_query": ["the month of its `placed_at`", "every order of a deleted customer", "`addresses.is_primary = 1`"],
        "dedupe_contacts": ["`+tag`", "`gmail.com` addresses only", "country code", "header row", "`Ann+News@Example.org`"],
    }
    for module, phrases in must.items():
        text = MODS[module].INSTRUCTION
        for phrase in phrases:
            assert phrase in text, (module, phrase)
