"""The tabular-data families join-orders, sqlite-query, json-aggregate and dedupe-contacts:
deterministic renders, and their traps re-derived from the rendered files, independently of
each family's own solution models."""

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
from fractions import Fraction
from pathlib import Path

import pytest

from learning_loop.core.config import REPO_ROOT
from learning_loop.tasks.render import render

MODULES = ("join_orders", "sqlite_query", "json_aggregate", "dedupe_contacts")


def _load(module: str):
    path = REPO_ROOT / "evaluation" / "families" / f"{module}.py"
    spec = importlib.util.spec_from_file_location(f"family_{module}", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FAMILY


FAMILIES = {m: _load(m) for m in MODULES}
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
    if params["reorder"]:
        assert len({files[n].splitlines()[0] for n in order_files}) > 1


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
    addresses = [dict(r) for r in con.execute("SELECT * FROM addresses")] if level == 3 else []
    con.close()
    if level == 3:
        primary = {a["customer_id"]: a["region"] for a in addresses if a["is_primary"]}
        assert len(primary) == len(customers) == len({a["customer_id"] for a in addresses})
        for c in customers.values():
            c["region"] = primary[c["id"]]
    deleted = lambda c: bool(c.get("deleted_at"))  # noqa: E731

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
    if level >= 2:
        assert any(in_month(o) and not o["deleted_at"] and deleted(customers[o["customer_id"]]) and customers[o["customer_id"]]["region"] == region for o in orders)
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


def _email(raw: str, level: int, all_domains: bool = False) -> str:
    e = raw.strip().lower()
    if level >= 2 and e:
        m = re.fullmatch(r"([^@+]*)(?:\+[^@]*)?@(.*)", e)
        assert m, raw
        local, domain = m.groups()
        if domain == "gmail.com" or all_domains:
            local = local.replace(".", "")
        e = f"{local}@{domain}"
    return e


def _phone(raw: str, level: int) -> str:
    d = re.sub(r"\D", "", raw)
    return d[1:] if level >= 2 and len(d) == 11 and d[0] == "1" else d


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
    level = params["level"]
    files = {k: v.decode() for k, v in _files(tmp_path, "data").items()}
    rows = []
    for name in sorted(files):
        for r in csv.DictReader(io.StringIO(files[name])):
            rows.append((r.get("name", r.get("Full Name")), r.get("email", r.get("E-mail")), r.get("phone", r.get("Phone Number"))))
    assert sorted(files) == (["contacts.csv", "crm_export.csv"] if level == 3 else ["contacts.csv"])

    def count(rows=rows, email=lambda e: _email(e, level), phone=lambda p: _phone(p, level), use=("e", "p"), empty=False):
        def keys(row):
            out = []
            if "e" in use and (row[1].strip() or empty):
                out.append(("e", email(row[1])))
            if "p" in use and (row[2].strip() or empty):
                out.append(("p", phone(row[2])))
            return out

        return _components(rows, keys)

    expected = _expected(tmp_path)
    assert count() == expected == params["people"]
    assert count(email=lambda e: e, phone=lambda p: p) > expected  # no normalization
    assert count(phone=lambda p: p) > expected  # emails normalized, phones raw
    assert count(email=lambda e: e) > expected  # phones normalized, emails raw
    assert count(use=("e",)) > expected and count(use=("p",)) > expected  # one key only
    assert len({(_email(e, level), _phone(p, level)) for _, e, p in rows}) > expected  # identical (email, phone) pairs
    assert count(empty=True) < expected  # empty values matched to each other
    assert len({n.strip().lower() for n, _, _ in rows}) < expected  # by name
    if level >= 2:
        assert count(email=lambda e: _email(e, level, all_domains=True)) < expected  # dots dropped for every domain
        assert count(email=lambda e: _email(e, 1)) > expected  # no +tag or gmail rule
        assert count(phone=lambda p: re.sub(r"\D", "", p)) > expected  # country code kept
        seen, greedy = set(), 0  # a row is new unless it matches an earlier row
        for _, e, p in rows:
            ks = {k for k in (("e", _email(e, level)) if e.strip() else None, ("p", _phone(p, level)) if p.strip() else None) if k}
            greedy += not (ks & seen)
            seen |= ks
        assert greedy > expected
    if level == 3:
        assert count(rows=rows[: len(files["contacts.csv"].splitlines()) - 1]) < expected  # contacts.csv only
