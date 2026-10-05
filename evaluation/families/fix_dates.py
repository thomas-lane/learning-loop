"""fix-dates: fix the date-arithmetic bugs injected into pure date helpers (partial credit).

`/app/dates.py` holds 4 / 5 / 7 small functions on ISO date strings (days in a month, month
end, adding months, ISO week; medium adds business days, hard adds age and ISO weeks per
year), and 1 / 2 / 4 of them carry a bug from a catalog of classic calendar mistakes:
leap-year rules, December and year wrap-around, month-end clamping, ISO week-year
boundaries, weekends and holidays. The functions take every date as an argument and never
read the clock, so the hidden checks are reproducible. The visible tests in
`/app/test_dates.py` expose all / half / a quarter of the bugs; a function with a bug no
visible test exposes gets no visible test. Hidden checks use fresh seed-specific dates
chosen to hit each boundary: century years, 31st-of-month moves, December targets, dates
whose ISO year differs from the calendar year, spans that cross a weekend without landing on
it, weekday holidays inside a span, birthdays a few days away after decades of leap days.

Traps, each declared as a shortcut that must fail:
- `visible-bugs-only` (when some bug has no visible test): fix only the bugs the visible
  tests expose;
- `hardcode-visible`: return the visible tests' expected values for their exact inputs and
  keep the buggy code for everything else;
- `wrong-fix-<function>`: every bug fixed correctly except one, which gets a plausible wrong
  fix from the catalog (e.g. `year % 4 == 0` as the leap rule, `min(day, 28)` as month-end
  clamping, `strftime("%Y-W%V")` for the ISO week, `(on - born).days // 365` for an age).

Expected values come from reference code independent of the module's source (the
`calendar` module, day-by-day loops), and `build` checks that the correct module passes
every hidden check. `build` redraws a visible test until it exposes its function's bug (or,
for a function without a visible bug, passes with the hidden-only bugs still in place).
"""

import calendar
import json
from datetime import date, timedelta

from learning_loop.tasks.runtime.grade import run_checks
from learning_loop.tasks.spec import Checks, Family, Reject, Solution, TaskSpec

MODULE = "dates"
PATH = f"/app/{MODULE}.py"

HEADER = '''"""Date helpers for billing and scheduling.

Dates are ISO 8601 strings, "YYYY-MM-DD", in the proleptic Gregorian calendar. The
functions are pure: every date they use is an argument, never the current date.
"""

from datetime import date, timedelta
'''

CORRECT = {
    "days_in_month": '''def days_in_month(year: int, month: int) -> int:
    """Number of days in `month` (1-12) of `year`. February has 29 days in a leap year: a year
    divisible by 4, except a year divisible by 100 that is not divisible by 400 (2000 was a
    leap year, 1900 was not)."""
    if month == 2:
        leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        return 29 if leap else 28
    return [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]
''',
    "month_end": '''def month_end(day: str) -> str:
    """The last day of the month that contains `day`, e.g. "2026-02-28" for "2026-02-10"."""
    d = date.fromisoformat(day)
    return date(d.year, d.month, days_in_month(d.year, d.month)).isoformat()
''',
    "add_months": '''def add_months(day: str, months: int) -> str:
    """`day` moved by `months` calendar months (a negative number moves back). When the target
    month is too short for the day, the result is that month's last day: "2026-01-31" plus
    one month is "2026-02-28"."""
    d = date.fromisoformat(day)
    year, month0 = divmod(d.year * 12 + d.month - 1 + months, 12)
    month = month0 + 1
    return date(year, month, min(d.day, days_in_month(year, month))).isoformat()
''',
    "iso_week": '''def iso_week(day: str) -> str:
    """The ISO 8601 week of `day` as "YYYY-Www", e.g. "2026-W05". ISO weeks start on Monday
    and week 1 is the week that contains the year's first Thursday, so the first days of
    January can belong to the last week of the previous year, and the last days of December
    to week 1 of the next year; YYYY is the year the week belongs to."""
    year, week, _ = date.fromisoformat(day).isocalendar()
    return f"{year}-W{week:02d}"
''',
    "add_business_days": '''def add_business_days(day: str, n: int, holidays: list[str]) -> str:
    """The date `n` business days after `day`. Business days are Monday to Friday, except
    the dates listed in `holidays`. `day` itself is never counted, so n = 0 returns `day`
    unchanged (even on a weekend). Raises ValueError if n < 0."""
    if n < 0:
        raise ValueError("n must be >= 0")
    d = date.fromisoformat(day)
    off = set(holidays)
    while n > 0:
        d += timedelta(days=1)
        if d.weekday() < 5 and d.isoformat() not in off:
            n -= 1
    return d.isoformat()
''',
    "age": '''def age(born: str, on: str) -> int:
    """Age in completed years on the day `on` of someone born on `born`. The age goes up on
    the birthday; someone born on 29 February has their birthday on 1 March in other years.
    Raises ValueError if `on` is before `born`."""
    b, d = date.fromisoformat(born), date.fromisoformat(on)
    if d < b:
        raise ValueError("on is before born")
    years = d.year - b.year
    if (d.month, d.day) < (b.month, b.day):
        years -= 1
    return years
''',
    "weeks_in_year": '''def weeks_in_year(year: int) -> int:
    """Number of ISO 8601 weeks in `year`, 52 or 53. A year has 53 weeks when 1 January is a
    Thursday, or when it is a leap year and 1 January is a Wednesday."""
    jan1 = date(year, 1, 1).weekday()  # Monday is 0
    leap = days_in_month(year, 2) == 29
    return 53 if jan1 == 3 or (leap and jan1 == 2) else 52
''',
}

_LEAP = "leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)"
_AGE = "    years = d.year - b.year\n    if (d.month, d.day) < (b.month, b.day):\n        years -= 1\n    return years\n"
_ISO = '    year, week, _ = date.fromisoformat(day).isocalendar()\n    return f"{year}-W{week:02d}"\n'

# func -> bug -> {"bug": swaps applied to CORRECT[func], "naive": swaps for a plausible wrong fix, or None}
BUGS = {
    "days_in_month": {
        "every_fourth_year": {
            "bug": [(_LEAP, "leap = year % 4 == 0")],
            "naive": [(_LEAP, "leap = year % 4 == 0 and year % 100 != 0")],
        },
        "no_400_rule": {
            "bug": [(_LEAP, "leap = year % 4 == 0 and year % 100 != 0")],
            "naive": [(_LEAP, "leap = year % 4 == 0")],
        },
    },
    "month_end": {
        "next_month_minus_one": {
            "bug": [
                (
                    "    return date(d.year, d.month, days_in_month(d.year, d.month)).isoformat()\n",
                    "    first_of_next = date(d.year, d.month + 1, 1)\n    return (first_of_next - timedelta(days=1)).isoformat()\n",
                )
            ],
            "naive": [
                (
                    "    return date(d.year, d.month, days_in_month(d.year, d.month)).isoformat()\n",
                    "    first_of_next = date(d.year, d.month % 12 + 1, 1)\n    return (first_of_next - timedelta(days=1)).isoformat()\n",
                )
            ],
        },
    },
    "add_months": {
        "no_clamp": {
            "bug": [("min(d.day, days_in_month(year, month))", "d.day")],
            "naive": [("min(d.day, days_in_month(year, month))", "min(d.day, 28)")],
        },
        "year_wrap": {
            "bug": [
                (
                    "    year, month0 = divmod(d.year * 12 + d.month - 1 + months, 12)\n    month = month0 + 1\n",
                    "    year = d.year + (d.month + months) // 12\n    month = (d.month + months) % 12 or 12\n",
                )
            ],
            "naive": None,
        },
    },
    "iso_week": {
        "calendar_year": {
            "bug": [(_ISO, '    d = date.fromisoformat(day)\n    return f"{d.year}-W{d.isocalendar()[1]:02d}"\n')],
            "naive": [
                (
                    _ISO,
                    (
                        "    d = date.fromisoformat(day)\n    week = d.isocalendar()[1]\n"
                        "    year = d.year - 1 if d.month == 1 and week >= 52 else d.year\n"
                        '    return f"{year}-W{week:02d}"\n'
                    ),
                )
            ],
        },
        "monday_weeks": {
            "bug": [(_ISO, '    return date.fromisoformat(day).strftime("%Y-W%W")\n')],
            "naive": [(_ISO, '    return date.fromisoformat(day).strftime("%Y-W%V")\n')],
        },
    },
    "add_business_days": {
        "calendar_then_roll": {
            "bug": [
                (
                    "    while n > 0:\n        d += timedelta(days=1)\n        if d.weekday() < 5 and d.isoformat() not in off:\n            n -= 1\n",
                    "    d += timedelta(days=n)\n    while d.weekday() >= 5 or d.isoformat() in off:\n        d += timedelta(days=1)\n",
                )
            ],
            "naive": [
                (
                    "    while n > 0:\n        d += timedelta(days=1)\n        if d.weekday() < 5 and d.isoformat() not in off:\n            n -= 1\n",
                    "    d += timedelta(days=n + 2 * (n // 5))\n    while d.weekday() >= 5 or d.isoformat() in off:\n        d += timedelta(days=1)\n",
                )
            ],
        },
        "saturday_counts": {
            "bug": [("if d.weekday() < 5 and", "if d.weekday() < 6 and")],
            "naive": None,
        },
        "ignores_holidays": {
            "bug": [("if d.weekday() < 5 and d.isoformat() not in off:", "if d.weekday() < 5:")],
            "naive": [
                ("if d.weekday() < 5 and d.isoformat() not in off:", "if d.weekday() < 5:"),
                ("    return d.isoformat()\n", "    while d.isoformat() in off or d.weekday() >= 5:\n        d += timedelta(days=1)\n    return d.isoformat()\n"),
            ],
        },
    },
    "age": {
        "year_difference": {
            "bug": [(_AGE, "    return d.year - b.year\n")],
            "naive": [(_AGE, "    return (d - b).days // 365\n")],
        },
        "month_only": {
            "bug": [("if (d.month, d.day) < (b.month, b.day):", "if d.month < b.month:")],
            "naive": None,
        },
    },
    "weeks_in_year": {
        "leap_years": {
            "bug": [("return 53 if jan1 == 3 or (leap and jan1 == 2) else 52", "return 53 if leap else 52")],
            "naive": [("return 53 if jan1 == 3 or (leap and jan1 == 2) else 52", "return 53 if jan1 == 3 else 52")],
        },
        "week_of_december_31": {
            "bug": [
                (
                    "    jan1 = date(year, 1, 1).weekday()  # Monday is 0\n    leap = days_in_month(year, 2) == 29\n    return 53 if jan1 == 3 or (leap and jan1 == 2) else 52\n",
                    "    return date(year, 12, 31).isocalendar()[1]\n",
                )
            ],
            "naive": None,
        },
    },
}


# --------------------------------------------------------------------------- #
# Reference values (independent of CORRECT) and the checks
# --------------------------------------------------------------------------- #


def _is_leap(y):
    return calendar.isleap(y)


def _last(y, m):
    return calendar.monthrange(y, m)[1]


def _plus_months(d, k):
    y, m = d.year + (d.month - 1 + k) // 12, (d.month - 1 + k) % 12 + 1
    return date(y, m, min(d.day, _last(y, m)))


def _business(d, n, off):
    while n:
        d += timedelta(days=1)
        if d.isoweekday() <= 5 and d not in off:
            n -= 1
    return d


def _age(b, d):
    return d.year - b.year - (1 if (d.month, d.day) < (b.month, b.day) else 0)


def _iso(d):
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}"


def _weeks(y):
    return date(y, 12, 28).isocalendar()[1]


def _eq(name, func, args, expected):
    return {"name": name, "func": func, "kind": "equal", "args": args, "expected": expected}


def _raises(name, func, args):
    return {"name": name, "func": func, "kind": "raises", "args": args}


def _years(lo, hi, pred):
    return [y for y in range(lo, hi) if pred(y)]


def _weekday(rng, lo, hi, weekdays):
    """A random date in [lo, hi) whose weekday (Monday 0) is in `weekdays`."""
    while True:
        d = lo + timedelta(days=rng.randrange((hi - lo).days))
        if d.weekday() in weekdays:
            return d


def _cases(rng, func):
    """The hidden checks for `func`, with fresh inputs; the visible tests draw from the same kinds."""
    lo, hi = date(1990, 1, 1), date(2060, 1, 1)
    if func == "days_in_month":
        leap = rng.choice(_years(1904, 2097, lambda y: y % 4 == 0 and y % 100 != 0))
        common = rng.choice(_years(1950, 2080, lambda y: y % 4 != 0))
        return [
            _eq("days_in_month_feb_leap", func, [leap, 2], 29),
            _eq("days_in_month_feb_century", func, [rng.choice([1700, 1800, 1900, 2100, 2200, 2300]), 2], 28),
            _eq("days_in_month_feb_400", func, [rng.choice([1600, 2000, 2400]), 2], 29),
            _eq("days_in_month_feb_common", func, [common, 2], 28),
            _eq("days_in_month_30", func, [common, rng.choice([4, 6, 9, 11])], 30),
            _eq("days_in_month_december", func, [leap, 12], 31),
        ]
    if func == "month_end":
        leap = rng.choice(_years(1904, 2097, lambda y: y % 4 == 0 and y % 100 != 0))
        century = rng.choice([1900, 2100, 2200])
        y = rng.randint(1990, 2059)
        m30 = rng.choice([4, 6, 9, 11])
        return [
            _eq("month_end_feb_leap", func, [date(leap, 2, rng.randint(1, 28)).isoformat()], date(leap, 2, 29).isoformat()),
            _eq("month_end_feb_century", func, [date(century, 2, rng.randint(1, 28)).isoformat()], date(century, 2, 28).isoformat()),
            _eq("month_end_december", func, [date(y, 12, rng.randint(1, 30)).isoformat()], date(y, 12, 31).isoformat()),
            _eq("month_end_30_days", func, [date(y, m30, rng.randint(1, 29)).isoformat()], date(y, m30, 30).isoformat()),
        ]
    if func == "add_months":
        y = rng.randint(1990, 2058)
        # the 31st of a month, moved to a shorter month and to a later 31-day month
        m31 = rng.choice([1, 3, 5, 8, 10])
        short = date(y, m31, 31)
        to31 = {1: 2, 3: 2, 5: 2, 8: 4, 10: 2}[m31]
        start = date(y, rng.randint(1, 11), rng.randint(1, 28))
        leap = rng.choice(_years(1992, 2057, lambda yy: yy % 4 == 0))
        back = date(y, rng.choice([1, 3]), rng.choice([rng.randint(1, 28), 31]))
        k = -(back.month + rng.randint(0, 6))
        far = rng.randint(13, 40)
        return [
            _eq("add_months_clamp", func, [short.isoformat(), 1], _plus_months(short, 1).isoformat()),
            _eq("add_months_to_31_day_month", func, [short.isoformat(), to31], _plus_months(short, to31).isoformat()),
            _eq("add_months_into_december", func, [start.isoformat(), 12 - start.month], _plus_months(start, 12 - start.month).isoformat()),
            _eq("add_months_february_leap", func, [date(leap, 1, rng.choice([29, 30, 31])).isoformat(), 1], date(leap, 2, 29).isoformat()),
            _eq("add_months_backwards", func, [back.isoformat(), k], _plus_months(back, k).isoformat()),
            _eq("add_months_years", func, [start.isoformat(), far], _plus_months(start, far).isoformat()),
        ]
    if func == "iso_week":
        mid_year = rng.choice(_years(1990, 2060, lambda yy: date(yy, 1, 1).weekday() in (1, 2, 3)))
        late = rng.choice(_years(1990, 2060, lambda yy: date(yy, 12, 31).weekday() in (0, 1, 2)))
        dec31 = date(late, 12, 31)
        late_day = dec31 - timedelta(days=rng.randint(0, dec31.weekday()))
        early = rng.choice(_years(1990, 2060, lambda yy: date(yy, 1, 1).weekday() in (4, 5, 6)))
        jan1 = date(early, 1, 1)
        early_day = jan1 + timedelta(days=rng.randint(0, 6 - jan1.weekday()))
        small = date(rng.randint(1990, 2059), 2, rng.randint(1, 20))
        mid = date(mid_year, rng.randint(3, 10), rng.randint(1, 28))
        return [
            _eq("iso_week_mid_year", func, [mid.isoformat()], _iso(mid)),
            _eq("iso_week_late_december", func, [late_day.isoformat()], _iso(late_day)),
            _eq("iso_week_early_january", func, [early_day.isoformat()], _iso(early_day)),
            _eq("iso_week_padded", func, [small.isoformat()], _iso(small)),
        ]
    if func == "add_business_days":
        fri = _weekday(rng, lo, hi, (4,))
        cross = _weekday(rng, lo, hi, (3, 4))
        n_cross = rng.randint(3, 4)
        start = _weekday(rng, lo, hi, (0, 1))
        n_hol = rng.randint(5, 8)
        plain_end = _business(start, n_hol, set())
        inside = [start + timedelta(days=i) for i in range(1, (plain_end - start).days) if (start + timedelta(days=i)).weekday() < 5]
        hol = rng.choice(inside)
        weekend_hol = _weekday(rng, start, start + timedelta(days=14), (5, 6))
        holidays = sorted({hol.isoformat(), weekend_hol.isoformat()})
        sat = _weekday(rng, lo, hi, (5,))
        return [
            _eq("business_days_friday_plus_one", func, [fri.isoformat(), 1, []], _business(fri, 1, set()).isoformat()),
            _eq("business_days_across_weekend", func, [cross.isoformat(), n_cross, []], _business(cross, n_cross, set()).isoformat()),
            _eq("business_days_holiday_inside", func, [start.isoformat(), n_hol, holidays], _business(start, n_hol, {hol, weekend_hol}).isoformat()),
            _eq("business_days_zero_on_weekend", func, [sat.isoformat(), 0, []], sat.isoformat()),
            _raises("business_days_negative", func, [start.isoformat(), -rng.randint(1, 5), []]),
        ]
    if func == "age":
        born = date(rng.randint(1940, 1990), rng.randint(1, 12), rng.randint(10, 28))
        years = rng.randint(30, 60)
        birthday = born.replace(year=born.year + years)
        before = birthday - timedelta(days=rng.randint(1, 3))
        same_month = birthday.replace(day=born.day - rng.randint(1, 9))
        leap_born = date(rng.choice(_years(1940, 1992, lambda yy: yy % 4 == 0)), 2, 29)
        later = rng.choice(_years(leap_born.year + 20, leap_born.year + 60, lambda yy: not _is_leap(yy)))
        feb28 = date(later, 2, 28)
        return [
            _eq("age_days_before_birthday", func, [born.isoformat(), before.isoformat()], _age(born, before)),
            _eq("age_on_birthday", func, [born.isoformat(), birthday.isoformat()], years),
            _eq("age_earlier_in_birth_month", func, [born.isoformat(), same_month.isoformat()], _age(born, same_month)),
            _eq("age_leap_day_birthday", func, [leap_born.isoformat(), feb28.isoformat()], _age(leap_born, feb28)),
            _raises("age_on_before_born", func, [born.isoformat(), (born - timedelta(days=rng.randint(1, 400))).isoformat()]),
        ]
    if func == "weeks_in_year":
        thursday = rng.choice(_years(1950, 2100, lambda yy: date(yy, 1, 1).weekday() == 3 and not _is_leap(yy)))
        leap_wed = rng.choice(_years(1950, 2100, lambda yy: date(yy, 1, 1).weekday() == 2 and _is_leap(yy)))
        leap_52 = rng.choice(_years(1950, 2100, lambda yy: date(yy, 1, 1).weekday() not in (2, 3) and _is_leap(yy)))
        w01 = rng.choice(_years(1950, 2100, lambda yy: date(yy, 12, 31).weekday() in (0, 1, 2)))
        return [_eq(f"weeks_in_year_{label}", func, [y], _weeks(y)) for label, y in (("thursday_start", thursday), ("leap_wednesday_start", leap_wed), ("leap_52", leap_52), ("december_31_in_week_1", w01))]
    raise KeyError(func)


# --------------------------------------------------------------------------- #
# The fix-code engine: buggy module, visible tests, shortcuts
# --------------------------------------------------------------------------- #


def _apply(func, swaps):
    src = CORRECT[func]
    for old, new in swaps:
        assert src.count(old) == 1, (func, old)
        src = src.replace(old, new)
    return src


def _module(funcs, chosen):
    """The module with `chosen` {func: source} in place of the correct functions."""
    return HEADER + "".join("\n\n" + chosen.get(f, CORRECT[f]) for f in funcs)


def _exec(src):
    ns = {}
    exec(compile(src, "<generated>", "exec"), ns)  # noqa: S102 - our own reference/buggy sources
    return ns


def _canonical(c):
    """`c` with its arguments as the verifier sees them: key.json is written with sorted keys,
    so dict arguments arrive in sorted key order."""
    return dict(c, args=json.loads(json.dumps(c["args"], sort_keys=True)))


def _passes(lib, checks):
    return all(run_checks(lib, checks).values())


def _test_function(c):
    call = f"{c['func']}({', '.join(repr(a) for a in c['args'])})"
    if c["kind"] == "equal":
        body = f"    got = {call}\n    assert got == {c['expected']!r}, f\"got {{got!r}}\"\n"
    elif c["kind"] == "raises":
        exc = c.get("exception", "ValueError")
        body = f"    try:\n        {call}\n    except {exc}:\n        return\n    raise AssertionError(\"expected {exc}\")\n"
    else:  # no_mutation
        rest = "".join(", " + repr(a) for a in c["args"][1:])
        body = f"    arg = {c['args'][0]!r}\n    {c['func']}(arg{rest})\n    assert arg == {c['args'][0]!r}, \"the argument was modified\"\n"
    return f"def test_{c['func']}():\n{body}"


TEST_RUNNER = """

if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {e!r}")
    raise SystemExit(1 if failed else 0)
"""


def _hardcoded(src, visible):
    """`src` plus wrappers that answer the given visible tests from their expectations."""
    for c in visible:
        f = c["func"]
        if c["kind"] == "equal":
            then = f"return {c['expected']!r}"
        elif c["kind"] == "raises":
            then = f"raise {c.get('exception', 'ValueError')}(\"invalid input\")"
        else:
            continue
        src += f"\n\n_{f}_original = {f}\n\n\ndef {f}(*args):\n    if list(args) == {c['args']!r}:\n        {then}\n    return _{f}_original(*args)\n"
    return src


def _write(source, then=""):
    return f"cat > {PATH} <<'PY'\n{source}PY\n{then}"


def _solution(source, then=""):
    return Solution(_write(source, then), lambda files, s=source: {PATH: s})


def build_fix(ctx, instruction, extra_imports=()):
    rng, p = ctx.rng, ctx.params
    funcs = p["functions"]
    bugged = sorted(rng.sample(funcs, p["n_bugs"]))
    bugs = {f: rng.choice(sorted(BUGS[f])) for f in bugged}
    if not any(BUGS[f][b]["naive"] for f, b in bugs.items()):
        raise Reject("no injected bug has a catalogued wrong fix")
    visible_bugs = sorted(rng.sample(bugged, max(1, round(len(bugged) * p["visible_bug_tests"]))))
    bug_src = {f: _apply(f, BUGS[f][b]["bug"]) for f, b in bugs.items()}
    hidden_only = {f: s for f, s in bug_src.items() if f not in visible_bugs}
    ref, buggy, partial = _exec(_module(funcs, {})), _exec(_module(funcs, bug_src)), _exec(_module(funcs, hidden_only))

    visible = []
    for f in funcs:
        if f in bugs and f not in visible_bugs:
            continue  # a hidden bug gets no visible test
        for _ in range(100):
            c = _canonical(dict(rng.choice(_cases(rng, f)), name=f"visible_{f}"))
            if _passes(ref, [c]) and _passes(partial, [c]) and (f not in visible_bugs or not _passes(buggy, [c])):
                visible.append(c)
                break
        else:
            raise Reject(f"no visible test for {f}")

    seen = [(c["func"], c["args"]) for c in visible]
    for _ in range(100):
        checks = [_canonical(c) for f in funcs for c in _cases(rng, f)]
        if not any((c["func"], c["args"]) in seen for c in checks):
            break  # hidden inputs differ from every visible test's, so hard-coding those cannot pass
    else:
        raise Reject("hidden checks keep repeating a visible test")
    failed = [c["name"] for c in checks if not _passes(ref, [c])]
    if failed:
        raise RuntimeError(f"{ctx.family}: the correct module fails hidden checks {failed}")

    names = ", ".join(funcs + list(extra_imports))
    tests = "\n\n".join(_test_function(c) for c in visible)
    test_src = f'"""Run with: python3 test_{MODULE}.py"""\n\nfrom {MODULE} import {names}  # noqa: F401\n\n\n{tests}' + TEST_RUNNER
    fixed = _module(funcs, {})
    shortcuts = {"hardcode-visible": _solution(_hardcoded(_module(funcs, bug_src), [c for c in visible if c["func"] in visible_bugs]))}
    if hidden_only:
        shortcuts["visible-bugs-only"] = _solution(_module(funcs, hidden_only))
    for f, b in bugs.items():
        if BUGS[f][b]["naive"]:
            shortcuts[f"wrong-fix-{f}"] = _solution(_module(funcs, {f: _apply(f, BUGS[f][b]["naive"])}))
    listed = ", ".join(f"`{f}`" for f in funcs)
    return TaskSpec(
        instruction=instruction.format(functions=listed, hint=p["hint"]),
        files={f"{MODULE}.py": _module(funcs, bug_src), f"test_{MODULE}.py": test_src},
        grader=Checks(PATH, MODULE, tuple(checks)),
        oracle=_solution(fixed, f"cd /app && python3 test_{MODULE}.py\n"),
        shortcuts=shortcuts,
        params={"functions": funcs, "bugs": bugs, "visible_bug_tests": visible_bugs, "n_checks": len(checks)},
    )


INSTRUCTION = """The date helpers in `/app/dates.py` ({functions}) are used for billing and scheduling, and some of their results are wrong around month ends, leap years, year boundaries and weekends. {hint}

Fix `dates.py` so that every function behaves exactly as its docstring describes. Keep the function names and signatures, and use only the Python standard library; the functions must keep taking every date as an argument (never the current date). You may add tests to `/app/test_dates.py` (run it with `python3 test_dates.py`), but do not change what the existing tests assert.
"""

HINTS = {
    "easy": "One function has a bug, and a test in `/app/test_dates.py` fails because of it.",
    "medium": "Some tests in `/app/test_dates.py` fail, and some bugs may not be covered by those tests at all.",
    "hard": "Several functions have bugs; the tests in `/app/test_dates.py` catch only a few of them.",
}


def build(ctx):
    return build_fix(ctx, INSTRUCTION)


FAMILY = Family(
    name="fix-dates",
    version=1,
    cluster="fix-code",
    category="debugging",
    skills=("python", "debugging", "dates", "reading-docs"),
    difficulties={
        "easy": {"functions": ["days_in_month", "month_end", "add_months", "iso_week"], "n_bugs": 1, "visible_bug_tests": 1.0, "hint": HINTS["easy"]},
        "medium": {
            "functions": ["days_in_month", "month_end", "add_months", "iso_week", "add_business_days"],
            "n_bugs": 2,
            "visible_bug_tests": 0.5,
            "hint": HINTS["medium"],
        },
        "hard": {
            "functions": ["days_in_month", "month_end", "add_months", "iso_week", "add_business_days", "age", "weeks_in_year"],
            "n_bugs": 4,
            "visible_bug_tests": 0.25,
            "hint": HINTS["hard"],
        },
    },
    build=build,
)
