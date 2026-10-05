"""format-converter: write a script that converts one text format into another.

The seed picks one of two conversions, each specified exactly in the instruction and shown
by an example pair in /app/examples/:

- `kv-csv`: a key=value log (bare or double-quoted values) becomes a CSV file with a given
  header, minimally quoted (a field with a comma or a double quote is quoted, inner quotes
  doubled).
- `fixed-jsonl`: a fixed-width export (1-based inclusive column ranges, text and integer
  fields) becomes JSON Lines in exactly the form `json.dumps(obj)` writes by default.

The agent writes /app/convert.py; the `Commands` grader runs `python3 convert.py FILE` on
hidden inputs (plus an empty file) and compares stdout and the exit code. Every hidden input
contains every trap of its difficulty, so each wrong method fails every non-empty check.

Traps (declared shortcuts; each is a full script an agent might write):
kv-csv: `hardcode-examples` (prints the example's CSV), `naive-split` (splits on spaces, so
quoted values with spaces break), `no-csv-quoting` (joins fields with commas); medium/hard add
`\\"` and `\\\\` escapes (`keeps-backslashes`: a regex that finds the quoted value but does not
unescape it) and comment/blank lines (`no-skip-comments`); hard adds bare values with `'`, `\\`
and `=`, which `shlex.split` misreads (`shlex-split`).
fixed-jsonl: `hardcode-examples`, `off-by-one-columns` (slices [start-1:end-1]),
`hand-built-json` (no escaping of the planted `"` in text); medium/hard add empty integer
fields (`empty-int-as-zero`) and comment lines (`no-skip-comments`); hard adds non-ASCII text
(`ensure-ascii-false`: `json.dumps(..., ensure_ascii=False)`), backslashes and lines whose
trailing empty fields are cut off.
"""

import csv
import io
import json
from datetime import datetime, timedelta

from learning_loop.tasks.spec import Commands, Family, Reject, Solution, TaskSpec

N_HIDDEN = 3

# --------------------------------------------------------------------------- #
# kv-csv
# --------------------------------------------------------------------------- #

KV_INSTRUCTION = """`/app/examples/sample.log` is an example of our services' key=value log format, and `/app/examples/sample.csv` is the CSV it should become.

Write a converter `/app/convert.py` such that `python3 /app/convert.py LOGFILE` prints the CSV for LOGFILE to standard output. It will be run on other log files in this format.

Input format:
- One record per line: `key=value` pairs separated by one or more spaces. Keys consist of lowercase letters, digits and underscores.
- A value is either bare or quoted. {bare} A quoted value starts with `"` and ends at the next {unescaped}`"`; it may contain spaces and commas{escapes}. The enclosing quotes are not part of the value. A value may be empty (`key=` or `key=""`).
{skip}
Output format:
- The first line is the header `{header}`. Then comes one row per record, in input order, with the record's values for these keys in this order. A missing key gives an empty field; other keys are ignored.
- Fields are separated by commas. A field that contains a comma or a double quote is enclosed in double quotes, and each double quote inside it is doubled; other fields are written as they are. Every line, including the last, ends with a newline.
"""

KV_BARE = {
    False: "A bare value runs up to the next space; it contains no spaces, double quotes or backslashes.",
    True: "A bare value runs up to the next space and may contain any character except a space and a double quote (for example `'`, `\\` or `=`).",
}
KV_SKIP = "- Blank lines and lines starting with `#` are not records; skip them.\n"

KV_PARSE = r'''def parse(line):
    """The key=value pairs of one line, as a dict."""
    record = {}
    i, n = 0, len(line)
    while i < n:
        if line[i] == " ":
            i += 1
            continue
        eq = line.index("=", i)
        key = line[i:eq]
        i = eq + 1
        if i < n and line[i] == '"':
            i += 1
            value = []
            while line[i] != '"':
                if line[i] == "\\":
                    i += 1
                value.append(line[i])
                i += 1
            i += 1
            record[key] = "".join(value)
        else:
            end = line.find(" ", i)
            end = n if end < 0 else end
            record[key] = line[i:end]
            i = end
    return record
'''
KV_PARSE_NAIVE = '''def parse(line):
    record = {}
    for token in line.split():
        if "=" in token:
            key, value = token.split("=", 1)
            record[key] = value.strip('"')
    return record
'''
KV_PARSE_REGEX = r'''PAIR = re.compile(r'(\w+)=("(?:[^"\\]|\\.)*"|\S*)')


def parse(line):
    record = {}
    for key, value in PAIR.findall(line):
        if value.startswith('"'):
            value = value[1:-1]
        record[key] = value
    return record
'''
KV_PARSE_SHLEX = '''def parse(line):
    record = {}
    for token in shlex.split(line):
        key, value = token.split("=", 1)
        record[key] = value
    return record
'''
KV_MAIN = '''

def main(path):
    out = csv.writer(sys.stdout, lineterminator="\\n")
    out.writerow(COLUMNS)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\\n")
            if not line.strip() or line.startswith("#"):
                continue
            record = parse(line)
            out.writerow([record.get(c, "") for c in COLUMNS])


main(sys.argv[1])
'''
KV_MAIN_NO_SKIP = KV_MAIN.replace('            if not line.strip() or line.startswith("#"):\n                continue\n', "")
KV_MAIN_JOIN = '''

def main(path):
    print(",".join(COLUMNS))
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\\n")
            if not line.strip() or line.startswith("#"):
                continue
            record = parse(line)
            print(",".join(record.get(c, "") for c in COLUMNS))


main(sys.argv[1])
'''

USERS = ["ann", "bo", "chen", "dara", "eli", "femi", "gus", "hana", "ivo", "juno"]
FULL_NAMES = ["Ann Lee", "Bo Diaz", "Lee, Ann", "Chen Wu", "Dara Okafor"]
ODD_USERS = ["o'neil", "d'arcy", "CORP\\femi", "it's-me"]
LEVELS = ["INFO", "WARN", "ERROR", "DEBUG"]
ACTIONS = ["login", "upload", "export", "delete", "sync", "retry"]
MSG_WORDS = ["disk", "full", "retrying", "upload", "of", "file", "failed", "after", "timeout", "user", "cache", "warm", "queue", "drained"]
PATHS = ["/api/items", "/api/users", "/login", "/static/app.js", "/export"]


def _kv_value(rng, key, level):
    """(raw text as written in the log, the value) for `key`."""
    if key == "time":
        return None
    if key == "level":
        v = rng.choice(LEVELS)
        return v, v
    if key == "pid":
        v = str(rng.randint(100, 32000))
        return v, v
    if key == "status":
        v = str(rng.choice([200, 201, 204, 400, 404, 500, 503]))
        return v, v
    if key == "action":
        v = rng.choice(ACTIONS)
        return v, v
    if key == "user":
        r = rng.random()
        if r < 0.25:
            v = rng.choice(FULL_NAMES)
            return f'"{v}"', v
        v = rng.choice(USERS)
        return v, v
    if key == "path":
        v = rng.choice(PATHS)
        if level == "hard" and rng.random() < 0.4:
            v += f"?id={rng.randint(1, 99)}"
        return v, v
    if key == "msg":
        words = rng.sample(MSG_WORDS, rng.randint(2, 5))
        if rng.random() < 0.4:
            words[rng.randrange(len(words))] += ","
        v = " ".join(words)
        if level != "easy" and rng.random() < 0.25:
            q = rng.choice(MSG_WORDS)
            v += f' "{q}"'
        if level != "easy" and rng.random() < 0.15:
            v += " C:\\tmp"
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"', v
    raise KeyError(key)


def _kv_line(rng, columns, extras, ts, level, plant=None):
    """One log line and its expected record. `plant` forces one trap into the line."""
    pairs, record = [("time", ts)], {"time": ts}
    keys = [k for k in columns if k != "time"] + extras
    for k in keys:
        if k in columns and k != "msg" and rng.random() < 0.12:
            continue  # a missing key
        raw, v = _kv_value(rng, k, level)
        pairs.append((k, raw))
        record[k] = v
    if plant == "comma":
        msg = f"{rng.choice(MSG_WORDS)}, {rng.choice(MSG_WORDS)} {rng.choice(MSG_WORDS)}"
        pairs = [(k, r) for k, r in pairs if k != "msg"] + [("msg", f'"{msg}"')]
        record["msg"] = msg
    elif plant == "escape":
        q = rng.choice(MSG_WORDS)
        msg = f'said "{q}" twice'
        pairs = [(k, r) for k, r in pairs if k != "msg"] + [("msg", '"said \\"' + q + '\\" twice"')]
        record["msg"] = msg
    elif plant == "backslash":
        msg = f"copied to D:\\{rng.choice(MSG_WORDS)}"
        pairs = [(k, r) for k, r in pairs if k != "msg"] + [("msg", '"' + msg.replace("\\", "\\\\") + '"')]
        record["msg"] = msg
    elif plant == "odd_bare":
        v = rng.choice(ODD_USERS)
        pairs = [(k, r) for k, r in pairs if k != "user"] + [("user", v)]
        record["user"] = v
    elif plant == "empty":
        pairs = [(k, r) for k, r in pairs if k != "user"] + [("user", rng.choice(["", '""']))]
        record["user"] = ""
    first, rest = pairs[0], pairs[1:]
    rng.shuffle(rest)
    sep = lambda: " " * rng.choice([1, 1, 1, 2])  # noqa: E731
    line = first[0] + "=" + first[1]
    for k, raw in rest:
        line += sep() + k + "=" + raw
    return line, {k: v for k, v in record.items() if k in columns}


def _kv_file(rng, p, columns, extras, start, plants):
    lines, records = [], []
    ts = start
    n = rng.randint(10, 16)
    plants = list(plants) + [None] * (n - len(plants))
    rng.shuffle(plants)
    for plant in plants:
        ts += timedelta(seconds=rng.randint(1, 300))
        line, rec = _kv_line(rng, columns, extras, ts.strftime("%Y-%m-%dT%H:%M:%SZ"), p["level"], plant)
        lines.append(line)
        records.append(rec)
    if p["comments"]:
        for text in ["# rotated by logrotate", "", "# host=web-2 restarted", "   "][: rng.randint(3, 4)]:
            lines.insert(rng.randint(0, len(lines)), text)
    return "\n".join(lines) + "\n", records


def _csv(columns, records):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(columns)
    for r in records:
        w.writerow([r.get(c, "") for c in columns])
    return buf.getvalue()


def _kv_scripts(p, columns, sample_csv):
    head = f"COLUMNS = {columns!r}\n\n\n"
    scripts = {
        "oracle": "import csv\nimport sys\n\n" + head + KV_PARSE + KV_MAIN,
        "naive-split": "import csv\nimport sys\n\n" + head + KV_PARSE_NAIVE + KV_MAIN,
        "no-csv-quoting": "import sys\n\n" + head + KV_PARSE + KV_MAIN_JOIN,
        "hardcode-examples": f"import sys\n\nSAMPLE = {sample_csv!r}\n\nsys.stdout.write(SAMPLE)\n",
    }
    if p["level"] != "easy":
        scripts["keeps-backslashes"] = "import csv\nimport re\nimport sys\n\n" + head + KV_PARSE_REGEX + KV_MAIN
        scripts["no-skip-comments"] = "import csv\nimport sys\n\n" + head + KV_PARSE + KV_MAIN_NO_SKIP
    if p["level"] == "hard":
        scripts["shlex-split"] = "import csv\nimport shlex\nimport sys\n\n" + head + KV_PARSE_SHLEX + KV_MAIN
    return scripts


def _build_kv(rng, p):
    optional = ["level", "status", "action", "path"]
    chosen = rng.sample(optional, p["n_columns"] - 3)
    columns = ["time"] + rng.sample(["user", "msg"] + chosen, p["n_columns"] - 1)
    extras = [k for k in optional if k not in chosen] + ["pid"]
    start = datetime(2026, 9, 1) + timedelta(days=rng.randint(0, 25), seconds=rng.randint(0, 80000))
    plants = ["comma", "comma", "empty"] + (["escape", "escape", "backslash"] if p["level"] != "easy" else []) + (["odd_bare", "odd_bare"] if p["level"] == "hard" else [])
    sample_log, sample_recs = _kv_file(rng, p, columns, extras, start, plants)
    inputs = [_kv_file(rng, p, columns, extras, start + timedelta(days=i + 1), plants) for i in range(N_HIDDEN)]
    sample_csv = _csv(columns, sample_recs)
    return {
        "instruction": KV_INSTRUCTION.format(
            bare=KV_BARE[p["level"] == "hard"],
            unescaped="unescaped " if p["level"] != "easy" else "",
            escapes=", and inside it `\\\"` stands for `\"` and `\\\\` for `\\`" if p["level"] != "easy" else ", but no double quotes or backslashes",
            skip=KV_SKIP if p["comments"] else "",
            header=",".join(columns),
        ),
        "examples": {"examples/sample.log": sample_log, "examples/sample.csv": sample_csv},
        "cases": [(f"input_{i + 1}.log", text, _csv(columns, recs)) for i, (text, recs) in enumerate(inputs)] + [("empty.log", "", _csv(columns, []))],
        "scripts": _kv_scripts(p, columns, sample_csv),
        "params": {"conversion": "kv-csv", "columns": columns},
    }


# --------------------------------------------------------------------------- #
# fixed-jsonl
# --------------------------------------------------------------------------- #

FX_INSTRUCTION = """`/app/examples/sample.txt` is an export in a fixed-width format, and `/app/examples/sample.jsonl` is the JSON Lines file it should become.

Write a converter `/app/convert.py` such that `python3 /app/convert.py FILE` prints the JSON Lines for FILE to standard output. It will be run on other files in this format.

Input format: one record per line. Positions count characters from 1, and a range includes both ends.

| field | positions | type |
|---|---|---|
{table}

- A text field's value is its content without leading and trailing spaces (possibly the empty string).
- An integer field is right-aligned and may have a minus sign{zeros}.{empty_int}
{short}{skip}
Output: one JSON object per record, one per line, in input order, with the fields above as keys in the table's order. Write each object exactly as Python's `json.dumps(obj)` writes it with its default settings: `", "` between members, `": "` after each key, and every non-ASCII character as a `\\uXXXX` escape. Every line, including the last, ends with a newline.
"""

FX_PARSE = '''def parse(line):
    record = {}
    for name, start, end, kind in FIELDS:
        raw = line[start - 1 : end].strip()
        if kind == "int":
            record[name] = int(raw) if raw else None
        else:
            record[name] = raw
    return record
'''
FX_MAIN = '''

def main(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\\n")
            if not line.strip() or line.startswith("#"):
                continue
            print(json.dumps(parse(line)))


main(sys.argv[1])
'''
FX_DUMP_BY_HAND = '''

def dump(record):
    parts = []
    for key, value in record.items():
        if value is None:
            text = "null"
        elif isinstance(value, int):
            text = str(value)
        else:
            text = '"' + value + '"'
        parts.append('"' + key + '": ' + text)
    return "{" + ", ".join(parts) + "}"
'''

NAMES = ["Ada Park", "Ben Ortiz", "Cara Nolan", "Dev Shah", "Eve Moss", "Finn Hale", "Gia Russo", "Hugo Lind", "Iris Bell", "Jon Reyes"]
QUOTED_NAMES = ['Robert "Bob" Ray', 'Kate "KJ" Jones', 'Li "Lee" Ming', 'Sam "Ace" Cole']
BACKSLASH_NAMES = ["CORP\\jdoe", "LAB\\mkent", "OPS\\rlee"]
UNICODE_NAMES = ["José Núñez", "Zoë Brandt", "Søren Holm", "Ana Lúcia", "Chloé Dubois", "Jürgen Weiß"]
CITIES = ["Oslo", "Lima", "Porto", "Austin", "Leeds", "Kyoto", "Perth", "Quito", "Riga", "Turin"]
UNICODE_CITIES = ["Malmö", "Zürich", "Bogotá", "Kraków", "Łódź"]


def _fx_fields(rng, p):
    """[(name, start, end, kind)] with 1-based inclusive positions."""
    spec = [("id", "int", 6), ("name", "text", rng.randint(18, 22))]
    pool = [("city", "text", rng.randint(10, 13)), ("qty", "int", rng.randint(5, 6)), ("balance", "int", rng.randint(8, 9)), ("code", "text", 7)]
    spec += rng.sample(pool, p["n_fields"] - 2)
    head, rest = spec[:1], spec[1:]
    rng.shuffle(rest)
    fields, pos = [], 1
    for name, kind, width in head + rest:
        fields.append((name, pos, pos + width - 1, kind))
        pos += width
    return fields


def _fx_value(rng, name, p, plant):
    level = p["level"]
    if name == "id":
        return rng.randint(1, 99999)
    if name == "name":
        if plant == "quote":
            return rng.choice(QUOTED_NAMES)
        if plant == "backslash":
            return rng.choice(BACKSLASH_NAMES)
        if plant == "unicode":
            return rng.choice(UNICODE_NAMES)
        return rng.choice(NAMES)
    if name == "city":
        if plant == "empty_text":
            return ""
        if level == "hard" and rng.random() < 0.2:
            return rng.choice(UNICODE_CITIES)
        return rng.choice(CITIES)
    if name == "qty":
        return None if plant == "empty_int" else rng.randint(0, 500)
    if name == "balance":
        return None if plant == "empty_int" else rng.randint(-50000, 90000)
    if name == "code":
        return "" if plant == "empty_text" else f"{rng.choice('ABCDEFGH')}{rng.choice('KLMNPRST')}-{rng.randint(10, 99)}"
    raise KeyError(name)


def _fx_cell(rng, value, kind, width, zeros):
    if kind == "text":
        return value.ljust(width)
    if value is None:
        return " " * width
    text = str(value)
    if zeros and value >= 0 and rng.random() < 0.3:
        text = text.zfill(min(width, len(text) + rng.randint(1, 3)))
    return text.rjust(width)


def _fx_file(rng, p, fields, plants):
    n = rng.randint(10, 15)
    plants = list(plants) + [None] * (n - len(plants))
    rng.shuffle(plants)
    lines, records = [], []
    for plant in plants:
        rec, line = {}, ""
        for name, start, end, kind in fields:
            v = _fx_value(rng, name, p, plant)
            rec[name] = v
            line += _fx_cell(rng, v, kind, end - start + 1, p["leading_zeros"])
        if p["short_lines"]:
            line = line.rstrip(" ")
        lines.append(line)
        records.append(rec)
    if p["comments"]:
        header = "# " + " ".join(name for name, _, _, _ in fields)
        lines.insert(0, header)
        lines.insert(rng.randint(2, len(lines)), "# continued")
    return "\n".join(lines) + "\n", records


def _jsonl(records):
    return "".join(json.dumps(r) + "\n" for r in records)


def _fx_scripts(p, fields, sample_jsonl):
    head = f"FIELDS = {fields!r}\n\n\n"
    oracle = "import json\nimport sys\n\n" + head + FX_PARSE + FX_MAIN
    scripts = {
        "oracle": oracle,
        "off-by-one-columns": oracle.replace("line[start - 1 : end]", "line[start - 1 : end - 1]"),
        "hand-built-json": "import sys\n\n" + head + FX_PARSE + FX_DUMP_BY_HAND + FX_MAIN.replace("json.dumps(parse(line))", "dump(parse(line))"),
        "hardcode-examples": f"import sys\n\nSAMPLE = {sample_jsonl!r}\n\nsys.stdout.write(SAMPLE)\n",
    }
    if p["level"] != "easy":
        scripts["empty-int-as-zero"] = oracle.replace("int(raw) if raw else None", "int(raw or 0)")
        scripts["no-skip-comments"] = oracle.replace('            if not line.strip() or line.startswith("#"):\n                continue\n', "")
    if p["level"] == "hard":
        scripts["ensure-ascii-false"] = oracle.replace("json.dumps(parse(line))", "json.dumps(parse(line), ensure_ascii=False)")
    return scripts


def _build_fixed(rng, p):
    fields = _fx_fields(rng, p)
    plants = ["quote", "quote"]
    if p["level"] != "easy":
        plants += ["empty_int", "empty_int", "backslash"]
    if p["level"] == "hard":
        plants += ["unicode", "unicode", "empty_text"]
    has_int = any(kind == "int" and name != "id" for name, _, _, kind in fields)
    if p["level"] != "easy" and not has_int:
        raise Reject("no optional integer field for the empty-integer trap")
    sample, sample_recs = _fx_file(rng, p, fields, plants)
    inputs = [_fx_file(rng, p, fields, plants) for _ in range(N_HIDDEN)]
    table = "\n".join(f"| `{name}` | {start}-{end} | {'integer' if kind == 'int' else 'text'} |" for name, start, end, kind in fields)
    sample_jsonl = _jsonl(sample_recs)
    return {
        "instruction": FX_INSTRUCTION.format(
            table=table,
            zeros=" or leading zeros" if p["leading_zeros"] else "",
            empty_int=" An integer field that holds only spaces is `null`." if p["level"] != "easy" else "",
            short="- A line may end early when its last fields are empty; the missing positions count as spaces.\n" if p["short_lines"] else "",
            skip="- Blank lines and lines starting with `#` are not records; skip them.\n" if p["comments"] else "",
        ),
        "examples": {"examples/sample.txt": sample, "examples/sample.jsonl": sample_jsonl},
        "cases": [(f"input_{i + 1}.txt", text, _jsonl(recs)) for i, (text, recs) in enumerate(inputs)] + [("empty.txt", "", "")],
        "scripts": _fx_scripts(p, fields, sample_jsonl),
        "params": {"conversion": "fixed-jsonl", "fields": [list(f) for f in fields]},
    }


# --------------------------------------------------------------------------- #
# Assembling an instance
# --------------------------------------------------------------------------- #


def _write(source):
    return f"cat > /app/convert.py <<'PY'\n{source}PY\n"


def build(ctx):
    rng, p = ctx.rng, dict(ctx.params, level=ctx.difficulty)
    conversion = rng.choice(["kv-csv", "fixed-jsonl"])
    task = _build_kv(rng, p) if conversion == "kv-csv" else _build_fixed(rng, p)
    checks = tuple(
        {"name": name.split(".")[0], "argv": ["python3", "convert.py", name], "inputs": {name: text}, "stdout": expected, "exit": 0}
        for name, text, expected in task["cases"]
    )
    scripts = task["scripts"]
    oracle = scripts.pop("oracle")
    return TaskSpec(
        instruction=task["instruction"],
        files=task["examples"],
        grader=Commands(("/app/convert.py",), checks),
        oracle=Solution(_write(oracle), lambda f, s=oracle: {"/app/convert.py": s}),
        shortcuts={k: Solution(_write(v), lambda f, s=v: {"/app/convert.py": s}) for k, v in scripts.items()},
        params=task["params"],
    )


FAMILY = Family(
    name="format-converter",
    version=1,
    cluster="write-code",
    category="coding",
    skills=("python", "parsing", "escaping", "file-formats"),
    difficulties={
        "easy": {"n_columns": 4, "n_fields": 4, "comments": False, "leading_zeros": False, "short_lines": False},
        "medium": {"n_columns": 5, "n_fields": 5, "comments": True, "leading_zeros": True, "short_lines": False},
        "hard": {"n_columns": 6, "n_fields": 6, "comments": True, "leading_zeros": True, "short_lines": True},
    },
    build=build,
)
