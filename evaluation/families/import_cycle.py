"""import-cycle: find the one import cycle among a package's modules and report it in a
canonical rotation.

Construction: a package `/app/<pkg>/` (with subpackages on hard). The import graph is a
random DAG over "units" (the modules outside the cycle, and the cycle as one unit) plus one
simple cycle of 3 (easy/medium) or 4 (hard) modules, so exactly one cycle exists. The module
whose name sorts first is never in the cycle and imports into it. Each import is written in
one of several forms: `import a.b`, `from a.b import name`, `from a import b`, and on
medium/hard `import a.b as x`, `import os, a.b` and relative forms (`from .b import name`,
`from . import b`, `from ..sub import b`); hard puts one import inside a function. Fake
imports in comments, docstrings and string constants point "upward" in the DAG, so a
reader that counts them sees extra cycles.

Traps, by construction or checked: the cycle has a `from <package> import <module>` (or
relative) edge, so taking only `import x` lines or only `node.module` of a `from` statement
misses it; counting text in comments and strings finds a fake cycle first (checked);
printing the whole DFS stack reports a non-cycle path that starts at the first module; the
cycle has 3+ modules, so listing it in reverse (importer after imported) fails. Medium/hard:
a cycle edge is relative (skipping relative imports fails). Hard: a cycle edge is an import
inside a function (looking only at top-level statements fails).

Every solution is one Python source (`SOLVER`) run with a mode, as in csv-revenue.
"""

from learning_loop.tasks.spec import Family, ParsedAnswer, Reject, Solution, TaskSpec

PACKAGES = ["inventory", "ledger", "storefront", "telemetry", "scheduler", "helpdesk"]
NAMES = [
    "accounts", "alerts", "audit", "billing", "cache", "catalog", "config", "db", "events", "exports",
    "forms", "handlers", "jobs", "mailer", "metrics", "models", "orders", "payments", "permissions",
    "pricing", "queue", "reports", "routes", "search", "serializers", "sessions", "signals", "storage",
    "tasks", "tokens", "users", "validators", "views", "webhooks",
]
SUBPACKAGES = ["api", "core", "workers"]

INSTRUCTION = """The `{pkg}` package in `/app/{pkg}/` has a circular import that we need to break. Find the import cycle among its modules. Exactly one cycle exists.

- The modules are the `.py` files in the package{subs}, named by their dotted names (e.g. `{pkg}.{example}`). The `__init__.py` files are empty and are not modules for this purpose.
- Module A imports module B when A has an import statement that imports B: `import {pkg}.b`, `from {pkg}.b import name`, `from {pkg} import b`{rel}. Import statements inside functions count; text in comments and strings does not.

Write the cycle to `/app/answer.json` as a JSON list of dotted module names. Start with the name that sorts first (plain string comparison), then list the modules in import order, so that each module imports the next one and the last imports the first. Do not repeat the first module at the end.
"""

SOLVER = r'''
import ast
import re


def _from_targets(mods, importer, module, level, names, mode):
    """Modules a `from <dots><module> import <names>` statement in `importer` imports."""
    if level:
        parts = importer.split(".")[:-level]
        base = ".".join(parts + ([module] if module else []))
    else:
        base = module or ""
    out = set()
    for n in names:
        if mode != "from-module-only" and base + "." + n in mods:
            out.add(base + "." + n)
        elif base in mods:
            out.add(base)
    return out


def _ast_edges(mods, name, source, mode):
    tree = ast.parse(source)
    nodes = tree.body if mode == "top-level-only" else list(ast.walk(tree))
    out = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names if a.name in mods}
        elif isinstance(node, ast.ImportFrom):
            if mode == "absolute-only" and node.level:
                continue
            out |= _from_targets(mods, name, node.module, node.level, [a.name for a in node.names], mode)
    return out


FROM_RE = re.compile(r"from\s+(\.*)([\w.]*)\s+import\s+([\w, ]+)")
IMPORT_RE = re.compile(r"(?:^|[^\w.])import\s+([\w.]+(?:\s+as\s+\w+)?(?:\s*,\s*[\w.]+(?:\s+as\s+\w+)?)*)")


def _text_edges(mods, name, source, mode):
    out = set()
    for line in source.splitlines():
        if mode == "import-only":
            m = re.match(r"\s*import\s+(.+)$", line)
            parts = m.group(1).split(",") if m else []
            out |= {p.split()[0] for p in parts if p.split() and p.split()[0] in mods}
            continue
        for m in FROM_RE.finditer(line):
            out |= _from_targets(mods, name, m.group(2), len(m.group(1)), [n.strip() for n in m.group(3).split(",") if n.strip()], mode)
        rest = FROM_RE.sub(" ", line)
        for m in IMPORT_RE.finditer(rest):
            out |= {p.split()[0] for p in m.group(1).split(",") if p.split() and p.split()[0] in mods}
    return out


def graph(sources, mode):
    mods = set(sources)
    edges = _text_edges if mode in ("grep-anywhere", "import-only") else _ast_edges
    return {name: edges(mods, name, src, mode) - {name} for name, src in sources.items()}


def find_cycle(g, mode):
    state, stack = {}, []

    def dfs(u):
        state[u] = 1
        stack.append(u)
        for v in sorted(g.get(u, ())):
            if state.get(v) == 1:
                return list(stack) if mode == "dfs-path" else stack[stack.index(v):]
            if v not in state:
                found = dfs(v)
                if found:
                    return found
        state[u] = 2
        stack.pop()
        return None

    for n in sorted(g):
        if n not in state:
            found = dfs(n)
            if found:
                return found
    return None


def answer(sources, mode):
    """sources: {dotted module name: source}. Returns the cycle as a list ([] if none)."""
    cycle = find_cycle(graph(sources, mode), mode)
    if not cycle:
        return []
    if mode == "dfs-path":
        return cycle
    if mode == "reversed":
        cycle = cycle[::-1]
    i = cycle.index(min(cycle))
    return cycle[i:] + cycle[:i]
'''

_SHELL = """python3 - <<'PY'
{solver}
import json
import os

sources = {{}}
for d, _, names in os.walk("/app/{pkg}"):
    for n in names:
        if n.endswith(".py") and n != "__init__.py":
            p = os.path.join(d, n)
            sources[os.path.relpath(p, "/app")[:-3].replace("/", ".")] = open(p).read()
with open("/app/answer.json", "w") as f:
    f.write(json.dumps(answer(sources, {mode!r})) + "\\n")
PY
"""

_NS: dict = {}
exec(SOLVER, _NS)  # noqa: S102 - our own source; the model form of every solution


def _sources(files):
    return {rel[:-3].replace("/", "."): data.decode() for rel, data in files.items() if rel.endswith(".py") and not rel.endswith("__init__.py")}


def _solution(mode, pkg):
    def model(files):
        return {"/app/answer.json": "%s\n" % _json(_NS["answer"](_sources(files), mode))}

    return Solution(_SHELL.format(solver=SOLVER, mode=mode, pkg=pkg), model)


def _json(v):
    import json

    return json.dumps(v)


def _relative(importer, target):
    """(dots, parent path or "", leaf) for a relative import of `target` from `importer`."""
    ip, tp = importer.split(".")[:-1], target.split(".")
    common = 0
    while common < min(len(ip), len(tp) - 1) and ip[common] == tp[common]:
        common += 1
    dots = "." * (len(ip) - common + 1)
    rest = tp[common:]
    return dots, ".".join(rest[:-1]), rest[-1]


def _sym(module):
    return "load_" + module.split(".")[-1]


def _statement(importer, target, form):
    """(import line, reference expression, function block or None) for one edge."""
    leaf, parent, sym = target.split(".")[-1], target.rsplit(".", 1)[0], _sym(target)
    if form == "import":
        return f"import {target}", f"{target}.{sym}", None
    if form == "alias":
        return f"import {target} as {leaf}_mod", f"{leaf}_mod.{sym}", None
    if form == "multi":
        return f"import os, {target}", f"{target}.{sym}", None
    if form == "from-name":
        return f"from {target} import {sym}", sym, None
    if form == "from-pkg":
        return f"from {parent} import {leaf}", f"{leaf}.{sym}", None
    dots, rparent, rleaf = _relative(importer, target)
    if form == "relative-name":
        return f"from {dots}{rparent + '.' if rparent else ''}{rleaf} import {sym}", sym, None
    if form == "relative-mod":
        return f"from {dots}{rparent} import {rleaf}", f"{rleaf}.{sym}", None
    if form == "function":
        return None, f"lazy_{leaf}()", f"def lazy_{leaf}():\n    from {target} import {sym}\n\n    return {sym}()\n"
    raise ValueError(form)


def _fake(kind, target):
    sym = _sym(target)
    if kind == "comment":
        return f"# from {target} import {sym}  (no longer needed here)"
    if kind == "string":
        return f'MIGRATION_HINT = "replace `import {target}` with the new API"'
    return f"Example::\n\n    from {target} import {sym}\n"


def build(ctx):
    rng, p = ctx.rng, ctx.params
    pkg = rng.choice(PACKAGES)
    leaves = rng.sample(NAMES, p["modules"])
    mods = []
    for leaf in leaves:
        sub = rng.choice(SUBPACKAGES + [None]) if p["subpackages"] else None
        mods.append(f"{pkg}.{sub}.{leaf}" if sub else f"{pkg}.{leaf}")
    tail = min(mods)
    rest = [m for m in mods if m != tail]
    rng.shuffle(rest)
    cycle, others = rest[: p["cycle"]], rest[p["cycle"]:]
    # Units in topological order (importers after what they import); the tail is last.
    units = [[m] for m in others]
    units.insert(rng.randint(0, len(units)), list(cycle))
    units.append([tail])
    edges = {}  # (importer, target) -> form, filled below

    def add(a, b):
        edges.setdefault((a, b), None)

    for i, unit in enumerate(units):
        below = units[:i]
        if not below:
            continue
        for _ in range(rng.randint(1, min(2, len(below)))):
            add(rng.choice(unit), rng.choice(rng.choice(below)))
    add(tail, rng.choice(cycle))
    cyc_edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]
    for e in cyc_edges:
        add(*e)
    # Forms: the cycle's edges carry the required ones, the rest are drawn.
    forms = list(p["forms"])
    required = list(p["required"])
    rng.shuffle(required)
    for e, f in zip(rng.sample(cyc_edges, len(cyc_edges)), required + [None] * len(cyc_edges)):
        edges[e] = f
    for e in edges:
        if edges[e] is None:
            edges[e] = rng.choice([f for f in forms if f != "function"])
    # Fake imports: from a lower unit's module to a higher unit's module.
    pos = {m: i for i, u in enumerate(units) for m in u}
    # The first fake import goes from the cycle module a DFS from the first module (the tail)
    # enters first back to the tail: that name sorts before every other, so a reader that
    # counts the fake finds the cycle through the tail before the real one.
    true_graph = {}
    for a, b in edges:
        true_graph.setdefault(a, set()).add(b)
    seen, order = set(), []

    def visit(u):
        seen.add(u)
        order.append(u)
        for v in sorted(true_graph.get(u, ())):
            if v not in seen:
                visit(v)

    visit(tail)
    entry = next(m for m in order if m in cycle)
    fakes = [(entry, tail, p["fakes"][0])]
    for kind in p["fakes"][1:]:
        a, b = rng.sample(mods, 2)
        while pos[a] == pos[b]:  # both in the cycle: not an upward edge
            a, b = rng.sample(mods, 2)
        if pos[a] > pos[b]:
            a, b = b, a
        fakes.append((a, b, kind))

    files = {}
    for sub in sorted({m.split(".")[1] for m in mods if m.count(".") == 2}):
        files[f"{pkg}/{sub}/__init__.py"] = ""
    files[f"{pkg}/__init__.py"] = ""
    for m in mods:
        leaf = m.split(".")[-1]
        doc = f"{leaf.replace('_', ' ').capitalize()} for {pkg}."
        lines, refs, blocks = ["from __future__ import annotations", ""], [], []
        for (a, b), form in sorted(edges.items()):
            if a != m:
                continue
            stmt, ref, block = _statement(a, b, form)
            if stmt:
                lines.append(stmt)
            if block:
                blocks.append(block)
            refs.append(ref)
        consts, comments = [], []
        for a, b, kind in fakes:
            if a != m:
                continue
            if kind == "docstring":
                doc += "\n\n" + _fake(kind, b)
            elif kind == "string":
                consts.append(_fake(kind, b))
            else:
                comments.append("    " + _fake(kind, b))
        body = f'"""{doc}"""\n\n' + "\n".join(lines) + "\n"
        if consts:
            body += "\n" + "\n".join(consts) + "\n"
        body += f"\n\ndef {_sym(m)}():\n" + "".join(c + "\n" for c in comments) + f'    return {{"module": "{leaf}"}}\n'
        for block in blocks:
            body += "\n\n" + block
        body += f"\n\ndef setup():\n    return [{', '.join(refs)}]\n"
        files[m.replace(".", "/") + ".py"] = body

    expected = cycle[cycle.index(min(cycle)):] + cycle[: cycle.index(min(cycle))]
    fbytes = {k: v.encode() for k, v in files.items()}
    if _NS["answer"](_sources(fbytes), "oracle") != expected:
        raise Reject("the oracle does not find the planted cycle")
    names = ["grep-anywhere", "import-only", "from-module-only", "dfs-path", "reversed"]
    if "relative-mod" in forms:
        names.append("absolute-only")
    if "function" in forms:
        names.append("top-level-only")
    shortcuts = {n: _solution(n, pkg) for n in names}
    rel = ""
    if "relative-mod" in forms:
        rel = ", or the relative forms (`from .b import name`, `from . import b`" + (", `from ..sub import b`" if p["subpackages"] else "") + ")"
    return TaskSpec(
        instruction=INSTRUCTION.format(
            pkg=pkg,
            subs=" and its subpackages" if p["subpackages"] else "",
            example=min(mods).split(".", 1)[1],
            rel=rel,
        ),
        files=files,
        grader=ParsedAnswer("/app/answer.json", "json", expected),
        oracle=_solution("oracle", pkg),
        shortcuts=shortcuts,
        params={"package": pkg, "modules": len(mods), "cycle": expected},
    )


FAMILY = Family(
    name="import-cycle",
    version=1,
    cluster="diagnosis",
    category="debugging",
    skills=("python", "imports", "graphs", "code-reading", "diagnosis"),
    difficulties={
        "easy": {"modules": 6, "cycle": 3, "subpackages": False, "forms": ["import", "from-name", "from-pkg"], "required": ["from-pkg", "from-name"], "fakes": ["comment"]},
        "medium": {
            "modules": 9,
            "cycle": 3,
            "subpackages": False,
            "forms": ["import", "from-name", "from-pkg", "alias", "multi", "relative-name", "relative-mod"],
            "required": ["from-pkg", "relative-name"],
            "fakes": ["comment", "string"],
        },
        "hard": {
            "modules": 12,
            "cycle": 4,
            "subpackages": True,
            "forms": ["import", "from-name", "from-pkg", "alias", "multi", "relative-name", "relative-mod", "function"],
            "required": ["relative-mod", "function", "from-pkg"],
            "fakes": ["comment", "string", "docstring"],
        },
    },
    build=build,
)
