"""extract-subset: extract only the `.conf` files from a bundle of nested tar.gz archives.

/app/bundle.tar.gz holds plain files and inner archives (`.tar.gz`; on hard also `.tgz`, and
one inner archive holds another). Every file whose name ends in `.conf`, at any level, goes
to /app/out with its path kept: a file at P in an inner archive stored at A.tar.gz (or A.tgz)
goes to <A>/P under the output directory of the archive that holds A. The grader is a
FileTree over /app/out: exactly those files.

Construction: every level has `.conf` files next to decoys whose paths contain `conf` but do
not end in `.conf` (`etc/api.conf.example`, `docs/configuration.md`, `conf.d/README.txt`), and
every inner archive holds `.conf` files, so extracting everything, matching `*conf*`, skipping
the inner archives, flattening paths, or extracting inner archives at the top of /app/out
all fail. On medium/hard several inner archives share a member path (`conf.d/limits.conf`).
On hard one inner archive is a `.tgz` and another contains a nested `.tgz`, so treating only
`.tar.gz` as archives or descending only one level fails. Every trap holds by construction,
so the first draw is always used.

The archives are built byte-for-byte deterministically: sorted members, fixed mtime, uid/gid 0,
GNU format, and gzip with mtime 0.
"""

import gzip
import io
import tarfile

from learning_loop.tasks.spec import Family, FileTree, gzip_bytes, Solution, TaskSpec, tree_entry

TAR_MTIME = 1788220800  # 2026-09-01T00:00:00Z
SERVICES = ["api", "web", "worker", "billing", "search", "mailer", "auth", "reports"]
PLUGINS = ["ratelimit", "audit", "cache", "metrics"]
KEYS = ["port", "workers", "timeout", "log_level", "max_body", "pool_size", "retries", "region"]

INSTRUCTION = """`/app/bundle.tar.gz` is a release bundle. Besides plain files it contains other gzipped tar archives{kinds}{nesting}.

Extract every file whose name ends in `.conf` into `/app/out/`, from the bundle itself and from every archive inside it, keeping their paths:

- a file at path `P` in the bundle goes to `/app/out/P`;
- an archive at path `A{exts}` in the bundle is not extracted as a file; a file at path `P` inside it goes to `/app/out/A/P` (`etc/x.conf` inside `pkg/y.tar.gz` goes to `/app/out/pkg/y/etc/x.conf`){nested_rule}

Do not flatten the paths. `/app/out/` must contain only those `.conf` files, with their contents unchanged: no other files and no archives.
"""

NESTED_RULE = ";\n- the same rule applies inside inner archives, at every level: an archive at `B.tgz` inside `pkg/y.tar.gz` puts its file `P` at `/app/out/pkg/y/B/P`."

SOLVER = '''
import io
import posixpath
import tarfile


def members(blob):
    """(path, bytes) of the regular files in a gzipped tar, in archive order."""
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        for m in tf.getmembers():
            if m.isfile():
                yield m.name, tf.extractfile(m).read()


def extract(blob, mode, prefix="", depth=0, out=None):
    """{output path relative to /app/out: bytes} for the files `mode` extracts."""
    out = {} if out is None else out
    kinds = (".tar.gz",) if mode == "tar-gz-only" else (".tar.gz", ".tgz")
    for name, data in members(blob):
        kind = next((k for k in kinds if name.endswith(k)), None)
        if kind is not None:
            if mode == "one-level" and depth >= 1:
                continue
            inner = prefix if mode in ("inner-at-root", "flatten") else posixpath.join(prefix, name[: -len(kind)])
            extract(data, mode, inner, depth + 1, out)
            continue
        if mode == "extract-everything":
            wanted = True
        elif mode == "name-contains":
            wanted = "conf" in name
        else:
            wanted = name.endswith(".conf")
        if wanted:
            out[posixpath.basename(name) if mode == "flatten" else posixpath.join(prefix, name)] = data
    return out
'''

_PY_SHELL = """python3 - <<'PY'
{solver}
import os

with open("/app/bundle.tar.gz", "rb") as f:
    files = extract(f.read(), {mode!r})
for rel, data in sorted(files.items()):
    path = os.path.join("/app/out", rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
PY
"""

SKIP_NESTED = "mkdir -p /app/out && tar -xzf /app/bundle.tar.gz -C /app/out --wildcards '*.conf'\n"

_NS: dict = {}
exec(SOLVER, _NS)  # noqa: S102 - our own source; the model form of the Python solutions


def _py_solution(mode):
    return Solution(_PY_SHELL.format(solver=SOLVER, mode=mode), lambda f: {f"/app/out/{rel}": data for rel, data in _NS["extract"](f["bundle.tar.gz"], mode).items()})


def _skip_nested(files):
    return {f"/app/out/{name}": data for name, data in _NS["members"](files["bundle.tar.gz"]) if name.endswith(".conf")}


def targz(members):
    """A gzipped tar of {path: bytes}, byte-for-byte deterministic."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.GNU_FORMAT) as tf:
        for name in sorted(members):
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(members[name]), TAR_MTIME, 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            tf.addfile(info, io.BytesIO(members[name]))
    return gzip_bytes(buf.getvalue())


def _conf(rng, title):
    keys = rng.sample(KEYS, rng.randint(2, 5))
    return (f"# {title}\n" + "".join(f"{k} = {rng.randint(1, 9000)}\n" for k in keys)).encode()


def _text(rng, title):
    return f"{title}\n\nBuild {rng.randint(100, 999)}, see the operations guide.\n".encode()


def _service(rng, svc, shared_limits, plugin=None):
    m = {
        f"etc/{svc}.conf": _conf(rng, svc),
        f"etc/{svc}.conf.example": _conf(rng, f"{svc} example"),
        "bin/start.sh": f"#!/bin/sh\nexec /opt/{svc}/run --config /etc/{svc}.conf\n".encode(),
        "README.md": _text(rng, f"# {svc}"),
    }
    if shared_limits:
        m["conf.d/limits.conf"] = _conf(rng, f"{svc} limits")
        m["conf.d/README.txt"] = _text(rng, "Drop-in configuration")
    if plugin is not None:
        m[f"plugins/{plugin}.tgz"] = targz({f"etc/{plugin}.conf": _conf(rng, plugin), "README.md": _text(rng, f"# {plugin}"), f"etc/{plugin}.conf.sample": _conf(rng, f"{plugin} sample")})
    return m


def build(ctx):
    rng, p = ctx.rng, ctx.params
    services = rng.sample(SERVICES, p["services"])
    outer = {
        "README.md": _text(rng, "# Release bundle"),
        "VERSION": f"2026.9.{rng.randint(1, 30)}\n".encode(),
        "etc/bundle.conf": _conf(rng, "bundle"),
        "etc/bundle.conf.bak": _conf(rng, "bundle backup"),
        "docs/configuration.md": _text(rng, "# Configuration"),
    }
    folder = "services/" if p["subdir"] else ""
    for i, svc in enumerate(services):
        plugin = rng.choice(PLUGINS) if p["nested"] and i == 0 else None
        ext = ".tgz" if p["tgz"] and i == len(services) - 1 else ".tar.gz"
        outer[f"{folder}{svc}{ext}"] = targz(_service(rng, svc, shared_limits=p["shared"] and i < 2, plugin=plugin))
    bundle = targz(outer)
    expected = _NS["extract"](bundle, "oracle")
    shortcuts = {mode: _py_solution(mode) for mode in ("extract-everything", "name-contains", "flatten", "inner-at-root")}
    shortcuts["skip-nested"] = Solution(SKIP_NESTED, _skip_nested)
    if p["nested"]:
        shortcuts["one-level"] = _py_solution("one-level")
    if p["tgz"]:
        shortcuts["tar-gz-only"] = _py_solution("tar-gz-only")
    hard = p["tgz"]
    return TaskSpec(
        instruction=INSTRUCTION.format(
            kinds=" (`.tar.gz` or `.tgz`)" if hard else " (`.tar.gz`)",
            nesting=", and archives can contain further archives" if p["nested"] else "",
            exts=".tar.gz` or `A.tgz" if hard else ".tar.gz",
            nested_rule=NESTED_RULE if p["nested"] else ".",
        ),
        files={"bundle.tar.gz": bundle},
        grader=FileTree("/app/out", {rel: tree_entry(data) for rel, data in sorted(expected.items())}),
        oracle=_py_solution("oracle"),
        shortcuts=shortcuts,
        params={"services": services, "extracted": len(expected)},
    )


FAMILY = Family(
    name="extract-subset",
    version=1,
    cluster="filesystem",
    category="shell",
    skills=("shell", "tar", "gzip", "archives"),
    difficulties={
        "easy": {"services": 2, "subdir": False, "shared": False, "nested": False, "tgz": False},
        "medium": {"services": 3, "subdir": True, "shared": True, "nested": False, "tgz": False},
        "hard": {"services": 4, "subdir": True, "shared": True, "nested": True, "tgz": True},
    },
    build=build,
)
