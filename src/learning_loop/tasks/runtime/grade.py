"""Shared task grader.

Standard-library only: the renderer copies this file verbatim into every task's `tests/`
directory, where the separate verifier container runs it, and generation imports it on the
host to grade each instance's oracle, shortcut and no-op outputs with the same code.

    python3 -I -B grade.py --root / --out /logs/verifier/reward.json [--key key.json] [--probe '<json>']

`--root` is the directory that stands for the container's `/` (artifacts are read from
`<root>/<path>`); `--key` defaults to `key.json` next to this file; `--probe` runs
`probe.py` (also next to this file) first and refuses to grade when the environment
violates the expectation. When `LL_TESTS_SHA256` is set (the renderer puts it in task.toml's
`[verifier] env`, which reaches the container without passing through the image build), the
grader also refuses unless `tests_digest` of its own directory matches, so a verifier image
built from stale files never grades. The grader writes exactly one key, `reward`, in [0, 1].

Grader kinds (the `kind` field of the key):

    exact     text of `path`, stripped, equals `expected`
    numeric   number in `path` is within max(abs_tol, rel_tol * |expected|) of `expected`
    parsed    `path` parsed as `format` equals `expected`: json, toml (object key order
              ignored), dotenv (KEY=VALUE lines; duplicate keys fail), lines (non-empty
              stripped lines, in order) or line-set (the same, order ignored)
    tree      the regular files under the directory `root` are exactly `expected`
              ({relative path: {"sha256", optional "mode"}}); reward = correct entries /
              (expected entries + unexpected entries)
    checks    call functions of the Python module at `path` with hidden inputs; reward is the
              fraction of checks passed
    commands  run commands (e.g. `python3 convert.py in.csv`) in a scratch directory holding
              copies of the artifacts in `files` and each check's inputs; compare stdout, exit
              code and output files; reward is the fraction of checks passed

`checks` and `commands` execute agent-written code. In the verifier it runs as the
unprivileged user `nobody`, which cannot read the key (its directory is made 0700 first) and
receives only the check inputs, and every `nobody` process is killed before the reward is
written. `checks` runs a child `python3 -I -B` that prints raw observations on one marker
line; this process accepts only plain JSON values of the expected shape for known checks, so
an object whose `__eq__` always returns true still fails. At generation (`trusted=True`) the
same code grades our own reference and wrong solutions in this process or as the current
user.
"""

import argparse
import hashlib
import json
import math
import os
import pwd
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time

MAX_ARTIFACT_BYTES = 10 * 1024 * 1024
CHILD_TIMEOUT_SEC = 30
MARKER = "@@CHECK_OBSERVATIONS@@ "
EXIT_PROBE_FAILED = 3
EXIT_GRADER_ERROR = 4
EXIT_STALE_FILES = 5
TESTS_DIGEST_ENV = "LL_TESTS_SHA256"
TESTS_DIGEST_FILES = ("grade.py", "key.json", "probe.py")
CHILD_ENV = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8", "TZ": "UTC", "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}


class GraderError(Exception):
    """The grader cannot produce a trustworthy reward; no reward is written."""


def tests_digest(directory):
    """sha256 over the grader's own files (names and contents), in a fixed order."""
    h = hashlib.sha256()
    for name in TESTS_DIGEST_FILES:
        with open(os.path.join(directory, name), "rb") as f:
            h.update(name.encode() + b"\0" + hashlib.sha256(f.read()).hexdigest().encode() + b"\n")
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Views: what the grader sees of the container state
# --------------------------------------------------------------------------- #


def read_artifact(root, path):
    """Bytes of the regular file `<root>/<path>`, or None (missing, symlink, not a file, too big)."""
    real = os.path.join(root, path.lstrip("/"))
    try:
        fd = os.open(real, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        return None
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        return None
    with os.fdopen(fd, "rb") as f:
        data = f.read(MAX_ARTIFACT_BYTES + 1)
    return None if len(data) > MAX_ARTIFACT_BYTES else data


class DiskView:
    """The transferred artifacts on disk under `root` (the verifier's `/`)."""

    def __init__(self, root):
        self.root = root

    def read(self, path):
        return read_artifact(self.root, path)

    def tree(self, path):
        """{relative path: ("F", sha256, mode) | ("L", target) | ("O",)} under `path` (no links followed)."""
        base = os.path.join(self.root, path.lstrip("/"))
        out = {}
        if not os.path.isdir(base) or os.path.islink(base):
            return out
        for dirpath, dirnames, filenames in os.walk(base):
            for name in filenames + [d for d in dirnames if os.path.islink(os.path.join(dirpath, d))]:
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, base)
                st = os.lstat(full)
                if stat.S_ISLNK(st.st_mode):
                    out[rel] = ("L", os.readlink(full))
                elif stat.S_ISREG(st.st_mode):
                    data = read_artifact("/", full)
                    out[rel] = ("F", hashlib.sha256(data).hexdigest(), stat.S_IMODE(st.st_mode)) if data is not None else ("O",)
                else:
                    out[rel] = ("O",)
        return out


class MemoryView:
    """A container state held in memory: {absolute path: (bytes, mode)}. Generation uses it
    for the initial files overlaid with what a solution model predicts."""

    def __init__(self, state):
        self.state = state

    def read(self, path):
        entry = self.state.get(path)
        return entry[0] if entry is not None else None

    def tree(self, path):
        prefix = path.rstrip("/") + "/"
        return {p[len(prefix):]: ("F", hashlib.sha256(data).hexdigest(), mode) for p, (data, mode) in self.state.items() if p.startswith(prefix)}


def _text(data):
    if data is None:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _canon(v):
    return json.dumps(v, sort_keys=True, separators=(",", ":"))


# --------------------------------------------------------------------------- #
# Answer kinds
# --------------------------------------------------------------------------- #


def grade_exact(key, view):
    text = _text(view.read(key["path"]))
    if text is None:
        return 0.0, ["missing or unreadable: " + key["path"]]
    ok = text.strip() == key["expected"]
    return (1.0 if ok else 0.0), ["expected: %r" % key["expected"][:500], "actual:   %r" % text.strip()[:500]]


def grade_numeric(key, view):
    text = _text(view.read(key["path"]))
    if text is None:
        return 0.0, ["missing or unreadable: " + key["path"]]
    try:
        value = float(text.strip())
    except ValueError:
        return 0.0, ["not a number: %r" % text.strip()[:200]]
    exp = key["expected"]
    tol = max(key.get("abs_tol", 0.0), key.get("rel_tol", 0.0) * abs(exp))
    ok = math.isfinite(value) and abs(value - exp) <= tol
    return (1.0 if ok else 0.0), ["expected: %r (tolerance %r)" % (exp, tol), "actual:   %r" % value]


def _reject_constant(name):
    raise ValueError("non-finite JSON constant " + name)


def parse_text(text, fmt):
    """`text` parsed as `fmt`; raises ValueError when it does not parse."""
    if fmt == "json":
        return json.loads(text, parse_constant=_reject_constant)
    if fmt == "toml":
        import tomllib

        return tomllib.loads(text)
    if fmt == "dotenv":
        out = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError("not KEY=VALUE: %r" % line[:100])
            k, v = line.split("=", 1)
            k = k.strip()
            if k in out:
                raise ValueError("duplicate key %r" % k)
            out[k] = v.strip()
        return out
    if fmt == "lines":
        return [ln.strip() for ln in text.splitlines() if ln.strip()]
    if fmt == "line-set":
        return sorted(ln.strip() for ln in text.splitlines() if ln.strip())
    raise GraderError("unknown parsed format %r" % (fmt,))


def grade_parsed(key, view):
    text = _text(view.read(key["path"]))
    if text is None:
        return 0.0, ["missing or unreadable: " + key["path"]]
    try:
        value = parse_text(text, key["format"])
    except (ValueError, TypeError) as e:
        return 0.0, ["does not parse as %s: %s" % (key["format"], e)]
    ok = _canon(value) == _canon(key["expected"])
    return (1.0 if ok else 0.0), ["expected: " + _canon(key["expected"])[:500], "actual:   " + _canon(value)[:500]]


def grade_tree(key, view):
    actual = view.tree(key["root"])
    expected = key["expected"]
    log, correct = [], 0
    for rel, want in sorted(expected.items()):
        got = actual.get(rel)
        if got is None:
            log.append("missing: " + rel)
        elif got[0] != "F" or got[1] != want["sha256"]:
            log.append("wrong content or not a regular file: " + rel)
        elif "mode" in want and "%04o" % got[2] != want["mode"]:
            log.append("wrong mode %04o (expected %s): %s" % (got[2], want["mode"], rel))
        else:
            correct += 1
    extras = sorted(set(actual) - set(expected))
    log += ["unexpected: " + rel for rel in extras[:50]]
    total = len(expected) + len(extras)
    log.append("%d/%d entries correct" % (correct, total))
    return (correct / total if total else 1.0), log


# --------------------------------------------------------------------------- #
# Running agent code as `nobody`
# --------------------------------------------------------------------------- #


def _kill_user_processes(uid):
    for _ in range(50):
        found = False
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                with open("/proc/%s/status" % pid) as f:
                    uids = next(line for line in f if line.startswith("Uid:")).split()[1:]
            except (OSError, StopIteration):
                continue
            if str(uid) in uids:
                found = True
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except OSError:
                    pass
        if not found:
            return
        time.sleep(0.05)


class _Sandbox:
    """Runs agent code: as `nobody` in the verifier (which must be root), or as the current
    user at generation (`trusted`), always with a fixed environment and a timeout."""

    def __init__(self, trusted, key_dir):
        self.trusted = trusted
        self.user = None
        if not trusted:
            if os.geteuid() != 0:
                raise GraderError("graders that run agent code must run as root to isolate it")
            self.user = pwd.getpwnam("nobody")
            os.chmod(key_dir, 0o700)

    def workdir(self):
        work = tempfile.mkdtemp(prefix="grade-")
        os.chmod(work, 0o755)
        return work

    def handover(self, work):
        """Give `work` (and everything in it) to the sandbox user."""
        if self.user is None:
            return
        for dirpath, dirnames, filenames in os.walk(work):
            for name in [dirpath] + [os.path.join(dirpath, n) for n in dirnames + filenames]:
                os.lchown(name, self.user.pw_uid, self.user.pw_gid)

    def run(self, argv, cwd, stdin, timeout):
        """(exit code or None on timeout, stdout, stderr)."""
        if self.trusted and argv and argv[0] == "python3":
            argv = [sys.executable] + list(argv[1:])
        kw = {}
        if self.user is not None:
            kw = {"user": self.user.pw_uid, "group": self.user.pw_gid, "extra_groups": []}
        try:
            proc = subprocess.run(
                argv, input=stdin, capture_output=True, text=True, errors="replace", cwd=cwd,
                env=dict(CHILD_ENV, HOME=cwd), timeout=timeout, start_new_session=True, **kw,
            )
            return proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            return None, "", "timed out after %ss" % timeout
        except OSError as e:
            return 127, "", "could not start: %s" % e
        finally:
            if self.user is not None:
                _kill_user_processes(self.user.pw_uid)


def _place(work, rel, data):
    path = os.path.join(work, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


# --------------------------------------------------------------------------- #
# checks: call functions of an agent-written module
# --------------------------------------------------------------------------- #

OBSERVE = '''
def observe(lib, checks):
    import copy
    import json

    def plain(v):
        try:
            return {"value": json.loads(json.dumps(v, allow_nan=False))}
        except Exception:
            return {"unserializable": True}

    out = {}
    for c in checks:
        fn = lib.get(c["func"]) if isinstance(lib, dict) else None
        if not callable(fn):
            out[c["name"]] = {"missing": True}
            continue
        args = copy.deepcopy(c["args"])
        try:
            got = fn(*args)
        except BaseException as e:  # anything the artifact raises is an observation
            out[c["name"]] = {"raised": type(e).__name__[:200], "value_error": isinstance(e, ValueError)}
            continue
        if c["kind"] == "no_mutation":
            p = plain(args[0])
            out[c["name"]] = {"after": p["value"]} if "value" in p else p
        elif c["kind"] == "raises":
            out[c["name"]] = {"returned": True}
        elif c["kind"] == "value":
            ok_type = isinstance(got, (int, float)) and not isinstance(got, bool)
            out[c["name"]] = plain(got) if ok_type else {"value": None}
        else:
            out[c["name"]] = plain(got)
    return out
'''

exec(OBSERVE)  # defines observe() here too, from the exact source the child runs  # noqa: S102


def judge(checks, observations):
    """Check name -> passed. Accepts only the observation shapes `observe` produces.

    Check kinds: `value` (a number within `abs_tol`, default 1e-9 relative), `equal` (the
    return value equals `expected` as JSON, so 1 and 1.0 or True and 1 differ), `raises`
    (raises `exception` by class name, default any ValueError) and `no_mutation` (the first
    argument is unchanged after the call)."""

    def num(x):
        return type(x) in (int, float) and math.isfinite(x)

    results = {}
    obs_all = observations if isinstance(observations, dict) else {}
    for c in checks:
        o = obs_all.get(c["name"])
        ok = False
        if isinstance(o, dict):
            if c["kind"] == "raises":
                if set(o) == {"raised", "value_error"}:
                    ok = o["raised"] == c["exception"] if "exception" in c else o["value_error"] is True
            elif c["kind"] == "no_mutation":
                ok = set(o) == {"after"} and _canon(o["after"]) == _canon(c["args"][0])
            elif c["kind"] == "equal":
                ok = set(o) == {"value"} and _canon(o["value"]) == _canon(c["expected"])
            elif set(o) == {"value"} and num(o["value"]):
                exp = c["expected"]
                tol = c.get("abs_tol")
                ok = abs(o["value"] - exp) <= (tol if tol is not None else 1e-9 * max(1.0, abs(exp)))
        results[c["name"]] = ok is True
    return results


def _inputs(checks):
    return [{k: v for k, v in c.items() if k not in ("expected", "abs_tol", "exception")} for c in checks]


def _child_source(module):
    return (
        "import json\nimport sys\n"
        + OBSERVE
        + "\nsys.path.insert(0, sys.argv[1])\n"
        + "checks = json.loads(sys.stdin.read())\n"
        + "try:\n    import %s as mod\n    lib = dict(vars(mod))\n" % module
        + "except BaseException as e:\n    print('import failed: ' + type(e).__name__)\n    lib = None\n"
        + "obs = observe(lib, checks) if lib is not None else {}\n"
        + "sys.stdout.write('\\n' + %r + json.dumps(obs) + '\\n')\nsys.stdout.flush()\n" % MARKER
    )


def run_checks(lib, checks):
    """Check name -> passed for `lib` (a dict of our own functions), in this process, with
    the same observe/judge round trip the verifier uses. Generation-time only."""
    return judge(checks, json.loads(json.dumps(observe(lib, _inputs(checks)))))


def _observe_trusted(key, view):
    """Generation-time only: our own reference/buggy sources, run in-process."""
    source = view.read(key["path"])
    if source is None:
        return {}
    ns = {}
    try:
        exec(compile(source, key["path"], "exec"), ns)  # noqa: S102 - trusted generator output
    except Exception:
        return {}
    return json.loads(json.dumps(observe(ns, _inputs(key["checks"]))))


def _observe_isolated(key, view, key_dir, log):
    box = _Sandbox(False, key_dir)
    work = box.workdir()
    try:
        source = view.read(key["path"])
        if source is None:
            log.append("missing or unreadable: " + key["path"])
        else:
            _place(work, key["module"] + ".py", source)
        _place(work, "child.py", _child_source(key["module"]).encode())
        for name in os.listdir(work):
            os.chmod(os.path.join(work, name), 0o644)
        code, stdout, _ = box.run([sys.executable, "-I", "-B", os.path.join(work, "child.py"), work], work, json.dumps(_inputs(key["checks"])), CHILD_TIMEOUT_SEC)
        log.append("child exit code %s" % code)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    lines = [ln[len(MARKER):] for ln in stdout.splitlines() if ln.startswith(MARKER)]
    if len(lines) != 1:
        log.append("expected exactly one observation line, got %d" % len(lines))
        return {}
    try:
        return json.loads(lines[0])
    except ValueError:
        log.append("observation line is not JSON")
        return {}


def grade_checks(key, view, trusted=False, key_dir=None):
    if not key["checks"]:
        raise GraderError("a checks key needs at least one check")
    log = []
    obs = _observe_trusted(key, view) if trusted else _observe_isolated(key, view, key_dir, log)
    results = judge(key["checks"], obs)
    log += [("PASS " if ok else "FAIL ") + name for name, ok in results.items()]
    passed = sum(1 for ok in results.values() if ok)
    log.append("%d/%d checks passed" % (passed, len(results)))
    return passed / len(results), log


# --------------------------------------------------------------------------- #
# commands: run agent-written programs on hidden inputs
# --------------------------------------------------------------------------- #


def grade_commands(key, view, trusted=False, key_dir=None):
    """Each check: {"name", "argv", optional "stdin", "inputs" {rel: text}, and what must match:
    "stdout" (trailing newlines ignored), "exit", "outputs" {rel: text}}. The artifacts in
    `files` (absolute /app paths) are copied to the same paths relative to the scratch dir."""
    if not key["checks"]:
        raise GraderError("a commands key needs at least one check")
    box = _Sandbox(trusted, key_dir)
    sources = {path: view.read(path) for path in key["files"]}
    log, passed = [], 0
    for c in key["checks"]:
        work = box.workdir()
        try:
            for path, data in sources.items():
                if data is not None:
                    _place(work, os.path.relpath(path, key.get("workdir", "/app")), data)
            for rel, text in c.get("inputs", {}).items():
                _place(work, rel, text.encode())
            box.handover(work)
            code, stdout, stderr = box.run(c["argv"], work, c.get("stdin", ""), key.get("timeout_sec", 10))
            problems = []
            if code is None:
                problems.append(stderr)
            if "exit" in c and code != c["exit"]:
                problems.append("exit %s, expected %s" % (code, c["exit"]))
            if "stdout" in c and stdout.rstrip("\n") != c["stdout"].rstrip("\n"):
                problems.append("stdout differs: %r" % stdout[:200])
            for rel, want in c.get("outputs", {}).items():
                got = _text(read_artifact(work, rel))
                if got is None or got.rstrip("\n") != want.rstrip("\n"):
                    problems.append("output %s differs or is missing" % rel)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        passed += not problems
        log.append(("PASS " if not problems else "FAIL ") + c["name"] + ("" if not problems else ": " + "; ".join(problems)))
    log.append("%d/%d checks passed" % (passed, len(key["checks"])))
    return passed / len(key["checks"]), log


def grade(key, view, trusted=False, key_dir=None):
    """(reward, log lines) for `key` over `view` (a DiskView or MemoryView). `trusted=True`
    runs agent-code graders in this process or as the current user: generation only."""
    kind = key.get("kind")
    if kind == "exact":
        return grade_exact(key, view)
    if kind == "numeric":
        return grade_numeric(key, view)
    if kind == "parsed":
        return grade_parsed(key, view)
    if kind == "tree":
        return grade_tree(key, view)
    if kind == "checks":
        return grade_checks(key, view, trusted=trusted, key_dir=key_dir)
    if kind == "commands":
        return grade_commands(key, view, trusted=trusted, key_dir=key_dir)
    raise GraderError("unknown grader kind %r" % (kind,))


def remove_reward_files(out_dir):
    for name in ("reward.json", "reward.txt"):
        path = os.path.join(out_dir, name)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        elif os.path.lexists(path):
            os.unlink(path)


def write_reward(out, value):
    """Replace any planted reward file, then write {"reward": value} without following links."""
    remove_reward_files(os.path.dirname(out) or ".")
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o644)
    with os.fdopen(fd, "w") as f:
        json.dump({"reward": value}, f)


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/")
    ap.add_argument("--out", default="/logs/verifier/reward.json")
    ap.add_argument("--key", default=os.path.join(here, "key.json"))
    ap.add_argument("--probe", default=None, help="JSON expectation; grade only if the environment matches it")
    a = ap.parse_args(argv)
    want = os.environ.get(TESTS_DIGEST_ENV)
    if want is not None and tests_digest(here) != want:
        remove_reward_files(os.path.dirname(a.out) or ".")
        print("verifier files differ from the rendered task (stale image build); not grading")
        return EXIT_STALE_FILES
    if want is not None:
        print("verifier files match the rendered task")
    if a.probe is not None:
        sys.path.insert(0, here)
        import probe

        result = probe.run(json.loads(a.probe))
        print("probe: " + json.dumps(result, sort_keys=True))
        if not result["ok"]:
            remove_reward_files(os.path.dirname(a.out) or ".")
            print("environment probe failed; not grading: " + ", ".join(result["violations"]))
            return EXIT_PROBE_FAILED
    with open(a.key) as f:
        key = json.load(f)
    try:
        reward, log = grade(key, DiskView(a.root), key_dir=os.path.dirname(os.path.abspath(a.key)))
    except GraderError as e:
        remove_reward_files(os.path.dirname(a.out) or ".")
        print("grader error; no reward written: %s" % e)
        return EXIT_GRADER_ERROR
    for line in log:
        print(line)
    write_reward(a.out, reward)
    print("reward: %r" % reward)
    return 0


if __name__ == "__main__":
    sys.exit(main())
