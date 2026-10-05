"""Shared task grader.

Standard-library only: the renderer copies this file verbatim into every task's `tests/`
directory, where the separate verifier container runs it, and generation imports it on the
host to grade each instance's oracle, shortcut and no-op outputs with the same code.

    python3 -I -B grade.py --root / --out /logs/verifier/reward.json [--key key.json] [--probe '<json>']

`--root` is the directory that stands for the container's `/` (artifacts are read from
`<root>/<path>`); `--key` defaults to `key.json` next to this file; `--probe` runs
`probe.py` (also next to this file) first and refuses to grade when the environment
violates the expectation. The grader writes exactly one key, `reward`, in [0, 1].

Grader kinds (the `kind` field of the key):

    exact    text of `path`, stripped, equals `expected`
    numeric  number in `path` is within max(abs_tol, rel_tol * |expected|) of `expected`
    json     JSON in `path` equals `expected` (object key order ignored)
    checks   call functions of the Python module at `path` with hidden inputs; reward is the
             fraction of checks passed

Only `checks` executes agent-written code. In the verifier it runs in a child
`python3 -I -B` process as the unprivileged user `nobody`, which cannot read the key (its
directory is made 0700 first) and receives only the check inputs. The child prints raw
observations on one marker line; this process accepts only plain JSON numbers and literal
booleans for known checks, so an object whose `__eq__` always returns true still fails.
Every `nobody` process is killed before the reward is written.
"""

import argparse
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


class GraderError(Exception):
    """The grader cannot produce a trustworthy reward; no reward is written."""


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


def _text(data):
    if data is None:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def grade_exact(key, read):
    text = _text(read(key["path"]))
    if text is None:
        return 0.0, ["missing or unreadable: " + key["path"]]
    ok = text.strip() == key["expected"]
    return (1.0 if ok else 0.0), ["expected: %r" % key["expected"], "actual:   %r" % text.strip()[:200]]


def grade_numeric(key, read):
    text = _text(read(key["path"]))
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


def grade_json(key, read):
    text = _text(read(key["path"]))
    if text is None:
        return 0.0, ["missing or unreadable: " + key["path"]]
    try:
        value = json.loads(text, parse_constant=_reject_constant)
    except ValueError as e:
        return 0.0, ["not valid JSON: %s" % e]
    canon = lambda v: json.dumps(v, sort_keys=True, separators=(",", ":"))  # noqa: E731
    ok = canon(value) == canon(key["expected"])
    return (1.0 if ok else 0.0), ["expected: " + canon(key["expected"])[:500], "actual:   " + canon(value)[:500]]


# --------------------------------------------------------------------------- #
# checks: call functions of an agent-written module
# --------------------------------------------------------------------------- #

OBSERVE = '''
def observe(lib, checks):
    import copy
    import json

    def plain(v):
        try:
            return json.loads(json.dumps(v, allow_nan=False))
        except Exception:
            return None

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
            out[c["name"]] = {"after": plain(args[0])}
        elif c["kind"] == "raises":
            out[c["name"]] = {"returned": True}
        else:
            ok_type = isinstance(got, (int, float)) and not isinstance(got, bool)
            out[c["name"]] = {"value": plain(got) if ok_type else None}
    return out
'''

exec(OBSERVE)  # defines observe() here too, from the exact source the child runs  # noqa: S102


def judge(checks, observations):
    """Check name -> passed. Accepts only the observation shapes `observe` produces."""

    def num(x):
        return type(x) in (int, float) and math.isfinite(x)

    results = {}
    obs_all = observations if isinstance(observations, dict) else {}
    for c in checks:
        o = obs_all.get(c["name"])
        ok = False
        if isinstance(o, dict):
            if c["kind"] == "raises":
                ok = set(o) == {"raised", "value_error"} and o["value_error"] is True
            elif c["kind"] == "no_mutation":
                ok = set(o) == {"after"} and json.dumps(o["after"]) == json.dumps(c["args"][0])
            elif set(o) == {"value"} and num(o["value"]):
                exp = c["expected"]
                tol = c.get("abs_tol")
                ok = abs(o["value"] - exp) <= (tol if tol is not None else 1e-9 * max(1.0, abs(exp)))
        results[c["name"]] = ok is True
    return results


def _inputs(checks):
    return [{k: v for k, v in c.items() if k not in ("expected", "abs_tol")} for c in checks]


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


def _observe_isolated(key, read, key_dir, log):
    if os.geteuid() != 0:
        raise GraderError("the checks grader must run as root to isolate the artifact")
    nobody = pwd.getpwnam("nobody")
    os.chmod(key_dir, 0o700)
    work = tempfile.mkdtemp(prefix="grade-")
    stdout = ""
    try:
        os.chmod(work, 0o755)
        source = read(key["path"])
        if source is None:
            log.append("missing or unreadable: " + key["path"])
        else:
            with open(os.path.join(work, key["module"] + ".py"), "wb") as f:
                f.write(source)
        with open(os.path.join(work, "child.py"), "w") as f:
            f.write(_child_source(key["module"]))
        for name in os.listdir(work):
            os.chmod(os.path.join(work, name), 0o644)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-B", os.path.join(work, "child.py"), work],
                input=json.dumps(_inputs(key["checks"])), capture_output=True, text=True, cwd=work,
                user=nobody.pw_uid, group=nobody.pw_gid, extra_groups=[], start_new_session=True,
                env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LC_ALL": "C.UTF-8", "HOME": work},
                timeout=CHILD_TIMEOUT_SEC,
            )
            stdout = proc.stdout
            log.append("child exit code %d" % proc.returncode)
        except subprocess.TimeoutExpired:
            log.append("child timed out after %ds" % CHILD_TIMEOUT_SEC)
    finally:
        _kill_user_processes(nobody.pw_uid)
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


def _observe_trusted(key, read):
    """Generation-time only: our own reference/buggy sources, run in-process."""
    source = read(key["path"])
    if source is None:
        return {}
    ns = {}
    try:
        exec(compile(source, key["path"], "exec"), ns)  # noqa: S102 - trusted generator output
    except Exception:
        return {}
    return json.loads(json.dumps(observe(ns, _inputs(key["checks"]))))


def grade_checks(key, read, trusted=False, key_dir=None):
    if not key["checks"]:
        raise GraderError("a checks key needs at least one check")
    log = []
    obs = _observe_trusted(key, read) if trusted else _observe_isolated(key, read, key_dir, log)
    results = judge(key["checks"], obs)
    log += [("PASS " if ok else "FAIL ") + name for name, ok in results.items()]
    passed = sum(1 for ok in results.values() if ok)
    log.append("%d/%d checks passed" % (passed, len(results)))
    return passed / len(results), log


def grade(key, read, trusted=False, key_dir=None):
    """(reward, log lines) for `key`; `read(path)` returns an artifact's bytes or None.
    `trusted=True` runs `checks` in-process and is only for generation-time self-checks."""
    kind = key.get("kind")
    if kind == "exact":
        return grade_exact(key, read)
    if kind == "numeric":
        return grade_numeric(key, read)
    if kind == "json":
        return grade_json(key, read)
    if kind == "checks":
        return grade_checks(key, read, trusted=trusted, key_dir=key_dir)
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
        reward, log = grade(key, lambda p: read_artifact(a.root, p), key_dir=os.path.dirname(os.path.abspath(a.key)))
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
