"""Hidden verifier tests (separate verifier container). Writes partial credit to
/logs/verifier/reward.json. Hand-written task; rendered by evaluation.generators.fix_stats.hidden_test_source (keep them in sync).

The agent's artifact (/app/stats.py) is never imported by this process. A copy runs in a
child `python3 -I -B` process as the unprivileged user `nobody`, which cannot read /tests
(made 0700 first) and gets only the check inputs. The child prints raw observations as one
marker line; this process accepts only strictly typed observations for known checks,
compares them with the expected values, kills every process left by that user, removes any
reward file it may have planted, and only then writes reward.json.
"""

import json
import os
import pwd
import shutil
import signal
import subprocess
import sys
import tempfile
import time

CHECKS = json.loads('[{"args": [[1, 2, 3, 4]], "expected": 2.5, "func": "mean", "kind": "value", "name": "mean"}, {"args": [[3, 1, 2]], "expected": 2, "func": "median", "kind": "value", "name": "median_odd"}, {"args": [[4, 1, 3, 2]], "expected": 2.5, "func": "median", "kind": "value", "name": "median_even"}, {"args": [[3, 1, 2]], "func": "median", "kind": "no_mutation", "name": "median_no_mutation"}, {"abs_tol": 1e-06, "args": [[2, 4, 4, 4, 5, 5, 7, 9]], "expected": 2.138089935, "func": "stdev", "kind": "value", "name": "stdev_sample"}, {"args": [[10, 20, 30, 40], 0], "expected": 10, "func": "percentile", "kind": "value", "name": "percentile_0"}, {"args": [[10, 20, 30, 40], 100], "expected": 40, "func": "percentile", "kind": "value", "name": "percentile_100"}, {"args": [[10, 20, 30, 40], 50], "expected": 25, "func": "percentile", "kind": "value", "name": "percentile_50"}, {"abs_tol": 1e-09, "args": [[1, 2, 3, 4, 5], 90], "expected": 4.6, "func": "percentile", "kind": "value", "name": "percentile_90"}]')
ARTIFACT = "/app/stats.py"
VERIFIER_DIR = "/logs/verifier"
CHILD_TIMEOUT_SEC = 30
CHILD_SOURCE = 'import json\nimport sys\n\nMARKER = "@@FIX_STATS_OBSERVATIONS@@ "\n\ndef observe(lib, checks):\n    import copy\n    import json\n\n    def plain(v):\n        try:\n            return json.loads(json.dumps(v, allow_nan=False))\n        except Exception:\n            return None\n\n    out = {}\n    for c in checks:\n        fn = lib.get(c["func"]) if isinstance(lib, dict) else None\n        if not callable(fn):\n            out[c["name"]] = {"missing": True}\n            continue\n        args = copy.deepcopy(c["args"])\n        try:\n            got = fn(*args)\n        except BaseException as e:  # noqa: BLE001 - anything the artifact raises is an observation\n            out[c["name"]] = {"raised": type(e).__name__[:200], "value_error": isinstance(e, ValueError)}\n            continue\n        if c["kind"] == "no_mutation":\n            out[c["name"]] = {"after": plain(args[0])}\n        elif c["kind"] == "raises":\n            out[c["name"]] = {"returned": True}\n        else:\n            ok_type = isinstance(got, (int, float)) and not isinstance(got, bool)\n            out[c["name"]] = {"value": plain(got) if ok_type else None}\n    return out\n\nsys.path.insert(0, sys.argv[1])\nchecks = json.loads(sys.stdin.read())\ntry:\n    import stats\n\n    lib = dict(vars(stats))\nexcept BaseException as e:  # noqa: BLE001\n    print("import failed: " + type(e).__name__)\n    lib = None\nobs = observe(lib, checks) if lib is not None else {}\nsys.stdout.write("\\n" + MARKER + json.dumps(obs) + "\\n")\nsys.stdout.flush()\n'
MARKER = "@@FIX_STATS_OBSERVATIONS@@ "

def judge(checks, observations):
    import json
    import math

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


def kill_user_processes(uid):
    for _ in range(50):
        found = False
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                with open(f"/proc/{pid}/status") as f:
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


def observe_isolated():
    if os.geteuid() != 0:
        print("grader must run as root to isolate the artifact; failing closed")
        return {}
    nobody = pwd.getpwnam("nobody")
    os.chmod("/tests", 0o700)
    work = tempfile.mkdtemp(prefix="grade-")
    stdout = ""
    try:
        os.chmod(work, 0o755)
        if os.path.isfile(ARTIFACT) and not os.path.islink(ARTIFACT):
            with open(ARTIFACT, "rb") as src, open(os.path.join(work, "stats.py"), "wb") as dst:
                dst.write(src.read())
        else:
            print("missing artifact: " + ARTIFACT)
        with open(os.path.join(work, "child.py"), "w") as f:
            f.write(CHILD_SOURCE)
        for name in os.listdir(work):
            os.chmod(os.path.join(work, name), 0o644)
        inputs = [{k: v for k, v in c.items() if k not in ("expected", "abs_tol")} for c in CHECKS]
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-B", os.path.join(work, "child.py"), work],
                input=json.dumps(inputs), capture_output=True, text=True, cwd=work,
                user=nobody.pw_uid, group=nobody.pw_gid, extra_groups=[], start_new_session=True,
                env={"PATH": "/usr/local/bin:/usr/bin:/bin", "LC_ALL": "C.UTF-8", "HOME": work},
                timeout=CHILD_TIMEOUT_SEC,
            )
            stdout = proc.stdout
            print(f"child exit code {proc.returncode}")
        except subprocess.TimeoutExpired:
            print(f"child timed out after {CHILD_TIMEOUT_SEC}s")
    finally:
        kill_user_processes(nobody.pw_uid)
        shutil.rmtree(work, ignore_errors=True)
    lines = [ln[len(MARKER):] for ln in stdout.splitlines() if ln.startswith(MARKER)]
    if len(lines) != 1:
        print(f"expected exactly one observation line, got {len(lines)}")
        return {}
    try:
        return json.loads(lines[0])
    except ValueError:
        print("observation line is not JSON")
        return {}


def write_reward(value):
    for name in ("reward.json", "reward.txt"):
        path = os.path.join(VERIFIER_DIR, name)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        elif os.path.lexists(path):
            os.unlink(path)
    fd = os.open(os.path.join(VERIFIER_DIR, "reward.json"), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(fd, "w") as f:
        # reward.json may hold several numeric keys; Harbor averages each key across
        # trials (a key missing from some tasks counts as 0 there, so keep keys uniform).
        json.dump({"reward": value}, f)


results = judge(CHECKS, observe_isolated())
for name, ok in results.items():
    print(("PASS " if ok else "FAIL ") + name)
passed = sum(1 for ok in results.values() if ok is True)
write_reward(passed / len(CHECKS))
print(f"{passed}/{len(CHECKS)} checks passed")
