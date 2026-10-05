"""Environment probe: checks that a running task container matches its profile.

Standard-library only: the renderer copies this file into every task's `tests/` directory
(the grader runs it before grading), and the episode runs the same source in the agent
container before turn 0. The expectation comes from the task's profile:

    {"network": "none", "env": {"TZ": "UTC", ...}, "tools": ["python3", "timeout"],
     "hostname": "task", "allowed_processes": ["sleep"]}

    python3 -I -B probe.py '<expectation json>'   # prints {"ok", "violations", "observed"}

Checks, each reported as one violation string when it fails:

    egress:dns, egress:tcp   with network "none", a DNS lookup and a TCP connection to a
                             public address must both fail
    env:<NAME>               the variable has exactly the expected value
    tool:<name>              the tool is on PATH
    hostname                 the container hostname is the expected one
    process:<comm>           every process other than this probe and its ancestors (the exec
                             chain) has an allowed command name; `/proc` is required
"""

import json
import os
import shutil
import socket
import sys
import threading

EGRESS_HOST = "example.com"
EGRESS_ADDR = ("1.1.1.1", 53)
EGRESS_TIMEOUT_SEC = 2.0


def _dns_open():
    result = []

    def lookup():
        try:
            socket.getaddrinfo(EGRESS_HOST, 80)
            result.append(True)
        except OSError:
            result.append(False)

    t = threading.Thread(target=lookup, daemon=True)
    t.start()
    t.join(EGRESS_TIMEOUT_SEC)
    return bool(result and result[0])  # an unanswered lookup counts as blocked


def _tcp_open():
    try:
        socket.create_connection(EGRESS_ADDR, timeout=EGRESS_TIMEOUT_SEC).close()
        return True
    except OSError:
        return False


def _proc_stat(pid):
    with open("/proc/%s/stat" % pid) as f:
        data = f.read()
    comm = data[data.index("(") + 1 : data.rindex(")")]
    ppid = int(data[data.rindex(")") + 2 :].split()[1])
    return comm, ppid


def _processes():
    """[(pid, comm)] of processes other than this one and its ancestors; None without /proc."""
    if not os.path.isdir("/proc"):
        return None
    ancestors = set()
    pid = os.getpid()
    while pid > 1 and pid not in ancestors:
        ancestors.add(pid)
        try:
            pid = _proc_stat(pid)[1]
        except (OSError, ValueError, IndexError):
            break
    out = []
    for name in sorted(os.listdir("/proc"), key=lambda s: (len(s), s)):
        if not name.isdigit() or int(name) in ancestors:
            continue
        try:
            out.append((int(name), _proc_stat(name)[0]))
        except (OSError, ValueError, IndexError):
            continue  # exited while we looked
    return out


def observe(expect):
    network = expect.get("network")
    return {
        "egress": {"dns": _dns_open(), "tcp": _tcp_open()} if network == "none" else None,
        "env": {k: os.environ.get(k) for k in sorted(expect.get("env", {}))},
        "tools": {t: shutil.which(t) for t in expect.get("tools", [])},
        "hostname": socket.gethostname(),
        "processes": _processes(),
    }


def check(observed, expect):
    v = []
    if expect.get("network") == "none":
        egress = observed.get("egress") or {}
        v += ["egress:" + k for k in ("dns", "tcp") if egress.get(k) is not False]
    for k, want in sorted(expect.get("env", {}).items()):
        if observed["env"].get(k) != want:
            v.append("env:" + k)
    v += ["tool:" + t for t, path in sorted(observed["tools"].items()) if not path]
    if "hostname" in expect and observed["hostname"] != expect["hostname"]:
        v.append("hostname")
    if "allowed_processes" in expect:
        procs = observed["processes"]
        if procs is None:
            v.append("process:no_proc")
        else:
            allowed = set(expect["allowed_processes"])
            v += sorted({"process:" + comm for _, comm in procs if comm not in allowed})
    return v


def run(expect):
    observed = observe(expect)
    violations = check(observed, expect)
    return {"ok": not violations, "violations": violations, "observed": observed}


def main(argv):
    result = run(json.loads(argv[1]))
    sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
    return 0 if result["ok"] else 3


if __name__ == "__main__":
    sys.exit(main(sys.argv))
