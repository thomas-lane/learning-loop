"""Pod-side idle watchdog: stop (or terminate) this Runpod pod when the coordinator's heartbeat goes stale.

Started on the pod by the coordinator (learning_loop.hosts.pods), one instance per pod:

    python3 scripts/pod_watchdog.py HEARTBEAT_FILE IDLE_SEC INTERVAL_SEC LOG_FILE [stop|terminate]

`terminate` is used for pods the coordinator created; if the pod-scoped key may not terminate
its pod, the watchdog stops it instead (no GPU billing; `loop pod cleanup` removes it later).

The coordinator touches HEARTBEAT_FILE periodically (heartbeat_sec) while a command runs. If the file is
older than IDLE_SEC (laptop asleep, coordinator crashed, network lost), the pod stops itself with
the pod-scoped RUNPOD_API_KEY that Runpod injects into every pod (read from PID 1's environment,
since SSH sessions may not inherit it), via `runpodctl stop pod` when available, else the REST
API. The account API key never leaves the coordinator. Standard library only (system python3).
WATCHDOG_STOP_CMD replaces the action in tests (it gets WATCHDOG_ACTION=stop|terminate).
"""

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone


def log(path: str, msg: str) -> None:
    with open(path, "a") as f:
        f.write(json.dumps({"t": datetime.now(timezone.utc).isoformat(), "msg": msg}) + "\n")


def pod_env() -> dict[str, str]:
    env = dict(os.environ)
    try:
        with open("/proc/1/environ", "rb") as f:
            for kv in f.read().split(b"\0"):
                k, _, v = kv.decode(errors="replace").partition("=")
                if k.startswith("RUNPOD_") and k not in env:
                    env[k] = v
    except OSError:
        pass
    return env


def _pod_action(env: dict[str, str], action: str) -> tuple[bool, str]:
    """`stop` or `terminate` this pod with the pod-scoped key: runpodctl, else the REST API."""
    pod, key = env.get("RUNPOD_POD_ID"), env.get("RUNPOD_API_KEY")
    if not pod:
        return False, "RUNPOD_POD_ID not available"
    if shutil.which("runpodctl"):
        argv = ["runpodctl", "stop" if action == "stop" else "remove", "pod", pod]
        r = subprocess.run(argv, env={**os.environ, **env}, capture_output=True, text=True)
        if r.returncode == 0:
            return True, f"runpodctl {argv[1]}"
        detail = f"runpodctl exit {r.returncode}: {(r.stderr or r.stdout).strip()[:200]}"
    else:
        detail = "runpodctl not installed"
    if not key:
        return False, f"{detail}; RUNPOD_API_KEY not available"
    url, method = (f"https://rest.runpod.io/v1/pods/{pod}/stop", "POST") if action == "stop" else (f"https://rest.runpod.io/v1/pods/{pod}", "DELETE")
    req = urllib.request.Request(url, method=method, headers={"Authorization": f"Bearer {key}", "User-Agent": "learning-loop-watchdog/1"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return 200 <= r.status < 300, f"rest {method} {r.status} (after {detail})"
    except Exception as e:  # noqa: BLE001 - logged and retried
        return False, f"{detail}; rest {method} failed: {type(e).__name__}: {e}"


def stop_pod(env: dict[str, str], action: str = "stop") -> tuple[bool, str]:
    override = os.environ.get("WATCHDOG_STOP_CMD")
    if override:
        return subprocess.call(override, shell=True, env={**os.environ, "WATCHDOG_ACTION": action}) == 0, f"override {action}"
    if action == "terminate":
        ok, how = _pod_action(env, "terminate")
        if ok:
            return ok, how
        ok2, how2 = _pod_action(env, "stop")  # at least stop GPU billing
        return ok2, f"terminate failed ({how}); stop: {how2}"
    return _pod_action(env, "stop")


def main() -> int:
    heartbeat, idle, interval, logfile = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    action = sys.argv[5] if len(sys.argv) > 5 else "stop"
    if action not in ("stop", "terminate"):
        raise SystemExit(f"unknown action {action!r}")
    started = time.time()
    log(logfile, f"watchdog started pid={os.getpid()} idle={idle:.0f}s interval={interval:.0f}s action={action}")
    while True:
        time.sleep(interval)
        try:
            last = os.path.getmtime(heartbeat)
        except OSError:
            last = started  # no heartbeat yet: count from start
        age = time.time() - last
        if age <= idle:
            continue
        ok, how = stop_pod(pod_env(), action)
        log(logfile, f"heartbeat stale ({age:.0f}s > {idle:.0f}s): {action} {'requested' if ok else 'FAILED'} via {how}")
        if ok:
            return 0


if __name__ == "__main__":
    sys.exit(main())
