"""Live, read-only dashboard for run directories (`loop dashboard`).

Serves an HTML view of one run, or an index of every run under the runs directory, on a local
HTTP server. A run page shows: a header (status, current cycle and stage, elapsed time,
estimated pod spend); the unassisted evaluation across cycles (charts, a per-cycle table with
deltas against cycle 0 on matched items, the paired comparison, family and difficulty
breakdowns); the cycle timeline with stage progress; outcomes; recent episodes with transcript
pages (grader output, the editor's targeted turn); proposals and verifications; pod and serving
events; and the tail of the coordinator log. Pages reload every 10 s unless paused or a section
is expanded.

Episode rows, success, token totals, stop categories and paired comparisons use the definitions
of `loop report` (`metrics.make_rows`, `summarize`, `paired_comparison` over the stage items
`report.collect_run` reads), so the numbers match the report of the same files. The run may be
in progress: every file is read tolerantly (missing or half-written JSON and torn JSONL lines
show as n/a) and nothing is ever written, locked or modified. Only files inside the served run
directories are read (paths from URLs and symlinks pointing elsewhere are refused); the
operator-supplied console log and the created-pod ledger are the only other files read. Binds to
127.0.0.1 unless told otherwise.
"""

from __future__ import annotations

import html
import json
import math
import os
import re
import socket
import statistics
import threading
import time
import traceback
import webbrowser
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, quote, unquote, urlsplit

from pydantic import ValidationError

from .metrics import EpisodeRow, _pair_key, agg, eval_rows, make_rows, paired_comparison, summarize
from .records import EpisodeRole, EpisodeSummary, TaskInstance
from .report import run_model_identity

STAGES = ["eval", "collect", "edit", "verify", "audit"]
STATES = ["done", "running", "pending", "failed", "infra_failed"]
DIFF_ORDER = {"easy": 0, "medium": 1, "hard": 2}
BRANCH_RE = re.compile(r"^(original|edited)-r\d+$")
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")  # run ids and work-item ids
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
LOG_REL = "logs/coordinator.log"
REFRESH_SEC = 10
MAX_TEXT_BYTES = 2 << 20


def default_ledger() -> Path:
    from .pods import KNOWN_HOSTS_DIR, LEDGER_NAME

    return KNOWN_HOSTS_DIR / LEDGER_NAME


# --------------------------------------------------------------------------- #
# Tolerant reading (never raises for missing, half-written or torn files)
# --------------------------------------------------------------------------- #

_JSON_CACHE: dict[str, tuple[tuple[int, int], Any]] = {}
_DERIVED: dict[tuple[str, str], tuple[tuple[int, int], Any]] = {}
_TURNS: dict[str, tuple[int, int]] = {}


def _stat_key(p: Path) -> tuple[int, int] | None:
    try:
        st = p.stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def read_json(p: Path) -> Any:
    """Parsed JSON, cached by (mtime, size); None when missing or not (yet) valid. Callers must
    not mutate the result."""
    key = _stat_key(p)
    if key is None:
        return None
    hit = _JSON_CACHE.get(str(p))
    if hit and hit[0] == key:
        return hit[1]
    try:
        data = json.loads(p.read_bytes())
    except (OSError, ValueError):
        return None
    _JSON_CACHE[str(p)] = (key, data)
    return data


def derived(p: Path, name: str, fn: Callable[[Any], Any]) -> Any:
    """fn(parsed JSON of p), cached by the file's (mtime, size); None when the file is missing or
    not valid JSON."""
    key = _stat_key(p)
    if key is None:
        return None
    hit = _DERIVED.get((str(p), name))
    if hit and hit[0] == key:
        return hit[1]
    data = read_json(p)
    val = None if data is None else fn(data)
    _DERIVED[(str(p), name)] = (key, val)
    return val


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    """JSON objects of a JSONL file; torn or garbage lines are skipped."""
    out: list[dict[str, Any]] = []
    try:
        raw = p.read_text(errors="replace")
    except OSError:
        return out
    for line in raw.splitlines():
        try:
            v = json.loads(line)
        except ValueError:
            continue
        if isinstance(v, dict):
            out.append(v)
    return out


def read_text(p: Path, limit: int = MAX_TEXT_BYTES) -> str | None:
    try:
        with open(p, "rb") as f:
            data = f.read(limit + 1)
    except OSError:
        return None
    text = data[:limit].decode("utf-8", "replace")
    return text + (f"\n… [truncated at {limit:,} bytes]" if len(data) > limit else "")


def tail(p: Path, n: int = 40, nbytes: int = 65536) -> list[str] | None:
    try:
        with open(p, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - nbytes))
            data = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    lines = [ANSI_RE.sub("", ln.split("\r")[-1]) for ln in data.splitlines()]
    return lines[-n:]


def live_turns(p: Path) -> int | None:
    """Model responses in an events.jsonl so far, counted incrementally over complete lines."""
    off, n = _TURNS.get(str(p), (0, 0))
    try:
        if p.stat().st_size < off:
            off, n = 0, 0
        with open(p, "rb") as f:
            f.seek(off)
            chunk = f.read()
    except OSError:
        return None
    cut = chunk.rfind(b"\n")
    if cut >= 0:
        n += chunk[: cut + 1].count(b'"kind":"response"')  # events.jsonl is compact JSON (JsonlAppender)
        off += cut + 1
    _TURNS[str(p)] = (off, n)
    return n


def g(d: Any, *keys: Any, default: Any = None) -> Any:
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        elif isinstance(d, list) and isinstance(k, int) and -len(d) <= k < len(d):
            d = d[k]
        else:
            return default
    return default if d is None else d


def _int(v: Any) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _num(v: Any) -> float | int | None:
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _parse_summary(d: Any) -> EpisodeSummary | bool:
    try:
        return EpisodeSummary.model_validate(d)
    except (ValidationError, TypeError, ValueError):
        return False  # present but not a valid summary


def _parse_instances(d: Any) -> dict[str, TaskInstance]:
    out: dict[str, TaskInstance] = {}
    for k, v in _dict(_dict(d).get("instances")).items():
        try:
            out[k] = TaskInstance.model_validate(v)
        except (ValidationError, TypeError, ValueError):
            pass
    return out


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #


def esc(v: Any) -> str:
    return html.escape("n/a" if v is None else str(v), quote=True)


def parse_t(v: Any) -> float | None:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, str) and v:
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def fmt_t(ts: float | None) -> str:
    if ts is None:
        return "n/a"
    lt = time.localtime(ts)
    return time.strftime("%H:%M:%S" if lt[:3] == time.localtime()[:3] else "%b %d %H:%M", lt)


def fmt_dur(sec: float | None) -> str:
    if sec is None:
        return "n/a"
    sec = int(max(0, sec))
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}h {m:02d}m" if h else (f"{m}m {s:02d}s" if m else f"{s}s")


def fmt_num(v: Any, nd: int = 0) -> str:
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return "n/a"
    if nd == 0 and abs(v) >= 10000:
        return f"{v / 1000:.1f}k"
    return f"{v:,.{nd}f}"


def fmt_c(v: float | None, pct: bool = False) -> str:
    """Compact number for chart labels: 0.62 -> 62%, 12345 -> 12.3k."""
    if v is None:
        return "n/a"
    if pct:
        return f"{v * 100:.0f}%"
    if abs(v) >= 1000:
        return f"{v / 1000:.1f}k".replace(".0k", "k")
    return f"{v:.0f}" if abs(v) >= 10 else f"{v:.2g}"


def fmt_signed(v: float | None, nd: int = 0) -> str:
    if v is None:
        return "n/a"
    return ("+" if v > 0 else ("−" if v < 0 else "±")) + fmt_num(abs(v), nd)


def pill(text: Any, kind: Any = None) -> str:
    return f'<span class="pill {esc(kind or text)}">{esc(text)}</span>'


def trunc_pre(text: Any, n: int = 2000) -> str:
    text = "" if text is None else str(text)
    if len(text) <= n:
        return f"<pre>{esc(text)}</pre>"
    return (f"<pre>{esc(text[:n])}\n… [{len(text) - n:,} more chars]</pre>"
            f"<details><summary>show all {len(text):,} chars</summary><pre>{esc(text)}</pre></details>")


def bar(counts: Counter, total: int) -> str:
    total = max(total, sum(counts.values()), 1)
    segs = "".join(f'<i class="s-{s}" style="width:{100 * counts.get(s, 0) / total:.2f}%" title="{s}: {counts.get(s, 0)}"></i>'
                   for s in STATES if counts.get(s))
    return f'<div class="bar">{segs}</div>'


def table(headers: list[str], rows: list[list[str]], cls: str = "") -> str:
    th = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    if not rows:
        body = f'<tr><td colspan="{len(headers)}" class="muted">nothing yet</td></tr>'
    return f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def short_repl(rep: Any, n: int = 160) -> str:
    if rep is None:
        return "n/a"
    if isinstance(rep, dict):
        args = rep.get("arguments")
        s = f"{rep.get('name')}: " + (str(args["command"]) if isinstance(args, dict) and set(args) == {"command"} else json.dumps(args, ensure_ascii=False))
    else:
        s = str(rep)
    return f"<code>{esc(s[:n])}</code>" + (f"<details><summary>full</summary><pre>{esc(s)}</pre></details>" if len(s) > n else "")


# --------------------------------------------------------------------------- #
# One run: paths, containment, links
# --------------------------------------------------------------------------- #


def inside(root: Path, p: Path) -> bool:
    """True when p (after resolving symlinks and '..') lies within root (already resolved)."""
    try:
        p.resolve().relative_to(root)
    except (ValueError, OSError, RuntimeError):
        return False
    return True


@dataclass(frozen=True)
class RunView:
    """What a run page needs: the resolved run directory, its URL prefix, the console log to tail
    (None: logs/coordinator.log in the run) and the created-pod ledger."""

    dir: Path
    base: str
    log: Path | None = None
    ledger: Path | None = None
    index_link: bool = False

    # every read of a run file goes through these, so a symlink out of the run is never followed
    def json(self, p: Path) -> Any:
        return read_json(p) if inside(self.dir, p) else None

    def jsonl(self, p: Path) -> list[dict[str, Any]]:
        return read_jsonl(p) if inside(self.dir, p) else []

    def text(self, p: Path, limit: int = MAX_TEXT_BYTES) -> str | None:
        return read_text(p, limit) if inside(self.dir, p) else None

    def summary(self, p: Path) -> EpisodeSummary | bool | None:
        return derived(p, "summary", _parse_summary) if inside(self.dir, p) else None

    def item_dir(self, rel: Any) -> Path | None:
        """Run-relative (or absolute) directory recorded in a manifest, if it lies inside the run."""
        if not isinstance(rel, str) or not rel or "\x00" in rel:
            return None
        p = Path(rel)
        p = p if p.is_absolute() else self.dir / p
        return p if inside(self.dir, p) else None

    def rel(self, p: Path) -> str | None:
        try:
            return p.resolve().relative_to(self.dir).as_posix()
        except (ValueError, OSError):
            return None

    def ep_link(self, path: Any, label: Any) -> str:
        """Link to an episode directory's transcript page; plain text when it is outside this run
        (e.g. an edit replay's source run)."""
        d = self.item_dir(path)
        rel = self.rel(d) if d is not None else None
        if rel is None:
            return esc(label)
        return f'<a href="{self.base}/ep?p={quote(rel)}">{esc(label)}</a>'


def resolve_episode(run_dir: Path, rel: str) -> Path | None:
    """The episode directory a transcript URL names, or None unless it is an existing directory
    under the run's cycles/ (no '..', absolute paths or symlinks out)."""
    if not rel or "\x00" in rel or Path(rel).is_absolute():
        return None
    try:
        d = (run_dir / rel).resolve()
        parts = d.relative_to(run_dir).parts
    except (ValueError, OSError, RuntimeError):
        return None
    if len(parts) < 2 or parts[0] != "cycles" or not d.is_dir():
        return None
    return d


# --------------------------------------------------------------------------- #
# Run model (manifests, cycle records, training state)
# --------------------------------------------------------------------------- #


def load_state(v: RunView) -> dict[str, Any]:
    rd = v.dir
    run = _dict(v.json(rd / "run.json"))
    exp = _dict(run.get("experiment"))
    kind = run.get("kind") or "learning"
    learning = kind == "learning"
    panels = _dict(run.get("panels"))
    n_cycles = _int(exp.get("cycles")) if learning else None
    planned: dict[str, int] = {}
    if learning:
        dev = g(exp, "evaluation", "dev_panels", default=[])
        planned["eval"] = sum(len(panels.get(p) or []) for p in (dev if isinstance(dev, list) else [])) * (
            _int(g(exp, "evaluation", "attempts_per_instance")) or 0)
        planned["collect"] = len(panels.get(g(exp, "tasks", "collection_panel", default="train")) or []) * (
            _int(g(exp, "tasks", "attempts_per_instance")) or 0)
    try:
        cyc_dirs = sorted(p for p in (rd / "cycles").glob("cycle-[0-9][0-9][0-9]") if p.is_dir())
    except OSError:
        cyc_dirs = []
    last_idx = n_cycles if n_cycles is not None else len(cyc_dirs) - 1
    cycles = []
    for c in range(max(last_idx + 1, len(cyc_dirs))):
        cdir = rd / "cycles" / f"cycle-{c:03d}"
        cy: dict[str, Any] = {"n": c, "dir": cdir, "exists": cdir.is_dir(), "rec": _dict(v.json(cdir / "cycle.json")),
                              "final": learning and c == last_idx, "st": {}}
        for s in STAGES:
            man = v.json(cdir / s / "manifest.json")
            if not isinstance(man, dict):
                continue
            items = man.get("items") or {}
            items = [i for i in (items.values() if isinstance(items, dict) else items) if isinstance(i, dict)]
            cnt = Counter(i.get("status", "pending") for i in items)
            plan = planned.get(s)
            if s == "edit":
                col = cy["st"].get("collect")
                if col and col["status"] == "done":
                    plan = sum(1 for i in col["items"] if i.get("status") == "done" and g(i, "meta", "success") is True) * (
                        _int(g(exp, "editor", "proposals_per_source")) or 1)
            if s == "verify":
                ed = cy["st"].get("edit")
                if ed and ed["status"] == "done":
                    plan = sum(1 for i in ed["items"] if g(i, "meta", "status") == "proposed")
            if plan and plan > len(items) and man.get("status") != "done":
                cnt["pending"] += plan - len(items)
            active = 0.0
            for iv in man.get("active_intervals") or []:
                a, b = parse_t(g(iv, 0)), parse_t(g(iv, 1)) or time.time()
                if a:
                    active += max(0.0, b - a)
            total = sum(cnt.values())
            eta = None
            if man.get("status") != "done" and cnt.get("done") and total > cnt["done"]:
                eta = active / cnt["done"] * (cnt.get("pending", 0) + cnt.get("running", 0))
            cy["st"][s] = {"status": man.get("status", "n/a"), "cycle": man.get("cycle"), "items": items, "cnt": cnt,
                           "total": total, "active": active, "eta": eta}
        cy["dataset"] = v.json(cdir / "dataset" / "manifest.json")
        tdir = cdir / "train"
        cy["train"] = None
        if tdir.is_dir():
            done = (tdir / "checkpoint_path.txt").exists() or cy["rec"].get("update") == "trained"
            launch = _dict(v.json(tdir / "remote_launch.json"))
            req_t = None
            try:
                req_t = (tdir / "request.json").stat().st_mtime
            except OSError:
                pass
            cy["train"] = {"done": done, "host": launch.get("host", "local"), "since": parse_t(launch.get("started_at")) or req_t,
                           "relaunches": len(v.jsonl(tdir / "relaunches.jsonl")), "result": g(v.json(tdir / "work" / "result.json"), "status")}
        cycles.append(cy)
    return {
        "v": v, "dir": rd, "run": run, "exp": exp, "kind": kind, "cycles": cycles, "planned": planned,
        "run_id": run.get("run_id") or rd.name,
        "tasks": (derived(rd / "run.json", "instances", _parse_instances) if inside(rd, rd / "run.json") else None) or {},
        "collection_panel": g(exp, "tasks", "collection_panel"),
        "identity": run_model_identity(rd, run),
    }


def cycle_done(S: dict[str, Any], cy: dict[str, Any]) -> bool:
    if cy["rec"].get("status") == "done":
        return True
    if S["kind"] == "learning" or cy["rec"]:
        return False
    # evaluation / edit-replay runs keep no cycle.json: done when every stage is and reports exist
    sts = [st["status"] for st in cy["st"].values()]
    return bool(sts) and all(s in ("done", "skipped") for s in sts) and (S["dir"] / "reports").is_dir()


def current_position(S: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    for cy in S["cycles"]:
        if cycle_done(S, cy):
            continue
        if not cy["exists"]:
            return cy, "not started"
        stage = "starting"
        for s in STAGES:
            if s in cy["st"]:
                st = cy["st"][s]
                stage = f"{s} ({st['cnt'].get('done', 0)}/{st['total']} done)" if st["status"] != "done" else f"{s} done"
        if cy["dataset"] is not None:
            stage = f"dataset ({g(cy['dataset'], 'n_examples')} pairs)"
        if cy["train"]:
            t = cy["train"]
            stage = "train done" if t["done"] else f"train on {t['host']} ({fmt_dur(time.time() - t['since']) if t['since'] else 'n/a'})"
        return cy, stage
    return None, "all cycles done" if S["cycles"] else "no cycles yet"


def coordinator_alive(run_dir: Path) -> tuple[bool | None, dict[str, Any]]:
    """Whether the PID recorded in the run's .lock is alive, without taking the lock: True/False
    on this host, None when unknown (no holder recorded, or the holder ran on another host). The
    file keeps the last holder after it exits, hence the PID check."""
    lock = _dict(read_json(run_dir / ".lock")) if inside(run_dir, run_dir / ".lock") else {}
    pid = _int(lock.get("pid"))
    if pid is None:
        return None, lock
    if lock.get("host") not in {os.uname().nodename, socket.gethostname(), socket.getfqdn()}:
        return None, lock
    try:
        os.kill(pid, 0)
        return True, lock
    except ProcessLookupError:
        return False, lock
    except OSError:
        return True, lock  # exists, owned by another user


def run_status(S: dict[str, Any], alive: bool | None) -> tuple[str, str]:
    """(label, pill class) for the run as a whole."""
    cy, _ = current_position(S)
    if cy is None and S["cycles"]:
        return "finished", "done"
    if alive:
        return "running", "running"
    if alive is False:
        return "coordinator not running", "failed"
    return "unknown (no lock holder on this host)", "pending"


def pod_ledger(ledger: Path | None) -> list[dict[str, Any]]:
    pods: dict[str, dict[str, Any]] = {}
    for e in read_jsonl(ledger) if ledger else []:
        pid = e.get("pod_id")
        if not isinstance(pid, str) or not pid:
            continue
        p = pods.setdefault(pid, {"pod": pid, "name": None, "cost": None, "created": None, "ended": None})
        if e.get("event") == "created":
            p.update(name=e.get("name"), cost=_num(e.get("cost_per_hr")), created=parse_t(e.get("t")))
        elif e.get("event") == "terminated":
            p["ended"] = parse_t(e.get("t"))
    now = time.time()
    for p in pods.values():
        p["up"] = ((p["ended"] or now) - p["created"]) if p["created"] else None
        p["spend"] = p["up"] / 3600 * p["cost"] if p["up"] is not None and p["cost"] is not None else None
    return sorted(pods.values(), key=lambda p: p["created"] or 0)


# --------------------------------------------------------------------------- #
# Episode rows (the definitions `loop report` uses)
# --------------------------------------------------------------------------- #


def stage_cycles(S: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    """Per cycle: the episode rows of one stage (eval or collect), built like report.collect_run:
    every manifest item with an output directory whose summary.json parses, the panel from the
    item (or the collection panel), task metadata from run.json. Running and pending items have
    no output yet; failed items never have one; infra-failed items count, as in the report."""
    v: RunView = S["v"]
    role = EpisodeRole.EVAL if stage == "eval" else EpisodeRole.COLLECT
    out = []
    for cy in S["cycles"]:
        st = cy["st"].get(stage)
        if not st:
            continue
        rec: dict[str, Any] = {"n": cy["n"], "final": cy["final"], "status": st["status"], "cnt": st["cnt"], "total": st["total"],
                               "rows": [], "rel": {}, "unreadable": 0, "ckpt": g(cy["rec"], "learner_in", "checkpoint_id"),
                               "running": st["status"] != "done"}
        mcycle = _int(st.get("cycle"))
        for it in st["items"]:
            d = v.item_dir(it.get("output"))
            if d is None:
                continue
            p = d / "summary.json"
            s = v.summary(p)
            if not isinstance(s, EpisodeSummary):
                if p.exists():
                    rec["unreadable"] += 1
                continue
            panel = g(it, "meta", "panel") or (S["collection_panel"] if s.role == EpisodeRole.COLLECT else None)
            row = make_rows([s], S["tasks"], run_id=S["run_id"], cycle=cy["n"] if mcycle is None else mcycle, panel=panel,
                            model_identity=S["identity"])[0]
            rec["rows"].append(row)
            rec["rel"][s.episode_id] = v.rel(d)
        rec["rows"] = eval_rows(rec["rows"]) if role == EpisodeRole.EVAL else [r for r in rec["rows"] if r.episode.role == role]
        if not rec["ckpt"] and rec["rows"]:
            rec["ckpt"] = rec["rows"][0].episode.checkpoint_id
        out.append(rec)
    return out


def light_summary(rows: list[EpisodeRow]) -> dict[str, Any]:
    """The fields of metrics.summarize() the breakdowns show, with the same definitions (ungraded
    episodes count as not successful; token means over episodes that report usage), without its
    instance-level bootstraps."""
    eps = [r.episode for r in rows]
    n = len(eps)
    n_success = sum(1 for e in eps if e.success is True)
    n_graded = sum(1 for e in eps if e.success is not None)
    tot = agg(e.usage.total for e in eps)
    return {
        "n_episodes": n,
        "n_graded": n_graded,
        "n_ungraded": n - n_graded,
        "n_success": n_success,
        "success_rate": n_success / n if n else None,
        "mean_partial_reward": agg(e.partial_reward for e in eps)["mean"],
        "total_tokens_mean": tot["mean"],
        "n_with_tokens": tot["n"],
        "total_tokens_mean_successful": agg(e.usage.total for e in eps if e.success is True)["mean"],
        "stop_categories": dict(Counter(e.stop_category.value for e in eps)),
        "n_malformed_turns_sum": sum(e.n_malformed_turns for e in eps),
    }


def stats(rows: list[EpisodeRow], full: bool = True) -> dict[str, Any]:
    """metrics.summarize() (or light_summary) plus per-episode means the dashboard shows."""
    eps = [r.episode for r in rows]
    use = [e for e in eps if e.usage.total is not None]
    use_ok = [e for e in use if e.success is True]
    return {
        **(summarize(rows) if full else light_summary(rows)),
        "n_use_ok": len(use_ok),
        # total tokens of the episodes that report usage / successes among those same episodes
        "tps": sum(e.usage.total or 0 for e in use) / len(use_ok) if use_ok else None,
        "out": agg(e.usage.output_tokens for e in eps)["mean"],
        "turns": agg(e.n_requests for e in eps)["mean"],
        "tools": agg(e.n_tool_calls for e in eps)["mean"],
        "sec": agg(e.timing.total_sec for e in eps)["mean"],
        "n_malformed_eps": sum(1 for e in eps if e.n_malformed_turns),
    }


def family_of(r: EpisodeRow) -> str:
    return r.family or (r.episode.instance_id.split("/")[0] if r.episode.instance_id else "?")


def rate_text(s: dict[str, Any]) -> str:
    if not s["n_episodes"]:
        return "n/a"
    return f"{s['n_success']}/{s['n_episodes']} ({s['success_rate']:.0%})"


def all_episodes(S: dict[str, Any]) -> list[dict[str, Any]]:
    """Every episode directory (eval, collect, verify/audit branches) with its raw summary,
    running first, then newest first. Raw (not validated) so running episodes show too."""
    v: RunView = S["v"]
    out = []
    for cy in S["cycles"]:
        for s in ("eval", "collect"):
            for it in (cy["st"].get(s) or {}).get("items", []):
                iid = it.get("item_id")
                if not isinstance(iid, str) or not SAFE_NAME.match(iid):
                    continue
                d = cy["dir"] / s / "items" / iid
                out.append(_ep_row(v, d, cy["n"], s, it.get("status"), g(it, "meta", "instance_id"), parse_t(it.get("updated_at"))))
        for s in ("verify", "audit"):
            for it in (cy["st"].get(s) or {}).get("items", []):
                iid = it.get("item_id")
                if not isinstance(iid, str) or not SAFE_NAME.match(iid):
                    continue
                d = cy["dir"] / s / "items" / iid
                if not inside(v.dir, d):
                    continue
                try:
                    subs = sorted(p for p in d.iterdir() if p.is_dir() and BRANCH_RE.match(p.name))
                except OSError:
                    subs = []
                for b in subs:
                    sm = b / "episode" / "summary.json"
                    try:
                        t, done = sm.stat().st_mtime, True
                    except OSError:
                        t, done = parse_t(it.get("updated_at")), False
                    out.append(_ep_row(v, b, cy["n"], f"{s}:{b.name}", "done" if done else "running", None, t))
    out.sort(key=lambda r: (r["status"] != "running", -(r["t"] or 0)))
    return out


def _ep_row(v: RunView, d: Path, cycle: int, stage: str, status: Any, inst: Any, t: float | None) -> dict[str, Any]:
    summ = _dict(v.json(d / "summary.json")) or _dict(v.json(d / "episode" / "summary.json"))
    usage = _dict(summ.get("usage"))
    i, o = _int(usage.get("input_tokens")), _int(usage.get("output_tokens"))
    turns = summ.get("n_requests")
    if status == "running" and inside(v.dir, d / "episode" / "events.jsonl"):
        turns = live_turns(d / "episode" / "events.jsonl")
    return {"cycle": cycle, "stage": stage, "status": status, "t": t, "instance": inst or summ.get("instance_id"),
            "success": summ.get("success"), "partial": summ.get("partial_reward"), "stop": summ.get("stop_reason"),
            "turns": turns, "tokens": i + o if i is not None and o is not None else None, "sec": g(summ, "timing", "total_sec"),
            "rel": v.rel(d)}


# --------------------------------------------------------------------------- #
# Run page sections
# --------------------------------------------------------------------------- #


def run_pods(S: dict[str, Any]) -> list[dict[str, Any]]:
    """Ledger entries of the pods this run's pod-lifecycle.jsonl names."""
    v: RunView = S["v"]
    mine = {e.get("pod") for e in v.jsonl(v.dir / "logs" / "pod-lifecycle.jsonl") if e.get("event") in ("created", "ready")}
    return [p for p in pod_ledger(v.ledger) if p["pod"] in mine]


def sec_header(S: dict[str, Any]) -> str:
    v: RunView = S["v"]
    exp = S["exp"]
    alive, lock = coordinator_alive(v.dir)
    prov = _dict(v.json(v.dir / "provenance.json"))
    start = parse_t(prov.get("recorded_at")) or parse_t(lock.get("since"))
    cy, stage = current_position(S)
    status = run_status(S, alive)
    pods = run_pods(S)
    spend = sum(p["spend"] or 0 for p in pods)
    live = [p for p in pod_ledger(v.ledger) if p["created"] and not p["ended"]]
    learner = g(cy, "rec", "learner_in", "checkpoint_id") if cy else None
    n_c = exp.get("cycles")
    cyc = "–"
    if cy:
        cyc = f"{cy['n']}"
        if S["kind"] == "learning":
            cyc += f" <small>of {esc(n_c)} training cycles + final eval (cycle {esc(n_c)})</small>" + (" final evaluation" if cy["final"] else "")
    kv = [
        ("status", pill(*status)),
        ("kind", esc(S["kind"])),
        ("started", f"{esc(fmt_t(start))} · {esc(fmt_dur(time.time() - start) if start else 'n/a')} ago"),
        ("cycle", cyc),
        ("stage", esc(stage)),
        ("learner", f"<code>{esc(learner)}</code>"),
        ("pod spend (this run, est.)", (f"${spend:.2f}" if pods else "n/a")
         + "".join(f" · <code>{esc(p['pod'])}</code> up {fmt_dur(p['up'])} @ ${esc(p['cost'])}/h" for p in live)),
        ("coordinator", f"pid {esc(lock.get('pid'))} on {esc(lock.get('host'))}"),
    ]
    rows = "".join(f'<div class="kv"><span>{k}</span><b>{val}</b></div>' for k, val in kv)
    back = '<p><a href="/">← all runs</a></p>' if v.index_link else ""
    return (f'<header>{back}<h1>{esc(S["run_id"])}</h1><p class="muted">{esc(exp.get("name"))} — {esc(exp.get("description"))}'
            f' · learner {esc(g(exp, "learner", "model_profile"))} · editor {esc(g(exp, "editor", "mode"))}</p><div class="kvs">{rows}</div></header>')


def sec_timeline(S: dict[str, Any]) -> str:
    show = [s for s in STAGES if s != "audit" or any("audit" in c["st"] for c in S["cycles"])]
    rows = []
    for cy in S["cycles"]:
        rec = cy["rec"]
        state = rec.get("status") or ("done" if cycle_done(S, cy) else ("not started" if not cy["exists"] else "n/a"))
        cells = [f"<b>{cy['n']}</b>{' <small>final</small>' if cy['final'] else ''}<br>{pill(state, state if state in STATES else 'pending')}"]
        for s in show:
            st = cy["st"].get(s)
            if not st:
                cells.append('<span class="muted">–</span>')
                continue
            c = st["cnt"]
            detail = " ".join(f"{k} {c[k]}" for k in STATES if c.get(k))
            eta = f" · ETA ~{fmt_dur(st['eta'])}" if st["eta"] else ""
            cells.append(f"{pill(st['status'])} {c.get('done', 0)}/{st['total']}{bar(c, st['total'])}<small>{esc(detail)} · {fmt_dur(st['active'])}{eta}</small>")
        ds = cy["dataset"]
        cells.append(f"{esc(g(ds, 'n_examples'))} pairs" if ds is not None else '<span class="muted">–</span>')
        t, td, cr = cy["train"], _dict(rec.get("training_data")), _dict(rec.get("checkpoint_record"))
        losses = g(cr, "metrics", "losses", default=[])
        tr = '<span class="muted">–</span>'
        if t:
            tr = pill("done") if t["done"] else pill("running") + f" on {esc(t['host'])} for {fmt_dur(time.time() - t['since']) if t['since'] else 'n/a'}"
            if t["relaunches"]:
                tr += f" <small>{t['relaunches']} relaunch(es)</small>"
            if t["result"]:
                tr += f" <small>result: {esc(t['result'])}</small>"
        if isinstance(losses, list) and losses:
            tr += (f"<br><small>loss {fmt_num(losses[0], 3)} → {fmt_num(losses[-1], 3)} · {esc(cr.get('optimizer_steps'))} steps · "
                   f"{fmt_dur(_num(g(cr, 'metrics', 'durations_sec', 'total')))}</small>")
        cells.append(tr)
        upd = rec.get("update")
        out = g(rec, "learner_out", "checkpoint_id")
        drops = _dict(td.get("dropped_by_reason"))
        tdtxt = f"exported {esc(td.get('n_exported'))} / trained {esc(td.get('n_trained'))} / dropped {esc(td.get('n_dropped'))}" if td else ""
        if drops:
            tdtxt += " (" + ", ".join(f"{esc(k)}: {esc(val)}" for k, val in drops.items()) + ")"
        cells.append((pill(upd, "done" if upd == "trained" else "pending") if upd else '<span class="muted">–</span>')
                     + (f"<br><code>{esc(out)}</code>" if upd else "") + (f"<br><small>{tdtxt}</small>" if tdtxt else ""))
        rows.append(cells)
    return "<h2>Cycle timeline</h2>" + table(["cycle"] + show + ["dataset", "train", "update · checkpoint · training data"], rows, "timeline")


def sec_outcomes(S: dict[str, Any]) -> str:
    out = ["<h2>Outcomes</h2><p class='muted'>Cells: successes/episodes (rate; ungraded episodes count as not successful) · "
           "mean partial reward · mean tokens (in+out, episodes with usage) · mean turns. Grouped by family · difficulty from run.json.</p>"]
    for stage in ("eval", "collect"):
        grid: dict[tuple[str, str], dict[int, list[EpisodeRow]]] = defaultdict(lambda: defaultdict(list))
        for rec in stage_cycles(S, stage):
            for r in rec["rows"]:
                grid[(family_of(r), r.difficulty or "?")][rec["n"]].append(r)
                grid[("all", "")][rec["n"]].append(r)
        cyc = sorted({c for val in grid.values() for c in val})
        keys = sorted(grid, key=lambda k: (k[0] == "all", k[0], DIFF_ORDER.get(k[1], 9)))
        rows = []
        for k in keys:
            cells = [f"<b>{esc(k[0])}</b> {esc(k[1])}"]
            for c in cyc:
                rs = grid[k].get(c) or []
                if not rs:
                    cells.append('<span class="muted">–</span>')
                    continue
                s = stats(rs, full=False)
                rate = s["success_rate"] or 0.0
                cells.append(f'<b>{s["n_success"]}/{s["n_episodes"]}</b> ({rate:.0%})<div class="bar"><i class="s-done" style="width:{rate * 100:.0f}%"></i>'
                             f'<i class="s-failed" style="width:{100 - rate * 100:.0f}%"></i></div>'
                             f"<small>p {fmt_num(s['mean_partial_reward'], 2)} · tok {fmt_num(s['total_tokens_mean'])} · {fmt_num(s['turns'], 1)} turns</small>")
            rows.append(cells)
        label = "eval (unassisted, dev panels)" if stage == "eval" else "collect (training panel)"
        out.append(f"<h3>{label}</h3>" + table(["group"] + [f"cycle {c}" for c in cyc], rows))
    return "".join(out)


def sec_activity(S: dict[str, Any], eps: list[dict[str, Any]]) -> str:
    v: RunView = S["v"]
    rows = []
    for r in eps[:30]:
        if r["status"] == "running":
            succ = pill("running")
        elif r["success"] is True:
            succ = pill("ok", "done")
        elif r["success"] is False:
            succ = pill("fail", "failed")
        else:
            succ = pill(r["status"] or "n/a")
        label = (r["rel"] or "").rsplit("/items/", 1)[-1] or "?"
        rows.append([esc(fmt_t(r["t"])), f"{r['cycle']} · {esc(r['stage'])}", esc(r["instance"]), succ, esc(fmt_num(r["partial"], 2)),
                     esc(r["stop"]), esc(r["turns"]), esc(fmt_num(r["tokens"])), esc(fmt_num(r["sec"], 0)),
                     v.ep_link(r["rel"], label) if r["rel"] else "n/a"])
    return f"<h2>Recent activity <small class='muted'>({len(eps)} episodes total; running first)</small></h2>" + table(
        ["time", "cycle · stage", "instance", "result", "partial", "stop", "turns", "tokens", "sec", "transcript"], rows)


def sec_edits(S: dict[str, Any]) -> str:
    v: RunView = S["v"]
    out = ["<h2>Edits &amp; verification</h2>"]
    any_ = False
    for cy in S["cycles"]:
        ed = cy["st"].get("edit")
        if not ed:
            continue
        any_ = True
        vers: dict[Any, list[tuple[str, dict[str, Any], Path, Any]]] = {}
        for s in ("verify", "audit"):
            for it in (cy["st"].get(s) or {}).get("items", []):
                iid = it.get("item_id")
                if not isinstance(iid, str) or not SAFE_NAME.match(iid):
                    continue
                d = cy["dir"] / s / "items" / iid
                vers.setdefault(g(it, "meta", "proposal_id"), []).append((s, it, d, _dict(v.json(d / "verification.json"))))
        props, reasons = [], Counter()
        for it in ed["items"]:
            iid = it.get("item_id")
            p = _dict(v.json(cy["dir"] / "edit" / "items" / iid / "proposal.json")) if isinstance(iid, str) and SAFE_NAME.match(iid) else {}
            props.append((it, p))
            for rr in p.get("rejection_reasons") or []:
                reasons[str(rr).split(":", 1)[0]] += 1
        st_cnt = Counter(str(p.get("status") or g(it, "meta", "status") or it.get("status")) for it, p in props)
        out.append(f"<h3>cycle {cy['n']}: {len(props)} proposals — " + ", ".join(f"{esc(k)} {n}" for k, n in st_cnt.most_common())
                   + (" · rejection reasons: " + ", ".join(f"{esc(k)} ×{n}" for k, n in reasons.most_common()) if reasons else "") + "</h3>")
        rows = []
        for it, p in props:
            src = g(it, "meta", "source_dir")
            vcell = []
            for s, vit, d, ver in vers.get(p.get("proposal_id") or it.get("item_id"), []):
                if not ver:
                    vcell.append(f"{esc(s)}: {pill(vit.get('status') or 'n/a')}")
                    continue
                names = [f"{b.get('branch')}-r{_int(b.get('repetition')) or 0:02d}" for b in ver.get("branches") or [] if isinstance(b, dict)]
                links = " ".join(v.ep_link(str(d / n), n) for n in names)
                vcell.append(f"{esc(s)}: {pill('accepted', 'done') if ver.get('accepted') else pill('rejected', 'failed')} "
                             f"{esc(', '.join(map(str, ver.get('reasons') or [])))}<br><small>cost {fmt_num(ver.get('mean_cost_original'))} → "
                             f"{fmt_num(ver.get('mean_cost_edited'))} · saving <b>{fmt_num(ver.get('mean_saving'))}</b> · {links}</small>")
            status = p.get("status") or it.get("status") or "n/a"
            rows.append([f"<code>{esc(p.get('proposal_id') or it.get('item_id'))}</code>", esc(p.get("instance_id") or g(it, "meta", "instance_id")),
                         v.ep_link(src, g(it, "meta", "source_episode_id", default="source")) if src else "n/a",
                         pill(status, {"proposed": "done", "invalid": "failed"}.get(p.get("status"), "pending")),
                         esc(p.get("turn_index")), short_repl(p.get("replacement")),
                         esc("; ".join(map(str, p.get("rejection_reasons") or [])) or "–"), "<br>".join(vcell) or "–"])
        out.append(table(["proposal", "instance", "source", "status", "turn", "replacement", "rejection reasons", "verification"], rows))
    return "".join(out) if any_ else out[0] + "<p class='muted'>no edit stage yet</p>"


def sec_pods(S: dict[str, Any]) -> str:
    v: RunView = S["v"]
    pev = v.jsonl(v.dir / "logs" / "pod-lifecycle.jsonl")
    rows = [[esc(fmt_t(parse_t(e.get("t")))), pill(e.get("event") or "?", "pending"), f"<code>{esc(e.get('pod') or e.get('pod_id'))}</code>",
             f"<small>{esc(json.dumps({k: val for k, val in e.items() if k not in ('t', 'event', 'pod', 'pod_id')}, ensure_ascii=False, default=str))}</small>"]
            for e in pev]
    out = "<h2>Pod &amp; serving</h2><h3>pod lifecycle (this run)</h3>" + table(["time", "event", "pod", "details"], rows)
    sev = v.jsonl(v.dir / "logs" / "serving-lifecycle.jsonl")
    rows = [[esc(fmt_t(parse_t(e.get("t")))), esc(e.get("role")), pill(e.get("event") or "?", "pending"), f"<code>{esc(e.get('checkpoint_id'))}</code>",
             f"<small>pid {esc(e.get('pid'))} {esc(e.get('host') or '')} {esc(e.get('log') or '')}</small>"] for e in sev]
    out += ("<h3>serving lifecycle</h3><p class='muted'>appended when the coordinator stops serving (e.g. before training), so the "
            "live server may not appear yet.</p>" + table(["time", "role", "event", "checkpoint", "details"], rows))
    pods = pod_ledger(v.ledger)
    rows = [[f"<code>{esc(p['pod'])}</code>", esc(p["name"]), esc(fmt_t(p["created"])), esc(fmt_t(p["ended"])) if p["ended"] else pill("running"),
             esc(fmt_dur(p["up"])), f"${esc(p['cost'])}/h", f"${p['spend']:.2f}" if p["spend"] is not None else "n/a"] for p in reversed(pods)]
    total = sum(p["spend"] or 0 for p in pods)
    return out + (f"<h3>created-pod ledger <small class='muted'>({esc(v.ledger)}; all runs; est. all-time spend ${total:.2f}, "
                  "compute only)</small></h3>") + table(["pod", "name", "created", "terminated", "uptime", "rate", "est. spend"], rows)


def sec_log(S: dict[str, Any]) -> str:
    v: RunView = S["v"]
    p = v.log or v.dir / LOG_REL
    if v.log is None and not p.exists():
        return (f"<h2>Coordinator log</h2><p class='muted'>no <code>{LOG_REL}</code> in this run. <code>loop run</code> prints its "
                "progress to the console; start the dashboard with <code>--log FILE</code> to tail a file you saved it to.</p>")
    lines = tail(p) if (v.log is not None or inside(v.dir, p)) else None
    try:
        age = f" · last write {fmt_dur(time.time() - p.stat().st_mtime)} ago"
    except OSError:
        age = ""
    body = "<pre class='log'>" + esc("\n".join(lines)) + "</pre>" if lines is not None else "<p class='muted'>n/a (cannot read)</p>"
    return f"<h2>Coordinator log <small class='muted'>{esc(p)}{esc(age)}</small></h2>{body}"


def guard(fn: Callable[..., str], *a: Any) -> str:
    try:
        return fn(*a)
    except Exception:  # noqa: BLE001 - one broken section must not take the page down
        return f"<h2>{esc(fn.__name__)}</h2><pre class='err'>{esc(traceback.format_exc())}</pre>"


# --------------------------------------------------------------------------- #
# Across cycles (unassisted evaluation)
# --------------------------------------------------------------------------- #


def delta_cls(d_tok: float | None, d_rate: float | None) -> str:
    """Fewer tokens with equal-or-better success is good; fewer tokens with lower success is mixed."""
    if d_tok is None:
        return ""
    if d_tok < 0:
        return "good" if (d_rate is None or d_rate >= 0) else "warn"
    return "bad" if d_tok > 0 else ""


def tok_delta_cell(cur: float | None, base: float | None, d_rate: float | None) -> str:
    if cur is None or base is None:
        return '<span class="muted">n/a</span>'
    d = cur - base
    pct = f" ({d / base * 100:+.0f}%)" if base else ""
    arrow = "▼ " if d < 0 else ("▲ " if d > 0 else "")
    return f'<span class="{delta_cls(d, d_rate)}">{arrow}{esc(fmt_signed(d))}{esc(pct)}</span>'


def nice_step(span: float, n: int = 4) -> float:
    raw = span / n if span > 0 else 1
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def svg_chart(title: str, xs: list[int], series: list[tuple[str, str, dict[int, tuple[float | None, bool]]]], pct: bool = False,
              ymax: float | None = None, small: bool = False, note: str = "", xlab: dict[int, str] | None = None) -> str:
    """Line chart over cycles. series: [(name, css-var, {cycle: (value, partial)})]. Hollow points = partial."""
    W, H = (300, 150) if small else (440, 230)
    L, R, T, B = (38, 10, 12, 26) if small else (50, 16, 18, 36)
    vals = [val for _, _, pts in series for val, _ in pts.values() if val is not None]
    head = f'<div class="chart{" sm" if small else ""}"><div class="ct">{title}</div>'
    if not vals:
        return head + f'<div class="muted empty">no finished cycles yet</div>{note}</div>'
    if pct:
        top, step = 1.0, 0.25
    else:
        top = ymax if ymax is not None else max(vals)
        step = nice_step(top or 1)
        top = max(step, math.ceil((top or 1) / step - 1e-9) * step)
    x0, x1 = min(xs), max(xs)
    pad = 22  # keep points and their labels clear of the y-axis labels

    def px(c: int) -> float:
        return L + (W - L - R) / 2 if x1 == x0 else L + pad + (c - x0) * (W - L - R - 2 * pad) / (x1 - x0)

    def py(val: float) -> float:
        return T + (H - T - B) * (1 - val / top)

    have = {c for _, _, pts in series for c, (val, _) in pts.items() if val is not None}
    p = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(re.sub("<[^>]+>", "", title))}">']
    k = 0
    while k * step <= top + 1e-9:
        y = k * step
        p.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{py(y):.1f}" y2="{py(y):.1f}"/>'
                 f'<text class="ax" x="{L - 5}" y="{py(y) + 3.5:.1f}" text-anchor="end">{esc(fmt_c(y, pct))}</text>')
        k += 1
    for c in xs:
        p.append(f'<text class="ax{"" if c in have else " dim"}" x="{px(c):.1f}" y="{H - B + 14}" text-anchor="middle">{esc((xlab or {}).get(c, c))}</text>')
    if not small:
        p.append(f'<text class="ax" x="{(L + W - R) / 2:.1f}" y="{H - 3}" text-anchor="middle">cycle (checkpoint evaluated)</text>')
    # with two series, label the higher point above and the lower one below so labels never collide
    rank = {}
    for c in xs:
        vs = sorted((pts[c][0], si) for si, (_, _, pts) in enumerate(series) if c in pts and pts[c][0] is not None)
        for k, (_, si) in enumerate(vs):
            rank[(c, si)] = "below" if len(vs) > 1 and k == 0 else "above"
    for si, (name, var, pts) in enumerate(series):
        seq = [(c, pts[c][0], pts[c][1]) for c in xs if c in pts and pts[c][0] is not None]
        for (ca, va, pa), (cb, vb, pb) in zip(seq, seq[1:]):
            p.append(f'<line class="ln{" prov" if pa or pb else ""}" style="stroke:var({var})" x1="{px(ca):.1f}" y1="{py(va):.1f}" x2="{px(cb):.1f}" y2="{py(vb):.1f}"/>')
        for c, val, part in seq:
            ly = py(val) + (16 if rank.get((c, si)) == "below" else -9)
            lab = fmt_c(val, pct) + ("*" if part else "")
            tip = f"{name} · cycle {c}: {fmt_c(val, pct) if pct else fmt_num(val)}" + (" (cycle still running: partial)" if part else "")
            p.append(f'<g><title>{esc(tip)}</title><circle cx="{px(c):.1f}" cy="{py(val):.1f}" r="10" fill="transparent"/>'
                     f'<circle class="pt" cx="{px(c):.1f}" cy="{py(val):.1f}" r="4.5" style="stroke:var({var});fill:{"var(--card)" if part else f"var({var})"}"/>'
                     f'<text class="pl" x="{px(c):.1f}" y="{ly:.1f}" text-anchor="middle">{esc(lab)}</text></g>')
    p.append("</svg>")
    legend = ""
    if len(series) > 1:
        legend = '<div class="lg">' + "".join(f'<span><i style="background:var({var})"></i>{esc(n)}</span>' for n, var, _ in series) + "</div>"
    return head + legend + "".join(p) + note + "</div>"


def matched(base_rows: list[EpisodeRow], rows: list[EpisodeRow]) -> tuple[list[EpisodeRow], list[EpisodeRow], int]:
    """Both sides restricted to the (panel, instance, attempt, seed) keys present in both."""
    keys = {_pair_key(r) for r in base_rows} & {_pair_key(r) for r in rows}
    return [r for r in base_rows if _pair_key(r) in keys], [r for r in rows if _pair_key(r) in keys], len(keys)


def sec_across(S: dict[str, Any]) -> str:
    v: RunView = S["v"]
    ev = stage_cycles(S, "eval")
    xs = [cy["n"] for cy in S["cycles"]] or [0]
    out = ["<h2>Across cycles (unassisted evaluation)</h2>",
           "<p class='muted'>Each cycle evaluates its incoming checkpoint on the same dev-panel instances with the same seeds, without editor help. "
           "Success counts ungraded (e.g. infra-failed) episodes as not successful, as in <code>loop report</code>. "
           "Tokens = input + output from provider usage (episodes without usage are left out of token means, never counted as 0). "
           "Only finished episodes enter the numbers.</p>"]
    if not ev:
        return "".join(out) + "<p class='muted'>no evaluation stage yet</p>"
    for r in ev:
        r["stats"] = stats(r["rows"])
    base = next((r for r in ev if r["n"] == 0), None)
    done_cycles = [r for r in ev if not r["running"] and r["stats"]["n_episodes"]]

    # ---- charts: finished cycles only. A running cycle's finished episodes are whichever items
    # ran first (not a representative subset); the paired table compares them like-for-like.
    def pts(key: str) -> dict[int, tuple[float | None, bool]]:
        return {r["n"]: (r["stats"][key], False) for r in done_cycles}

    xlab = {r["n"]: f"{r['n']} (running {r['cnt'].get('done', 0)}/{r['total']})" for r in ev if r["running"]}
    fin = {cy["n"]: f"{cy['n']} final" for cy in S["cycles"] if cy["final"]}
    note = ""
    if len(done_cycles) < 2:
        note = "<div class='muted cn'>Only one finished cycle so far; later cycles appear here as their evaluation finishes.</div>"
    if xlab:
        note += ("<div class='muted cn'>Charts show finished cycles only: a running cycle's finished episodes are whichever items ran "
                 "first, not a representative sample. Its finished items are compared like-for-like in the paired table.</div>")
    labels = {**fin, **xlab}
    charts = [
        svg_chart("Success rate <small>(unassisted eval)</small>", xs, [("success rate", "--s1", pts("success_rate"))], pct=True, xlab=labels),
        svg_chart("Mean tokens per episode", xs, [("all episodes", "--s1", pts("total_tokens_mean")),
                                                  ("successful only", "--s2", pts("total_tokens_mean_successful"))], xlab=labels),
        svg_chart("Tokens per success <small>(total tokens ÷ successes)</small>", xs, [("tokens per success", "--s1", pts("tps"))], xlab=labels),
    ]
    out.append(f"<div class='charts'>{''.join(charts)}</div>{note}")
    fams = sorted({family_of(r) for rec in ev for r in rec["rows"]})
    if fams:
        fam_pts = {f: {r["n"]: (agg(x.episode.usage.total for x in r["rows"] if family_of(x) == f)["mean"], False) for r in done_cycles} for f in fams}
        top = max([val for d in fam_pts.values() for val, _ in d.values() if val is not None] or [0]) or None
        out.append("<h3>Mean tokens per episode by family <small class='muted'>(all episodes; shared y-scale)</small></h3><div class='charts sm'>"
                   + "".join(svg_chart(esc(f), xs, [(f, "--s1", fam_pts[f])], ymax=top, small=True) for f in fams) + "</div>")

    # ---- per-cycle table
    rows = []
    for r in ev:
        a = r["stats"]
        cyc = f"<b>{r['n']}</b>{' <small>final</small>' if r['final'] else ''}<br><code>{esc(r['ckpt'])}</code>"
        if r["running"]:
            c = r["cnt"]
            cyc += (f"<br>{pill('running')} <small>{esc(c.get('done', 0))}/{esc(r['total'])} done · {esc(c.get('running', 0))} running · "
                    f"{esc(c.get('pending', 0))} pending</small>")
        extra = []
        if a["n_with_tokens"] < a["n_episodes"]:
            extra.append(f"usage {a['n_with_tokens']}/{a['n_episodes']}")
        if r["unreadable"]:
            extra.append(f"{r['unreadable']} unreadable")
        if r["cnt"].get("failed"):
            extra.append(f"{r['cnt']['failed']} failed items have no result")
        if r["cnt"].get("infra_failed"):
            extra.append(f"{r['cnt']['infra_failed']} infra-failed included")
        nep = f"{a['n_episodes']}" + (f"<br><small>{esc(' · '.join(extra))}</small>" if extra else "")
        rate = rate_text(a) + (f"<br><small>{a['n_ungraded']} ungraded</small>" if a["n_ungraded"] else "")
        if r is base or base is None:
            drate_c = dtok_c = '<span class="muted">baseline</span>' if r is base else '<span class="muted">n/a (no cycle 0)</span>'
        else:
            # deltas compare the same (instance, attempt, seed) items in both cycles, so a running cycle is like-for-like too
            bm_rows, cm_rows, nk = matched(base["rows"], r["rows"])
            bm, cm = stats(bm_rows), stats(cm_rows)
            mnote = f"<br><small>on {nk} matched items</small>" if (nk != len(base["rows"]) or nk != len(r["rows"])) else ""
            d_rate = cm["success_rate"] - bm["success_rate"] if cm["success_rate"] is not None and bm["success_rate"] is not None else None
            drate_c = ('<span class="muted">n/a</span>' if d_rate is None else
                       f'<span class="{"good" if d_rate > 0 else ("bad" if d_rate < 0 else "")}">'
                       f'{"▲ " if d_rate > 0 else ("▼ " if d_rate < 0 else "")}{d_rate * 100:+.0f} pp</span>') + mnote
            dtok_c = (f"all {tok_delta_cell(cm['total_tokens_mean'], bm['total_tokens_mean'], d_rate)}"
                      f"<br>succ {tok_delta_cell(cm['total_tokens_mean_successful'], bm['total_tokens_mean_successful'], d_rate)}"
                      f"<br>per success {tok_delta_cell(cm['tps'], bm['tps'], d_rate)}") + mnote
        stops = ", ".join(f"{esc(k)} {n}" for k, n in Counter(a["stop_categories"]).most_common())
        mal = f"{fmt_num(a['n_malformed_turns_sum'])}" + (f" <small>in {a['n_malformed_eps']} eps</small>" if a["n_malformed_turns_sum"] else "") if a["n_episodes"] else "n/a"
        rows.append([cyc, nep, rate, drate_c, esc(fmt_num(a["mean_partial_reward"], 2)), esc(fmt_num(a["total_tokens_mean"])),
                     esc(fmt_num(a["total_tokens_mean_successful"])) + (f" <small>n={a['n_use_ok']}</small>" if a["n_use_ok"] else ""),
                     esc(fmt_num(a["tps"])), dtok_c, esc(fmt_num(a["out"])), esc(fmt_num(a["turns"], 1)), esc(fmt_num(a["tools"], 1)),
                     esc(fmt_num(a["sec"], 0)), mal, f"<small>{stops or 'n/a'}</small>"])
    out.append("<h3>Per cycle</h3><p class='muted'>Absolute columns cover every finished episode of the cycle (a running cycle's are partial "
               "and unrepresentative). Δ columns compare against cycle 0 on the same (instance, attempt, seed) items only.</p>"
               + table(["cycle · checkpoint", "episodes", "success", "Δ success vs c0 (matched)", "mean partial", "mean tokens (all)",
                        "mean tokens (successful)", "tokens / success", "Δ tokens vs c0 (matched)", "mean output tok", "mean turns",
                        "mean tool calls", "mean sec", "malformed turns", "stop categories"], rows, "across"))

    # ---- paired comparison (metrics.paired_comparison, as in `loop report` / `loop compare`)
    if base:
        base_rel = {_pair_key(x): base["rel"].get(x.episode.episode_id) for x in base["rows"]}
        rows, details = [], []
        for r in ev:
            if r is base:
                continue
            try:
                c = paired_comparison(base["rows"], r["rows"], "c0", f"c{r['n']}")
            except ValueError as e:  # duplicate keys: not a pairable evaluation
                rows.append([f"<b>{r['n']}</b> vs 0", f"<span class='bad'>{esc(e)}</span>"] + ["n/a"] * 10)
                continue
            cur_rel = {_pair_key(x): r["rel"].get(x.episode.episode_id) for x in r["rows"]}
            t = c["transitions"]
            pairs = c["pairs"]
            ta, tb = "total_tokens_c0", f"total_tokens_c{r['n']}"
            both = [(p[ta], p[tb]) for p in pairs if p["token_delta_both_succeed"] is not None]
            ds = [b - a for a, b in both]
            sb, sc = sum(a for a, _ in both), sum(b for _, b in both)
            drows = []
            for p in pairs:
                k = (p["panel"], p["instance_id"], p["attempt_index"], p["seed"])
                tr = p["transition"]
                d = p[tb] - p[ta] if p[ta] is not None and p[tb] is not None else None  # shown for every pair; coloured only where both succeed
                drows.append([esc(p["instance_id"]), esc(p["attempt_index"]),
                              pill(tr.replace("_", " "), {"both_succeed": "done", "gained": "done", "lost": "failed"}.get(tr, "pending")),
                              v.ep_link(base_rel.get(k), fmt_num(p[ta])), v.ep_link(cur_rel.get(k), fmt_num(p[tb])),
                              f'<span class="{delta_cls(d, 0) if tr == "both_succeed" else ""}">{esc(fmt_signed(d))}</span>' if d is not None else "n/a"])
            notes = []
            if c["n_seed_mismatch"]:
                notes.append(f"<small class='bad'>{c['n_seed_mismatch']} item(s) with different seeds (not paired)!</small>")
            if not c["token_units"]["comparable"]:
                notes.append(f"<small class='bad'>{esc(c['token_units']['note'])}</small>")
            n_both = t["both_succeed"]
            rows.append([f"<b>{r['n']}</b> vs 0" + (f" {pill('running')}" if r["running"] else ""),
                         f"{c['n_pairs']} <small>of {len(base['rows'])} baseline</small>" + "".join(f"<br>{n}" for n in notes),
                         f"<span>{n_both}</span>", f"<span class='{'good' if t['gained'] else ''}'>{t['gained']}</span>",
                         f"<span class='{'bad' if t['lost'] else ''}'>{t['lost']}</span>", f"{t['both_fail']}", f"{t['undetermined']}",
                         f"{len(both)}" + (f" <small>of {n_both}</small>" if len(both) != n_both else ""),
                         f'<span class="{delta_cls(statistics.mean(ds), 0)}">{esc(fmt_signed(statistics.mean(ds)))}</span>' if ds else "n/a",
                         f'<span class="{delta_cls(statistics.median(ds), 0)}">{esc(fmt_signed(statistics.median(ds)))}</span>' if ds else "n/a",
                         (f"{fmt_num(sb)} → {fmt_num(sc)} " + (f'<span class="{delta_cls(sc - sb, 0)}">({(sc - sb) / sb * 100:+.0f}%)</span>' if sb else "")) if ds else "n/a",
                         f"<span class='good'>{sum(1 for x in ds if x < 0)}</span> / <span class='bad'>{sum(1 for x in ds if x > 0)}</span> / "
                         f"{sum(1 for x in ds if x == 0)}" if ds else "n/a"])
            details.append(f"<details><summary>cycle {r['n']} vs 0: {c['n_pairs']} matched pairs</summary>"
                           + table(["instance", "attempt", "transition", "tokens c0", f"tokens c{r['n']}", "Δ tokens"], drows) + "</details>")
        out.append("<h3>Paired vs cycle 0</h3><p class='muted'>Only matched pairs are compared: the same (panel, instance, attempt, seed) "
                   "finished in both cycles. Token changes use only pairs where both episodes succeeded, so a lost success cannot "
                   "look like a saving.</p>"
                   + table(["cycle", "matched pairs", "both succeed", "gained", "lost", "both fail", "undetermined", "both-succeed pairs with usage",
                            "mean Δ tokens", "median Δ tokens", "Σ tokens c0 → cN", "cheaper / costlier / same"], rows) + "".join(details))

    # ---- per family / difficulty
    def grid_table(key: Callable[[EpisodeRow], str], label: str) -> str:
        groups = sorted({key(x) for r in ev for x in r["rows"]}, key=lambda val: (DIFF_ORDER.get(val, 9), val))
        trows = []
        for grp in groups + ["all"]:
            cells = [f"<b>{esc(grp)}</b>"]
            for r in ev:
                a = stats([x for x in r["rows"] if grp == "all" or key(x) == grp], full=False)
                if not a["n_episodes"]:
                    cells.append('<span class="muted">–</span>')
                    continue
                cells.append(f"{rate_text(a)}<br><small>tok {esc(fmt_num(a['total_tokens_mean']))} · succ {esc(fmt_num(a['total_tokens_mean_successful']))}</small>")
            trows.append(cells)
        return table([label] + [f"cycle {r['n']}" + (" (running)" if r["running"] else "") for r in ev], trows)

    out.append("<h3>By family and difficulty</h3><p class='muted'>Cells: successes/episodes (rate) · mean tokens all · mean tokens successful-only.</p>"
               "<div class='two'>" + grid_table(family_of, "family") + grid_table(lambda x: x.difficulty or "?", "difficulty") + "</div>")

    n_eval = S["planned"].get("eval") or max((r["stats"]["n_episodes"] for r in ev), default=0) or "a few"
    out.append(f"<p class='caption'>Small samples: each cycle has only {esc(n_eval)} evaluation episodes, so differences between cycles "
               "are noisy and a few episodes can swing every number here. Nothing on this page is a significance test; "
               "<code>loop report</code> and <code>loop compare</code> give instance-level intervals.</p>")

    # ---- collection (secondary)
    rows = []
    for r in stage_cycles(S, "collect"):
        a = stats(r["rows"], full=False)
        cyc = f"<b>{r['n']}</b>" + (f" {pill('running')} <small>{esc(r['cnt'].get('done', 0))}/{esc(r['total'])}</small>" if r["running"] else "")
        rows.append([cyc, f"{a['n_episodes']}" + (f" <small>usage {a['n_with_tokens']}/{a['n_episodes']}</small>" if a["n_with_tokens"] < a["n_episodes"] else ""),
                     rate_text(a), esc(fmt_num(a["mean_partial_reward"], 2)), esc(fmt_num(a["total_tokens_mean"])),
                     esc(fmt_num(a["total_tokens_mean_successful"])), esc(fmt_num(a["tps"])), esc(fmt_num(a["turns"], 1)),
                     f"<small>{', '.join(f'{esc(k)} {n}' for k, n in Counter(a['stop_categories']).most_common()) or 'n/a'}</small>"])
    out.append("<details class='sec'><summary>Collection episodes per cycle (training panel, editor-source episodes; a different panel, "
               "not the evaluation)</summary>"
               + table(["cycle", "episodes", "success", "mean partial", "mean tokens (all)", "mean tokens (successful)", "tokens / success",
                        "mean turns", "stop categories"], rows) + "</details>")
    return "".join(out)


# --------------------------------------------------------------------------- #
# Transcript page
# --------------------------------------------------------------------------- #


def find_proposal(v: RunView, rel: str) -> dict[str, Any] | None:
    """The edit proposal whose source is this episode directory, if any."""
    try:
        mans = sorted((v.dir / "cycles").glob("cycle-*/edit/manifest.json"))
    except OSError:
        return None
    for man_p in mans:
        items = _dict(v.json(man_p)).get("items") or {}
        for it in (items.values() if isinstance(items, dict) else items):
            if not isinstance(it, dict):
                continue
            src = g(it, "meta", "source_dir")
            d = v.item_dir(src)
            iid = it.get("item_id")
            if d is not None and v.rel(d) == rel and isinstance(iid, str) and SAFE_NAME.match(iid):
                return _dict(v.json(man_p.parent / "items" / iid / "proposal.json")) or None
    return None


def live_messages(v: RunView, events_p: Path) -> list[dict[str, Any]] | None:
    """The conversation so far, rebuilt from events.jsonl (last request's messages + later turns)."""
    if not inside(v.dir, events_p):
        return None
    msgs: list[Any] = []
    tail_: list[dict[str, Any]] = []
    try:
        fh = open(events_p, encoding="utf-8", errors="replace")
    except OSError:
        return None
    with fh:
        for line in fh:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if not isinstance(e, dict):
                continue
            k, d = e.get("kind"), _dict(e.get("data"))
            if k == "request":
                msgs, tail_ = d.get("messages") or [], []
            elif k == "response":
                m = d.get("history_message") or g(d, "raw", "choices", 0, "message")
                if isinstance(m, dict):
                    tail_.append(m)
            elif k == "tool_result":
                tail_.append({"role": "tool", "tool_call_id": d.get("call_id"), "content": d.get("raw_output")})
    return [m for m in msgs if isinstance(m, dict)] + tail_


def render_messages(msgs: list[Any], prop: dict[str, Any]) -> str:
    out, turn = [], -1
    hit = prop.get("tool_call_id")
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "?"))
        head = esc(role)
        if role == "assistant":
            turn += 1
            head = f"assistant · turn {turn}"
        elif role == "tool":
            head = f"tool output <small>{esc(m.get('tool_call_id'))}</small>"
        body = []
        if m.get("reasoning_content"):
            body.append(f"<details><summary>reasoning ({len(str(m['reasoning_content'])):,} chars)</summary><pre>{esc(m['reasoning_content'])}</pre></details>")
        content = m.get("content")
        if isinstance(content, list):
            content = "\n".join(str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content)
        if role == "system":
            body.append(f"<details><summary>system prompt ({len(str(content or '')):,} chars)</summary><pre>{esc(content)}</pre></details>")
        elif content:
            body.append(trunc_pre(content, 1500 if role == "tool" else 4000))
        for tc in m.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = _dict(tc.get("function"))
            args = fn.get("arguments")
            try:
                a = json.loads(args) if isinstance(args, str) else args
            except ValueError:
                a = args
            if isinstance(a, dict) and "command" in a and len(a) <= 2:
                shown = "$ " + str(a["command"])
            else:
                shown = a if isinstance(a, str) else json.dumps(a, indent=1, ensure_ascii=False)
            edited = bool(tc.get("id")) and tc.get("id") == hit
            body.append(f'<div class="call{" edited" if edited else ""}"><b>→ {esc(fn.get("name"))}</b> <small>{esc(tc.get("id"))}</small>{trunc_pre(shown, 3000)}'
                        + (f"<div class='repl'><b>editor's replacement ({esc(prop.get('status'))}):</b>{short_repl(prop.get('replacement'), 400)}</div>" if edited else "")
                        + "</div>")
        out.append(f'<div class="msg r-{esc(role)}"><div class="role">{head}</div>{"".join(body)}</div>')
    return "".join(out)


def episode_page(v: RunView, rel: str) -> tuple[int, str]:
    d = resolve_episode(v.dir, rel)
    if d is None:
        return 404, page("not found", f"<p>no such episode directory in this run · <a href='{v.base}/'>dashboard</a></p>", refresh=False)
    rel = v.rel(d) or rel
    summ = _dict(v.json(d / "summary.json")) or _dict(v.json(d / "episode" / "summary.json"))
    msgs = g(v.json(d / "episode" / "messages.json"), "messages")
    note = ""
    if not isinstance(msgs, list):
        msgs = live_messages(v, d / "episode" / "events.jsonl") or []
        note = "<p class='muted'>Episode not finished (or messages.json missing): transcript reconstructed live from events.jsonl.</p>"
    prop = find_proposal(v, rel) or {}
    parts = [f"<p><a href='{v.base}/'>← dashboard</a></p><h1>{esc(rel)}</h1>"]
    if summ:
        kv = [("instance", summ.get("instance_id")), ("role", summ.get("role")), ("success", summ.get("success")), ("partial", summ.get("partial_reward")),
              ("stop", f"{summ.get('stop_reason')} ({summ.get('stop_category')})"), ("requests", summ.get("n_requests")), ("tool calls", summ.get("n_tool_calls")),
              ("tokens in/out", f"{g(summ, 'usage', 'input_tokens')} / {g(summ, 'usage', 'output_tokens')}"), ("seconds", fmt_num(g(summ, "timing", "total_sec"), 1)),
              ("checkpoint", summ.get("checkpoint_id")), ("infra error", summ.get("infra_error"))]
        parts.append('<div class="kvs">' + "".join(f'<div class="kv"><span>{esc(k)}</span><b>{esc(val)}</b></div>' for k, val in kv) + "</div>")
    else:
        turns = live_turns(d / "episode" / "events.jsonl") if inside(v.dir, d / "episode" / "events.jsonl") else None
        parts.append(f"{pill('running')} <small>{esc(turns)} model responses so far</small>")
    if d.parent.parent.name in ("verify", "audit") or BRANCH_RE.match(d.name):
        vdir = d.parent if BRANCH_RE.match(d.name) else d
        ver = _dict(v.json(vdir / "verification.json"))
        if ver:
            try:
                sibs = sorted(p for p in vdir.iterdir() if p.is_dir() and BRANCH_RE.match(p.name))
            except OSError:
                sibs = []
            parts.append(f"<p>verification <code>{esc(ver.get('verification_id'))}</code>: {'accepted' if ver.get('accepted') else 'rejected'} — "
                         f"{esc(', '.join(map(str, ver.get('reasons') or [])))}; saving {fmt_num(ver.get('mean_saving'))} · siblings: "
                         + " ".join(v.ep_link(str(p), p.name) for p in sibs) + "</p>")
    if prop:
        parts.append(f"<div class='msg r-prop'><div class='role'>edit proposal {esc(prop.get('proposal_id'))} · {esc(prop.get('status'))} · "
                     f"turn {esc(prop.get('turn_index'))}</div><p>{esc(prop.get('justification') or '')}</p>"
                     f"<p>rejection reasons: {esc('; '.join(map(str, prop.get('rejection_reasons') or [])) or '–')}</p>"
                     f"replacement: {short_repl(prop.get('replacement'), 400)}</div>")
    parts.append(note + render_messages(msgs, prop))
    try:
        verifiers = sorted(d.glob("harbor/*/verifier"))
    except OSError:
        verifiers = []
    for vf in verifiers:
        txt = v.text(vf / "test-stdout.txt")
        rw = v.json(vf / "reward.json")
        parts.append(f"<h2>Grader ({esc(vf.parent.name)}) reward {esc(json.dumps(rw) if rw is not None else 'n/a')}</h2>" + trunc_pre(txt if txt is not None else "n/a", 6000))
    return 200, page(f"episode {d.name}", "".join(parts), refresh=not summ)


# --------------------------------------------------------------------------- #
# Page shell, run page, index
# --------------------------------------------------------------------------- #

CSS = """
:root{--bg:#f7f7f5;--fg:#1d1d1f;--muted:#6b6b70;--card:#fff;--line:#e2e2e0;--ok:#2f9e5b;--bad:#d64545;--warn:#d98a1c;--run:#2f72d6;--idle:#c8c8cc;--code:#f0f0ee;--s1:#2a78d6;--s2:#eb6834}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#151517;--fg:#e8e8ea;--muted:#9a9aa1;--card:#1e1e21;--line:#2f2f34;--ok:#45b874;--bad:#e46a6a;--warn:#e0a040;--run:#5b93e6;--idle:#45454b;--code:#26262a;--s1:#3987e5;--s2:#d95926}}
:root[data-theme="dark"]{--bg:#151517;--fg:#e8e8ea;--muted:#9a9aa1;--card:#1e1e21;--line:#2f2f34;--ok:#45b874;--bad:#e46a6a;--warn:#e0a040;--run:#5b93e6;--idle:#45454b;--code:#26262a;--s1:#3987e5;--s2:#d95926}
*{box-sizing:border-box}body{margin:0;padding:12px 16px 40px;background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,system-ui,Segoe UI,sans-serif;max-width:1500px;margin:auto}
h1{font-size:20px;margin:6px 0}h2{font-size:16px;margin:26px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}h3{font-size:14px;margin:14px 0 6px}
a{color:var(--run)}code,pre{font:12px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace}code{background:var(--code);padding:0 3px;border-radius:3px;word-break:break-all}
pre{background:var(--code);padding:8px;border-radius:6px;white-space:pre-wrap;word-break:break-word;margin:4px 0;max-height:70vh;overflow:auto}
.muted{color:var(--muted)}small{color:var(--muted)}.kvs{display:flex;flex-wrap:wrap;gap:8px}.kv{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 10px;min-width:120px}
.kv span{display:block;font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}.kv b{font-weight:600}
.tw{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:8px}table{border-collapse:collapse;width:100%}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{font-size:11px;color:var(--muted);text-transform:uppercase;white-space:nowrap}
tr:last-child td{border-bottom:0}.timeline td{min-width:130px}
.pill{display:inline-block;padding:0 7px;border-radius:9px;font-size:11px;font-weight:600;background:var(--idle);color:var(--fg)}
.pill.done{background:var(--ok);color:#fff}.pill.running{background:var(--run);color:#fff}.pill.failed,.pill.infra_failed{background:var(--bad);color:#fff}
.bar{display:flex;height:7px;border-radius:4px;overflow:hidden;background:var(--line);margin:3px 0;min-width:80px}.bar i{display:block;height:100%}
.s-done{background:var(--ok)}.s-running{background:var(--run)}.s-pending{background:var(--idle)}.s-failed{background:var(--bad)}.s-infra_failed{background:var(--warn)}
.msg{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--idle);border-radius:6px;padding:6px 10px;margin:8px 0}
.r-assistant{border-left-color:var(--run)}.r-tool{border-left-color:var(--ok)}.r-user{border-left-color:var(--warn)}.r-prop{border-left-color:var(--bad)}
.role{font-weight:600;font-size:12px;margin-bottom:2px}.call{margin-top:4px}.call.edited{outline:2px solid var(--bad);border-radius:6px;padding:4px}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr));gap:10px}
.chart{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 10px}.chart.sm{padding:6px 8px}.ct{font-weight:600;font-size:13px}
.chart svg{display:block;width:100%;max-width:620px;height:auto;overflow:visible}
.charts.sm{grid-template-columns:repeat(auto-fill,minmax(min(100%,260px),1fr))}.chart.sm svg{max-width:420px}.chart .grid{stroke:var(--line);stroke-width:1}
.chart .ax{fill:var(--muted);font-size:11px}.chart .ax.dim{opacity:.45}.chart .ln{stroke-width:2;stroke-linecap:round}.chart .ln.prov{stroke-dasharray:4 4}
.chart .pt{stroke-width:2}.chart .pl{fill:var(--fg);font-size:11px;font-weight:600;paint-order:stroke;stroke:var(--card);stroke-width:3px}
.lg{display:flex;gap:12px;font-size:12px;color:var(--muted);margin:2px 0}.lg i{display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:5px;vertical-align:middle}
.empty{padding:40px 0;text-align:center}.cn{margin:6px 0}.good{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}.warn{color:var(--warn);font-weight:600}
.caption{border-left:3px solid var(--warn);padding:4px 10px;background:var(--card);border-radius:4px}.across td{min-width:70px}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:10px}details.sec{margin:12px 0}details.sec>summary{cursor:pointer;font-weight:600}
.repl{margin-top:4px}.log{max-height:none}.err{color:var(--bad)}.top{display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap}
"""

JS = """<script>
const cb=document.getElementById('pause');
try{cb.checked=localStorage.getItem('dash-pause')==='1'}catch(e){}
cb.onchange=()=>{try{localStorage.setItem('dash-pause',cb.checked?'1':'0')}catch(e){}};
(function tick(){setTimeout(()=>{if(!cb.checked&&!document.querySelector('details[open]'))location.reload();else tick()},%d)})();
</script>""" % (REFRESH_SEC * 1000)


def page(title: str, body: str, refresh: bool = True) -> str:
    ctl = (f"<label class='muted'><input type=checkbox id=pause> pause auto-refresh ({REFRESH_SEC} s; also paused while a section is expanded)</label>"
           if refresh else "")
    return (f"<!doctype html><html lang='en'><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{esc(title)}</title><style>{CSS}</style></head><body><div class='top'><small>rendered {esc(time.strftime('%H:%M:%S'))} · "
            f"read-only</small>{ctl}</div>{body}{JS if refresh else ''}</body></html>")


def run_page(v: RunView) -> tuple[int, str]:
    S = load_state(v)
    eps = all_episodes(S)
    body = (guard(sec_header, S) + guard(sec_across, S) + guard(sec_timeline, S) + guard(sec_activity, S, eps)
            + guard(sec_outcomes, S) + guard(sec_edits, S) + guard(sec_pods, S) + guard(sec_log, S))
    return 200, page(f"{S['run_id']} · dashboard", body)


def run_overview(v: RunView) -> dict[str, Any]:
    """One index row: kind, experiment, status, position, coordinator, start and last activity."""
    S = load_state(v)
    alive, lock = coordinator_alive(v.dir)
    cy, stage = current_position(S)
    prov = _dict(v.json(v.dir / "provenance.json"))
    start = parse_t(prov.get("recorded_at"))
    if start is None:
        try:
            start = (v.dir / "run.json").stat().st_mtime
        except OSError:
            start = None
    last = None
    try:
        for p in (v.dir / "cycles").glob("cycle-*/*.json"):
            last = max(last or 0, p.stat().st_mtime)
        for p in (v.dir / "cycles").glob("cycle-*/*/manifest.json"):
            last = max(last or 0, p.stat().st_mtime)
    except OSError:
        pass
    return {"name": v.dir.name, "run_id": S["run_id"], "kind": S["kind"], "experiment": S["exp"].get("name"), "status": run_status(S, alive),
            "cycle": cy["n"] if cy else None, "stage": stage, "alive": alive, "lock": lock, "start": start, "last": last}


# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #


class Dashboard:
    """Routes: `/` (index of runs, or a redirect to the one run), `/run/<run-id>/` (run page),
    `/run/<run-id>/ep?p=<episode dir relative to the run>` (transcript)."""

    def __init__(self, runs_dir: Path | None = None, run_dir: Path | None = None, log: Path | None = None, ledger: Path | None = None):
        if run_dir is None and runs_dir is None:
            raise ValueError("need a runs directory or a run directory")
        self.single = Path(run_dir).resolve() if run_dir is not None else None
        self.runs_dir = self.single.parent if self.single is not None else Path(runs_dir).resolve()  # type: ignore[arg-type]
        self.log = Path(log).expanduser().resolve() if log is not None else None
        self.ledger = Path(ledger) if ledger is not None else default_ledger()

    def runs(self) -> list[Path]:
        if self.single is not None:
            return [self.single]
        try:
            return [p.resolve() for p in self.runs_dir.iterdir()
                    if SAFE_NAME.match(p.name) and p.is_dir() and (p / "run.json").is_file() and p.resolve().parent == self.runs_dir]
        except OSError:
            return []

    def resolve_run(self, name: str) -> Path | None:
        """The run directory a URL names, or None unless it is one of the served runs."""
        if self.single is not None:
            return self.single if name == self.single.name else None
        if not SAFE_NAME.match(name):
            return None
        p = self.runs_dir / name
        try:
            rp = p.resolve()
        except (OSError, RuntimeError):
            return None
        if rp.parent != self.runs_dir or not rp.is_dir() or not (rp / "run.json").is_file():
            return None
        return rp

    def view(self, run_dir: Path) -> RunView:
        return RunView(dir=run_dir, base=f"/run/{quote(run_dir.name)}", log=self.log if self.single is not None else None,
                       ledger=self.ledger, index_link=self.single is None)

    def index_page(self) -> tuple[int, str]:
        rows = []
        overviews = []
        for rd in self.runs():
            try:
                overviews.append(run_overview(self.view(rd)))
            except Exception:  # noqa: BLE001 - one broken run must not hide the others
                overviews.append({"name": rd.name, "run_id": rd.name, "kind": "?", "experiment": None, "status": ("unreadable", "failed"), "cycle": None,
                                  "stage": "n/a", "alive": None, "lock": {}, "start": None, "last": None})
        overviews.sort(key=lambda o: -(o["start"] or 0))
        now = time.time()
        for o in overviews:
            if o["alive"] is True:
                coord = pill("alive", "running") + f" <small>pid {esc(o['lock'].get('pid'))}</small>"
            elif o["alive"] is False:
                coord = pill("not running", "pending") + f" <small>last pid {esc(o['lock'].get('pid'))}</small>"
            else:
                coord = '<span class="muted">unknown</span>' + (f" <small>{esc(o['lock'].get('host'))}</small>" if o["lock"].get("host") else "")
            label = esc(o["name"]) + (f"<br><small>run_id {esc(o['run_id'])}</small>" if o["run_id"] != o["name"] else "")
            rows.append([f'<a href="/run/{quote(o["name"])}/">{label}</a>', esc(o["kind"]), esc(o["experiment"]), pill(*o["status"]),
                         esc(o["cycle"]) if o["cycle"] is not None else "–", esc(o["stage"]), coord, esc(fmt_t(o["start"])),
                         f"{esc(fmt_dur(now - o['last']))} ago" if o["last"] else "n/a"])
        body = (f"<h1>Runs <small class='muted'>{esc(self.runs_dir)}</small></h1><p class='muted'>Every directory with a run.json, newest "
                "first. The coordinator column checks whether the PID recorded in the run's .lock is alive on this host (the lock "
                "itself is never taken).</p>"
                + table(["run", "kind", "experiment", "status", "cycle", "stage", "coordinator", "started", "last activity"], rows))
        return 200, page("runs · dashboard", body)

    def handle(self, url: str) -> tuple[int, str, dict[str, str]]:
        """(HTTP status, HTML, extra headers) for a request path; never raises."""
        u = urlsplit(url)
        parts = [unquote(p) for p in u.path.split("/") if p]
        try:
            if not parts:
                if self.single is not None:
                    return 302, "", {"Location": f"/run/{quote(self.single.name)}/"}
                code, doc = self.index_page()
                return code, doc, {}
            if parts[0] == "run" and len(parts) in (2, 3):
                rd = self.resolve_run(parts[1])
                if rd is not None:
                    v = self.view(rd)
                    if len(parts) == 2:
                        code, doc = run_page(v)
                        return code, doc, {}
                    if parts[2] == "ep":
                        code, doc = episode_page(v, (parse_qs(u.query).get("p") or [""])[0])
                        return code, doc, {}
            return 404, page("not found", "<p>not found · <a href='/'>dashboard</a></p>", refresh=False), {}
        except Exception:  # noqa: BLE001 - show the error instead of dropping the connection
            return 500, page("error", f"<pre class='err'>{esc(traceback.format_exc())}</pre>", refresh=False), {}


def make_handler(dash: Dashboard) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "loop-dashboard"

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            code, doc, headers = dash.handle(self.path)
            data = doc.encode("utf-8")
            self.send_response(HTTPStatus(code))
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            for k, val in headers.items():
                self.send_header(k, val)
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - quiet by default
            pass

    return Handler


def make_server(dash: Dashboard, host: str = "127.0.0.1", port: int = 8090) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), make_handler(dash))
    server.daemon_threads = True
    return server


def serve(dash: Dashboard, host: str = "127.0.0.1", port: int = 8090, open_browser: bool = False) -> int:
    server = make_server(dash, host, port)
    actual_port = server.server_address[1]
    url = f"http://{'localhost' if host in ('127.0.0.1', 'localhost') else host}:{actual_port}/"
    what = dash.single if dash.single is not None else f"runs in {dash.runs_dir}"
    print(f"Dashboard for {what} at {url}  (read-only; Ctrl-C to stop)", flush=True)
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"note: bound to {host}; other machines on the network can read run outputs and transcripts", flush=True)
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0
