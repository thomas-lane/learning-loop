"""`loop dashboard`: run index, run page and transcript pages rendered from a fixture run;
numbers match `loop report`, content is escaped, half-written files are tolerated, URLs cannot
reach outside the runs, and rendering never writes into a run."""

import asyncio
import html
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from learning_loop.cli import main
from learning_loop.core.config import REPO_ROOT
from learning_loop.core.storage import read_json
from learning_loop.orchestration import coordinator as co
from learning_loop.reporting import dashboard as dash
from learning_loop.reporting.metrics import eval_rows, group_summaries, summarize
from learning_loop.reporting.report import collect_run

EXP = REPO_ROOT / "experiments" / "fixture-two-cycles.yaml"
MACHINE = REPO_ROOT / "configs" / "machines" / "examples" / "fixture.yaml"


@pytest.fixture(scope="module")
def fixture_run(tmp_path_factory) -> Path:
    runs = tmp_path_factory.mktemp("runs")
    ctx = co.create_run(EXP, MACHINE, run_id="fx", runs_dir=runs)
    asyncio.run(co.run_all(ctx, log=lambda m: None))
    return ctx.run_dir


@pytest.fixture
def runs(fixture_run, tmp_path) -> Path:
    """A private copy of the fixture run (tests may damage it) in its own runs directory."""
    d = tmp_path / "runs"
    shutil.copytree(fixture_run, d / "fx", symlinks=True)
    return d


def board(runs: Path, **kw) -> dash.Dashboard:
    return dash.Dashboard(runs_dir=runs, ledger=runs.parent / "no-ledger.jsonl", **kw)


def get(d: dash.Dashboard, url: str) -> tuple[int, str]:
    code, doc, _ = d.handle(url)
    return code, doc


def errors(doc: str) -> list[str]:
    return re.findall(r"<pre class='err'>(.*?)</pre>", doc, re.S)


def first_eval_item(run_dir: Path, cycle: int = 0) -> str:
    m = read_json(run_dir / "cycles" / f"cycle-{cycle:03d}" / "eval" / "manifest.json")
    return next(iter(m["items"].values()))["output"]


def snapshot(root: Path) -> dict[str, tuple[int, int] | None]:
    out = {}
    for p in sorted(root.rglob("*")):
        st = p.lstat()
        out[p.relative_to(root).as_posix()] = None if p.is_dir() else (st.st_mtime_ns, st.st_size)
    return out


# --------------------------------------------------------------------------- #
# Pages and numbers
# --------------------------------------------------------------------------- #


def test_run_page_sections_and_numbers_match_report(runs):
    rd = runs / "fx"
    code, doc = get(board(runs), "/run/fx/")
    assert code == 200 and not errors(doc)
    for section in ("Across cycles (unassisted evaluation)", "Per cycle", "Paired vs cycle 0", "By family and difficulty", "Cycle timeline",
                    "Recent activity", "Outcomes", "Edits &amp; verification", "Pod &amp; serving", "Coordinator log", "Small samples"):
        assert section in doc, section
    assert "<svg" in doc and "pause auto-refresh" in doc

    # per-cycle evaluation numbers are the report's (same rows, same summarize definitions)
    report = {s["cycle"]: s for s in group_summaries(eval_rows(collect_run(rd).rows), ("cycle",))}
    S = dash.load_state(board(runs).view(rd.resolve()))
    ev = {r["n"]: r for r in dash.stage_cycles(S, "eval")}
    assert set(ev) == set(report) == {0, 1, 2}
    for c, rep in report.items():
        mine = dash.stats(ev[c]["rows"])
        for k in ("n_episodes", "n_success", "success_rate", "total_tokens_mean", "total_tokens_mean_successful", "mean_partial_reward",
                  "stop_categories", "n_with_tokens"):
            assert mine[k] == rep[k], (c, k)
        light = dash.light_summary(ev[c]["rows"])
        assert {k: light[k] for k in light} == {k: summarize(ev[c]["rows"])[k] for k in light}
        assert dash.rate_text(mine) in doc
    # the collection table uses the same rows as the report's collect role
    col = [r for r in collect_run(rd).rows if r.episode.role.value == "collect"]
    assert sum(len(r["rows"]) for r in dash.stage_cycles(S, "collect")) == len(col)


def test_index_lists_runs_newest_first_with_coordinator_state(runs):
    shutil.copytree(runs / "fx", runs / "fx-newer", symlinks=True)
    prov = read_json(runs / "fx" / "provenance.json")
    (runs / "fx-newer" / "provenance.json").write_text(json.dumps({**prov, "recorded_at": "2999-01-01T00:00:00+00:00"}))
    (runs / "fx-newer" / "run.json").write_text(json.dumps({**read_json(runs / "fx" / "run.json"), "run_id": "fx-newer"}))
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    (runs / "fx-newer" / ".lock").write_text(json.dumps({"pid": dead.pid, "host": os.uname().nodename, "since": "2999-01-01T00:00:00+00:00"}))
    (runs / "fx" / ".lock").write_text(json.dumps({"pid": os.getpid(), "host": os.uname().nodename, "since": "2026-01-01T00:00:00+00:00"}))
    (runs / "not-a-run").mkdir()
    code, doc = get(board(runs), "/")
    assert code == 200 and not errors(doc)
    assert doc.index('href="/run/fx-newer/"') < doc.index('href="/run/fx/"')
    assert "not-a-run" not in doc
    assert "finished" in doc and "learning" in doc and "fixture-two-cycles" in doc
    assert f"last pid {dead.pid}" in doc and f"pid {os.getpid()}" in doc  # dead vs alive holder


def test_run_in_progress_shows_position_and_running_cycle(runs):
    rd = runs / "fx"
    shutil.rmtree(rd / "cycles" / "cycle-002")
    cy1 = read_json(rd / "cycles" / "cycle-001" / "cycle.json")
    (rd / "cycles" / "cycle-001" / "cycle.json").write_text(json.dumps({**cy1, "status": "running"}))
    man = read_json(rd / "cycles" / "cycle-001" / "collect" / "manifest.json")
    item = next(iter(man["items"]))
    man["items"][item] = {**man["items"][item], "status": "running", "output": None}
    (rd / "cycles" / "cycle-001" / "collect" / "manifest.json").write_text(json.dumps({**man, "status": "running"}))
    S = dash.load_state(board(runs).view(rd.resolve()))
    cy, stage = dash.current_position(S)
    assert cy["n"] == 1 and "train" in stage
    code, doc = get(board(runs), "/run/fx/")
    assert code == 200 and not errors(doc)
    assert "not started" in doc  # cycle 2 (final evaluation) has no directory yet


def test_transcript_page_escapes_content_and_shows_edit_and_grader(runs, tmp_path):
    rd = runs / "fx"
    # an editor-source episode (collection) is shown with the edited call highlighted
    man = read_json(rd / "cycles" / "cycle-000" / "edit" / "manifest.json")
    src = next(i["meta"]["source_dir"] for i in man["items"].values() if i["meta"].get("status") == "proposed")
    code, doc = get(board(runs), f"/run/fx/ep?p={src}")
    assert code == 200 and not errors(doc)
    assert "edit proposal" in doc and 'class="call edited"' in doc and "editor's replacement" in doc

    rel = first_eval_item(rd)
    ep = rd / rel
    msgs = read_json(ep / "episode" / "messages.json")
    msgs["messages"].append({"role": "assistant", "content": "<script>alert(1)</script>",
                             "tool_calls": [{"id": "x", "function": {"name": "<b>bash</b>", "arguments": '{"command": "echo <img src=x>"}'}}]})
    msgs["messages"].append({"role": "tool", "tool_call_id": "x", "content": "</pre><iframe>"})
    (ep / "episode" / "messages.json").write_text(json.dumps(msgs))
    verifier = ep / "harbor" / "trial-1" / "verifier"
    verifier.mkdir(parents=True)
    (verifier / "test-stdout.txt").write_text("1 passed <b>bold</b>")
    (verifier / "reward.json").write_text('{"reward": 1}')
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET")
    evil = ep / "harbor" / "trial-2" / "verifier"
    evil.mkdir(parents=True)
    (evil / "test-stdout.txt").symlink_to(secret)  # a symlink out of the run is never followed

    code, doc = get(board(runs), f"/run/fx/ep?p={rel}")
    assert code == 200 and not errors(doc)
    for raw in ("<script>alert(1)</script>", "<b>bash</b>", "<img src=x>", "<iframe>", "<b>bold</b>"):
        assert raw not in doc
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in doc and "&lt;iframe&gt;" in doc
    assert "Grader (trial-1)" in doc and "1 passed &lt;b&gt;bold&lt;/b&gt;" in doc
    assert "TOP-SECRET" not in doc


def test_branch_transcript_links_siblings_and_verification(runs):
    rd = runs / "fx"
    ver = next((rd / "cycles" / "cycle-000" / "verify" / "items").iterdir())
    branch = next(p for p in ver.iterdir() if p.is_dir())
    code, doc = get(board(runs), f"/run/fx/ep?p={branch.relative_to(rd).as_posix()}")
    assert code == 200 and not errors(doc)
    assert "verification <code>" in doc and "original-r00" in doc and "edited-r00" in doc


def test_tolerates_truncated_summary_torn_jsonl_and_broken_manifest(runs):
    rd = runs / "fx"
    rel = first_eval_item(rd, 1)
    s = (rd / rel / "summary.json").read_text()
    (rd / rel / "summary.json").write_text(s[: len(s) // 2])
    with open(rd / rel / "episode" / "events.jsonl", "a") as f:
        f.write('{"kind":"response","data":{"his')
    (rd / "logs").mkdir(exist_ok=True)
    (rd / "logs" / "pod-lifecycle.jsonl").write_text('{"t": "2026-01-01T00:00:00+00:00", "event": "created", "pod": "p1"}\n{"t": "20')
    (rd / "logs" / "serving-lifecycle.jsonl").write_text('{"role": "learner", "event": "st')
    m = (rd / "cycles" / "cycle-001" / "collect" / "manifest.json").read_text()
    (rd / "cycles" / "cycle-001" / "collect" / "manifest.json").write_text(m[:40])
    (rd / "cycles" / "cycle-001" / "cycle.json").write_text("{")
    (rd / ".lock").write_text("")
    code, doc = get(board(runs), "/run/fx/")
    assert code == 200 and not errors(doc)
    assert "1 unreadable" in doc
    code, doc = get(board(runs), f"/run/fx/ep?p={rel}")
    assert code == 200 and not errors(doc)
    code, doc = get(board(runs), "/")
    assert code == 200 and not errors(doc)


def test_coordinator_log_from_run_or_override(runs, tmp_path):
    rd = runs / "fx"
    _, doc = get(board(runs), "/run/fx/")
    assert "no <code>logs/coordinator.log</code>" in doc
    (rd / "logs").mkdir(exist_ok=True)
    (rd / "logs" / "coordinator.log").write_text("cycle 0: evaluating\n\x1b[31mcycle 1: <collecting>\x1b[0m\n")
    _, doc = get(board(runs), "/run/fx/")
    assert "cycle 1: &lt;collecting&gt;" in doc and "\x1b" not in doc
    console = tmp_path / "console.log"
    console.write_text("run dir: x\ncycle 2: evaluating c001\n")
    _, doc = get(board(runs, run_dir=rd, log=console), "/run/fx/")
    assert "cycle 2: evaluating c001" in doc and "cycle 1: &lt;collecting&gt;" not in doc


def test_coordinator_log_falls_back_to_the_old_submit_location(runs):
    """Runs fetched before `loop submit` wrote logs/coordinator.log have it at the run root."""
    rd = runs / "fx"
    (rd / "coordinator.log").write_text("cycle 0: old location\n")
    _, doc = get(board(runs), "/run/fx/")
    assert "cycle 0: old location" in doc
    (rd / "logs").mkdir(exist_ok=True)
    (rd / "logs" / "coordinator.log").write_text("cycle 0: new location\n")
    _, doc = get(board(runs), "/run/fx/")
    assert "cycle 0: new location" in doc and "old location" not in doc


# --------------------------------------------------------------------------- #
# Path safety and read-only behaviour
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "url",
    ["/run/fx/ep?p=/etc", "/run/fx/ep?p=../outside", "/run/fx/ep?p=..%2F..%2Foutside", "/run/fx/ep?p=tasks", "/run/fx/ep?p=run.json",
     "/run/fx/ep?p=cycles/../tasks", "/run/fx/ep?p=cycles/cycle-000/eval/items/evil", "/run/fx/ep?p=", "/run/fx/ep",
     "/run/..%2Foutside/", "/run/%2e%2e/", "/run/.hidden/", "/run/linked/", "/run/outside/", "/run/fx/other", "/etc/passwd", "/run/"],
)
def test_refuses_paths_outside_the_runs(runs, url):
    outside = runs.parent / "outside"
    (outside / "cycles").mkdir(parents=True)
    (outside / "run.json").write_text('{"run_id": "TOP-SECRET"}')
    (outside / "summary.json").write_text('{"instance_id": "TOP-SECRET"}')
    (runs / "linked").symlink_to(outside, target_is_directory=True)
    (runs / ".hidden").mkdir()
    (runs / ".hidden" / "run.json").write_text('{"run_id": "TOP-SECRET"}')
    (runs / "fx" / "cycles" / "cycle-000" / "eval" / "items" / "evil").symlink_to(outside, target_is_directory=True)
    code, doc = get(board(runs), url)
    assert code == 404 and "TOP-SECRET" not in doc
    _, index = get(board(runs), "/")
    assert "linked" not in index and "TOP-SECRET" not in index


def test_single_run_mode_serves_only_that_run(runs):
    shutil.copytree(runs / "fx", runs / "fx2", symlinks=True)
    d = board(runs, run_dir=runs / "fx")
    code, _, headers = d.handle("/")
    assert code == 302 and headers["Location"] == "/run/fx/"
    assert get(d, "/run/fx/")[0] == 200
    assert get(d, "/run/fx2/")[0] == 404


def test_rendering_writes_nothing(runs):
    before = snapshot(runs)
    d = board(runs)
    _, page = get(d, "/run/fx/")
    links = sorted(set(re.findall(r'href="(/run/fx/ep\?p=[^"]+)"', page)))
    assert links
    for url in ["/", "/run/fx/", *links]:
        code, doc = get(d, html.unescape(url))
        assert code == 200 and not errors(doc), url
    assert snapshot(runs) == before


def test_live_server_and_cli_usage_errors(runs, tmp_path):
    server = dash.make_server(board(runs), "127.0.0.1", 0)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as r:
            assert r.status == 200 and "fx" in r.read().decode()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/run/fx/") as r:
            assert r.headers["Cache-Control"] == "no-store" and "Cycle timeline" in r.read().decode()
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/run/fx/ep?p=../../etc")
        assert e.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
    assert main(["dashboard", "--log", str(tmp_path / "x.log")]) == 2  # --log needs RUN_DIR
    assert main(["dashboard", str(tmp_path)]) == 2  # not a run directory
