"""`loop docs` viewer: rendering, path safety, a live server, and index/navigation coverage."""

import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from learning_loop import docserver
from learning_loop.config import REPO_ROOT


def test_render_headings_tables_mermaid_and_escaping():
    body, mermaid, toc = docserver.render_markdown(
        "# Title\n\n## Data flow\n\n### Run `loop docs`\n\n## Data flow\n\n"
        "| a | b |\n|---|---|\n| 1 | 2 |\n\n```mermaid\nflowchart LR\n  A-->B\n```\n\n```text\n<run-id>\n```\n"
    )
    assert '<h2 id="data-flow">' in body and '<h2 id="data-flow-1">' in body  # GitHub-style, deduplicated
    assert '<h3 id="run-loop-docs">' in body
    assert "<table>" in body
    assert mermaid and '<pre class="mermaid">flowchart LR\n  A--&gt;B\n</pre>' in body
    assert "&lt;run-id&gt;" in body
    assert [t[2] for t in toc] == ["data-flow", "run-loop-docs", "data-flow-1"]


@pytest.mark.parametrize(
    "url",
    ["/../pyproject.toml", "/%2e%2e/pyproject.toml", "/runs/x/run.json", "/.venv/pyvenv.cfg", "/evaluation/jobs/x/result.json",
     "/configs/machines/local/lab.yaml", "/src/learning_loop/__pycache__/x.pyc", "/docs/missing.md", "/uv.lock.bin"],
)
def test_refuses_paths_outside_the_served_set(url):
    assert docserver.resolve_request(url) is None


def test_serves_markdown_and_repository_text_files():
    assert docserver.resolve_request("/") == REPO_ROOT / "docs" / "index.md"
    assert docserver.resolve_request("/docs/cli.md") == REPO_ROOT / "docs" / "cli.md"
    assert docserver.resolve_request("/experiments/pilot.yaml") == REPO_ROOT / "experiments" / "pilot.yaml"


def test_every_document_is_in_the_index_and_navigation():
    docs = {p.relative_to(REPO_ROOT).as_posix() for p in (REPO_ROOT / "docs").glob("*.md")}
    maintained = docs | {"README.md", "AGENTS.md", "evaluation/README.md"}
    nav = {rel for rel, _ in docserver.NAV}
    assert maintained == nav, f"update NAV in docserver.py: {maintained ^ nav}"
    index = (REPO_ROOT / "docs" / "index.md").read_text()
    for rel in maintained - {"docs/index.md"}:
        link = Path(rel).name if rel.startswith("docs/") else f"../{rel}"
        assert f"]({link})" in index, f"docs/index.md does not link {rel}"


def test_live_server_pages_and_404():
    server = docserver.make_server("127.0.0.1", 0)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as r:  # redirect to the index
            page = r.read().decode()
            assert r.url.endswith("/docs/index.md") and "Documentation index" in page and "On this page" in page
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/docs/architecture.md") as r:
            page = r.read().decode()
            assert 'class="mermaid"' in page and "mermaid.min.js" in page and 'class="current"' in page
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/experiments/pilot.yaml") as r:
            assert "<pre><code>" in r.read().decode()
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/runs/anything/run.json")
        assert e.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
