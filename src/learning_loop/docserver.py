"""Local documentation viewer (`loop docs`).

Serves the repository's Markdown rendered as HTML on a local HTTP server, starting at
docs/index.md and re-reading files on
every request, so edits show up on reload. Relative links between documents work as in the
repository; Markdown files are rendered, other text files in the repository are shown as source.
Mermaid diagrams are drawn by mermaid.js loaded from a CDN (the source text shows when offline).

Only files inside the repository are served, never run outputs, virtualenvs, caches, private
machine profiles or hidden directories. Binds to 127.0.0.1 unless told otherwise.
"""

from __future__ import annotations

import html
import re
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt

from .config import REPO_ROOT

# Navigation order (other Markdown files are still reachable through links).
HOME = "docs/index.md"
NAV: list[tuple[str, str]] = [
    (HOME, "Index"),
    ("README.md", "Overview"),
    ("docs/architecture.md", "Architecture"),
    ("docs/experiment.md", "Method"),
    ("docs/run-layout.md", "Run layout"),
    ("docs/cli.md", "CLI reference"),
    ("docs/configuration.md", "Configuration"),
    ("docs/operations.md", "Operations"),
    ("docs/runpod.md", "Runpod deployment"),
    ("docs/glossary.md", "Glossary"),
    ("evaluation/README.md", "Tasks and agent"),
    ("AGENTS.md", "Developer guide"),
]

EXCLUDED_PARTS = {".venv", "runs", "artifacts", "node_modules", "__pycache__", ".pytest_cache", ".git"}
EXCLUDED_PREFIXES = ("evaluation/jobs", "configs/machines/local")
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".json", ".jsonl", ".txt", ".sh", ".jinja", ".cfg", ".ini", ".lock", ""}
MAX_TEXT_BYTES = 1 << 20
MERMAID_JS = "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def slugify(text: str, seen: dict[str, int]) -> str:
    """GitHub-style heading anchors (lowercase, punctuation dropped, spaces -> '-'), deduplicated."""
    base = re.sub(r"[^\w\- ]", "", text.strip().lower()).replace(" ", "-")
    n = seen.get(base, 0)
    seen[base] = n + 1
    return base if n == 0 else f"{base}-{n}"


def _markdown() -> MarkdownIt:
    md = MarkdownIt("commonmark", {"html": True}).enable(["table", "strikethrough"])

    def heading_open(self, tokens, idx, options, env):
        inline = tokens[idx + 1]
        text = "".join(t.content for t in (inline.children or []) if t.type in ("text", "code_inline")) or inline.content
        slug = slugify(text, env.setdefault("slugs", {}))
        tokens[idx].attrSet("id", slug)
        level = int(tokens[idx].tag[1])
        if level in (2, 3):
            env.setdefault("toc", []).append((level, text, slug))
        return self.renderToken(tokens, idx, options, env)

    default_fence = md.renderer.rules["fence"]

    def fence(self, tokens, idx, options, env):
        tok = tokens[idx]
        if tok.info.strip().split(" ")[0] == "mermaid":
            env["mermaid"] = True
            return f'<pre class="mermaid">{html.escape(tok.content)}</pre>\n'
        return default_fence(tokens, idx, options, env)

    md.add_render_rule("heading_open", heading_open)
    md.add_render_rule("fence", fence)
    return md


_MD = _markdown()


def render_markdown(text: str) -> tuple[str, bool, list[tuple[int, str, str]]]:
    """(html body, uses mermaid, contents as (level, heading text, anchor) for h2/h3)."""
    env: dict = {}
    body = _MD.render(text, env)
    return body, bool(env.get("mermaid")), env.get("toc", [])


def _toc(entries: list[tuple[int, str, str]]) -> str:
    if len(entries) < 2:
        return ""
    items = "".join(f'<li class="l{lvl}"><a href="#{slug}">{html.escape(text)}</a></li>' for lvl, text, slug in entries)
    return f'<div class="toc"><div class="toc-title">On this page</div><ul>{items}</ul></div>'


def _nav(current: str) -> str:
    items = []
    for rel, label in NAV:
        cls = ' class="current"' if rel == current else ""
        items.append(f'<li><a{cls} href="/{rel}">{html.escape(label)}</a></li>')
    return "<ul>" + "".join(items) + "</ul>"


def page(title: str, body: str, current: str, mermaid: bool = False, banner: str = "", toc: str = "") -> str:
    script = (
        f'<script src="{MERMAID_JS}"></script>'
        '<script>mermaid.initialize({startOnLoad: true, theme: '
        '(matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "default")});</script>'
        if mermaid
        else ""
    )
    banner_html = f'<div class="banner">{banner}</div>' if banner else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} - learn-from-experience docs</title>
<style>
:root {{ --fg:#1f2328; --bg:#ffffff; --muted:#59636e; --line:#d1d9e0; --code:#f6f8fa; --link:#0969da; --warn:#fff8c5; }}
@media (prefers-color-scheme: dark) {{ :root {{ --fg:#e6edf3; --bg:#0d1117; --muted:#9198a1; --line:#3d444d;
  --code:#151b23; --link:#4493f8; --warn:#3b2e00; }} }}
* {{ box-sizing: border-box; }}
body {{ margin:0; font:16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  color:var(--fg); background:var(--bg); display:flex; }}
nav {{ width:230px; flex:none; padding:20px 16px; border-right:1px solid var(--line); position:sticky; top:0;
  height:100vh; overflow-y:auto; }}
nav .title {{ font-weight:600; margin-bottom:12px; }}
nav ul {{ list-style:none; padding:0; margin:0; }}
nav li a {{ display:block; padding:4px 8px; border-radius:6px; color:var(--fg); text-decoration:none; }}
nav li a:hover {{ background:var(--code); }}
nav li a.current {{ background:var(--code); font-weight:600; }}
nav .path {{ margin-top:16px; color:var(--muted); font-size:13px; word-break:break-all; }}
nav .toc {{ margin-top:16px; border-top:1px solid var(--line); padding-top:12px; font-size:14px; }}
nav .toc-title {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; margin-bottom:6px; }}
nav .toc li a {{ padding:2px 8px; color:var(--muted); }}
nav .toc li.l3 a {{ padding-left:20px; font-size:13px; }}
main {{ flex:1; min-width:0; max-width:1100px; padding:24px 40px 64px; }}
a {{ color:var(--link); }}
h1, h2 {{ border-bottom:1px solid var(--line); padding-bottom:.3em; }}
code, pre {{ font:13.5px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; background:var(--code); border-radius:6px; }}
code {{ padding:.15em .35em; }}
pre {{ padding:12px 16px; overflow-x:auto; }}
pre code {{ padding:0; background:none; }}
pre.mermaid {{ background:none; text-align:center; }}
table {{ border-collapse:collapse; display:block; overflow-x:auto; margin:12px 0; }}
th, td {{ border:1px solid var(--line); padding:6px 12px; vertical-align:top; }}
th {{ background:var(--code); }}
.banner {{ background:var(--warn); border:1px solid var(--line); border-radius:6px; padding:8px 12px; margin-bottom:16px; }}
</style></head>
<body><nav><div class="title">learn-from-experience</div>{_nav(current)}<div class="path">{html.escape(current)}</div>{toc}</nav>
<main>{banner_html}{body}</main>{script}</body></html>"""


# --------------------------------------------------------------------------- #
# File access
# --------------------------------------------------------------------------- #


def resolve_request(url_path: str, root: Path = REPO_ROOT) -> Path | None:
    """Repository file for a URL path, or None if it is missing or not allowed to be served."""
    rel = unquote(urlsplit(url_path).path).lstrip("/") or HOME
    root = root.resolve()
    try:
        target = (root / rel).resolve()
        rel_parts = target.relative_to(root).parts
    except (ValueError, OSError):
        return None  # outside the repository (.., absolute paths, symlinks out)
    rel_posix = "/".join(rel_parts)
    if any(p in EXCLUDED_PARTS or p.startswith(".") for p in rel_parts):
        return None
    if any(rel_posix == pre or rel_posix.startswith(pre + "/") for pre in EXCLUDED_PREFIXES):
        return None
    if not target.is_file():
        return None
    if target.suffix != ".md" and target.suffix not in TEXT_SUFFIXES:
        return None
    return target


def render_file(path: Path, root: Path = REPO_ROOT) -> str:
    rel = path.resolve().relative_to(root.resolve()).as_posix()
    text = path.read_text(errors="replace")
    if path.suffix == ".md":
        body, mermaid, toc = render_markdown(text)
        title = next((ln.lstrip("# ").strip() for ln in text.splitlines() if ln.startswith("# ")), rel)
        return page(title, body, rel, mermaid=mermaid, banner=_stale_banner(rel), toc=_toc(toc))
    if path.stat().st_size > MAX_TEXT_BYTES:
        body = f"<h1>{html.escape(rel)}</h1><p>File too large to display ({path.stat().st_size} bytes).</p>"
    else:
        body = f"<h1>{html.escape(rel)}</h1><pre><code>{html.escape(text)}</code></pre>"
    return page(rel, body, rel)


def _stale_banner(rel: str) -> str:
    if rel not in ("docs/cli.md", "docs/configuration.md"):
        return ""
    from .docgen import check_docs

    try:
        stale = {p.resolve().relative_to(REPO_ROOT.resolve()).as_posix() for p in check_docs()}
    except Exception as e:  # noqa: BLE001 - the viewer must still show the page
        return f"Could not check the generated section: {html.escape(str(e))}"
    if rel in stale:
        return "The generated part of this page is out of date. Run <code>uv run loop docs-gen</code>."
    return ""


def not_found(url_path: str) -> str:
    body = (
        f"<h1>Not found</h1><p><code>{html.escape(unquote(url_path))}</code> is not a document this "
        "viewer serves (only repository Markdown and text files; never run outputs, private "
        f'profiles or hidden directories).</p><p><a href="/{HOME}">Back to the index</a></p>'
    )
    return page("Not found", body, "")


# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #


def make_handler(root: Path = REPO_ROOT) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "loop-docs"

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            if urlsplit(self.path).path == "/":
                self.send_response(HTTPStatus.FOUND)
                self.send_header("Location", f"/{HOME}")
                self.end_headers()
                return
            target = resolve_request(self.path, root)
            if target is None:
                self._send(HTTPStatus.NOT_FOUND, not_found(self.path))
                return
            try:
                self._send(HTTPStatus.OK, render_file(target, root))
            except Exception as e:  # noqa: BLE001 - show the error instead of dropping the connection
                self._send(HTTPStatus.INTERNAL_SERVER_ERROR, page("Error", f"<h1>Error</h1><pre>{html.escape(repr(e))}</pre>", ""))

        def _send(self, status: HTTPStatus, body: str) -> None:
            data = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args) -> None:  # noqa: A002 - quiet by default
            pass

    return Handler


def make_server(host: str = "127.0.0.1", port: int = 8000, root: Path = REPO_ROOT) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(root))


def serve(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = False) -> int:
    server = make_server(host, port)
    actual_port = server.server_address[1]
    url = f"http://{'localhost' if host in ('127.0.0.1', 'localhost') else host}:{actual_port}/"
    print(f"Serving documentation at {url}  (Ctrl-C to stop)", flush=True)
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"note: bound to {host}; other machines on the network can read the repository documents", flush=True)
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0
