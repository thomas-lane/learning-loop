"""Rendering is host-independent (marker `docker`): every family and difficulty renders to the
same bytes, modes, mtimes and symlinks on this host as inside the pinned profile image, which
has another Python, zlib and SQLite. A difference would give the same instance different
content (and content hashes) on different machines."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from learning_loop.core.config import REPO_ROOT
from learning_loop.tasks.spec import PROFILES

pytestmark = pytest.mark.docker

MANIFEST = r'''
import hashlib, json, os, stat, sys, tempfile
from pathlib import Path
from evaluation.families import FAMILIES
from learning_loop.tasks.render import render

def manifest(root):
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in sorted(dirnames + filenames):
            p = os.path.join(dirpath, name)
            st = os.lstat(p)
            rel = os.path.relpath(p, root)
            if stat.S_ISLNK(st.st_mode):
                out[rel] = ["L", os.readlink(p), st.st_mtime_ns]
            elif stat.S_ISDIR(st.st_mode):
                out[rel] = ["D", stat.S_IMODE(st.st_mode), st.st_mtime_ns]
            else:
                out[rel] = ["F", stat.S_IMODE(st.st_mode), st.st_mtime_ns, hashlib.sha256(Path(p).read_bytes()).hexdigest()]
    return out

result = {}
with tempfile.TemporaryDirectory() as tmp:
    for name, fam in sorted(FAMILIES.items()):
        for d in fam.difficulties:
            out = Path(tmp) / f"{name}__{d}"
            render(fam, d, 1, out)
            result[f"{name}/{d}"] = manifest(out)
json.dump(result, sys.stdout, sort_keys=True)
'''


def _docker_ok() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


if not _docker_ok():  # pragma: no cover
    pytest.skip("Docker daemon not available", allow_module_level=True)


def _host() -> dict:
    env = {"PYTHONPATH": f"{REPO_ROOT}:{REPO_ROOT / 'src'}", "PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8", "PYTHONHASHSEED": "0"}
    out = subprocess.run([sys.executable, "-B", "-c", MANIFEST], capture_output=True, text=True, env=env, cwd=REPO_ROOT, timeout=1800)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout)


def _container() -> dict:
    image = PROFILES["python@1"].base_image
    cmd = ["docker", "run", "--rm", "--network", "none", "-v", f"{REPO_ROOT}:/src:ro", "-w", "/src",
           "-e", "PYTHONPATH=/src:/src/src", "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "LC_ALL=C.UTF-8", "-e", "PYTHONHASHSEED=0",
           image, "python3", "-B", "-c", MANIFEST]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout)


def test_every_family_renders_identically_on_the_host_and_in_the_profile_image():
    host, container = _host(), _container()
    assert sorted(host) == sorted(container)
    differences = {
        inst: sorted(p for p in set(host[inst]) | set(container[inst]) if host[inst].get(p) != container[inst].get(p))
        for inst in host
    }
    assert {k: v for k, v in differences.items() if v} == {}
