"""Declared task-state fingerprint.

This file is deliberately standard-library only and Python-3.8 compatible: its
source is shipped verbatim into task containers (`python3 -I -B -`) so that the
Docker and local fixture environments compute fingerprints with the *same*
code.

What is covered, for each declared path (recursively, symlinks not followed):

    D <path> <mode> <uid>:<gid>                       directory
    F <path> <mode> <uid>:<gid> <size> <sha256>       regular file (content hash)
    L <path> <mode> <uid>:<gid> -> <target>           symlink (target text)
    O <path> <kind> <mode> <uid>:<gid>                fifo/socket/device
    M <path>                                          declared path missing
    C <cwd> dir|missing                               tool working directory

Modification/access times are *not* covered (they depend on wall-clock time
and would make every replay differ). Excludes are fnmatch patterns matched
against the displayed absolute path; a matching directory is pruned. Process,
network and clock state are not covered at all: tasks that depend on them
cannot declare deterministic replay.
"""

import fnmatch
import hashlib
import json
import os
import stat
import sys

FORMAT_VERSION = 1


def _q(path):
    return json.dumps(path)


def _sha256(path):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError as e:
        return "ERR:%s" % (e.errno,)
    return h.hexdigest()


def _kind(mode):
    if stat.S_ISFIFO(mode):
        return "fifo"
    if stat.S_ISSOCK(mode):
        return "socket"
    if stat.S_ISCHR(mode):
        return "chardev"
    if stat.S_ISBLK(mode):
        return "blockdev"
    return "other"


class _Walker:
    def __init__(self, excludes, path_map):
        self.excludes = list(excludes)
        self.path_map = list(path_map)  # [(real_prefix, shown_prefix)]
        self.lines = []

    def shown(self, real):
        for real_prefix, shown_prefix in self.path_map:
            if real == real_prefix or real.startswith(real_prefix.rstrip("/") + "/"):
                return shown_prefix.rstrip("/") + real[len(real_prefix.rstrip("/")):] or "/"
        return real

    def excluded(self, shown):
        return any(fnmatch.fnmatchcase(shown, pat) for pat in self.excludes)

    def entry(self, real, shown):
        try:
            st = os.lstat(real)
        except OSError:
            self.lines.append("M %s" % _q(shown))
            return False
        mode = "%04o" % stat.S_IMODE(st.st_mode)
        own = "%d:%d" % (st.st_uid, st.st_gid)
        if stat.S_ISLNK(st.st_mode):
            self.lines.append("L %s %s %s -> %s" % (_q(shown), mode, own, _q(os.readlink(real))))
            return False
        if stat.S_ISDIR(st.st_mode):
            self.lines.append("D %s %s %s" % (_q(shown), mode, own))
            return True
        if stat.S_ISREG(st.st_mode):
            self.lines.append("F %s %s %s %d %s" % (_q(shown), mode, own, st.st_size, _sha256(real)))
            return False
        self.lines.append("O %s %s %s %s" % (_q(shown), _kind(st.st_mode), mode, own))
        return False

    def walk(self, real):
        shown = self.shown(real)
        if self.excluded(shown):
            return
        if not self.entry(real, shown):
            return
        try:
            names = sorted(os.listdir(real))
        except OSError as e:
            self.lines.append("E %s %s" % (_q(shown), e.errno))
            return
        for name in names:
            self.walk(os.path.join(real, name))


def compute(paths, excludes=(), cwd=None, path_map=()):
    """Return (sha256, lines). `paths`/`cwd` are real paths; `path_map` maps
    real prefixes to the displayed (declared) prefixes."""
    w = _Walker(excludes, path_map)
    if cwd is not None:
        w.lines.append("C %s %s" % (_q(w.shown(cwd)), "dir" if os.path.isdir(cwd) else "missing"))
    for p in paths:
        w.walk(p)
    body = "fingerprint-v%d\n" % FORMAT_VERSION + "\n".join(w.lines)
    return hashlib.sha256(body.encode("utf-8", "surrogateescape")).hexdigest(), w.lines


def main(argv):
    spec = json.loads(argv[1])
    digest, lines = compute(spec["paths"], spec.get("exclude", []), spec.get("cwd"))
    sys.stdout.write(json.dumps({"sha256": digest, "lines": lines}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
