#!/bin/bash
cd /app
python3 - <<'PY'
import re
src = open("stats.py").read()
src = src.replace(
    "    xs.sort()\n    return xs[len(xs) // 2]",
    "    s = sorted(xs)\n    n = len(s)\n    mid = n // 2\n"
    "    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2",
)
src = src.replace("/ len(xs))", "/ (len(xs) - 1))")
src = src.replace("k = (len(s)) * p / 100", "k = (len(s) - 1) * p / 100")
open("stats.py", "w").write(src)
PY
python3 test_stats.py
