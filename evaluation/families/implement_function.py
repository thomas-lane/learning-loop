"""implement-function: implement a function from its docstring and a few examples.

`/app/<module>.py` holds stubs that raise `NotImplementedError`. Each stub's docstring
specifies the function precisely and shows doctest examples. The functions come from
`CATALOG`, a set of small, precisely specified functions; each entry draws its own spec
parameters from the seed (separators, run lengths, how touching intervals or trailing
slashes are treated), so instances differ in behavior, not only in their examples.

Grading: hidden `Checks` call the functions on seed-specific inputs that are not the
visible examples, plus every edge case the docstring states (`equal` for results, `raises`
for the stated ValueErrors, `no_mutation` where the docstring promises it).

Traps, each a declared shortcut that fails by construction because a hidden check targets it:
`hardcode-examples` (returns the visible examples' outputs and nothing else), and per catalog
entry the plausible wrong implementations: ignoring a stated edge case (unsorted input, an
empty string, `..` above the start of a relative path), off-by-one boundaries (a run of
exactly the minimum length, the last window, intervals that only touch) and wrong escaping
(toggling on every quote instead of honoring doubled quotes).

Difficulty: easy is one easy-tier function with four examples, one of them an edge case;
medium one medium-tier function with three typical examples; hard two functions in one
module (one hard-tier, one medium-tier) with two typical examples each.
"""

from dataclasses import dataclass
from typing import Any, Callable

from learning_loop.tasks.runtime.grade import run_checks
from learning_loop.tasks.spec import Checks, Family, Reject, Solution, TaskSpec

INSTRUCTION_ONE = """Implement the function `{name}` in `/app/{module}` so that it does what its docstring says.

The docstring specifies the behavior precisely and shows a few examples; you can run the examples with `python3 -m doctest -v {module}` in `/app`. The function will also be called with other inputs, including the edge cases the docstring describes.

Keep the function name and signature, use only the Python standard library, and do not print anything.
"""

INSTRUCTION_MANY = """Implement the functions {names} in `/app/{module}` so that each does what its docstring says.

The docstrings specify the behavior precisely and show a few examples; you can run the examples with `python3 -m doctest -v {module}` in `/app`. The functions will also be called with other inputs, including the edge cases the docstrings describe.

Keep the function names and signatures, use only the Python standard library, and do not print anything.
"""

N_HIDDEN_TYPICAL = 4
WORDS = ["alpha", "bravo", "cedar", "delta", "ember", "fjord", "grove", "harbor", "iris", "jade", "kite", "lumen", "maple", "nova", "opal", "pine", "quill", "reef", "slate", "tide"]
SHORT_WORDS = ["a", "an", "at", "is", "of", "on", "the", "to", "go", "it", "cat", "dog", "sun", "map"]
PATH_NAMES = ["usr", "lib", "home", "etc", "var", "src", "docs", "tmp", "data", "bin", "log", "app"]


@dataclass(frozen=True)
class Entry:
    """One catalog function. `body(p)` is the reference body, `wrong(p)` maps shortcut names
    to wrong bodies, `typical(rng, p)` draws one argument list, `edges(rng, p)` the stated edge
    cases as (label, args) and `shown_edge` the label an easy instance shows as an example."""

    name: str
    module: str
    tier: str
    signature: str
    params: Callable[[Any], dict]
    doc: Callable[[dict], str]
    body: Callable[[dict], str]
    wrong: Callable[[dict], dict[str, str]]
    typical: Callable[[Any, dict], list]
    edges: Callable[[Any, dict], list[tuple[str, list]]]
    shown_edge: str
    no_mutation: Callable[[Any, dict], list] | None = None  # args whose first element must not change


def _sub(src: str, **values) -> str:
    for k, v in values.items():
        src = src.replace(f"@{k}@", repr(v))
    return src


# --------------------------------------------------------------------------- #
# compress_ranges (easy)
# --------------------------------------------------------------------------- #

COMPRESS_BODY = """    values = sorted(set(nums))
    items = []
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[j + 1] == values[j] + 1:
            j += 1
        if j - i + 1 >= @MIN_RUN@:
            items.append(str(values[i]) + @SEP@ + str(values[j]))
        else:
            items.extend(str(v) for v in values[i : j + 1])
        i = j + 1
    return @JOINER@.join(items)
"""


def _compress_doc(p):
    return (
        "Describe a collection of non-negative integers compactly.\n\n"
        "The input may be in any order and may contain duplicates; each distinct value is described\n"
        "once, in increasing order. A maximal run of consecutive values (such as 4, 5, 6) with\n"
        f"{p['min_run']} or more values is written as first{p['sep']}last; the values of a shorter run are\n"
        f"written one by one. Items are separated by {p['joiner']!r}. An empty input gives ''.\n"
        "The input list is not modified."
    )


def _compress_runs(rng, lengths):
    out, v = [], rng.randint(0, 9)
    for n in lengths:
        out += list(range(v, v + n))
        v += n + rng.randint(2, 6)
    return out


def _compress_typical(rng, p):
    lengths = [rng.choice([1, 1, 2, 3, 4, 5]) for _ in range(rng.randint(3, 5))]
    return [_compress_runs(rng, lengths)]


def _compress_edges(rng, p):
    exact = _compress_runs(rng, [p["min_run"], p["min_run"] - 1, rng.randint(p["min_run"] + 1, 6)])
    messy = _compress_runs(rng, [rng.randint(2, 5), 1, rng.randint(p["min_run"], 5)])
    messy = messy + rng.sample(messy, 3)
    while messy == sorted(messy):
        rng.shuffle(messy)
    rng.shuffle(messy)
    return [("unsorted", [messy]), ("exact_run", [exact]), ("empty", [[]]), ("single", [[rng.randint(0, 99)]])]


COMPRESS = Entry(
    name="compress_ranges",
    module="ranges.py",
    tier="easy",
    signature="nums: list[int]) -> str",
    params=lambda rng: {"sep": rng.choice(["-", "..", "~"]), "min_run": rng.choice([2, 3]), "joiner": rng.choice([",", ", "])},
    doc=_compress_doc,
    body=lambda p: _sub(COMPRESS_BODY, MIN_RUN=p["min_run"], SEP=p["sep"], JOINER=p["joiner"]),
    wrong=lambda p: {
        "unsorted-input": _sub(COMPRESS_BODY.replace("values = sorted(set(nums))", "values = list(nums)"), MIN_RUN=p["min_run"], SEP=p["sep"], JOINER=p["joiner"]),
        "run-length-off-by-one": _sub(COMPRESS_BODY.replace(">= @MIN_RUN@", "> @MIN_RUN@"), MIN_RUN=p["min_run"], SEP=p["sep"], JOINER=p["joiner"]),
    },
    typical=_compress_typical,
    edges=_compress_edges,
    shown_edge="unsorted",
)

# --------------------------------------------------------------------------- #
# rle_encode (easy)
# --------------------------------------------------------------------------- #

RLE_FMT = """    def run(ch, n):
        count = "" if @OMIT@ and n == 1 else str(n)
        return count + ch if @COUNT_FIRST@ else ch + count

"""
RLE_BODY = RLE_FMT + """    out = []
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append(run(s[i], j - i))
        i = j
    return "".join(out)
"""
RLE_DROPS_LAST = RLE_FMT + """    out = []
    prev, n = s[:1], 0
    for ch in s:
        if ch == prev:
            n += 1
        else:
            out.append(run(prev, n))
            prev, n = ch, 1
    return "".join(out)
"""
RLE_EMPTY_CRASH = RLE_FMT + """    out = []
    prev, n = s[0], 0
    for ch in s:
        if ch == prev:
            n += 1
        else:
            out.append(run(prev, n))
            prev, n = ch, 1
    out.append(run(prev, n))
    return "".join(out)
"""


def _rle_doc(p):
    first = p["count_first"]
    one = 'as just the letter ("b" -> "b")' if p["omit_one"] else ('with the count 1 ("b" -> "1b")' if first else 'with the count 1 ("b" -> "b1")')
    return (
        "Run-length encode a string of lowercase letters.\n\n"
        + ('Each maximal run of one repeated letter becomes the run\'s length followed by the letter\n("aaa" -> "3a"). '
           if first else 'Each maximal run of one repeated letter becomes the letter followed by the run\'s length\n("aaa" -> "a3"). ')
        + f"A run of a single letter is written {one}.\n"
        + "Lengths can have several digits (twelve a's give " + ('"12a"' if first else '"a12"') + "). The empty string encodes to\nthe empty string."
    )


def _rle_from(rng, lengths):
    letters, out = "abcdefghjkmnpqrstuwxyz", []
    prev = ""
    for n in lengths:
        ch = rng.choice([c for c in letters if c != prev])
        out.append(ch * n)
        prev = ch
    return "".join(out)


def _rle_typical(rng, p):
    return [_rle_from(rng, [rng.choice([1, 1, 2, 3, 4, 6]) for _ in range(rng.randint(3, 6))])]


def _rle_edges(rng, p):
    return [
        ("long_run", [_rle_from(rng, [rng.randint(1, 3), rng.randint(10, 14), 1])]),
        ("singles", [_rle_from(rng, [1] * rng.randint(3, 5))]),
        ("empty", [""]),
        ("one_char", [_rle_from(rng, [1])]),
    ]


def _rle_sub(src, p):
    return _sub(src, OMIT=p["omit_one"], COUNT_FIRST=p["count_first"])


RLE = Entry(
    name="rle_encode",
    module="rle.py",
    tier="easy",
    signature="s: str) -> str",
    params=lambda rng: {"count_first": rng.choice([True, False]), "omit_one": rng.choice([True, False])},
    doc=_rle_doc,
    body=lambda p: _rle_sub(RLE_BODY, p),
    wrong=lambda p: {"drops-last-run": _rle_sub(RLE_DROPS_LAST, p), "empty-input-crash": _rle_sub(RLE_EMPTY_CRASH, p)},
    typical=_rle_typical,
    edges=_rle_edges,
    shown_edge="long_run",
)

# --------------------------------------------------------------------------- #
# window_sums (easy)
# --------------------------------------------------------------------------- #

WINDOW_BODY = """    if k < 1:
        raise ValueError("k must be at least 1")
    return [sum(xs[i : i + k]) for i in range(len(xs) - k + 1)]
"""


def _window_typical(rng, p):
    xs = [rng.randint(-20, 50) for _ in range(rng.randint(5, 8))]
    return [xs, rng.randint(2, 4)]


def _window_edges(rng, p):
    xs = [rng.randint(-20, 50) for _ in range(rng.randint(3, 5))]
    ys = [rng.randint(-20, 50) for _ in range(rng.randint(3, 6))]
    return [
        ("k_too_big", [xs, len(xs) + rng.randint(1, 3)]),
        ("k_equals_len", [ys, len(ys)]),
        ("k_zero", [ys, 0]),
        ("k_negative", [xs, -rng.randint(1, 3)]),
        ("empty", [[], 1]),
    ]


WINDOW = Entry(
    name="window_sums",
    module="windows.py",
    tier="easy",
    signature="xs: list[int], k: int) -> list[int]",
    params=lambda rng: {},
    doc=lambda p: (
        "The sum of every window of k consecutive elements of xs, from left to right.\n\n"
        "A list of n elements has n - k + 1 such windows; if k is larger than len(xs) there are none\n"
        "and the result is []. Raise ValueError if k is less than 1."
    ),
    body=lambda p: WINDOW_BODY,
    wrong=lambda p: {
        "last-window-off-by-one": WINDOW_BODY.replace("range(len(xs) - k + 1)", "range(len(xs) - k)"),
        "no-k-check": WINDOW_BODY.replace('    if k < 1:\n        raise ValueError("k must be at least 1")\n', ""),
    },
    typical=_window_typical,
    edges=_window_edges,
    shown_edge="k_too_big",
)

# --------------------------------------------------------------------------- #
# merge_intervals (medium)
# --------------------------------------------------------------------------- #

MERGE_BODY = """    out = []
    for start, end in sorted(intervals):
        if out and start @CMP@ out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out
"""


def _merge_src(p, flip=False, unsorted=False, contained=False, in_place=False):
    merge = p["touching"] == "merge"
    src = MERGE_BODY.replace("@CMP@", "<=" if merge != flip else "<")
    if unsorted:
        src = src.replace("sorted(intervals)", "intervals")
    if contained:
        src = src.replace("max(out[-1][1], end)", "end")
    if in_place:
        src = src.replace("    for start, end in sorted(intervals):", "    intervals.sort()\n    for start, end in intervals:")
    return src


def _merge_doc(p):
    touch = (
        "Intervals that only touch, such as [1, 3] and [3, 5], are merged too (into [1, 5])."
        if p["touching"] == "merge"
        else "Intervals that only touch, such as [1, 3] and [3, 5], share no number and stay separate."
    )
    return (
        "Merge half-open intervals.\n\n"
        "Each interval is a pair [start, end] of integers with start < end; it contains the numbers x\n"
        "with start <= x < end. The input may be in any order. Return the union as a list of\n"
        "[start, end] pairs sorted by start, merging intervals that overlap.\n"
        f"{touch}\nAn empty input gives []. The input list is not modified."
    )


def _interval(rng, lo, hi):
    a = rng.randint(lo, hi)
    return [a, a + rng.randint(1, 6)]


def _merge_typical(rng, p):
    xs = [_interval(rng, 0, 40) for _ in range(rng.randint(4, 6))]
    while len({tuple(x) for x in xs}) < 2:
        xs.append(_interval(rng, 0, 40))
    while xs == sorted(xs):
        rng.shuffle(xs)
    return [xs]


def _merge_edges(rng, p):
    a = rng.randint(0, 20)
    b = a + rng.randint(2, 5)
    c = b + rng.randint(2, 5)
    far = c + rng.randint(3, 8)
    touching = [[b, c], [far, far + 2], [a, b]]  # [a, b] and [b, c] only touch
    outer = [a, a + rng.randint(8, 12)]
    contained = [outer, [outer[0] + 2, outer[0] + 4], [outer[1] + 3, outer[1] + 5]]  # the second lies inside the first
    return [("touching", [touching]), ("contained", [contained]), ("empty", [[]]), ("single", [[_interval(rng, 0, 30)]])]


MERGE = Entry(
    name="merge_intervals",
    module="intervals.py",
    tier="medium",
    signature="intervals: list[list[int]]) -> list[list[int]]",
    params=lambda rng: {"touching": rng.choice(["merge", "separate"])},
    doc=_merge_doc,
    body=_merge_src,
    wrong=lambda p: {
        "touching-off-by-one": _merge_src(p, flip=True),
        "unsorted-input": _merge_src(p, unsorted=True),
        "contained-interval": _merge_src(p, contained=True),
        "sorts-in-place": _merge_src(p, in_place=True),
    },
    typical=_merge_typical,
    edges=_merge_edges,
    shown_edge="touching",
    no_mutation=_merge_typical,  # an unsorted list, which sorting in place would reorder
)

# --------------------------------------------------------------------------- #
# top_words (medium)
# --------------------------------------------------------------------------- #

TOP_BODY = """    import re

    counts = {}
    for word in re.findall(r"[A-Za-z]+", text):
        word = word.lower()
        if len(word) >= @MIN_LEN@:
            counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [[w, c] for w, c in ranked[:k]]
"""
TOP_FIRST_SEEN = """    import re
    from collections import Counter

    words = [w.lower() for w in re.findall(r"[A-Za-z]+", text)]
    counts = Counter(w for w in words if len(w) >= @MIN_LEN@)
    return [[w, c] for w, c in counts.most_common(k)]
"""


def _top_words_text(rng, counts, noise=True):
    words = [w for w, n in counts.items() for _ in range(n)]
    rng.shuffle(words)
    out = []
    for w in words:
        w = w.capitalize() if rng.random() < 0.25 else w
        out.append(w + (rng.choice([",", ".", "!", ";", ""]) if noise and rng.random() < 0.3 else ""))
    return " ".join(out)


def _top_typical(rng, p):
    vocab = rng.sample(WORDS, 5)
    counts = {w: rng.randint(1, 4) for w in vocab}
    counts.update({w: rng.randint(1, 2) for w in rng.sample(SHORT_WORDS, 3)})
    return [_top_words_text(rng, counts), rng.randint(2, 4)]


def _top_edges(rng, p):
    m = p["min_len"]
    a, b, c = sorted(rng.sample(WORDS, 3))
    # tie: c (alphabetically last) is seen first, a last; both counted twice
    tie = f"{c.capitalize()} {b} {b} {b} {c}, {a}. {a.upper()}!"
    exact = rng.choice([w for w in SHORT_WORDS if len(w) == m] or [w for w in WORDS if len(w) == m])
    exact_text = " ".join([exact] * 4 + [a, a, b])
    tok = rng.choice(WORDS)
    joined = f"{tok}_{a} {tok}_{a} v2{b} {tok}42 {c}"
    return [
        ("tie_order", [tie, 3]),
        ("exact_min_length", [exact_text, 2]),
        ("non_letters", [joined, 3]),
        ("k_too_big", [_top_words_text(rng, {a: 2, b: 1}), 5]),
    ]


TOP = Entry(
    name="top_words",
    module="words.py",
    tier="medium",
    signature="text: str, k: int) -> list[list]",
    params=lambda rng: {"min_len": rng.choice([2, 3, 4])},
    doc=lambda p: (
        "The k most frequent words in text, as [word, count] pairs.\n\n"
        "A word is a maximal run of ASCII letters (a-z, A-Z); every other character separates words.\n"
        f"Case is ignored and words are returned in lower case. Words shorter than {p['min_len']} letters are\n"
        "ignored. The most frequent word comes first; words with the same count are in alphabetical\n"
        "order. If there are fewer than k distinct words, return all of them. k is at least 1."
    ),
    body=lambda p: _sub(TOP_BODY, MIN_LEN=p["min_len"]),
    wrong=lambda p: {
        "first-seen-ties": _sub(TOP_FIRST_SEEN, MIN_LEN=p["min_len"]),
        "min-length-off-by-one": _sub(TOP_BODY.replace(">= @MIN_LEN@", "> @MIN_LEN@"), MIN_LEN=p["min_len"]),
        "word-regex-w": _sub(TOP_BODY.replace('r"[A-Za-z]+"', 'r"\\w+"'), MIN_LEN=p["min_len"]),
    },
    typical=_top_typical,
    edges=_top_edges,
    shown_edge="tie_order",
)

# --------------------------------------------------------------------------- #
# version_compare (medium)
# --------------------------------------------------------------------------- #

VERSION_PARTS = """    def parts(v):
        if @ALLOW_V@ and v[:1] in ("v", "V"):
            v = v[1:]
        return [int(x) for x in v.split(".")]

    pa, pb = parts(a), parts(b)
"""
VERSION_BODY = VERSION_PARTS + """    n = max(len(pa), len(pb))
    pa += [0] * (n - len(pa))
    pb += [0] * (n - len(pb))
    return (pa > pb) - (pa < pb)
"""
VERSION_ZIP = VERSION_PARTS + """    for x, y in zip(pa, pb):
        if x != y:
            return 1 if x > y else -1
    return 0
"""
VERSION_NO_PAD = VERSION_PARTS + "    return (pa > pb) - (pa < pb)\n"
VERSION_STRING = "    return (a > b) - (a < b)\n"


def _version(rng, n=None):
    return ".".join(str(rng.randint(0, 12)) for _ in range(n or rng.randint(2, 4)))


def _version_typical(rng, p):
    a = _version(rng)
    if rng.random() < 0.5:
        parts = a.split(".")
        i = rng.randrange(len(parts))
        parts[i] = str(int(parts[i]) + rng.randint(1, 9))
        b = ".".join(parts)
    else:
        b = _version(rng)
    return [a, b] if rng.random() < 0.5 else [b, a]


def _version_edges(rng, p):
    base = _version(rng, 2)
    major = rng.randint(1, 5)
    out = [
        ("trailing_zero", [base + ".0", base]),
        ("longer_is_greater", [base, base + "." + str(rng.randint(1, 9))]),
        ("multi_digit", [f"{major}.{rng.randint(10, 19)}", f"{major}.{rng.randint(2, 9)}"]),
        ("leading_zero", [f"{major}.0{rng.randint(1, 9)}", f"{major}.{rng.randint(1, 9)}"]),
    ]
    if p["allow_v"]:
        out.append(("v_prefix", ["v" + base, base + ".0"]))
    return out


VERSION = Entry(
    name="version_compare",
    module="versions.py",
    tier="medium",
    signature="a: str, b: str) -> int",
    params=lambda rng: {"allow_v": rng.choice([True, False])},
    doc=lambda p: (
        "Compare two version strings: -1 if a is lower than b, 0 if they are equal, 1 if a is higher.\n\n"
        "A version is one or more non-negative integers separated by dots, such as \"1.10.2\".\n"
        + ('It may be preceded by "v" or "V", which is ignored ("v1.2" equals "1.2").\n' if p["allow_v"] else "")
        + "Versions compare component by component as numbers, so \"1.10\" is higher than \"1.9\" and\n"
        "\"1.02\" equals \"1.2\". Missing trailing components count as 0: \"1.2\" equals \"1.2.0\" and is\n"
        "lower than \"1.2.1\"."
    ),
    body=lambda p: _sub(VERSION_BODY, ALLOW_V=p["allow_v"]),
    wrong=lambda p: {
        "zip-truncates": _sub(VERSION_ZIP, ALLOW_V=p["allow_v"]),
        "no-zero-padding": _sub(VERSION_NO_PAD, ALLOW_V=p["allow_v"]),
        "string-compare": VERSION_STRING,
    },
    typical=_version_typical,
    edges=_version_edges,
    shown_edge="trailing_zero",
)

# --------------------------------------------------------------------------- #
# normalize_path (hard)
# --------------------------------------------------------------------------- #

PATH_BODY = """    absolute = path.startswith("/")
    parts = []
    for comp in path.split("/"):
        if comp in ("", "."):
            continue
        if comp == "..":
            if parts and parts[-1] != "..":
                parts.pop()
            elif not absolute:
                parts.append("..")
            continue
        parts.append(comp)
    result = ("/" if absolute else "") + "/".join(parts)
    if not result:
        return "."
    if @KEEP@ and path.endswith("/") and result != "/":
        result += "/"
    return result
"""
PATH_LIBRARY = """    import posixpath

    result = posixpath.normpath(path)
    if @KEEP@ and path.endswith("/") and result not in ("/", "//", "."):
        result += "/"
    return result
"""


def _path_doc(p):
    trail = (
        "If the input ends with a slash, the result keeps one trailing slash (\"a/b/\" stays \"a/b/\"),\n"
        "unless the result is \"/\" or \".\"; otherwise the result has no trailing slash."
        if p["keep_trailing"]
        else "The result has no trailing slash (\"a/b/\" becomes \"a/b\"), except the root \"/\" itself."
    )
    return (
        "Normalize a POSIX path lexically (the file system is never consulted).\n\n"
        "- A run of slashes counts as one slash, also at the start (\"//a\" becomes \"/a\").\n"
        "- \".\" components are removed.\n"
        "- \"..\" removes the component before it. At the root of an absolute path it is dropped\n"
        "  (\"/../a\" becomes \"/a\"); in a relative path with nothing left to remove it is kept\n"
        "  (\"../a\" stays \"../a\" and \"a/../..\" becomes \"..\").\n"
        f"- {trail}\n"
        "- A relative path that normalizes to nothing, and the empty string, give \".\"."
    )


def _rel_path(rng, n):
    comps = []
    for _ in range(n):
        r = rng.random()
        comps.append("." if r < 0.12 else ".." if r < 0.3 else rng.choice(PATH_NAMES))
    return comps


def _path_typical(rng, p):
    comps = _rel_path(rng, rng.randint(3, 6))
    path = "/".join(comps)
    if rng.random() < 0.6:
        path = "/" + path
    if rng.random() < 0.3:
        path = path.replace("/", "//", 1)
    return [path]


def _path_edges(rng, p):
    a, b, c = rng.sample(PATH_NAMES, 3)
    return [
        ("double_slash_root", [f"//{a}/{b}/../{c}"]),
        ("leading_dotdot", [f"{a}/../../{b}/./{c}"]),
        ("above_root", [f"/../{a}/../../{b}"]),
        ("trailing_slash", [f"{a}//{b}/./"]),
        ("dot_only", ["./"]),
        ("empty", [""]),
        ("all_up", [f"{a}/{b}/../../.."]),
    ]


def _path_sub(src, p):
    return _sub(src, KEEP=p["keep_trailing"])


PATH = Entry(
    name="normalize_path",
    module="paths.py",
    tier="hard",
    signature="path: str) -> str",
    params=lambda rng: {"keep_trailing": rng.choice([True, False])},
    doc=_path_doc,
    body=lambda p: _path_sub(PATH_BODY, p),
    wrong=lambda p: {
        "library-normpath": _path_sub(PATH_LIBRARY, p),
        "drops-leading-dotdot": _path_sub(PATH_BODY.replace("            elif not absolute:\n                parts.append(\"..\")\n", ""), p),
        "keeps-dotdot-at-root": _path_sub(PATH_BODY.replace("            elif not absolute:\n", "            else:\n"), p),
    },
    typical=_path_typical,
    edges=_path_edges,
    shown_edge="leading_dotdot",
)

# --------------------------------------------------------------------------- #
# split_fields (hard)
# --------------------------------------------------------------------------- #

SPLIT_BODY = """    fields = []
    i, n = 0, len(line)
    while True:
        if i < n and line[i] == '"':
            i += 1
            value = []
            while True:
                if i >= n:
                    @UNTERMINATED@
                if line[i] == '"':
                    if i + 1 < n and line[i + 1] == '"':
                        value.append('"')
                        i += 2
                        continue
                    i += 1
                    break
                value.append(line[i])
                i += 1
            fields.append("".join(value))
            if i >= n:
                return fields
            if line[i] != @SEP@:
                @AFTER_QUOTE@
            i += 1
        else:
            j = line.find(@SEP@, i)
            if j < 0:
                fields.append(line[i:])
                return fields
            fields.append(line[i:j])
            i = j + 1
"""
SPLIT_RAISE = {"@UNTERMINATED@": 'raise ValueError("unterminated quoted field")', "@AFTER_QUOTE@": 'raise ValueError("text after a closing quote")'}
SPLIT_LENIENT = {"@UNTERMINATED@": "break", "@AFTER_QUOTE@": "j = line.find(@SEP@, i)\n                fields[-1] += line[i:] if j < 0 else line[i:j]\n                if j < 0:\n                    return fields\n                i = j"}
SPLIT_TOGGLE = """    fields, cur, quoted = [], [], False
    for ch in line:
        if ch == '"':
            quoted = not quoted
        elif ch == @SEP@ and not quoted:
            fields.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if quoted:
        raise ValueError("unterminated quoted field")
    fields.append("".join(cur))
    return fields
"""


def _split_src(p, mode):
    src = SPLIT_BODY
    for k, v in (SPLIT_RAISE if mode == "strict" else SPLIT_LENIENT).items():
        src = src.replace(k, v)
    return _sub(src, SEP=p["sep"])


def _split_doc(p):
    s = p["sep"]
    return (
        f"Split one line of {s!r}-separated text into a list of field values.\n\n"
        f"- Fields are separated by {s!r}. Two adjacent separators give an empty field, a line that ends\n"
        "  with a separator ends with an empty field, and the empty line is one empty field.\n"
        "- A field that starts with a double quote is quoted: it ends at the matching closing quote,\n"
        "  which must be followed by a separator or the end of the line. Inside it the separator is an\n"
        "  ordinary character and two double quotes stand for one. The enclosing quotes are not part\n"
        "  of the value.\n"
        "- A double quote anywhere else in a field is an ordinary character.\n"
        "- Spaces are part of the values and are never stripped.\n"
        "- Raise ValueError if a quoted field is not closed, or if its closing quote is followed by\n"
        "  anything other than a separator or the end of the line."
    )


def _plain_field(rng, p):
    return rng.choice(["", "x"] + WORDS[:8] + [f"{rng.choice(WORDS)} {rng.choice(WORDS)}", str(rng.randint(0, 999))])


def _quoted(value):
    return '"' + value.replace('"', '""') + '"'


def _split_typical(rng, p):
    s = p["sep"]
    fields = []
    for _ in range(rng.randint(3, 5)):
        r = rng.random()
        if r < 0.3:
            fields.append(_quoted(f"{rng.choice(WORDS)}{s} {rng.choice(WORDS)}"))
        elif r < 0.4:
            fields.append(_quoted(f"{rng.choice(WORDS)} {rng.randint(2, 9)}"))
        else:
            fields.append(_plain_field(rng, p))
    return [s.join(fields)]


def _split_edges(rng, p):
    s = p["sep"]
    w1, w2, w3 = rng.sample(WORDS, 3)
    return [
        ("doubled_quotes", [s.join([w1, _quoted(f'say "{w2}"{s} then go'), w3])]),
        ("empty_fields", [s.join(["", w1, "", ""])]),
        ("inner_quote", [s.join([f'{w1}"{w2}', w3])]),
        ("unterminated", [s.join([w1, f'"{w2}{s} {w3}'])]),
        ("text_after_quote", [s.join([f'"{w1}"{w2}', w3])]),
        ("empty_line", [""]),
    ]


SPLIT = Entry(
    name="split_fields",
    module="fields.py",
    tier="hard",
    signature="line: str) -> list[str]",
    params=lambda rng: {"sep": rng.choice([",", ";", "|"])},
    doc=_split_doc,
    body=lambda p: _split_src(p, "strict"),
    wrong=lambda p: {
        "naive-split": _sub("    return line.split(@SEP@)\n", SEP=p["sep"]),
        "toggle-quotes": _sub(SPLIT_TOGGLE, SEP=p["sep"]),
        "lenient-quotes": _split_src(p, "lenient"),
    },
    typical=_split_typical,
    edges=_split_edges,
    shown_edge="doubled_quotes",
)

CATALOG = {e.name: e for e in (COMPRESS, RLE, WINDOW, MERGE, TOP, VERSION, PATH, SPLIT)}

# --------------------------------------------------------------------------- #
# Assembling an instance
# --------------------------------------------------------------------------- #

MODULE_DOC = '"""{title}"""\n'


def _call(name, args):
    return f"{name}({', '.join(repr(a) for a in args)})"


def _docstring(entry, p, examples):
    text = entry.doc(p) + "\n\n" + "".join(f">>> {_call(entry.name, args)}\n{result!r}\n" for args, result in examples)
    lines = text.rstrip("\n").split("\n")
    body = "\n".join(("    " + ln) if ln else "" for ln in lines)
    return f'    """{body[4:]}\n    """\n'


def _function(entry, docstring, body):
    return f"def {entry.name}({entry.signature}:\n{docstring}{body}"


def _module(parts):
    return '"""Small helpers."""\n\n\n' + "\n\n".join(parts)


def _exec(src):
    ns = {}
    exec(compile(src, "<generated>", "exec"), ns)  # noqa: S102 - our own reference/wrong sources
    return ns


def _argnames(entry):
    return [a.split(":")[0].strip() for a in entry.signature.split(")")[0].split(",")]


def _hardcoded(entry, examples):
    """What an agent writes when it special-cases the docstring examples."""
    names = _argnames(entry)
    lines = []
    for args, result in examples:
        if len(args) == 1:
            cond = f"{names[0]} == {args[0]!r}"
        else:
            cond = f"({', '.join(names)}) == ({', '.join(repr(a) for a in args)})"
        lines.append(f"    if {cond}:\n        return {result!r}\n")
    return "".join(lines) + "    raise NotImplementedError\n"


def _write(module, source):
    return f"cat > /app/{module} <<'PY'\n{source}PY\n"


def _checks_for(rng, entry, p, ref, examples):
    shown = [args for args, _ in examples]
    cases = []
    for _ in range(N_HIDDEN_TYPICAL):
        for _ in range(50):
            args = entry.typical(rng, p)
            if args not in shown and args not in [a for _, a in cases]:
                break
        else:
            raise Reject("no fresh typical input")
        cases.append(("typical", args))
    cases += entry.edges(rng, p)
    checks = []
    for i, (label, args) in enumerate(cases):
        name = f"{entry.name}_{label}_{i}"
        try:
            result = ref[entry.name](*[_copy(a) for a in args])
        except ValueError:
            checks.append({"name": name, "func": entry.name, "kind": "raises", "args": args})
            continue
        checks.append({"name": name, "func": entry.name, "kind": "equal", "args": args, "expected": result})
    if entry.no_mutation is not None:
        checks.append({"name": f"{entry.name}_no_mutation", "func": entry.name, "kind": "no_mutation", "args": entry.no_mutation(rng, p)})
    return checks


def _copy(v):
    return [_copy(x) for x in v] if isinstance(v, list) else v


def _examples(rng, entry, p, ref, n, show_edge):
    examples, seen = [], []
    edge = dict(entry.edges(rng, p))[entry.shown_edge] if show_edge else None
    while len(examples) < n - (1 if edge else 0):
        args = entry.typical(rng, p)
        if args in seen:
            continue
        seen.append(args)
        examples.append((args, ref[entry.name](*_copy(args))))
    if edge is not None:
        examples.insert(rng.randint(0, len(examples)), (edge, ref[entry.name](*_copy(edge))))
    return examples


def build(ctx):
    rng, cfg = ctx.rng, ctx.params
    entries = [CATALOG[rng.choice(sorted(n for n, e in CATALOG.items() if e.tier == tier))] for tier in cfg["tiers"]]
    module = entries[0].module if len(entries) == 1 else "toolkit.py"
    stubs, oracle_parts, all_checks, specs = [], [], [], {}
    wrong_by_entry, hard_parts = {}, []
    for entry in entries:
        p = entry.params(rng)
        ref = _exec(_function(entry, "", entry.body(p)))
        examples = _examples(rng, entry, p, ref, cfg["examples"], cfg["show_edge"])
        doc = _docstring(entry, p, examples)
        if '"""' in doc[7:-8] or "\\" in doc:
            raise Reject("an example does not fit in a docstring")
        stubs.append(_function(entry, doc, "    raise NotImplementedError\n"))
        oracle_parts.append(_function(entry, doc, entry.body(p)))
        hard_parts.append(_function(entry, doc, _hardcoded(entry, examples)))
        wrong_by_entry[entry.name] = {k: _function(entry, doc, v) for k, v in entry.wrong(p).items()}
        all_checks += _checks_for(rng, entry, p, ref, examples)
        specs[entry.name] = p
    # every wrong variant fails a planted check on its own function
    for entry in entries:
        mine = [c for c in all_checks if c["func"] == entry.name]
        for name, src in wrong_by_entry[entry.name].items():
            if all(run_checks(_exec(src), mine).values()):
                raise Reject(f"no hidden check catches {name}")
    oracle = _module(oracle_parts)
    shortcuts = {"hardcode-examples": _module(hard_parts)}
    for i, entry in enumerate(entries):
        for name, src in wrong_by_entry[entry.name].items():
            if name in shortcuts:
                raise AssertionError(f"duplicate shortcut name {name}")
            shortcuts[name] = _module(oracle_parts[:i] + [src] + oracle_parts[i + 1 :])
    names = [e.name for e in entries]
    if len(names) == 1:
        instruction = INSTRUCTION_ONE.format(name=names[0], module=module)
    else:
        instruction = INSTRUCTION_MANY.format(names=" and ".join(f"`{n}`" for n in names), module=module)
    return TaskSpec(
        instruction=instruction,
        files={module: _module(stubs)},
        grader=Checks(f"/app/{module}", module[:-3], tuple(all_checks)),
        oracle=Solution(_write(module, oracle), lambda f, s=oracle: {f"/app/{module}": s}),
        shortcuts={k: Solution(_write(module, v), lambda f, s=v: {f"/app/{module}": s}) for k, v in shortcuts.items()},
        params={"functions": names, "spec": specs, "n_checks": len(all_checks)},
    )


FAMILY = Family(
    name="implement-function",
    version=1,
    cluster="write-code",
    category="coding",
    skills=("python", "reading-specs", "edge-cases"),
    difficulties={
        "easy": {"tiers": ["easy"], "examples": 4, "show_edge": True},
        "medium": {"tiers": ["medium"], "examples": 3, "show_edge": False},
        "hard": {"tiers": ["hard", "medium"], "examples": 2, "show_edge": False},
    },
    build=build,
)
