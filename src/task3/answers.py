"""Task 3 answer extraction, equivalence grouping and grading.

MATH: answers compared with math-verify (symmetric equivalence for grouping; gold vs answer
for grading). MathVista: the released metadata's question/answer type, choices and precision,
following MathVista's normalization (multiple choice -> option letter; integer/float with
precision; text/list as strings). Missing extractions form one NO_ANSWER group that is always
graded incorrect. Groups are formed WITHOUT gold answers; grading uses the same rules.
"""
import re
from difflib import SequenceMatcher

NO_ANSWER = "NO_ANSWER"


# ------------------------------------------------------------------ extraction
def last_boxed(text: str):
    """Last literal \\boxed{ with balanced braces; None if absent, empty or unclosed."""
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    start = j
    while j < len(text):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                inner = text[start:j].strip()
                return inner or None
        j += 1
    return None


# ------------------------------------------------------------------ MATH
_PARSE_CACHE = {}


def _mv_parse(s):
    from math_verify import parse
    if s not in _PARSE_CACHE:
        try:
            _PARSE_CACHE[s] = parse("\\boxed{" + s + "}")
        except Exception:  # noqa: BLE001
            _PARSE_CACHE[s] = []
    return _PARSE_CACHE[s]


def math_equal(a, b):
    from math_verify import verify
    if a == b:
        return True
    pa, pb = _mv_parse(a), _mv_parse(b)
    if not pa or not pb:
        return False
    try:
        return bool(verify(pa, pb) or verify(pb, pa))
    except Exception:  # noqa: BLE001
        return False


def math_correct(gold, ans):
    from math_verify import verify
    if ans is None or ans == NO_ANSWER:
        return False
    pg, pa = _mv_parse(gold), _mv_parse(ans)
    if not pg or not pa:
        return gold.strip() == ans.strip()
    try:
        return bool(verify(pg, pa))
    except Exception:  # noqa: BLE001
        return False


def group_answers(answers, equal):
    """Map each answer string to a canonical group key (first member of its group, in the
    order given). None -> NO_ANSWER. Uses only the answers themselves."""
    reps, key = [], {}
    for a in answers:
        if a is None:
            continue
        if a in key:
            continue
        for r in reps:
            if equal(a, r):
                key[a] = r
                break
        else:
            reps.append(a)
            key[a] = a
    return key


# ------------------------------------------------------------------ MathVista
_LATEX_TEXT = re.compile(r"\\(?:text|mathrm|textbf|mathbf)\{([^{}]*)\}")


def _clean(s):
    s = s.strip().strip("$").strip()
    s = _LATEX_TEXT.sub(r"\1", s)
    for t in ("\\,", "\\;", "\\!", "\\%", "%", "^\\circ", "^{\\circ}", "\\circ", "\\left", "\\right"):
        s = s.replace(t, "")
    return s.strip()


def _first_number(s):
    s = s.replace(",", "")
    m = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", s)
    if not m:
        frac = re.search(r"\\d?frac\{(-?\d+)\}\{(\d+)\}", s)
        return float(frac.group(1)) / float(frac.group(2)) if frac else None
    x = m[0]
    if "/" in x:
        n, d = x.split("/")
        return float(n) / float(d) if float(d) else None
    return float(x)


def mathvista_normalize(ans, meta):
    """MathVista rules (question_type, answer_type, precision, choices). Returns a string
    or NO_ANSWER."""
    if ans is None:
        return NO_ANSWER
    s = _clean(str(ans))
    if not s:
        return NO_ANSWER
    if meta["question_type"] == "multi_choice":
        choices = list(meta["choices"])
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        t = s.strip().rstrip(".").strip()
        m = re.fullmatch(r"\(?([A-Z])\)?(?:[.:)]\s*.*)?", t)
        if m and m.group(1) in letters:
            return m.group(1)
        low = {str(c).strip().lower(): l for c, l in zip(choices, letters)}
        if t.lower() in low:
            return low[t.lower()]
        # MathVista: otherwise the most similar choice
        sims = [SequenceMatcher(None, t.lower(), str(c).lower()).ratio() for c in choices]
        return letters[max(range(len(choices)), key=lambda i: sims[i])]
    at = meta["answer_type"]
    if at == "integer":
        x = _first_number(s)
        return str(int(round(x))) if x is not None else NO_ANSWER
    if at == "float":
        x = _first_number(s)
        p = int(meta.get("precision") or 0)
        return str(round(x, p)) if x is not None else NO_ANSWER
    return s.lower()


def mathvista_gold(meta):
    if meta["question_type"] == "multi_choice":
        letters = [chr(ord("A") + i) for i in range(len(meta["choices"]))]
        return letters[[str(c) for c in meta["choices"]].index(str(meta["answer"]))]
    return mathvista_normalize(meta["answer"], meta)