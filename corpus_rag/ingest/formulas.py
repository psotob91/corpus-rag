"""Extract equation-like lines heuristically (no LaTeX/VLM yet).

RESCUE: ms-rie orchestrator/formula_extractor.py

Deterministic-first baseline: flag lines that are dense in math symbols or end
with an equation number like "(3)". LaTeX conversion / VLM is a later upgrade,
so ``latex`` is left empty and ``source`` is "heuristic".
  {"id": "<slug>-e<N>", "page": p, "latex": "", "source": "heuristic", "text": raw}
"""
from __future__ import annotations

import re

# Characters that signal mathematical content.
_MATH_CHARS = set("=+-*/^<>∑∏∫∂√≈≠≤≥"
                  "±×÷∇∞αβγδθ"
                  "λμπσφω→∈∑")

# Equation number at end of line, e.g. "... (3)" or "... (12a)".
_EQ_NUM_RE = re.compile(r"\(\s*\d+[a-z]?\s*\)\s*$")

# A line with a relation operator surrounded by symbols is likely an equation.
_REL_RE = re.compile(r"[A-Za-z0-9\)\]]\s*[=<>≈≠≤≥]\s*[A-Za-z0-9\(\[\-]")

_MIN_LEN = 3
_MATH_DENSITY = 0.18  # fraction of non-space chars that are math symbols


def _is_formula(line: str) -> bool:
    s = line.strip()
    if len(s) < _MIN_LEN:
        return False
    # Pure prose sentences ending in a period are not equations.
    non_space = [c for c in s if not c.isspace()]
    if not non_space:
        return False
    math_count = sum(1 for c in non_space if c in _MATH_CHARS)
    density = math_count / len(non_space)
    has_eq_num = bool(_EQ_NUM_RE.search(s))
    has_relation = bool(_REL_RE.search(s))
    # Require a relation/operator so we don't flag every numbered list item.
    if has_eq_num and (has_relation or density >= _MATH_DENSITY):
        return True
    if density >= _MATH_DENSITY and has_relation:
        return True
    return False


def extract_formulas(doc, slug: str) -> list[dict]:
    """Heuristically extract equation-like lines from an open PyMuPDF document.

    ``doc`` is an open ``fitz.Document``. Returns a list of formula dicts in
    reading order, numbered ``<slug>-e1``, ``<slug>-e2``, ...
    """
    out: list[dict] = []
    n = 0
    for page in doc:
        text = page.get_text("text")
        for raw_line in text.splitlines():
            if _is_formula(raw_line):
                n += 1
                out.append(
                    {
                        "id": f"{slug}-e{n}",
                        "page": page.number + 1,
                        "latex": "",
                        "source": "heuristic",
                        "text": raw_line.strip(),
                    }
                )
    return out
