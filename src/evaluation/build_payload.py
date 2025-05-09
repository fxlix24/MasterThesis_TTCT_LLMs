# src/evaluation/build_payload.py
"""
Convert a Request ORM object into the compact, judge-friendly payload
we agreed on.

Usage
-----
    from evaluation.build_payload import build_payload
    payload = build_payload(request)
"""

from __future__ import annotations
from typing import Dict
import re
# ── helper -------------------------------------------------------------

def _clean_response(text: str):
    """
    Return (use, detail) or None if the line is meta / empty.
    Splits at the first [- – — :] to separate the core idea from its elaboration.
    """
    txt = text.strip()
    if not txt:
        return None

    head, *tail = re.split(r"\s*[-–—:]\s*", txt, maxsplit=1)
    use = head.rstrip(". ").strip()
    detail = tail[0].rstrip(". ").strip() if tail else ""
    return (use, detail)

# ── build input payload -----------------------------------------------
def build_payload(req) -> Dict[str, object]:
    ideas = []
    seen = set()

    for r in req.responses:
        parsed = _clean_response(r.bullet_text)
        if not parsed:
            continue
        use, detail = parsed
        key = use.lower()
        if key in seen:
            continue                     # skip duplicate uses
        seen.add(key)
        ideas.append({"use": use, "detail": detail})

    return {
        "request_id": req.id,
        "original_prompt": req.prompt,   # drop this if your judge doesn’t need it
        "ideas": ideas,                  # structured responses
    }
