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

# ── build input payload -----------------------------------------------
def build_payload(req) -> Dict[str, object]:
    ideas = []
    seen = set()

    for r in req.responses:
        use = r.bullet_point
        detail = r.bullet_details
        key = use.lower()
        if key in seen:
            continue                     # skip duplicate uses
        seen.add(key)
        ideas.append({"use": use, "detail": detail})

    return {
        "request_id": req.id,
        "original_prompt": req.prompt,   
        "ideas": ideas,                  # structured responses
    }
