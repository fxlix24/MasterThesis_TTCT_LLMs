# core_store.py
import os, re, json
from sqlalchemy.orm import Session
from database import Request, Response, engine
from typing import Optional, Tuple

class AbstractLLMStore:
    def __init__(self, phase: int = os.getenv("PROJECT_PHASE")):
        self.phase   = phase
        self.session = Session(engine)

    def save(self, prompt: str, model_name: str, full_text: str, used_tokens: int) -> Request:
        # 1) Create Request
        req = Request(
            prompt=prompt,
            model=model_name,
            experiment_phase=self.phase,
            total_tokens=used_tokens,
        )
        self.session.add(req)
        self.session.flush()

        # 2) Split and Store Responses
        for num, point, detail in _iter_responses(full_text):
            # Store the raw bullet text mainly for completeness / debugging:
            bullet_text = point if not detail else f"{point} – {detail}"

            self.session.add(Response(
                request_id   = req.id,
                bullet_number= num,
                bullet_text  = bullet_text,
                bullet_point = point,
                bullet_details = detail,
            ))

        self.session.commit()
        print(f"Stored request #{req.id} with {len(req.responses)} ideas.")
        return req
    
# --------------------------------------------------------------------- #
# 1. helper: extract numbered bullets – keeps #1 even at start of text  #
# --------------------------------------------------------------------- #
_BULLET_RE = re.compile(
    r'''
        (?:^|\n)                    # start of text OR a new line
        \s*                         # optional indent
        (?:\d+\.\s* | [*\-•]\s+)    # bullet marker: "1.", "-", "•", …
        (.*?)                       # the bullet itself (non-greedy)

        (?=                         # … until we hit one of these:
            \n\s*(?:\d+\.\s*|[*\-•]) #   1) another bullet
          | \n\s*#+\s               #   2) a Markdown heading (###, ##, #…)
          | \n{2,}                  #   3) a completely blank line
          | \Z                      #   4) end of text
        )
    ''',
    re.DOTALL | re.VERBOSE | re.MULTILINE,
)

def extract_bullets(text: str):
    bullets = [m.group(1).strip() for m in _BULLET_RE.finditer(text)]
    return list(enumerate(bullets, start=1))

_delim = re.compile(
    r"""
    (                      
        \s[-–—]\s          
      | :\s+               
      | :\*\*\s+            
    )
    """,
    re.VERBOSE,
)

# --------------------------------------------------------------------- #
# 2. helper: extract idea and explanatory details                       #
# --------------------------------------------------------------------- #

def _clean_response(text: str) -> Optional[Tuple[str, str]]:
    txt = (text or "").strip()
    if not txt:
        return None

    m = _delim.search(txt)
    if not m:                        # no delimiter found → whole line is the idea
        return (txt.rstrip(". "), "")

    head = txt[:m.start()].rstrip(". ")
    tail = txt[m.end():].lstrip().rstrip(". ")

    # If the delimiter was ":** " we snipped the closing **.
    # Put it back and strip it from the detail if necessary.
    if m.group().startswith(":**"):
        head = head + "**"
        if tail.startswith("**"):
            tail = tail[2:].lstrip()

    # final tidy-up
    return (head.strip(), tail.strip())

# --------------------------------------------------------------------- #
# 3. helper: extract idea and explanatory details from JSON             #
# --------------------------------------------------------------------- #

def _iter_responses(raw: str):
    """
    Yields (n, idea, explanation) tuples from either a JSON list or plaintext bullets.
    """
    # ── 1) try JSON first (handle markdown code blocks) ──────────────
    json_text = raw.strip()
    
    # Remove markdown code block wrapper if present
    if json_text.startswith('```json\n') and json_text.endswith('\n```'):
        json_text = json_text[8:-4]  # Remove ```json\n from start and \n``` from end
    elif json_text.startswith('```\n') and json_text.endswith('\n```'):
        json_text = json_text[4:-4]  # Remove ```\n from start and \n``` from end
    elif json_text.startswith('```') and json_text.endswith('```'):
        json_text = json_text[3:-3].strip()  # Remove ``` from both ends
    
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        # Try to salvage partial JSON by attempting to fix common truncation issues
        try:
            # If it looks like JSON but is truncated, try to close it
            if json_text.strip().startswith('[') and not json_text.strip().endswith(']'):
                # Try to close the array - this is a simple heuristic
                salvage_attempt = json_text.rstrip().rstrip(',') + ']'
                data = json.loads(salvage_attempt)
            else:
                data = None
        except (json.JSONDecodeError, Exception):
            data = None

    # Handle both direct arrays and nested objects
    if data is not None:
        ideas_list = None
        
        # Case 1: Direct array of objects with "idea" key
        if isinstance(data, list) and all(isinstance(d, dict) and "idea" in d for d in data):
            ideas_list = data
        
        # Case 2: Object with a key containing an array of idea objects
        elif isinstance(data, dict):
            # Look for any key that contains a list of idea objects
            for key, value in data.items():
                if isinstance(value, list) and all(isinstance(d, dict) and "idea" in d for d in value):
                    ideas_list = value
                    break
        
        # If we found a valid ideas list, process it
        if ideas_list:
            for n, item in enumerate(ideas_list, 1):
                idea = item["idea"].strip()
                explanation = (item.get("explanation") or "").strip()
                yield n, idea, explanation
            return

    # ── 2) fall back to the old plaintext logic ─────────────────────
    for n, bullet in extract_bullets(raw):
        idea, detail = _clean_response(bullet)
        if idea:  # Only yield if we have a valid idea
            yield n, idea, detail