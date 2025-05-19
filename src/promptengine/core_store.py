# core_store.py
import os, re
from sqlalchemy.orm import Session
from database import Request, Response, engine

class AbstractLLMStore:
    def __init__(self, phase: str = os.getenv("PROJECT_PHASE")):
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
        for num, idea in extract_bullets(full_text):
            self.session.add(Response(
                request_id=req.id,
                bullet_number=num,
                bullet_text=idea,
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
